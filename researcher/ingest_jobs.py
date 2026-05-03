"""Async ingest job tracking and progress emission.

Backs ``fr_researcher_2b22a2f3`` (async ingest job model) and
``fr_researcher_bbf3cf69`` (progress events).

Long-running ingest (depth=readme+code on a non-trivial repo, or
``ingest_idea`` followed by claim parsing + relevance filtering) holds
a single bus request open for minutes. That's fragile under transport
flakiness and ties up an MCP slot. Convert to an async job model:

  - Caller invokes an ``ingest_*_async`` skill which returns
    ``{job_id, accepted_at}`` immediately and schedules the work as
    an asyncio task.
  - The task emits ``research.ingest.progress`` bus events as it moves
    through phases — caller subscribes via ``bus_wait_for_event``
    instead of polling.
  - ``ingest_status(job_id)`` returns the current phase / progress /
    completed result for direct polling when an event-driven caller
    isn't available.

Phase enum (loose — pipeline functions emit what's meaningful for
their own work):

  accepted  → started
  started   → (one or more of) cloning / ast_scanning / distilling / storing
  any phase → done | error

Job state lives in-process on the agent; restarts wipe in-flight
jobs (same shape as ``IngestWatcherRegistry``). Persisting jobs
across restarts is a deliberate future extension.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

PROGRESS_TOPIC = "research.ingest.progress"

# Canonical phase names. Pipeline functions emit a subset; subscribers
# should treat any unknown phase as informational.
PHASES: tuple[str, ...] = (
    "accepted",
    "started",
    "cloning",
    "ast_scanning",
    "distilling",
    "storing",
    "done",
    "error",
)

TERMINAL_PHASES: frozenset[str] = frozenset({"done", "error"})


@dataclass
class JobRecord:
    """Per-job state held in :class:`IngestJobStore`."""

    job_id: str
    skill: str           # ingest_github | ingest_file | ingest_idea
    args: dict[str, Any]
    accepted_at: float
    phase: str = "accepted"
    progress_pct: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_status(self) -> dict[str, Any]:
        """Public-facing snapshot — what ``ingest_status`` returns."""
        return {
            "job_id": self.job_id,
            "skill": self.skill,
            "phase": self.phase,
            "progress_pct": self.progress_pct,
            "accepted_at": self.accepted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "history": list(self.history),
        }


class IngestJobStore:
    """In-memory async-safe registry for active and recent ingest jobs.

    Concurrency: an ``asyncio.Lock`` guards every state mutation so
    progress callbacks firing from a worker task don't race the
    caller's status reads. Job creation, transitions, and lookups all
    go through the lock.

    Bounded retention: keeps the most recent ``max_completed`` finished
    jobs (default 64) so an agent that runs for days doesn't grow its
    job table without limit. In-flight jobs are never evicted.
    """

    def __init__(self, max_completed: int = 64):
        self._jobs: dict[str, JobRecord] = {}
        self._completed_order: list[str] = []  # FIFO of finished job_ids
        self._max_completed = max_completed
        self._lock = asyncio.Lock()

    async def create(self, skill: str, args: dict[str, Any]) -> JobRecord:
        async with self._lock:
            job = JobRecord(
                job_id=f"job_{uuid.uuid4().hex[:12]}",
                skill=skill,
                args=dict(args),
                accepted_at=time.time(),
            )
            self._jobs[job.job_id] = job
            return job

    async def get(self, job_id: str) -> Optional[JobRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list(
        self,
        *,
        skill: str | None = None,
        phases: Iterable[str] | None = None,
    ) -> list[JobRecord]:
        async with self._lock:
            wanted = set(phases) if phases else None
            return [
                j for j in self._jobs.values()
                if (skill is None or j.skill == skill)
                and (wanted is None or j.phase in wanted)
            ]

    async def transition(
        self,
        job_id: str,
        *,
        phase: str,
        progress_pct: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Optional[JobRecord]:
        """Move a job to ``phase`` and append a history entry.

        Returns the updated record, or None if the job_id is unknown
        (defensive — a publish callback firing after the job was evicted
        shouldn't crash the worker).
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("transition on unknown job_id=%s phase=%s", job_id, phase)
                return None
            now = time.time()
            job.phase = phase
            if progress_pct is not None:
                job.progress_pct = max(0, min(100, int(progress_pct)))
            if phase == "started" and job.started_at is None:
                job.started_at = now
            if phase in TERMINAL_PHASES:
                job.completed_at = now
                if phase == "done" and job.progress_pct < 100:
                    job.progress_pct = 100
                self._mark_completed_locked(job_id)
            entry: dict[str, Any] = {"phase": phase, "at": now}
            if progress_pct is not None:
                entry["progress_pct"] = job.progress_pct
            if detail:
                entry["detail"] = dict(detail)
            job.history.append(entry)
            return job

    async def set_result(self, job_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.result = dict(result)

    async def set_error(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.error = error

    def _mark_completed_locked(self, job_id: str) -> None:
        """Append to FIFO and evict oldest finished job if over cap.
        Caller must hold ``self._lock``."""
        if job_id in self._completed_order:
            return
        self._completed_order.append(job_id)
        while len(self._completed_order) > self._max_completed:
            evict = self._completed_order.pop(0)
            self._jobs.pop(evict, None)


# Type alias for the publish callable an agent passes in. Keeps the
# JobStore decoupled from BaseAgent — tests can inject a recorder.
PublishFn = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _publish_progress(
    publish: PublishFn,
    job: JobRecord,
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    """Fire one ``research.ingest.progress`` bus event for ``job``.

    Best-effort: a publish failure (e.g. transient bus disconnect) is
    logged but does not abort the worker — progress events are
    informational, not load-bearing for correctness.
    """
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "skill": job.skill,
        "phase": job.phase,
        "progress_pct": job.progress_pct,
        "accepted_at": job.accepted_at,
    }
    if job.started_at is not None:
        payload["started_at"] = job.started_at
        payload["elapsed_ms"] = int((time.time() - job.started_at) * 1000)
    if job.completed_at is not None:
        payload["completed_at"] = job.completed_at
    if job.error:
        payload["error"] = job.error
    if detail:
        payload["detail"] = dict(detail)
    try:
        await publish(PROGRESS_TOPIC, payload)
    except asyncio.CancelledError:
        # Re-raise so the calling task observes the cancellation;
        # we just want this swallow-noise pattern to not eat the
        # cancel. ``Exception`` does not catch ``CancelledError``
        # on Python 3.11+, so without this branch a cancelled
        # ``publish()`` (e.g. connector mid-shutdown) would tear
        # down the worker even though progress events are meant
        # to be best-effort. Logging at debug — operators don't
        # need to see every cancelled publish.
        logger.debug(
            "publish %s cancelled for job_id=%s phase=%s",
            PROGRESS_TOPIC, job.job_id, job.phase,
        )
        raise
    except Exception as e:  # pragma: no cover — best-effort
        logger.warning(
            "publish %s failed for job_id=%s phase=%s: %s",
            PROGRESS_TOPIC, job.job_id, job.phase, e,
        )


async def run_ingest_job(
    store: IngestJobStore,
    publish: PublishFn,
    job: JobRecord,
    work: Callable[[Callable[..., Awaitable[None]]], Awaitable[dict[str, Any]]],
) -> None:
    """Drive a job through started → … → done/error and emit events.

    ``work`` is the actual ingest coroutine. It receives a
    ``progress`` callable it may invoke to report intermediate phases::

        async def work(progress):
            await progress(phase="cloning", progress_pct=10)
            ...
            await progress(phase="storing", progress_pct=90)
            return {"entry_id": "...", ...}

    The wrapper auto-emits ``started`` before ``work`` runs and
    ``done`` (or ``error`` on exception) after.
    """
    async def progress(
        phase: str,
        *,
        progress_pct: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        updated = await store.transition(
            job.job_id, phase=phase, progress_pct=progress_pct, detail=detail,
        )
        if updated is not None:
            await _publish_progress(publish, updated, detail=detail)

    # The try/except spans EVERY await — including the
    # ``progress("started")``, ``store.set_result``, and
    # ``progress("done")`` calls — not just ``work()``. A
    # ``CancelledError`` raised at any of those points (e.g.
    # connector mid-shutdown cancels the publish) would otherwise
    # bypass the error-path bookkeeping and leave the JobRecord
    # stuck at the last successful transition (``accepted`` if
    # ``progress("started")`` itself was the cancelled await).
    # ``ingest_status`` callers would then poll a non-terminal
    # phase indefinitely. Wrapping the whole body forces the
    # cancellation through the same ``set_error`` + emit-error
    # path as a mid-work cancel.
    try:
        await progress("started", progress_pct=0)
        result = await work(progress)
        await store.set_result(job.job_id, result)
        await progress(
            "done",
            progress_pct=100,
            detail={"result_keys": sorted(result.keys())},
        )
    except asyncio.CancelledError:
        # Cancellation can hit at any point in the body. We need
        # to distinguish "work failed mid-flight" (record error)
        # from "work succeeded, cancellation hit during the
        # bookkeeping tail" (don't downgrade to error — the
        # entry was actually persisted).
        #
        # ``store.set_result`` runs after ``work()`` returns
        # successfully; if the JobRecord's ``result`` is non-None
        # at the cancel point, the work outcome is durable and
        # marking ``phase=error`` would lie to subscribers.
        #
        # ``progress("done")`` calls ``store.transition`` BEFORE
        # ``_publish_progress``; if the publish call raises
        # CancelledError, ``phase`` is already ``"done"`` on the
        # JobRecord. Same case as above — don't reclassify.
        current = await store.get(job.job_id)
        if current is not None and (
            current.phase == "done" or current.result is not None
        ):
            # Mid-bookkeeping cancel after a successful work().
            # If the phase is still "started" / a custom phase
            # (set_result ran, progress("done") didn't), finalise
            # to done so subscribers see a terminal phase that
            # reflects reality.
            if current.phase != "done":
                await store.transition(
                    job.job_id, phase="done", progress_pct=100,
                )
            raise

        # Work didn't complete: record the cancellation. ``progress``
        # itself may also be cancelled while emitting the error
        # event; fall back to a direct phase transition so the
        # JobRecord always lands terminal.
        logger.info("ingest job %s cancelled", job.job_id)
        await store.set_error(job.job_id, "CancelledError: cancelled")
        try:
            await progress("error")
        except asyncio.CancelledError:
            await store.transition(job.job_id, phase="error")
        raise
    except Exception as e:
        # ``logger.warning`` (no stack) rather than
        # ``logger.exception`` because a large fraction of failures
        # here are expected user-facing errors —
        # ``handle_ingest_github_async`` translates pipeline error
        # dicts (invalid GitHub URL, clone permission denied, …)
        # into ``RuntimeError`` so the job surfaces ``phase=error``,
        # and a full traceback for those would be log noise. The
        # error message and exception type still land on the
        # JobRecord and stream through the progress event.
        logger.warning("ingest job %s failed: %s: %s", job.job_id, type(e).__name__, e)
        await store.set_error(job.job_id, f"{type(e).__name__}: {e}")
        await progress("error")
        return
