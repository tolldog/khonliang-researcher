"""Tests for researcher/ingest_jobs.py — async ingest job model + progress events.

Covers fr_researcher_2b22a2f3 (async ingest jobs) +
fr_researcher_bbf3cf69 (research.ingest.progress events).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from researcher.ingest_jobs import (
    IngestJobStore,
    JobRecord,
    PROGRESS_TOPIC,
    TERMINAL_PHASES,
    run_ingest_job,
)


# ---------------------------------------------------------------------------
# JobStore unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_assigns_unique_job_id_and_records_args():
    store = IngestJobStore()
    a = await store.create("ingest_github", {"repo_url": "https://x"})
    b = await store.create("ingest_github", {"repo_url": "https://y"})
    assert a.job_id != b.job_id
    assert a.skill == "ingest_github"
    assert a.args == {"repo_url": "https://x"}
    assert a.phase == "accepted"
    assert a.accepted_at > 0
    assert a.started_at is None
    assert a.completed_at is None


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_id():
    store = IngestJobStore()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_transition_records_phase_history_and_started_at():
    store = IngestJobStore()
    j = await store.create("ingest_idea", {"text": "x"})
    await store.transition(j.job_id, phase="started", progress_pct=0)
    await store.transition(j.job_id, phase="distilling", progress_pct=30)
    await store.transition(j.job_id, phase="storing", progress_pct=80)

    fresh = await store.get(j.job_id)
    assert fresh is not None
    assert fresh.phase == "storing"
    assert fresh.progress_pct == 80
    assert fresh.started_at is not None
    assert fresh.completed_at is None
    phases = [h["phase"] for h in fresh.history]
    assert phases == ["started", "distilling", "storing"]


@pytest.mark.asyncio
async def test_transition_to_done_marks_completed_and_pegs_progress_at_100():
    store = IngestJobStore()
    j = await store.create("ingest_idea", {"text": "x"})
    await store.transition(j.job_id, phase="done")
    fresh = await store.get(j.job_id)
    assert fresh.phase == "done"
    assert fresh.completed_at is not None
    assert fresh.progress_pct == 100  # auto-pegged on done


@pytest.mark.asyncio
async def test_transition_to_error_records_completion_without_pegging():
    store = IngestJobStore()
    j = await store.create("ingest_idea", {"text": "x"})
    await store.transition(j.job_id, phase="started", progress_pct=0)
    await store.transition(j.job_id, phase="error")
    fresh = await store.get(j.job_id)
    assert fresh.phase == "error"
    assert fresh.completed_at is not None
    # Error doesn't peg progress at 100 — it failed mid-flight.
    assert fresh.progress_pct == 0


@pytest.mark.asyncio
async def test_transition_on_unknown_id_is_logged_not_raised():
    store = IngestJobStore()
    # Returns None — defensive; a publish callback firing after eviction
    # shouldn't crash the worker.
    assert await store.transition("missing", phase="done") is None


@pytest.mark.asyncio
async def test_completed_jobs_are_evicted_above_max():
    store = IngestJobStore(max_completed=2)
    ids = []
    for i in range(4):
        j = await store.create("ingest_idea", {"i": i})
        ids.append(j.job_id)
        await store.transition(j.job_id, phase="done")
    # Only the last 2 finished jobs survive; in-flight jobs are never
    # evicted (none here, but covered by next test).
    surviving = [jid for jid in ids if await store.get(jid) is not None]
    assert surviving == ids[2:]


@pytest.mark.asyncio
async def test_in_flight_jobs_not_evicted_when_completed_overflow():
    store = IngestJobStore(max_completed=1)
    flight = await store.create("ingest_idea", {"in": "flight"})  # never done
    for i in range(3):
        j = await store.create("ingest_idea", {"i": i})
        await store.transition(j.job_id, phase="done")
    assert await store.get(flight.job_id) is not None


@pytest.mark.asyncio
async def test_list_filters_by_skill_and_phase():
    store = IngestJobStore()
    g = await store.create("ingest_github", {})
    f = await store.create("ingest_file", {})
    await store.transition(g.job_id, phase="started", progress_pct=0)
    await store.transition(f.job_id, phase="done")
    by_skill = await store.list(skill="ingest_github")
    assert {j.job_id for j in by_skill} == {g.job_id}
    in_flight = await store.list(phases=["accepted", "started", "cloning",
                                          "ast_scanning", "distilling", "storing"])
    assert {j.job_id for j in in_flight} == {g.job_id}


# ---------------------------------------------------------------------------
# run_ingest_job — drives a job through started → done/error and emits events
# ---------------------------------------------------------------------------


class _Recorder:
    """Records every (topic, payload) for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, topic: str, payload: dict[str, Any]) -> None:
        self.events.append((topic, dict(payload)))


@pytest.mark.asyncio
async def test_run_ingest_job_happy_path_emits_started_and_done():
    store = IngestJobStore()
    rec = _Recorder()
    job = await store.create("ingest_idea", {"text": "x"})

    async def work(progress):
        await progress("distilling", progress_pct=30)
        return {"idea_id": "id_42"}

    await run_ingest_job(store, rec, job, work)

    phases = [p["phase"] for _, p in rec.events]
    assert phases == ["started", "distilling", "done"]
    # Every event is on the canonical topic
    assert all(t == PROGRESS_TOPIC for t, _ in rec.events)
    # Final state stored
    fresh = await store.get(job.job_id)
    assert fresh.phase == "done"
    assert fresh.result == {"idea_id": "id_42"}
    assert fresh.error is None


@pytest.mark.asyncio
async def test_run_ingest_job_error_path_emits_error_and_records_message():
    store = IngestJobStore()
    rec = _Recorder()
    job = await store.create("ingest_github", {"repo_url": "x"})

    async def work(progress):
        await progress("cloning", progress_pct=10)
        raise RuntimeError("clone failed: 403")

    await run_ingest_job(store, rec, job, work)

    phases = [p["phase"] for _, p in rec.events]
    assert phases == ["started", "cloning", "error"]
    fresh = await store.get(job.job_id)
    assert fresh.phase == "error"
    assert "clone failed: 403" in fresh.error
    assert fresh.result is None


@pytest.mark.asyncio
async def test_progress_event_payload_includes_job_metadata():
    store = IngestJobStore()
    rec = _Recorder()
    job = await store.create("ingest_idea", {"source_label": "demo"})

    async def work(progress):
        await progress("storing", progress_pct=80, detail={"entry_id": "e1"})
        return {"entry_id": "e1"}

    await run_ingest_job(store, rec, job, work)

    # Inspect the storing event specifically — the FR's
    # documented payload shape.
    storing = next(p for _, p in rec.events if p["phase"] == "storing")
    assert storing["job_id"] == job.job_id
    assert storing["skill"] == "ingest_idea"
    assert storing["progress_pct"] == 80
    assert storing["accepted_at"] == job.accepted_at
    assert "started_at" in storing
    assert "elapsed_ms" in storing
    assert storing["detail"] == {"entry_id": "e1"}

    # Done event carries completed_at and a result_keys detail
    done = next(p for _, p in rec.events if p["phase"] == "done")
    assert "completed_at" in done
    assert done["progress_pct"] == 100
    assert done["detail"]["result_keys"] == ["entry_id"]


@pytest.mark.asyncio
async def test_publish_failure_does_not_abort_worker():
    """A flaky bus must not break the ingest. Progress events are
    informational; the job's result still lands in the store."""
    store = IngestJobStore()
    job = await store.create("ingest_idea", {})
    flaky_calls = {"n": 0}

    async def flaky_publish(topic: str, payload: dict) -> None:
        flaky_calls["n"] += 1
        if flaky_calls["n"] == 2:
            raise ConnectionError("transient bus disconnect")

    async def work(progress):
        await progress("distilling", progress_pct=30)
        return {"idea_id": "ok"}

    await run_ingest_job(store, flaky_publish, job, work)
    fresh = await store.get(job.job_id)
    assert fresh.phase == "done"
    assert fresh.result == {"idea_id": "ok"}


@pytest.mark.asyncio
async def test_to_status_returns_serializable_snapshot():
    store = IngestJobStore()
    j = await store.create("ingest_github", {"repo_url": "https://x"})
    await store.transition(j.job_id, phase="started", progress_pct=0)
    await store.transition(j.job_id, phase="cloning", progress_pct=10)
    snap = (await store.get(j.job_id)).to_status()
    assert snap["job_id"] == j.job_id
    assert snap["skill"] == "ingest_github"
    assert snap["phase"] == "cloning"
    assert snap["progress_pct"] == 10
    assert snap["history"][0]["phase"] == "started"
    # List is a copy — mutating it doesn't affect the record.
    snap["history"].clear()
    assert (await store.get(j.job_id)).history  # original intact


def test_terminal_phases_are_constants():
    """Lock the public phase contract — subscribers filter on these."""
    assert TERMINAL_PHASES == frozenset({"done", "error"})


def test_progress_topic_is_canonical():
    assert PROGRESS_TOPIC == "research.ingest.progress"


@pytest.mark.asyncio
async def test_run_ingest_job_cancel_during_done_publish_keeps_phase_done():
    """If ``CancelledError`` lands during the FINAL
    ``progress("done")`` publish — work() already succeeded, the
    JobRecord already has ``phase=done`` and ``result`` populated —
    the cancel handler must NOT downgrade to ``phase=error``.
    Doing so would lie to subscribers about the persistence
    outcome (the entry IS saved). Same shape: a cancel that
    arrives between ``set_result`` and ``progress("done")`` should
    still finalise to ``done`` since the durable write happened."""
    store = IngestJobStore()
    job = await store.create("ingest_idea", {})

    publish_calls: list = []

    async def cancel_after_done(topic, payload):
        publish_calls.append(payload["phase"])
        if payload["phase"] == "done":
            # Work succeeded; cancel hits during the done event's
            # publish AFTER store.transition(phase=done) already ran.
            raise asyncio.CancelledError()

    async def work(progress):
        return {"idea_id": "ok"}

    with pytest.raises(asyncio.CancelledError):
        await run_ingest_job(store, cancel_after_done, job, work)

    fresh = await store.get(job.job_id)
    assert fresh.phase == "done", (
        f"work() succeeded — cancel during done-publish must not "
        f"reclassify to error; got phase={fresh.phase!r}"
    )
    assert fresh.result == {"idea_id": "ok"}
    assert fresh.error is None


@pytest.mark.asyncio
async def test_run_ingest_job_cancel_during_set_result_still_finalises_done():
    """Cancellation that lands at ``await store.set_result(...)``
    — between ``work()`` returning and the JobRecord recording the
    result — must still finalise to ``done``. Several call sites
    (``ingest_idea``) have already persisted their durable state
    inside ``work()`` itself; reclassifying to ``error`` would lie
    to the caller about a successfully-saved entry."""
    store = IngestJobStore()
    job = await store.create("ingest_idea", {})

    real_set_result = store.set_result
    cancel_state = {"first_call": True}

    async def cancel_set_result(job_id, result):
        # First call (from the happy-path ``set_result`` in run_ingest_job)
        # raises CancelledError BEFORE writing the result, simulating a
        # connector mid-shutdown cancel that lands exactly at this
        # await point. Subsequent calls (from the cancel-handler's
        # forced finalisation) go through to the real method.
        if cancel_state["first_call"]:
            cancel_state["first_call"] = False
            raise asyncio.CancelledError()
        return await real_set_result(job_id, result)

    store.set_result = cancel_set_result  # type: ignore[method-assign]

    async def work(progress):
        return {"idea_id": "persisted"}

    with pytest.raises(asyncio.CancelledError):
        await run_ingest_job(store, _Recorder(), job, work)

    fresh = await store.get(job.job_id)
    assert fresh.phase == "done"
    assert fresh.result == {"idea_id": "persisted"}
    assert fresh.error is None


@pytest.mark.asyncio
async def test_run_ingest_job_cancel_during_finalise_emits_done_event():
    """When the cancel handler forces the JobRecord to ``phase=done``
    after a mid-tail cancel, it must also emit the ``done`` progress
    event. Subscribers following ``research.ingest.progress``
    (instead of polling ``ingest_status``) would otherwise see the
    last pre-terminal phase and wait forever for a terminal event
    that the durable state already supports."""
    store = IngestJobStore()
    job = await store.create("ingest_idea", {})

    rec = _Recorder()
    cancel_state = {"first_call": True}
    real_set_result = store.set_result

    async def cancel_set_result(job_id, result):
        if cancel_state["first_call"]:
            cancel_state["first_call"] = False
            raise asyncio.CancelledError()
        return await real_set_result(job_id, result)

    store.set_result = cancel_set_result  # type: ignore[method-assign]

    async def work(progress):
        return {"idea_id": "persisted"}

    with pytest.raises(asyncio.CancelledError):
        await run_ingest_job(store, rec, job, work)

    phases = [p["phase"] for _, p in rec.events]
    assert "done" in phases, (
        f"cancel-during-finalise must still emit done event for "
        f"event-driven subscribers; got phases={phases!r}"
    )


@pytest.mark.asyncio
async def test_list_phases_empty_returns_empty_not_all():
    """``phases=[]`` (caller intersected two phase sets to nothing)
    must return ``[]``, not every job in the store. Earlier code
    used ``if phases else None`` which collapsed the empty list to
    "no filter" and leaked the full job list."""
    store = IngestJobStore()
    a = await store.create("ingest_github", {})
    b = await store.create("ingest_idea", {})
    await store.transition(a.job_id, phase="started", progress_pct=0)
    await store.transition(b.job_id, phase="done")

    assert await store.list(phases=[]) == []
    # Sanity: ``phases=None`` still returns everything (existing contract).
    assert {j.job_id for j in await store.list(phases=None)} == {a.job_id, b.job_id}


@pytest.mark.asyncio
async def test_run_ingest_job_cancel_during_started_emit_still_terminates():
    """If ``CancelledError`` fires while ``progress("started")``
    itself is awaiting (e.g. the publish call mid-cancel), the
    JobRecord must still land in a terminal phase. Earlier code
    only wrapped ``work()`` in the try/except, so a cancel here
    bypassed bookkeeping and left the record at ``phase=accepted``
    indefinitely — ``ingest_status`` callers would poll forever."""
    store = IngestJobStore()
    job = await store.create("ingest_idea", {})

    publish_calls = {"n": 0}

    async def cancel_on_first_publish(topic, payload):
        publish_calls["n"] += 1
        if publish_calls["n"] == 1:
            # Simulate the connector being cancelled mid-publish.
            raise asyncio.CancelledError()

    async def work(progress):  # never reached if started is cancelled
        return {"idea_id": "never"}

    with pytest.raises(asyncio.CancelledError):
        await run_ingest_job(store, cancel_on_first_publish, job, work)

    fresh = await store.get(job.job_id)
    assert fresh.phase == "error"
    assert "Cancelled" in (fresh.error or "")
