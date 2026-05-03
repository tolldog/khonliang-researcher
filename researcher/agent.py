"""Researcher as a bus agent.

Alternative to ``researcher.server`` (MCP over stdio). Instead of Claude
talking to the researcher directly, the researcher registers with the bus
and Claude talks to the bus.

Usage::

    # Install into the bus
    python -m researcher.agent install --id researcher-primary --bus http://localhost:8787 --config config.yaml

    # Start (normally done by the bus on boot)
    python -m researcher.agent --id researcher-primary --bus http://localhost:8787 --config config.yaml

    # Uninstall
    python -m researcher.agent uninstall --id researcher-primary --bus http://localhost:8787

The agent wraps all MCP tools from ``create_research_server`` as bus
handlers via ``BaseAgent.from_mcp()``. Tool code is identical — only the
transport changes from stdio to bus HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from types import MethodType

from khonliang_bus import BaseAgent, Skill
from khonliang_bus.connector import BusConnector

from researcher.ingest_jobs import IngestJobStore, run_ingest_job
from researcher.ingest_watcher import IngestWatcherRegistry, IngestWatcherStore

logger = logging.getLogger(__name__)


# Per-call cap for ``ingest_from_artifact``'s body fetch. Sized
# to the bus's REST-surface ceiling (``HARD_MAX_CHARS=20000``,
# see ``khonliang_bus.bus.artifacts.HARD_MAX_CHARS`` and
# ``khonliang_store.store.local_store.HARD_MAX_CHARS`` — both
# enforce 20k on the read path) so the requested size matches
# what we'll actually receive. Larger payloads will return
# ``truncated=True`` from store; full retrieval gates on a
# streaming endpoint (out of scope for this FR).
_INGEST_FETCH_CAP_CHARS = 20_000


async def stage_payload(agent: BaseAgent, args: dict) -> dict:
    """Persist a payload as a store artifact with provenance.

    Bus-skill handler shape: takes the request envelope's
    ``args`` dict directly so the wiring in
    ``_extend_with_native_handlers`` can register it as a
    handler without an extra adapter. ``args`` reads:

    * ``content`` (required, str): raw payload text. Binary
      payloads (PDF bytes, etc.) need decoding upstream
      first; the function rejects non-string inputs with a
      clear error envelope.
    * ``kind_hint`` (str, default ""): dispatcher hint
      stored in metadata for the future auto-detected
      dispatcher (sister fr_researcher_1ca5499e).
    * ``title`` (str, default ""): human-readable artifact
      name. Falls back to a 80-char first-non-empty-line
      preview when omitted.
    * ``content_type`` (str, default "text/plain").
    * ``source`` (dict, default {}): provenance dict
      (url, fetched_at, fetcher) attached to metadata.

    Returns ``{"artifact_id": ...}`` on success, or the
    store's error envelope verbatim. Thin wrapper over
    ``agent.request(agent_type='store',
    operation='artifact_create')`` — routing through the
    ``store`` agent type means the store backend can move
    (composite / local-only / etc.) without researcher
    caring.

    Module-level so tests can call it with a mock ``agent``
    without wiring through ``BaseAgent.from_mcp``. Direct
    Python callers should pass an ``args`` dict the same
    way the bus would.
    """
    # Distinguish missing (required-field error) from wrong type
    # (validation error) so callers don't see "must be a string"
    # when they simply forgot the field. Mirrors the
    # ``artifact_id is required`` shape in ``ingest_from_artifact``.
    if "content" not in args:
        return {"error": "content is required"}
    content = args["content"]
    if not isinstance(content, str):
        return {"error": "content must be a string"}
    # ``strip() == ""`` rather than just ``not content`` so a
    # whitespace-only payload ("\n\n", "   ") doesn't get
    # staged as a "valid" empty artifact.
    if not content.strip():
        return {"error": "content is required"}
    # Type-strict on string fields: silent ``str()`` coercion
    # would let callers pass a number/object and quietly persist
    # the repr() into artifact metadata, masking caller bugs
    # well downstream of the actual mistake.
    kind_hint_raw = args.get("kind_hint", "")
    if not isinstance(kind_hint_raw, str):
        return {"error": "kind_hint must be a string"}
    kind_hint = kind_hint_raw.strip()
    title_raw = args.get("title", "")
    if not isinstance(title_raw, str):
        return {"error": "title must be a string"}
    title = title_raw.strip()
    if not title:
        # Short content preview so the artifact has a human
        # name in ``artifact_list``. First non-empty stripped
        # line of the input, capped at 80 chars with an
        # ellipsis. Loop ensures a leading blank line doesn't
        # produce an empty title — ``content.partition("\n")``
        # always takes the first line, even if it's empty.
        preview = ""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                preview = stripped
                break
        if len(preview) > 80:
            # 79 + ellipsis so the rendered title is exactly 80
            # chars — the previous ``[:80] + "…"`` overshot by one.
            title = preview[:79] + "…"
        else:
            title = preview or "staged payload"
    content_type_raw = args.get("content_type", "text/plain")
    if not isinstance(content_type_raw, str):
        return {"error": "content_type must be a string"}
    # Strip + fallback so a whitespace-only or empty string
    # doesn't reach store as an invalid MIME type.
    content_type = content_type_raw.strip() or "text/plain"
    source = args["source"] if "source" in args else {}
    if not isinstance(source, dict):
        return {"error": "source must be an object"}
    # Provenance + the dispatcher hint live in metadata
    # together — store doesn't define a schema for either,
    # so the convention we set here is what the dispatcher
    # FR will read. ``source.*`` fields stay nested so they
    # don't collide with future top-level keys.
    metadata: dict = {"source": source}
    if kind_hint:
        metadata["kind_hint"] = kind_hint
    result = await agent.request(
        agent_type="store",
        operation="artifact_create",
        args={
            "kind": "staged_payload",
            "title": title,
            "content": content,
            "content_type": content_type,
            "producer": agent.agent_id,
            "metadata": metadata,
        },
    )
    payload = _unwrap_request_envelope(result)
    if isinstance(payload, dict) and "error" in payload:
        return payload
    if not isinstance(payload, dict):
        return {"error": "store returned unexpected response shape"}
    # Tolerate both response shapes: a flat metadata dict
    # (``LocalArtifactStore.create`` returns this today, with
    # ``id`` at the top level) AND a nested
    # ``{"artifact": {"id": ...}}`` envelope (the bus's REST
    # ``view_response`` shape; other callers in the
    # researcher repo already treat both as valid — see
    # ``LibrarianAgent._artifact_id``).
    artifact_id = payload.get("id")
    if not artifact_id:
        nested = payload.get("artifact")
        if isinstance(nested, dict):
            artifact_id = nested.get("id")
    if not artifact_id:
        return {"error": "store created artifact without id"}
    return {"artifact_id": artifact_id}


async def ingest_from_artifact(
    agent: BaseAgent, pipeline, args: dict,
) -> dict:
    """Pull bytes from store, route through ``pipeline.ingest_idea``.

    Bus-skill handler shape, like :func:`stage_payload`. ``args``
    reads:

    * ``artifact_id`` (required, str): id of a previously-
      staged store artifact.
    * ``hints`` (dict, default {}): forwarded to the future
      auto-detected dispatcher (not consumed today).
    * ``source_label`` (str, default ""): override for the
      ingest_idea source label; falls back to the artifact's
      ``producer`` when omitted.

    Returns ``{"idea_id", "artifact_id", "source_label",
    "hints"}`` so downstream consumers can trace lineage from
    the resulting idea back to the staged artifact.
    Auto-detected dispatch is the sister FR; today this skill
    treats the artifact as informal text and feeds it to the
    existing idea pipeline. ``hints`` is accepted but not yet
    consumed — wired through so future dispatcher logic can
    break ties without an API change here.

    Module-level for the same reason as :func:`stage_payload`.
    """
    artifact_id_raw = args.get("artifact_id", "")
    if not isinstance(artifact_id_raw, str):
        return {"error": "artifact_id must be a string"}
    artifact_id = artifact_id_raw.strip()
    if not artifact_id:
        return {"error": "artifact_id is required"}
    hints = args["hints"] if "hints" in args else {}
    if not isinstance(hints, dict):
        return {"error": "hints must be an object"}
    source_label_raw = args.get("source_label", "")
    if not isinstance(source_label_raw, str):
        return {"error": "source_label must be a string"}
    source_label_override = source_label_raw.strip()

    # Pull the artifact body. ``_INGEST_FETCH_CAP_CHARS`` matches
    # the bus's HARD_MAX_CHARS=20000 clamp so we ask for
    # what we can actually receive; larger payloads will return
    # ``truncated=True`` from store, and ingesting a partial
    # payload would mislead downstream tooling — handled by the
    # empty-content / future-streaming guards rather than
    # silently passing through.
    result = await agent.request(
        agent_type="store",
        operation="artifact_get",
        args={
            "id": artifact_id,
            "offset": 0,
            "max_chars": _INGEST_FETCH_CAP_CHARS,
        },
    )
    payload = _unwrap_request_envelope(result)
    if isinstance(payload, dict) and "error" in payload:
        return payload
    if not isinstance(payload, dict):
        return {"error": "store returned unexpected response shape"}
    if payload.get("truncated") is True:
        # Store / bus had to clamp the read at HARD_MAX_CHARS.
        # Ingesting partial content would produce an idea
        # whose claims and search queries don't reflect the
        # full source — surface the truncation as a clean
        # error so the caller can wait on streaming support
        # (out of scope FR) or split the source upstream.
        return {
            "error": (
                "store returned truncated content; "
                "ingest_from_artifact requires the full body"
            ),
        }
    # Accept any of ``text``, ``body``, ``content`` for the
    # artifact body. Bus's ``view_response`` uses ``text``;
    # ``/v1/artifacts/{id}/content`` puts the same payload
    # under ``content``; ``body`` is a historical alias kept
    # for backwards compatibility. Tolerating all three means
    # a future store surface tweak doesn't quietly turn a
    # successful fetch into "empty content" here.
    text = (
        payload.get("text")
        or payload.get("body")
        or payload.get("content")
        or ""
    )
    # ``text.strip()`` so a whitespace-only body — "\n\n", "   ",
    # an artifact whose only content is page-break newlines —
    # surfaces the empty-content error here rather than slipping
    # into ``ingest_idea`` with garbage.
    if not isinstance(text, str) or not text.strip():
        return {"error": "store returned empty content"}

    artifact_meta = (
        payload.get("artifact")
        if isinstance(payload.get("artifact"), dict)
        else {}
    )
    # Default the ingest source_label to the staged artifact's
    # producer so the resulting idea points back at where the
    # payload originated. Caller can override via
    # ``source_label``.
    source_label = source_label_override or str(artifact_meta.get("producer") or "")

    try:
        idea_id = await pipeline.ingest_idea(text, source_label)
    except RuntimeError as exc:
        return {"error": f"ingest failed: {exc}"}
    return {
        "idea_id": idea_id,
        "artifact_id": artifact_id,
        "source_label": source_label,
        "hints": hints,
    }


def _unwrap_request_envelope(result):
    """Pull ``result["result"]`` out of the bus request envelope.

    ``BaseAgent.request`` returns the raw bus response — the
    handler's return value lives nested under ``result``. Tests
    can fake either shape (full envelope or raw result) by
    routing through this helper.
    """
    if isinstance(result, dict):
        return result.get("result", result)
    return result


def create_researcher_agent(
    agent_id: str,
    bus_url: str,
    config_path: str,
) -> BaseAgent:
    """Build a researcher bus agent wrapping all MCP tools.

    Constructs the research pipeline + MCP server (same as
    ``researcher.server.main``), then wraps every MCP tool as a bus
    handler via ``BaseAgent.from_mcp()``.
    """
    from researcher.pipeline import create_pipeline
    from researcher.server import create_research_server

    pipeline = create_pipeline(config_path)
    mcp_server = create_research_server(pipeline)

    agent = BaseAgent.from_mcp(
        mcp_server,
        agent_type="researcher",
        agent_id=agent_id,
        bus_url=bus_url,
        config_path=config_path,
    )

    # Derive version from installed package metadata
    try:
        from importlib.metadata import version
        agent.version = version("khonliang-researcher")
    except Exception:
        agent.version = "0.0.0"

    logger.info(
        "Researcher agent %s created with %d skills",
        agent_id,
        len(agent.register_skills()),
    )

    _extend_with_native_handlers(agent, pipeline)

    return agent


def _extend_with_native_handlers(agent: BaseAgent, pipeline) -> None:
    """Attach native bus handlers on top of the MCP bridge."""
    original_register_skills = agent.register_skills

    def register_skills(self):
        skills = list(original_register_skills())
        names = {skill.name for skill in skills}
        extras = [
            Skill(
                "watch_ingest_queue",
                "Start a long-running ingest watcher publishing ingest.* bus events.",
                {"interval_s": {"type": "integer", "default": 5}},
            ),
            Skill(
                "list_ingest_watchers",
                "List active ingest watchers.",
                {},
            ),
            Skill(
                "stop_ingest_watcher",
                "Stop an ingest watcher.",
                {"watcher_id": {"type": "string", "required": True}},
            ),
            Skill(
                "stage_payload",
                "Persist raw ingest content as a store artifact, returning "
                "an artifact_id. Provenance metadata (source dict, kind_hint) "
                "is attached so the artifact is self-describing. Pair with "
                "ingest_from_artifact to ingest without re-transmitting the "
                "payload (re-route after misclassification, retry after a "
                "wedged worker, etc.). Routes through the bus to "
                "agent_type='store', operation='artifact_create'.",
                {
                    "content": {"type": "string", "required": True},
                    "kind_hint": {"type": "string", "default": ""},
                    "title": {"type": "string", "default": ""},
                    "content_type": {
                        "type": "string", "default": "text/plain",
                    },
                    "source": {"type": "object", "default": {}},
                },
                since="0.3.0",
            ),
            Skill(
                "ingest_from_artifact",
                "Ingest a previously-staged store artifact through the "
                "researcher idea pipeline. Pulls the body via the bus "
                "(agent_type='store', operation='artifact_get') and "
                "routes it through pipeline.ingest_idea (the canonical "
                "informal-text entry today). Returns {idea_id, "
                "artifact_id} so downstream consumers can trace the "
                "lineage. Auto-detected dispatch is a separate FR; the "
                "hints arg is wired through so future routing logic can "
                "break ties without an API change here.",
                {
                    "artifact_id": {"type": "string", "required": True},
                    "hints": {"type": "object", "default": {}},
                    "source_label": {"type": "string", "default": ""},
                },
                since="0.3.0",
            ),
            Skill(
                "ingest_github_async",
                "Schedule a GitHub-repo ingest as a background job. "
                "Returns {job_id, accepted_at} immediately; progress "
                "fires on bus topic 'research.ingest.progress'. Poll "
                "with ingest_status(job_id) for the race-free "
                "authority on terminal state. depth must be one of "
                "'readme' / 'readme+code' / 'full'. "
                "fr_researcher_2b22a2f3 + fr_researcher_bbf3cf69.",
                {
                    "repo_url": {"type": "string", "required": True},
                    "label": {"type": "string", "default": ""},
                    "depth": {"type": "string", "default": "readme+code"},
                },
                since="0.4.0",
            ),
            Skill(
                "ingest_file_async",
                "Schedule a local-file ingest as a background job. "
                "Returns {job_id, accepted_at} immediately; progress "
                "fires on 'research.ingest.progress'.",
                {"path": {"type": "string", "required": True}},
                since="0.4.0",
            ),
            Skill(
                "ingest_idea_async",
                "Schedule an idea-text ingest as a background job. "
                "Returns {job_id, accepted_at} immediately; progress "
                "fires on 'research.ingest.progress'.",
                {
                    "text": {"type": "string", "required": True},
                    "source_label": {"type": "string", "default": ""},
                },
                since="0.4.0",
            ),
            Skill(
                "ingest_status",
                "Look up an async ingest job's current phase, "
                "progress_pct, started_at, completed_at, result, "
                "error, and history. Race-free authority on terminal "
                "state — a fast job can move through started → done "
                "before a caller subscribed via bus_wait_for_event "
                "after the spawn returned, and ingest_status's "
                "history field replays every phase transition the "
                "job went through. Returns {error: 'not found'} when "
                "the job_id is unknown. Three causes: (a) the agent "
                "process restarted (the JobStore is in-memory and "
                "wipes on restart — including in-flight jobs); "
                "(b) the job is older than the completed-job "
                "retention cap (default 64); (c) the job_id was "
                "never issued by this agent.",
                {"job_id": {"type": "string", "required": True}},
                since="0.4.0",
            ),
        ]
        for skill in extras:
            if skill.name not in names:
                skills.append(skill)
        return skills

    async def _get_ingest_registry(self) -> IngestWatcherRegistry:
        registry = getattr(self, "_ingest_watcher_registry", None)
        if registry is None:
            store = IngestWatcherStore(str(pipeline.config.get("db_path", "data/researcher.db")))
            registry = IngestWatcherRegistry(
                store=store,
                publish=self.publish,
                snapshot_fn=pipeline.get_ingest_snapshot,
            )
            self._ingest_watcher_registry = registry
        return registry

    async def handle_watch_ingest_queue(self, args):
        interval_s = int(args.get("interval_s", 5))
        if interval_s <= 0:
            return {"error": "interval_s must be positive"}
        registry = await _get_ingest_registry(self)
        watcher_id = await registry.start(interval_s=interval_s)
        return {"watcher_id": watcher_id, "interval_s": interval_s}

    async def handle_list_ingest_watchers(self, args):
        registry = await _get_ingest_registry(self)
        return {"watchers": registry.list_watchers()}

    async def handle_stop_ingest_watcher(self, args):
        watcher_id = str(args.get("watcher_id", "")).strip()
        if not watcher_id:
            return {"error": "watcher_id is required"}
        registry = await _get_ingest_registry(self)
        stopped = await registry.stop(watcher_id)
        return {"watcher_id": watcher_id, "stopped": stopped}

    async def handle_stage_payload(self, args):
        return await stage_payload(self, args)

    async def handle_ingest_from_artifact(self, args):
        return await ingest_from_artifact(self, pipeline, args)

    def _get_job_store(self) -> IngestJobStore:
        store = getattr(self, "_ingest_job_store", None)
        if store is None:
            store = IngestJobStore()
            self._ingest_job_store = store
        return store

    def _get_ingest_semaphore(self) -> asyncio.Semaphore:
        """Bound the number of concurrent ingest workers.

        Without this, a burst of async-ingest calls can spawn an
        arbitrary number of repo clones / distill jobs in parallel
        and exhaust process resources long before the JobStore's
        completed-job retention cap helps. The cap is configurable
        via ``config.ingest_async_concurrency`` (default 4); jobs
        beyond the cap stay in ``phase=accepted`` until a slot
        opens, which subscribers see as a delayed
        ``accepted → started`` transition (with the wait time
        visible in the history timestamps).
        """
        sem = getattr(self, "_ingest_semaphore", None)
        if sem is None:
            cap = int(pipeline.config.get("ingest_async_concurrency", 4))
            sem = asyncio.Semaphore(max(1, cap))
            self._ingest_semaphore = sem
        return sem

    async def _spawn_ingest_job(self, skill: str, args: dict, work):
        """Common scaffolding for the three ingest_*_async skills.

        Creates a JobRecord, schedules a worker task gated on the
        agent's ingest semaphore, and returns
        ``{job_id, accepted_at, skill}`` immediately. ``work`` is an
        ``async (progress) -> dict`` coroutine that does the actual
        ingest and may call ``progress(phase, progress_pct=...,
        detail=...)`` at phase boundaries to fire
        ``research.ingest.progress`` bus events.

        Race note: events are best-effort for monitoring; a fast job
        can transition through ``started → done`` before a caller
        subscribed via ``bus_wait_for_event`` after this call
        returned. ``ingest_status(job_id)`` is the race-free authority
        for terminal state, and its ``history`` field replays every
        phase transition the job went through.

        Lifecycle: every spawned task is retained in
        ``self._ingest_tasks`` so ``shutdown()`` can cancel still-
        running ingests instead of letting them publish progress
        events into a closed connector. The task removes itself from
        the set when it completes.
        """
        store = self._get_job_store()
        job = await store.create(skill, args)
        semaphore = self._get_ingest_semaphore()

        async def driver():
            # Bound concurrency: jobs beyond ``ingest_async_concurrency``
            # park at ``phase=accepted`` until a slot opens. The wait
            # time shows up as the accepted_at→started_at delta in the
            # job's history, so a subscriber can spot pile-ups.
            try:
                async with semaphore:
                    await run_ingest_job(store, self.publish, job, work)
            except asyncio.CancelledError:
                # Two cancel sites converge here:
                # (a) cancelled DURING ``run_ingest_job`` —
                #     ``run_ingest_job`` already recorded
                #     ``phase=error`` and re-raised, so phase is no
                #     longer ``accepted`` and we just propagate.
                # (b) cancelled WHILE QUEUED on the semaphore —
                #     ``run_ingest_job`` never entered, so the
                #     ``phase=accepted`` JobRecord would otherwise
                #     stay stuck forever and ``ingest_status``
                #     callers would poll indefinitely. Detect that
                #     case by checking the current phase and record
                #     the cancellation so the supervision surface
                #     stays honest.
                current = await store.get(job.job_id)
                if current is not None and current.phase == "accepted":
                    await store.set_error(
                        job.job_id, "CancelledError: cancelled before start",
                    )
                    await store.transition(job.job_id, phase="error")
                raise

        # Schedule and retain — progress events + ingest_status are
        # the supervision surface for happy-path monitoring, but the
        # agent also needs a handle on each task so ``shutdown()``
        # can cancel them rather than leaving repo clones / LLM
        # work running after the connector is closed.
        tasks = getattr(self, "_ingest_tasks", None)
        if tasks is None:
            tasks = set()
            self._ingest_tasks = tasks
        task = asyncio.create_task(driver(), name=f"ingest-job-{job.job_id}")
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {
            "job_id": job.job_id,
            "skill": job.skill,
            "accepted_at": job.accepted_at,
        }

    _VALID_INGEST_DEPTHS = ("readme", "readme+code", "full")

    async def handle_ingest_github_async(self, args):
        repo_url = str(args.get("repo_url", "")).strip()
        if not repo_url:
            return {"error": "repo_url is required"}
        label = str(args.get("label", ""))
        depth = str(args.get("depth", "readme+code")).strip()
        # Validate at the API boundary so a typo or surrounding
        # whitespace doesn't silently degrade to README-only ingest
        # while still reporting back as if the requested depth were
        # honoured (raised by Copilot review on PR #37).
        if depth not in _VALID_INGEST_DEPTHS:
            return {
                "error": (
                    f"invalid depth: {depth!r} "
                    f"(expected one of {list(_VALID_INGEST_DEPTHS)})"
                ),
            }

        async def work(progress):
            result = await pipeline.ingest_github_repo(
                repo_url, label=label, depth=depth,
                progress_callback=progress,
            )
            # ``ingest_github_repo`` reports invalid URLs / clone
            # failures by returning ``{"error": "..."}`` instead of
            # raising. Translate that into an exception here so
            # ``run_ingest_job`` surfaces ``phase=error`` and stores
            # the message; otherwise subscribers would see
            # ``phase=done`` on a failed ingest (Copilot review on
            # PR #37).
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(result["error"])
            return result

        return await self._spawn_ingest_job(
            "ingest_github", {"repo_url": repo_url, "label": label, "depth": depth}, work,
        )

    async def handle_ingest_file_async(self, args):
        path = str(args.get("path", "")).strip()
        if not path:
            return {"error": "path is required"}

        async def work(progress):
            from researcher.fetcher import fetch_file
            await progress("distilling", progress_pct=20)
            result = await fetch_file(path)
            if not result.content.strip():
                # Empty extracted text means there is no entry_id to
                # hand back. ``run_ingest_job`` would otherwise mark
                # this ``phase=done``, which would mislead a caller
                # into thinking the ingest succeeded. Surface as an
                # error so the worker fires ``phase=error`` and the
                # message is preserved on the job. Don't fire the
                # ``storing`` phase before this check: ``storing``
                # implies a write was attempted and a phase-based
                # monitor would otherwise see ``storing → error``
                # for an ingest that never tried to persist anything.
                raise RuntimeError(
                    f"no text content extracted from: {path}"
                )
            await progress("storing", progress_pct=70)
            import hashlib
            from khonliang.knowledge.store import EntryStatus, KnowledgeEntry, Tier
            entry_id = hashlib.sha256(path.encode()).hexdigest()[:16]
            entry = KnowledgeEntry(
                id=entry_id,
                tier=Tier.IMPORTED,
                title=result.title or path,
                content=result.content,
                source=result.url,
                scope="research",
                tags=["paper", f"format:{result.format.value}"],
                status=EntryStatus.INGESTED,
                metadata={
                    "url": result.url,
                    "format": result.format.value,
                    "fetched_at": result.fetched_at,
                    **result.metadata,
                },
            )
            pipeline.knowledge.add(entry)
            return {"entry_id": entry_id, "title": entry.title, "format": result.format.value}

        return await self._spawn_ingest_job("ingest_file", {"path": path}, work)

    async def handle_ingest_idea_async(self, args):
        text = str(args.get("text", ""))
        if not text.strip():
            return {"error": "text is required"}
        source_label = str(args.get("source_label", ""))

        async def work(progress):
            await progress("distilling", progress_pct=30)
            idea_id = await pipeline.ingest_idea(text, source_label)
            await progress("storing", progress_pct=80)
            return {"idea_id": idea_id, "source_label": source_label}

        return await self._spawn_ingest_job(
            "ingest_idea", {"source_label": source_label}, work,
        )

    async def handle_ingest_status(self, args):
        job_id = str(args.get("job_id", "")).strip()
        if not job_id:
            return {"error": "job_id is required"}
        store = self._get_job_store()
        job = await store.get(job_id)
        if job is None:
            return {"error": "not found", "job_id": job_id}
        return job.to_status()

    async def start(self):
        skills = self._all_skills()
        collabs = self.register_collaborations()
        self._connector = BusConnector(
            bus_url=self.bus_url,
            agent_id=self.agent_id,
            on_request=self._dispatch_request,
        )
        try:
            await self._connector.connect_and_register(
                agent_type=self.agent_type,
                version=self.version,
                pid=os.getpid(),
                skills=[s.to_dict() for s in skills],
                collaborations=[
                    {
                        "name": c.name,
                        "description": c.description,
                        "requires": c.requires,
                        "steps": c.steps,
                    }
                    for c in collabs
                ],
            )
        except Exception:
            await self._http.aclose()
            raise

        registry = await _get_ingest_registry(self)
        await registry.rehydrate()

        logger.info(
            "Agent %s started (%d skills, WebSocket)",
            self.agent_id,
            len(skills),
        )
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        except NotImplementedError:
            pass
        try:
            await self._connector.run()
        finally:
            await registry.shutdown()
            await self._http.aclose()

    async def shutdown(self):
        registry = getattr(self, "_ingest_watcher_registry", None)
        if registry is not None:
            await registry.shutdown()
        # Cancel any in-flight async ingest jobs so they don't publish
        # progress events into a connector that's about to close.
        # We snapshot the set first so the per-task ``done_callback``
        # mutation doesn't race the iteration. Best-effort: a task
        # already in the middle of an awaited library call observes
        # ``CancelledError`` on the next checkpoint, which
        # ``run_ingest_job`` translates into ``phase=error`` with
        # ``CancelledError`` recorded; tasks queued on the semaphore
        # are handled by the driver's own except branch.
        #
        # Bounded with a hard timeout so a single slow operation —
        # e.g. ``repo_tree()`` blocking inside a 120s ``git clone``
        # subprocess that doesn't observe cancellation promptly —
        # can't hold the agent's shutdown indefinitely. After the
        # timeout we abandon the survivors: detach them from the
        # ingest-task set AND silence the
        # ``Task was destroyed but it is pending!`` warning that
        # ``asyncio.run`` would otherwise emit at loop teardown.
        # The survivors continue to run on the loop until they
        # observe cancellation or finish naturally; the agent's
        # ``shutdown()`` returns within the bound and the bus
        # connector closes regardless. Loop teardown will eventually
        # collect them, but the agent is already unregistered.
        tasks = list(getattr(self, "_ingest_tasks", ()) or ())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.shield(
                        asyncio.gather(*tasks, return_exceptions=True),
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                survivors = [t for t in tasks if not t.done()]
                for t in survivors:
                    # Suppress the runtime warning at loop teardown.
                    # Setting ``_log_destroy_pending`` is the
                    # documented escape hatch for "I deliberately
                    # abandoned this task and accept the
                    # consequences."
                    t._log_destroy_pending = False
                logger.warning(
                    "ingest shutdown abandoned %d task(s) still running "
                    "after 10s cancel grace period; they will be "
                    "collected at loop teardown",
                    len(survivors),
                )
        await BaseAgent.shutdown(self)

    import signal

    agent.register_skills = MethodType(register_skills, agent)
    agent._handlers["watch_ingest_queue"] = MethodType(handle_watch_ingest_queue, agent)
    agent._handlers["list_ingest_watchers"] = MethodType(handle_list_ingest_watchers, agent)
    agent._handlers["stop_ingest_watcher"] = MethodType(handle_stop_ingest_watcher, agent)
    agent._handlers["stage_payload"] = MethodType(handle_stage_payload, agent)
    agent._handlers["ingest_from_artifact"] = MethodType(handle_ingest_from_artifact, agent)
    agent._handlers["ingest_github_async"] = MethodType(handle_ingest_github_async, agent)
    agent._handlers["ingest_file_async"] = MethodType(handle_ingest_file_async, agent)
    agent._handlers["ingest_idea_async"] = MethodType(handle_ingest_idea_async, agent)
    agent._handlers["ingest_status"] = MethodType(handle_ingest_status, agent)
    agent._get_job_store = MethodType(_get_job_store, agent)
    agent._get_ingest_semaphore = MethodType(_get_ingest_semaphore, agent)
    agent._spawn_ingest_job = MethodType(_spawn_ingest_job, agent)
    agent.start = MethodType(start, agent)
    agent.shutdown = MethodType(shutdown, agent)


def main():
    """CLI entry point for the researcher agent."""
    import argparse

    from khonliang_bus import add_version_flag

    # Check for install/uninstall commands before full agent init
    parser = argparse.ArgumentParser(
        prog="researcher.agent",
        description="khonliang-researcher bus agent",
    )
    add_version_flag(parser)
    parser.add_argument("command", nargs="?", choices=["install", "uninstall"],
                        help="install or uninstall from the bus")
    parser.add_argument("--id", default="researcher-primary")
    parser.add_argument("--bus", default="http://localhost:8787")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command in ("install", "uninstall"):
        # Use BaseAgent's CLI handling for install/uninstall
        # (lightweight — doesn't build the full pipeline)
        BaseAgent.from_cli([
            args.command,
            "--id", args.id,
            "--bus", args.bus,
            "--config", args.config,
        ])
        return

    # Full agent startup — builds pipeline, wraps MCP tools
    agent = create_researcher_agent(
        agent_id=args.id,
        bus_url=args.bus,
        config_path=args.config,
    )
    asyncio.run(agent.start())


if __name__ == "__main__":
    main()
