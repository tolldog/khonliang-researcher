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
import json
import logging
import os
import sys
from types import MethodType

from khonliang_bus import BaseAgent, Skill, Welcome, WelcomeEntryPoint
from khonliang_bus.connector import BusConnector

from researcher.ingest_jobs import (
    IngestJobStore,
    IngestQueueFull,
    _publish_progress,
    run_ingest_job,
)
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
    # ``view_response`` shape; khonliang-librarian's
    # ``LibrarianAgent._artifact_id`` treats both as valid too).
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


async def distill_repo_docs_handler(
    agent: BaseAgent, pipeline, args: dict,
) -> dict:
    """Bus-skill handler for ``distill_repo_docs``.

    Validates ``args``, builds the store-routing closure, and delegates to
    :func:`researcher.repo_docs.distill_repo_docs`. Module-level so tests
    can drive it without wiring through ``BaseAgent.from_mcp`` — pass a
    fake agent (with ``request`` + ``agent_id``) and a fake pipeline (with
    ``pool``).
    """
    from researcher.repo_docs import distill_repo_docs

    content_raw = args.get("content")
    if not isinstance(content_raw, dict):
        return {"error": "content must be an object mapping path -> body"}
    if not content_raw:
        return {"error": "content is required"}
    # Defer per-file type-checking to repo_docs.normalize_corpus — it raises
    # TypeError with the offending path, which we surface as a clean envelope.
    repo_name_raw = args.get("repo_name", "")
    if not isinstance(repo_name_raw, str):
        return {"error": "repo_name must be a string"}
    model_role_raw = args.get("model_role", "summarizer")
    if not isinstance(model_role_raw, str) or not model_role_raw.strip():
        return {"error": "model_role must be a non-empty string"}
    prompt_version_raw = args.get("prompt_version", "v1")
    if not isinstance(prompt_version_raw, str) or not prompt_version_raw.strip():
        return {"error": "prompt_version must be a non-empty string"}

    async def store_request(operation: str, op_args: dict) -> dict:
        return await agent.request(
            agent_type="store",
            operation=operation,
            args=op_args,
        )

    try:
        return await distill_repo_docs(
            content=content_raw,
            pool=pipeline.pool,
            store_request=store_request,
            repo_name=repo_name_raw.strip(),
            model_role=model_role_raw.strip(),
            prompt_version=prompt_version_raw.strip(),
            producer=getattr(agent, "agent_id", ""),
        )
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}


async def ask_librarian(agent: BaseAgent, args: dict) -> dict:
    """Bus-skill handler proxying to librarian-primary (fr_researcher_11e9524a).

    Thin pass-through over ``researcher.librarian_client.call_librarian`` —
    lets an external MCP/bus consumer that only knows about researcher-primary
    reach librarian's classification/taxonomy/gap-finding skills without
    needing to discover and address ``librarian-primary`` itself. ``args``
    reads:

    * ``operation`` (required, str): one of librarian's 7 skill names
      (classify_paper, taxonomy_report, rebuild_neighborhoods,
      suggest_missing_nodes, promote_investigation, identify_gaps,
      library_health).
    * ``librarian_args`` (dict, default {}): forwarded verbatim as that
      skill's args.

    Never raises: librarian being down, slow, or unregistered degrades to
    ``{"available": False, "reason": ...}`` rather than failing the caller
    (optional-coordinator principle — librarian absence degrades quality,
    not function). Module-level so tests can call it with a mock agent.
    """
    from researcher.librarian_client import call_librarian

    if not isinstance(args, dict):
        return {"error": "args must be an object"}
    operation_raw = args.get("operation", "")
    if not isinstance(operation_raw, str):
        return {"error": "operation must be a string"}
    operation = operation_raw.strip()
    if not operation:
        return {"error": "operation is required"}
    librarian_args = args.get("librarian_args", {})
    if not isinstance(librarian_args, dict):
        return {"error": "librarian_args must be an object"}
    return await call_librarian(agent, operation, librarian_args)


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

    # Re-derive the SelfCatalog's ownership from the REAL running bus
    # agent_id, not whatever config["bus_agent_id"] happened to say (or its
    # "researcher-primary" default). pipeline.py builds `self.catalog`
    # before any agent_id exists — it's shared by both this bus-agent entry
    # point and the transport-agnostic MCP-stdio one (researcher.server),
    # neither of which pipeline.py itself knows about. Rebuilding here means
    # a custom `--id` (a second, domain-scoped researcher instance) always
    # wins: its catalog rows, and its later register_source call, are
    # stamped/advertised under the id this process actually registers on
    # the bus with — codex P1, a config drift (bus_agent_id != --id) would
    # otherwise silently mis-stamp every row and let two instances
    # overwrite each other's librarian registration.
    if getattr(pipeline, "catalog", None) is not None:
        from researcher.self_catalog import build_self_catalog

        pipeline.catalog = build_self_catalog(pipeline.config, owner_agent=agent_id)

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

    # Cold-start orientation surface (fr_khonliang-bus-lib_6a82732c).
    # ``BaseAgent.from_mcp`` returns a BaseAgent instance, not a subclass,
    # so we set WELCOME as an instance attribute (Python resolves
    # instance-attr before class-attr — handle_welcome reads self.WELCOME).
    agent.WELCOME = Welcome(
        role="ingest + corpus authority",
        mission=(
            "Ingests external knowledge — papers, RSS feeds, GitHub repos, "
            "free-form text — and distills it into a queryable corpus. "
            "Surfaces evidence on demand for any consumer agent. Corpus "
            "health and taxonomy belong to librarian; FR lifecycle belongs "
            "to developer."
        ),
        not_responsible_for=[
            "FR / spec / milestone lifecycle (developer)",
            "corpus classification + taxonomy (librarian)",
            "code review (reviewer)",
        ],
        delegates_to={
            "developer": "FR / spec / milestone lifecycle changes",
            "librarian": "classification, taxonomy, neighborhoods, gaps — standalone bus agent (librarian-primary); reach it directly, or via ask_librarian if you only address researcher",
            "store": "artifact-mediated large payloads (stage_payload / ingest_from_artifact)",
        },
        entry_points=[
            WelcomeEntryPoint(
                skill="brief_on",
                when_to_use="topic-in-context brief over the corpus — multi-query retrieval, reuses distilled summaries",
            ),
            WelcomeEntryPoint(
                skill="find_relevant",
                when_to_use="embedding-based corpus search by topic; filter by project relevance",
            ),
            WelcomeEntryPoint(
                skill="fetch_paper",
                when_to_use="ingest a paper / blog post / arxiv URL into the corpus",
            ),
            WelcomeEntryPoint(
                skill="stage_payload",
                when_to_use="persist raw ingest content as a store artifact for ingest_from_artifact later",
            ),
            WelcomeEntryPoint(
                skill="distill_paper",
                when_to_use="run LLM distillation on a stored paper — produces summary + triples + applicability",
            ),
            WelcomeEntryPoint(
                skill="ask_librarian",
                when_to_use="proxy a classify_paper/taxonomy_report/identify_gaps/etc. call to librarian-primary without addressing it directly; degrades to available:false rather than failing if librarian is down",
            ),
        ],
        guide_skill="research_guide",
    )

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

    # Built once (not None only when pipeline.catalog is initialized) and
    # reused by every catalog_* handler below — CatalogSkills is a thin
    # stateless wrapper over the same long-lived SelfCatalog instance.
    from researcher.self_catalog import build_catalog_skills

    # getattr, not pipeline.catalog: some tests wire a bare stub/SimpleNamespace
    # pipeline through here with no catalog attribute at all — that must
    # behave like catalog=None (self-catalog disabled), not AttributeError.
    catalog_skills = build_catalog_skills(getattr(pipeline, "catalog", None))

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
                "ingest_url_with_body",
                "Ingest a URL whose page body was fetched OUTSIDE the service "
                "(browser-grade WebFetch, Playwright, an external distiller) — "
                "the recovery path when fetch_paper is blocked (403/429/503, or "
                "a known anti-bot host) and the "
                "service can't retrieve the page itself. Stores an entry in the "
                "same Tier.IMPORTED / paper / INGESTED shape as fetch_paper "
                "success, then the distillation worker picks it up. Returns "
                "{entry_id, url, source} where url is the caller's input and "
                "source is the stored canonical (arxiv-normalized) URL used for "
                "dedupe/backlinks.",
                {
                    "url": {"type": "string", "required": True},
                    "body": {"type": "string", "required": True},
                    "title": {"type": "string", "default": ""},
                    "content_type": {"type": "string", "default": "text/markdown"},
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
            Skill(
                "distill_repo_docs",
                "Distill a repository's docs corpus (READMEs, "
                "ARCHITECTURE.md, conventions, CLAUDE.md, ...) into a "
                "compact bulleted list of normative claims — invariants, "
                "must / must-not rules, architectural decisions — suitable "
                "for prompt augmentation during code review. Cached by "
                "content sha256 + model + prompt_version: identical inputs "
                "return the existing artifact without re-running the LLM. "
                "The stored artifact lives on store-primary as "
                "kind='researcher_distillation' with a deterministic "
                "art_repodocs_<hex> id. See fr_researcher_86a810a3.",
                {
                    "content": {"type": "object", "required": True},
                    "repo_name": {"type": "string", "default": ""},
                    "model_role": {"type": "string", "default": "summarizer"},
                    "prompt_version": {"type": "string", "default": "v1"},
                },
                since="0.5.0",
            ),
            Skill(
                "distill_paper_async",
                "Schedule LLM distillation of a stored paper as a background "
                "job. The synchronous distill_paper blocks the agent and times "
                "out over MCP on large papers (summarize is a 7B-model call on "
                "the full body); this returns {job_id, accepted_at} immediately "
                "and runs the distill off the request path. Progress fires on "
                "'research.ingest.progress'; poll ingest_status(job_id) for the "
                "race-free terminal state, whose result carries {entry_id, "
                "title, success, triples, assessments} — where `triples` and "
                "`assessments` are integer COUNTS, not the payloads (call "
                "paper_digest / paper_context for the full distilled content). "
                "bug_khonliang-researcher_d4068c16.",
                {"entry_id": {"type": "string", "required": True}},
                since="0.5.0",
            ),
            Skill(
                "ask_librarian",
                "Proxy a call to one of librarian-primary's 7 skills "
                "(classify_paper, taxonomy_report, rebuild_neighborhoods, "
                "suggest_missing_nodes, promote_investigation, "
                "identify_gaps, library_health) over the bus, for callers "
                "that only address researcher-primary. Routes through "
                "agent_type='librarian'. Never fails the caller — if "
                "librarian-primary is down, slow, or unreachable, returns "
                "{available: false, reason} instead of raising "
                "(optional-coordinator principle: librarian absence "
                "degrades quality, not function). fr_researcher_11e9524a.",
                {
                    "operation": {"type": "string", "required": True},
                    "librarian_args": {"type": "object", "default": {}},
                },
                since="0.6.0",
            ),
            Skill(
                "catalog_query",
                "Structured query over researcher's own SelfCatalog index "
                "cards (papers/ideas) — the librarian's federation surface, "
                "callable directly too. project is mandatory (isolation is "
                "load-bearing). filters match kind/record_id/schema_version/"
                "embedding_status columns, updated_after/updated_before date "
                "bounds, or facet keys. Returns {error: ...} when the "
                "catalog is disabled (no db_path configured). "
                "fr_researcher_bbe95f12.",
                {
                    "project": {"type": "string", "required": True},
                    "filters": {"type": "object", "default": None},
                    "jmespath_expr": {"type": "string", "default": None},
                    "fields": {"type": "array", "default": None},
                    "limit": {"type": "integer", "default": 100},
                    "scope": {"type": "string", "default": "project"},
                },
                since="0.6.0",
            ),
            Skill(
                "catalog_search",
                "Similarity search over researcher's SelfCatalog index-card "
                "text (title + abstract/summary tier — never full paper "
                "bodies). Vector search requires a query_vector AND embedded "
                "rows; otherwise falls back to text LIKE. fr_researcher_bbe95f12.",
                {
                    "project": {"type": "string", "required": True},
                    "query_text": {"type": "string", "default": None},
                    "limit": {"type": "integer", "default": 20},
                    "query_vector": {"type": "string", "default": None},
                },
                since="0.6.0",
            ),
            Skill(
                "catalog_stats",
                "SelfCatalog health counters (by kind/status, spec versions, "
                "embedding backlog), optionally scoped to one project. "
                "fr_researcher_bbe95f12.",
                {"project": {"type": "string", "default": None}},
                since="0.6.0",
            ),
            Skill(
                "list_since",
                "SelfCatalog records updated after since_ts (epoch seconds), "
                "oldest first — registered under this exact name (not "
                "catalog_list_since) because the librarian's own federation "
                "code calls every source's resync primitive as 'list_since' "
                "(CatalogSkills.list_since). fr_researcher_bbe95f12.",
                {
                    "project": {"type": "string", "required": True},
                    "since_ts": {"type": "number", "required": True},
                    "limit": {"type": "integer", "default": 100},
                },
                since="0.6.0",
            ),
            Skill(
                "catalog_fetch",
                "Exact-id lookup for one corpus entry (paper or idea) by its "
                "SelfCatalog record_id — this IS the entry's KnowledgeEntry "
                "id. Distinct from paper_context/paper_digest (fuzzy, "
                "multi-result search over the whole corpus): this is what "
                "an IndexRecord's `ref` field points at for bounded, exact "
                "expansion back to the record it describes. Returns "
                "{error: 'not found'} for an unknown record_id. "
                "fr_researcher_bbe95f12.",
                {"record_id": {"type": "string", "required": True}},
                since="0.6.0",
            ),
            Skill(
                "catalog_mark_stale",
                "Bulk-flag researcher's SelfCatalog rows for a project "
                "pending re-embedding after a CatalogSpec version bump. Call "
                "with the FULL new spec (not just its version) when the "
                "ecosystem's library.catalog_spec_published event fires. "
                "fr_researcher_bbe95f12.",
                {
                    "project": {"type": "string", "required": True},
                    "spec": {"type": "object", "required": True},
                },
                since="0.6.0",
            ),
            Skill(
                "catalog_backfill",
                "One-time (idempotent) catalog backfill for corpus entries "
                "that predate self-cataloging — the two completion-path "
                "hooks (distill/ingest_idea) only publish index cards for "
                "FUTURE ingests, so an upgrade against an already-populated "
                "corpus needs this run once to make catalog_query/"
                "catalog_search see the pre-existing dataset. Walks the "
                "WHOLE knowledge store; NOT run automatically on agent "
                "startup (unbounded work against a live, actively-written "
                "db) — call manually post-deploy, or re-run any time "
                "(already-cataloged entries are skipped, so it's cheap once "
                "caught up). Returns {papers, ideas, skipped, errors}. "
                "fr_researcher_bbe95f12.",
                {},
                since="0.6.0",
            ),
        ]
        # Skills whose handlers hard-depend on catalog_skills (built from
        # pipeline.catalog) and return {"error": ...} for every call when
        # it's None — i.e. self-cataloging is disabled (no db_path
        # configured, or khonliang-librarian-lib isn't installed). Don't
        # advertise them in that case: a client picking capabilities from
        # register_skills() has no way to know these are dead until it
        # actually calls one, and the skill list should reflect what the
        # agent can actually do. catalog_fetch is deliberately NOT in this
        # set — its handler reads pipeline.knowledge directly and works
        # regardless of whether self-cataloging is enabled.
        _catalog_dependent_skills = {
            "catalog_query",
            "catalog_search",
            "catalog_stats",
            "list_since",
            "catalog_mark_stale",
            "catalog_backfill",
        }
        for skill in extras:
            if skill.name in _catalog_dependent_skills and catalog_skills is None:
                continue
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
        interval_s = args.get("interval_s", 5)
        # Strict isinstance (rejecting bool, which is an int subclass) like the
        # other handlers — ``int(...)`` would crash on a non-numeric string
        # with an unhandled ValueError instead of returning an error envelope,
        # and silently truncate floats / coerce bools.
        if not isinstance(interval_s, int) or isinstance(interval_s, bool):
            return {"error": "interval_s must be an integer"}
        if interval_s <= 0:
            return {"error": "interval_s must be positive"}
        registry = await _get_ingest_registry(self)
        watcher_id = await registry.start(interval_s=interval_s)
        return {"watcher_id": watcher_id, "interval_s": interval_s}

    async def handle_list_ingest_watchers(self, args):
        registry = await _get_ingest_registry(self)
        return {"watchers": registry.list_watchers()}

    async def handle_stop_ingest_watcher(self, args):
        watcher_id_raw = args.get("watcher_id", "")
        # Strict type check rather than str()-coercion, which would stringify
        # None/123/objects into a bogus id ("None", "123") and mask a caller
        # type error as a silent "watcher not found".
        if not isinstance(watcher_id_raw, str):
            return {"error": "watcher_id must be a string"}
        watcher_id = watcher_id_raw.strip()
        if not watcher_id:
            return {"error": "watcher_id is required"}
        registry = await _get_ingest_registry(self)
        stopped = await registry.stop(watcher_id)
        return {"watcher_id": watcher_id, "stopped": stopped}

    async def handle_stage_payload(self, args):
        return await stage_payload(self, args)

    async def handle_ingest_from_artifact(self, args):
        return await ingest_from_artifact(self, pipeline, args)

    async def handle_ask_librarian(self, args):
        return await ask_librarian(self, args)

    async def handle_ingest_url_with_body(self, args):
        # Strict isinstance validation like the sibling handlers — never
        # str()-coerce, so caller type bugs surface as a clean error.
        url_raw = args.get("url", "")
        if not isinstance(url_raw, str):
            return {"error": "url must be a string"}
        url = url_raw.strip()
        if not url:
            return {"error": "url is required"}
        # Contract is "Ingest a URL" — reject non-http(s)/non-hostname inputs so
        # they can't pollute dedupe/backlinking or store a non-URL `source`.
        from researcher.fetcher import is_http_url

        if not is_http_url(url):
            return {"error": "url must be an absolute http(s) URL"}
        # Distinguish missing (required-field error) from wrong type, like
        # stage_payload does for `content`.
        if "body" not in args:
            return {"error": "body is required"}
        body = args.get("body")
        if not isinstance(body, str):
            return {"error": "body must be a string"}
        if not body.strip():
            return {"error": "body is required"}
        title_raw = args.get("title", "")
        if not isinstance(title_raw, str):
            return {"error": "title must be a string"}
        content_type_raw = args.get("content_type", "text/markdown")
        if not isinstance(content_type_raw, str):
            return {"error": "content_type must be a string"}
        content_type = content_type_raw.strip() or "text/markdown"
        try:
            entry_id = await pipeline.ingest_url_with_body(
                url, body, title=title_raw.strip(), content_type=content_type,
            )
        except Exception as e:  # noqa: BLE001 — surface as a clean error envelope
            # Return/log only the exception type — its str can embed the
            # caller URL's query tokens/userinfo, which would leak into bus
            # transcripts. Matches the MCP tool's sanitized error path.
            logger.warning("ingest_url_with_body bus handler failed: %s", type(e).__name__)
            return {"error": f"ingest failed: {type(e).__name__}"}
        if not entry_id:
            return {"error": "no extractable content in body"}
        # Echo the stored entry's canonical `source` (arxiv-normalized the same
        # way the pipeline does) so callers can dedupe/backlink without
        # re-implementing the rule. `url` stays the caller's original input.
        from researcher.fetcher import extract_arxiv_id

        arxiv_id = extract_arxiv_id(url)
        source = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url
        return {"entry_id": entry_id, "url": url, "source": source}

    async def handle_distill_repo_docs(self, args):
        return await distill_repo_docs_handler(self, pipeline, args)

    async def handle_catalog_query(self, args):
        if catalog_skills is None:
            return {"error": "self-catalog is disabled (no db_path configured)"}
        project_raw = args.get("project", "")
        if not isinstance(project_raw, str) or not project_raw.strip():
            return {"error": "project is required"}
        return catalog_skills.catalog_query(
            project_raw.strip(),
            filters=args.get("filters"),
            jmespath_expr=args.get("jmespath_expr"),
            fields=args.get("fields"),
            limit=args.get("limit", 100),
            scope=args.get("scope", "project"),
        )

    async def handle_catalog_search(self, args):
        if catalog_skills is None:
            return {"error": "self-catalog is disabled (no db_path configured)"}
        project_raw = args.get("project", "")
        if not isinstance(project_raw, str) or not project_raw.strip():
            return {"error": "project is required"}
        return catalog_skills.catalog_search(
            project_raw.strip(),
            args.get("query_text"),
            limit=args.get("limit", 20),
            query_vector=args.get("query_vector"),
        )

    async def handle_catalog_stats(self, args):
        if catalog_skills is None:
            return {"error": "self-catalog is disabled (no db_path configured)"}
        return catalog_skills.catalog_stats(project=args.get("project"))

    async def handle_list_since(self, args):
        if catalog_skills is None:
            return {"error": "self-catalog is disabled (no db_path configured)"}
        project_raw = args.get("project", "")
        if not isinstance(project_raw, str) or not project_raw.strip():
            return {"error": "project is required"}
        since_ts = args.get("since_ts")
        if since_ts is None:
            return {"error": "since_ts is required"}
        try:
            since_ts = float(since_ts)
        except (TypeError, ValueError):
            return {"error": "since_ts must be a number"}
        return catalog_skills.list_since(
            project_raw.strip(), since_ts, limit=args.get("limit", 100)
        )

    async def handle_catalog_mark_stale(self, args):
        if catalog_skills is None:
            return {"error": "self-catalog is disabled (no db_path configured)"}
        project_raw = args.get("project", "")
        if not isinstance(project_raw, str) or not project_raw.strip():
            return {"error": "project is required"}
        spec = args.get("spec")
        if not isinstance(spec, dict):
            return {"error": "spec must be an object (the full CatalogSpec)"}
        return catalog_skills.catalog_mark_stale(project_raw.strip(), spec)

    async def handle_catalog_fetch(self, args):
        record_id_raw = args.get("record_id", "")
        if not isinstance(record_id_raw, str) or not record_id_raw.strip():
            return {"error": "record_id is required"}
        record_id = record_id_raw.strip()
        entry = pipeline.knowledge.get(record_id)
        if entry is None:
            return {"error": "not found", "record_id": record_id}
        # Papers store their distilled summary under a sibling id
        # (f"{entry.id}_summary", see pipeline._store_distillation) rather
        # than on the entry itself — surface it alongside the raw entry so
        # a `ref` follow (paper kind) gets the abstract-tier content the
        # catalog card actually described, not just the entry's metadata.
        summary_entry = pipeline.knowledge.get(f"{record_id}_summary")
        summary = None
        if summary_entry is not None:
            try:
                summary = json.loads(summary_entry.content)
            except (json.JSONDecodeError, TypeError):
                summary = None
        return {
            "record_id": record_id,
            "title": entry.title,
            "content": entry.content if summary is None else None,
            "summary": summary,
            "url": entry.metadata.get("url", ""),
            "status": str(getattr(entry, "status", "")),
        }

    async def handle_catalog_backfill(self, args):
        if catalog_skills is None:
            return {"error": "self-catalog is disabled (no db_path configured)"}
        return pipeline.backfill_self_catalog()

    def _get_job_store(self) -> IngestJobStore:
        store = getattr(self, "_ingest_job_store", None)
        if store is None:
            # max_inflight bounds accepted-but-not-finished jobs (the semaphore
            # only bounds *running* ones); a burst of spawns beyond it is
            # rejected synchronously rather than queuing unbounded parked tasks.
            max_inflight = int(pipeline.config.get("ingest_max_inflight", 128))
            store = IngestJobStore(max_inflight=max(1, max_inflight))
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
        try:
            job = await store.create(skill, args)
        except IngestQueueFull as e:
            # Backpressure: the in-flight cap is hit. Reject synchronously with
            # an error envelope — no JobRecord, no parked task — so the caller
            # can retry/back off instead of growing an unbounded queue.
            logger.warning("ingest job rejected (queue full): %s", e)
            return {"error": str(e), "retryable": True}
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
                #     ``phase=error`` and emitted the matching
                #     progress event, then re-raised. Phase is no
                #     longer ``accepted`` so we skip the recovery
                #     block and just propagate.
                # (b) cancelled WHILE QUEUED on the semaphore —
                #     ``run_ingest_job`` never entered, so the
                #     ``phase=accepted`` JobRecord would otherwise
                #     stay stuck forever and ``ingest_status``
                #     callers would poll indefinitely. Detect that
                #     case by checking the current phase, record
                #     the cancellation, AND emit the matching
                #     ``research.ingest.progress`` event so an
                #     event-driven subscriber sees the terminal
                #     phase rather than only the supervision-poll
                #     state.
                current = await store.get(job.job_id)
                if current is not None and current.phase == "accepted":
                    await store.set_error(
                        job.job_id, "CancelledError: cancelled before start",
                    )
                    final = await store.transition(job.job_id, phase="error")
                    if final is not None:
                        await _publish_progress(self.publish, final)
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
        # ``isinstance(str)``-validate at the API boundary rather
        # than ``str()``-coerce. ``str(None)`` / ``str(123)`` would
        # otherwise enqueue a job for the literal stringified value;
        # other handlers in this module (``stage_payload``,
        # ``ingest_from_artifact``) already do the strict check.
        repo_url = args.get("repo_url", "")
        if not isinstance(repo_url, str):
            return {"error": f"repo_url must be a string, got {type(repo_url).__name__}"}
        repo_url = repo_url.strip()
        if not repo_url:
            return {"error": "repo_url is required"}
        label = args.get("label", "")
        if not isinstance(label, str):
            return {"error": f"label must be a string, got {type(label).__name__}"}
        depth = args.get("depth", "readme+code")
        if not isinstance(depth, str):
            return {"error": f"depth must be a string, got {type(depth).__name__}"}
        depth = depth.strip()
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
        path = args.get("path", "")
        if not isinstance(path, str):
            return {"error": f"path must be a string, got {type(path).__name__}"}
        # Validate emptiness on a stripped copy, but pass the raw
        # ``path`` through to ``fetch_file``. Leading/trailing
        # whitespace is legal in POSIX filenames; stripping the
        # path itself would silently mis-target a file like
        # ``" notes.txt "`` to ``"notes.txt"`` and either ingest
        # the wrong file or report ``not found`` for a path the
        # synchronous ``ingest_file`` skill would accept verbatim.
        if not path.strip():
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
        # Validate type explicitly rather than ``str()``-coercing.
        # ``str(123)`` / ``str(None)`` would otherwise silently
        # enqueue a job with a bogus body — the other handlers in
        # this module (``stage_payload``, ``ingest_from_artifact``)
        # already do isinstance checks; align with them.
        text = args.get("text", "")
        if not isinstance(text, str):
            return {"error": f"text must be a string, got {type(text).__name__}"}
        if not text.strip():
            return {"error": "text is required"}
        source_label = args.get("source_label", "")
        if not isinstance(source_label, str):
            return {
                "error": (
                    f"source_label must be a string, got "
                    f"{type(source_label).__name__}"
                ),
            }

        async def work(progress):
            # ``pipeline.ingest_idea`` performs both distill AND
            # store atomically — by the time it returns, the
            # KnowledgeEntry is already persisted. Emit
            # ``storing`` BEFORE the call so the phase represents
            # intent ("we're about to write") rather than past
            # tense ("we wrote, this event is informational").
            # Emitting ``storing`` AFTER the persist would mean a
            # cancellation between the persist and the event
            # marks ``phase=error`` even though the idea was
            # actually saved — false-failure for the caller.
            await progress("distilling", progress_pct=30)
            await progress("storing", progress_pct=80)
            idea_id = await pipeline.ingest_idea(text, source_label)
            return {"idea_id": idea_id, "source_label": source_label}

        return await self._spawn_ingest_job(
            "ingest_idea", {"source_label": source_label}, work,
        )

    async def handle_distill_paper_async(self, args):
        # isinstance-validate at the boundary like the sibling async handlers —
        # str()-coercing would enqueue a job for a bogus entry_id.
        entry_id = args.get("entry_id", "")
        if not isinstance(entry_id, str):
            return {"error": f"entry_id must be a string, got {type(entry_id).__name__}"}
        if not entry_id.strip():
            return {"error": "entry_id is required"}
        entry_id = entry_id.strip()

        async def work(progress):
            # distill() is monolithic (summarize → extract+assess → store) and
            # persists internally, so there's no mid-pipeline hook. Emit a single
            # "distilling" intent marker BEFORE the call; the job's terminal
            # "done" (run_ingest_job, on return) signals completion. Don't emit a
            # post-call "storing": distill already stored, so a cancellation
            # between its persist and that emit would mark a false phase=error.
            await progress("distilling", progress_pct=30)
            result = await pipeline.distill(entry_id)
            return {
                "entry_id": entry_id,
                "title": result.title,
                "success": result.success,
                # skipped == another drainer is already distilling this paper; a
                # distinct, non-error outcome (not a failure) — bug abfe679b.
                "skipped": getattr(result, "skipped", False),
                # errored == a transient DB-open failure in distill()'s pre-LLM
                # window; the entry was left retryable (not FAILED) — bug 706df96b.
                "errored": getattr(result, "errored", False),
                "triples": len(result.triples),
                "assessments": len(result.assessments),
            }

        return await self._spawn_ingest_job(
            "distill_paper", {"entry_id": entry_id}, work,
        )

    async def handle_ingest_status(self, args):
        # Strict isinstance — ``None`` / ``{}`` / ``123`` should
        # surface as a validation error, not get silently coerced
        # to a non-existent job_id and report ``{"error": "not
        # found"}``. That ambiguity made caller bugs
        # indistinguishable from a genuinely missing job.
        job_id = args.get("job_id", "")
        if not isinstance(job_id, str):
            return {"error": f"job_id must be a string, got {type(job_id).__name__}"}
        job_id = job_id.strip()
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
        # Drain the distillation queue continuously so ingested entries get
        # distilled without a manual start_distillation call (the worker
        # idle-polls when empty).
        self._start_distill_worker()

        # Register as a knowledge source with the librarian (fr_researcher_
        # bbe95f12). Best-effort: the librarian may not be up yet, or may
        # never be deployed in a given environment — registration failure
        # must not block the researcher agent's own startup. No retry loop
        # here; re-registration is idempotent (register() upserts by
        # source_id), so a later librarian-side rebuild/restart just needs
        # this agent's own next restart (or a future re-register skill) to
        # reconcile.
        _catalog = getattr(pipeline, "catalog", None)
        if _catalog is not None:
            try:
                from librarian_lib import CONTRACT_VERSION

                # source_id == catalog.source (== the resolved owner_agent
                # build_self_catalog derived it from, NOT necessarily
                # self.agent_id — the two SHOULD match when config's
                # bus_agent_id is set to this agent's real --id, but the
                # registry entry must key off whatever `source` value this
                # process's rows are actually stamped with). A static
                # "researcher" source_id here would let a second
                # researcher instance's registration silently overwrite the
                # first's (register_source upserts by source_id), stranding
                # the first instance's catalog rows with no reachable owner.
                #
                # system_tier (not a fixed `projects` list): researcher
                # catalogs papers under whichever project scored highest
                # per-entry, PLUS the generic "research" fallback bucket for
                # entries with no project above threshold — a static list
                # from config["projects"] would omit that fallback bucket
                # and would also need updating every time a project is
                # added, which register_source's own "projects" contract
                # isn't meant to track.
                await self.request(
                    agent_type="librarian",
                    operation="register_source",
                    args={
                        "source_id": _catalog.source,
                        "kind": "corpus",
                        "owner_agent": self.agent_id,
                        "system_tier": True,
                        "contract_version": CONTRACT_VERSION,
                        "record_count": _catalog.stats().get("total"),
                    },
                )
            except Exception:
                logger.warning(
                    "librarian register_source failed (librarian may be "
                    "unavailable) — continuing without federation registration",
                    exc_info=True,
                )

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
            await self._stop_distill_worker()
            await registry.shutdown()
            await self._http.aclose()

    async def shutdown(self):
        # Stop the distillation drain loop first so it isn't mid-distill
        # against stores that are about to tear down.
        await self._stop_distill_worker()
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
                tracked = self._ingest_tasks
                for t in survivors:
                    # Suppress the runtime warning at loop teardown.
                    # Setting ``_log_destroy_pending`` is the
                    # documented escape hatch for "I deliberately
                    # abandoned this task and accept the
                    # consequences."
                    t._log_destroy_pending = False
                    # Detach from the agent's tracked set too, so a
                    # second shutdown signal doesn't re-iterate the
                    # same already-cancelled-but-stuck tasks and
                    # eat another 10s cancel grace period waiting
                    # on them. The per-task ``done_callback`` would
                    # eventually do this when the task finishes
                    # naturally, but we don't want to wait.
                    tracked.discard(t)
                logger.warning(
                    "ingest shutdown abandoned %d task(s) still running "
                    "after 10s cancel grace period; they will be "
                    "collected at loop teardown",
                    len(survivors),
                )
        await BaseAgent.shutdown(self)

    def _start_distill_worker(self):
        """Launch the continuous distillation drain loop as a retained
        background task.

        ``DistillWorker.run()`` processes pending ``INGESTED`` entries then
        idle-polls when the queue is empty, so ingest paths need not kick it.
        Without this, the bus-agent deployment never drained the queue —
        ``worker_status`` showed ``running=False`` with items pending forever
        (bug_researcher_0cadd7ea); the only drainer was the standalone
        ``python -m researcher.worker`` process, which the agent path never
        launches.

        Embedded by default. A deployment that ALSO runs the standalone worker
        against the same DB should set ``distill_worker.embedded: false`` in
        config to avoid two drainers racing on the same ``INGESTED`` rows.
        ``pause_between`` / ``idle_poll`` under that key tune the loop.
        """
        cfg = {}
        config = getattr(pipeline, "config", None)
        if isinstance(config, dict):
            cfg = config.get("distill_worker") or {}
        if not cfg.get("embedded", True):
            logger.info("embedded distillation worker disabled by config")
            return
        if getattr(self, "_distill_worker", None) is not None:
            return  # idempotent — already running
        from researcher.worker import DistillWorker

        worker_kwargs = {
            k: cfg[k] for k in ("pause_between", "idle_poll") if k in cfg
        }
        self._distill_worker = DistillWorker(pipeline, **worker_kwargs)
        self._distill_task = asyncio.create_task(
            self._distill_worker.run(), name="distill-worker",
        )
        logger.info("embedded distillation worker started")

    async def _stop_distill_worker(self):
        """Stop the embedded distillation worker (no-op if not started).

        Clears the handles first so a re-entrant ``shutdown()`` doesn't double
        the cancel grace period, then asks ``run()`` to exit (``stop()``) and
        cancels the task to interrupt an idle-poll sleep immediately.
        """
        worker = getattr(self, "_distill_worker", None)
        task = getattr(self, "_distill_task", None)
        self._distill_worker = None
        self._distill_task = None
        if worker is None:
            return
        worker.stop()
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:  # pragma: no cover — best-effort teardown
                logger.warning(
                    "embedded distillation worker errored during shutdown",
                    exc_info=True,
                )

    import signal

    agent.register_skills = MethodType(register_skills, agent)
    agent._handlers["watch_ingest_queue"] = MethodType(handle_watch_ingest_queue, agent)
    agent._handlers["list_ingest_watchers"] = MethodType(handle_list_ingest_watchers, agent)
    agent._handlers["stop_ingest_watcher"] = MethodType(handle_stop_ingest_watcher, agent)
    agent._handlers["stage_payload"] = MethodType(handle_stage_payload, agent)
    agent._handlers["ingest_from_artifact"] = MethodType(handle_ingest_from_artifact, agent)
    agent._handlers["ask_librarian"] = MethodType(handle_ask_librarian, agent)
    agent._handlers["ingest_url_with_body"] = MethodType(handle_ingest_url_with_body, agent)
    agent._handlers["distill_repo_docs"] = MethodType(handle_distill_repo_docs, agent)
    agent._handlers["catalog_query"] = MethodType(handle_catalog_query, agent)
    agent._handlers["catalog_search"] = MethodType(handle_catalog_search, agent)
    agent._handlers["catalog_stats"] = MethodType(handle_catalog_stats, agent)
    # Registered as "list_since" (not "catalog_list_since") — the librarian's
    # own federation code calls every registered source's resync primitive
    # by this exact name (CatalogSkills.list_since).
    agent._handlers["list_since"] = MethodType(handle_list_since, agent)
    agent._handlers["catalog_mark_stale"] = MethodType(handle_catalog_mark_stale, agent)
    agent._handlers["catalog_fetch"] = MethodType(handle_catalog_fetch, agent)
    agent._handlers["catalog_backfill"] = MethodType(handle_catalog_backfill, agent)
    agent._handlers["ingest_github_async"] = MethodType(handle_ingest_github_async, agent)
    agent._handlers["ingest_file_async"] = MethodType(handle_ingest_file_async, agent)
    agent._handlers["ingest_idea_async"] = MethodType(handle_ingest_idea_async, agent)
    agent._handlers["distill_paper_async"] = MethodType(handle_distill_paper_async, agent)
    agent._handlers["ingest_status"] = MethodType(handle_ingest_status, agent)
    agent._get_job_store = MethodType(_get_job_store, agent)
    agent._get_ingest_semaphore = MethodType(_get_ingest_semaphore, agent)
    agent._spawn_ingest_job = MethodType(_spawn_ingest_job, agent)
    agent._start_distill_worker = MethodType(_start_distill_worker, agent)
    agent._stop_distill_worker = MethodType(_stop_distill_worker, agent)
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
        # Lightweight CLI handling — doesn't build the full pipeline.
        # The runtime agent is a dynamic ``from_mcp`` subclass with no
        # importable class, so give the install path a stub carrying
        # the right identity: ``from_cli`` is a classmethod that
        # constructs ``cls(...)``, and calling it on BaseAgent directly
        # registers module_name="agent" / agent_type="base" with the
        # bus — a launch spec that can't start
        # (bug_developer_agent_main_install_uses_base_class_4bb0a5cf).
        class _ResearcherInstallStub(BaseAgent):
            agent_type = "researcher"
            module_name = "researcher.agent"

        _ResearcherInstallStub.from_cli([
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
