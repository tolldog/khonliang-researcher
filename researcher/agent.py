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

    Thin wrapper over
    ``agent.request(agent_type='store', operation='artifact_create')``:
    validates the inbound shape, generates a sensible title when
    one wasn't provided, builds the metadata dict, and surfaces
    the new ``{artifact_id}`` (or the store's error envelope) to
    the caller. Routing through the ``store`` agent type means
    the store backend can move (composite / local-only / etc.)
    without researcher caring.

    Module-level so tests can call it with a mock ``agent``
    without wiring through ``BaseAgent.from_mcp``.
    """
    content = args.get("content")
    if not isinstance(content, str):
        return {"error": "content must be a string"}
    if not content:
        return {"error": "content is required"}
    kind_hint = str(args.get("kind_hint") or "").strip()
    title = str(args.get("title") or "").strip()
    if not title:
        # Short content preview so the artifact has a human
        # name in ``artifact_list``. First non-empty stripped
        # line of the input, capped at 80 chars with an
        # ellipsis. Single-pass to avoid materializing a full
        # ``splitlines()`` list for large payloads.
        first_line, _, _ = content.partition("\n")
        preview = first_line.strip()
        if len(preview) > 80:
            title = preview[:80] + "…"
        else:
            title = preview or "staged payload"
    content_type = str(args.get("content_type") or "text/plain")
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
    artifact_id = payload.get("id")
    if not artifact_id:
        return {"error": "store created artifact without id"}
    return {"artifact_id": artifact_id}


async def ingest_from_artifact(
    agent: BaseAgent, pipeline, args: dict,
) -> dict:
    """Pull bytes from store, route through ``pipeline.ingest_idea``.

    Auto-detected dispatch is the sister FR; today this skill
    treats the artifact as informal text and feeds it to the
    existing idea pipeline. ``hints`` is accepted but not yet
    consumed — wired through so future dispatcher logic can
    break ties without an API change here.

    Module-level for the same reason as :func:`stage_payload`.
    """
    artifact_id = str(args.get("artifact_id") or "").strip()
    if not artifact_id:
        return {"error": "artifact_id is required"}
    hints = args["hints"] if "hints" in args else {}
    if not isinstance(hints, dict):
        return {"error": "hints must be an object"}
    source_label_override = str(args.get("source_label") or "").strip()

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
    text = payload.get("text") or payload.get("body") or ""
    if not isinstance(text, str) or not text:
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
                since="0.5.0",
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
                since="0.5.0",
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
        await BaseAgent.shutdown(self)

    import signal

    agent.register_skills = MethodType(register_skills, agent)
    agent._handlers["watch_ingest_queue"] = MethodType(handle_watch_ingest_queue, agent)
    agent._handlers["list_ingest_watchers"] = MethodType(handle_list_ingest_watchers, agent)
    agent._handlers["stop_ingest_watcher"] = MethodType(handle_stop_ingest_watcher, agent)
    agent._handlers["stage_payload"] = MethodType(handle_stage_payload, agent)
    agent._handlers["ingest_from_artifact"] = MethodType(handle_ingest_from_artifact, agent)
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
