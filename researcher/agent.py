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
