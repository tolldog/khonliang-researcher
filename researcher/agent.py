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
import sys

from khonliang_bus import BaseAgent

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

    return agent


def main():
    """CLI entry point for the researcher agent."""
    import argparse

    # Check for install/uninstall commands before full agent init
    parser = argparse.ArgumentParser(description="khonliang-researcher bus agent")
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
