"""Generic researcher — BaseResearchAgent with zero extensions.

Proves the researcher-lib framework works standalone. Has ~18 generic
research skills (search, distill, synthesize, explore, ideas, bundling)
with no developer-specific tools (no FRs, no code scanning, no pipeline
management).

This is what a new domain would start from — extend BaseResearchAgent,
add domain engines and rules, deploy.

Usage::

    python -m researcher.generic_agent --id generic-researcher --bus http://localhost:8788 --config config.yaml
"""

from __future__ import annotations

import asyncio
import logging
import sys

from khonliang_researcher import BaseResearchAgent

logger = logging.getLogger(__name__)


class GenericResearcher(BaseResearchAgent):
    """Generic researcher — just the base, no extensions."""

    agent_type = "researcher"
    module_name = "researcher.generic_agent"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generic researcher bus agent")
    parser.add_argument("command", nargs="?", choices=["install", "uninstall"])
    parser.add_argument("--id", default="generic-researcher")
    parser.add_argument("--bus", default="http://localhost:8788")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command in ("install", "uninstall"):
        from khonliang_bus import BaseAgent
        BaseAgent.from_cli([
            args.command,
            "--id", args.id,
            "--bus", args.bus,
            "--config", args.config,
        ])
        return

    agent = GenericResearcher(
        agent_id=args.id,
        bus_url=args.bus,
        config_path=args.config,
    )
    asyncio.run(agent.start())


if __name__ == "__main__":
    main()
