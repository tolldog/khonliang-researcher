"""Background distillation worker.

Continuously processes the pending queue:
  1. Pick next ingested paper
  2. Pre-filter by relevance
  3. Run summarizer → extractor → assessor
  4. Store results, mark as distilled
  5. Pause between papers (configurable)
  6. Repeat until queue empty, then idle-poll

Runs as a standalone process or embedded in the MCP server.

Usage:
    # Standalone
    python -m researcher.worker

    # Or via CLI
    khonliang-researcher worker
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Optional

from khonliang.knowledge.store import EntryStatus, Tier
from khonliang_researcher import BaseQueueWorker

from researcher.pipeline import ResearchPipeline, create_pipeline

logger = logging.getLogger(__name__)


class DistillWorker(BaseQueueWorker):
    """Background worker that drains the distillation queue."""

    def __init__(self, pipeline: ResearchPipeline, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    def count_pending(self) -> int:
        return len(self.pipeline.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED))

    def get_next(self):
        """Get next ingested paper, skipping papers that failed too many times."""
        for entry in self.pipeline.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED):
            retries = self._failed_ids.get(entry.id, 0)
            if retries < self.max_retries_per_item:
                return entry
        return None

    async def process_item(self, entry) -> bool:
        """Pre-filter by relevance, then distill."""
        skipped = await self.pipeline.filter_irrelevant(entry.id)
        if skipped:
            logger.info("  SKIPPED (low relevance): %s", entry.title[:60])
            return True

        result = await self.pipeline.distill(entry.id)
        if result.success:
            triples = len(result.triples) if result.triples else 0
            logger.info(
                "  OK: %d triples, assessments: %s",
                triples,
                list(result.assessments.keys()),
            )
        return result.success


def main():
    """Run worker as standalone process."""
    import argparse

    parser = argparse.ArgumentParser(description="Distillation worker")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch", type=int, default=None, help="Process N papers then exit")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds between papers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    pipeline = create_pipeline(args.config)
    worker = DistillWorker(pipeline, pause_between=args.pause)

    # Graceful shutdown on SIGINT/SIGTERM
    def handle_signal(sig, frame):
        logger.info("Shutdown signal received...")
        worker.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.batch:
        asyncio.run(worker.run_batch(limit=args.batch))
    else:
        asyncio.run(worker.run())


if __name__ == "__main__":
    main()
