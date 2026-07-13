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
from khonliang_researcher.worker import SKIP

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
        """Get next ingested paper, skipping papers that failed too many times.

        Reclaims papers orphaned by a dead distiller first (bug abfe679b). The
        run loop calls get_next every drain / idle-poll cycle, so this recovers
        orphans that appear AFTER startup too — a sibling drainer killed mid-
        distill — with no restart. Safe every call: recovery only touches papers
        whose lock owner is gone, never a live in-flight distill.
        """
        self.pipeline.recover_stalled_processing()
        for entry in self.pipeline.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED):
            retries = self._failed_ids.get(entry.id, 0)
            if retries >= self.max_retries_per_item:
                continue
            # Skip papers a sibling drainer is actively distilling (live lock) —
            # pick a different one instead of racing to a no-op claim. Avoids
            # wasting a batch slot on a contention skip (abfe679b).
            if self.pipeline.locks.is_locked_live(entry.id):
                continue
            return entry
        return None

    async def process_item(self, entry) -> bool:
        """Pre-filter by relevance, then distill."""
        skipped = await self.pipeline.filter_irrelevant(entry.id)
        if skipped:
            logger.info("  SKIPPED (low relevance): %s", entry.title[:60])
            return True

        result = await self.pipeline.distill(entry.id)
        if getattr(result, "skipped", False):
            # A live owner claimed the paper in the tiny window between get_next
            # (which already filters live-locked papers) and distill's claim.
            # It's neither a success nor a failure — return the SKIP sentinel so
            # BaseQueueWorker counts it as `declined` (not processed, not a retry
            # failure): the paper IS being distilled by the sibling and will
            # finish or be recovered on that owner's crash (abfe679b).
            logger.info("  SKIPPED (locked by another drainer): %s", entry.title[:60])
            return SKIP
        if getattr(result, "errored", False):
            # A transient DB-open error in distill()'s pre-LLM window (bug
            # 706df96b) — the entry's status was left untouched (not FAILED),
            # so treat this the same as a live-lock skip: don't burn a retry,
            # just come back around next drain cycle once the DB recovers.
            logger.info("  SKIPPED (transient DB error): %s", entry.title[:60])
            return SKIP
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
