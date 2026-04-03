"""Background distillation worker.

Continuously processes the pending queue:
  1. Pick next undistilled paper
  2. Run summarizer → extractor → assessor
  3. Store results, move to distilled
  4. Pause between papers (configurable)
  5. Repeat until queue empty, then idle-poll

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

from khonliang.knowledge.store import Tier

from researcher.pipeline import ResearchPipeline, create_pipeline

logger = logging.getLogger(__name__)


class DistillWorker:
    """Background worker that drains the distillation queue."""

    def __init__(
        self,
        pipeline: ResearchPipeline,
        pause_between: float = 2.0,
        idle_poll: float = 30.0,
        max_failures: int = 3,
    ):
        self.pipeline = pipeline
        self.pause_between = pause_between  # seconds between papers
        self.idle_poll = idle_poll  # seconds between empty-queue checks
        self.max_failures = max_failures  # consecutive failures before skipping
        self._running = False
        self._stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "started_at": None,
        }

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "running": self._running,
            "pending": self._count_pending(),
        }

    def _count_pending(self) -> int:
        return sum(
            1
            for e in self.pipeline.knowledge.get_by_tier(Tier.IMPORTED)
            if "undistilled" in (e.tags or [])
        )

    def _get_next(self):
        """Get next undistilled paper, or None."""
        for entry in self.pipeline.knowledge.get_by_tier(Tier.IMPORTED):
            if "undistilled" in (entry.tags or []):
                return entry
        return None

    async def run(self):
        """Main worker loop. Runs until stopped or queue exhausted."""
        self._running = True
        self._stats["started_at"] = time.time()
        consecutive_failures = 0

        logger.info(
            "Distill worker started. %d papers pending.", self._count_pending()
        )

        while self._running:
            entry = self._get_next()

            if entry is None:
                logger.info("Queue empty. Idling for %.0fs...", self.idle_poll)
                await asyncio.sleep(self.idle_poll)
                continue

            logger.info(
                "[%d/%d] Distilling: %s",
                self._stats["processed"] + 1,
                self._stats["processed"] + self._count_pending(),
                entry.title[:60],
            )

            try:
                result = await self.pipeline.distill(entry.id)

                if result.success:
                    self._stats["processed"] += 1
                    consecutive_failures = 0
                    triples = len(result.triples) if result.triples else 0
                    logger.info(
                        "  OK: %d triples, assessments: %s",
                        triples,
                        list(result.assessments.keys()),
                    )
                else:
                    self._stats["failed"] += 1
                    consecutive_failures += 1
                    logger.warning("  FAILED: %s", entry.title[:60])

            except Exception as e:
                self._stats["failed"] += 1
                consecutive_failures += 1
                logger.error("  ERROR: %s — %s", entry.title[:60], e)

            # Skip paper if too many consecutive failures (model may be down)
            if consecutive_failures >= self.max_failures:
                logger.warning(
                    "Too many consecutive failures (%d). Pausing for 60s...",
                    consecutive_failures,
                )
                await asyncio.sleep(60)
                consecutive_failures = 0

            # Pause between papers to avoid overwhelming Ollama
            if self._running:
                await asyncio.sleep(self.pause_between)

        logger.info(
            "Worker stopped. Processed: %d, Failed: %d, Skipped: %d",
            self._stats["processed"],
            self._stats["failed"],
            self._stats["skipped"],
        )

    def stop(self):
        """Signal the worker to stop after current paper."""
        self._running = False

    async def run_batch(self, limit: Optional[int] = None):
        """Process up to `limit` papers then stop. None = all pending."""
        self._running = True
        self._stats["started_at"] = time.time()
        count = 0

        pending = self._count_pending()
        target = min(pending, limit) if limit else pending
        logger.info("Processing %d papers...", target)

        while self._running and (limit is None or count < limit):
            entry = self._get_next()
            if entry is None:
                break

            count += 1
            logger.info("[%d/%d] %s", count, target, entry.title[:60])

            try:
                result = await self.pipeline.distill(entry.id)
                if result.success:
                    self._stats["processed"] += 1
                else:
                    self._stats["failed"] += 1
            except Exception as e:
                self._stats["failed"] += 1
                logger.error("  ERROR: %s", e)

            if self._running and count < target:
                await asyncio.sleep(self.pause_between)

        self._running = False
        return self._stats


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
