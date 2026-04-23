"""Long-running ingest watcher.

Publishes ``ingest.*`` bus events from researcher queue state so
subscribers can react without polling ``worker_status``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

PublishFn = Callable[[str, dict[str, Any]], Awaitable[None]]
SnapshotFn = Callable[[], list[dict[str, Any]]]

TOPIC_URL_QUEUED = "ingest.url_queued"
TOPIC_URL_DISTILLING = "ingest.url_distilling"
TOPIC_URL_DISTILLED = "ingest.url_distilled"
TOPIC_URL_FAILED = "ingest.url_failed"
TOPIC_QUEUE_DRAINED = "ingest.queue_drained"
TOPIC_POLL_ERROR = "ingest.poll_error"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_watcher_registry (
    watcher_id TEXT PRIMARY KEY,
    interval_s INTEGER NOT NULL,
    started_at REAL NOT NULL,
    last_poll_at REAL NOT NULL DEFAULT 0,
    last_active_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ingest_watcher_dedupe (
    watcher_id TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    transition_kind TEXT NOT NULL,
    dedupe_id TEXT NOT NULL,
    emitted_at REAL NOT NULL,
    PRIMARY KEY (watcher_id, entry_id, transition_kind, dedupe_id)
);
"""


class IngestWatcherStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def register_watcher(self, watcher_id: str, interval_s: int, started_at: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingest_watcher_registry
                    (watcher_id, interval_s, started_at, last_poll_at, last_active_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (watcher_id, int(interval_s), float(started_at), 0.0, 0),
            )

    def touch(self, watcher_id: str, at: float, *, active_count: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE ingest_watcher_registry
                SET last_poll_at = ?, last_active_count = ?
                WHERE watcher_id = ?
                """,
                (float(at), int(active_count), watcher_id),
            )

    def get_last_active_count(self, watcher_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_active_count FROM ingest_watcher_registry WHERE watcher_id = ?",
                (watcher_id,),
            ).fetchone()
        return int(row["last_active_count"]) if row else 0

    def list_watchers(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ingest_watcher_registry ORDER BY started_at ASC"
            ).fetchall()
        return [
            {
                "watcher_id": row["watcher_id"],
                "interval_s": int(row["interval_s"]),
                "started_at": float(row["started_at"]),
                "last_poll_at": float(row["last_poll_at"]),
                "last_active_count": int(row["last_active_count"]),
            }
            for row in rows
        ]

    def remove_watcher(self, watcher_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM ingest_watcher_registry WHERE watcher_id = ?",
                (watcher_id,),
            )
            conn.execute(
                "DELETE FROM ingest_watcher_dedupe WHERE watcher_id = ?",
                (watcher_id,),
            )

    def was_emitted(
        self,
        watcher_id: str,
        entry_id: str,
        transition_kind: str,
        dedupe_id: str,
    ) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM ingest_watcher_dedupe
                WHERE watcher_id = ? AND entry_id = ? AND transition_kind = ? AND dedupe_id = ?
                LIMIT 1
                """,
                (watcher_id, entry_id, transition_kind, dedupe_id),
            ).fetchone()
        return row is not None

    def mark_emitted(
        self,
        watcher_id: str,
        entry_id: str,
        transition_kind: str,
        dedupe_id: str,
        at: float,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO ingest_watcher_dedupe
                    (watcher_id, entry_id, transition_kind, dedupe_id, emitted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (watcher_id, entry_id, transition_kind, dedupe_id, float(at)),
            )


@dataclass
class IngestWatcherConfig:
    watcher_id: str
    interval_s: int
    started_at: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.watcher_id,
            "interval_s": self.interval_s,
            "started_at": self.started_at,
        }


class IngestWatcher:
    def __init__(
        self,
        config: IngestWatcherConfig,
        store: IngestWatcherStore,
        publish: PublishFn,
        snapshot_fn: SnapshotFn,
        now_fn: Callable[[], float] = time.time,
    ):
        self.config = config
        self.store = store
        self._publish = publish
        self._snapshot_fn = snapshot_fn
        self._now = now_fn
        self._active_count = 0

    @property
    def active_count(self) -> int:
        return self._active_count

    async def poll_once(self) -> int:
        try:
            snapshot = self._snapshot_fn()
        except Exception as e:
            await self._emit_poll_error(str(e))
            return 0

        emitted = 0
        active_count = 0
        for row in snapshot:
            transition = _transition_for_status(str(row.get("status", "")))
            if not transition:
                continue
            if transition in {"url_queued", "url_distilling"}:
                active_count += 1
            if await self._emit_transition(row, transition):
                emitted += 1

        previous_active = max(
            self._active_count,
            self.store.get_last_active_count(self.config.watcher_id),
        )
        if previous_active > 0 and active_count == 0:
            if await self._emit(
                TOPIC_QUEUE_DRAINED,
                entry_id="_queue_",
                transition_kind="queue_drained",
                dedupe_id=str(int(self._now())),
                payload={
                    "drained_at": self._now(),
                    "total_items_processed": len(snapshot),
                },
            ):
                emitted += 1

        self._active_count = active_count
        self.store.touch(self.config.watcher_id, self._now(), active_count=active_count)
        return emitted

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Ingest watcher %s poll failed: %s", self.config.watcher_id, e)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.config.interval_s)
            except asyncio.TimeoutError:
                pass

    async def _emit_transition(self, row: dict[str, Any], transition: str) -> bool:
        entry_id = str(row.get("entry_id", ""))
        url = row.get("url", "")
        if transition == "url_queued":
            return await self._emit(
                TOPIC_URL_QUEUED,
                entry_id=entry_id,
                transition_kind=transition,
                dedupe_id="queued",
                payload={"entry_id": entry_id, "url": url, "queued_at": self._now()},
            )
        if transition == "url_distilling":
            return await self._emit(
                TOPIC_URL_DISTILLING,
                entry_id=entry_id,
                transition_kind=transition,
                dedupe_id="distilling",
                payload={"entry_id": entry_id, "url": url, "started_at": self._now()},
            )
        if transition == "url_distilled":
            return await self._emit(
                TOPIC_URL_DISTILLED,
                entry_id=entry_id,
                transition_kind=transition,
                dedupe_id="distilled",
                payload={
                    "entry_id": entry_id,
                    "url": url,
                    "distilled_at": self._now(),
                    "summary_preview": row.get("summary_preview", ""),
                },
            )
        if transition == "url_failed":
            return await self._emit(
                TOPIC_URL_FAILED,
                entry_id=entry_id,
                transition_kind=transition,
                dedupe_id="failed",
                payload={
                    "entry_id": entry_id,
                    "url": url,
                    "stage": "distilling",
                    "error_kind": "pipeline_failure",
                    "error_message": row.get("error_message", "") or "distillation failed",
                },
            )
        return False

    async def _emit(
        self,
        topic: str,
        *,
        entry_id: str,
        transition_kind: str,
        dedupe_id: str,
        payload: dict[str, Any],
    ) -> bool:
        if self.store.was_emitted(
            watcher_id=self.config.watcher_id,
            entry_id=entry_id,
            transition_kind=transition_kind,
            dedupe_id=dedupe_id,
        ):
            return False
        try:
            await self._publish(topic, payload)
        except Exception as e:
            logger.warning(
                "Ingest watcher %s publish %s failed: %s",
                self.config.watcher_id,
                topic,
                e,
            )
            return False
        self.store.mark_emitted(
            watcher_id=self.config.watcher_id,
            entry_id=entry_id,
            transition_kind=transition_kind,
            dedupe_id=dedupe_id,
            at=self._now(),
        )
        return True

    async def _emit_poll_error(self, reason: str) -> None:
        try:
            await self._publish(
                TOPIC_POLL_ERROR,
                {
                    "watcher_id": self.config.watcher_id,
                    "reason": reason,
                    "at": self._now(),
                },
            )
        except Exception as e:
            logger.warning("Ingest watcher %s poll_error publish failed: %s", self.config.watcher_id, e)


@dataclass
class _LiveWatcher:
    watcher: IngestWatcher
    stop_event: asyncio.Event
    task: asyncio.Task


class IngestWatcherRegistry:
    def __init__(
        self,
        store: IngestWatcherStore,
        publish: PublishFn,
        snapshot_fn: SnapshotFn,
    ):
        self.store = store
        self._publish = publish
        self._snapshot_fn = snapshot_fn
        self._watchers: dict[str, _LiveWatcher] = {}
        self._lock = asyncio.Lock()

    async def start(self, *, interval_s: int) -> str:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        config = IngestWatcherConfig(
            watcher_id=_new_watcher_id(),
            interval_s=interval_s,
            started_at=time.time(),
        )
        await self._spawn(config, persist=True)
        return config.watcher_id

    async def rehydrate(self) -> list[str]:
        spawned: list[str] = []
        for row in self.store.list_watchers():
            watcher_id = row["watcher_id"]
            async with self._lock:
                if watcher_id in self._watchers:
                    continue
            config = IngestWatcherConfig(
                watcher_id=watcher_id,
                interval_s=int(row["interval_s"]),
                started_at=float(row["started_at"]),
            )
            await self._spawn(config, persist=False)
            spawned.append(watcher_id)
        return spawned

    async def _spawn(self, config: IngestWatcherConfig, *, persist: bool) -> None:
        async with self._lock:
            if config.watcher_id in self._watchers:
                return
        watcher = IngestWatcher(
            config=config,
            store=self.store,
            publish=self._publish,
            snapshot_fn=self._snapshot_fn,
        )
        if persist:
            self.store.register_watcher(config.watcher_id, config.interval_s, config.started_at)
        stop_event = asyncio.Event()
        task = asyncio.create_task(watcher.run(stop_event), name=f"ingest_watcher_{config.watcher_id}")
        async with self._lock:
            if config.watcher_id in self._watchers:
                stop_event.set()
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                return
            self._watchers[config.watcher_id] = _LiveWatcher(watcher, stop_event, task)

    async def stop(self, watcher_id: str) -> bool:
        async with self._lock:
            live = self._watchers.pop(watcher_id, None)
        if live is None:
            self.store.remove_watcher(watcher_id)
            return False
        live.stop_event.set()
        try:
            await asyncio.wait_for(live.task, timeout=5.0)
        except asyncio.TimeoutError:
            live.task.cancel()
            try:
                await live.task
            except (asyncio.CancelledError, Exception):
                pass
        except Exception:
            pass
        self.store.remove_watcher(watcher_id)
        return True

    def list_watchers(self) -> list[dict[str, Any]]:
        rows = self.store.list_watchers()
        for row in rows:
            live = self._watchers.get(row["watcher_id"])
            row["active_count"] = live.watcher.active_count if live else 0
        return rows

    async def shutdown(self) -> None:
        ids = list(self._watchers.keys())
        for watcher_id in ids:
            try:
                await self.stop(watcher_id)
            except Exception as e:
                logger.warning("ingest watcher shutdown: stop %s failed: %s", watcher_id, e)


def _transition_for_status(status: str) -> Optional[str]:
    mapping = {
        "ingested": "url_queued",
        "processing": "url_distilling",
        "distilled": "url_distilled",
        "failed": "url_failed",
    }
    return mapping.get(status.lower())


def _new_watcher_id() -> str:
    return f"iw_{uuid.uuid4().hex[:12]}"
