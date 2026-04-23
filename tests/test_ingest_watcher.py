from __future__ import annotations

import pytest

from researcher.ingest_watcher import (
    IngestWatcher,
    IngestWatcherConfig,
    IngestWatcherRegistry,
    IngestWatcherStore,
    TOPIC_QUEUE_DRAINED,
    TOPIC_URL_DISTILLED,
    TOPIC_URL_DISTILLING,
    TOPIC_URL_FAILED,
    TOPIC_URL_QUEUED,
)


@pytest.mark.asyncio
async def test_ingest_watcher_emits_status_transitions(tmp_path):
    events: list[tuple[str, dict]] = []

    async def publish(topic: str, payload: dict) -> None:
        events.append((topic, payload))

    snapshot = [
        {"entry_id": "a", "url": "https://example.com/a", "status": "ingested", "summary_preview": ""},
        {"entry_id": "b", "url": "https://example.com/b", "status": "processing", "summary_preview": ""},
        {"entry_id": "c", "url": "https://example.com/c", "status": "distilled", "summary_preview": "done"},
        {"entry_id": "d", "url": "https://example.com/d", "status": "failed", "summary_preview": "", "error_message": "boom"},
    ]

    watcher = IngestWatcher(
        config=IngestWatcherConfig("iw_test", 5, 1.0),
        store=IngestWatcherStore(str(tmp_path / "watcher.db")),
        publish=publish,
        snapshot_fn=lambda: snapshot,
        now_fn=lambda: 123.0,
    )

    emitted = await watcher.poll_once()

    assert emitted == 4
    assert [topic for topic, _ in events] == [
        TOPIC_URL_QUEUED,
        TOPIC_URL_DISTILLING,
        TOPIC_URL_DISTILLED,
        TOPIC_URL_FAILED,
    ]


@pytest.mark.asyncio
async def test_ingest_watcher_emits_queue_drained_once(tmp_path):
    events: list[tuple[str, dict]] = []

    async def publish(topic: str, payload: dict) -> None:
        events.append((topic, payload))

    snapshots = [
        [{"entry_id": "a", "url": "https://example.com/a", "status": "processing", "summary_preview": ""}],
        [{"entry_id": "a", "url": "https://example.com/a", "status": "distilled", "summary_preview": "done"}],
        [{"entry_id": "a", "url": "https://example.com/a", "status": "distilled", "summary_preview": "done"}],
    ]

    def snapshot_fn():
        return snapshots.pop(0)

    watcher = IngestWatcher(
        config=IngestWatcherConfig("iw_test", 5, 1.0),
        store=IngestWatcherStore(str(tmp_path / "watcher.db")),
        publish=publish,
        snapshot_fn=snapshot_fn,
        now_fn=lambda: 123.0,
    )

    await watcher.poll_once()
    await watcher.poll_once()
    await watcher.poll_once()

    queue_drained = [payload for topic, payload in events if topic == TOPIC_QUEUE_DRAINED]
    assert len(queue_drained) == 1


@pytest.mark.asyncio
async def test_ingest_watcher_registry_start_list_stop(tmp_path):
    async def publish(topic: str, payload: dict) -> None:
        return None

    registry = IngestWatcherRegistry(
        store=IngestWatcherStore(str(tmp_path / "watcher.db")),
        publish=publish,
        snapshot_fn=lambda: [],
    )

    watcher_id = await registry.start(interval_s=5)
    listed = registry.list_watchers()

    assert len(listed) == 1
    assert listed[0]["watcher_id"] == watcher_id

    stopped = await registry.stop(watcher_id)
    assert stopped is True
    assert registry.list_watchers() == []


@pytest.mark.asyncio
async def test_stop_returns_true_when_only_persisted_state_existed(tmp_path):
    """Registry.stop() must report True when it cleaned up persisted state even
    if no live watcher was running in this process. Returning False there would
    look like a no-op failure to callers despite the delete actually happening.
    """

    async def publish(topic: str, payload: dict) -> None:
        return None

    store = IngestWatcherStore(str(tmp_path / "watcher.db"))

    # Simulate a prior-process watcher: a persisted row with no live task.
    store.register_watcher("iw_orphan", 5, 100.0)

    registry = IngestWatcherRegistry(
        store=store,
        publish=publish,
        snapshot_fn=lambda: [],
    )

    # Sanity: nothing live, but registry row is present.
    assert any(row["watcher_id"] == "iw_orphan" for row in store.list_watchers())

    stopped = await registry.stop("iw_orphan")
    assert stopped is True, "stop() must return True when persisted state was cleaned"
    assert not any(row["watcher_id"] == "iw_orphan" for row in store.list_watchers())

    # Second call: nothing live AND nothing persisted -> genuinely nothing to stop.
    stopped_again = await registry.stop("iw_orphan")
    assert stopped_again is False


@pytest.mark.asyncio
async def test_ingest_watcher_retries_publish_after_failure(tmp_path):
    events: list[tuple[str, dict]] = []
    attempts = {"count": 0}

    async def publish(topic: str, payload: dict) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("boom")
        events.append((topic, payload))

    snapshot = [{"entry_id": "a", "url": "https://example.com/a", "status": "distilled", "summary_preview": "done"}]

    watcher = IngestWatcher(
        config=IngestWatcherConfig("iw_test", 5, 1.0),
        store=IngestWatcherStore(str(tmp_path / "watcher.db")),
        publish=publish,
        snapshot_fn=lambda: snapshot,
        now_fn=lambda: 123.0,
    )

    emitted_first = await watcher.poll_once()
    emitted_second = await watcher.poll_once()

    assert emitted_first == 0
    assert emitted_second == 1
    assert [topic for topic, _ in events] == [TOPIC_URL_DISTILLED]
