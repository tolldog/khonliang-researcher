"""Tests for researcher/feed_store.py — persistent RSS feed registry.

Covers fr_researcher_b8b5c008 acceptance criteria: seed migration,
register_feed idempotency (duplicate URL refused), and disable_feed
soft-delete preserving history.
"""

from __future__ import annotations

import pytest

from researcher.feed_store import FeedError, FeedStore
from researcher.rss import DEFAULT_FEEDS


def test_seed_if_empty_populates_default_feeds(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    seeded = store.seed_if_empty(DEFAULT_FEEDS)
    assert seeded == len(DEFAULT_FEEDS) == 11
    feeds = store.list_feeds(enabled_only=False)
    assert len(feeds) == 11
    assert {f["source"] for f in feeds} == {cfg.source for cfg in DEFAULT_FEEDS.values()}


def test_seed_if_empty_is_idempotent(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    first = store.seed_if_empty(DEFAULT_FEEDS)
    second = store.seed_if_empty(DEFAULT_FEEDS)
    assert first == 11
    assert second == 0
    assert len(store.list_feeds(enabled_only=False)) == 11


def test_register_feed_rejects_duplicate_url(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    store.register_feed(name="A Blog", url="https://a.example/rss.xml", source="a")
    with pytest.raises(FeedError, match="already exists"):
        store.register_feed(name="A Blog Again", url="https://a.example/rss.xml", source="a2")
    assert len(store.list_feeds(enabled_only=False)) == 1


def test_register_feed_then_get_and_list(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    feed = store.register_feed(
        name="A Blog", url="https://a.example/rss.xml", source="a", metadata={"category": "infra"},
    )
    assert feed["name"] == "A Blog"
    assert feed["enabled"] is True
    assert feed["metadata"] == {"category": "infra"}

    fetched = store.get_feed(feed["feed_id"])
    assert fetched == feed

    feeds = store.list_feeds()
    assert len(feeds) == 1
    assert feeds[0]["feed_id"] == feed["feed_id"]


def test_disable_feed_soft_deletes_and_preserves_history(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    feed = store.register_feed(name="A Blog", url="https://a.example/rss.xml", source="a")

    assert store.disable_feed(feed["feed_id"]) is True

    assert store.list_feeds(enabled_only=True) == []
    all_feeds = store.list_feeds(enabled_only=False)
    assert len(all_feeds) == 1
    assert all_feeds[0]["feed_id"] == feed["feed_id"]
    assert all_feeds[0]["enabled"] is False

    # Row still resolvable directly by id — this is a soft-delete, not a purge.
    still_there = store.get_feed(feed["feed_id"])
    assert still_there is not None
    assert still_there["enabled"] is False


def test_disable_feed_unknown_id_returns_false(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    assert store.disable_feed("feed_doesnotexist") is False


def test_update_feed_edits_fields_in_place(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    feed = store.register_feed(name="A Blog", url="https://a.example/rss.xml", source="a")

    updated = store.update_feed(feed["feed_id"], name="A Blog (renamed)", metadata={"k": "v"})
    assert updated["name"] == "A Blog (renamed)"
    assert updated["metadata"] == {"k": "v"}
    assert updated["url"] == "https://a.example/rss.xml"  # unspecified fields unchanged


def test_update_feed_rejects_unsupported_field(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    feed = store.register_feed(name="A Blog", url="https://a.example/rss.xml", source="a")
    with pytest.raises(FeedError, match="unsupported fields"):
        store.update_feed(feed["feed_id"], created_at=0.0)


def test_update_feed_unknown_id_returns_none(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    assert store.update_feed("feed_doesnotexist", name="x") is None


def test_register_feed_requires_all_fields(tmp_path):
    store = FeedStore(str(tmp_path / "feeds.db"))
    with pytest.raises(FeedError):
        store.register_feed(name="", url="https://a.example/rss.xml", source="a")
