"""Tests for researcher/server.py's feed-registry MCP tools end-to-end.

Covers a codex finding on PR #71 (after rebase): _feed_store() created the
FeedStore's schema but never seeded DEFAULT_FEEDS, so a brand-new db_path
left list_feeds/get_feed/update_feed/disable_feed operating on an empty
registry until some unrelated call path (browse_feeds) happened to trigger
seeding first via rss._load_feeds_from_store.
"""

from __future__ import annotations

import pytest

from researcher.rss import DEFAULT_FEEDS
from researcher.server import create_research_server


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    # Away from the repo root's real feeds.opml (26 entries) so these tests
    # exercise the DEFAULT_FEEDS fallback path specifically and deterministically.
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"db_path: {tmp_path / 'test.db'}\n")

    from researcher.pipeline import create_pipeline

    pipeline = create_pipeline(str(config_path))
    return create_research_server(pipeline)


async def _call(mcp, name, args=None):
    result = await mcp.call_tool(name, args or {})
    return result[-1]["result"]


@pytest.mark.asyncio
async def test_list_feeds_seeded_on_first_use_via_server_tool(mcp):
    result = await _call(mcp, "list_feeds")
    assert f"{len(DEFAULT_FEEDS)} feed(s)" in result
    for cfg in DEFAULT_FEEDS.values():
        assert cfg.name in result


@pytest.mark.asyncio
async def test_get_feed_works_on_a_brand_new_db_path(mcp):
    listing = await _call(mcp, "list_feeds")
    first_feed_id = listing.split("\n")[1].split(" | ")[0]

    result = await _call(mcp, "get_feed", {"feed_id": first_feed_id})
    assert first_feed_id in result
    assert "error" not in result


@pytest.mark.asyncio
async def test_list_feeds_and_load_feeds_from_store_use_the_same_seed_source(tmp_path, monkeypatch):
    # codex finding on PR #71, round 4 of the final pre-merge check:
    # server.py's _feed_store() used to always seed from DEFAULT_FEEDS
    # while rss.py's _load_feeds_from_store seeded from _seed_source()
    # (feeds.opml when present) -- so the registry's initial contents
    # depended on which tool ran first. Both must seed identically.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "feeds.opml").write_text(
        '<?xml version="1.0"?><opml version="1.0"><body>'
        '<outline text="Only In OPML" xmlUrl="http://opml-only.example/feed"/>'
        "</body></opml>"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"db_path: {tmp_path / 'test.db'}\n")

    from researcher.pipeline import create_pipeline

    pipeline = create_pipeline(str(config_path))
    mcp = create_research_server(pipeline)

    # list_feeds (server-side tool) runs first here, before any browse_feeds
    # call ever touches _load_feeds_from_store.
    result = await _call(mcp, "list_feeds")
    assert "Only In OPML" in result
    assert "1 feed(s)" in result
