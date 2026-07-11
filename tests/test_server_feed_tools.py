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
