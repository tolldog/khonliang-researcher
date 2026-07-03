"""Tests for researcher/rss.py — feed parsing + keyword search edge cases.

Covers the audit fixes: empty-query ZeroDivisionError (bug_researcher_cf797a64)
and OPML key-collision + RSS 1.0/RDF parsing (bug_researcher_875ac59a).
"""

from __future__ import annotations

import pytest

from researcher.rss import DEFAULT_FEEDS, RSSEngine, _parse_feed


# ---------------------------------------------------------------------------
# Keyword search — empty query must not divide by zero
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_empty_query_returns_empty_not_zerodivision():
    engine = RSSEngine()
    # A blank query must short-circuit before `score = ... / len(keywords)`
    # (len 0 -> ZeroDivisionError, which BaseEngine.query would swallow into a
    # silent empty result). `_cache_time = inf` keeps the cache non-stale so
    # execute() does no network refresh.
    engine._cache_time = float("inf")
    for blank in ("", "   ", "\n\t"):
        assert await engine.execute(blank) == []


# ---------------------------------------------------------------------------
# RSS 1.0 (RDF) parsing — must not silently yield zero entries
# ---------------------------------------------------------------------------

RSS1_RDF = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://purl.org/rss/1.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel rdf:about="http://example.org/"><title>Example</title></channel>
  <item rdf:about="http://example.org/a">
    <title>First Post</title>
    <link>http://example.org/a</link>
    <description>Body &lt;b&gt;one&lt;/b&gt;</description>
    <dc:date>2026-05-30</dc:date>
  </item>
  <item rdf:about="http://example.org/b">
    <title>Second Post</title>
    <link>http://example.org/b</link>
    <description>Body two</description>
  </item>
</rdf:RDF>
"""


def test_parse_feed_handles_rss1_rdf():
    results = _parse_feed(RSS1_RDF, source="ex")
    assert [r.title for r in results] == ["First Post", "Second Post"]
    assert results[0].url == "http://example.org/a"
    assert results[0].content == "Body one"  # HTML stripped
    assert results[0].metadata["published"] == "2026-05-30"


RSS2 = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>R2 Post</title><link>http://x/r2</link>
    <description>desc</description><pubDate>Fri, 30 May 2026</pubDate></item>
</channel></rss>
"""


def test_parse_feed_still_handles_rss2():
    results = _parse_feed(RSS2, source="ex")
    assert [r.title for r in results] == ["R2 Post"]
    assert results[0].url == "http://x/r2"


# ---------------------------------------------------------------------------
# OPML key collision — colliding sanitized names must not overwrite
# ---------------------------------------------------------------------------


def test_load_opml_disambiguates_colliding_keys(tmp_path):
    from researcher.rss import load_opml

    # Two feeds whose sanitized+truncated keys collide on the first 30 chars.
    opml = tmp_path / "feeds.opml"
    opml.write_text(
        """<?xml version="1.0"?>
<opml version="1.0"><body>
  <outline text="AI Research Weekly Digest Number One" xmlUrl="http://a/feed"/>
  <outline text="AI Research Weekly Digest Number Two" xmlUrl="http://b/feed"/>
</body></opml>
"""
    )
    feeds = load_opml(str(opml))
    # Both feeds survive (no silent overwrite), pointing at distinct URLs.
    urls = sorted(fc.url for fc in feeds.values())
    assert urls == ["http://a/feed", "http://b/feed"]
    assert len(feeds) == 2


# ---------------------------------------------------------------------------
# Persistent feed registry wiring (fr_researcher_b8b5c008) — must never touch
# a real path by default; only opts in when a caller passes db_path.
# ---------------------------------------------------------------------------


def test_rssengine_defaults_to_default_feeds_without_db_path():
    # No feeds/opml_path/db_path given: must use DEFAULT_FEEDS directly and
    # never construct a FeedStore (which would create/seed a file at
    # whatever the default path happened to be — see bug filed after this
    # FR accidentally seeded the live production data/researcher.db during
    # a test run).
    engine = RSSEngine()
    assert engine.feeds == DEFAULT_FEEDS


def test_rssengine_loads_and_seeds_from_explicit_db_path(tmp_path):
    db_path = str(tmp_path / "feeds.db")
    engine = RSSEngine(db_path=db_path)
    assert len(engine.feeds) == len(DEFAULT_FEEDS)
    assert set(engine.feeds) == set(DEFAULT_FEEDS)

    # A feed registered after the seed is picked up by a fresh engine
    # reading the same store (runtime add, no restart needed).
    from researcher.feed_store import FeedStore

    FeedStore(db_path).register_feed(name="New Blog", url="https://new.example/rss.xml", source="new")
    engine2 = RSSEngine(db_path=db_path)
    assert len(engine2.feeds) == len(DEFAULT_FEEDS) + 1
    assert "new" in {cfg.source for cfg in engine2.feeds.values()}


def test_rssengine_falls_back_to_default_feeds_on_unreadable_db_path(tmp_path):
    # db_path pointing at a directory (not a file) can't be opened by
    # sqlite3.connect; the loader must degrade to DEFAULT_FEEDS rather
    # than raise or return zero feeds.
    bad_path = tmp_path / "not_a_file"
    bad_path.mkdir()
    engine = RSSEngine(db_path=str(bad_path))
    assert engine.feeds == DEFAULT_FEEDS
