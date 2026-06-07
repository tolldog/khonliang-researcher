"""Regression tests for second-pass pipeline correctness fixes.

Covers:
- bug_researcher_b5597e90 — research_idea per-query budget floored at 1
- bug_researcher_73b28040 — strike() removes all _url_index aliases
- bug_researcher_571dad81 — get_historical_feature_requests tolerates non-dict FR JSON
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from researcher.pipeline import ResearchPipeline


@pytest.mark.asyncio
async def test_research_idea_per_query_budget_floored_at_one(monkeypatch):
    """When an idea has more search queries than max_papers, integer division
    (max_papers // len(queries)) is 0 — which would search every query for zero
    papers. The per-query budget must be floored at 1."""
    import researcher.search_engines as se

    captured: list[int] = []

    async def fake_search_papers(query, max_results=10, **kwargs):
        captured.append(max_results)
        return []  # empty → fetch/distill loops are no-ops

    monkeypatch.setattr(se, "search_papers", fake_search_papers)

    queries = [f"q{i}" for i in range(11)]  # 11 queries > 10 papers
    idea = SimpleNamespace(
        title="idea", tags=[], metadata={"search_queries": queries},
    )

    class _Knowledge:
        def get(self, _id):
            return idea

        def get_by_status(self, *a, **k):
            return []

        def set_status(self, *a, **k):
            pass

        def add(self, *a, **k):
            pass

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.knowledge = _Knowledge()
    pipe.digest = SimpleNamespace(record=lambda **k: None)
    pipe._url_index = {}

    stats = await pipe.research_idea("idea_x", max_papers=10)

    assert "error" not in stats
    # naive 10 // 11 == 0; must be floored to 1 for every query.
    assert captured == [1] * 11


def test_strike_removes_all_url_index_aliases():
    """ingest_paper indexes an entry under several URL aliases (canonical, raw,
    original, arxiv-abs). strike() must remove ALL of them, not just the
    canonical one, or a re-import via an alias returns the stale deleted id."""
    entry = SimpleNamespace(title="Paper", metadata={"url": "https://canon"})

    class _Knowledge:
        def get(self, _id):
            # summary lookup ("<id>_summary") returns None; entry lookup returns entry
            return None if str(_id).endswith("_summary") else entry

        def remove(self, *a, **k):
            pass

    class _Triples:
        def get(self, *a, **k):
            return []

        def remove(self, *a, **k):
            pass

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.knowledge = _Knowledge()
    pipe.triples = _Triples()
    pipe.digest = SimpleNamespace(record=lambda **k: None)
    # e1 indexed under four aliases; e2 is an unrelated entry that must survive.
    pipe._url_index = {
        "https://canon": "e1",
        "https://raw": "e1",
        "https://orig": "e1",
        "https://arxiv.org/abs/2511.0001": "e1",
        "https://other": "e2",
    }

    result = pipe.strike("e1")

    assert result["paper"] is True
    # Every e1 alias gone; the unrelated e2 alias survives.
    assert pipe._url_index == {"https://other": "e2"}


def test_get_historical_feature_requests_tolerates_non_dict_json():
    """An FR entry whose content is valid JSON but not an object (list/scalar)
    must not blow up the **fr_data spread; degrade that row to a title-only
    record rather than aborting the whole listing."""
    bad = SimpleNamespace(
        id="fr_bad", title="Legacy FR", tags=["fr"], content='["not", "an", "object"]',
        metadata={},
    )
    good = SimpleNamespace(
        id="fr_good", title="Good FR", tags=["fr"],
        content='{"title": "Good FR", "extra": 1}', metadata={},
    )

    class _Knowledge:
        def get_by_tier(self, _tier):
            return [bad, good]

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.knowledge = _Knowledge()

    frs = pipe.get_historical_feature_requests()

    assert len(frs) == 2  # neither row aborted the listing
    by_id = {f["id"]: f for f in frs}
    assert by_id["fr_bad"]["title"] == "Legacy FR"  # fell back to entry.title
    assert by_id["fr_good"]["extra"] == 1  # dict content still spread in


def test_readability_fallback_cfg_handles_malformed_config():
    """fetcher.readability_fallback extraction must fail-closed (None), never
    raise, on absent/malformed config — `fetcher: null` is the case that broke
    pipeline init (PR #47)."""
    from researcher.pipeline import _readability_fallback_cfg

    assert _readability_fallback_cfg(None) is None
    assert _readability_fallback_cfg("not-a-dict") is None
    assert _readability_fallback_cfg({}) is None
    assert _readability_fallback_cfg({"fetcher": None}) is None  # fetcher: null
    assert _readability_fallback_cfg({"fetcher": "str"}) is None
    cfg = {"proxy": "https://r.jina.ai/{url}", "hosts": ["x.com"]}
    assert _readability_fallback_cfg({"fetcher": {"readability_fallback": cfg}}) == cfg


# ---------------------------------------------------------------------------
# ingest_url_with_body (fr_researcher_22486af4, layer 2)
# ---------------------------------------------------------------------------


def _body_pipe():
    pipe = ResearchPipeline.__new__(ResearchPipeline)
    captured = {}
    pipe._url_index = {}
    pipe.knowledge = SimpleNamespace(add=lambda e: captured.__setitem__("entry", e))
    pipe.digest = SimpleNamespace(record=lambda **k: None)
    return pipe, captured


@pytest.mark.asyncio
async def test_ingest_url_with_body_stores_research_entry():
    from khonliang.knowledge.store import EntryStatus, Tier

    pipe, captured = _body_pipe()
    eid = await pipe.ingest_url_with_body(
        "https://x.substack.com/p/a", "# Heading\n\nthe article body",
        content_type="text/markdown",
    )
    assert eid
    e = captured["entry"]
    # Same shape as fetch_paper success.
    assert e.tier == Tier.IMPORTED
    assert e.status == EntryStatus.INGESTED
    assert e.scope == "research"  # retrievable via find_relevant/brief_on
    assert e.tags == ["paper"]
    assert e.source == "https://x.substack.com/p/a"  # source=URL, not file://
    assert "the article body" in e.content
    assert e.metadata["source"] == "url_with_body"


@pytest.mark.asyncio
async def test_ingest_url_with_body_strips_html():
    pipe, captured = _body_pipe()
    await pipe.ingest_url_with_body(
        "https://x.com/a",
        "<html><body><h1>T</h1><p>hello world</p></body></html>",
        content_type="text/html",
    )
    e = captured["entry"]
    assert "hello world" in e.content
    assert "<p>" not in e.content


@pytest.mark.asyncio
async def test_ingest_url_with_body_empty_body_returns_none():
    pipe, captured = _body_pipe()
    assert await pipe.ingest_url_with_body("https://x.com/a", "   ") is None
    assert "entry" not in captured  # nothing stored


@pytest.mark.asyncio
async def test_ingest_url_with_body_dedupes_on_url():
    pipe, captured = _body_pipe()
    first = await pipe.ingest_url_with_body("https://x.com/a", "body one")
    second = await pipe.ingest_url_with_body("https://x.com/a", "body two")
    assert second == first  # same url -> existing entry, not re-stored


@pytest.mark.asyncio
async def test_ingest_url_with_body_rejects_non_http_url():
    """The method contracts on a URL and stores it as the entry source/dedupe
    key — a file:// or bare-string input must raise, not silently ingest."""
    pipe, captured = _body_pipe()
    for bad in ("file:///etc/passwd", "not-a-url", "ftp://h/x"):
        with pytest.raises(ValueError, match="absolute http"):
            await pipe.ingest_url_with_body(bad, "# T\n\nbody")
    assert "entry" not in captured  # nothing stored


@pytest.mark.asyncio
async def test_ingest_url_with_body_strips_url_before_storing():
    """is_http_url tolerates surrounding whitespace; the stored source/dedupe
    key must be the stripped URL, or a padded variant breaks dedupe/backlinks."""
    pipe, captured = _body_pipe()
    eid = await pipe.ingest_url_with_body("  https://x.com/a  ", "body")
    e = captured["entry"]
    assert e.source == "https://x.com/a"  # no surrounding spaces
    assert pipe._url_index.get("https://x.com/a") == eid  # indexed under stripped key
    # A subsequent unpadded ingest collapses to the same entry (dedupe intact).
    assert await pipe.ingest_url_with_body("https://x.com/a", "body two") == eid


@pytest.mark.asyncio
async def test_ingest_url_with_body_blank_content_type_normalized():
    """A direct pipeline caller passing content_type='' must NOT misroute
    through the HTML converter (_detect_format('', '') defaults to HTML) and
    must not store the blank value — normalize to text/markdown up-front."""
    pipe, captured = _body_pipe()
    await pipe.ingest_url_with_body(
        "https://x.com/a", "# Title\n\nplain markdown body", content_type="   ",
    )
    e = captured["entry"]
    assert e.metadata["content_type"] == "text/markdown"  # normalized, not ""
    # Markdown passthrough kept the heading text (HTML strip would not apply).
    assert "plain markdown body" in e.content


# ---------------------------------------------------------------------------
# ingest_idea search-query promotion (bug_developer_609eecb0 / dog_912b9f0d)
# ---------------------------------------------------------------------------


def _idea_pipe(parsed):
    pipe = ResearchPipeline.__new__(ResearchPipeline)
    captured = {}
    pipe.knowledge = SimpleNamespace(add=lambda e: captured.__setitem__("entry", e))
    pipe.digest = SimpleNamespace(record=lambda **k: None)
    pipe.idea_parser = SimpleNamespace(handle=lambda text: _as_async(parsed))
    return pipe, captured


async def _as_async(value):
    return value


@pytest.mark.asyncio
async def test_ingest_idea_promotes_per_claim_search_queries():
    # Parser put the query per-claim and left top-level search_queries empty —
    # ingest_idea must promote (and dedupe) them so research_idea isn't starved.
    pipe, captured = _idea_pipe({
        "success": True, "title": "An idea",
        "claims": [
            {"claim": "c1", "search_query": "query one"},
            {"claim": "c2", "search_query": "query two"},
            {"claim": "c3", "search_query": "query one"},  # duplicate
        ],
        "search_queries": [],
    })
    await pipe.ingest_idea("some idea text")
    assert captured["entry"].metadata["search_queries"] == ["query one", "query two"]


@pytest.mark.asyncio
async def test_ingest_idea_unions_top_level_and_per_claim_queries():
    pipe, captured = _idea_pipe({
        "success": True, "title": "x",
        "claims": [{"claim": "c", "search_query": "from claim"}],
        "search_queries": ["top level", "  "],  # blank entries filtered
    })
    await pipe.ingest_idea("t")
    assert captured["entry"].metadata["search_queries"] == ["top level", "from claim"]
