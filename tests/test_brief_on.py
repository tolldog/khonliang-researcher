"""Tests for the ``brief_on`` MCP tool.

Two layers:

1. ``test_brief_on_multi_query_expansion_*`` — unit tests that mock the
   retrieval layer (``pipeline.search``) and assert (a) all three queries
   fire, (b) results are unioned and deduped by entry id, (c) entries
   that surface in multiple queries outrank entries that surface in only
   one, and (d) the per-source distill-reuse path loads the stored
   ``<id>_summary`` entry rather than running a new distillation.

2. ``test_brief_on_real_corpus_smoke`` — smoke test that opens the real
   researcher knowledge store (if one exists locally) and verifies the
   acceptance criterion from FR fr_researcher_5ad96ffe: a brief_on call
   for 'local Ollama code review models' in the context of the
   'khonliang reviewer agent' surfaces >=3 of the known corpus entries.
   Skipped when the DB is not present so CI stays green.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, List

import pytest

from khonliang.knowledge.store import EntryStatus, KnowledgeEntry, Tier


# ---------------------------------------------------------------------------
# FastMCP private-API isolation
# ---------------------------------------------------------------------------
#
# FastMCP does not currently expose a public API to retrieve a registered
# tool's underlying Python function. We reach into ``_tool_manager._tools``
# here; this is the ONLY place that happens so a FastMCP upgrade that
# changes the registry shape fails in one, obvious spot.
#
# If/when FastMCP exposes a public accessor (e.g. ``mcp.get_tool(name)``),
# replace this helper's body and the rest of the suite is unaffected.
def _get_registered_tool_fn(mcp: Any, name: str) -> Callable | None:
    for attr in ("_tool_manager", "tool_manager"):
        mgr = getattr(mcp, attr, None)
        if mgr is None:
            continue
        tools = getattr(mgr, "_tools", None) or getattr(mgr, "tools", None)
        if not tools:
            continue
        entry = tools.get(name) if isinstance(tools, dict) else None
        if entry is None:
            continue
        return getattr(entry, "fn", None) or getattr(entry, "func", None) or entry
    return None


# ---------------------------------------------------------------------------
# Mocked unit tests
# ---------------------------------------------------------------------------


class _FakeKnowledge:
    """Minimal stand-in for KnowledgeStore — only .get() is used by brief_on."""

    def __init__(self, summaries: dict[str, KnowledgeEntry] | None = None):
        self._summaries = summaries or {}

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        return self._summaries.get(entry_id)


class _FakeRelevance:
    threshold = 0.5


class _FakePipeline:
    """Captures every search() call so tests can assert multi-query fan-out."""

    def __init__(self, search_by_query: dict[str, List[KnowledgeEntry]], summaries=None):
        self._search_by_query = search_by_query
        self.calls: List[tuple[str, int]] = []
        self.knowledge = _FakeKnowledge(summaries)
        self.relevance = _FakeRelevance()

    def search(self, query: str, limit: int = 10) -> List[KnowledgeEntry]:
        self.calls.append((query, limit))
        return list(self._search_by_query.get(query, []))


def _entry(eid: str, title: str, content: str = "body") -> KnowledgeEntry:
    return KnowledgeEntry(
        id=eid,
        tier=Tier.IMPORTED,
        title=title,
        content=content,
        scope="research",
        source=f"http://example.com/{eid}",
        status=EntryStatus.DISTILLED,
        tags=[],
        metadata={},
    )


def _summary(parent_id: str, key_finding: str) -> KnowledgeEntry:
    """Shape matches what pipeline._store_distillation actually writes."""
    return KnowledgeEntry(
        id=f"{parent_id}_summary",
        tier=Tier.DERIVED,
        title=f"Summary: {parent_id}",
        content=json.dumps({
            "title": f"paper {parent_id}",
            "abstract": "abstract text",
            "key_findings": [key_finding],
        }),
        scope="research",
        source=parent_id,
        status=EntryStatus.DISTILLED,
        tags=["summary"],
        metadata={"parent_id": parent_id},
    )


class _DummyWorker:
    def __init__(self, *a, **kw):
        pass

    stats = {"running": False, "pending": 0, "processed": 0, "failed": 0}

    def count_pending(self):
        return 0


class _DummySynth:
    def __init__(self, *a, **kw):
        pass


class _Stub:
    def __getattr__(self, _name):
        return _Stub()

    def __call__(self, *a, **kw):
        return _Stub()


@pytest.fixture
def call_brief_on(monkeypatch):
    """Return a callable that invokes the ``brief_on`` MCP tool.

    Uses pytest's ``monkeypatch`` so any attribute swaps on
    ``researcher.worker`` / ``researcher.synthesizer`` are automatically
    reverted at test teardown. No global state leaks between tests.
    """
    import researcher.worker as worker_mod
    import researcher.synthesizer as synth_mod
    from researcher.server import create_research_server

    # create_research_server instantiates a DistillWorker and a
    # Synthesizer. Neither is exercised by brief_on, so substitute
    # no-op stand-ins; monkeypatch reverts after each test.
    monkeypatch.setattr(worker_mod, "DistillWorker", _DummyWorker)
    monkeypatch.setattr(synth_mod, "Synthesizer", _DummySynth)

    def _invoke(pipeline, **kwargs) -> dict:
        # Minimal pipeline shim: brief_on only touches .knowledge,
        # .relevance, and .search(). The surrounding server wiring
        # wants a few more attributes — stub them.
        pipeline.triples = _Stub()
        pipeline.pool = _Stub()
        pipeline.config = {"projects": {}}
        pipeline.digest = _Stub()

        mcp = create_research_server(pipeline)
        tool_fn = _get_registered_tool_fn(mcp, "brief_on")
        assert tool_fn is not None, "brief_on tool not found on MCP server"

        result = tool_fn(**kwargs)
        return json.loads(result)

    return _invoke


def test_brief_on_runs_three_queries_when_context_supplied(call_brief_on):
    """Multi-query expansion: topic / topic+context / context all fire."""
    e1 = _entry("aaa", "topic-only hit")
    e2 = _entry("bbb", "both hit")
    e3 = _entry("ccc", "context-only hit")

    pipeline = _FakePipeline({
        "ollama code review": [e1, e2],
        "ollama code review reviewer agent": [e2],
        "reviewer agent": [e2, e3],
    })

    out = call_brief_on(
        pipeline,
        topic="ollama code review",
        in_context_of="reviewer agent",
        detail="brief",
    )

    # Three distinct searches.
    queries = [c[0] for c in pipeline.calls]
    assert queries == [
        "ollama code review",
        "ollama code review reviewer agent",
        "reviewer agent",
    ]

    diag = out["retrieval_diagnostics"]
    assert diag["queries_run"] == queries
    assert diag["total_hits"] == 3  # union across the three queries

    # Entry that surfaced in all three queries outranks single-query hits.
    assert out["source_ids"][0] == "bbb"
    assert set(out["source_ids"]) == {"aaa", "bbb", "ccc"}


def test_brief_on_single_query_when_no_context(call_brief_on):
    pipeline = _FakePipeline({"ollama code review": [_entry("aaa", "x")]})
    out = call_brief_on(pipeline, topic="ollama code review")
    assert [c[0] for c in pipeline.calls] == ["ollama code review"]
    assert out["retrieval_diagnostics"]["queries_run"] == ["ollama code review"]


def test_brief_on_reuses_stored_distill_summary_no_redistill(call_brief_on):
    """When a <id>_summary entry exists, brief_on reads its key_findings
    rather than re-running the distiller. This is the FR's 'reuse the
    existing distill_paper primitive' invariant."""
    e = _entry("xyz", "Paper X")
    summary = _summary("xyz", "KEY_FINDING_SENTINEL: local models suffice")

    pipeline = _FakePipeline(
        {"query": [e]},
        summaries={"xyz_summary": summary},
    )

    out = call_brief_on(pipeline, topic="query", detail="brief")
    # The sentinel finding from the stored summary must appear in the brief.
    assert "KEY_FINDING_SENTINEL" in out["brief"]
    assert out["source_ids"] == ["xyz"]


def test_brief_on_falls_back_to_content_when_no_summary(call_brief_on):
    e = _entry("nodistill", "Raw Paper", content="First line is the claim.\nrest...")
    pipeline = _FakePipeline({"q": [e]})

    out = call_brief_on(pipeline, topic="q", detail="brief")
    assert "First line is the claim." in out["brief"]


def test_brief_on_brief_detail_under_2000_chars(call_brief_on):
    """Acceptance: detail='brief' output fits in <=2000 chars."""
    entries = [_entry(f"id{i:02d}", f"Title {i} " * 20) for i in range(10)]
    pipeline = _FakePipeline({"topic": entries[:10], "topic ctx": entries[5:], "ctx": entries})

    out = call_brief_on(
        pipeline, topic="topic", in_context_of="ctx", detail="brief", top_k=10,
    )
    assert len(out["brief"]) <= 2000


def test_brief_on_empty_topic_rejected(call_brief_on):
    pipeline = _FakePipeline({})
    out = call_brief_on(pipeline, topic="")
    assert out["source_ids"] == []
    assert "non-empty topic" in out["brief"]
    # Contract: per_query_hits is always present, even on early-return.
    diag = out["retrieval_diagnostics"]
    assert diag["per_query_hits"] == {}


def test_brief_on_no_hits_returns_empty_diagnostics(call_brief_on):
    pipeline = _FakePipeline({"nothing": []})
    out = call_brief_on(pipeline, topic="nothing")
    assert out["source_ids"] == []
    assert out["retrieval_diagnostics"]["total_hits"] == 0
    # queries_run is still populated so callers can tune.
    assert out["retrieval_diagnostics"]["queries_run"] == ["nothing"]


def test_brief_on_return_shape(call_brief_on):
    pipeline = _FakePipeline({"t": [_entry("id1", "T")]})
    out = call_brief_on(pipeline, topic="t")
    assert set(out.keys()) == {"brief", "source_ids", "retrieval_diagnostics"}
    diag = out["retrieval_diagnostics"]
    assert set(diag.keys()) >= {"queries_run", "total_hits", "top_k_chosen", "per_query_hits"}


def test_brief_on_per_query_limit_respects_top_k(call_brief_on):
    """top_k > 10 must propagate to pipeline.search so enough candidates
    are pulled per query to fill the caller's requested top_k."""
    entries = [_entry(f"id{i:02d}", f"Title {i}") for i in range(25)]
    pipeline = _FakePipeline({"topic": entries})

    out = call_brief_on(pipeline, topic="topic", detail="brief", top_k=20)
    # Every recorded search must have been issued with limit >= 20.
    assert all(limit >= 20 for _q, limit in pipeline.calls), pipeline.calls
    # And the top_k_chosen honours the caller's request.
    assert out["retrieval_diagnostics"]["top_k_chosen"] == 20


def test_brief_on_full_detail_emits_untruncated_key_claim(call_brief_on):
    """detail='full' must not truncate the key_claim at 220 chars."""
    long_claim = "SENTINEL_FULL " + ("x" * 400)  # well past 220
    e = _entry("longclaim", "Long-claim paper")
    summary = KnowledgeEntry(
        id="longclaim_summary",
        tier=Tier.DERIVED,
        title="Summary: longclaim",
        content=json.dumps({
            "title": "paper longclaim",
            "abstract": "abstract text",
            "key_findings": [long_claim],
        }),
        scope="research",
        source="longclaim",
        status=EntryStatus.DISTILLED,
        tags=["summary"],
        metadata={"parent_id": "longclaim"},
    )
    pipeline = _FakePipeline({"topic": [e]}, summaries={"longclaim_summary": summary})

    out = call_brief_on(pipeline, topic="topic", detail="full")
    assert long_claim in out["brief"], (
        "full() formatter should emit the untruncated key_claim"
    )


# ---------------------------------------------------------------------------
# Real-corpus smoke test (gated on the local DB being present)
# ---------------------------------------------------------------------------


_REAL_DB_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "data" / "researcher.db",
    Path(os.environ.get("KHONLIANG_RESEARCHER_DB", "")) if os.environ.get("KHONLIANG_RESEARCHER_DB") else None,
]


def _find_real_db() -> Path | None:
    for candidate in _REAL_DB_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    return None


@pytest.mark.skipif(
    _find_real_db() is None,
    reason="real researcher knowledge store not present; skipping corpus smoke test",
)
def test_brief_on_real_corpus_smoke(call_brief_on):
    """FR acceptance: brief_on(topic='local Ollama code review models',
    in_context_of='khonliang reviewer agent') must surface >=3 of the
    known-corpus entries (CodeGPT Ollama guide, Local AI Master 2026,
    DEV Community Ollama Cloud comparison, Anthropic $25/PR, Greptile
    AI Code Review). The baseline (synthesize_topic) returned 1 of 5."""
    from khonliang.knowledge.store import KnowledgeStore

    db_path = _find_real_db()
    assert db_path is not None
    store = KnowledgeStore(str(db_path))

    # Minimal pipeline shim — brief_on only reaches into
    # pipeline.search / pipeline.knowledge / pipeline.relevance.
    class _ThinPipeline:
        def __init__(self, store):
            self.knowledge = store
            self.relevance = _FakeRelevance()

        def search(self, query: str, limit: int = 10):
            return store.search(query, scope="research", limit=limit)

    out = call_brief_on(
        _ThinPipeline(store),
        topic="local Ollama code review models",
        in_context_of="khonliang reviewer agent",
        detail="brief",
        top_k=10,
    )

    known_prefixes = [
        "1478ad79",  # CodeGPT Ollama guide
        "a90b0a21",  # Local AI Master 2026
        "2e51a600",  # DEV Community Ollama Cloud comparison
        "3ac9482c",  # Anthropic $25/PR
        "99c76cb1",  # Greptile AI Code Review
    ]
    surfaced = [
        p for p in known_prefixes
        if any(sid.startswith(p) for sid in out["source_ids"])
    ]
    assert len(surfaced) >= 3, (
        f"Expected >=3 known corpus entries in source_ids, got {surfaced} "
        f"out of candidates {known_prefixes}. source_ids={out['source_ids']}"
    )

    # And the brief itself stays within the detail=brief budget.
    assert len(out["brief"]) <= 2000
