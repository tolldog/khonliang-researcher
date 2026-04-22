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
from typing import Any, List

import pytest

from khonliang.knowledge.store import EntryStatus, KnowledgeEntry, Tier


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


def _build_server_with_fake_pipeline(pipeline) -> Any:
    """Spin up a research server wrapping the fake pipeline.

    The server normally requires a real pipeline (knowledge + triples +
    pool). We only exercise brief_on, which touches pipeline.search,
    pipeline.knowledge.get, and pipeline.relevance — fake those and
    stub out the pieces the tool constructor otherwise needs.
    """
    from researcher import server as server_mod

    # The full create_research_server wires a Synthesizer and DistillWorker
    # onto the pipeline; those aren't on our fake. Rather than stub the
    # whole thing, invoke the tool directly by poking at the registered
    # MCP tool table via a thin wrapper server build.
    #
    # Easiest path: construct an MCP with just our tool by reusing the
    # decorator pattern from the real server. We avoid the DistillWorker
    # and Synthesizer constructors entirely.
    raise NotImplementedError  # placeholder — unused; see direct-call tests below


def _call_brief_on(pipeline, **kwargs) -> dict:
    """Invoke the tool body directly without a full MCP server.

    We import ``create_research_server`` side-effect-free and then fish
    the registered tool out. If that's too invasive, we can instead
    inline-call the same code path; here the tool body is a closure over
    ``pipeline`` so we build a minimal replica.
    """
    # Strategy: rather than reconstruct the full server (which needs a
    # Synthesizer / Worker / ModelPool), call the closure body directly
    # by importing the module and constructing the tool ourselves with
    # the same logic. To avoid code drift, we instead invoke the real
    # function via FastMCP's tool registry after a minimal server build
    # that skips worker/synth setup.
    from researcher.server import create_research_server

    # Minimal pipeline needs: .knowledge, .triples, .relevance, .pool,
    # .search(), .config, plus enough for Synthesizer + DistillWorker
    # constructors to not crash. We stub those.
    class _Stub:
        def __getattr__(self, _name):
            return _Stub()

        def __call__(self, *a, **kw):
            return _Stub()

    pipeline.triples = _Stub()
    pipeline.pool = _Stub()
    pipeline.config = {"projects": {}}
    pipeline.digest = _Stub()

    # DistillWorker(pipeline) and Synthesizer(knowledge, triples, pool)
    # may try to do real work; patch them out.
    import researcher.server as srv_mod
    import researcher.worker as worker_mod
    import researcher.synthesizer as synth_mod

    class _DummyWorker:
        def __init__(self, *a, **kw):
            pass

        stats = {"running": False, "pending": 0, "processed": 0, "failed": 0}

        def count_pending(self):
            return 0

    class _DummySynth:
        def __init__(self, *a, **kw):
            pass

    original_worker = srv_mod.DistillWorker if hasattr(srv_mod, "DistillWorker") else None
    # create_research_server imports DistillWorker / Synthesizer inside the
    # function body, so we patch the source modules.
    worker_mod.DistillWorker = _DummyWorker  # type: ignore[attr-defined]
    synth_mod.Synthesizer = _DummySynth  # type: ignore[attr-defined]

    try:
        mcp = create_research_server(pipeline)
    finally:
        if original_worker is not None:
            srv_mod.DistillWorker = original_worker  # type: ignore[attr-defined]

    # FastMCP stores tools in a private registry. Find brief_on and call.
    tool_fn = None
    # Modern FastMCP exposes tools via ``_tool_manager._tools`` or similar.
    for attr in ("_tool_manager", "tool_manager"):
        mgr = getattr(mcp, attr, None)
        if mgr is None:
            continue
        tools = getattr(mgr, "_tools", None) or getattr(mgr, "tools", None)
        if not tools:
            continue
        entry = tools.get("brief_on") if isinstance(tools, dict) else None
        if entry is None:
            continue
        tool_fn = getattr(entry, "fn", None) or getattr(entry, "func", None) or entry
        break

    assert tool_fn is not None, "brief_on tool not found on MCP server"

    result = tool_fn(**kwargs)
    return json.loads(result)


def test_brief_on_runs_three_queries_when_context_supplied():
    """Multi-query expansion: topic / topic+context / context all fire."""
    e1 = _entry("aaa", "topic-only hit")
    e2 = _entry("bbb", "both hit")
    e3 = _entry("ccc", "context-only hit")

    pipeline = _FakePipeline({
        "ollama code review": [e1, e2],
        "ollama code review reviewer agent": [e2],
        "reviewer agent": [e2, e3],
    })

    out = _call_brief_on(
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


def test_brief_on_single_query_when_no_context():
    pipeline = _FakePipeline({"ollama code review": [_entry("aaa", "x")]})
    out = _call_brief_on(pipeline, topic="ollama code review")
    assert [c[0] for c in pipeline.calls] == ["ollama code review"]
    assert out["retrieval_diagnostics"]["queries_run"] == ["ollama code review"]


def test_brief_on_reuses_stored_distill_summary_no_redistill():
    """When a <id>_summary entry exists, brief_on reads its key_findings
    rather than re-running the distiller. This is the FR's 'reuse the
    existing distill_paper primitive' invariant."""
    e = _entry("xyz", "Paper X")
    summary = _summary("xyz", "KEY_FINDING_SENTINEL: local models suffice")

    pipeline = _FakePipeline(
        {"query": [e]},
        summaries={"xyz_summary": summary},
    )

    out = _call_brief_on(pipeline, topic="query", detail="brief")
    # The sentinel finding from the stored summary must appear in the brief.
    assert "KEY_FINDING_SENTINEL" in out["brief"]
    assert out["source_ids"] == ["xyz"]


def test_brief_on_falls_back_to_content_when_no_summary():
    e = _entry("nodistill", "Raw Paper", content="First line is the claim.\nrest...")
    pipeline = _FakePipeline({"q": [e]})

    out = _call_brief_on(pipeline, topic="q", detail="brief")
    assert "First line is the claim." in out["brief"]


def test_brief_on_brief_detail_under_2000_chars():
    """Acceptance: detail='brief' output fits in <=2000 chars."""
    entries = [_entry(f"id{i:02d}", f"Title {i} " * 20) for i in range(10)]
    pipeline = _FakePipeline({"topic": entries[:10], "topic ctx": entries[5:], "ctx": entries})

    out = _call_brief_on(
        pipeline, topic="topic", in_context_of="ctx", detail="brief", top_k=10,
    )
    assert len(out["brief"]) <= 2000


def test_brief_on_empty_topic_rejected():
    pipeline = _FakePipeline({})
    out = _call_brief_on(pipeline, topic="")
    assert out["source_ids"] == []
    assert "non-empty topic" in out["brief"]


def test_brief_on_no_hits_returns_empty_diagnostics():
    pipeline = _FakePipeline({"nothing": []})
    out = _call_brief_on(pipeline, topic="nothing")
    assert out["source_ids"] == []
    assert out["retrieval_diagnostics"]["total_hits"] == 0
    # queries_run is still populated so callers can tune.
    assert out["retrieval_diagnostics"]["queries_run"] == ["nothing"]


def test_brief_on_return_shape():
    pipeline = _FakePipeline({"t": [_entry("id1", "T")]})
    out = _call_brief_on(pipeline, topic="t")
    assert set(out.keys()) == {"brief", "source_ids", "retrieval_diagnostics"}
    diag = out["retrieval_diagnostics"]
    assert set(diag.keys()) >= {"queries_run", "total_hits", "top_k_chosen"}


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
def test_brief_on_real_corpus_smoke():
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

    out = _call_brief_on(
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
