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
    """Stand-in for researcher.RelevanceScorer.

    brief_on reads ``.threshold`` when ``project`` is set, and calls
    ``._embed`` from the second-stage embedding fallback. Tests that
    want to exercise the fallback path pass a dict of
    ``{text: embedding_vec}``; everything else returns None, which the
    fallback treats as "embedding unavailable" and short-circuits.
    """

    threshold = 0.5

    def __init__(self, embeddings: dict[str, list[float]] | None = None):
        self._embeddings = embeddings or {}

    async def _embed(self, text: str):
        return self._embeddings.get(text)


class _FakePipeline:
    """Captures every search() call so tests can assert multi-query fan-out.

    ``embeddings`` lets a test wire up stage-2 fallback: a dict of
    ``{embed_text: vector}`` that the fake relevance scorer returns
    from ``_embed``. Omit (or pass ``None``) for pure stage-1 tests.
    """

    def __init__(
        self,
        search_by_query: dict[str, List[KnowledgeEntry]],
        summaries=None,
        embeddings: dict[str, list[float]] | None = None,
    ):
        self._search_by_query = search_by_query
        self.calls: List[tuple[str, int]] = []
        self.knowledge = _FakeKnowledge(summaries)
        self.relevance = _FakeRelevance(embeddings)

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
        # .relevance, .search(), and (for stage-2 fallback)
        # .relevance._embed. The surrounding server wiring
        # wants a few more attributes — stub them.
        pipeline.triples = _Stub()
        pipeline.pool = _Stub()
        pipeline.config = {"projects": {}}
        pipeline.digest = _Stub()

        mcp = create_research_server(pipeline)
        tool_fn = _get_registered_tool_fn(mcp, "brief_on")
        assert tool_fn is not None, "brief_on tool not found on MCP server"

        # brief_on became async in fr_researcher_c4df6fc5 (second-stage
        # embedding fallback uses await on pipeline.relevance._embed).
        # Drive it via asyncio.run; inspect.iscoroutinefunction check
        # keeps the helper forward-compatible if it ever reverts to sync.
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(tool_fn):
            result = asyncio.run(tool_fn(**kwargs))
        else:
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
# Second-stage embedding fallback (fr_researcher_c4df6fc5)
# ---------------------------------------------------------------------------


def _stub_knowledge_with_tiers(summaries, extras=None):
    """Knowledge stub that also exposes .get_by_tier for the fallback path.

    brief_on's fallback no longer calls get_by_tier directly (it uses
    pipeline.search), so this just backstops in case a refactor adds
    a get_by_tier call — returning an empty list is a safe default.
    """
    fake = _FakeKnowledge(summaries)
    fake.get_by_tier = lambda _tier: list((extras or {}).values())
    return fake


def test_brief_on_embedding_fallback_closes_recall_gap(call_brief_on):
    """Stage-2 fires when stage-1 union < top_k.

    Two FTS hits for top_k=3; stage 2 embeds a broader candidate pool
    and returns one entry above threshold. Result: 3 hits, one tagged
    as ``embedding`` in source_by_id.
    """
    # Stage-1 FTS hits.
    a = _entry("aaa", "fts hit A")
    b = _entry("bbb", "fts hit B")
    # Stage-2-only candidate — present in FTS for a broader query
    # (individual word), absent from the original topic/context queries.
    c = _entry("ccc", "semantic neighbor", content="semantically close body")

    # Two-word topic so the stage-1 query key ("alpha beta") differs from
    # stage-2's per-token candidate queries ("alpha" / "beta") — the fake
    # pipeline keys by EXACT query string, so a single-word topic would feed
    # stage 2's candidate pool straight into stage 1 and stage 2 would
    # never fire (fts_hits would already reach top_k).
    pipeline = _FakePipeline(
        {
            # Stage-1 query: only a and b surface.
            "alpha beta": [a, b],
            # Stage-2 token query: c enters the candidate pool only here.
            "beta": [c],
        },
        embeddings={
            # Query embedding (topic only, since no context provided).
            "alpha beta": [1.0, 0.0],
            # ccc is aligned with the query — high similarity => merged.
            "semantic neighbor\n\nsemantically close body": [0.95, 0.31],
        },
    )

    out = call_brief_on(pipeline, topic="alpha beta", detail="brief", top_k=3)

    # Diagnostics: 2 from FTS, 1 from embedding, no short-circuit.
    diag = out["retrieval_diagnostics"]
    assert diag["fts_hits"] == 2, diag
    assert diag["embedding_hits"] == 1, diag
    assert diag["union_size"] == 3, diag
    assert diag["embedding_short_circuit"] is False, diag

    # source_by_id labels the fallback hit.
    assert diag["source_by_id"].get("ccc") == "embedding"
    assert diag["source_by_id"].get("aaa") == "fts"
    assert diag["source_by_id"].get("bbb") == "fts"

    # FTS hits rank ahead of embedding hits (any query_hit_count > 0
    # beats an embedding-only hit with count 0).
    assert out["source_ids"][-1] == "ccc"


def test_brief_on_fallback_skipped_when_stage1_fills_top_k(call_brief_on):
    """If stage 1 already has >= top_k, stage 2 must NOT fire.

    Detectable via ``embedding_hits == 0`` and
    ``embedding_short_circuit == False`` (short-circuit means tried-and-
    failed; not-fired means the guard skipped the stage entirely).
    """
    entries = [_entry(f"id{i}", f"t{i}") for i in range(5)]
    pipeline = _FakePipeline(
        {"topic": entries},
        embeddings={"topic": [1.0, 0.0]},  # would work if called
    )

    out = call_brief_on(pipeline, topic="topic", top_k=3)
    diag = out["retrieval_diagnostics"]
    assert diag["fts_hits"] >= 3
    assert diag["embedding_hits"] == 0
    assert diag["embedding_short_circuit"] is False


def test_brief_on_fallback_short_circuits_when_embed_unavailable(call_brief_on):
    """No embeddings wired up => ``_embed`` returns None => short-circuit.

    Diagnostics record ``embedding_short_circuit: true`` and the brief
    falls back to stage-1-only output.
    """
    a = _entry("aaa", "only fts hit")
    # Empty embeddings dict — _FakeRelevance._embed returns None for
    # every input, mirroring a dead/misconfigured Ollama instance.
    pipeline = _FakePipeline({"topic": [a]}, embeddings={})

    out = call_brief_on(pipeline, topic="topic", top_k=5)
    diag = out["retrieval_diagnostics"]
    assert diag["fts_hits"] == 1
    assert diag["embedding_hits"] == 0
    assert diag["embedding_short_circuit"] is True
    assert out["source_ids"] == ["aaa"]


def test_brief_on_fallback_below_threshold_not_merged(call_brief_on):
    """Candidates below the similarity threshold must not pad the result.

    Even when stage 1 underfills top_k, a weakly-related candidate
    stays out — per the FR's 'prefer shorter brief to padded one'.
    """
    a = _entry("aaa", "stage 1 hit")
    c = _entry("ccc", "weak match", content="weak body")
    # Two-word topic: keeps the stage-2 token query ("beta") distinct from
    # the stage-1 key so c reaches stage 2's pool without inflating stage 1.
    pipeline = _FakePipeline(
        {"alpha beta": [a], "beta": [c]},
        embeddings={
            "alpha beta": [1.0, 0.0],
            # Orthogonal vector => similarity 0.0, below threshold.
            "weak match\n\nweak body": [0.0, 1.0],
        },
    )

    out = call_brief_on(pipeline, topic="alpha beta", top_k=5)
    diag = out["retrieval_diagnostics"]
    assert diag["fts_hits"] == 1
    assert diag["embedding_hits"] == 0, (
        "below-threshold candidate should NOT be merged; diag=%r" % diag
    )
    assert diag["embedding_short_circuit"] is False


def test_brief_on_fallback_respects_project_scope(call_brief_on):
    """When ``project`` is set, stage 2 is skipped per FR out-of-scope note.

    Per-project relevance-score filtering is a stage-1 concern; stage 2
    would bypass it. Current contract: embedding_hits == 0 when project
    is supplied.
    """
    a = _entry("aaa", "fts hit")
    a.metadata = {"relevance_scores": {"khonliang": 0.9}}
    c = _entry("ccc", "sem match", content="semantic body")
    pipeline = _FakePipeline(
        {"topic": [a, c]},
        embeddings={
            "topic": [1.0, 0.0],
            "sem match\n\nsemantic body": [0.99, 0.0],
        },
    )

    out = call_brief_on(pipeline, topic="topic", project="khonliang", top_k=5)
    diag = out["retrieval_diagnostics"]
    # Stage 2 must be gated off entirely when project is set.
    assert diag["embedding_hits"] == 0
    assert diag["embedding_short_circuit"] is False


def test_brief_on_diagnostics_field_shape(call_brief_on):
    """FR acceptance: retrieval_diagnostics gains the 5 new fields."""
    pipeline = _FakePipeline({"topic": [_entry("aaa", "t")]})
    out = call_brief_on(pipeline, topic="topic")
    diag = out["retrieval_diagnostics"]
    # Old fields still present.
    assert {"queries_run", "total_hits", "top_k_chosen", "per_query_hits"} <= set(diag)
    # New fields from fr_researcher_c4df6fc5.
    assert {
        "fts_hits",
        "embedding_hits",
        "union_size",
        "embedding_short_circuit",
        "source_by_id",
    } <= set(diag)


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
