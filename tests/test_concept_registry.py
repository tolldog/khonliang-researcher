"""FR fr_khonliang_0f3c7542 — concept-graph-over-RAG consumer of the generic
described-registry primitive (index + nearest-match expand).

Composes the lib graph builder + KnowledgeStore + TripleStore behind a two-call
contract. The load-bearing acceptance test asserts the two-call path returns
FEWER tokens than the equivalent flat top-k retrieval for the same query, and
that expand batches >=2 ids in one call.
"""

from __future__ import annotations

import pytest

from khonliang.knowledge.store import KnowledgeEntry, KnowledgeStore, Tier
from khonliang.knowledge.triples import TripleStore

from researcher.concept_registry import build_concept_registry


# ---------------------------------------------------------------------------
# Fixture: ~9 concepts, each with one chunky research section, wired into a
# small graph. Sized so full-section content dominates one-line descriptions.
# ---------------------------------------------------------------------------

_CONCEPTS = {
    "GRPO": "Group Relative Policy Optimization is a reinforcement learning "
    "method that removes the value model and estimates advantages from grouped "
    "sample rewards, reducing memory and compute for policy training on math.",
    "MAGRPO": "Multi-Agent GRPO extends group relative optimization to a team of "
    "cooperating agents, sharing a group baseline across agents so credit is "
    "assigned jointly during collaborative rollouts and debate.",
    "PPO": "Proximal Policy Optimization clips the surrogate objective to keep "
    "policy updates within a trust region, requiring a separate learned value "
    "function to estimate advantages during each optimization step.",
    "DPO": "Direct Preference Optimization aligns a language model directly on "
    "pairwise preference data without an explicit reward model, reframing the "
    "RLHF objective as a simple classification loss over chosen and rejected.",
    "ConsensusEngine": "The consensus engine aggregates multiple agent votes "
    "into a single decision using weighted confidence and quorum thresholds, "
    "and is the runtime that consumes trained collaborative policies.",
    "LoCoMo": "LoCoMo is a long-conversation memory benchmark used to evaluate "
    "graph-augmented associative memory against flat retrieval baselines over "
    "multi-session dialogue with temporal reasoning questions.",
    "GAAMA": "Graph Augmented Associative Memory stores agent memories as a "
    "typed graph and retrieves connected neighborhoods on demand, surpassing "
    "flat RAG baselines on the LoCoMo long-context benchmark.",
    "RAG": "Retrieval Augmented Generation stuffs top-k retrieved chunks into "
    "the prompt as flat context, which over-retrieves noisy sections and spends "
    "tokens on material the model never uses for the answer.",
    "Embedding": "Dense embeddings map text into a vector space where cosine "
    "similarity approximates semantic relatedness, powering nearest-neighbor "
    "retrieval and the relevance scoring behind flat RAG pipelines.",
}

_EDGES = [
    ("GRPO", "improved_by", "MAGRPO"),
    ("GRPO", "compared_to", "PPO"),
    ("GRPO", "compared_to", "DPO"),
    ("MAGRPO", "used_by", "ConsensusEngine"),
    ("GAAMA", "evaluated_on", "LoCoMo"),
    ("GAAMA", "surpasses", "RAG"),
    ("RAG", "uses", "Embedding"),
]


def _stores(tmp_path):
    knowledge = KnowledgeStore(str(tmp_path / "knowledge.db"))
    triples = TripleStore(str(tmp_path / "triples.db"))
    for i, (name, content) in enumerate(_CONCEPTS.items()):
        knowledge.add(
            KnowledgeEntry(
                id=f"paper:{i}",
                tier=Tier.DERIVED,
                title=name,
                content=content,
                scope="research",
                source=f"paper:{i}",
                confidence=0.9,
            )
        )
    for j, (s, p, o) in enumerate(_EDGES):
        triples.add(s, p, o, confidence=0.9, source=f"paper:{j % len(_CONCEPTS)}")
    return knowledge, triples


def _tokens(text: str) -> int:
    """One tokenizer used on BOTH sides. Whitespace proxy (tiktoken absent);
    stated in the test + PR so the comparison is apples-to-apples."""
    return len(text.split())


# ---------------------------------------------------------------------------
# index()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_returns_concepts_with_descriptions(tmp_path):
    knowledge, triples = _stores(tmp_path)
    reg = build_concept_registry(knowledge, triples)
    entries = await reg.index()
    ids = {e.id for e in entries}
    # every graph node appears
    assert {"GRPO", "MAGRPO", "GAAMA", "RAG"} <= ids
    # descriptions are synthesized relation summaries, not full section text
    grpo = next(e for e in entries if e.id == "GRPO")
    assert "GRPO" in grpo.description
    assert "improved_by MAGRPO" in grpo.description or "MAGRPO" in grpo.description
    # one-liner stays short
    assert len(grpo.description) <= 121


@pytest.mark.asyncio
async def test_index_limit_keeps_most_connected(tmp_path):
    knowledge, triples = _stores(tmp_path)
    reg = build_concept_registry(knowledge, triples)
    entries = await reg.index(limit=3)
    assert len(entries) == 3
    # GRPO (3 outgoing edges) is the most-connected -> survives the cap
    assert "GRPO" in {e.id for e in entries}


# ---------------------------------------------------------------------------
# expand() — batching + depth + RAG sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_batches_two_ids_in_one_call(tmp_path):
    knowledge, triples = _stores(tmp_path)
    reg = build_concept_registry(knowledge, triples)
    result = await reg.expand(["GRPO", "GAAMA"], depth=1)
    assert set(result.keys()) == {"GRPO", "GAAMA"}
    # detail = matching KnowledgeStore sections (RAG detail)
    assert "policy" in result["GRPO"].detail.lower()
    # connected concepts walked one hop
    grpo_connected = {c["id"] for c in result["GRPO"].connected}
    assert "MAGRPO" in grpo_connected


@pytest.mark.asyncio
async def test_expand_depth_two_walks_further(tmp_path):
    knowledge, triples = _stores(tmp_path)
    reg = build_concept_registry(knowledge, triples)
    d1 = await reg.expand(["GRPO"], depth=1)
    d2 = await reg.expand(["GRPO"], depth=2)
    ids1 = {c["id"] for c in d1["GRPO"].connected}
    ids2 = {c["id"] for c in d2["GRPO"].connected}
    # ConsensusEngine is two hops out (GRPO -> MAGRPO -> ConsensusEngine)
    assert "ConsensusEngine" not in ids1
    assert "ConsensusEngine" in ids2


@pytest.mark.asyncio
async def test_expand_resolves_case_insensitively(tmp_path):
    knowledge, triples = _stores(tmp_path)
    reg = build_concept_registry(knowledge, triples)
    result = await reg.expand(["grpo"])
    assert result["grpo"].id == "GRPO"


@pytest.mark.asyncio
async def test_expand_unknown_concept_absent(tmp_path):
    knowledge, triples = _stores(tmp_path)
    reg = build_concept_registry(knowledge, triples)
    result = await reg.expand(["NotAConcept", "GRPO"])
    assert set(result.keys()) == {"GRPO"}


# ---------------------------------------------------------------------------
# ACCEPTANCE: two-call path is cheaper than flat top-k for the same query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_call_is_cheaper_than_flat_topk(tmp_path):
    """The concept consumer returns fewer tokens than the equivalent flat
    top-k retrieval for the same query.

    Baseline = the repo's EXISTING flat path: knowledge.search(query, k) summing
    full section content (this is what get_paper_context stuffs into the prompt).
    Two-call = index() (all one-liners) + expand([picked ids]) — counted
    together, conservatively.
    """
    knowledge, triples = _stores(tmp_path)
    reg = build_concept_registry(knowledge, triples)

    # A query that lexically hits SEVERAL sections -> flat over-retrieves.
    query = "policy optimization advantage retrieval baseline"
    k = 6

    # --- Flat baseline: same query, same scope, k full sections stuffed ---
    flat_entries = knowledge.search(query, scope="research", limit=k)
    flat_text = "\n\n".join(e.content for e in flat_entries)
    flat_tokens = _tokens(flat_text)

    # --- Two-call path: cheap index, then expand only the picked concept ---
    index_entries = await reg.index()
    index_text = "\n".join(f"{e.id} {e.description}" for e in index_entries)
    picked = ["GRPO"]  # LLM would pick the closest from the cheap index
    expanded = await reg.expand(picked, depth=1)
    expand_text = "\n\n".join(
        x.detail + " " + " ".join(c["id"] for c in x.connected)
        for x in expanded.values()
    )
    two_call_tokens = _tokens(index_text) + _tokens(expand_text)

    assert flat_entries, "flat baseline must actually retrieve something"
    assert two_call_tokens < flat_tokens, (
        f"two-call ({two_call_tokens}) must be cheaper than "
        f"flat top-{k} ({flat_tokens})"
    )
