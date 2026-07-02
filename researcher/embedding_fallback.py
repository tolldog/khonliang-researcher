"""Embedding-similarity second-stage retrieval for ``brief_on``.

``brief_on``'s primary retrieval is SQL FTS (``pipeline.search``) with
multi-query expansion. FTS nails literal-phrase matches; it misses
entries whose topic vocabulary diverges from the query vocabulary
(synonyms, paraphrases, acronym-rich titles). This module adds an
optional second stage: when the first-stage union is smaller than the
caller's ``top_k``, embed the query and a bounded candidate pool, rank
by cosine similarity, and merge hits above a confidence threshold.

Design notes
------------
- **No persistent embedding index.** The researcher DB stores FTS
  shadow tables but no content-embedding column. Building/maintaining
  a durable index is a separate maintenance concern (out of scope per
  FR ``fr_researcher_c4df6fc5``). This module instead embeds on the
  fly against a capped candidate pool, reusing the existing Ollama
  ``/api/embed`` pipeline via ``RelevanceScorer._embed``.
- **Threshold-based.** Low-confidence candidates are dropped rather
  than padded into the result. The FR prefers a shorter brief to a
  padded one.
- **Clean short-circuit.** When the embedding path is unavailable
  (no ``RelevanceScorer``, embedding-model offline, empty candidate
  pool), the caller is handed back a ``short_circuit=True`` result and
  a single diagnostic log line. No crash, no retries, no new index
  builds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from khonliang.knowledge.store import KnowledgeEntry, Tier

from researcher.relevance import cosine_similarity


logger = logging.getLogger(__name__)


# Embedding-similarity cutoff below which a candidate is not merged
# into the result set. Tuned conservatively: the FR acceptance only
# needs >=4/5 on a known-good query, and padding with low-confidence
# hits degrades the brief. Callers can override via ``threshold``.
DEFAULT_SIMILARITY_THRESHOLD = 0.55

# Upper bound on how many candidates get embedded per brief_on call.
# Embedding is the dominant cost here, so capping protects latency.
# Sized to comfortably cover the broader FTS candidate pool returned
# for typical multi-word queries (observed: ~50-100 matches on the
# current researcher corpus) without making brief_on visibly slow.
DEFAULT_CANDIDATE_POOL_CAP = 100


@dataclass
class EmbeddingHit:
    """A candidate entry ranked by embedding similarity."""

    entry: KnowledgeEntry
    similarity: float


async def _embed_or_none(pipeline: Any, text: str) -> Optional[List[float]]:
    """Call the pipeline's existing embedder. Returns ``None`` on any failure.

    Reuses ``pipeline.relevance._embed`` so this module does not
    introduce a new embedding model or a new HTTP client.
    """
    scorer = getattr(pipeline, "relevance", None)
    if scorer is None:
        return None
    embed_fn = getattr(scorer, "_embed", None)
    if embed_fn is None:
        return None
    try:
        return await embed_fn(text)
    except Exception as exc:  # defensive: never let embedding kill brief_on
        logger.warning("brief_on embedding call failed: %s", exc)
        return None


def _iter_candidate_entries(
    pipeline: Any,
    candidate_queries: Sequence[str],
    exclude_ids: set,
    cap: int,
) -> List[KnowledgeEntry]:
    """Build a bounded candidate pool for embedding-similarity ranking.

    Strategy: run each of ``candidate_queries`` through the pipeline's
    FTS (same entry point as stage 1, just with a larger ``limit`` and
    more permissive query variants) and union the results. This
    intentionally over-fetches — the embedding step below is what
    actually filters down to semantically-close hits.

    Entries already surfaced by stage 1 (``exclude_ids``) are omitted,
    as are empty-content entries. Capped at ``cap`` so embed-call
    cost stays bounded.
    """
    pool: List[KnowledgeEntry] = []
    seen: set = set()
    for q in candidate_queries:
        q = (q or "").strip()
        if not q:
            continue
        try:
            results = pipeline.search(q, limit=cap)
        except Exception:
            results = []
        for entry in results:
            if entry.id in exclude_ids or entry.id in seen:
                continue
            if not (entry.content or "").strip():
                continue
            pool.append(entry)
            seen.add(entry.id)
            if len(pool) >= cap:
                return pool
    return pool


def _default_candidate_queries(topic: str, context: str) -> List[str]:
    """Broader lenient queries used to build the stage-2 candidate pool.

    Includes the original topic/context plus individual tokens so the
    FTS BM25 layer returns a wider net. Embedding similarity then
    does the actual ranking.
    """
    queries: List[str] = []
    topic = (topic or "").strip()
    context = (context or "").strip()
    if topic:
        queries.append(topic)
    if context:
        queries.append(context)
    if topic and context:
        queries.append(f"{topic} {context}")
    # Individual tokens: catches entries that only share one
    # vocabulary word with the query.
    seen_tokens: set = set()
    for token in (topic + " " + context).split():
        tok = token.strip().lower()
        if len(tok) <= 2 or tok in seen_tokens:
            continue
        seen_tokens.add(tok)
        queries.append(tok)
    return queries


def _entry_embedding_text(entry: KnowledgeEntry) -> str:
    """Build the text we actually embed for a candidate.

    Title + first 1500 chars of content matches
    ``RelevanceScorer``'s own ``CONTENT_PREFIX_LEN``, keeping embed
    cost bounded and the input comparable to what ``relevance.score``
    uses.
    """
    title = entry.title or ""
    body = (entry.content or "")[:1500]
    return f"{title}\n\n{body}"


async def second_stage_embedding_hits(
    pipeline: Any,
    topic: str,
    context: str,
    exclude_ids: Iterable[str],
    needed: int,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    candidate_pool_cap: int = DEFAULT_CANDIDATE_POOL_CAP,
) -> tuple[List[EmbeddingHit], bool]:
    """Return embedding-ranked hits and whether the pipeline short-circuited.

    Args:
        pipeline: Researcher pipeline (must expose ``relevance`` and
            ``search``; anything else is treated as a short-circuit).
        topic: Brief topic — used both to build the query embedding
            and to derive broader candidate-pool FTS queries.
        context: Optional scoping phrase (``in_context_of``).
        exclude_ids: Entry ids already surfaced by stage 1 — omitted
            from the candidate pool.
        needed: How many additional hits stage 2 may contribute.
            ``<=0`` is a no-op returning ``([], False)``.
        threshold: Minimum cosine similarity for a hit to be merged.
        candidate_pool_cap: Upper bound on candidates embedded.

    Returns:
        ``(hits, short_circuited)``. ``hits`` is sorted by descending
        similarity and trimmed to ``needed``. ``short_circuited`` is
        ``True`` when the embedding path was unavailable (no scorer,
        query embedding failed, no candidates). In that case ``hits``
        is empty.
    """
    if needed <= 0:
        return [], False

    exclude = set(exclude_ids)

    query_text = topic if not context else f"{topic} {context}"
    query_embedding = await _embed_or_none(pipeline, query_text)
    if not query_embedding:
        logger.info(
            "brief_on embedding fallback: query-embedding unavailable, "
            "short-circuiting to first-stage results"
        )
        return [], True

    candidate_queries = _default_candidate_queries(topic, context)
    candidates = _iter_candidate_entries(
        pipeline, candidate_queries, exclude, candidate_pool_cap
    )
    if not candidates:
        logger.info(
            "brief_on embedding fallback: no candidate entries outside "
            "first-stage union, short-circuiting"
        )
        return [], True

    scored: List[EmbeddingHit] = []
    for entry in candidates:
        emb = await _embed_or_none(pipeline, _entry_embedding_text(entry))
        if not emb:
            continue
        sim = cosine_similarity(query_embedding, emb)
        if sim >= threshold:
            scored.append(EmbeddingHit(entry=entry, similarity=sim))

    scored.sort(key=lambda h: h.similarity, reverse=True)
    return scored[:needed], False
