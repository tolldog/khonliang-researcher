"""
Concept-graph-over-RAG consumer of the generic described-registry primitive.

First new consumer of ``khonliang.registry.DescribedRegistry`` (FR
fr_khonliang_0f3c7542): reduce flat-RAG to an as-needed concept graph behind a
two-call API.

- ``index()``  -> concepts + one-line descriptions (synthesized from each
  concept's own graph relations — the "item's own summary").
- ``expand([ids], depth)`` -> for each chosen concept: matching KnowledgeStore
  sections (the RAG detail) + connected concepts walked out to ``depth`` via the
  existing concept-graph.

This composes existing primitives — ``build_concept_graph`` (the graph
builder), ``KnowledgeStore.search`` (RAG sections), and ``TripleStore`` (edges)
— rather than duplicating ``concept_tree`` / ``knowledge_search`` /
``concept_context``. It is the unified two-call entry point *over* them: the LLM
scans the cheap index, picks the closest concept(s), and issues ONE batched
expand instead of receiving a flat top-k chunk dump.

The graph algorithm still lives in the khonliang-researcher lib; this adapter
consumes its structured ``EntityNode.connections`` output (BFS mirroring
``trace_chain``'s depth/branch semantics) instead of the rendered ASCII string.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from khonliang.registry import (
    DescribedRegistry,
    ExpandedItem,
    IndexEntry,
)

# Length caps that keep the index genuinely cheap.
_DESC_MAX_RELATIONS = 3
_DESC_MAX_CHARS = 120
_DETAIL_SECTION_LIMIT = 3
_DETAIL_SECTION_CHARS = 600


def _describe(name: str, node: Any) -> str:
    """
    Synthesize a one-line description for a concept from its own graph relations.

    No per-node summary is stored, so the concept's "own summary" is its top
    outgoing relations, e.g. ``GRPO — improved_by MAGRPO; used_by ConsensusEngine``.
    Deterministic, no LLM.
    """
    rels: List[str] = []
    for target, predicates in node.connections.items():
        if not predicates:
            continue
        rels.append(f"{predicates[0]} {target}")
        if len(rels) >= _DESC_MAX_RELATIONS:
            break
    if not rels:
        desc = name
    else:
        desc = f"{name} — " + "; ".join(rels)
    if len(desc) > _DESC_MAX_CHARS:
        desc = desc[: _DESC_MAX_CHARS - 1].rstrip() + "…"
    return desc


class ConceptGraphAdapter:
    """
    ``ItemAdapter`` mapping the concept graph onto the described-registry
    contract. Thin: it composes the lib graph builder + the stores, holds no
    graph algorithm of its own.

    Args:
        knowledge: KnowledgeStore (RAG sections + build_concept_graph targets).
        triples:   TripleStore (graph edges).
        min_confidence: edge-confidence floor for graph construction.
        max_branches:   per-node branch cap when walking connected concepts
                        (mirrors ``trace_chain`` semantics).
        search_scope:   KnowledgeStore scope for section retrieval. Defaults to
                        ``None`` = search all scopes, so concepts that entered the
                        graph from non-research sources (``scan:``/``idea:``
                        triples with ``capability`` knowledge) still expand to
                        real sections instead of ``(no sections)``.
    """

    def __init__(
        self,
        knowledge: Any,
        triples: Any,
        *,
        min_confidence: float = 0.5,
        max_branches: int = 3,
        search_scope: Optional[str] = None,
    ) -> None:
        self.knowledge = knowledge
        self.triples = triples
        self.min_confidence = min_confidence
        self.max_branches = max_branches
        self.search_scope = search_scope

    def _graph(self) -> Dict[str, Any]:
        # Compose the lib graph builder — no local graph algorithm.
        from khonliang_researcher import build_concept_graph

        return build_concept_graph(
            self.triples,
            min_confidence=self.min_confidence,
            knowledge=self.knowledge,
        )

    async def catalog(
        self, scope: Optional[str] = None, limit: Optional[int] = None
    ) -> List[IndexEntry]:
        # ``scope`` is accepted for Protocol conformance but inert here: the
        # composed lib ``build_concept_graph`` builds nodes from all triples
        # (``source_prefix`` only gates document counts, not the node set), so
        # there is nothing to scope-filter without scope-aware graph construction
        # in the lib. Deferred to an FR phase.
        graph = self._graph()
        nodes = list(graph.values())
        # Most-connected concepts first so a soft limit keeps the salient ones.
        nodes.sort(key=lambda n: len(n.connections), reverse=True)
        if limit is not None:
            nodes = nodes[:limit]
        entries: List[IndexEntry] = []
        for node in nodes:
            meta: Dict[str, Any] = {}
            if getattr(node, "document_count", 0):
                meta["documents"] = node.document_count
            if node.connections:
                meta["connections"] = len(node.connections)
            entries.append(
                IndexEntry(
                    id=node.name,
                    description=_describe(node.name, node),
                    meta=meta,
                )
            )
        return entries

    async def expand(
        self, ids: Sequence[str], depth: int = 1
    ) -> Dict[str, ExpandedItem]:
        from khonliang_researcher import resolve_entity

        graph = self._graph()
        out: Dict[str, ExpandedItem] = {}
        for raw_id in ids:
            canonical = resolve_entity(graph, raw_id)
            if canonical is None:
                continue
            node = graph[canonical]
            detail = self._sections_for(canonical)
            connected = self._walk(graph, canonical, depth) if depth > 0 else []
            out[raw_id] = ExpandedItem(
                id=canonical,
                detail=detail,
                connected=connected,
            )
        return out

    def _sections_for(self, concept: str) -> str:
        """Matching KnowledgeStore sections for a concept (the RAG detail)."""
        entries = self.knowledge.search(
            concept, scope=self.search_scope, limit=_DETAIL_SECTION_LIMIT
        )
        if not entries:
            return ""
        parts: List[str] = []
        for e in entries:
            content = (e.content or "")[:_DETAIL_SECTION_CHARS]
            title = getattr(e, "title", "") or e.id
            parts.append(f"[{title}] {content}")
        return "\n\n".join(parts)

    def _walk(
        self, graph: Dict[str, Any], start: str, depth: int
    ) -> List[Dict[str, Any]]:
        """
        BFS connected concepts out to ``depth`` hops, mirroring ``trace_chain``:
        sort each node's neighbors by predicate count desc, take the top
        ``max_branches``. Returns structured ``{id, relation, depth}`` — the
        structured analogue of the ASCII tree ``concept_tree`` renders.
        """
        connected: List[Dict[str, Any]] = []
        visited = {start}
        frontier = [start]
        for hop in range(1, depth + 1):
            next_frontier: List[str] = []
            for node_name in frontier:
                node = graph.get(node_name)
                if node is None:
                    continue
                neighbors = sorted(
                    node.connections.items(),
                    key=lambda kv: len(kv[1]),
                    reverse=True,
                )[: self.max_branches]
                for target, predicates in neighbors:
                    if target in visited:
                        continue
                    visited.add(target)
                    next_frontier.append(target)
                    relation = predicates[0] if predicates else ""
                    connected.append(
                        {"id": target, "relation": relation, "depth": hop}
                    )
            frontier = next_frontier
            if not frontier:
                break
        return connected


def build_concept_registry(
    knowledge: Any,
    triples: Any,
    *,
    min_confidence: float = 0.5,
    max_branches: int = 3,
    max_index: int = 200,
    max_depth: int = 3,
    search_scope: Optional[str] = None,
) -> DescribedRegistry:
    """Wire a ``DescribedRegistry`` over the concept graph."""
    adapter = ConceptGraphAdapter(
        knowledge,
        triples,
        min_confidence=min_confidence,
        max_branches=max_branches,
        search_scope=search_scope,
    )
    return DescribedRegistry(adapter, max_index=max_index, max_depth=max_depth)
