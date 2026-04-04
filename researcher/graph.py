"""Concept graph and matrix views over the distilled knowledge base.

Two views of the same data:

1. **Matrix**: Concepts × Papers — which papers cover which concepts,
   with relationship types and scores. Good for finding coverage gaps.

2. **Concept Graph**: Network of concepts connected through papers.
   Traverse chains like "GRPO → MAGRPO → C3 → Weight Learning".
   Similar to LinkedIn connection graphs but for research concepts.

Both built from the TripleStore + KnowledgeStore data.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from khonliang.knowledge.store import KnowledgeStore, Tier
from khonliang.knowledge.triples import TripleStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matrix View
# ---------------------------------------------------------------------------

@dataclass
class MatrixCell:
    """A cell in the concept × paper matrix."""
    predicates: List[str] = field(default_factory=list)
    confidence: float = 0.0


def build_concept_matrix(
    triples: TripleStore,
    min_confidence: float = 0.5,
    min_connections: int = 2,
    max_concepts: int = 50,
) -> Dict[str, Any]:
    """Build a concept × paper matrix from the triple store.

    Returns:
        {
            "concepts": ["GRPO", "consensus", ...],
            "papers": ["paper:abc123", ...],
            "matrix": {"GRPO": {"paper:abc123": {"predicates": [...], "confidence": 0.9}}},
            "concept_counts": {"GRPO": 5, ...},
        }
    """
    all_triples = triples.get(min_confidence=min_confidence, limit=5000)

    # Build concept → paper connections
    concept_papers: Dict[str, Dict[str, MatrixCell]] = defaultdict(dict)
    paper_set: Set[str] = set()

    for t in all_triples:
        source = t.source or ""

        # Subject is a concept, source is a paper
        if source.startswith("paper:"):
            if source not in concept_papers[t.subject]:
                concept_papers[t.subject][source] = MatrixCell()
            cell = concept_papers[t.subject][source]
            cell.predicates.append(t.predicate)
            cell.confidence = max(cell.confidence, t.confidence)
            paper_set.add(source)

            # Object is also a concept connected to this paper
            if source not in concept_papers[t.object]:
                concept_papers[t.object][source] = MatrixCell()
            cell2 = concept_papers[t.object][source]
            cell2.predicates.append(t.predicate)
            cell2.confidence = max(cell2.confidence, t.confidence)

    # Filter to concepts with enough connections
    filtered = {
        concept: papers
        for concept, papers in concept_papers.items()
        if len(papers) >= min_connections
    }

    # Sort by connection count, limit
    sorted_concepts = sorted(
        filtered.keys(),
        key=lambda c: len(filtered[c]),
        reverse=True,
    )[:max_concepts]

    # Build output
    matrix = {}
    for concept in sorted_concepts:
        matrix[concept] = {
            paper: {"predicates": cell.predicates, "confidence": cell.confidence}
            for paper, cell in filtered[concept].items()
        }

    return {
        "concepts": sorted_concepts,
        "papers": sorted(paper_set),
        "matrix": matrix,
        "concept_counts": {c: len(filtered[c]) for c in sorted_concepts},
    }


def format_matrix(matrix_data: Dict[str, Any], knowledge: KnowledgeStore) -> str:
    """Format the matrix as readable text."""
    concepts = matrix_data["concepts"]
    matrix = matrix_data["matrix"]
    counts = matrix_data["concept_counts"]

    # Resolve paper IDs to titles
    paper_titles = {}
    for paper_id in matrix_data["papers"]:
        entry_id = paper_id.replace("paper:", "")
        entry = knowledge.get(entry_id)
        if entry:
            paper_titles[paper_id] = entry.title[:40]
        else:
            paper_titles[paper_id] = entry_id[:16]

    lines = [f"## Concept × Paper Matrix ({len(concepts)} concepts, {len(matrix_data['papers'])} papers)\n"]

    for concept in concepts[:30]:
        papers = matrix[concept]
        lines.append(f"### {concept} ({counts[concept]} papers)")
        for paper_id, cell in sorted(papers.items(), key=lambda x: -x[1]["confidence"]):
            title = paper_titles.get(paper_id, paper_id)
            preds = ", ".join(set(cell["predicates"]))
            lines.append(f"  [{cell['confidence']:.0%}] {title} — {preds}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Concept Graph
# ---------------------------------------------------------------------------

@dataclass
class ConceptNode:
    """A node in the concept graph."""
    name: str
    paper_count: int = 0
    connections: Dict[str, List[str]] = field(default_factory=dict)
    # connections = {"MAGRPO": ["improves_on"], "consensus": ["applies_to"]}


def build_concept_graph(
    triples: TripleStore,
    min_confidence: float = 0.5,
) -> Dict[str, ConceptNode]:
    """Build a concept graph from triples.

    Nodes are concepts (subjects/objects). Edges are predicates.
    """
    all_triples = triples.get(min_confidence=min_confidence, limit=5000)

    nodes: Dict[str, ConceptNode] = {}

    for t in all_triples:
        # Create/update subject node
        if t.subject not in nodes:
            nodes[t.subject] = ConceptNode(name=t.subject)
        # Create/update object node
        if t.object not in nodes:
            nodes[t.object] = ConceptNode(name=t.object)

        # Add edge from subject → object
        subj_node = nodes[t.subject]
        if t.object not in subj_node.connections:
            subj_node.connections[t.object] = []
        if t.predicate not in subj_node.connections[t.object]:
            subj_node.connections[t.object].append(t.predicate)

        # Count papers
        if t.source and t.source.startswith("paper:"):
            subj_node.paper_count = max(subj_node.paper_count, 1)
            nodes[t.object].paper_count = max(nodes[t.object].paper_count, 1)

    return nodes


def trace_chain(
    graph: Dict[str, ConceptNode],
    start: str,
    max_depth: int = 4,
    max_branches: int = 3,
) -> str:
    """Trace a concept chain from a starting node.

    Returns a tree-like text representation:
        GRPO
        ├── improved_by → MAGRPO
        │   ├── applied_to → LLM Collaboration
        │   └── extends → C3
        └── used_by → ConsensusEngine
    """
    if start not in graph:
        # Try case-insensitive match
        matches = [k for k in graph if k.lower() == start.lower()]
        if matches:
            start = matches[0]
        else:
            return f"Concept '{start}' not found in graph."

    lines = [start]
    visited = {start}
    _trace_recursive(graph, start, lines, visited, "", max_depth, max_branches, 0)
    return "\n".join(lines)


def _trace_recursive(
    graph: Dict[str, ConceptNode],
    node_name: str,
    lines: List[str],
    visited: Set[str],
    prefix: str,
    max_depth: int,
    max_branches: int,
    depth: int,
):
    if depth >= max_depth:
        return

    node = graph.get(node_name)
    if not node:
        return

    connections = [
        (target, preds)
        for target, preds in node.connections.items()
        if target not in visited
    ]

    # Sort by number of predicates (more connected = more interesting)
    connections.sort(key=lambda x: -len(x[1]))
    connections = connections[:max_branches]

    for i, (target, predicates) in enumerate(connections):
        is_last = i == len(connections) - 1
        branch = "└── " if is_last else "├── "
        continuation = "    " if is_last else "│   "

        pred_str = ", ".join(predicates[:2])
        lines.append(f"{prefix}{branch}{pred_str} → {target}")

        visited.add(target)
        _trace_recursive(
            graph, target, lines, visited,
            prefix + continuation, max_depth, max_branches, depth + 1,
        )


def find_paths(
    graph: Dict[str, ConceptNode],
    start: str,
    end: str,
    max_depth: int = 5,
) -> List[List[Tuple[str, str, str]]]:
    """Find paths between two concepts.

    Returns list of paths, each path is [(node, predicate, next_node), ...].
    """
    if start not in graph or end not in graph:
        return []

    paths = []
    _find_paths_recursive(graph, start, end, [], set(), paths, max_depth)
    return paths


def _find_paths_recursive(
    graph: Dict[str, ConceptNode],
    current: str,
    end: str,
    path: List[Tuple[str, str, str]],
    visited: Set[str],
    results: List,
    max_depth: int,
):
    if len(path) >= max_depth:
        return
    if current == end:
        results.append(list(path))
        return

    visited.add(current)
    node = graph.get(current)
    if not node:
        return

    for target, predicates in node.connections.items():
        if target not in visited:
            for pred in predicates[:1]:  # Take first predicate per edge
                path.append((current, pred, target))
                _find_paths_recursive(
                    graph, target, end, path, visited, results, max_depth,
                )
                path.pop()

    visited.discard(current)
