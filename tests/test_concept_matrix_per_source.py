"""FR cecbd365 — the concept matrix credits each co-sourcing paper with its OWN
confidence (via Triple.confidence_for), not the aggregate max, and gates a
sub-threshold co-source out of the matrix entirely.

Drives the real TripleStore (post lib PR #85, which exposes per-source
confidence) so this is an integration test of the lib accessor + the matrix
consumer together.
"""

from __future__ import annotations

from khonliang.knowledge.triples import TripleStore

from researcher.graph import build_concept_matrix


def _store(tmp_path):
    return TripleStore(str(tmp_path / "triples.db"))


def test_matrix_credits_each_paper_at_its_own_confidence(tmp_path):
    store = _store(tmp_path)
    # Two papers of different strength assert the SAME fact -> one unioned triple
    # with per-source confidences {p1: 0.9, p2: 0.6} and aggregate max 0.9.
    store.add("Alpha", "relates_to", "Beta", confidence=0.9, source="paper:p1")
    store.add("Alpha", "relates_to", "Beta", confidence=0.6, source="paper:p2")

    m = build_concept_matrix(store, min_confidence=0.5, min_connections=1)
    cells = m["matrix"]["Alpha"]
    assert cells["paper:p1"]["confidence"] == 0.9
    # the weaker paper keeps its OWN 0.6 — not the 0.9 it used to inherit
    assert cells["paper:p2"]["confidence"] == 0.6


def test_matrix_gates_subthreshold_cosource(tmp_path):
    store = _store(tmp_path)
    # strong (0.9) + weak (0.2) assert the same fact; the aggregate max (0.9)
    # clears the fetch gate, so the weak paper used to ride along at 0.9 and
    # wrongly pass min_confidence. It must now be dropped on its own 0.2.
    store.add("Alpha", "relates_to", "Beta", confidence=0.9, source="paper:strong")
    store.add("Alpha", "relates_to", "Beta", confidence=0.2, source="paper:weak")

    m = build_concept_matrix(store, min_confidence=0.5, min_connections=1)
    cells = m["matrix"]["Alpha"]
    assert "paper:strong" in cells
    assert cells["paper:strong"]["confidence"] == 0.9
    assert "paper:weak" not in cells  # gated by its own 0.2 < 0.5
    # the gated paper is also absent from the global paper set
    assert "paper:weak" not in m["papers"]


def test_matrix_object_concept_also_uses_per_source(tmp_path):
    # The object side of a triple is credited the same way as the subject.
    store = _store(tmp_path)
    store.add("Alpha", "relates_to", "Beta", confidence=0.9, source="paper:p1")
    store.add("Alpha", "relates_to", "Beta", confidence=0.6, source="paper:p2")

    m = build_concept_matrix(store, min_confidence=0.5, min_connections=1)
    beta = m["matrix"]["Beta"]
    assert beta["paper:p1"]["confidence"] == 0.9
    assert beta["paper:p2"]["confidence"] == 0.6
