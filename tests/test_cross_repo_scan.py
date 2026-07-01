"""Tests for the cross-repo integration scan (FR fr_khonliang-researcher_33561994,
Phase 1 — on-demand report).

Covers the pure classification / filtering / dedup logic in
``researcher.cross_repo_scan`` plus the pipeline orchestration method with
stubbed concept/corpus/gap collaborators (no live LLM/bus). Assertions map
directly to the FR acceptance criteria:

  - a scan over >=2 repos returns findings classified duplication |
    complementarity | latent-concept;
  - generic infra concepts are filtered;
  - findings carry provenance (repos, concepts, corpus sources);
  - a previously-filed/dismissed candidate is deduped out;
  - NOTHING is auto-filed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from researcher.cross_repo_scan import (
    COMPLEMENTARITY,
    DUPLICATION,
    LATENT_CONCEPT,
    Finding,
    build_report,
    classify_cross_repo_findings,
    is_infra_concept,
    normalize_concept,
)


# --------------------------------------------------------------- pure logic ---

def test_duplication_two_repos_same_concept():
    footprints = {
        "consensus voting": {"khonliang": 0.8, "researcher": 0.7},
    }
    out = classify_cross_repo_findings(
        footprints, repos=["khonliang", "researcher"]
    )
    assert len(out) == 1
    f = out[0]
    assert f.signal_class == DUPLICATION
    assert sorted(f.repos) == ["khonliang", "researcher"]
    # provenance: the concept + both repos are carried on the finding
    assert f.concept == "consensus voting"
    assert "shared-lib" in f.summary


def test_single_repo_is_not_duplication():
    footprints = {"consensus voting": {"khonliang": 0.8, "researcher": 0.1}}
    out = classify_cross_repo_findings(
        footprints, repos=["khonliang", "researcher"], threshold=0.4
    )
    # only one repo above threshold -> no duplication (and no gap -> nothing)
    assert [f for f in out if f.signal_class == DUPLICATION] == []


def test_complementarity_off_real_gap_signal():
    footprints = {"triple provenance": {"researcher": 0.75, "developer": 0.05}}
    # developer lists 'triple provenance' as a gap (planned/exploring)
    gaps = {"developer": {normalize_concept("triple provenance")}}
    out = classify_cross_repo_findings(
        footprints, gaps=gaps, repos=["researcher", "developer"]
    )
    comp = [f for f in out if f.signal_class == COMPLEMENTARITY]
    assert len(comp) == 1
    f = comp[0]
    assert f.detail["provider"] == "researcher"
    assert f.detail["gap_repo"] == "developer"
    assert sorted(f.repos) == ["developer", "researcher"]


def test_score_asymmetry_without_gap_is_not_complementarity():
    # researcher implements it, developer doesn't — but developer does NOT
    # list it as a gap. Must NOT manufacture complementarity from a score diff.
    footprints = {"triple provenance": {"researcher": 0.75, "developer": 0.05}}
    out = classify_cross_repo_findings(
        footprints, gaps={}, repos=["researcher", "developer"]
    )
    assert [f for f in out if f.signal_class == COMPLEMENTARITY] == []


def test_latent_concept_no_repo_uses_it():
    footprints = {
        # corpus knows this concept (high score for some paper target) but no
        # target repo implements it.
        "speculative decoding": {"paper_target": 0.9, "khonliang": 0.1},
    }
    latent = [
        {"concept": "speculative decoding", "score": 0.9, "sources": ["paper:abc123"]},
    ]
    out = classify_cross_repo_findings(
        footprints, latent=latent, repos=["khonliang", "researcher"]
    )
    lat = [f for f in out if f.signal_class == LATENT_CONCEPT]
    assert len(lat) == 1
    f = lat[0]
    # provenance: corpus source ids are carried
    assert f.corpus_sources == ["paper:abc123"]
    assert sorted(f.repos) == ["khonliang", "researcher"]


def test_latent_skipped_when_a_repo_already_uses_it():
    footprints = {"speculative decoding": {"khonliang": 0.8}}
    latent = [{"concept": "speculative decoding", "score": 0.9, "sources": ["p1"]}]
    out = classify_cross_repo_findings(
        footprints, latent=latent, repos=["khonliang", "researcher"]
    )
    assert [f for f in out if f.signal_class == LATENT_CONCEPT] == []


@pytest.mark.parametrize(
    "concept",
    ["HTTP client", "structured logging", "yaml config", "JSON serialization"],
)
def test_infra_concepts_filtered(concept):
    assert is_infra_concept(concept) is True
    footprints = {concept: {"a": 0.9, "b": 0.9}}
    out = classify_cross_repo_findings(footprints, repos=["a", "b"])
    assert out == []


def test_infra_denylist_does_not_over_match():
    # "api" is infra but must not match "rapid prototyping" via substring.
    assert is_infra_concept("rapid prototyping") is False
    # a real concept that merely contains an infra word as a substring survives
    assert is_infra_concept("consensus routing") is False


def test_dedup_against_filed_fr_by_key():
    footprints = {"consensus voting": {"a": 0.8, "b": 0.7}}
    dup = Finding(DUPLICATION, "consensus voting", ["a", "b"], "")
    already_filed = [{"dedup_key": dup.dedup_key()}]
    out = classify_cross_repo_findings(
        footprints, repos=["a", "b"], already_filed=already_filed
    )
    assert out == []


def test_dedup_against_dismissed_by_concept():
    footprints = {"consensus voting": {"a": 0.8, "b": 0.7}}
    dismissed = [{"concept": "Consensus Voting"}]  # concept-only, any class
    out = classify_cross_repo_findings(
        footprints, repos=["a", "b"], dismissed=dismissed
    )
    assert out == []


def test_ranking_duplication_before_latent():
    footprints = {
        "consensus voting": {"a": 0.8, "b": 0.7},          # duplication
        "speculative decoding": {"paper": 0.9, "a": 0.0},  # latent
    }
    latent = [{"concept": "speculative decoding", "score": 0.9, "sources": ["p1"]}]
    out = classify_cross_repo_findings(
        footprints, latent=latent, repos=["a", "b"]
    )
    assert out[0].signal_class == DUPLICATION
    assert out[-1].signal_class == LATENT_CONCEPT


def test_build_report_marks_nothing_filed():
    footprints = {"consensus voting": {"a": 0.8, "b": 0.7}}
    findings = classify_cross_repo_findings(footprints, repos=["a", "b"])
    report = build_report(findings, repos=["a", "b"])
    assert report["auto_filed"] is False
    assert report["by_class"][DUPLICATION] == 1
    # every finding is a provenance-carrying dict
    for f in report["findings"]:
        assert set(["signal_class", "concept", "repos", "dedup_key"]) <= set(f)


# --------------------------------------------------------- pipeline wiring ----

class _StubTriple:
    def __init__(self, subject, obj, source):
        self.subject = subject
        self.object = obj
        self.source = source


class _StubTriples:
    def __init__(self, triples):
        self._triples = triples

    def get(self, min_confidence=0.3, limit=5000):
        return self._triples


class _StubKnowledge:
    """Only ``get_by_tier`` is exercised by the scan (capability gaps)."""

    def __init__(self, derived):
        self._derived = derived

    def get_by_tier(self, tier):
        return self._derived


class _StubDigest:
    def __init__(self):
        self.records = []

    def record(self, **kwargs):
        self.records.append(kwargs)


def _make_pipeline(*, footprints, gaps_entries, triples, evidence_sources):
    """Build a ResearchPipeline with stubbed collaborators, monkeypatching the
    one lib call (build_project_scores) at call sites via a footprints stub."""
    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.knowledge = _StubKnowledge(gaps_entries)
    pipe.triples = _StubTriples(triples)
    pipe.digest = _StubDigest()
    pipe._evidence_sources = evidence_sources
    # stub list_evidence_sources to return registered repos
    pipe.list_evidence_sources = lambda owned_locally=None: evidence_sources
    pipe._footprints = footprints
    return pipe


def _cap_entry(target, concept, status):
    return SimpleNamespace(
        tags=["capability", f"cap:{target}"],
        title=concept,
        metadata={"capability_status": status, "target": target, "concept": concept},
    )


def test_pipeline_scan_end_to_end(monkeypatch):
    footprints = {
        "consensus voting": {"khonliang": 0.8, "researcher": 0.7},   # duplication
        "triple provenance": {"researcher": 0.75, "developer": 0.02},  # complement
        "http client": {"khonliang": 0.9, "researcher": 0.9},        # infra -> filtered
        "speculative decoding": {"paper_target": 0.95, "khonliang": 0.0},  # latent
    }
    gaps_entries = [
        _cap_entry("developer", "triple provenance", "planned"),
        _cap_entry("khonliang", "some existing thing", "exists"),  # not a gap
    ]
    triples = [
        _StubTriple("speculative decoding", "llm inference", "paper:spec1"),
    ]
    evidence_sources = [
        {"project": "khonliang"},
        {"project": "researcher"},
        {"project": "developer"},
    ]
    pipe = _make_pipeline(
        footprints=footprints,
        gaps_entries=gaps_entries,
        triples=triples,
        evidence_sources=evidence_sources,
    )

    monkeypatch.setattr(
        "khonliang_researcher.build_project_scores",
        lambda knowledge, triples: pipe._footprints,
    )

    report = pipe.scan_cross_repo_integration(
        repos=["khonliang", "researcher", "developer"]
    )

    classes = {f["signal_class"] for f in report["findings"]}
    assert DUPLICATION in classes
    assert COMPLEMENTARITY in classes
    assert LATENT_CONCEPT in classes

    # infra filtered
    assert all(f["concept"] != "http client" for f in report["findings"])

    # provenance: latent finding carries corpus sources
    latent = [f for f in report["findings"] if f["signal_class"] == LATENT_CONCEPT]
    assert latent and latent[0]["corpus_sources"] == ["paper:spec1"]

    # nothing auto-filed
    assert report["auto_filed"] is False
    # dedup gap noted because no filed/dismissed supplied
    assert "dedup_gap" in report


def test_pipeline_dedup_removes_filed_candidate(monkeypatch):
    footprints = {"consensus voting": {"khonliang": 0.8, "researcher": 0.7}}
    pipe = _make_pipeline(
        footprints=footprints,
        gaps_entries=[],
        triples=[],
        evidence_sources=[{"project": "khonliang"}, {"project": "researcher"}],
    )
    monkeypatch.setattr(
        "khonliang_researcher.build_project_scores",
        lambda knowledge, triples: pipe._footprints,
    )

    dup = Finding(DUPLICATION, "consensus voting", ["khonliang", "researcher"], "")
    report = pipe.scan_cross_repo_integration(
        repos=["khonliang", "researcher"],
        already_filed=[{"dedup_key": dup.dedup_key()}],
    )
    assert report["finding_count"] == 0
    assert report["auto_filed"] is False
    # dedup source supplied -> no dedup gap note
    assert "dedup_gap" not in report


def test_pipeline_requires_two_repos(monkeypatch):
    pipe = _make_pipeline(
        footprints={},
        gaps_entries=[],
        triples=[],
        evidence_sources=[{"project": "onlyone"}],
    )
    monkeypatch.setattr(
        "khonliang_researcher.build_project_scores",
        lambda knowledge, triples: {},
    )
    report = pipe.scan_cross_repo_integration()
    assert report["finding_count"] == 0
    assert "error" in report
    assert report["auto_filed"] is False
