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


def test_complementarity_names_strongest_provider():
    # Two repos implement the concept; the finding must name the strongest one
    # (highest score), not the alphabetical first (codex P2).
    footprints = {
        "triple provenance": {"alpha": 0.42, "zeta": 0.9, "gappy": 0.05},
    }
    gaps = {"gappy": {normalize_concept("triple provenance")}}
    out = classify_cross_repo_findings(
        footprints, gaps=gaps, repos=["alpha", "zeta", "gappy"]
    )
    comp = [f for f in out if f.signal_class == COMPLEMENTARITY]
    assert comp and comp[0].detail["provider"] == "zeta"
    assert comp[0].score == pytest.approx(0.9)


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
    def __init__(self, subject, obj, source, sources=None):
        self.subject = subject
        self.object = obj
        self.source = source
        # multi-source provenance; primary token is `source`
        self.sources = sources if sources is not None else ([source] if source else [])


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
        lambda knowledge, triples, **kw: pipe._footprints,
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
        lambda knowledge, triples, **kw: pipe._footprints,
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


def test_latent_excludes_scores_from_out_of_scope_repos(monkeypatch):
    # A concept implemented ONLY by an excluded registered repo (repo_c) must
    # NOT be reported as latent for a repo_a,repo_b subset scan (codex P2).
    footprints = {
        "repo_c only concept": {"repo_c": 0.9, "repo_a": 0.0, "repo_b": 0.0},
        # a genuine latent concept: corpus (paper target) rates it, no repo uses it
        "genuine latent": {"paper_target": 0.85, "repo_a": 0.0, "repo_b": 0.0},
    }
    evidence_sources = [
        {"project": "repo_a"}, {"project": "repo_b"}, {"project": "repo_c"},
    ]
    pipe = _make_pipeline(
        footprints=footprints,
        gaps_entries=[],
        triples=[_StubTriple("genuine latent", "x", "paper:g1")],
        evidence_sources=evidence_sources,
    )
    monkeypatch.setattr(
        "khonliang_researcher.build_project_scores",
        lambda knowledge, triples, **kw: pipe._footprints,
    )
    report = pipe.scan_cross_repo_integration(repos=["repo_a", "repo_b"])
    latent_concepts = {
        f["concept"] for f in report["findings"]
        if f["signal_class"] == LATENT_CONCEPT
    }
    assert "repo_c only concept" not in latent_concepts
    assert "genuine latent" in latent_concepts


def test_latent_provenance_includes_all_multi_source_tokens(monkeypatch):
    footprints = {"genuine latent": {"paper_target": 0.85, "repo_a": 0.0, "repo_b": 0.0}}
    # triple whose PRIMARY source differs from a co-source token; both must surface
    triple = _StubTriple(
        "genuine latent", "x", "paper:primary",
        sources=["paper:primary", "paper:secondary"],
    )
    pipe = _make_pipeline(
        footprints=footprints,
        gaps_entries=[],
        triples=[triple],
        evidence_sources=[{"project": "repo_a"}, {"project": "repo_b"}],
    )
    monkeypatch.setattr(
        "khonliang_researcher.build_project_scores",
        lambda knowledge, triples, **kw: pipe._footprints,
    )
    report = pipe.scan_cross_repo_integration(repos=["repo_a", "repo_b"])
    latent = [f for f in report["findings"] if f["signal_class"] == LATENT_CONCEPT]
    assert latent
    assert set(latent[0]["corpus_sources"]) == {"paper:primary", "paper:secondary"}


def test_pipeline_passes_low_threshold_to_footprint_builder(monkeypatch):
    # A threshold below build_project_scores' 0.3 default must be forwarded as
    # min_score so [threshold, 0.3) concepts aren't silently dropped (codex P2).
    seen = {}

    def _fake_scores(knowledge, triples, min_score=0.3, **kw):
        seen["min_score"] = min_score
        return {"low concept": {"repo_a": 0.25, "repo_b": 0.25}}

    pipe = _make_pipeline(
        footprints={},
        gaps_entries=[],
        triples=[],
        evidence_sources=[{"project": "repo_a"}, {"project": "repo_b"}],
    )
    monkeypatch.setattr("khonliang_researcher.build_project_scores", _fake_scores)

    report = pipe.scan_cross_repo_integration(
        repos=["repo_a", "repo_b"], threshold=0.2
    )
    assert seen["min_score"] == pytest.approx(0.2)
    # the sub-0.3 duplication is now visible
    assert any(f["concept"] == "low concept" for f in report["findings"])


def test_pipeline_threshold_floor_never_exceeds_default(monkeypatch):
    seen = {}

    def _fake_scores(knowledge, triples, min_score=0.3, **kw):
        seen["min_score"] = min_score
        return {}

    pipe = _make_pipeline(
        footprints={},
        gaps_entries=[],
        triples=[],
        evidence_sources=[{"project": "a"}, {"project": "b"}],
    )
    monkeypatch.setattr("khonliang_researcher.build_project_scores", _fake_scores)
    pipe.scan_cross_repo_integration(repos=["a", "b"], threshold=0.6)
    # threshold above default must not raise the floor (stays 0.3)
    assert seen["min_score"] == pytest.approx(0.3)


def test_pipeline_dedups_explicit_duplicate_repo_args(monkeypatch):
    # repos="x,x" must NOT pass the >=2 guard as a bogus single-repo scan
    # (codex P3).
    pipe = _make_pipeline(
        footprints={},
        gaps_entries=[],
        triples=[],
        evidence_sources=[{"project": "x"}],
    )
    monkeypatch.setattr(
        "khonliang_researcher.build_project_scores",
        lambda knowledge, triples, **kw: {},
    )
    report = pipe.scan_cross_repo_integration(repos=["x", "x"])
    assert report["repos"] == ["x"]
    assert "error" in report
    assert report["finding_count"] == 0


def test_pipeline_rejects_unknown_repo_names(monkeypatch):
    # A typo/stale name must fail fast, not silently compare against a repo with
    # no footprint/gap data (codex P2).
    pipe = _make_pipeline(
        footprints={},
        gaps_entries=[],
        triples=[],
        evidence_sources=[{"project": "repo_a"}, {"project": "repo_b"}],
    )
    monkeypatch.setattr(
        "khonliang_researcher.build_project_scores",
        lambda knowledge, triples, **kw: {},
    )
    report = pipe.scan_cross_repo_integration(repos=["repo_a", "typo_repo"])
    assert "unknown repo" in report.get("error", "")
    assert "typo_repo" in report["error"]
    assert report["finding_count"] == 0
    assert report["auto_filed"] is False


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
