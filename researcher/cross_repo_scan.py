"""Cross-repo integration-opportunity scan (FR fr_khonliang-researcher_33561994, Phase 1).

On-demand classification of cross-repo integration candidates from concept
footprints + capability gaps + latent corpus concepts. This module holds the
*pure* logic — no LLM, no bus, no store calls — so it is unit-testable with
stubbed data. The pipeline layer gathers the inputs (``build_project_scores``,
capability gaps, corpus queries) and hands them here.

THREE SIGNAL CLASSES:
  1. DUPLICATION       — >=2 repos score a concept above threshold -> shared-lib
                         extraction candidate (highest value, DRY debt).
  2. COMPLEMENTARITY   — repo A implements a concept that repo B carries as a
                         *gap* (planned/exploring capability) -> connector /
                         consume-via-lib candidate. Keyed off a real gap signal,
                         NOT a bare score asymmetry.
  3. LATENT-CONCEPT    — a corpus concept NO target repo uses but the corpus says
                         is relevant -> both plausibly should. Needs the corpus.

OUTPUT DISCIPLINE (load-bearing): this module NEVER files/promotes anything. It
returns a report (list of finding dicts). Generic infrastructure concepts are
filtered as noise. Findings dedup against already-filed FRs and previously
dismissed candidates so repeated runs don't re-surface the same idea.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


# Signal-class labels (stable strings used in findings + dedup keys).
DUPLICATION = "duplication"
COMPLEMENTARITY = "complementarity"
LATENT_CONCEPT = "latent-concept"


# Generic infrastructure concepts filtered as noise: a cross-repo "match" on
# these carries no integration value (every repo does HTTP/logging/config).
# Substring-matched against the normalized concept, so "http client",
# "structured logging", "yaml config" all filter. Overridable by the caller.
DEFAULT_INFRA_CONCEPTS: Set[str] = {
    "http",
    "https",
    "rest",
    "api endpoint",
    "logging",
    "logger",
    "config",
    "configuration",
    "serialization",
    "serializer",
    "deserialization",
    "json",
    "yaml",
    "toml",
    "cli",
    "command line",
    "argparse",
    "environment variable",
    "file io",
    "filesystem",
    "path handling",
    "error handling",
    "exception handling",
    "retry",
    "caching",
    "cache",
    "timeout",
    "async",
    "asyncio",
    "threading",
    "concurrency",
    "unit testing",
    "test fixture",
    "database connection",
    "connection pool",
    "authentication",
    "authorization",
}


def normalize_concept(concept: str) -> str:
    """Canonical form for matching/dedup: lowercased, collapsed whitespace,
    punctuation-to-space, trailing/leading stripped."""
    text = (concept or "").lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_infra_concept(concept: str, infra: Optional[Iterable[str]] = None) -> bool:
    """True if ``concept`` is generic infrastructure (noise).

    Matched by whole-word/phrase overlap against the normalized concept, so
    ``"HTTP client"`` -> infra but ``"HTTP/2 multiplexing research"`` is judged
    by whether any infra token appears as a word. We use word-boundary phrase
    containment to avoid ``"api"`` matching ``"rapid"``.
    """
    norm = normalize_concept(concept)
    if not norm:
        return True  # empty/garbage concept is not a real finding
    terms = DEFAULT_INFRA_CONCEPTS if infra is None else {normalize_concept(t) for t in infra}
    words = norm.split()
    word_set = set(words)
    padded = f" {norm} "  # word-boundary sentinel for phrase matching
    for term in terms:
        if not term:
            continue
        if " " in term:
            # phrase: require it to appear as whole contiguous words, not
            # inside a longer word (" command line " matches, but
            # "precommand line" must not) — Copilot correctness fix.
            if f" {term} " in padded:
                return True
        else:
            # single token: require an exact word match (not substring)
            if term in word_set:
                return True
    return False


@dataclass
class Finding:
    """One classified, provenance-carrying integration candidate."""

    signal_class: str
    concept: str
    repos: List[str]
    summary: str
    score: float = 0.0
    corpus_sources: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> str:
        """Stable key so periodic runs don't re-surface the same idea.

        Keyed on (signal-class, normalized concept, sorted repos) — NOT on the
        human-facing summary/title, which can drift run to run.
        """
        return "|".join(
            [
                self.signal_class,
                normalize_concept(self.concept),
                ",".join(sorted(self.repos)),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_class": self.signal_class,
            "concept": self.concept,
            "repos": list(self.repos),
            "summary": self.summary,
            "score": round(self.score, 4),
            "corpus_sources": list(self.corpus_sources),
            "dedup_key": self.dedup_key(),
            **({"detail": self.detail} if self.detail else {}),
        }


def _build_dedup_index(
    already_filed: Optional[Sequence[Mapping[str, Any]]],
    dismissed: Optional[Sequence[Mapping[str, Any]]],
) -> Set[str]:
    """Build the set of dedup keys to suppress.

    Each entry may supply an explicit ``dedup_key``, or the fields
    (``signal_class``/``concept``/``repos``) from which one is derived. Entries
    that carry only a concept (e.g. a filed FR that names a concept but not a
    signal class) suppress that concept across ALL classes — the idea has
    already been captured.
    """
    keys: Set[str] = set()
    concept_only: Set[str] = set()
    for entry in list(already_filed or []) + list(dismissed or []):
        if not isinstance(entry, Mapping):
            continue
        explicit = entry.get("dedup_key")
        if explicit:
            keys.add(str(explicit))
            continue
        concept = entry.get("concept")
        if not concept:
            continue
        norm = normalize_concept(str(concept))
        signal_class = entry.get("signal_class")
        repos = entry.get("repos")
        if signal_class and repos:
            f = Finding(
                signal_class=str(signal_class),
                concept=str(concept),
                repos=[str(r) for r in repos],
                summary="",
            )
            keys.add(f.dedup_key())
        else:
            concept_only.add(norm)
    # Sentinel prefix distinguishes concept-only suppression at match time.
    for norm in concept_only:
        keys.add(f"concept:{norm}")
    return keys


def _is_deduped(finding: Finding, dedup_keys: Set[str]) -> bool:
    if finding.dedup_key() in dedup_keys:
        return True
    if f"concept:{normalize_concept(finding.concept)}" in dedup_keys:
        return True
    return False


def classify_cross_repo_findings(
    footprints: Mapping[str, Mapping[str, float]],
    *,
    gaps: Optional[Mapping[str, Set[str]]] = None,
    latent: Optional[Sequence[Mapping[str, Any]]] = None,
    already_filed: Optional[Sequence[Mapping[str, Any]]] = None,
    dismissed: Optional[Sequence[Mapping[str, Any]]] = None,
    repos: Optional[Sequence[str]] = None,
    threshold: float = 0.4,
    infra: Optional[Iterable[str]] = None,
    max_findings: int = 50,
) -> List[Finding]:
    """Classify cross-repo integration candidates. Pure — no side effects.

    Args:
        footprints: ``{concept: {repo: score}}`` — per-repo concept coverage,
            as produced by ``build_project_scores`` restricted to target repos.
        gaps: ``{repo: {normalized_concept, ...}}`` — concepts each repo carries
            as a *gap* (planned/exploring capability). Drives COMPLEMENTARITY.
        latent: sequence of ``{concept, score, sources}`` corpus concepts that
            no target repo uses but the corpus judges relevant. Drives LATENT.
        already_filed: prior filed FRs (dedup source).
        dismissed: previously dismissed candidates (dedup source).
        repos: the target repos in scope (used to bound duplication/latent to
            the repos actually being compared).
        threshold: min score for a repo to "implement" a concept.
        infra: override the infra denylist.
        max_findings: cap on returned findings.

    Returns:
        List[Finding], highest-value first (duplication > complementarity >
        latent), infra-filtered and deduped. Nothing is filed.
    """
    gaps = gaps or {}
    repo_set = set(repos) if repos is not None else None
    dedup_keys = _build_dedup_index(already_filed, dismissed)

    findings: List[Finding] = []

    # ---- 1. DUPLICATION: >=2 target repos implement the same concept --------
    for concept, repo_scores in footprints.items():
        if is_infra_concept(concept, infra):
            continue
        implementers = sorted(
            r for r, s in repo_scores.items()
            if s >= threshold and (repo_set is None or r in repo_set)
        )
        if len(implementers) < 2:
            continue
        strength = min(repo_scores[r] for r in implementers)
        findings.append(
            Finding(
                signal_class=DUPLICATION,
                concept=concept,
                repos=implementers,
                score=strength,
                summary=(
                    f"{len(implementers)} repos implement '{concept}' "
                    f"({', '.join(implementers)}) -> shared-lib extraction candidate"
                ),
                detail={"per_repo_score": {r: round(repo_scores[r], 4) for r in implementers}},
            )
        )

    # ---- 2. COMPLEMENTARITY: repo A implements what repo B lists as a gap ---
    # Normalize each repo's gap set ONCE up front (the pipeline already emits
    # normalized concepts, but normalize defensively for hand-built callers) so
    # the inner concept×gap_repo loop is plain set membership, not a recomputed
    # comprehension per pair (Copilot perf).
    norm_gaps: Dict[str, Set[str]] = {
        repo: {normalize_concept(g) for g in gap_concepts}
        for repo, gap_concepts in gaps.items()
    }
    for concept, repo_scores in footprints.items():
        if is_infra_concept(concept, infra):
            continue
        norm = normalize_concept(concept)
        implementers = sorted(
            r for r, s in repo_scores.items()
            if s >= threshold and (repo_set is None or r in repo_set)
        )
        if not implementers:
            continue
        for gap_repo, gap_concepts in norm_gaps.items():
            if repo_set is not None and gap_repo not in repo_set:
                continue
            if gap_repo in implementers:
                continue  # already implements it, not a gap for this repo
            if norm not in gap_concepts:
                continue
            # Name the strongest implementer as the provider (highest footprint
            # score), not an arbitrary alphabetical first — a weaker repo would
            # misattribute the finding and its score (codex P2).
            provider = max(implementers, key=lambda r: repo_scores[r])
            findings.append(
                Finding(
                    signal_class=COMPLEMENTARITY,
                    concept=concept,
                    repos=sorted({provider, gap_repo}),
                    score=repo_scores[provider],
                    summary=(
                        f"'{provider}' implements '{concept}' which '{gap_repo}' "
                        f"carries as a gap -> connector / consume-via-lib candidate"
                    ),
                    detail={"provider": provider, "gap_repo": gap_repo},
                )
            )

    # ---- 3. LATENT CORPUS CONCEPT: corpus says relevant, no repo uses it ----
    for item in latent or []:
        concept = str(item.get("concept", ""))
        if not concept or is_infra_concept(concept, infra):
            continue
        norm = normalize_concept(concept)
        # Skip if any target repo already implements it above threshold.
        repo_scores = footprints.get(concept) or {}
        used = any(
            s >= threshold and (repo_set is None or r in repo_set)
            for r, s in repo_scores.items()
        )
        if not used:
            # also match by normalized name against footprint keys
            for fp_concept, fp_scores in footprints.items():
                if normalize_concept(fp_concept) != norm:
                    continue
                if any(
                    s >= threshold and (repo_set is None or r in repo_set)
                    for r, s in fp_scores.items()
                ):
                    used = True
                    break
        if used:
            continue
        target_repos = sorted(repo_set) if repo_set else []
        sources = [str(s) for s in (item.get("sources") or [])]
        findings.append(
            Finding(
                signal_class=LATENT_CONCEPT,
                concept=concept,
                repos=target_repos,
                score=float(item.get("score", 0.0)),
                corpus_sources=sources,
                summary=(
                    f"corpus concept '{concept}' used by no target repo but "
                    f"judged relevant -> both plausibly should adopt it"
                ),
            )
        )

    # ---- dedup + rank -------------------------------------------------------
    kept: List[Finding] = []
    seen_local: Set[str] = set()
    for f in findings:
        if _is_deduped(f, dedup_keys):
            continue
        key = f.dedup_key()
        if key in seen_local:
            continue  # collapse identical findings within a single run
        seen_local.add(key)
        kept.append(f)

    _class_rank = {DUPLICATION: 0, COMPLEMENTARITY: 1, LATENT_CONCEPT: 2}
    kept.sort(key=lambda f: (_class_rank.get(f.signal_class, 9), -f.score, f.concept))
    return kept[:max_findings]


def build_report(
    findings: Sequence[Finding],
    *,
    repos: Sequence[str],
    dedup_gap_note: str = "",
) -> Dict[str, Any]:
    """Assemble the structured report return value (still no side effects)."""
    by_class: Dict[str, int] = {DUPLICATION: 0, COMPLEMENTARITY: 0, LATENT_CONCEPT: 0}
    for f in findings:
        by_class[f.signal_class] = by_class.get(f.signal_class, 0) + 1
    report: Dict[str, Any] = {
        "repos": list(repos),
        "finding_count": len(findings),
        "by_class": by_class,
        "findings": [f.to_dict() for f in findings],
        "auto_filed": False,  # load-bearing invariant: report only, nothing filed
    }
    if dedup_gap_note:
        report["dedup_gap"] = dedup_gap_note
    return report
