"""Researcher's SelfCatalog adoption — corpus entries self-catalog for librarian.

fr_researcher_bbe95f12: researcher becomes a knowledge owner under the
khonliang-librarian-lib contract (fr_khonliang-librarian-lib_500f69b9).
Corpus entries (distilled papers, ingested ideas) publish an index card —
title + abstract/summary tier text, project facet, ingest/distill status,
a `ref` pointer back into researcher for bounded expansion — into a sqlite
sidecar (`SelfCatalog`) that the librarian agent federates reads over.

Design notes:
- The catalog stores INDEX CARDS, never full payloads. `text` is the
  embeddable abstract tier (title + distilled abstract), never the raw
  paper body — the corpus KnowledgeStore keeps full-text duty.
- `project` is mandatory on every record. Researcher already scores paper
  applicability per-project via `AssessorRole` (see `pipeline.distill`);
  the highest-scoring project above `relevance_threshold` becomes the
  record's `project` facet. When no project clears the threshold (or for
  entries that aren't per-project-scored, like ideas at ingest time), we
  fall back to `"research"` — the same generic scope value the knowledge
  store already uses for un-scoped corpus entries.
- Concept-graph triples are NOT cataloged here — those flow through the
  concept-authority intake, a distinct FR (per the FR description).
  `backed_by` link inversions (papers referenced by FRs) are also out of
  scope here — that resolution happens on the librarian side via mentions.
- This module never guesses a relative default db_path: `build_self_catalog`
  requires `config["db_path"]` to already be the pipeline's resolved
  absolute path (which `create_pipeline` always populates) and returns
  None (a no-op catalog) if it's absent, rather than defaulting to a bare
  relative path that could collide with an unrelated cwd's data (see
  CLAUDE.md's near-miss note on `data/researcher.db`).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from librarian_lib import CatalogSkills, IndexRecord, SelfCatalog

logger = logging.getLogger(__name__)

#: Source id stamped on every record this catalog owns.
CATALOG_SOURCE = "researcher"

#: Fallback project facet for entries with no per-project score above
#: threshold (or that aren't per-project-scored at all, e.g. ideas at
#: ingest time) — mirrors KnowledgeEntry's own generic `scope="research"`.
FALLBACK_PROJECT = "research"

#: Contract schema_version for records this module writes. Bump in lockstep
#: with any change to the facet/text shape below.
SCHEMA_VERSION = 1


def build_self_catalog(
    config: dict[str, Any], owner_agent: Optional[str] = None
) -> Optional[SelfCatalog]:
    """Build the researcher's SelfCatalog sidecar next to its main db.

    Returns None (no-op) when ``config["db_path"]`` is absent — this is
    deliberate: guessing a relative default here risks writing into
    whatever the current process's cwd happens to be (the exact mistake
    that nearly seeded rows into the live production db). Callers that get
    None back should skip cataloging, not fabricate a path.
    """
    db_path = config.get("db_path")
    if not db_path:
        logger.warning(
            "self_catalog: no db_path in config — catalog disabled (no-op)"
        )
        return None
    catalog_db_path = Path(db_path).parent / "self_catalog.db"
    return SelfCatalog(
        db_path=str(catalog_db_path),
        source=CATALOG_SOURCE,
        owner_agent=owner_agent or config.get("bus_agent_id") or "researcher-primary",
    )


def pick_primary_project(
    assessments: dict[str, Any], threshold: float
) -> tuple[Optional[str], dict[str, float]]:
    """Pick the best-scoring project from `distill()`'s per-project assessments.

    Returns ``(project_name_or_None, {project: score, ...})``. The second
    element carries every project's score (not just the winner) so callers
    can stash the full spread as an extension facet — multi-project
    relevance is kept, per the FR, even though only the primary project
    becomes the record's mandatory `project` field. Returns
    ``(None, scores)`` when no project's score clears `threshold` (a bare
    `assessments` dict with malformed entries is tolerated: those entries
    are skipped rather than raising).
    """
    scores: dict[str, float] = {}
    for name, assessment in (assessments or {}).items():
        if not isinstance(assessment, dict):
            continue
        try:
            score = float(assessment.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        scores[name] = score
    best_name: Optional[str] = None
    best_score = threshold
    for name, score in scores.items():
        if score >= best_score:
            best_name = name
            best_score = score
    return best_name, scores


def paper_index_record(
    entry: Any,
    result: Any,
    relevance_threshold: float,
) -> Optional[IndexRecord]:
    """Build the post-distill index card for a paper (or None if nothing to catalog).

    Called at the `distill()` completion path once a paper has a summary +
    per-project assessments. Returns None when `result.summary` is empty
    (a failed/skipped distill has nothing embeddable to catalog).
    """
    summary = getattr(result, "summary", None)
    if not summary:
        return None
    primary_project, scores = pick_primary_project(
        getattr(result, "assessments", {}) or {}, relevance_threshold
    )
    project = primary_project or FALLBACK_PROJECT

    abstract = summary.get("abstract", "") if isinstance(summary, dict) else ""
    keywords = summary.get("keywords", []) if isinstance(summary, dict) else []
    text_parts = [entry.title, abstract]
    if keywords:
        text_parts.append("Keywords: " + ", ".join(str(k) for k in keywords))
    text = "\n\n".join(p for p in text_parts if p)

    return IndexRecord(
        project=project,
        source=CATALOG_SOURCE,
        record_id=entry.id,
        schema_version=SCHEMA_VERSION,
        kind="paper",
        updated_at=time.time(),
        facets={
            "distill_status": "distilled",
            "primary_project": primary_project,
            "relevance_scores": scores,
            "ingest_date": entry.metadata.get("fetched_at") or entry.created_at,
            "source_url": entry.metadata.get("url", ""),
        },
        text=text,
        ref={"skill": "paper_context", "args": {"query": entry.title}},
    )


def idea_index_record(entry: Any) -> Optional[IndexRecord]:
    """Build the ingest-time index card for a free-form idea/blog entry.

    Ideas aren't scored per-project the way papers are (`ingest_idea` has
    no assessor pass), so there's no per-project score to pick a primary
    project from — every idea catalogs under `FALLBACK_PROJECT`
    ("research"), the same generic scope `KnowledgeEntry` already gives
    un-scoped corpus entries. `text` is the idea body itself (not a
    summary) — ideas are short informal notes, not full paper bodies, so
    embedding the whole thing stays within the "abstract tier, not raw
    corpus text" rule the FR sets for papers.
    """
    if not entry.content or not entry.content.strip():
        return None
    return IndexRecord(
        project=FALLBACK_PROJECT,
        source=CATALOG_SOURCE,
        record_id=entry.id,
        schema_version=SCHEMA_VERSION,
        kind="idea",
        updated_at=time.time(),
        facets={
            "distill_status": "ingested",
            "source_type": entry.metadata.get("source_type", "freeform"),
        },
        text=f"{entry.title}\n\n{entry.content}",
        ref={"skill": "paper_context", "args": {"query": entry.title}},
    )


def build_catalog_skills(catalog: Optional[SelfCatalog]) -> Optional[CatalogSkills]:
    """Wrap a SelfCatalog for bus-skill registration, or None when catalog is None."""
    if catalog is None:
        return None
    return CatalogSkills(catalog)
