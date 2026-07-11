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

# Imported defensively: khonliang-librarian-lib is a bare-name local-editable
# dependency (no PyPI package) across this ecosystem's repos — a deploy that
# hasn't yet run `pip install -e ../khonliang-librarian-lib` in the
# production venv must not crash the whole pipeline import on restart. The
# rest of this module (and its two pipeline.py call sites) already treats
# the catalog as best-effort/optional; a missing library degrades the same
# way an absent `db_path` does — catalog disabled, everything else works.
try:
    from librarian_lib import CONTRACT_VERSION, CatalogSkills, IndexRecord, SelfCatalog

    _LIBRARIAN_LIB_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only when the dep is absent
    CatalogSkills = IndexRecord = SelfCatalog = None  # type: ignore[assignment,misc]
    CONTRACT_VERSION = "0"  # never actually advertised: register_source is skipped when catalog is None
    _LIBRARIAN_LIB_AVAILABLE = False

logger = logging.getLogger(__name__)

if not _LIBRARIAN_LIB_AVAILABLE:
    logger.warning(
        "khonliang-librarian-lib is not installed — self-catalog disabled "
        "(no-op). Install it editable (`pip install -e "
        "../khonliang-librarian-lib`) to enable corpus self-cataloging."
    )

#: Default source/owner_agent when config carries neither. Also the
#: default `source` argument for `paper_index_record`/`idea_index_record`
#: when called directly (e.g. tests) without a live catalog to read
#: `.source` from.
DEFAULT_SOURCE = "researcher-primary"

#: Fallback project facet for entries with no per-project score above
#: threshold (or that aren't per-project-scored at all, e.g. ideas at
#: ingest time) — mirrors KnowledgeEntry's own generic `scope="research"`.
FALLBACK_PROJECT = "research"

#: Contract schema_version for records this module writes. Bump in lockstep
#: with any change to the facet/text shape below.
SCHEMA_VERSION = 1

#: Cap on embedded idea text. Ideas ingested via ``ingest_from_artifact`` /
#: ``stage_payload`` can carry up to the store's 20k-char fetch cap (blogs,
#: staged artifacts) — well past "index card" size. The FR's "abstract
#: tier, never full bodies" rule applies to papers explicitly, but the same
#: reasoning holds for any embeddable catalog text, so long idea bodies are
#: truncated rather than embedded whole.
IDEA_TEXT_CAP = 2000


def build_self_catalog(
    config: dict[str, Any], owner_agent: Optional[str] = None
) -> Optional[SelfCatalog]:
    """Build the researcher's SelfCatalog sidecar next to its main db.

    Returns None (no-op) when ``config["db_path"]`` is absent — this is
    deliberate: guessing a relative default here risks writing into
    whatever the current process's cwd happens to be (the exact mistake
    that nearly seeded rows into the live production db). Callers that get
    None back should skip cataloging, not fabricate a path. Also returns
    None (already logged at import time) when ``khonliang-librarian-lib``
    itself isn't installed.
    """
    if not _LIBRARIAN_LIB_AVAILABLE:
        return None
    db_path = config.get("db_path")
    if not db_path:
        logger.warning(
            "self_catalog: no db_path in config — catalog disabled (no-op)"
        )
        return None
    # Named from the main db's own FULL basename — `.name`, not `.stem`
    # (not a fixed "self_catalog.db" either): two researcher deployments
    # that each keep a separate main db in the SAME directory (e.g.
    # domain-scoped instances sharing a data/ dir) would otherwise
    # silently share one sidecar file — their index cards, stats, and
    # mark-stale operations would mix even though the main KnowledgeStores
    # stay isolated. `.stem` alone isn't enough: two main dbs that differ
    # only by extension (`researcher.db` / `researcher.sqlite3`) share the
    # same stem and would still collide (codex P2) — `.name` includes the
    # extension, so only two deployments pointed at the literal same
    # db_path can ever share a sidecar.
    main_db = Path(db_path)
    catalog_db_path = main_db.parent / f"{main_db.name}.self_catalog.db"
    # source == owner_agent (one string, reused for both): SelfCatalog's
    # `source` is the label stamped on every row, and the librarian's
    # registry keys sources by a `source_id` that must map 1:1 to a bus
    # agent_id for federated calls to route to the right process. Reusing
    # the actual owner_agent value as `source` (instead of a fleet-wide
    # constant like "researcher") keeps multiple researcher instances
    # (e.g. domain-scoped deployments, each with its own bus agent_id and
    # its own self_catalog.db) from colliding on one registry entry — a
    # static "researcher" source_id would mean the second instance's
    # register_source call silently overwrites the first's, making the
    # first instance's rows permanently unroutable even though its sqlite
    # file is untouched.
    resolved_owner = owner_agent or config.get("bus_agent_id") or DEFAULT_SOURCE
    return SelfCatalog(
        db_path=str(catalog_db_path),
        source=resolved_owner,
        owner_agent=resolved_owner,
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
    source: str = DEFAULT_SOURCE,
) -> Optional[IndexRecord]:
    """Build the post-distill index card for a paper (or None if nothing to catalog).

    Called at the `distill()` completion path once a paper has a summary +
    per-project assessments. Returns None when `result.summary` is empty
    (a failed/skipped distill has nothing embeddable to catalog). `source`
    must match the owning `SelfCatalog.source` exactly (`upsert()` rejects
    a mismatch) — callers with a live catalog should pass `catalog.source`,
    not rely on the default.
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
        source=source,
        record_id=entry.id,
        schema_version=SCHEMA_VERSION,
        kind="paper",
        updated_at=time.time(),
        facets={
            # "status" (not e.g. "distill_status") is the CORE_FACETS key
            # librarian_lib.SelfCatalog.stats()'s by_status breakdown and
            # any status-filtered catalog_query call actually read
            # (json_extract(facets_json, '$.status')) — a different key
            # name here would silently bucket every row under "unset" and
            # never match a status filter (codex P1).
            "status": "distilled",
            "primary_project": primary_project,
            "relevance_scores": scores,
            "ingest_date": entry.metadata.get("fetched_at") or entry.created_at,
            "source_url": entry.metadata.get("url", ""),
        },
        text=text,
        # catalog_fetch is an exact record_id lookup (researcher.agent's own
        # new bus skill, backed by pipeline.knowledge.get) — NOT
        # paper_context/paper_digest, which are fuzzy multi-result search
        # surfaces that can resolve to the wrong entry on a duplicate or
        # similar title. `ref` is meant to retrieve THIS exact record.
        ref={"skill": "catalog_fetch", "args": {"record_id": entry.id}},
    )


def idea_index_record(entry: Any, source: str = DEFAULT_SOURCE) -> Optional[IndexRecord]:
    """Build the ingest-time index card for a free-form idea/blog entry.

    Ideas aren't scored per-project the way papers are (`ingest_idea` has
    no assessor pass), so there's no per-project score to pick a primary
    project from — every idea catalogs under `FALLBACK_PROJECT`
    ("research"), the same generic scope `KnowledgeEntry` already gives
    un-scoped corpus entries. `text` is the idea body itself (not a
    summary) — most ideas are short informal notes, but `ingest_idea` is
    also the entry point for staged artifacts / blogs up to the store's
    20k-char fetch cap, so the body is truncated to `IDEA_TEXT_CAP` rather
    than embedded whole (same "index card, not full payload" reasoning the
    FR applies to papers). `source` must match the owning
    `SelfCatalog.source` exactly, same caveat as `paper_index_record`.
    """
    if not entry.content or not entry.content.strip():
        return None
    body = entry.content.strip()
    truncated = len(body) > IDEA_TEXT_CAP
    if truncated:
        body = body[:IDEA_TEXT_CAP]
    return IndexRecord(
        project=FALLBACK_PROJECT,
        source=source,
        record_id=entry.id,
        schema_version=SCHEMA_VERSION,
        kind="idea",
        updated_at=time.time(),
        facets={
            # See paper_index_record — "status" is the contract key, not a
            # bespoke "distill_status".
            "status": "ingested",
            "source_type": entry.metadata.get("source_type", "freeform"),
            "text_truncated": truncated,
        },
        text=f"{entry.title}\n\n{body}",
        ref={"skill": "catalog_fetch", "args": {"record_id": entry.id}},
    )


def build_catalog_skills(catalog: Optional[SelfCatalog]) -> Optional[CatalogSkills]:
    """Wrap a SelfCatalog for bus-skill registration, or None when catalog is None."""
    if catalog is None:
        return None
    return CatalogSkills(catalog)
