from __future__ import annotations

from types import SimpleNamespace

import pytest

from researcher.self_catalog import (
    build_catalog_skills,
    build_self_catalog,
    idea_index_record,
    paper_index_record,
    pick_primary_project,
)


def _entry(**overrides):
    defaults = dict(
        id="paper123",
        title="A Great Paper",
        content="full raw body text — should never be embedded",
        metadata={"url": "https://arxiv.org/abs/1234", "fetched_at": 1700000000.0},
        created_at=1700000000.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _result(**overrides):
    defaults = dict(
        entry_id="paper123",
        title="A Great Paper",
        summary={
            "abstract": "This paper does a thing.",
            "keywords": ["foo", "bar"],
        },
        triples=[],
        assessments={
            "khonliang": {"score": 0.8},
            "genealogy": {"score": 0.1},
        },
        success=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# build_self_catalog: never guesses a relative default db_path
# ---------------------------------------------------------------------------


def test_build_self_catalog_returns_none_without_db_path():
    assert build_self_catalog({}) is None
    assert build_self_catalog({"db_path": ""}) is None
    assert build_self_catalog({"db_path": None}) is None


def test_build_self_catalog_places_sidecar_next_to_main_db(tmp_path):
    db_path = tmp_path / "researcher.db"
    catalog = build_self_catalog({"db_path": str(db_path)})
    assert catalog is not None
    assert catalog.db_path == str(tmp_path / "researcher.self_catalog.db")
    # source == owner_agent (both default to DEFAULT_SOURCE) so a registry
    # source_id maps 1:1 to the bus agent_id that owns this catalog file.
    assert catalog.source == "researcher-primary"
    assert catalog.owner_agent == "researcher-primary"


def test_build_self_catalog_owner_agent_override(tmp_path):
    db_path = tmp_path / "researcher.db"
    catalog = build_self_catalog(
        {"db_path": str(db_path)}, owner_agent="researcher-secondary"
    )
    assert catalog.owner_agent == "researcher-secondary"
    assert catalog.source == "researcher-secondary"


def test_build_self_catalog_sidecars_dont_collide_when_main_dbs_share_a_dir(tmp_path):
    # Two deployments keeping separate main dbs in the same directory (e.g.
    # domain-scoped researcher instances sharing a data/ dir) must get
    # separate sidecar files — a fixed "self_catalog.db" name would collapse
    # both instances' index cards into one file even though their main
    # KnowledgeStores stay isolated.
    cat_a = build_self_catalog({"db_path": str(tmp_path / "a.db")})
    cat_b = build_self_catalog({"db_path": str(tmp_path / "b.db")})
    assert cat_a.db_path != cat_b.db_path
    assert cat_a.db_path == str(tmp_path / "a.self_catalog.db")
    assert cat_b.db_path == str(tmp_path / "b.self_catalog.db")


# ---------------------------------------------------------------------------
# pick_primary_project
# ---------------------------------------------------------------------------


def test_pick_primary_project_picks_highest_scorer_above_threshold():
    name, scores = pick_primary_project(
        {"khonliang": {"score": 0.8}, "genealogy": {"score": 0.1}}, threshold=0.3
    )
    assert name == "khonliang"
    assert scores == {"khonliang": 0.8, "genealogy": 0.1}


def test_pick_primary_project_none_when_nothing_clears_threshold():
    name, scores = pick_primary_project(
        {"khonliang": {"score": 0.2}, "genealogy": {"score": 0.1}}, threshold=0.3
    )
    assert name is None
    assert scores == {"khonliang": 0.2, "genealogy": 0.1}


def test_pick_primary_project_tolerates_malformed_entries():
    name, scores = pick_primary_project(
        {"khonliang": {"score": 0.8}, "bad": None, "worse": {"score": "not-a-number"}},
        threshold=0.3,
    )
    assert name == "khonliang"
    assert scores == {"khonliang": 0.8}


def test_pick_primary_project_empty_assessments():
    name, scores = pick_primary_project({}, threshold=0.3)
    assert name is None
    assert scores == {}


# ---------------------------------------------------------------------------
# paper_index_record
# ---------------------------------------------------------------------------


def test_paper_index_record_none_when_no_summary():
    assert paper_index_record(_entry(), _result(summary=None), 0.3) is None


def test_paper_index_record_shape_and_project_facet():
    record = paper_index_record(_entry(), _result(), 0.3)
    assert record is not None
    assert record.project == "khonliang"
    assert record.source == "researcher-primary"
    assert record.record_id == "paper123"
    assert record.kind == "paper"
    assert record.schema_version == 1
    assert "A Great Paper" in record.text
    assert "This paper does a thing." in record.text
    # Never embeds the raw full-text body.
    assert "full raw body text" not in record.text
    assert record.facets["primary_project"] == "khonliang"
    assert record.facets["relevance_scores"] == {"khonliang": 0.8, "genealogy": 0.1}
    assert record.ref == {"skill": "catalog_fetch", "args": {"record_id": "paper123"}}


def test_paper_index_record_falls_back_to_research_project_below_threshold():
    result = _result(assessments={"khonliang": {"score": 0.1}})
    record = paper_index_record(_entry(), result, 0.3)
    assert record.project == "research"
    assert record.facets["primary_project"] is None


def test_paper_index_record_falls_back_with_no_assessments_at_all():
    record = paper_index_record(_entry(), _result(assessments={}), 0.3)
    assert record.project == "research"


# ---------------------------------------------------------------------------
# idea_index_record
# ---------------------------------------------------------------------------


def test_idea_index_record_shape():
    entry = _entry(
        id="idea456",
        title="An Idea",
        content="some free-form idea text",
        metadata={"source_type": "freeform"},
    )
    record = idea_index_record(entry)
    assert record is not None
    assert record.project == "research"
    assert record.kind == "idea"
    assert record.record_id == "idea456"
    assert "An Idea" in record.text
    assert "some free-form idea text" in record.text


def test_idea_index_record_none_for_empty_content():
    entry = _entry(content="   ")
    assert idea_index_record(entry) is None


def test_idea_index_record_marks_short_body_not_truncated():
    entry = _entry(content="short body")
    record = idea_index_record(entry)
    assert record.facets["text_truncated"] is False


def test_idea_index_record_truncates_long_body():
    from researcher.self_catalog import IDEA_TEXT_CAP

    long_body = "x" * (IDEA_TEXT_CAP + 5000)
    entry = _entry(content=long_body)
    record = idea_index_record(entry)
    assert record.facets["text_truncated"] is True
    # Body portion of the text is capped; the exact overhead from the
    # title prefix doesn't matter, just that the full 20k-char blog body
    # never lands in the catalog whole.
    assert len(record.text) < len(long_body)


# ---------------------------------------------------------------------------
# Missing khonliang-librarian-lib must disable the catalog, not crash import
# (deploy hazard: it's a bare-name local-editable dep, not on PyPI, so a
# production venv that hasn't run the sibling `pip install -e` yet must not
# take the whole pipeline down on restart).
# ---------------------------------------------------------------------------


def test_build_self_catalog_disabled_when_librarian_lib_unavailable(monkeypatch, tmp_path):
    import researcher.self_catalog as sc

    monkeypatch.setattr(sc, "_LIBRARIAN_LIB_AVAILABLE", False)
    assert sc.build_self_catalog({"db_path": str(tmp_path / "researcher.db")}) is None


# ---------------------------------------------------------------------------
# build_catalog_skills
# ---------------------------------------------------------------------------


def test_build_catalog_skills_none_passthrough():
    assert build_catalog_skills(None) is None


def test_build_catalog_skills_wraps_catalog(tmp_path):
    catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})
    skills = build_catalog_skills(catalog)
    assert skills is not None
    assert skills.catalog is catalog


# ---------------------------------------------------------------------------
# End-to-end: upsert then query round-trip through the real SelfCatalog
# ---------------------------------------------------------------------------


def test_paper_record_round_trips_through_catalog_upsert_and_query(tmp_path):
    catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})
    record = paper_index_record(_entry(), _result(), 0.3)
    catalog.upsert(record)

    result = catalog.query("khonliang")
    assert result["count"] == 1
    row = result["rows"][0]
    assert row["record_id"] == "paper123"
    assert row["kind"] == "paper"
    assert row["facets"]["primary_project"] == "khonliang"

    # Cross-project isolation: querying a different project sees nothing.
    assert catalog.query("genealogy")["count"] == 0


def test_idea_record_round_trips_through_catalog_upsert_and_query(tmp_path):
    catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})
    entry = _entry(id="idea789", title="An Idea", content="idea body")
    record = idea_index_record(entry)
    catalog.upsert(record)

    result = catalog.query("research")
    assert result["count"] == 1
    assert result["rows"][0]["kind"] == "idea"


# ---------------------------------------------------------------------------
# Pipeline wiring: distill()/_store_distillation and ingest_idea() call the
# catalog upsert helpers at their completion paths (fr_researcher_bbe95f12).
# Uses ResearchPipeline.__new__ + hand-set attributes, matching the existing
# style in tests/test_pipeline_second_pass.py.
# ---------------------------------------------------------------------------


class _FakeCatalog:
    def __init__(self, source="researcher-primary"):
        self.upserted = []
        self.source = source

    def upsert(self, record):
        self.upserted.append(record)
        return {"project": record.project, "record_id": record.record_id}


def test_catalog_upsert_paper_helper_calls_catalog(tmp_path):
    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.config = {"relevance_threshold": 0.3}
    pipe.catalog = _FakeCatalog()

    entry = _entry()
    result = _result()
    pipe._catalog_upsert_paper(entry, result)

    assert len(pipe.catalog.upserted) == 1
    assert pipe.catalog.upserted[0].project == "khonliang"


def test_catalog_upsert_paper_helper_is_noop_when_catalog_disabled():
    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.config = {}
    pipe.catalog = None
    # Must not raise even though there's no catalog to upsert into.
    pipe._catalog_upsert_paper(_entry(), _result())


def test_catalog_upsert_paper_helper_never_raises_on_missing_attribute():
    """Pipeline objects built via __new__ in other test modules never set
    ``catalog`` at all — the helper must degrade to a no-op, not AttributeError."""
    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.config = {}
    pipe._catalog_upsert_paper(_entry(), _result())  # no AttributeError


def test_catalog_upsert_paper_helper_swallows_catalog_exceptions():
    from researcher.pipeline import ResearchPipeline

    class _BoomCatalog:
        source = "researcher-primary"

        def upsert(self, record):
            raise RuntimeError("boom")

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.config = {}
    pipe.catalog = _BoomCatalog()
    # Must not propagate — a catalog failure can't flip an already-distilled
    # paper to a failure state.
    pipe._catalog_upsert_paper(_entry(), _result())


def test_catalog_upsert_idea_helper_calls_catalog():
    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.catalog = _FakeCatalog()
    entry = _entry(id="idea1", title="Idea", content="body")
    pipe._catalog_upsert_idea(entry)

    assert len(pipe.catalog.upserted) == 1
    assert pipe.catalog.upserted[0].kind == "idea"


@pytest.mark.asyncio
async def test_ingest_idea_end_to_end_upserts_to_catalog():
    from types import SimpleNamespace as NS

    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    captured = {}
    pipe.knowledge = NS(add=lambda e: captured.__setitem__("entry", e))
    pipe.digest = NS(record=lambda **k: None)
    pipe.idea_parser = NS(
        handle=lambda text: _as_async(
            {"success": True, "title": "An Idea", "claims": [], "search_queries": []}
        )
    )
    pipe.extractor = NS(handle=lambda text: _as_async({"success": False}))
    pipe.triples = NS(add=lambda **k: None)
    pipe.config = {}
    pipe.catalog = _FakeCatalog()

    await pipe.ingest_idea("some free-form idea text")

    assert len(pipe.catalog.upserted) == 1
    assert pipe.catalog.upserted[0].kind == "idea"
    assert pipe.catalog.upserted[0].project == "research"


async def _as_async(value):
    return value


# ---------------------------------------------------------------------------
# backfill_self_catalog: pre-existing corpus entries that predate
# self-cataloging (codex P1).
# ---------------------------------------------------------------------------


def _knowledge_stub(entries: dict):
    from khonliang.knowledge.store import EntryStatus, Tier

    class _Knowledge:
        def get(self, eid):
            return entries.get(eid)

        def get_by_tier(self, tier):
            return [e for e in entries.values() if getattr(e, "tier", Tier.IMPORTED) == tier]

    return _Knowledge()


def test_backfill_self_catalog_noop_without_catalog():
    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.catalog = None
    pipe.config = {}
    pipe.knowledge = _knowledge_stub({})
    assert pipe.backfill_self_catalog() == {"papers": 0, "ideas": 0, "skipped": 0, "errors": 0}


def test_backfill_self_catalog_publishes_pre_existing_papers_and_ideas(tmp_path):
    import json as _json

    from khonliang.knowledge.store import EntryStatus, Tier
    from researcher.pipeline import ResearchPipeline

    paper = SimpleNamespace(
        id="p1", tier=Tier.IMPORTED, tags=["paper"], status=EntryStatus.DISTILLED,
        title="A Paper", metadata={"url": "https://x"}, created_at=1700000000.0,
    )
    summary = SimpleNamespace(
        tier=Tier.DERIVED,
        content=_json.dumps({"abstract": "abs text", "keywords": []}),
        metadata={"assessments": {"khonliang": {"score": 0.9}}},
    )
    idea = SimpleNamespace(
        id="i1", tier=Tier.IMPORTED, tags=["idea"], status=EntryStatus.INGESTED,
        title="An Idea", content="idea body", metadata={"source_type": "freeform"},
    )
    entries = {"p1": paper, "p1_summary": summary, "i1": idea}

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.config = {"relevance_threshold": 0.3}
    pipe.knowledge = _knowledge_stub(entries)
    pipe.catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})

    stats = pipe.backfill_self_catalog()
    assert stats == {"papers": 1, "ideas": 1, "skipped": 0, "errors": 0}

    assert pipe.catalog.query("khonliang")["count"] == 1
    assert pipe.catalog.query("research")["count"] == 1

    # Re-running is idempotent: both entries are now skipped.
    stats2 = pipe.backfill_self_catalog()
    assert stats2 == {"papers": 0, "ideas": 0, "skipped": 2, "errors": 0}


def test_backfill_self_catalog_skips_undistilled_papers(tmp_path):
    from khonliang.knowledge.store import EntryStatus, Tier
    from researcher.pipeline import ResearchPipeline

    paper = SimpleNamespace(
        id="p2", tier=Tier.IMPORTED, tags=["paper"], status=EntryStatus.INGESTED,
        title="Not yet distilled", metadata={},
    )
    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.config = {}
    pipe.knowledge = _knowledge_stub({"p2": paper})
    pipe.catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})

    stats = pipe.backfill_self_catalog()
    assert stats == {"papers": 0, "ideas": 0, "skipped": 0, "errors": 0}
    assert pipe.catalog.stats()["total"] == 0


# ---------------------------------------------------------------------------
# strike() drops the catalog card too (codex P1).
# ---------------------------------------------------------------------------


def test_strike_removes_catalog_card(tmp_path):
    from researcher.pipeline import ResearchPipeline

    entry = SimpleNamespace(title="Paper", metadata={"url": "https://canon"})

    class _Knowledge:
        def get(self, eid):
            return None if str(eid).endswith("_summary") else entry

        def remove(self, *a, **k):
            pass

    class _Triples:
        db_path = str(tmp_path / "absent.db")

        def remove_source(self, *a, **k):
            return False

    catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})
    from librarian_lib import IndexRecord

    catalog.upsert(
        IndexRecord(
            project="khonliang", source=catalog.source, record_id="e1",
            schema_version=1, kind="paper", text="t",
        )
    )
    assert catalog.query("khonliang")["count"] == 1

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.knowledge = _Knowledge()
    pipe.triples = _Triples()
    pipe.digest = SimpleNamespace(record=lambda **k: None)
    pipe._url_index = {}
    pipe.catalog = catalog

    pipe.strike("e1")

    assert catalog.query("khonliang")["count"] == 0


def test_catalog_delete_noop_when_record_absent(tmp_path):
    """Striking an entry that was never cataloged (or already removed) is a no-op."""
    from researcher.pipeline import ResearchPipeline

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})
    pipe._catalog_delete("never-cataloged")  # must not raise
