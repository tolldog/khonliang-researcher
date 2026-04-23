from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from khonliang.knowledge.store import EntryStatus, KnowledgeEntry, Tier

from researcher.pipeline import create_pipeline


def _make_config(tmp_path):
    config = {
        "db_path": str(tmp_path / "researcher.db"),
        "models": {},
        "projects": {},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_register_evidence_source_infers_owned_locally(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    repo_path = tmp_path / "khonliang-researcher"
    repo_path.mkdir()

    pipeline.register_evidence_source(
        "researcher",
        str(repo_path),
        description="Research app",
    )

    rows = pipeline.list_evidence_sources()
    assert len(rows) == 1
    assert rows[0]["project"] == "researcher"
    assert rows[0]["owned_locally"] is True


def test_register_evidence_source_does_not_mark_remote_urls_owned(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))

    pipeline.register_evidence_source(
        "external",
        "https://github.com/example/project",
    )

    rows = pipeline.list_evidence_sources()
    assert rows[0]["owned_locally"] is False


def test_list_evidence_sources_filters_owned_locally(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    local_repo = tmp_path / "khonliang-local"
    local_repo.mkdir()
    pipeline.register_evidence_source("local", str(local_repo))
    pipeline.register_evidence_source(
        "external",
        "https://github.com/example/project",
        owned_locally=False,
    )

    owned = pipeline.list_evidence_sources(True)
    external = pipeline.list_evidence_sources(False)

    assert [row["project"] for row in owned] == ["local"]
    assert [row["project"] for row in external] == ["external"]


def test_list_evidence_sources_preserves_explicit_owned_locally_false(tmp_path):
    """Regression: an explicit ``owned_locally=False`` for a local checkout
    must persist through ``list_evidence_sources``. Previously the listing
    logic OR-ed the stored flag with a path-existence check, which silently
    flipped the value back to ``True`` for any existing path — making the
    explicit API/CLI override impossible to persist. Callers that want the
    legacy inference can combine ``owned_locally or path_exists`` themselves.
    """
    pipeline = create_pipeline(_make_config(tmp_path))
    repo_path = tmp_path / "ollama-khonliang"
    repo_path.mkdir()
    pipeline.register_evidence_source(
        "khonliang",
        str(repo_path),
        owned_locally=False,
    )

    rows = pipeline.list_evidence_sources()

    assert rows[0]["project"] == "khonliang"
    # Stored value is honoured verbatim — explicit False is not clobbered
    # even though the path exists on disk.
    assert rows[0]["owned_locally"] is False
    # path_exists is exposed separately so callers can combine them if they
    # want the union (``owned_locally or path_exists``).
    assert rows[0]["path_exists"] is True


@pytest.mark.asyncio
async def test_consume_research_request_searches_and_records(monkeypatch, tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))

    async def fake_search(query: str, engines=None, max_results: int = 8, **kwargs):
        assert "code review" in query
        return [
            SimpleNamespace(
                title="Paper A",
                url="https://example.com/a",
                source="arxiv",
            ),
            SimpleNamespace(
                title="Paper B",
                url="https://example.com/b",
                source="semantic_scholar",
            ),
        ][:max_results]

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    result = await pipeline.consume_research_request(
        topic="code review",
        audience="developer-researcher",
        branch="llm-code-review",
        rationale="gap report",
    )

    assert result["topic"] == "code review"
    assert result["audience"] == "developer-researcher"
    assert len(result["papers"]) == 2
    assert result["papers"][0]["title"] == "Paper A"


@pytest.mark.asyncio
async def test_consume_research_request_threads_suggested_sources(monkeypatch, tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))

    captured: dict = {}

    async def fake_search(query, engines=None, max_results=20, **kwargs):
        captured["query"] = query
        captured["engines"] = engines
        captured["max_results"] = max_results
        return []

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    await pipeline.consume_research_request(
        topic="code review",
        suggested_sources=["arxiv", "semantic_scholar"],
        max_results=5,
    )

    assert captured["engines"] == ["arxiv", "semantic_scholar"]
    assert captured["max_results"] == 5


@pytest.mark.asyncio
async def test_consume_research_request_omits_engines_when_no_sources(monkeypatch, tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))

    captured: dict = {}

    async def fake_search(query, engines=None, max_results=20, **kwargs):
        captured["engines"] = engines
        return []

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    # Missing param → engines defaults to None (search all engines)
    await pipeline.consume_research_request(topic="t1")
    assert captured["engines"] is None

    # Empty list → same (still None, not [] — avoids search_papers returning empty)
    await pipeline.consume_research_request(topic="t2", suggested_sources=[])
    assert captured["engines"] is None

    # List of empty/whitespace strings → also None
    await pipeline.consume_research_request(topic="t3", suggested_sources=["", "  "])
    assert captured["engines"] is None


def test_get_ingest_snapshot_excludes_non_url_entries(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))

    # URL-backed paper — included
    pipeline.knowledge.add(
        KnowledgeEntry(
            id="paper-ok",
            tier=Tier.IMPORTED,
            title="Paper with URL",
            content="body",
            source="https://example.com/a",
            scope="research",
            tags=["paper"],
            status=EntryStatus.INGESTED,
            metadata={"url": "https://example.com/a"},
        )
    )
    # Idea — excluded (no url, tag idea)
    pipeline.knowledge.add(
        KnowledgeEntry(
            id="idea-xx",
            tier=Tier.IMPORTED,
            title="An idea",
            content="raw",
            source="idea",
            scope="research",
            tags=["idea"],
            status=EntryStatus.INGESTED,
            metadata={"claims": []},
        )
    )
    # Paper-tagged but no URL — excluded (bad data guard)
    pipeline.knowledge.add(
        KnowledgeEntry(
            id="paper-nourl",
            tier=Tier.IMPORTED,
            title="Broken paper",
            content="body",
            source="?",
            scope="research",
            tags=["paper"],
            status=EntryStatus.INGESTED,
            metadata={},
        )
    )

    rows = pipeline.get_ingest_snapshot()
    assert [row["entry_id"] for row in rows] == ["paper-ok"]
    assert rows[0]["url"] == "https://example.com/a"
