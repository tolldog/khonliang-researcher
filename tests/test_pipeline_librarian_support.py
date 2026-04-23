from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

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


def test_list_evidence_sources_self_heals_stale_owned_locally_metadata(tmp_path):
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
    assert rows[0]["owned_locally"] is True


@pytest.mark.asyncio
async def test_consume_research_request_searches_and_records(monkeypatch, tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))

    async def fake_search(query: str, max_results: int = 8):
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
