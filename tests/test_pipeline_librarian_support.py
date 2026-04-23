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


@pytest.mark.asyncio
async def test_consume_research_request_strips_engine_whitespace(monkeypatch, tmp_path):
    """R7 regression: engine names with surrounding whitespace must be
    *stripped* before hitting ``search_papers``. Previously the filter
    check used ``.strip()`` but appended the unstripped original, so a
    caller passing ``[" arxiv ", "semantic_scholar"]`` produced an engines
    list with a whitespace-wrapped first entry that ``search_papers``
    silently ignored (engine lookup keys on the bare name).
    """
    pipeline = create_pipeline(_make_config(tmp_path))

    captured: dict = {}

    async def fake_search(query, engines=None, max_results=8, **kwargs):
        captured["engines"] = engines
        return []

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    await pipeline.consume_research_request(
        topic="whitespace",
        suggested_sources=[" arxiv ", "semantic_scholar", "\tcore\n"],
    )

    # Each engine is stripped verbatim — no surrounding whitespace survives.
    assert captured["engines"] == ["arxiv", "semantic_scholar", "core"]


def test_list_evidence_sources_legacy_data_without_owned_locally_key_auto_infers_from_path(
    tmp_path,
):
    """R7 regression: rows registered before the ``owned_locally`` metadata
    flag existed (or by a path where the key was simply never set) must
    *auto-infer* from path existence rather than collapsing to False.

    This exercises the pre-R5 legacy-data case: a repo entry whose
    metadata dict lacks the key entirely. After R5's fix switched to
    ``bool(meta.get('owned_locally', False))``, those rows silently
    flipped to ``owned_locally=False`` even when the path existed on
    disk. The three-way resolution (absent → infer; explicit → honour)
    restores legacy compatibility without losing R5's intent.
    """
    pipeline = create_pipeline(_make_config(tmp_path))
    repo_path = tmp_path / "legacy-repo"
    repo_path.mkdir()

    # Simulate legacy data: add the entry directly with metadata that has
    # NO owned_locally key at all. This mimics what an older
    # register_repo() implementation would have persisted.
    pipeline.knowledge.add(
        KnowledgeEntry(
            id="repo_legacy",
            tier=Tier.DERIVED,
            title="Repo: legacy",
            content="Legacy pre-R5 entry",
            source="registry",
            scope="registry",
            tags=["repo", "evidence-source", "repo:legacy"],
            status=EntryStatus.DISTILLED,
            metadata={
                "project": "legacy",
                "repo_path": str(repo_path),
                "scope": "",
                "depends_on": [],
                # Note: NO "owned_locally" key — pre-R5 data shape.
                "upstream_url": "",
                "license": "",
            },
        )
    )

    rows = pipeline.list_evidence_sources()
    assert len(rows) == 1
    row = rows[0]
    assert row["project"] == "legacy"
    # Legacy row: key absent → auto-infer from path. Path exists → True.
    assert row["owned_locally"] is True
    assert row["path_exists"] is True
    # Flag surfaces that the stored metadata had NO explicit value.
    assert row["owned_locally_explicit"] is False

    # And: once a caller registers the same project explicitly with False,
    # that authoritative value must win on the next listing (R5 intent).
    pipeline.register_evidence_source(
        "legacy",
        str(repo_path),
        owned_locally=False,
    )
    rows_after = pipeline.list_evidence_sources()
    assert len(rows_after) == 1
    updated = rows_after[0]
    assert updated["owned_locally"] is False
    assert updated["owned_locally_explicit"] is True
    assert updated["path_exists"] is True


@pytest.mark.asyncio
async def test_consume_research_request_continues_batch_on_per_paper_failure(
    monkeypatch, tmp_path
):
    """Per-paper error isolation: a transient failure on one paper in a
    batch must not abort the rest. Failed papers land in ``failed`` with the
    captured exception; successful ones keep flowing through ``ingested``.
    The digest-backed research-request record reflects the mixed outcome
    via ``status=partial`` (some succeeded, some failed).
    """
    pipeline = create_pipeline(_make_config(tmp_path))

    async def fake_search(query, engines=None, max_results=8, **kwargs):
        return [
            SimpleNamespace(title="Paper 1", url="https://example.com/p1", source="arxiv"),
            SimpleNamespace(title="Paper 2", url="https://example.com/p2", source="arxiv"),
            SimpleNamespace(title="Paper 3", url="https://example.com/p3", source="arxiv"),
        ]

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    ingest_calls: list[str] = []

    async def fake_ingest(url: str):
        ingest_calls.append(url)
        if url == "https://example.com/p2":
            raise RuntimeError("transient fetch boom")
        return f"entry-{url.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(pipeline, "ingest_paper", fake_ingest)

    result = await pipeline.consume_research_request(
        topic="batch isolation",
        auto_fetch=True,
    )

    # All three URLs were attempted (batch did not abort on #2's failure).
    assert ingest_calls == [
        "https://example.com/p1",
        "https://example.com/p2",
        "https://example.com/p3",
    ]

    # Papers #1 and #3 landed.
    assert result["ingested"] == ["entry-p1", "entry-p3"]
    # Back-compat alias still populated.
    assert result["ingested_entry_ids"] == ["entry-p1", "entry-p3"]

    # Paper #2 is captured with the exception details.
    assert len(result["failed"]) == 1
    failure = result["failed"][0]
    assert failure["url"] == "https://example.com/p2"
    assert failure["error_type"] == "RuntimeError"
    assert "transient fetch boom" in failure["error"]

    # Summary reflects the requested/ingested/failed counts.
    assert result["summary"]["requested"] == 3
    assert result["summary"]["ingested"] == 2
    assert result["summary"]["failed"] == 1

    # Mixed outcome → status is "partial".
    assert result["status"] == "partial"


@pytest.mark.asyncio
async def test_consume_research_request_cancelled_error_propagates(
    monkeypatch, tmp_path
):
    """``asyncio.CancelledError`` from ingest_paper must propagate out of
    the batch loop. The broad Exception catch that isolates per-paper
    failures must not swallow cancellation — otherwise task cancellation
    semantics break silently.
    """
    pipeline = create_pipeline(_make_config(tmp_path))

    async def fake_search(query, engines=None, max_results=8, **kwargs):
        return [
            SimpleNamespace(title="Paper 1", url="https://example.com/p1", source="arxiv"),
        ]

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    import asyncio as _asyncio

    async def fake_ingest(url: str):
        raise _asyncio.CancelledError()

    monkeypatch.setattr(pipeline, "ingest_paper", fake_ingest)

    with pytest.raises(_asyncio.CancelledError):
        await pipeline.consume_research_request(
            topic="cancel propagation",
            auto_fetch=True,
        )


@pytest.mark.asyncio
async def test_consume_research_request_all_failed_status(monkeypatch, tmp_path):
    """When every attempted paper fails, status is ``failed`` (not partial)."""
    pipeline = create_pipeline(_make_config(tmp_path))

    async def fake_search(query, engines=None, max_results=8, **kwargs):
        return [
            SimpleNamespace(title="P1", url="https://example.com/p1", source="arxiv"),
            SimpleNamespace(title="P2", url="https://example.com/p2", source="arxiv"),
        ]

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    async def fake_ingest(url: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "ingest_paper", fake_ingest)

    result = await pipeline.consume_research_request(
        topic="all fail",
        auto_fetch=True,
    )

    assert result["ingested"] == []
    assert len(result["failed"]) == 2
    assert result["status"] == "failed"
    assert result["summary"] == {
        "requested": 2,
        "ingested": 0,
        "failed": 2,
        "distilled": 0,
    }


@pytest.mark.asyncio
async def test_consume_research_request_all_succeeded_status(monkeypatch, tmp_path):
    """When every paper lands, status is ``completed``."""
    pipeline = create_pipeline(_make_config(tmp_path))

    async def fake_search(query, engines=None, max_results=8, **kwargs):
        return [
            SimpleNamespace(title="P1", url="https://example.com/p1", source="arxiv"),
            SimpleNamespace(title="P2", url="https://example.com/p2", source="arxiv"),
        ]

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    async def fake_ingest(url: str):
        return f"entry-{url.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(pipeline, "ingest_paper", fake_ingest)

    result = await pipeline.consume_research_request(
        topic="all good",
        auto_fetch=True,
    )

    assert result["ingested"] == ["entry-p1", "entry-p2"]
    assert result["failed"] == []
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_consume_research_request_skipped_status_without_auto_fetch(
    monkeypatch, tmp_path
):
    """Search-only requests (auto_fetch=False) report ``status=skipped`` —
    no ingest work was attempted, so completed/partial/failed don't apply.
    """
    pipeline = create_pipeline(_make_config(tmp_path))

    async def fake_search(query, engines=None, max_results=8, **kwargs):
        return [SimpleNamespace(title="P1", url="https://example.com/p1", source="arxiv")]

    monkeypatch.setattr("researcher.pipeline.search_papers", fake_search)

    result = await pipeline.consume_research_request(topic="search only")

    assert result["status"] == "skipped"
    assert result["ingested"] == []
    assert result["failed"] == []


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
