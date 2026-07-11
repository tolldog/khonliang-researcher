"""Agent-level tests for the catalog_* bus handlers (fr_researcher_bbe95f12).

Mirrors the fake-agent harness in test_agent_ingest_async.py: wires
``_extend_with_native_handlers`` against a stub pipeline and drives the
registered ``_handlers`` dict directly, without a real BaseAgent/bus.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from researcher.self_catalog import build_self_catalog


def _build_fake_agent(pipeline_stub: Any) -> Any:
    from researcher.agent import _extend_with_native_handlers

    agent = SimpleNamespace(
        agent_id="test-researcher",
        agent_type="researcher",
        version="0.0.0-test",
        bus_url="http://test",
        config_path="/tmp/test.yaml",
        _handlers={},
        register_skills=lambda: [],
        register_collaborations=lambda: [],
    )
    _extend_with_native_handlers(agent, pipeline_stub)
    return agent


def _pipeline_without_catalog() -> Any:
    return SimpleNamespace(config={})


def _pipeline_with_catalog(tmp_path) -> Any:
    catalog = build_self_catalog({"db_path": str(tmp_path / "researcher.db")})
    return SimpleNamespace(config={}, catalog=catalog), catalog


@pytest.mark.asyncio
async def test_catalog_query_disabled_when_no_catalog():
    agent = _build_fake_agent(_pipeline_without_catalog())
    handler = agent._handlers["catalog_query"]
    result = await handler({"project": "khonliang"})
    assert "error" in result


@pytest.mark.asyncio
async def test_catalog_query_requires_project():
    agent = _build_fake_agent(_pipeline_without_catalog())
    handler = agent._handlers["catalog_query"]
    result = await handler({})
    assert "error" in result


@pytest.mark.asyncio
async def test_catalog_query_returns_upserted_records(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)

    from librarian_lib import IndexRecord

    catalog.upsert(
        IndexRecord(
            project="khonliang",
            source="researcher",
            record_id="p1",
            schema_version=1,
            kind="paper",
            text="Title\n\nAbstract",
        )
    )

    handler = agent._handlers["catalog_query"]
    result = await handler({"project": "khonliang"})
    assert result["count"] == 1
    assert result["rows"][0]["record_id"] == "p1"


@pytest.mark.asyncio
async def test_catalog_search_text_fallback(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)

    from librarian_lib import IndexRecord

    catalog.upsert(
        IndexRecord(
            project="khonliang",
            source="researcher",
            record_id="p1",
            schema_version=1,
            kind="paper",
            text="Consensus voting for multi-agent systems",
        )
    )

    handler = agent._handlers["catalog_search"]
    result = await handler({"project": "khonliang", "query_text": "consensus"})
    assert result["mode"] == "text_fallback"
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_catalog_stats(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)
    handler = agent._handlers["catalog_stats"]
    result = await handler({})
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_catalog_list_since_requires_since_ts(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)
    handler = agent._handlers["catalog_list_since"]
    result = await handler({"project": "khonliang"})
    assert "error" in result


@pytest.mark.asyncio
async def test_catalog_list_since_returns_recent(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)

    from librarian_lib import IndexRecord

    catalog.upsert(
        IndexRecord(
            project="khonliang", source="researcher", record_id="p1",
            schema_version=1, kind="paper", text="t",
        )
    )
    handler = agent._handlers["catalog_list_since"]
    result = await handler({"project": "khonliang", "since_ts": 0})
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_catalog_mark_stale_requires_spec_object(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)
    handler = agent._handlers["catalog_mark_stale"]
    result = await handler({"project": "khonliang", "spec": "not-a-dict"})
    assert "error" in result


@pytest.mark.asyncio
async def test_catalog_mark_stale_bulk_flags_rows(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)

    from librarian_lib import IndexRecord

    catalog.upsert(
        IndexRecord(
            project="khonliang", source="researcher", record_id="p1",
            schema_version=1, kind="paper", text="t",
        )
    )
    handler = agent._handlers["catalog_mark_stale"]
    result = await handler({"project": "khonliang", "spec": {"version": 2}})
    assert result["updated"] == 1
