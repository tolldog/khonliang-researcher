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
            source="researcher-primary",
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
            source="researcher-primary",
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
async def test_list_since_requires_since_ts(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)
    handler = agent._handlers["list_since"]
    result = await handler({"project": "khonliang"})
    assert "error" in result


@pytest.mark.asyncio
async def test_list_since_returns_recent(tmp_path):
    pipeline, catalog = _pipeline_with_catalog(tmp_path)
    agent = _build_fake_agent(pipeline)

    from librarian_lib import IndexRecord

    catalog.upsert(
        IndexRecord(
            project="khonliang", source="researcher-primary", record_id="p1",
            schema_version=1, kind="paper", text="t",
        )
    )
    handler = agent._handlers["list_since"]
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
            project="khonliang", source="researcher-primary", record_id="p1",
            schema_version=1, kind="paper", text="t",
        )
    )
    handler = agent._handlers["catalog_mark_stale"]
    result = await handler({"project": "khonliang", "spec": {"version": 2}})
    assert result["updated"] == 1


# ---------------------------------------------------------------------------
# catalog_fetch: the exact-id lookup an IndexRecord's `ref` points at.
# ---------------------------------------------------------------------------


def _pipeline_with_knowledge(entries: dict) -> Any:
    return SimpleNamespace(
        config={},
        knowledge=SimpleNamespace(get=lambda eid: entries.get(eid)),
    )


@pytest.mark.asyncio
async def test_catalog_fetch_requires_record_id():
    agent = _build_fake_agent(_pipeline_with_knowledge({}))
    handler = agent._handlers["catalog_fetch"]
    result = await handler({})
    assert "error" in result


@pytest.mark.asyncio
async def test_catalog_fetch_not_found():
    agent = _build_fake_agent(_pipeline_with_knowledge({}))
    handler = agent._handlers["catalog_fetch"]
    result = await handler({"record_id": "missing"})
    assert result["error"] == "not found"


@pytest.mark.asyncio
async def test_catalog_fetch_returns_entry_and_summary():
    import json as _json

    paper = SimpleNamespace(
        title="A Paper", content="raw body", metadata={"url": "https://x"}, status="distilled",
    )
    summary_entry = SimpleNamespace(content=_json.dumps({"abstract": "abs"}))
    agent = _build_fake_agent(
        _pipeline_with_knowledge({"p1": paper, "p1_summary": summary_entry})
    )
    handler = agent._handlers["catalog_fetch"]
    result = await handler({"record_id": "p1"})
    assert result["title"] == "A Paper"
    assert result["summary"] == {"abstract": "abs"}
    # Summary present → raw content isn't duplicated into the response.
    assert result["content"] is None


@pytest.mark.asyncio
async def test_catalog_fetch_falls_back_to_content_without_summary():
    idea = SimpleNamespace(title="An Idea", content="idea body", metadata={}, status="ingested")
    agent = _build_fake_agent(_pipeline_with_knowledge({"idea1": idea}))
    handler = agent._handlers["catalog_fetch"]
    result = await handler({"record_id": "idea1"})
    assert result["summary"] is None
    assert result["content"] == "idea body"


# ---------------------------------------------------------------------------
# create_researcher_agent rebuilds the catalog with the REAL bus agent_id
# (codex P1: a config bus_agent_id that drifted from the actual --id, or was
# never set at all, must not leave the catalog mis-stamped under the
# "researcher-primary" default while the agent itself runs under a
# different id).
# ---------------------------------------------------------------------------


def test_create_researcher_agent_rebuilds_catalog_with_runtime_agent_id(monkeypatch, tmp_path):
    from researcher import agent as agent_mod

    fake_pipeline = SimpleNamespace(config={"db_path": str(tmp_path / "researcher.db")}, catalog=object())
    fake_server = object()
    fake_agent = type("FakeAgent", (), {})()
    fake_agent.register_skills = lambda: []
    fake_agent.version = "0.0.0"

    monkeypatch.setattr("researcher.pipeline.create_pipeline", lambda _p: fake_pipeline)
    monkeypatch.setattr("researcher.server.create_research_server", lambda _pipe: fake_server)
    monkeypatch.setattr(agent_mod.BaseAgent, "from_mcp", staticmethod(lambda *a, **kw: fake_agent))
    monkeypatch.setattr(agent_mod, "_extend_with_native_handlers", lambda *a, **kw: None)

    agent_mod.create_researcher_agent(
        agent_id="researcher-secondary",
        bus_url="http://localhost:9999",
        config_path=str(tmp_path / "config.yaml"),
    )

    # A real SelfCatalog (not the placeholder object()) was rebuilt, stamped
    # with the actual runtime agent_id — not "researcher-primary" — even
    # though nothing in `fake_pipeline.config` set bus_agent_id.
    assert fake_pipeline.catalog is not None
    assert fake_pipeline.catalog.source == "researcher-secondary"
    assert fake_pipeline.catalog.owner_agent == "researcher-secondary"


def test_create_researcher_agent_skips_rebuild_when_catalog_disabled(monkeypatch, tmp_path):
    """A pipeline with catalog=None (no db_path, or lib not installed) stays None."""
    from researcher import agent as agent_mod

    fake_pipeline = SimpleNamespace(config={}, catalog=None)
    fake_agent = type("FakeAgent", (), {})()
    fake_agent.register_skills = lambda: []
    fake_agent.version = "0.0.0"

    monkeypatch.setattr("researcher.pipeline.create_pipeline", lambda _p: fake_pipeline)
    monkeypatch.setattr("researcher.server.create_research_server", lambda _pipe: object())
    monkeypatch.setattr(agent_mod.BaseAgent, "from_mcp", staticmethod(lambda *a, **kw: fake_agent))
    monkeypatch.setattr(agent_mod, "_extend_with_native_handlers", lambda *a, **kw: None)

    agent_mod.create_researcher_agent(
        agent_id="researcher-primary",
        bus_url="http://localhost:9999",
        config_path=str(tmp_path / "config.yaml"),
    )
    assert fake_pipeline.catalog is None
