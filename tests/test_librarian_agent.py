from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import yaml

from khonliang.knowledge.store import EntryStatus, KnowledgeEntry, Tier
from researcher import librarian_agent
from khonliang_researcher import GapReport
from researcher.librarian_agent import LibrarianAgent


def _make_config(tmp_path):
    config = {
        "db_path": str(tmp_path / "researcher.db"),
        "models": {},
        "projects": {},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_librarian_agent_registers_expected_skills(tmp_path):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    names = {skill.name for skill in agent.register_skills()}

    assert names == {
        "library_health",
        "rebuild_neighborhoods",
        "taxonomy_report",
        "suggest_missing_nodes",
        "classify_paper",
        "promote_investigation",
        "identify_gaps",
    }


@pytest.mark.asyncio
async def test_rebuild_neighborhoods_persists_snapshot_and_classification(tmp_path, monkeypatch):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    agent.pipeline.knowledge.add(
        KnowledgeEntry(
            id="paper1",
            tier=Tier.IMPORTED,
            title="Multi-agent code review",
            content="paper",
            source="paper1",
            scope="research",
            tags=["paper"],
            status=EntryStatus.DISTILLED,
            metadata={"url": "https://example.com/paper1"},
        )
    )
    agent.pipeline.triples.add(
        subject="Multi Agent Code Review",
        predicate="specializes",
        obj="Multi Agent Review",
        confidence=0.9,
        source="paper:paper1",
    )
    agent.pipeline.triples.add(
        subject="Multi Agent Code Review",
        predicate="applies_to",
        obj="Pull Request Review",
        confidence=0.8,
        source="paper:paper1",
    )

    async def fake_post(url, json):
        return SimpleNamespace(json=lambda: {"artifact": {"id": "art_123"}})

    published = []

    async def fake_publish(topic, payload):
        published.append((topic, payload))

    monkeypatch.setattr(agent._http, "post", fake_post)
    monkeypatch.setattr(agent, "publish", fake_publish)

    result = await agent.handle_rebuild_neighborhoods({"audience": "", "reason": "test"})

    assert result["artifact_id"] == "art_123"
    assert agent.store.latest_snapshot() is not None
    assert agent.store.get_classification("paper1") is not None
    assert [topic for topic, _ in published] == ["library.rebuilt"]


@pytest.mark.asyncio
async def test_taxonomy_report_caps_brief_output(tmp_path, monkeypatch):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    async def fake_ensure_snapshot(audience: str = "", reason: str = "bootstrap"):
        return {
            "groups": [
                {"code": f"G{i}", "label": f"group {i}", "audience": ""}
                for i in range(6)
            ],
            "relationships": [
                {"source": f"G{i}", "target": f"G{i + 1}", "predicate": "related_to"}
                for i in range(6)
            ],
        }

    monkeypatch.setattr(agent, "_ensure_snapshot", fake_ensure_snapshot)

    result = await agent.handle_taxonomy_report(
        {"detail": "brief", "max_groups": 2, "max_relationships": 3}
    )

    assert len(result["groups"]) == 2
    assert len(result["relationships"]) == 3
    assert result["summary"]["group_count"] == 6
    assert result["summary"]["relationship_count"] == 6
    assert result["summary"]["groups_truncated"] is True
    assert result["summary"]["relationships_truncated"] is True


@pytest.mark.asyncio
async def test_identify_gaps_caps_brief_output(tmp_path, monkeypatch):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    async def fake_ensure_snapshot(audience: str = "", reason: str = "bootstrap"):
        return {"groups": [], "relationships": []}

    async def fake_publish(topic, payload):
        return None

    async def fake_post(url, json):
        return SimpleNamespace(json=lambda: {"artifact": {"id": "art_gap_123"}})

    monkeypatch.setattr(agent, "_ensure_snapshot", fake_ensure_snapshot)
    monkeypatch.setattr(agent.store, "list_classifications", lambda audience="": [])
    monkeypatch.setattr(agent, "publish", fake_publish)
    monkeypatch.setattr(agent._http, "post", fake_post)
    monkeypatch.setattr(
        librarian_agent,
        "identify_gap_candidates",
        lambda taxonomy, classifications, audience="": [
            GapReport(
                request_id=f"gap-{i}",
                topic=f"topic {i}",
                audience=audience,
                branch=f"branch-{i}",
                rationale="missing coverage",
            )
            for i in range(6)
        ],
    )

    result = await agent.handle_identify_gaps({"detail": "brief", "max_gaps": 2})

    assert len(result["gaps"]) == 2
    assert result["artifact_id"] == "art_gap_123"
    assert result["summary"]["total_gaps"] == 6
    assert result["summary"]["emitted_count"] == 2
    assert result["summary"]["truncated"] is True


@pytest.mark.asyncio
async def test_handle_bus_event_classifies_distilled_entry(tmp_path, monkeypatch):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    called = {}

    async def fake_classify(args):
        called["args"] = args
        return {"status": "classified"}

    monkeypatch.setattr(agent, "handle_classify_paper", fake_classify)

    await agent._handle_bus_event(
        {
            "topic": "ingest.url_distilled",
            "payload": {"entry_id": "paper-123"},
        }
    )

    assert called["args"] == {"paper_id": "paper-123", "detail": "brief"}


@pytest.mark.asyncio
async def test_handle_bus_event_rebuilds_on_queue_drained(tmp_path, monkeypatch):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    called = {}

    async def fake_rebuild(args):
        called["args"] = args
        return {"snapshot_id": "libsnap_1"}

    monkeypatch.setattr(agent, "handle_rebuild_neighborhoods", fake_rebuild)

    await agent._handle_bus_event(
        {
            "topic": "ingest.queue_drained",
            "payload": {"queue_depth": 0},
        }
    )

    assert called["args"] == {
        "audience": "",
        "reason": "event:ingest.queue_drained",
    }


def test_librarian_agent_cli_install_uses_librarian_class(monkeypatch):
    called = {}

    @classmethod
    def fake_from_cli(cls, argv=None):
        called["cls"] = cls
        called["argv"] = argv
        return None

    monkeypatch.setattr(LibrarianAgent, "from_cli", fake_from_cli)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            command="install",
            id="librarian-primary",
            bus="http://localhost:8788",
            config="config.yaml",
        ),
    )

    librarian_agent.main()

    assert called["cls"] is LibrarianAgent
    assert called["argv"][:4] == ["install", "--id", "librarian-primary", "--bus"]


def test_artifact_id_accepts_top_level_and_nested_shapes(tmp_path):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    assert agent._artifact_id({"id": "art_top"}) == "art_top"
    assert agent._artifact_id({"artifact": {"id": "art_nested"}}) == "art_nested"
    assert agent._artifact_id({}) == ""
