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
    classification = agent.store.get_classification("paper1")
    assert classification is not None
    assert classification.source_snapshot_id == result["snapshot_id"]
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
async def test_suggest_missing_nodes_normalizes_group_labels(tmp_path, monkeypatch):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    async def fake_ensure_snapshot(audience: str = "", reason: str = "bootstrap"):
        return {
            "groups": [
                {
                    "code": "DEV.001",
                    "label": "Claude CLI Optimization",
                    "audience": "developer-researcher",
                }
            ],
            "relationships": [],
        }

    monkeypatch.setattr(agent, "_ensure_snapshot", fake_ensure_snapshot)
    monkeypatch.setattr(librarian_agent, "suggest_entities", lambda graph, query: [])

    result = await agent.handle_suggest_missing_nodes(
        {"query": "claude cli optimization", "audience": "developer-researcher"}
    )

    assert [group["code"] for group in result["group_candidates"]] == ["DEV.001"]


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


@pytest.mark.asyncio
async def test_handle_bus_event_ignores_non_dict_payload(tmp_path, monkeypatch):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    called = {"classify": 0, "rebuild": 0}

    async def fake_classify(args):
        called["classify"] += 1
        return {"status": "classified"}

    async def fake_rebuild(args):
        called["rebuild"] += 1
        return {"snapshot_id": "libsnap_1"}

    monkeypatch.setattr(agent, "handle_classify_paper", fake_classify)
    monkeypatch.setattr(agent, "handle_rebuild_neighborhoods", fake_rebuild)

    await agent._handle_bus_event({"topic": "ingest.url_distilled", "payload": "bad"})

    assert called == {"classify": 0, "rebuild": 0}


@pytest.mark.asyncio
async def test_handle_bus_event_tolerates_non_dict_payload_and_warns(tmp_path, monkeypatch, caplog):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    async def fake_classify(args):
        return {"status": "classified"}

    async def fake_rebuild(args):
        return {"snapshot_id": "libsnap_1"}

    monkeypatch.setattr(agent, "handle_classify_paper", fake_classify)
    monkeypatch.setattr(agent, "handle_rebuild_neighborhoods", fake_rebuild)

    caplog.set_level("WARNING", logger="researcher.librarian_agent")

    # list payload — not a dict, must not raise, must log a warning
    await agent._handle_bus_event({"topic": "ingest.url_distilled", "payload": [1, 2]})
    # int payload — same
    await agent._handle_bus_event({"topic": "ingest.queue_drained", "payload": 42})
    # missing payload key entirely — also fine, but NO warning (raw_payload is None)
    await agent._handle_bus_event({"topic": "ingest.url_distilled"})

    warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert len(warnings) == 2
    assert all("non-dict payload" in rec.getMessage() for rec in warnings)


def test_paper_entries_excludes_ideas_and_url_less_entries(tmp_path):
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    # Real paper: tag=paper and has url
    agent.pipeline.knowledge.add(
        KnowledgeEntry(
            id="paper-ok",
            tier=Tier.IMPORTED,
            title="Real paper",
            content="body",
            source="https://example.com/a",
            scope="research",
            tags=["paper"],
            status=EntryStatus.DISTILLED,
            metadata={"url": "https://example.com/a"},
        )
    )
    # Idea: tag=idea, no url
    agent.pipeline.knowledge.add(
        KnowledgeEntry(
            id="idea-xx",
            tier=Tier.IMPORTED,
            title="An idea",
            content="raw idea text",
            source="idea",
            scope="research",
            tags=["idea"],
            status=EntryStatus.INGESTED,
            metadata={"claims": []},
        )
    )
    # Edge: tag=paper but no url — still excluded (guards bad data)
    agent.pipeline.knowledge.add(
        KnowledgeEntry(
            id="paper-nourl",
            tier=Tier.IMPORTED,
            title="Malformed paper",
            content="body",
            source="?",
            scope="research",
            tags=["paper"],
            status=EntryStatus.INGESTED,
            metadata={},
        )
    )

    ids = {entry.id for entry in agent._paper_entries()}
    assert ids == {"paper-ok"}


@pytest.mark.asyncio
async def test_rebuild_neighborhoods_concurrent_within_same_second_produces_distinct_ids(
    tmp_path, monkeypatch
):
    """Two rebuilds firing within the same wall-clock second (e.g. a manual
    trigger racing an ``ingest.queue_drained`` event) must produce distinct
    snapshot_ids. Second-resolution ``int(time.time())`` would collide and one
    snapshot would overwrite the other; nanosecond resolution prevents this.
    """
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    async def fake_post(url, json):
        return SimpleNamespace(json=lambda: {"artifact": {"id": "art_x"}})

    async def fake_publish(topic, payload):
        return None

    monkeypatch.setattr(agent._http, "post", fake_post)
    monkeypatch.setattr(agent, "publish", fake_publish)

    # Pin int(time.time()) to a fixed second so a naive `libsnap_{int(...)}`
    # implementation would produce identical ids across the two rebuilds.
    # time_ns() is NOT pinned, so nanosecond-based ids stay distinct.
    import time as _time

    monkeypatch.setattr(_time, "time", lambda: 1_700_000_000.0)

    first = await agent.handle_rebuild_neighborhoods({"audience": "", "reason": "manual"})
    second = await agent.handle_rebuild_neighborhoods(
        {"audience": "", "reason": "event:ingest.queue_drained"}
    )

    assert first["snapshot_id"] != second["snapshot_id"], (
        "snapshot_id collision: two rebuilds in the same second produced the same id "
        f"({first['snapshot_id']}). Use time.time_ns() for nanosecond resolution."
    )
    # Both must be persisted — neither silently overwritten by the other.
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "researcher.db"))
    try:
        rows = conn.execute(
            "SELECT snapshot_id FROM librarian_neighborhood_snapshots"
        ).fetchall()
    finally:
        conn.close()
    snapshot_ids = {row[0] for row in rows}
    assert first["snapshot_id"] in snapshot_ids
    assert second["snapshot_id"] in snapshot_ids


@pytest.mark.asyncio
async def test_classify_paper_caches_latest_snapshot_lookup(tmp_path, monkeypatch):
    """handle_classify_paper must call store.latest_snapshot at most once per
    classified path. Calling it twice in a single expression risks the two
    reads disagreeing if a concurrent rebuild lands between them.
    """
    agent = LibrarianAgent(
        agent_id="librarian-test",
        bus_url="http://localhost:8788",
        config_path=_make_config(tmp_path),
    )

    agent.pipeline.knowledge.add(
        KnowledgeEntry(
            id="paper-lookup",
            tier=Tier.IMPORTED,
            title="Cache test paper",
            content="paper",
            source="paper-lookup",
            scope="research",
            tags=["paper"],
            status=EntryStatus.DISTILLED,
            metadata={"url": "https://example.com/paper-lookup"},
        )
    )
    agent.pipeline.triples.add(
        subject="Cache Test Paper",
        predicate="specializes",
        obj="Cache Behavior",
        confidence=0.9,
        source="paper:paper-lookup",
    )
    agent.pipeline.triples.add(
        subject="Cache Test Paper",
        predicate="applies_to",
        obj="Classification Lookup",
        confidence=0.8,
        source="paper:paper-lookup",
    )

    async def fake_ensure_snapshot(audience: str = "", reason: str = "bootstrap"):
        # Minimal taxonomy shape the classifier will accept; actual classification
        # behaviour isn't the contract under test — cache count is.
        return {"groups": [], "relationships": []}

    async def fake_publish(topic, payload):
        return None

    monkeypatch.setattr(agent, "_ensure_snapshot", fake_ensure_snapshot)
    monkeypatch.setattr(agent, "publish", fake_publish)

    # Force the classifier to report "classified" so the source_snapshot_id
    # lookup is exercised (that's the code path with the double-read bug).
    def fake_classify(paper_id, triples, taxonomy, audience=""):
        return {
            "status": "classified",
            "paper_id": paper_id,
            "classification_code": "TEST.001",
            "audience_tags": [audience] if audience else [],
            "confidence": 0.9,
            "rationale": "forced for test",
        }

    monkeypatch.setattr(librarian_agent, "classify_paper_from_triples", fake_classify)

    # Spy on latest_snapshot.
    real_latest = agent.store.latest_snapshot
    calls = {"count": 0}

    def counting_latest(audience: str = ""):
        calls["count"] += 1
        return real_latest(audience)

    monkeypatch.setattr(agent.store, "latest_snapshot", counting_latest)

    result = await agent.handle_classify_paper(
        {"paper_id": "paper-lookup", "audience": ""}
    )

    assert result["status"] == "classified"
    assert calls["count"] == 1, (
        f"latest_snapshot should be called exactly once per classify path; "
        f"called {calls['count']} times. Cache the result in a local variable."
    )
