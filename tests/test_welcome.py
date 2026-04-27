"""Tests for the cold-start welcome surface (fr_khonliang-bus-lib_6a82732c).

Both researcher-primary (configured via instance attr in
create_researcher_agent) and librarian-primary (class attr on
LibrarianAgent) advertise editorial WELCOME content to ephemeral
external LLM sessions. These tests pin the surface so future edits
don't silently empty it.
"""

from __future__ import annotations

import pytest


def test_librarian_welcome_is_populated():
    """LibrarianAgent.WELCOME advertises a complete cold-start surface."""
    try:
        from researcher.librarian_agent import LibrarianAgent
    except ImportError as exc:
        # Narrow skip: only the known pre-existing researcher-lib export
        # gap (AmbiguityRecord) is tolerated. Any other import failure —
        # including welcome-side regressions that break this module —
        # must red-fail so we notice.
        if "AmbiguityRecord" in str(exc):
            pytest.skip(f"pre-existing researcher-lib gap: {exc}")
        raise
    from khonliang_bus import Welcome

    w = LibrarianAgent.WELCOME
    assert isinstance(w, Welcome), (
        "WELCOME must be a Welcome instance — handle_welcome relies on "
        "the dataclass shape and will raise TypeError for anything else."
    )
    assert w.role
    assert w.mission
    assert w.not_responsible_for, "boundaries must list at least one excluded responsibility"
    assert w.delegates_to, "librarian must declare at least one downstream delegate"
    assert w.entry_points
    advertised = {ep.skill for ep in w.entry_points}
    assert "taxonomy_report" in advertised
    assert "library_health" in advertised
    for ep in w.entry_points:
        assert ep.skill, "entry-point skill name must not be empty"
        assert ep.when_to_use, f"entry-point '{ep.skill}' missing when_to_use"


def test_researcher_welcome_is_populated_after_create_agent(monkeypatch, tmp_path):
    """``create_researcher_agent`` attaches WELCOME as instance attr.

    The researcher-side WELCOME isn't a class attr because the agent
    uses ``BaseAgent.from_mcp()``, so the test mocks the heavier deps
    (pipeline + MCP server) and asserts the attached editorial.
    """
    from khonliang_bus import Welcome
    from researcher import agent as agent_mod

    fake_pipeline = object()
    fake_server = object()

    def _fake_create_pipeline(_path):
        return fake_pipeline

    def _fake_create_server(_pipeline):
        return fake_server

    fake_agent = type("FakeAgent", (), {})()
    fake_agent.register_skills = lambda: []
    fake_agent.version = "0.0.0"

    def _fake_from_mcp(*_a, **_kw):
        return fake_agent

    def _fake_extend(*_a, **_kw):
        pass

    monkeypatch.setattr("researcher.pipeline.create_pipeline", _fake_create_pipeline)
    monkeypatch.setattr("researcher.server.create_research_server", _fake_create_server)
    monkeypatch.setattr(agent_mod.BaseAgent, "from_mcp", staticmethod(_fake_from_mcp))
    monkeypatch.setattr(agent_mod, "_extend_with_native_handlers", _fake_extend)

    out = agent_mod.create_researcher_agent(
        agent_id="researcher-primary",
        bus_url="http://localhost:9999",
        config_path=str(tmp_path / "config.yaml"),
    )

    assert isinstance(out.WELCOME, Welcome)
    assert out.WELCOME.role
    assert out.WELCOME.mission
    assert out.WELCOME.not_responsible_for, "researcher must declare not_responsible_for"
    assert out.WELCOME.delegates_to, "researcher must declare delegates_to"
    advertised = {ep.skill for ep in out.WELCOME.entry_points}
    assert "find_relevant" in advertised
    assert "brief_on" in advertised
    assert out.WELCOME.guide_skill == "research_guide", (
        "researcher must advertise research_guide as its guide_skill — "
        "the welcome FR's acceptance criterion expects this pointer to "
        "deeper context."
    )
    for ep in out.WELCOME.entry_points:
        assert ep.skill, "entry-point skill name must not be empty"
        assert ep.when_to_use, f"entry-point '{ep.skill}' missing when_to_use"
