"""Install/uninstall CLI dispatch must carry subclass identity.

``BaseAgent.from_cli`` is a classmethod that constructs ``cls(...)``;
calling it on the base class registers module_name="agent" /
agent_type="base" with the bus — a launch spec that can't start
(bug_developer_agent_main_install_uses_base_class_4bb0a5cf). Mirrors
test_librarian_agent_cli_install_uses_librarian_class.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from khonliang_bus import BaseAgent

from researcher import agent as researcher_agent
from researcher import generic_agent
from researcher.generic_agent import GenericResearcher


def _patch_args(monkeypatch, agent_id: str) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            command="install",
            id=agent_id,
            bus="http://localhost:8788",
            config="config.yaml",
        ),
    )


def test_researcher_agent_cli_install_carries_identity(monkeypatch):
    called = {}

    @classmethod
    def fake_from_cli(cls, argv=None):
        called["cls"] = cls
        called["argv"] = argv
        return None

    monkeypatch.setattr(BaseAgent, "from_cli", fake_from_cli)
    _patch_args(monkeypatch, "researcher-primary")

    researcher_agent.main()

    # The runtime agent is a dynamic from_mcp subclass, so install goes
    # through a stub — what matters is the identity it registers.
    assert called["cls"] is not BaseAgent
    assert issubclass(called["cls"], BaseAgent)
    assert called["cls"].module_name == "researcher.agent"
    assert called["cls"].agent_type == "researcher"
    assert called["argv"][:4] == ["install", "--id", "researcher-primary", "--bus"]


def test_generic_agent_cli_install_uses_subclass(monkeypatch):
    called = {}

    @classmethod
    def fake_from_cli(cls, argv=None):
        called["cls"] = cls
        called["argv"] = argv
        return None

    monkeypatch.setattr(BaseAgent, "from_cli", fake_from_cli)
    _patch_args(monkeypatch, "generic-researcher")

    generic_agent.main()

    assert called["cls"] is GenericResearcher
    assert called["cls"].module_name == "researcher.generic_agent"
    assert called["argv"][:4] == ["install", "--id", "generic-researcher", "--bus"]
