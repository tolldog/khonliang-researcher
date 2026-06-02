"""Regression tests for second-pass CLI robustness fixes.

Covers:
- bug_researcher_d908c692 — `idea full` guards an error result (no KeyError)
- bug_researcher_801011a5 — `idea ingest`/`idea full` guard a None knowledge.get
- bug_researcher_294441a5 — `distill` exits non-zero on failure / missing arg
"""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from researcher import cli as cli_mod
from researcher.cli import cli


def _runner(monkeypatch, stub) -> CliRunner:
    monkeypatch.setattr(cli_mod, "create_pipeline", lambda config: stub)
    return CliRunner()


def test_distill_missing_arg_exits_nonzero(monkeypatch):
    runner = _runner(monkeypatch, SimpleNamespace())
    result = runner.invoke(cli, ["distill"])  # no ENTRY_ID, no --all
    assert result.exit_code == 1
    assert "Provide an ENTRY_ID" in result.output


def test_idea_full_error_result_exits_clean_no_keyerror(monkeypatch):
    async def ingest_idea(text, source=""):
        return "idea1"

    async def research_idea(idea_id, max_papers, **kwargs):
        return {"error": "No search queries in idea metadata"}

    entry = SimpleNamespace(title="My Idea", metadata={"claims": []})
    stub = SimpleNamespace(
        ingest_idea=ingest_idea,
        research_idea=research_idea,
        knowledge=SimpleNamespace(get=lambda _id: entry),
    )
    runner = _runner(monkeypatch, stub)
    result = runner.invoke(cli, ["idea", "full", "some idea text"])
    assert result.exit_code == 1
    assert "No search queries" in result.output
    # Clean error, not a crash: no KeyError('papers_new') traceback.
    assert "KeyError" not in result.output
    assert not isinstance(result.exception, KeyError)


def test_idea_ingest_none_entry_exits_clean(monkeypatch):
    async def ingest_idea(text, source=""):
        return "idea1"

    stub = SimpleNamespace(
        ingest_idea=ingest_idea,
        knowledge=SimpleNamespace(get=lambda _id: None),  # reload returns None
    )
    runner = _runner(monkeypatch, stub)
    result = runner.invoke(cli, ["idea", "ingest", "some idea text"])
    assert result.exit_code == 1
    assert "could not be reloaded" in result.output
    # No AttributeError: 'NoneType' has no attribute 'title'.
    assert not isinstance(result.exception, AttributeError)
