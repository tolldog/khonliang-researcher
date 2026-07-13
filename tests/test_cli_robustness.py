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


def _all_output(result) -> str:
    """stdout + stderr regardless of how this Click version splits them.

    These commands emit errors via ``click.echo(..., err=True)``. Some Click
    versions mix stderr into ``result.output``; others capture it separately on
    ``result.stderr``. Concatenate both so the assertion doesn't depend on which
    (duplication is harmless for substring checks)."""
    parts = [result.output or ""]
    try:
        parts.append(result.stderr or "")
    except (ValueError, AttributeError):
        pass
    return "".join(parts)


def test_distill_missing_arg_exits_nonzero(monkeypatch):
    runner = _runner(monkeypatch, SimpleNamespace())
    result = runner.invoke(cli, ["distill"])  # no ENTRY_ID, no --all
    assert result.exit_code == 1
    assert "Provide an ENTRY_ID" in _all_output(result)


def test_distill_failure_branch_exits_nonzero(monkeypatch):
    """A single-paper distillation that returns success=False must exit 1
    (not just print), so scripts/CI can detect it."""
    async def distill(entry_id):
        return SimpleNamespace(success=False, title="bad-paper")

    stub = SimpleNamespace(distill=distill)
    runner = _runner(monkeypatch, stub)
    result = runner.invoke(cli, ["distill", "entry-123"])
    assert result.exit_code == 1
    assert "Distillation failed" in _all_output(result)


def test_distill_success_but_stuck_exits_nonzero(monkeypatch):
    # codex P2 round 10, bug_khonliang-researcher_706df96b: a distill that
    # succeeded but leaked its lock (stuck=True) must not read as a clean
    # exit 0 — that's how a cron job / operator would miss a condition that
    # needs a manual process restart to recover.
    async def distill(entry_id):
        return SimpleNamespace(
            success=True, skipped=False, errored=False, stuck=True,
            title="ok-but-stuck", summary=None, triples=[], assessments={},
        )

    stub = SimpleNamespace(distill=distill)
    runner = _runner(monkeypatch, stub)
    result = runner.invoke(cli, ["distill", "entry-123"])
    assert result.exit_code == 1
    assert "stuck" in _all_output(result).lower()


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
    assert "No search queries" in _all_output(result)
    # Clean error, not a crash: no KeyError('papers_new') traceback.
    assert "KeyError" not in _all_output(result)
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
    assert "could not be reloaded" in _all_output(result)
    # No AttributeError: 'NoneType' has no attribute 'title'.
    assert not isinstance(result.exception, AttributeError)
