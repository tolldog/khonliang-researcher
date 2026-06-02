"""Tests for researcher utility helpers."""

import asyncio

import pytest

from researcher.util import async_repo_tree, parse_branch_specs


def test_parse_branch_specs_accepts_repeated_and_semicolon_values():
    specs = parse_branch_specs([
        "cache:Prompt Caching,Token Cost",
        "sessions:Session Hygiene;llm:Prompt Hacking",
    ])

    assert specs == [
        {"label": "cache", "seeds": ["Prompt Caching", "Token Cost"]},
        {"label": "sessions", "seeds": ["Session Hygiene"]},
        {"label": "llm", "seeds": ["Prompt Hacking"]},
    ]


@pytest.mark.parametrize("spec", ["missing-colon", ":seed", "empty:"])
def test_parse_branch_specs_rejects_malformed_specs(spec):
    with pytest.raises(ValueError, match="Invalid branch spec"):
        parse_branch_specs(spec)


# ---------------------------------------------------------------------------
# async_repo_tree (fr_researcher_0539b91f) — non-clone branches. The clone path
# is covered end-to-end by the github-ingest tests (which patch it).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_repo_tree_yields_local_dir_unchanged(tmp_path):
    # A local path is not a GitHub URL, so it is yielded as-is — no clone.
    async with async_repo_tree(str(tmp_path)) as p:
        assert p == tmp_path


@pytest.mark.asyncio
async def test_async_repo_tree_missing_local_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        async with async_repo_tree(str(tmp_path / "does-not-exist")):
            pass


@pytest.mark.asyncio
async def test_async_repo_tree_clone_timeout_survives_processlookuperror(monkeypatch):
    """On clone timeout, proc.kill() can race the process exiting and raise
    ProcessLookupError; the timeout path must still surface a clean
    RepoTreeError, not leak ProcessLookupError to callers."""
    import researcher.util as util
    from researcher.util import RepoTreeError

    class _FakeProc:
        returncode = None

        async def communicate(self):  # pragma: no cover — wait_for preempts it
            return (b"", b"")

        def kill(self):
            raise ProcessLookupError()  # exited between timeout and kill

        async def wait(self):
            return None

    async def _fake_exec(*args, **kwargs):
        return _FakeProc()

    async def _immediate_timeout(coro, timeout):
        coro.close()  # avoid "coroutine was never awaited"
        raise asyncio.TimeoutError()

    monkeypatch.setattr(util, "_github_repo", lambda s: ("o/r", "https://github.com/o/r.git"))
    monkeypatch.setattr(util.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(util.asyncio, "wait_for", _immediate_timeout)

    with pytest.raises(RepoTreeError, match="timed out"):
        async with async_repo_tree("https://github.com/o/r"):
            pass
