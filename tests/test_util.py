"""Tests for researcher utility helpers."""

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
