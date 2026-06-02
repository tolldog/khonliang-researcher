"""Direct test for ``pipeline.ingest_github_repo(..., progress_callback=...)``.

The agent-level tests in ``test_agent_ingest_async.py`` stub the
pipeline entirely and the ``ingest_jobs`` tests exercise a fake
worker. That left no test for the actual integration point — a typo
in the keyword name (``progress_callback`` → ``progress_cb``) or a
missed phase name would slip through. This file pins the contract:

  - URL validation runs BEFORE the first progress event, so an
    invalid-URL rejection short-circuits without any callback fire.
  - When validation passes and the clone is reached, ``cloning``
    fires before the clone-failure path can return an error dict
    (a clone-failure callback recorder sees ``["cloning"]``, not
    ``[]``).
  - The kwarg name is exactly ``progress_callback``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_ingest_github_repo_invalid_url_short_circuits_before_emit():
    """An invalid URL is rejected by ``github_repo_key`` BEFORE the
    first progress event fires. This is observable: a callback that
    records every phase should still see no calls when the URL
    fails validation."""
    from researcher.pipeline import ResearchPipeline

    seen: list = []

    async def callback(phase: str, progress_pct: int) -> None:
        seen.append((phase, progress_pct))

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    # Don't run __init__ — we only exercise the URL-validation
    # branch, which doesn't touch self.knowledge / self.pool / etc.
    result = await ResearchPipeline.ingest_github_repo(
        pipe, repo_url="not-actually-a-url", progress_callback=callback,
    )
    assert "error" in result
    assert seen == []  # no progress events fired before the error


@pytest.mark.asyncio
async def test_ingest_github_repo_invokes_progress_callback_with_keyword_args():
    """When the URL parses, ``ingest_github_repo`` calls the callback
    with the documented ``phase=`` / ``progress_pct=`` kwargs at the
    ``cloning`` boundary. Asserts both the kwarg names and the
    initial phase name match the contract the agent's ingest_jobs
    wrapper depends on. Stops at the first call by raising from
    ``repo_tree`` (we only need to verify the integration shape, not
    drive a real clone)."""
    from researcher.pipeline import ResearchPipeline
    from researcher.util import RepoTreeError

    seen: list = []

    async def callback(**kwargs) -> None:
        # Capture the FULL kwarg dict so a renamed param shows up.
        seen.append(dict(kwargs))

    pipe = ResearchPipeline.__new__(ResearchPipeline)

    class _FakeCtx:
        async def __aenter__(self):
            raise RepoTreeError("simulated clone abort for test")
        async def __aexit__(self, *exc):
            return False

    with patch("researcher.util.github_repo_key", return_value="o/r"):
        with patch("researcher.util.async_repo_tree", return_value=_FakeCtx()):
            try:
                await ResearchPipeline.ingest_github_repo(
                    pipe,
                    repo_url="https://github.com/o/r",
                    progress_callback=callback,
                )
            except RepoTreeError:
                pass  # expected — we deliberately abort the clone
            except Exception:
                # Any other exception is also fine: the test only
                # cares that the callback was invoked once before
                # the abort.
                pass

    assert len(seen) == 1, f"expected exactly one progress call, got {seen!r}"
    assert seen[0] == {"phase": "cloning", "progress_pct": 10}


@pytest.mark.asyncio
async def test_ingest_github_repo_stores_research_scope(tmp_path):
    """Regression for bug_researcher_0be22a09: GitHub-ingested entries must be
    stored under scope='research' so Pipeline.search — the retrieval path behind
    find_relevant and brief_on — actually reaches them. The prior scope='external'
    left them in the knowledge table but invisible to retrieval (the search
    filters to scope=research/global).

    Drives the real ``ingest_github_repo`` with ``depth='readme'`` against an
    empty repo dir (no README, no AST scan → no LLM calls), stubbing only the
    store/triple/digest/score collaborators, and asserts the entry handed to
    ``knowledge.add`` carries scope='research'."""
    from contextlib import asynccontextmanager

    from researcher.pipeline import ResearchPipeline

    captured: dict = {}

    class _FakeKnowledge:
        def add(self, entry):
            captured["entry"] = entry

    class _FakeTriples:
        def add(self, **kwargs):
            pass

    class _FakeDigest:
        def record(self, **kwargs):
            pass

    async def _fake_score(entry_id):
        return {}

    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe.knowledge = _FakeKnowledge()
    pipe.triples = _FakeTriples()
    pipe.digest = _FakeDigest()
    pipe.score_relevance = _fake_score
    pipe._extract_package_metadata = lambda repo_path: {
        "description": "", "dependencies": [], "entry_points": [], "mcp_tools": [],
    }

    @asynccontextmanager
    async def _fake_repo_tree(url, prefix=""):
        yield tmp_path  # empty dir → no README, no LLM calls on the readme path

    with patch("researcher.util.github_repo_key", return_value="o/r"), \
         patch("researcher.util.async_repo_tree", _fake_repo_tree):
        result = await ResearchPipeline.ingest_github_repo(
            pipe, repo_url="https://github.com/o/r", depth="readme",
        )

    assert "error" not in result, result
    entry = captured.get("entry")
    assert entry is not None, "knowledge.add was never called"
    assert entry.scope == "research", (
        f"GitHub entries must be scope='research' to be retrievable via "
        f"find_relevant/brief_on; got {entry.scope!r}"
    )
    # external-origin distinction must still be preserved out-of-band.
    assert "external" in entry.tags
