"""Direct test for ``pipeline.ingest_github_repo(..., progress_callback=...)``.

The agent-level tests in ``test_agent_ingest_async.py`` stub the
pipeline entirely and the ``ingest_jobs`` tests exercise a fake
worker. That left no test for the actual integration point — a typo
in the keyword name (``progress_callback`` → ``progress_cb``) or a
missed phase name would slip through. This file pins the contract:
the pipeline emits ``cloning`` BEFORE the URL/clone error path can
fire, and the kwarg name is exactly ``progress_callback``.
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
        def __enter__(self):
            raise RepoTreeError("simulated clone abort for test")
        def __exit__(self, *exc):
            return False

    with patch("researcher.util.github_repo_key", return_value="o/r"):
        with patch("researcher.util.repo_tree", return_value=_FakeCtx()):
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
