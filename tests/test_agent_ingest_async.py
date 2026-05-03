"""Agent-level integration tests for the ingest_*_async surface.

JobStore + run_ingest_job have unit coverage in test_ingest_jobs.py;
this file exercises the *handlers* registered onto the bus agent —
the spawn → status → terminal flow that an MCP caller actually
sees. Critically covers the ``{"error": ...}`` translation that
``pipeline.ingest_github_repo`` does instead of raising, so a failed
ingest surfaces as ``phase=error`` rather than ``phase=done``.

Surfaced by Copilot review on PR #37 ("nothing here exercises the
agent-level async workflow end-to-end").
"""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from typing import Any

import pytest


def _build_fake_agent(pipeline_stub: Any) -> Any:
    """Minimal stand-in for a ``BaseAgent`` that satisfies what
    ``_extend_with_native_handlers`` reaches into.

    Real ``BaseAgent`` plumbs WebSocket, registration, etc.; for these
    tests we only need ``register_skills`` (called by the wrapper),
    ``_handlers`` (populated by the wrapper), and ``publish`` (called
    by the worker driver to emit events). Everything else is irrelevant
    because the test never starts the agent.
    """
    from researcher.agent import _extend_with_native_handlers

    captured_events: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        captured_events.append((topic, dict(payload)))

    agent = SimpleNamespace(
        agent_id="test-researcher",
        agent_type="researcher",
        version="0.0.0-test",
        bus_url="http://test",
        config_path="/tmp/test.yaml",
        _handlers={},
        register_skills=lambda: [],  # _extend wraps this; original returns nothing
        register_collaborations=lambda: [],
    )
    agent.publish = MethodType(publish, agent)
    agent.events = captured_events  # convenience for assertions
    _extend_with_native_handlers(agent, pipeline_stub)
    return agent


def _make_pipeline(stub_ingest_github_repo=None, stub_ingest_idea=None) -> Any:
    """Pipeline stub with the surface the async handlers reach into.

    Defaults: ``ingest_github_repo`` returns a happy result; the
    individual tests override either function to inject failures or
    measure call shape.
    """
    async def default_ingest_github_repo(
        repo_url, label="", depth="readme+code", progress_callback=None,
    ):
        if progress_callback is not None:
            await progress_callback(phase="cloning", progress_pct=10)
            await progress_callback(phase="ast_scanning", progress_pct=35)
            await progress_callback(phase="storing", progress_pct=75)
        return {
            "repo": "owner/repo",
            "entry_id": "ghrepo_abc",
            "capabilities": ["a", "b"],
            "depth": depth,
        }

    async def default_ingest_idea(text, source_label=""):
        return "idea_test"

    return SimpleNamespace(
        config={},  # ingest_async_concurrency falls back to default 4
        ingest_github_repo=stub_ingest_github_repo or default_ingest_github_repo,
        ingest_idea=stub_ingest_idea or default_ingest_idea,
    )


async def _await_terminal(agent, job_id: str, *, timeout: float = 3.0) -> dict:
    """Poll ingest_status until phase ∈ {done, error} or timeout.

    Mirrors how a polling-only caller (no event subscription) would
    drive the surface — exercises the race-free contract documented
    on the ingest_status skill.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        status = await agent._handlers["ingest_status"]({"job_id": job_id})
        if status.get("phase") in ("done", "error"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"job {job_id} did not reach terminal phase within {timeout}s; "
        f"last status={status!r}"
    )


# ---------------------------------------------------------------------------
# ingest_github_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_github_async_happy_path_reaches_done_with_result():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)

    accepted = await agent._handlers["ingest_github_async"]({
        "repo_url": "https://github.com/owner/repo",
    })
    assert "job_id" in accepted
    assert accepted["skill"] == "ingest_github"
    assert accepted["accepted_at"] > 0

    status = await _await_terminal(agent, accepted["job_id"])
    assert status["phase"] == "done"
    assert status["error"] is None
    assert status["result"]["entry_id"] == "ghrepo_abc"
    # Phase history must include the pipeline-emitted phases plus the
    # wrapper's started/done bookends.
    phases = [h["phase"] for h in status["history"]]
    assert phases[0] == "started"
    assert phases[-1] == "done"
    assert "cloning" in phases
    assert "ast_scanning" in phases
    assert "storing" in phases


@pytest.mark.asyncio
async def test_github_async_pipeline_error_dict_becomes_phase_error():
    """``pipeline.ingest_github_repo`` reports invalid URLs / clone
    failures by returning ``{"error": "..."}`` instead of raising.
    The handler must translate that into an exception so the worker
    surfaces ``phase=error`` (not ``phase=done`` with no error).
    Closes the most semantically important Copilot finding on PR #37.
    """
    async def failing_ingest(repo_url, label="", depth="readme+code", progress_callback=None):
        return {"error": f"Invalid GitHub URL: {repo_url}"}

    pipeline = _make_pipeline(stub_ingest_github_repo=failing_ingest)
    agent = _build_fake_agent(pipeline)

    accepted = await agent._handlers["ingest_github_async"]({
        "repo_url": "not-actually-a-url",
    })
    status = await _await_terminal(agent, accepted["job_id"])
    assert status["phase"] == "error"
    assert "Invalid GitHub URL" in status["error"]
    # ``result`` must remain None — the caller can rely on (phase ==
    # "done") implying ``result`` is non-null.
    assert status["result"] is None


@pytest.mark.asyncio
async def test_github_async_invalid_depth_rejected_at_api_boundary():
    """A typo or surrounding whitespace in ``depth`` must NOT silently
    degrade to README-only ingest while still reporting back the
    caller's invalid value as if it were honoured. Validate before
    spawn, return ``{"error": ...}`` synchronously.

    Stronger assertion: ``ingest_status`` returning ``not found`` for
    a probe id is true regardless of whether a job was secretly
    spawned, so it's a weak proof. Instead, also assert no progress
    events were ever published (a spawned job would emit ``started``
    on the same event loop tick) and a follow-up valid call gets a
    fresh job_id that DOES show up via ingest_status — proving the
    JobStore is unpolluted by the rejected attempt."""
    pipeline = _make_pipeline()
    pipeline_calls = {"n": 0}
    original = pipeline.ingest_github_repo

    async def counting_ingest(*args, **kw):
        pipeline_calls["n"] += 1
        return await original(*args, **kw)

    pipeline.ingest_github_repo = counting_ingest
    agent = _build_fake_agent(pipeline)

    out = await agent._handlers["ingest_github_async"]({
        "repo_url": "https://github.com/owner/repo",
        "depth": "readmecode",  # typo — missing the '+'
    })
    assert "error" in out
    assert "invalid depth" in out["error"]
    # Yield to the event loop in case anything scheduled a task —
    # nothing should have, but this gives any phantom worker a tick
    # to publish a started event so the assertion below catches it.
    await asyncio.sleep(0.01)
    # No worker ran: pipeline never invoked, no progress events fired.
    assert pipeline_calls["n"] == 0
    assert agent.events == [], (
        f"unexpected events after invalid-depth rejection: {agent.events!r}"
    )

    # Sanity: a follow-up valid call DOES spawn and reach the
    # JobStore — proving the rejection didn't break the normal path.
    valid = await agent._handlers["ingest_github_async"]({
        "repo_url": "https://github.com/owner/repo",
    })
    status = await _await_terminal(agent, valid["job_id"])
    assert status["phase"] == "done"


@pytest.mark.asyncio
async def test_github_async_repo_url_required():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)
    out = await agent._handlers["ingest_github_async"]({})
    assert out == {"error": "repo_url is required"}


# ---------------------------------------------------------------------------
# ingest_idea_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idea_async_happy_path_reaches_done_with_idea_id():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)

    accepted = await agent._handlers["ingest_idea_async"]({
        "text": "An informal claim about retrieval.",
        "source_label": "demo",
    })
    status = await _await_terminal(agent, accepted["job_id"])
    assert status["phase"] == "done"
    assert status["result"]["idea_id"] == "idea_test"
    # The worker emits its own distilling/storing phases between the
    # wrapper's started/done.
    phases = [h["phase"] for h in status["history"]]
    assert phases == ["started", "distilling", "storing", "done"]


@pytest.mark.asyncio
async def test_idea_async_text_required():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)
    out = await agent._handlers["ingest_idea_async"]({"text": "   "})
    assert out == {"error": "text is required"}


# ---------------------------------------------------------------------------
# ingest_file_async (file-specific behavior — fetch errors, empty content)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_async_path_required():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)
    out = await agent._handlers["ingest_file_async"]({"path": "  "})
    assert out == {"error": "path is required"}


@pytest.mark.asyncio
async def test_file_async_empty_content_becomes_phase_error(monkeypatch):
    """``fetch_file`` returning empty text means there's nothing to
    store and no entry_id to hand back. Without the explicit raise,
    ``run_ingest_job`` would mark the job ``phase=done`` with a
    warning dict, misleading the caller into thinking the ingest
    succeeded. Closes the Copilot finding on PR #37 pass 2."""
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)

    class _FakeFormat:
        value = "txt"

    class _FakeResult:
        content = ""  # the failure mode under test
        title = ""
        url = "file:///empty.txt"
        format = _FakeFormat()
        fetched_at = 0
        metadata = {}

    async def fake_fetch_file(path):
        return _FakeResult()

    # Stub the fetcher import in researcher.fetcher.
    import researcher.fetcher as fetcher_mod
    monkeypatch.setattr(fetcher_mod, "fetch_file", fake_fetch_file)

    accepted = await agent._handlers["ingest_file_async"]({"path": "/tmp/empty.txt"})
    status = await _await_terminal(agent, accepted["job_id"])
    assert status["phase"] == "error"
    assert "no text content extracted" in (status["error"] or "")
    assert status["result"] is None


# ---------------------------------------------------------------------------
# ingest_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_status_returns_not_found_for_unknown_job_id():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)
    out = await agent._handlers["ingest_status"]({"job_id": "job_does_not_exist"})
    assert out == {"error": "not found", "job_id": "job_does_not_exist"}


@pytest.mark.asyncio
async def test_ingest_status_job_id_required():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)
    out = await agent._handlers["ingest_status"]({})
    assert out == {"error": "job_id is required"}


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_async_jobs_respect_semaphore_cap():
    """Bound the number of concurrently-executing ingest workers.
    Concretely: with ``ingest_async_concurrency=2`` and four
    simultaneous ``ingest_github_async`` calls, no more than two
    workers run in parallel even though the JobStore holds all four.
    """
    in_flight = 0
    peak = 0
    enter = asyncio.Event()
    leave = asyncio.Event()

    async def slow_ingest(repo_url, label="", depth="readme+code", progress_callback=None):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        enter.set()  # signal the test that at least one worker started
        try:
            # Hold the slot until the test releases.
            await leave.wait()
            return {"repo": repo_url, "entry_id": "x"}
        finally:
            in_flight -= 1

    pipeline = _make_pipeline(stub_ingest_github_repo=slow_ingest)
    pipeline.config = {"ingest_async_concurrency": 2}
    agent = _build_fake_agent(pipeline)

    accepted = []
    for n in range(4):
        a = await agent._handlers["ingest_github_async"]({
            "repo_url": f"https://github.com/o/r{n}",
        })
        accepted.append(a)

    # Wait for at least one worker to enter, then give the loop time
    # for any others that fit under the cap to also enter.
    await asyncio.wait_for(enter.wait(), timeout=2.0)
    await asyncio.sleep(0.1)
    assert peak <= 2, f"peak in-flight workers exceeded cap: {peak}"
    assert peak >= 1

    leave.set()
    for a in accepted:
        await _await_terminal(agent, a["job_id"], timeout=3.0)
    # All four eventually run, but never more than 2 at once.
    assert peak <= 2


# ---------------------------------------------------------------------------
# Lifecycle: in-flight tasks are tracked and cancelled on shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_flight_jobs_are_cancelled_when_agent_tasks_are_cancelled():
    """Spawned ingest tasks are retained on the agent so a graceful
    shutdown can cancel them rather than leaving repo clones / LLM
    work running after the bus connector closes. ``run_ingest_job``
    translates the ``CancelledError`` into ``phase=error`` with the
    cancellation message, then re-raises so the asyncio runtime
    finalises the task properly."""
    leave = asyncio.Event()

    async def parking_ingest(repo_url, label="", depth="readme+code", progress_callback=None):
        await leave.wait()
        return {"repo": repo_url, "entry_id": "never-reached"}

    pipeline = _make_pipeline(stub_ingest_github_repo=parking_ingest)
    agent = _build_fake_agent(pipeline)

    accepted = await agent._handlers["ingest_github_async"]({
        "repo_url": "https://github.com/o/r",
    })
    # Let the worker enter ``leave.wait()``.
    await asyncio.sleep(0.05)
    tasks = list(agent._ingest_tasks)
    assert len(tasks) == 1
    assert not tasks[0].done()

    # Cancel all in-flight tasks (mimics what ``shutdown()`` does).
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass

    status = await agent._handlers["ingest_status"]({"job_id": accepted["job_id"]})
    assert status["phase"] == "error"
    assert "Cancelled" in (status["error"] or "")


@pytest.mark.asyncio
async def test_queued_jobs_are_cancelled_correctly_when_blocked_on_semaphore():
    """A job cancelled WHILE STILL QUEUED on the concurrency
    semaphore (i.e. ``run_ingest_job`` never entered) used to stay
    in ``phase=accepted`` forever — ``ingest_status`` callers would
    poll indefinitely. The driver's ``except CancelledError`` branch
    now records the cancel directly on the JobRecord.

    Setup: cap concurrency at 1, spawn two jobs (job A runs, job B
    queues on the semaphore), cancel B's task, verify B reaches
    ``phase=error`` with the cancellation message.
    """
    leave_a = asyncio.Event()

    async def parking_ingest(repo_url, label="", depth="readme+code", progress_callback=None):
        # Job A holds the only slot until the test releases it.
        await leave_a.wait()
        return {"repo": repo_url, "entry_id": "a"}

    pipeline = _make_pipeline(stub_ingest_github_repo=parking_ingest)
    pipeline.config = {"ingest_async_concurrency": 1}
    agent = _build_fake_agent(pipeline)

    a = await agent._handlers["ingest_github_async"]({"repo_url": "https://github.com/o/a"})
    b = await agent._handlers["ingest_github_async"]({"repo_url": "https://github.com/o/b"})

    # Let A enter the semaphore (it parks on leave_a). B is queued.
    await asyncio.sleep(0.05)
    a_status = await agent._handlers["ingest_status"]({"job_id": a["job_id"]})
    b_status = await agent._handlers["ingest_status"]({"job_id": b["job_id"]})
    assert a_status["phase"] in ("started", "cloning", "ast_scanning", "storing")
    assert b_status["phase"] == "accepted"

    # Find B's task and cancel it. Because B is queued on the
    # semaphore, run_ingest_job has not entered.
    tasks = {t.get_name(): t for t in agent._ingest_tasks}
    b_task = tasks[f"ingest-job-{b['job_id']}"]
    b_task.cancel()
    try:
        await b_task
    except asyncio.CancelledError:
        pass

    # B should now report phase=error with a cancellation message,
    # not phase=accepted (which would be the bug).
    b_final = await agent._handlers["ingest_status"]({"job_id": b["job_id"]})
    assert b_final["phase"] == "error"
    assert "cancelled before start" in (b_final["error"] or "").lower()

    # A still finishes cleanly when released, proving the cancel
    # didn't disturb the queue mechanics.
    leave_a.set()
    a_final = await _await_terminal(agent, a["job_id"], timeout=2.0)
    assert a_final["phase"] == "done"


# ---------------------------------------------------------------------------
# Bus events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_events_published_on_research_ingest_progress_topic():
    pipeline = _make_pipeline()
    agent = _build_fake_agent(pipeline)
    accepted = await agent._handlers["ingest_idea_async"]({"text": "x"})
    await _await_terminal(agent, accepted["job_id"])
    topics = {topic for topic, _ in agent.events}
    assert topics == {"research.ingest.progress"}
    phases = [p["phase"] for _, p in agent.events]
    assert phases[0] == "started"
    assert phases[-1] == "done"
    # All events tagged with the originating job_id and skill so
    # subscribers can filter their attention.
    job_id = accepted["job_id"]
    for _, p in agent.events:
        assert p["job_id"] == job_id
        assert p["skill"] == "ingest_idea"
