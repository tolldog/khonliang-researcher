"""Tests for the researcher's store-artifact integration skills.

Phase 5 of the khonliang-store roadmap (`fr_researcher_546c1308`).
Exercises ``stage_payload`` and ``ingest_from_artifact`` as
module-level coroutines so they can be tested without standing up
the full ``BaseAgent.from_mcp`` pipeline. The handlers wired into
the bus agent are thin shims over these.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from researcher.agent import ingest_from_artifact, stage_payload


class _MockAgent:
    """Minimal agent stub: records ``request()`` calls + returns a canned response.

    Exposes ``agent_id`` because :func:`stage_payload` reads it for
    the artifact's ``producer`` field. Async ``request`` matches
    ``BaseAgent.request`` shape (kwargs only).
    """

    def __init__(self, agent_id: str = "researcher-test", response: Any = None) -> None:
        self.agent_id = agent_id
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def request(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        return self.response


class _MockPipeline:
    """Stub pipeline whose ``ingest_idea`` returns a canned id.

    Records the ``(text, source_label)`` it was called with so
    tests can assert the right bytes were threaded through.
    """

    def __init__(self, idea_id: str = "idea_test", *, raise_runtime: Optional[str] = None) -> None:
        self.idea_id = idea_id
        self.raise_runtime = raise_runtime
        self.calls: list[tuple[str, str]] = []

    async def ingest_idea(self, text: str, source_label: str) -> str:
        self.calls.append((text, source_label))
        if self.raise_runtime is not None:
            raise RuntimeError(self.raise_runtime)
        return self.idea_id


# -- stage_payload ------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_payload_routes_to_store_artifact_create():
    """Happy path: serializes the request envelope correctly,
    returns the artifact_id from the store response.
    """
    agent = _MockAgent(
        response={"result": {"id": "art_xyz", "kind": "staged_payload"}},
    )
    result = await stage_payload(agent, {
        "content": "hello world",
        "kind_hint": "github-url",
        "source": {"url": "http://example.com", "fetcher": "test"},
    })
    assert result == {"artifact_id": "art_xyz"}
    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert call["agent_type"] == "store"
    assert call["operation"] == "artifact_create"
    inner = call["args"]
    assert inner["kind"] == "staged_payload"
    assert inner["content"] == "hello world"
    assert inner["producer"] == "researcher-test"
    # Provenance + kind_hint flow through metadata so a future
    # dispatcher can read them without us defining new fields.
    assert inner["metadata"] == {
        "source": {"url": "http://example.com", "fetcher": "test"},
        "kind_hint": "github-url",
    }


@pytest.mark.asyncio
async def test_stage_payload_generates_default_title_from_content():
    """Empty title field → preview-of-first-line capped at 80
    *total* chars (79 raw + ellipsis). The previous ``[:80] +
    "…"`` overshot the documented limit by one.
    """
    agent = _MockAgent(response={"result": {"id": "art_a"}})
    long_first_line = "x" * 200
    body = long_first_line + "\nsecond line"
    await stage_payload(agent, {"content": body})
    title = agent.calls[0]["args"]["title"]
    assert title == ("x" * 79) + "…"
    assert len(title) == 80


@pytest.mark.asyncio
async def test_stage_payload_skips_leading_blank_lines_for_default_title():
    """Markdown content often opens with a blank line before
    the heading. ``partition("\\n")`` would have taken the
    empty first line and produced ``"staged payload"`` as the
    fallback title, hiding the real first content. The current
    loop walks until it finds a non-empty stripped line.
    """
    agent = _MockAgent(response={"result": {"id": "art_a"}})
    body = "\n\n# Real Title\n\nbody text"
    await stage_payload(agent, {"content": body})
    assert agent.calls[0]["args"]["title"] == "# Real Title"


@pytest.mark.asyncio
async def test_stage_payload_uses_explicit_title_when_provided():
    agent = _MockAgent(response={"result": {"id": "art_a"}})
    await stage_payload(agent, {
        "content": "anything",
        "title": "  My Important Doc  ",
    })
    assert agent.calls[0]["args"]["title"] == "My Important Doc"


@pytest.mark.asyncio
async def test_stage_payload_rejects_non_string_content():
    agent = _MockAgent()
    result = await stage_payload(agent, {"content": 42})
    assert result == {"error": "content must be a string"}
    # Bad input must NOT round-trip to store.
    assert agent.calls == []


@pytest.mark.asyncio
async def test_stage_payload_rejects_empty_content():
    agent = _MockAgent()
    result = await stage_payload(agent, {"content": ""})
    assert result == {"error": "content is required"}
    assert agent.calls == []


@pytest.mark.asyncio
async def test_stage_payload_rejects_non_dict_source():
    """``source`` is documented as a provenance dict; falsey
    invalids (e.g. ``[]`` or ``"string"``) used to coerce
    silently when ``args.get("source") or {}`` ate the type
    check. ``"source" in args`` keeps the isinstance check
    honest.
    """
    agent = _MockAgent()
    result = await stage_payload(agent, {
        "content": "x", "source": "not-a-dict",
    })
    assert result == {"error": "source must be an object"}
    assert agent.calls == []


@pytest.mark.asyncio
async def test_stage_payload_passes_through_store_error_envelope():
    """A store-side error envelope (size cap, duplicate id,
    sqlite failure) flows back to the caller verbatim — same
    shape they'd see calling artifact_create directly.
    """
    agent = _MockAgent(
        response={"result": {"error": "content exceeds maximum size"}},
    )
    result = await stage_payload(agent, {"content": "x"})
    assert result == {"error": "content exceeds maximum size"}


@pytest.mark.asyncio
async def test_stage_payload_handles_missing_id_in_response():
    """Defensive: a store that returns ``{}`` (without an
    ``id`` and without an ``error``) shouldn't claim success.
    """
    agent = _MockAgent(response={"result": {"kind": "staged_payload"}})
    result = await stage_payload(agent, {"content": "x"})
    assert result == {"error": "store created artifact without id"}


# -- ingest_from_artifact -----------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_from_artifact_pulls_body_then_dispatches_to_pipeline():
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_x", "producer": "fetcher-a"},
        "text": "raw payload bytes",
    }})
    pipeline = _MockPipeline(idea_id="idea_42")
    result = await ingest_from_artifact(agent, pipeline, {
        "artifact_id": "art_x",
        "hints": {"force_type": "github-url"},
    })
    assert result == {
        "idea_id": "idea_42",
        "artifact_id": "art_x",
        "source_label": "fetcher-a",
        "hints": {"force_type": "github-url"},
    }
    # Pipeline was invoked with the artifact's bytes; the
    # source_label fell back to the artifact's ``producer``.
    assert pipeline.calls == [("raw payload bytes", "fetcher-a")]
    # Store fetch went through agent_type=store, operation=artifact_get.
    fetch = agent.calls[0]
    assert fetch["agent_type"] == "store"
    assert fetch["operation"] == "artifact_get"
    assert fetch["args"]["id"] == "art_x"


@pytest.mark.asyncio
async def test_ingest_from_artifact_explicit_source_label_wins():
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_x", "producer": "fetcher-default"},
        "text": "x",
    }})
    pipeline = _MockPipeline()
    await ingest_from_artifact(agent, pipeline, {
        "artifact_id": "art_x",
        "source_label": "operator-override",
    })
    # Override beats the artifact's producer field.
    assert pipeline.calls[0][1] == "operator-override"


@pytest.mark.asyncio
async def test_ingest_from_artifact_requires_artifact_id():
    agent = _MockAgent()
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(agent, pipeline, {})
    assert result == {"error": "artifact_id is required"}
    # Bad input doesn't reach the store or the pipeline.
    assert agent.calls == []
    assert pipeline.calls == []


@pytest.mark.asyncio
async def test_ingest_from_artifact_rejects_non_dict_hints():
    agent = _MockAgent()
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(
        agent, pipeline, {"artifact_id": "art_x", "hints": "not-a-dict"},
    )
    assert result == {"error": "hints must be an object"}


@pytest.mark.asyncio
async def test_ingest_from_artifact_passes_through_store_not_found():
    """A missing artifact's error envelope flows back without
    being re-wrapped — caller sees the same ``"artifact not
    found"`` they'd get from a direct artifact_get.
    """
    agent = _MockAgent(response={"result": {"error": "artifact not found"}})
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(
        agent, pipeline, {"artifact_id": "art_missing"},
    )
    assert result == {"error": "artifact not found"}
    # Pipeline never reached.
    assert pipeline.calls == []


@pytest.mark.asyncio
async def test_ingest_from_artifact_rejects_truncated_response():
    """Store's REST surface caps reads at HARD_MAX_CHARS=20000;
    larger artifacts come back with ``truncated=True`` and a
    partial body. Ingesting that would produce an idea whose
    claims/queries don't reflect the full source — surface a
    clear error so the caller can wait on streaming support
    rather than silently partially-ingesting.
    """
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_huge"},
        "text": "first 20K of a much larger artifact",
        "truncated": True,
    }})
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(
        agent, pipeline, {"artifact_id": "art_huge"},
    )
    assert "error" in result
    assert "truncated" in result["error"]
    # Pipeline never reached — better to fail loudly than
    # ingest a partial artifact and lie about completeness.
    assert pipeline.calls == []


@pytest.mark.asyncio
async def test_ingest_from_artifact_handles_empty_content():
    """A successful fetch with zero text is still useless to the
    pipeline — surface a clean error rather than threading
    empty bytes through ingest_idea.
    """
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_empty"},
        "text": "",
    }})
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(
        agent, pipeline, {"artifact_id": "art_empty"},
    )
    assert result == {"error": "store returned empty content"}


@pytest.mark.asyncio
async def test_ingest_from_artifact_surfaces_pipeline_failure():
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_x"},
        "text": "ok content",
    }})
    pipeline = _MockPipeline(raise_runtime="LLM down")
    result = await ingest_from_artifact(
        agent, pipeline, {"artifact_id": "art_x"},
    )
    assert result == {"error": "ingest failed: LLM down"}
