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
async def test_stage_payload_rejects_missing_content():
    """Missing key vs. wrong type are distinct errors so callers
    don't get "content must be a string" when they simply forgot
    to pass the field. Matches ``artifact_id is required`` in
    ``ingest_from_artifact`` for cross-skill consistency.
    """
    agent = _MockAgent()
    result = await stage_payload(agent, {})
    assert result == {"error": "content is required"}
    assert agent.calls == []


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
async def test_stage_payload_rejects_whitespace_only_content():
    """``"\\n\\n"`` / ``"   "`` shouldn't slip past the
    "content is required" check just because the bytes are
    technically non-empty — store would persist a useless
    artifact and the dispatcher would later choke on it.
    """
    agent = _MockAgent()
    result = await stage_payload(agent, {"content": "\n\n   \t  \n"})
    assert result == {"error": "content is required"}
    assert agent.calls == []


@pytest.mark.asyncio
async def test_stage_payload_rejects_non_string_kind_hint():
    """``kind_hint`` is metadata that the future dispatcher
    reads as a string. Silently coercing a number/object via
    ``str()`` would persist the repr() into the artifact and
    hide the caller bug behind a confusing dispatch later.
    """
    agent = _MockAgent()
    result = await stage_payload(agent, {"content": "x", "kind_hint": 42})
    assert result == {"error": "kind_hint must be a string"}
    assert agent.calls == []


@pytest.mark.asyncio
async def test_stage_payload_rejects_non_string_title():
    agent = _MockAgent()
    result = await stage_payload(agent, {"content": "x", "title": ["a", "b"]})
    assert result == {"error": "title must be a string"}
    assert agent.calls == []


@pytest.mark.asyncio
async def test_stage_payload_normalizes_whitespace_content_type():
    """``'  text/markdown  '`` is the same MIME type as
    ``'text/markdown'`` once you strip it; ``'   '`` (or
    ``''``) is no MIME type at all and should fall back to
    text/plain rather than reach store as invalid metadata.
    """
    agent = _MockAgent(response={"result": {"id": "art_a"}})
    await stage_payload(agent, {
        "content": "x",
        "content_type": "  text/markdown  ",
    })
    assert agent.calls[0]["args"]["content_type"] == "text/markdown"

    agent2 = _MockAgent(response={"result": {"id": "art_b"}})
    await stage_payload(agent2, {"content": "x", "content_type": "   "})
    assert agent2.calls[0]["args"]["content_type"] == "text/plain"


@pytest.mark.asyncio
async def test_stage_payload_rejects_non_string_content_type():
    agent = _MockAgent()
    result = await stage_payload(
        agent, {"content": "x", "content_type": {"mime": "text/plain"}},
    )
    assert result == {"error": "content_type must be a string"}
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
async def test_stage_payload_accepts_nested_artifact_id_shape():
    """Some artifact-create surfaces emit
    ``{"artifact": {"id": ...}}`` instead of the flat
    ``{"id": ...}`` shape (see LibrarianAgent._artifact_id for
    the same-repo precedent). Tolerate both so a future store
    surface tweak doesn't break the integration.
    """
    agent = _MockAgent(
        response={"result": {"artifact": {"id": "art_nested", "kind": "staged_payload"}}},
    )
    result = await stage_payload(agent, {"content": "x"})
    assert result == {"artifact_id": "art_nested"}


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
    # ``offset=0`` + ``max_chars=_INGEST_FETCH_CAP_CHARS`` (20000)
    # are load-bearing for the truncated-response guard: a
    # different ceiling here means a partial body could slip
    # through without ``truncated=True`` ever being set.
    assert fetch["args"]["offset"] == 0
    assert fetch["args"]["max_chars"] == 20_000


@pytest.mark.asyncio
async def test_ingest_from_artifact_accepts_content_field():
    """Bus's ``/v1/artifacts/{id}/content`` route returns the
    body under ``content``, not ``text``. Accept that shape so
    surface variation doesn't turn a successful fetch into
    "store returned empty content".
    """
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_x", "producer": "fetcher-a"},
        "content": "raw bytes via content field",
    }})
    pipeline = _MockPipeline(idea_id="idea_99")
    result = await ingest_from_artifact(agent, pipeline, {"artifact_id": "art_x"})
    assert result["idea_id"] == "idea_99"
    assert pipeline.calls == [("raw bytes via content field", "fetcher-a")]


@pytest.mark.asyncio
async def test_ingest_from_artifact_accepts_body_field():
    """Historical alias — kept for backwards compat with any
    callers that hand back ``{"body": ...}``.
    """
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_x"},
        "body": "raw bytes via body field",
    }})
    pipeline = _MockPipeline(idea_id="idea_legacy")
    result = await ingest_from_artifact(agent, pipeline, {"artifact_id": "art_x"})
    assert result["idea_id"] == "idea_legacy"
    assert pipeline.calls == [("raw bytes via body field", "")]


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
async def test_ingest_from_artifact_rejects_non_string_artifact_id():
    """A non-string ``artifact_id`` (int, None, dict) used to
    coerce via ``str()`` and produce a different lookup id —
    failing in store with a confusing not-found rather than
    fast at the boundary.
    """
    agent = _MockAgent()
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(
        agent, pipeline, {"artifact_id": 12345},
    )
    assert result == {"error": "artifact_id must be a string"}
    assert agent.calls == []
    assert pipeline.calls == []


@pytest.mark.asyncio
async def test_ingest_from_artifact_rejects_non_string_source_label():
    agent = _MockAgent()
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(
        agent, pipeline,
        {"artifact_id": "art_x", "source_label": {"name": "x"}},
    )
    assert result == {"error": "source_label must be a string"}
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
async def test_ingest_from_artifact_handles_whitespace_only_content():
    """An artifact whose body is just whitespace (page-break
    newlines from a poorly-extracted PDF, etc.) should surface
    as empty here — not slip into ``ingest_idea`` with garbage
    that fails further downstream with a less actionable error.
    """
    agent = _MockAgent(response={"result": {
        "artifact": {"id": "art_blank"},
        "text": "   \n\n\t  \n   ",
    }})
    pipeline = _MockPipeline()
    result = await ingest_from_artifact(
        agent, pipeline, {"artifact_id": "art_blank"},
    )
    assert result == {"error": "store returned empty content"}
    # Pipeline never reached.
    assert pipeline.calls == []


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
