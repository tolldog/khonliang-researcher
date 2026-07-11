"""Tests for researcher-as-librarian-consumer (fr_researcher_11e9524a).

Exercises ``researcher.librarian_client.call_librarian`` (and the 7
convenience wrappers) plus ``researcher.agent.ask_librarian`` as
module-level coroutines, mirroring ``test_store_integration.py``'s
pattern for ``stage_payload``/``ingest_from_artifact`` — a mock agent
stands in for ``BaseAgent`` so no live bus / librarian-primary is
required.
"""

from __future__ import annotations

from typing import Any

import pytest

from researcher import librarian_client
from researcher.agent import ask_librarian


class _MockAgent:
    """Records ``request()`` calls; returns a canned response or raises."""

    def __init__(self, response: Any = None, *, raise_exc: Exception | None = None) -> None:
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def request(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


# -- call_librarian: success path ---------------------------------------------


@pytest.mark.asyncio
async def test_call_librarian_routes_to_agent_type_librarian():
    agent = _MockAgent(response={"result": {"status": "classified", "record": {}}})

    result = await librarian_client.call_librarian(
        agent, "classify_paper", {"paper_id": "p1"},
    )

    assert agent.calls == [
        {
            "agent_type": "librarian",
            "operation": "classify_paper",
            "args": {"paper_id": "p1"},
            "timeout": librarian_client.DEFAULT_TIMEOUT,
        }
    ]
    assert result["available"] is True
    assert result["status"] == "classified"


@pytest.mark.asyncio
async def test_call_librarian_accepts_unwrapped_result_shape():
    """Some ``request`` mocks / bus shapes return the raw handler result
    (no ``{"result": ...}`` envelope) — tolerate both, like
    ``_unwrap_request_envelope`` does for the store path."""
    agent = _MockAgent(response={"detail": "brief", "groups": []})

    result = await librarian_client.call_librarian(agent, "taxonomy_report")

    assert result == {"available": True, "detail": "brief", "groups": []}


# -- call_librarian: graceful degradation -------------------------------------


@pytest.mark.asyncio
async def test_call_librarian_unknown_operation_short_circuits_without_a_call():
    agent = _MockAgent(response={"result": {}})

    result = await librarian_client.call_librarian(agent, "delete_everything")

    assert result["available"] is False
    assert "unknown librarian operation" in result["reason"]
    assert agent.calls == []


@pytest.mark.asyncio
async def test_call_librarian_swallows_transport_exceptions():
    agent = _MockAgent(raise_exc=TimeoutError("librarian-primary did not respond"))

    result = await librarian_client.call_librarian(agent, "library_health")

    assert result == {
        "available": False,
        "reason": "librarian unreachable: librarian-primary did not respond",
    }


@pytest.mark.asyncio
async def test_call_librarian_treats_bus_error_envelope_as_unavailable():
    """agent_type='librarian' with no agent registered (librarian-primary
    down) surfaces as a bus-level error envelope, not an exception —
    must still degrade gracefully rather than propagate the error."""
    agent = _MockAgent(response={"result": {"error": "no agent registered for type librarian"}})

    result = await librarian_client.call_librarian(agent, "identify_gaps")

    assert result == {
        "available": False,
        "reason": "no agent registered for type librarian",
    }


@pytest.mark.asyncio
async def test_call_librarian_rejects_unexpected_response_shape():
    agent = _MockAgent(response={"result": "not a dict"})

    result = await librarian_client.call_librarian(agent, "library_health")

    assert result["available"] is False
    assert "unexpected response shape" in result["reason"]


# -- convenience wrappers ------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_paper_wrapper_forwards_expected_args():
    agent = _MockAgent(response={"result": {"status": "classified"}})

    await librarian_client.classify_paper(agent, "p1", audience="developer")

    assert agent.calls[0]["operation"] == "classify_paper"
    assert agent.calls[0]["args"] == {
        "paper_id": "p1",
        "audience": "developer",
        "detail": "brief",
    }


@pytest.mark.asyncio
async def test_library_health_wrapper_degrades_when_unreachable():
    agent = _MockAgent(raise_exc=ConnectionRefusedError("no route to bus"))

    result = await librarian_client.library_health(agent)

    assert result["available"] is False
    assert "librarian unreachable" in result["reason"]


# -- ask_librarian bus-skill handler -------------------------------------------


@pytest.mark.asyncio
async def test_ask_librarian_requires_operation():
    agent = _MockAgent(response={"result": {}})

    result = await ask_librarian(agent, {})

    assert result == {"error": "operation is required"}
    assert agent.calls == []


@pytest.mark.asyncio
async def test_ask_librarian_rejects_non_dict_librarian_args():
    agent = _MockAgent(response={"result": {}})

    result = await ask_librarian(
        agent, {"operation": "classify_paper", "librarian_args": "nope"},
    )

    assert result == {"error": "librarian_args must be an object"}
    assert agent.calls == []


@pytest.mark.asyncio
async def test_ask_librarian_proxies_to_call_librarian():
    agent = _MockAgent(response={"result": {"detail": "brief", "groups": []}})

    result = await ask_librarian(
        agent, {"operation": "taxonomy_report", "librarian_args": {"branch": "ml"}},
    )

    assert agent.calls == [
        {
            "agent_type": "librarian",
            "operation": "taxonomy_report",
            "args": {"branch": "ml"},
            "timeout": librarian_client.DEFAULT_TIMEOUT,
        }
    ]
    assert result == {"available": True, "detail": "brief", "groups": []}


@pytest.mark.asyncio
async def test_ask_librarian_degrades_gracefully_when_librarian_down():
    agent = _MockAgent(raise_exc=TimeoutError("timed out"))

    result = await ask_librarian(agent, {"operation": "library_health"})

    assert result["available"] is False
    assert "librarian unreachable" in result["reason"]
