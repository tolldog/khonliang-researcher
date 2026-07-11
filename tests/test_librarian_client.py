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
async def test_call_librarian_unwraps_the_bus_result_envelope():
    """A successful dispatch always wraps the handler's return value as
    {"result": ..., "trace_id": ...} (bus/server.py
    :_dispatch_resolved_request) — call_librarian must unwrap it rather
    than returning the trace_id/envelope noise to the caller."""
    agent = _MockAgent(response={"result": {"detail": "brief", "groups": []}, "trace_id": "t-abc"})

    result = await librarian_client.call_librarian(agent, "taxonomy_report")

    assert result == {"available": True, "detail": "brief", "groups": []}


# -- call_librarian: reached-but-rejected vs. genuinely unreachable -----------


@pytest.mark.asyncio
async def test_call_librarian_surfaces_domain_error_as_available_true():
    """Librarian responding with its OWN validation/domain error (e.g. a
    caller-side mistake like a missing paper_id) means librarian is UP —
    available must stay True, with the rejection surfaced separately from
    the availability signal (khonliang-researcher PR #75 review finding)."""
    agent = _MockAgent(
        response={"result": {"error": "paper_id is required"}, "trace_id": "t-1"},
    )

    result = await librarian_client.call_librarian(agent, "classify_paper", {})

    assert result == {"available": True, "error": "paper_id is required"}


@pytest.mark.asyncio
async def test_call_librarian_treats_bare_top_level_error_as_unreachable():
    """A bus-level dispatch failure (no healthy agent for the type, i.e.
    librarian-primary isn't registered) comes back as a bare top-level
    {"error": ..., "trace_id": ...} with NO "result" key — the request
    never reached a librarian handler at all. That must degrade to
    available: False, distinct from a domain error librarian itself
    returned."""
    agent = _MockAgent(response={"error": "no healthy agent found for librarian", "trace_id": "t-2"})

    result = await librarian_client.call_librarian(agent, "identify_gaps")

    assert result == {
        "available": False,
        "reason": "no healthy agent found for librarian",
    }


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
async def test_call_librarian_rejects_non_dict_top_level_response():
    agent = _MockAgent(response="not a dict")

    result = await librarian_client.call_librarian(agent, "library_health")

    assert result["available"] is False
    assert "unexpected response shape" in result["reason"]


@pytest.mark.asyncio
async def test_call_librarian_rejects_non_dict_result_payload():
    agent = _MockAgent(response={"result": "not a dict", "trace_id": "t-3"})

    result = await librarian_client.call_librarian(agent, "library_health")

    assert result["available"] is False
    assert "unexpected response shape" in result["reason"]


# -- call_librarian: per-operation timeouts ------------------------------------


@pytest.mark.asyncio
async def test_rebuild_neighborhoods_gets_a_longer_default_timeout():
    """rebuild_neighborhoods is a corpus-wide scan, not an interactive
    lookup — it must not inherit the 10s interactive default (that would
    report a healthy-but-busy librarian as unavailable)."""
    agent = _MockAgent(response={"result": {"snapshot_id": "libsnap_1"}, "trace_id": "t-4"})

    await librarian_client.call_librarian(agent, "rebuild_neighborhoods")

    assert agent.calls[0]["timeout"] == librarian_client.OPERATION_TIMEOUTS["rebuild_neighborhoods"]
    assert agent.calls[0]["timeout"] > librarian_client.DEFAULT_TIMEOUT


@pytest.mark.asyncio
async def test_identify_gaps_gets_a_longer_default_timeout():
    agent = _MockAgent(response={"result": {"gaps": []}, "trace_id": "t-5"})

    await librarian_client.call_librarian(agent, "identify_gaps")

    assert agent.calls[0]["timeout"] == librarian_client.OPERATION_TIMEOUTS["identify_gaps"]
    assert agent.calls[0]["timeout"] > librarian_client.DEFAULT_TIMEOUT


@pytest.mark.asyncio
async def test_classify_paper_keeps_the_short_interactive_default():
    agent = _MockAgent(response={"result": {}, "trace_id": "t-6"})

    await librarian_client.call_librarian(agent, "classify_paper", {"paper_id": "p1"})

    assert agent.calls[0]["timeout"] == librarian_client.DEFAULT_TIMEOUT


@pytest.mark.asyncio
async def test_explicit_timeout_overrides_the_per_operation_default():
    agent = _MockAgent(response={"result": {}, "trace_id": "t-7"})

    await librarian_client.call_librarian(agent, "rebuild_neighborhoods", timeout=5.0)

    assert agent.calls[0]["timeout"] == 5.0


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
async def test_ask_librarian_rejects_non_dict_args_without_crashing():
    """A caller passing args=None (or any non-dict) must get the repo's
    normal {"error": ...} envelope, not an AttributeError from args.get(...)
    (khonliang-researcher PR #75 review finding)."""
    agent = _MockAgent(response={"result": {}})

    result = await ask_librarian(agent, None)

    assert result == {"error": "args must be an object"}
    assert agent.calls == []

    result = await ask_librarian(agent, "not a dict")

    assert result == {"error": "args must be an object"}
    assert agent.calls == []


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
