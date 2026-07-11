"""Thin bus client for researcher-as-librarian-consumer (fr_researcher_11e9524a).

Post-split (fr_librarian_bc0a06d7 / fr_librarian_e78d9b10), librarian runs as
its own standalone bus agent (``librarian-primary``, repo
``khonliang-librarian``) — no longer co-resident inside researcher's process.
This module is the reroute: researcher calls librarian's 7 skills
(classify_paper, taxonomy_report, rebuild_neighborhoods,
suggest_missing_nodes, promote_investigation, identify_gaps, library_health)
over the bus, the same way ``researcher/agent.py``'s ``stage_payload`` /
``ingest_from_artifact`` already call ``agent_type="store"`` — reusing
``BaseAgent.request(agent_type=..., operation=..., args=...)`` rather than
inventing a new transport.

Graceful degradation (optional-coordinator principle —
project_knowledge_estate_synthesis_2026_07_02: "every capability reachable
via owner skills directly; librarian absence degrades quality not
function"): every call here is best-effort. Timeouts, connection failures,
and bus-level dispatch failures (e.g. librarian-primary not currently
registered) are caught and normalized into a non-fatal
``{"available": False, "reason": ...}`` envelope — never raised. Callers
must treat ``available: False`` as "skip librarian-dependent enrichment for
this request," not as a hard failure of whatever they were doing.

``available`` distinguishes *unreachable* from *reached-but-rejected*: a
transport exception, timeout, or a bus-level dispatch failure (no healthy
agent, connection refused, non-2xx callback — see
``bus/server.py:_dispatch_resolved_request`` in khonliang-bus, which always
wraps a real handler's return value as ``{"result": ..., "trace_id": ...}``
and only emits a bare top-level ``{"error": ..., "trace_id": ...}`` when
dispatch itself never reached the handler) means ``available: False``. A
domain-level rejection from librarian itself (e.g. a caller-side validation
error, missing ``paper_id``) means librarian responded — that's
``available: True`` with the rejection surfaced under a separate ``error``
key, so a caller can tell "my request was malformed" from "librarian is
down" (khonliang-researcher PR #75 review finding).

Note there is deliberately no ingest-pipeline call site wired to this module:
the ``ingest.url_distilled`` / ``ingest.queue_drained`` bus events researcher
already publishes (``researcher/ingest_watcher.py``) are the sanctioned
fire-and-forget path librarian's own watcher subscribes to
(``librarian/agent.py:_watch_ingest_events`` in khonliang-librarian) — adding
a second, blocking call from the ingest hot path would duplicate that
contract and reintroduce the coupling this FR removes. This module exists
for call sites that want a synchronous answer (e.g. an on-demand consumer
skill), not for the ingest hot path.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

LIBRARIAN_AGENT_TYPE = "librarian"

# Default kept short relative to the generic 30s ``BaseAgent.request``
# default: most librarian skills are interactive / on-demand lookups
# (classify_paper, taxonomy_report, suggest_missing_nodes,
# promote_investigation, library_health), so a short timeout keeps a down or
# slow librarian-primary from stalling those callers for the full 30s before
# degrading gracefully.
DEFAULT_TIMEOUT = 10.0

# rebuild_neighborhoods and identify_gaps are corpus-wide SCANS (the FR itself
# describes them that way) — they legitimately take longer than an
# interactive lookup on a real corpus. A flat 10s timeout would report a
# healthy-but-busy librarian as unavailable, which is a functional regression
# from both the old in-process path and BaseAgent.request's own (longer)
# default. Give these two a longer per-operation timeout; everything else
# falls back to DEFAULT_TIMEOUT. Explicit ``timeout=`` on a call always wins.
OPERATION_TIMEOUTS: dict[str, float] = {
    "rebuild_neighborhoods": 60.0,
    "identify_gaps": 45.0,
}

# The 7 skills librarian-primary exposes (fr_librarian_bc0a06d7). Kept as an
# explicit allowlist so a typo'd operation name fails fast with a clear
# reason instead of round-tripping to the bus first.
KNOWN_OPERATIONS = frozenset({
    "classify_paper",
    "taxonomy_report",
    "rebuild_neighborhoods",
    "suggest_missing_nodes",
    "promote_investigation",
    "identify_gaps",
    "library_health",
})


async def call_librarian(
    agent: Any,
    operation: str,
    args: dict[str, Any] | None = None,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call one of librarian-primary's skills over the bus, never raising.

    ``timeout`` defaults to ``OPERATION_TIMEOUTS.get(operation,
    DEFAULT_TIMEOUT)`` when not given explicitly.

    Returns the librarian skill's result merged with ``{"available": True}``
    when librarian responded — including when it responded with a domain
    error (bad args, unknown paper_id, etc.), surfaced as
    ``{"available": True, "error": ...}`` — or ``{"available": False,
    "reason": ...}`` when librarian could not be reached at all (unknown
    operation, timeout, connection error, bus-level dispatch failure, or an
    unexpected response shape). See the module docstring for how the two are
    told apart.
    """
    if operation not in KNOWN_OPERATIONS:
        return {
            "available": False,
            "reason": f"unknown librarian operation: {operation!r}",
        }
    resolved_timeout = OPERATION_TIMEOUTS.get(operation, DEFAULT_TIMEOUT) if timeout is None else timeout
    try:
        result = await agent.request(
            agent_type=LIBRARIAN_AGENT_TYPE,
            operation=operation,
            args=args or {},
            timeout=resolved_timeout,
        )
    except Exception as exc:
        logger.warning("librarian call %r failed: %s", operation, exc)
        return {"available": False, "reason": f"librarian unreachable: {exc}"}

    if not isinstance(result, dict):
        return {
            "available": False,
            "reason": "librarian returned an unexpected response shape",
        }

    if "result" not in result:
        # Bus-level dispatch failure: no healthy agent for the type, a
        # timeout, connection refused, or a non-2xx callback response — the
        # request never reached librarian's handler. A successful dispatch
        # always wraps the handler's return value as
        # {"result": ..., "trace_id": ...} (bus/server.py
        # :_dispatch_resolved_request); a bare top-level "error" here means
        # librarian itself never saw this request.
        reason = result.get("error", "librarian dispatch failed")
        return {"available": False, "reason": str(reason)}

    payload = result["result"]
    if not isinstance(payload, dict):
        return {
            "available": False,
            "reason": "librarian returned an unexpected response shape",
        }
    if payload.get("error"):
        # Librarian responded — it's up — but rejected this specific
        # request (bad args, unknown id, etc.). Distinct from
        # unreachability: available stays True, error is a separate key.
        return {"available": True, "error": str(payload["error"])}
    return {"available": True, **payload}


async def classify_paper(
    agent: Any,
    paper_id: str,
    *,
    audience: str = "",
    detail: str = "brief",
    timeout: float | None = None,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "classify_paper",
        {"paper_id": paper_id, "audience": audience, "detail": detail},
        timeout=timeout,
    )


async def taxonomy_report(
    agent: Any,
    *,
    audience: str = "",
    branch: str = "",
    detail: str = "brief",
    max_groups: int = 25,
    max_relationships: int = 50,
    timeout: float | None = None,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "taxonomy_report",
        {
            "audience": audience,
            "branch": branch,
            "detail": detail,
            "max_groups": max_groups,
            "max_relationships": max_relationships,
        },
        timeout=timeout,
    )


async def rebuild_neighborhoods(
    agent: Any,
    *,
    audience: str = "",
    reason: str = "",
    timeout: float | None = None,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "rebuild_neighborhoods",
        {"audience": audience, "reason": reason},
        timeout=timeout,
    )


async def suggest_missing_nodes(
    agent: Any,
    query: str,
    *,
    audience: str = "",
    detail: str = "brief",
    timeout: float | None = None,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "suggest_missing_nodes",
        {"query": query, "audience": audience, "detail": detail},
        timeout=timeout,
    )


async def promote_investigation(
    agent: Any,
    workspace_id: str,
    *,
    target_branch: str = "",
    reason: str = "",
    timeout: float | None = None,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "promote_investigation",
        {
            "workspace_id": workspace_id,
            "target_branch": target_branch,
            "reason": reason,
        },
        timeout=timeout,
    )


async def identify_gaps(
    agent: Any,
    *,
    audience: str = "",
    branch: str = "",
    detail: str = "brief",
    max_gaps: int = 25,
    timeout: float | None = None,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "identify_gaps",
        {
            "audience": audience,
            "branch": branch,
            "detail": detail,
            "max_gaps": max_gaps,
        },
        timeout=timeout,
    )


async def library_health(
    agent: Any,
    *,
    detail: str = "brief",
    timeout: float | None = None,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "library_health",
        {"detail": detail},
        timeout=timeout,
    )
