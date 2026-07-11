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
and bus-level error envelopes (e.g. librarian-primary not currently
registered) are caught and normalized into a non-fatal
``{"available": False, "reason": ...}`` envelope — never raised. Callers
must treat ``available: False`` as "skip librarian-dependent enrichment for
this request," not as a hard failure of whatever they were doing.

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
# default: librarian-consuming callers are expected to be interactive /
# on-demand, not background jobs, and a short timeout keeps a down or
# slow librarian-primary from stalling the caller for the full 30s before
# degrading gracefully.
DEFAULT_TIMEOUT = 10.0

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
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call one of librarian-primary's skills over the bus, never raising.

    Returns the librarian skill's result merged with ``{"available": True}``
    on success, or ``{"available": False, "reason": ...}`` on any failure
    (unknown operation, timeout, connection error, bus-level error
    envelope, or an unexpected response shape).
    """
    if operation not in KNOWN_OPERATIONS:
        return {
            "available": False,
            "reason": f"unknown librarian operation: {operation!r}",
        }
    try:
        result = await agent.request(
            agent_type=LIBRARIAN_AGENT_TYPE,
            operation=operation,
            args=args or {},
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("librarian call %r failed: %s", operation, exc)
        return {"available": False, "reason": f"librarian unreachable: {exc}"}

    payload = result.get("result", result) if isinstance(result, dict) else result
    if isinstance(payload, dict) and payload.get("error"):
        return {"available": False, "reason": str(payload["error"])}
    if not isinstance(payload, dict):
        return {
            "available": False,
            "reason": "librarian returned an unexpected response shape",
        }
    return {"available": True, **payload}


async def classify_paper(
    agent: Any,
    paper_id: str,
    *,
    audience: str = "",
    detail: str = "brief",
    timeout: float = DEFAULT_TIMEOUT,
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
    timeout: float = DEFAULT_TIMEOUT,
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
    timeout: float = DEFAULT_TIMEOUT,
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
    timeout: float = DEFAULT_TIMEOUT,
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
    timeout: float = DEFAULT_TIMEOUT,
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
    timeout: float = DEFAULT_TIMEOUT,
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
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    return await call_librarian(
        agent,
        "library_health",
        {"detail": detail},
        timeout=timeout,
    )
