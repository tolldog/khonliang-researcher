"""Librarian agent for durable corpus organization."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from typing import Any

from khonliang.knowledge.store import EntryStatus, Tier
from khonliang_bus import BaseAgent, Skill, Welcome, WelcomeEntryPoint, handler
from khonliang_bus.connector import BusConnector
from khonliang_researcher import (
    AmbiguityRecord,
    GapReport,
    LibrarianStore,
    NeighborhoodSnapshot,
    PaperClassification,
    build_concept_graph,
    build_concept_taxonomy,
    classify_paper_from_triples,
    identify_gap_candidates,
    suggest_entities,
)

from researcher.pipeline import create_pipeline, is_paper_entry

logger = logging.getLogger(__name__)


class LibrarianAgent(BaseAgent):
    agent_type = "librarian"
    module_name = "researcher.librarian_agent"

    # Cold-start orientation surface (fr_khonliang-bus-lib_6a82732c).
    # Read by ephemeral external LLM sessions via the welcome skill.
    WELCOME = Welcome(
        role="corpus health + taxonomy authority",
        mission=(
            "Curates the durable library — classification, neighborhoods, "
            "taxonomy, gap identification, investigation promotion. "
            "Researcher ingests; librarian organizes. Co-resident in the "
            "researcher repo today; future extraction into a standalone "
            "repo is queued."
        ),
        not_responsible_for=[
            "paper / RSS / GitHub ingestion (researcher.fetch_paper / brief_on)",
            "evidence retrieval and synthesis (researcher.find_relevant / brief_on)",
            "FR / spec / milestone lifecycle (developer)",
        ],
        delegates_to={
            "researcher": "ingest + retrieval + synthesis + distillation",
        },
        entry_points=[
            WelcomeEntryPoint(
                skill="taxonomy_report",
                when_to_use="browse the durable library taxonomy — buckets, codes, entities",
            ),
            WelcomeEntryPoint(
                skill="library_health",
                when_to_use="summary of coverage, freshness, classified-vs-pending counts, ambiguity / gap totals",
            ),
            WelcomeEntryPoint(
                skill="identify_gaps",
                when_to_use="find under-covered taxonomy branches; emits research-request events for ingestion",
            ),
            WelcomeEntryPoint(
                skill="classify_paper",
                when_to_use="assign or refresh a stable library classification for a single paper",
            ),
            WelcomeEntryPoint(
                skill="rebuild_neighborhoods",
                when_to_use="deterministic taxonomy / neighborhood rebuild after batch ingest",
            ),
        ],
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.pipeline = create_pipeline(self.config_path or "config.yaml")
        self.store = LibrarianStore(str(self.pipeline.config.get("db_path", "data/researcher.db")))
        self._event_task: asyncio.Task | None = None
        self._event_subscriber_id = f"{self.agent_id}-ingest-events"

    def register_skills(self) -> list[Skill]:
        return [
            Skill("library_health", "Summary of durable library coverage and freshness.", {
                "detail": {"type": "string", "default": "brief"},
            }),
            Skill("rebuild_neighborhoods", "Deterministically rebuild taxonomy/neighborhood artifacts.", {
                "audience": {"type": "string", "default": ""},
                "reason": {"type": "string", "default": ""},
            }),
            Skill("taxonomy_report", "Browse the durable library taxonomy.", {
                "audience": {"type": "string", "default": ""},
                "branch": {"type": "string", "default": ""},
                "detail": {"type": "string", "default": "brief"},
                "max_groups": {"type": "integer", "default": 25},
                "max_relationships": {"type": "integer", "default": 50},
            }),
            Skill("suggest_missing_nodes", "Suggest existing nodes and taxonomy groups for a missing lookup.", {
                "query": {"type": "string", "required": True},
                "audience": {"type": "string", "default": ""},
                "detail": {"type": "string", "default": "brief"},
            }),
            Skill("classify_paper", "Assign or refresh a stable library classification for a paper.", {
                "paper_id": {"type": "string", "required": True},
                "audience": {"type": "string", "default": ""},
                "detail": {"type": "string", "default": "brief"},
            }),
            Skill("promote_investigation", "Promote a workspace artifact into the durable library via an artifact snapshot.", {
                "workspace_id": {"type": "string", "required": True},
                "target_branch": {"type": "string", "default": ""},
                "reason": {"type": "string", "default": ""},
            }),
            Skill("identify_gaps", "Identify under-covered taxonomy branches and emit research-request events.", {
                "audience": {"type": "string", "default": ""},
                "branch": {"type": "string", "default": ""},
                "detail": {"type": "string", "default": "brief"},
                "max_gaps": {"type": "integer", "default": 25},
            }),
        ]

    def _graph(self):
        return build_concept_graph(self.pipeline.triples, knowledge=self.pipeline.knowledge)

    def _taxonomy(self) -> dict[str, Any]:
        snapshot = self.store.latest_snapshot()
        if snapshot:
            return snapshot.content
        graph = self._graph()
        return build_concept_taxonomy(graph)

    @staticmethod
    def _limit_rows(rows: list[dict[str, Any]], max_items: int) -> tuple[list[dict[str, Any]], bool]:
        limited = rows[:max_items]
        return limited, len(rows) > len(limited)

    async def _create_artifact(
        self,
        *,
        kind: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        source_artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        response = await self._http.post(
            f"{self.bus_url}/v1/artifacts",
            json={
                "kind": kind,
                "title": title,
                "content": content,
                "producer": self.agent_id,
                "content_type": "application/json",
                "metadata": metadata,
                "source_artifacts": source_artifacts or [],
            },
        )
        return response.json()

    @staticmethod
    def _artifact_id(payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        nested = payload.get("artifact")
        if isinstance(nested, dict) and nested.get("id"):
            return str(nested["id"])
        if payload.get("id"):
            return str(payload["id"])
        return ""

    def _paper_entries(self):
        # Filter to real URL-backed papers so library_health counts and
        # rebuild_neighborhoods classification don't pollute with ideas
        # or other non-paper Tier.IMPORTED entries.
        return [
            entry
            for entry in self.pipeline.knowledge.get_by_tier(Tier.IMPORTED)
            if is_paper_entry(entry)
        ]

    async def _create_classification_delta_artifact(
        self,
        *,
        audience: str,
        reason: str,
        classified_count: int,
        ambiguous_count: int,
    ) -> str:
        artifact = await self._create_artifact(
            kind="librarian.classification_delta",
            title=f"Librarian classification delta {int(time.time())}",
            content=json.dumps(
                {
                    "audience": audience,
                    "reason": reason,
                    "classified_count": classified_count,
                    "ambiguous_count": ambiguous_count,
                },
                indent=2,
            ),
            metadata={
                "audience": audience,
                "reason": reason,
                "classified_count": classified_count,
                "ambiguous_count": ambiguous_count,
            },
        )
        return self._artifact_id(artifact)

    async def _wait_for_ingest_event(self, *, timeout: float = 30.0) -> dict[str, Any]:
        response = await self._http.post(
            f"{self.bus_url}/v1/wait",
            json={
                "topics": ["ingest.url_distilled", "ingest.queue_drained"],
                "subscriber_id": self._event_subscriber_id,
                "timeout": timeout,
                "ack_on_return": True,
            },
            timeout=timeout + 5,
        )
        return response.json()

    async def _handle_bus_event(self, event: dict[str, Any]) -> None:
        topic = str(event.get("topic", "")).strip()
        raw_payload = event.get("payload")
        if raw_payload is not None and not isinstance(raw_payload, dict):
            logger.warning(
                "librarian ingest-event watcher got non-dict payload for topic %s (type=%s); treating as empty",
                topic,
                type(raw_payload).__name__,
            )
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        if topic == "ingest.url_distilled":
            entry_id = str(payload.get("entry_id", "")).strip()
            if entry_id:
                await self.handle_classify_paper(
                    {"paper_id": entry_id, "detail": "brief"}
                )
            return
        if topic == "ingest.queue_drained":
            await self.handle_rebuild_neighborhoods(
                {
                    "audience": "",
                    "reason": "event:ingest.queue_drained",
                }
            )

    async def _watch_ingest_events(self) -> None:
        while True:
            try:
                result = await self._wait_for_ingest_event(timeout=30.0)
                if result.get("status") != "matched":
                    continue
                event = result.get("event") or {}
                if event:
                    await self._handle_bus_event(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("librarian ingest-event watcher failed")
                await asyncio.sleep(2)

    async def _ensure_snapshot(
        self, audience: str = "", reason: str = "bootstrap"
    ) -> tuple[dict[str, Any], str]:
        """Return ``(taxonomy_content, snapshot_id)`` for the latest snapshot.

        Returning both together closes a race window: callers that persist
        classifications tagged with ``source_snapshot_id`` must reference the
        *same* snapshot whose taxonomy they consumed. A separate
        ``self.store.latest_snapshot(audience)`` lookup done after
        classification could see a newer snapshot published by a concurrent
        rebuild, producing records whose ``source_snapshot_id`` does not
        actually match the taxonomy used.

        Callers that only need the taxonomy can discard the second element:
        ``taxonomy, _ = await self._ensure_snapshot(...)``.
        """
        snapshot = self.store.latest_snapshot(audience)
        if snapshot:
            return snapshot.content, snapshot.snapshot_id
        # rebuild persists the snapshot; re-read from the store rather than
        # relying on the handler response (which is intentionally compact and
        # no longer carries the full taxonomy payload).
        await self.handle_rebuild_neighborhoods({"audience": audience, "reason": reason})
        refreshed = self.store.latest_snapshot(audience)
        if refreshed:
            return refreshed.content, refreshed.snapshot_id
        return {}, ""

    @handler("library_health")
    async def handle_library_health(self, args: dict[str, Any]) -> dict[str, Any]:
        total_papers = len(self._paper_entries())
        health = self.store.health_summary(total_papers=total_papers)
        health["detail"] = args.get("detail", "brief")
        return health

    @handler("rebuild_neighborhoods")
    async def handle_rebuild_neighborhoods(self, args: dict[str, Any]) -> dict[str, Any]:
        """Rebuild taxonomy/neighborhood artifacts and persist a snapshot.

        The response is intentionally bounded: it returns a compact summary
        (snapshot_id, audience, paper_count, classification_count,
        ambiguous_count, created_at, artifact_id, delta_artifact_id) rather
        than the full taxonomy. This matches the token-economy goal — the
        common ``ingest.queue_drained`` automation path would otherwise emit
        oversized bus responses and risk timeouts.

        Callers that need the full taxonomy should read it via
        ``self.store.latest_snapshot(audience).content`` (local API) or a
        future ``get_taxonomy_snapshot`` bus skill. The compute-store-distill
        pattern (``fr_khonliang_8441aff3``) is the long-term home for that
        retrieval surface.
        """
        audience = str(args.get("audience", "")).strip()
        reason = str(args.get("reason", "")).strip()
        graph = self._graph()
        taxonomy = build_concept_taxonomy(graph)
        triples = self.pipeline.triples.get(min_confidence=0.5, limit=5000)
        # Nanosecond resolution prevents collisions when a manual rebuild and
        # an event-driven rebuild (e.g. ingest.queue_drained) fire within the
        # same wall-clock second. snapshot_id is TEXT; integer nanoseconds stay
        # sortable and dependency-free.
        snapshot_id = f"libsnap_{time.time_ns()}"

        paper_entries = self._paper_entries()
        paper_count = len(paper_entries)
        classified = 0
        ambiguous = 0
        for entry in paper_entries:
            result = classify_paper_from_triples(entry.id, triples, taxonomy, audience=audience)
            if result["status"] == "classified":
                self.store.upsert_classification(
                    PaperClassification(
                        paper_id=result["paper_id"],
                        classification_code=result["classification_code"],
                        audience_tags=result["audience_tags"],
                        confidence=result["confidence"],
                        rationale=result["rationale"],
                        source_snapshot_id=snapshot_id,
                    )
                )
                classified += 1
            elif result["status"] == "ambiguous":
                self.store.log_ambiguity(
                    AmbiguityRecord(
                        paper_id=entry.id,
                        candidates=result["candidates"],
                        reason=result["reason"],
                    )
                )
                ambiguous += 1

        snapshot_payload = {
            "taxonomy": taxonomy,
            "classified_count": classified,
            "ambiguous_count": ambiguous,
            "audience": audience,
        }
        artifact = await self._create_artifact(
            kind="librarian.taxonomy",
            title=f"Librarian taxonomy snapshot {snapshot_id}",
            content=json.dumps(snapshot_payload, indent=2),
            metadata={"audience": audience, "reason": reason},
        )
        artifact_id = self._artifact_id(artifact)
        delta_artifact_id = await self._create_classification_delta_artifact(
            audience=audience,
            reason=reason,
            classified_count=classified,
            ambiguous_count=ambiguous,
        )

        stored = self.store.store_snapshot(
            NeighborhoodSnapshot(
                snapshot_id=snapshot_id,
                audience=audience,
                artifact_id=artifact_id,
                reason=reason,
                content=taxonomy,
            )
        )
        created_at = float(stored.rebuilt_at) if stored is not None else time.time()

        await self.publish(
            "library.rebuilt",
            {
                "audience": audience,
                "artifact_id": artifact_id,
                "delta_artifact_id": delta_artifact_id,
                "reason": reason,
                "rebuilt_at": created_at,
                "changes_summary": {
                    "groups": len(taxonomy.get("groups", [])),
                    "relationships": len(taxonomy.get("relationships", [])),
                    "classified_count": classified,
                    "ambiguous_count": ambiguous,
                },
            },
        )
        return {
            "snapshot_id": snapshot_id,
            "audience": audience,
            "artifact_id": artifact_id,
            "delta_artifact_id": delta_artifact_id,
            "paper_count": paper_count,
            "classification_count": classified,
            "ambiguous_count": ambiguous,
            "created_at": created_at,
        }

    @handler("taxonomy_report")
    async def handle_taxonomy_report(self, args: dict[str, Any]) -> dict[str, Any]:
        taxonomy, _ = await self._ensure_snapshot(str(args.get("audience", "")).strip())
        audience = str(args.get("audience", "")).strip()
        branch = str(args.get("branch", "")).strip()
        detail = str(args.get("detail", "brief")).strip() or "brief"
        max_groups = max(1, int(args.get("max_groups", 25) or 25))
        max_relationships = max(1, int(args.get("max_relationships", 50) or 50))
        groups = list(taxonomy.get("groups", []))
        if audience:
            groups = [group for group in groups if group.get("audience") == audience]
        if branch:
            groups = [
                group for group in groups
                if group.get("code") == branch or branch in group.get("label", "")
            ]
        relationships = list(taxonomy.get("relationships", []))
        if branch:
            relationships = [
                rel for rel in relationships
                if rel.get("source") == branch or rel.get("target") == branch
            ]
        total_groups = len(groups)
        total_relationships = len(relationships)
        if detail != "full":
            groups, groups_truncated = self._limit_rows(groups, max_groups)
            relationships, relationships_truncated = self._limit_rows(
                relationships,
                max_relationships,
            )
        else:
            groups_truncated = False
            relationships_truncated = False
        return {
            "audience": audience,
            "branch": branch,
            "summary": {
                "group_count": total_groups,
                "relationship_count": total_relationships,
                "groups_truncated": groups_truncated,
                "relationships_truncated": relationships_truncated,
            },
            "groups": groups,
            "relationships": relationships,
            "detail": detail,
        }

    @handler("suggest_missing_nodes")
    async def handle_suggest_missing_nodes(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        audience = str(args.get("audience", "")).strip()
        graph = self._graph()
        taxonomy, _ = await self._ensure_snapshot(audience)
        suggestions = suggest_entities(graph, query)
        groups = list(taxonomy.get("groups", []))
        if audience:
            groups = [group for group in groups if group.get("audience") in {audience, "universal"}]
        normalized = query.lower().replace("-", " ").replace("_", " ")
        group_candidates = [
            group for group in groups
            if normalized in group.get("label", "").lower().replace("-", " ").replace("_", " ")
            or group.get("label", "").lower().replace("-", " ").replace("_", " ") in normalized
        ][:5]
        return {
            "query": query,
            "suggestions": [{"name": name, "score": score} for name, score in suggestions],
            "group_candidates": group_candidates,
            "detail": args.get("detail", "brief"),
        }

    @handler("classify_paper")
    async def handle_classify_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        paper_id = str(args.get("paper_id", "")).strip()
        audience = str(args.get("audience", "")).strip()
        # Capture both taxonomy and snapshot_id from a single store read so a
        # concurrent rebuild landing between "pick taxonomy" and "persist
        # source_snapshot_id" cannot make the persisted snapshot pointer
        # disagree with the taxonomy that was actually classified against.
        taxonomy, source_snapshot_id = await self._ensure_snapshot(audience)
        triples = self.pipeline.triples.get(min_confidence=0.5, limit=5000)
        result = classify_paper_from_triples(paper_id, triples, taxonomy, audience=audience)
        if result["status"] == "classified":
            record = self.store.upsert_classification(
                PaperClassification(
                    paper_id=result["paper_id"],
                    classification_code=result["classification_code"],
                    audience_tags=result["audience_tags"],
                    confidence=result["confidence"],
                    rationale=result["rationale"],
                    source_snapshot_id=source_snapshot_id,
                )
            )
            await self.publish(
                "library.classification_assigned",
                {
                    "paper_id": record.paper_id,
                    "classification_code": record.classification_code,
                    "audience_tags": record.audience_tags,
                    "classified_at": record.updated_at,
                    "confidence": record.confidence,
                },
            )
            return {"status": "classified", "record": asdict(record)}
        if result["status"] == "ambiguous":
            record = self.store.log_ambiguity(
                AmbiguityRecord(
                    paper_id=paper_id,
                    candidates=result["candidates"],
                    reason=result["reason"],
                )
            )
            await self.publish(
                "library.classification_ambiguous",
                {
                    "paper_id": paper_id,
                    "candidates": result["candidates"],
                    "reason": result["reason"],
                    "logged_at": record.logged_at,
                },
            )
            return {"status": "ambiguous", "record": asdict(record)}
        return result

    @handler("promote_investigation")
    async def handle_promote_investigation(self, args: dict[str, Any]) -> dict[str, Any]:
        workspace_id = str(args.get("workspace_id", "")).strip()
        target_branch = str(args.get("target_branch", "")).strip()
        reason = str(args.get("reason", "")).strip()
        response = await self._http.get(
            f"{self.bus_url}/v1/artifacts/{workspace_id}/content",
            params={"offset": 0, "max_chars": 100000},
        )
        data = response.json()
        content = data.get("content", "")
        if not content:
            return {"error": f"workspace artifact not found or empty: {workspace_id}"}
        try:
            workspace = json.loads(content)
        except json.JSONDecodeError:
            return {"error": f"workspace artifact is not JSON: {workspace_id}"}
        artifact = await self._create_artifact(
            kind="librarian.investigation_promotion",
            title=f"Promotion of {workspace_id}",
            content=json.dumps(
                {
                    "workspace_id": workspace_id,
                    "target_branch": target_branch,
                    "reason": reason,
                    "workspace": workspace,
                },
                indent=2,
            ),
            metadata={"target_branch": target_branch, "reason": reason},
            source_artifacts=[workspace_id],
        )
        return {
            "workspace_id": workspace_id,
            "target_branch": target_branch,
            "artifact_id": self._artifact_id(artifact),
        }

    @handler("identify_gaps")
    async def handle_identify_gaps(self, args: dict[str, Any]) -> dict[str, Any]:
        audience = str(args.get("audience", "")).strip()
        branch = str(args.get("branch", "")).strip()
        detail = str(args.get("detail", "brief")).strip() or "brief"
        max_gaps = max(1, int(args.get("max_gaps", 25) or 25))
        taxonomy, _ = await self._ensure_snapshot(audience)
        classifications = self.store.list_classifications(audience)
        gaps = identify_gap_candidates(taxonomy, classifications, audience=audience)
        if branch:
            gaps = [gap for gap in gaps if gap.branch == branch]
        total_gaps = len(gaps)
        gap_artifact = await self._create_artifact(
            kind="librarian.gap_report",
            title=f"Librarian gap scan {int(time.time())}",
            content=json.dumps(
                {
                    "audience": audience,
                    "branch": branch,
                    "detail": detail,
                    "total_gaps": total_gaps,
                    "gaps": [asdict(gap) for gap in gaps],
                },
                indent=2,
            ),
            metadata={
                "audience": audience,
                "branch": branch,
                "detail": detail,
                "total_gaps": total_gaps,
            },
        )
        artifact_id = self._artifact_id(gap_artifact)
        if detail != "full":
            gaps = gaps[:max_gaps]
        emitted = []
        for gap in gaps:
            stored = self.store.upsert_gap_report(gap)
            payload = {
                "request_id": stored.request_id,
                "topic": stored.topic,
                "audience": stored.audience,
                "branch": stored.branch,
                "priority": stored.priority,
                "rationale": stored.rationale,
                "suggested_sources": stored.suggested_sources,
                "detail": stored.detail,
            }
            await self.publish("library.gap_identified", payload)
            emitted.append(payload)
        return {
            "gaps": emitted,
            "detail": detail,
            "artifact_id": artifact_id,
            "summary": {
                "total_gaps": total_gaps,
                "emitted_count": len(emitted),
                "truncated": total_gaps > len(emitted),
            },
        }

    async def start(self) -> None:
        skills = self._all_skills()
        collabs = self.register_collaborations()
        self._connector = BusConnector(
            bus_url=self.bus_url,
            agent_id=self.agent_id,
            on_request=self._dispatch_request,
        )
        try:
            await self._connector.connect_and_register(
                agent_type=self.agent_type,
                version=self.version,
                pid=os.getpid(),
                skills=[s.to_dict() for s in skills],
                collaborations=[
                    {
                        "name": c.name,
                        "description": c.description,
                        "requires": c.requires,
                        "steps": c.steps,
                    }
                    for c in collabs
                ],
            )
        except Exception:
            await self._http.aclose()
            raise

        self._event_task = asyncio.create_task(self._watch_ingest_events())

        logger.info(
            "Agent %s started (%d skills, WebSocket)",
            self.agent_id,
            len(skills),
        )
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        except NotImplementedError:
            pass
        try:
            await self._connector.run()
        finally:
            if self._event_task is not None:
                self._event_task.cancel()
                try:
                    await self._event_task
                except asyncio.CancelledError:
                    pass
            await self._http.aclose()

    async def shutdown(self) -> None:
        if self._event_task is not None:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None
        await BaseAgent.shutdown(self)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Librarian bus agent")
    parser.add_argument("command", nargs="?", choices=["install", "uninstall"])
    parser.add_argument("--id", default="librarian-primary")
    parser.add_argument("--bus", default="http://localhost:8788")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command in ("install", "uninstall"):
        LibrarianAgent.from_cli([
            args.command,
            "--id", args.id,
            "--bus", args.bus,
            "--config", args.config,
        ])
        return

    agent = LibrarianAgent(
        agent_id=args.id,
        bus_url=args.bus,
        config_path=args.config,
    )
    asyncio.run(agent.start())


if __name__ == "__main__":
    main()
