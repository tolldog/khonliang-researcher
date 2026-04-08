"""Research pipeline: fetch → ingest → distill → extract triples.

Orchestrates khonliang components:
  - ResearchPool for threaded fetch/parse
  - KnowledgeStore for paper storage (Tier 2 raw, Tier 3 summaries)
  - TripleStore for relationship extraction
  - DigestStore for activity tracking
  - Roles for LLM distillation
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from khonliang.knowledge.store import KnowledgeStore, KnowledgeEntry, Tier, EntryStatus
from khonliang.knowledge.triples import TripleStore
from khonliang.digest.store import DigestStore
from khonliang.pool import ModelPool
from khonliang.research.pool import ResearchPool
from khonliang.research.models import ResearchTask

from researcher.fetcher import extract_arxiv_id, fetch_arxiv, fetch_url
from researcher.parser import parse_paper_list, PaperReference
from researcher.queue import PaperFetcher, ListParser
from researcher.idea import IdeaParserRole
from khonliang_researcher import RelevanceScorer
from researcher.roles import SummarizerRole, ExtractorRole, AssessorRole

logger = logging.getLogger(__name__)


@dataclass
class DistillResult:
    entry_id: str
    title: str
    summary: Optional[Dict[str, Any]] = None
    triples: List[Dict[str, Any]] = field(default_factory=list)
    assessments: Dict[str, Any] = field(default_factory=dict)
    success: bool = False


def update_capability_status(
    knowledge: KnowledgeStore,
    target: str,
    title: str,
    concept: str,
    status: str,
    fr_id: str = "",
):
    """Track what exists/is planned per project. Call on FR status changes.

    Status is monotonic: 'exists' is never downgraded to 'planned'.
    Tags and content are kept in sync with the resolved status.
    """
    def _resolve(current: str, new: str) -> str:
        if current == "exists" or new == "exists":
            return "exists"
        return new

    cap_id = f"cap_{target}_{hashlib.sha256(title.encode()).hexdigest()[:8]}"

    existing = knowledge.get(cap_id)
    if existing:
        metadata = existing.metadata or {}
        resolved = _resolve(metadata.get("capability_status", ""), status)
        metadata["target"] = target
        metadata["concept"] = concept
        metadata["capability_status"] = resolved
        if fr_id:
            metadata["fr_id"] = fr_id
        existing.metadata = metadata
        existing.content = f"{resolved}: {title}"
        existing.tags = [
            t for t in (existing.tags or [])
            if not t.startswith("cap:") and t != "capability"
        ] + ["capability", f"cap:{target}", f"cap:{resolved}"]
        knowledge.add(existing)
    else:
        entry = KnowledgeEntry(
            id=cap_id,
            tier=Tier.DERIVED,
            title=title,
            content=f"{status}: {title}",
            source="capability_tracker",
            scope="capability",
            tags=["capability", f"cap:{target}", f"cap:{status}"],
            status=EntryStatus.DISTILLED,
            metadata={
                "target": target,
                "concept": concept,
                "capability_status": status,
                "fr_id": fr_id,
            },
        )
        knowledge.add(entry)


class ResearchPipeline:
    """Main orchestrator for the research paper pipeline."""

    def __init__(
        self,
        knowledge: KnowledgeStore,
        triples: TripleStore,
        digest: DigestStore,
        pool: ModelPool,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.knowledge = knowledge
        self.triples = triples
        self.digest = digest
        self.pool = pool
        self.config = config or {}

        # LLM roles
        self.summarizer = SummarizerRole(pool)
        self.extractor = ExtractorRole(pool)
        self.assessor = AssessorRole(pool)
        self.idea_parser = IdeaParserRole(pool)

        # Persistent blackboard for relevance signal learning
        from khonliang.gateway.blackboard import Blackboard
        db_dir = Path(self.config.get("db_path", "data/researcher.db")).parent
        self.blackboard = Blackboard(persist_to=str(db_dir / "blackboard.db"))

        # Relevance scorer (lazy init on first use)
        self.relevance = RelevanceScorer(
            targets=self.config.get("projects", {}),
            ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
            model=self.config.get("models", {}).get("embedder", "nomic-embed-text"),
            threshold=self.config.get("relevance_threshold", 0.3),
            blackboard=self.blackboard,
        )

        # Research pool for threaded fetching
        self.research_pool = ResearchPool()
        self.research_pool.register(PaperFetcher())

        parser_client = pool.get_client("extractor")
        self.research_pool.register(ListParser(llm_client=parser_client))

        # URL dedup: check knowledge store for existing entries
        self._url_index: Dict[str, str] = {}  # url -> entry_id
        self._build_url_index()

        # One-time migration from tag-based status to EntryStatus
        self._migrate_status()

    def _build_url_index(self):
        """Build URL index from existing knowledge entries."""
        from researcher.fetcher import extract_arxiv_id
        for entry in self.knowledge.get_by_tier(Tier.IMPORTED):
            url = entry.metadata.get("url", "")
            if url:
                self._url_index[url] = entry.id
                # Also index canonical arxiv URL for cross-format dedup
                arxiv_id = extract_arxiv_id(url)
                if arxiv_id:
                    self._url_index[f"https://arxiv.org/abs/{arxiv_id}"] = entry.id
            original = entry.metadata.get("original_url", "")
            if original:
                self._url_index[original] = entry.id

    def _migrate_status(self):
        """One-time migration: backfill EntryStatus from tags for existing entries."""
        migrated = 0
        for entry in self.knowledge.get_by_status(EntryStatus.ACTIVE):
            tags = entry.tags or []
            if "undistilled" in tags:
                self.knowledge.set_status(entry.id, EntryStatus.INGESTED)
                migrated += 1
            elif "distilled" in tags or "summary" in tags:
                self.knowledge.set_status(entry.id, EntryStatus.DISTILLED)
                migrated += 1
        if migrated:
            logger.info("Migrated %d entries from tag-based to EntryStatus", migrated)

    def start(self):
        """Start the research pool workers."""
        self.research_pool.start(workers=5)

    def stop(self):
        """Stop the research pool."""
        self.research_pool.stop()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_paper(self, url: str) -> Optional[str]:
        """Fetch a paper and store as Tier 2. Returns entry_id or None if duplicate."""
        # Check for existing
        if url in self._url_index:
            logger.info("Paper already ingested: %s", url)
            return self._url_index[url]

        # Normalize arxiv URLs to canonical form for dedup
        arxiv_id = extract_arxiv_id(url)
        canonical_url = url
        if arxiv_id:
            canonical_url = f"https://arxiv.org/abs/{arxiv_id}"
            if canonical_url in self._url_index:
                logger.info("Paper already ingested (arxiv %s): %s", arxiv_id, url)
                return self._url_index[canonical_url]
            result = await fetch_arxiv(url)
        else:
            result = await fetch_url(url)

        if not result.content.strip():
            logger.warning("Empty content from %s", url)
            return None

        # Store as Tier 2 — use canonical URL for consistent IDs
        import hashlib
        entry_id = hashlib.sha256(canonical_url.encode()).hexdigest()[:16]

        entry = KnowledgeEntry(
            id=entry_id,
            tier=Tier.IMPORTED,
            title=result.title or url,
            content=result.content,
            source=url,
            scope="research",
            tags=["paper"],
            status=EntryStatus.INGESTED,
            metadata={
                "url": canonical_url,
                "original_url": url if url != canonical_url else "",
                "fetched_at": result.fetched_at,
                **result.metadata,
            },
        )
        self.knowledge.add(entry)
        self._url_index[canonical_url] = entry_id
        if url != canonical_url:
            self._url_index[url] = entry_id

        self.digest.record(
            summary=f"Ingested paper: {result.title or url}",
            source="pipeline",
            audience="research",
            tags=["ingested"],
            metadata={"entry_id": entry_id, "url": url},
        )

        logger.info("Ingested paper %s: %s", entry_id, result.title)
        return entry_id

    async def ingest_paper_list(self, url: str) -> List[PaperReference]:
        """Fetch an awesome-list and extract paper references."""
        from researcher.fetcher import fetch_raw

        raw = await fetch_raw(url)
        client = self.pool.get_client("extractor")
        papers = await parse_paper_list(raw, client)

        self.digest.record(
            summary=f"Parsed paper list from {url}: {len(papers)} papers found",
            source="pipeline",
            audience="research",
            tags=["list_parsed"],
            metadata={"url": url, "count": len(papers)},
        )

        return papers

    async def ingest_papers_from_list(
        self, papers: List[PaperReference], max_concurrent: int = 5
    ) -> List[str]:
        """Fetch and ingest multiple papers concurrently. Returns entry_ids."""
        sem = asyncio.Semaphore(max_concurrent)
        entry_ids = []

        async def _fetch_one(paper: PaperReference):
            async with sem:
                eid = await self.ingest_paper(paper.url)
                if eid:
                    entry_ids.append(eid)

        await asyncio.gather(
            *[_fetch_one(p) for p in papers],
            return_exceptions=True,
        )
        return entry_ids

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    async def score_relevance(self, entry_id: str) -> Dict[str, float]:
        """Score a paper's relevance to all projects. Returns {project: score}."""
        entry = self.knowledge.get(entry_id)
        if not entry:
            return {}
        scores = await self.relevance.score(entry.title, entry.content)
        if scores:
            entry.metadata["relevance_scores"] = scores
            self.knowledge.add(entry)
        return scores

    async def filter_irrelevant(self, entry_id: str) -> bool:
        """Check relevance and skip if below threshold for all projects.

        Returns True if paper was skipped, False if it should proceed.
        """
        entry = self.knowledge.get(entry_id)
        if not entry:
            return False

        relevant, scores = await self.relevance.is_relevant(entry.title, entry.content)
        if scores:
            entry.metadata["relevance_scores"] = scores
            self.knowledge.add(entry)

        if not relevant:
            self.knowledge.set_status(entry_id, EntryStatus.SKIPPED)
            max_score = max(scores.values()) if scores else 0
            # Record negative signal for adaptive learning
            await self.relevance.record_signal(
                entry.title, entry.content, "negative"
            )
            logger.info(
                "Skipped (relevance %.2f < %.2f): %s",
                max_score,
                self.relevance.threshold,
                entry.title[:60],
            )
            self.digest.record(
                summary=f"Skipped irrelevant paper: {entry.title} (max relevance: {max_score:.2f})",
                source="pipeline",
                audience="research",
                tags=["skipped", "relevance"],
                metadata={"entry_id": entry_id, "scores": scores},
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    async def distill(self, entry_id: str) -> DistillResult:
        """Run full distillation pipeline on a stored paper.

        Step 1: Summarize (7B model, sequential — needs raw paper)
        Step 2: Extract + Assess in parallel (3B model — both use summary)
        """
        entry = self.knowledge.get(entry_id)
        if not entry:
            return DistillResult(entry_id=entry_id, title="NOT FOUND")

        result = DistillResult(entry_id=entry_id, title=entry.title)
        self.knowledge.set_status(entry_id, EntryStatus.PROCESSING)

        # Step 1: Summarize (must complete before extraction/assessment)
        summary_resp = await self.summarizer.handle(entry.content)
        if not summary_resp.get("success"):
            logger.warning("Summarization failed for %s", entry_id)
            self.knowledge.set_status(entry_id, EntryStatus.FAILED)
            return result
        result.summary = summary_resp["summary"]

        # Step 2: Extract triples + assess all projects in parallel
        summary_text = json.dumps(result.summary, indent=2)
        projects = self.config.get("projects", {})

        async def _extract():
            resp = await self.extractor.handle(summary_text)
            return resp.get("triples", []) if resp.get("success") else []

        async def _assess(name, cfg):
            resp = await self.assessor.handle(
                summary_text,
                context={"project_description": cfg.get("description", "")},
            )
            return name, resp.get("assessment") if resp.get("success") else None

        # Fan out: 1 extraction + N assessments concurrently
        tasks = [_extract()]
        for proj_name, proj_cfg in projects.items():
            tasks.append(_assess(proj_name, proj_cfg))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        result.triples = results[0] if not isinstance(results[0], Exception) else []
        for r in results[1:]:
            if isinstance(r, tuple) and r[1] is not None:
                result.assessments[r[0]] = r[1]

        # Store results
        self._store_distillation(entry, result)

        # Record positive signal — paper was worth distilling
        await self.relevance.record_signal(
            entry.title, entry.content, "positive"
        )

        result.success = True
        return result

    def _store_distillation(self, entry: KnowledgeEntry, result: DistillResult):
        """Persist distillation results to knowledge + triple stores."""
        # Store summary as Tier 3
        if result.summary:
            summary_entry = KnowledgeEntry(
                id=f"{entry.id}_summary",
                tier=Tier.DERIVED,
                title=f"Summary: {entry.title}",
                content=json.dumps(result.summary, indent=2),
                source=entry.id,
                scope="research",
                tags=["summary"],
                status=EntryStatus.DISTILLED,
                metadata={
                    "parent_id": entry.id,
                    "url": entry.metadata.get("url", ""),
                    "assessments": result.assessments,
                },
            )
            self.knowledge.add(summary_entry)

        # Store triples
        for triple in result.triples:
            if isinstance(triple, dict) and triple.get("subject") and triple.get("object"):
                self.triples.add(
                    subject=triple["subject"],
                    predicate=triple.get("predicate", "related_to"),
                    obj=triple["object"],
                    confidence=float(triple.get("confidence", 0.7)),
                    source=f"paper:{entry.id}",
                )

        # Mark original entry as distilled
        self.knowledge.set_status(entry.id, EntryStatus.DISTILLED)

        self.digest.record(
            summary=f"Distilled paper: {entry.title} — {len(result.triples)} triples extracted",
            source="pipeline",
            audience="research",
            tags=["distilled"],
            metadata={
                "entry_id": entry.id,
                "triple_count": len(result.triples),
                "projects_assessed": list(result.assessments.keys()),
            },
        )

    async def distill_all_pending(self) -> List[DistillResult]:
        """Find and distill all papers that haven't been processed yet."""
        results = []
        for entry in self.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED):
            result = await self.distill(entry.id)
            results.append(result)
        return results

    def strike(self, entry_id: str) -> Dict[str, Any]:
        """Remove a paper, its summary, and its triples. Allows fresh re-import.

        Args:
            entry_id: The paper's entry ID

        Returns:
            Dict with counts of what was removed.
        """
        removed = {"paper": False, "summary": False, "triples": 0}

        entry = self.knowledge.get(entry_id)
        if not entry:
            return {"error": f"Entry {entry_id} not found", **removed}

        title = entry.title
        url = entry.metadata.get("url", "")

        # Remove summary (Tier 3)
        summary_id = f"{entry_id}_summary"
        if self.knowledge.get(summary_id):
            self.knowledge.remove(summary_id)
            removed["summary"] = True

        # Remove triples sourced from this paper
        all_triples = self.triples.get(limit=10000)
        for t in all_triples:
            if f"paper:{entry_id}" in (t.source or ""):
                self.triples.remove(t.subject, t.predicate, t.object)
                removed["triples"] += 1

        # Remove the paper itself
        self.knowledge.remove(entry_id)
        removed["paper"] = True

        # Remove from URL index
        if url in self._url_index:
            del self._url_index[url]

        self.digest.record(
            summary=f"Struck paper: {title} ({removed['triples']} triples removed)",
            source="pipeline",
            audience="research",
            tags=["strike"],
        )

        logger.info(
            "Struck %s: paper=%s, summary=%s, triples=%d",
            entry_id, removed["paper"], removed["summary"], removed["triples"],
        )
        return {"title": title, **removed}

    # ------------------------------------------------------------------
    # Idea ingestor
    # ------------------------------------------------------------------

    async def ingest_idea(self, text: str, source_label: str = "") -> str:
        """Parse informal text into claims + queries, store as Tier 2 idea.

        Returns the idea entry_id.
        """
        import hashlib

        parsed = await self.idea_parser.handle(text)
        if not parsed.get("success"):
            raise RuntimeError(f"Idea parsing failed: {parsed.get('error', 'unknown')}")

        entry_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        title = parsed.get("title", "Untitled idea")
        if source_label:
            title = f"{title} ({source_label})"

        entry = KnowledgeEntry(
            id=entry_id,
            tier=Tier.IMPORTED,
            title=title,
            content=text,
            source=source_label or "idea",
            scope="research",
            tags=["idea"],
            status=EntryStatus.INGESTED,
            metadata={
                "source_type": parsed.get("source_type", "freeform"),
                "claims": parsed.get("claims", []),
                "search_queries": parsed.get("search_queries", []),
                "keywords": parsed.get("keywords", []),
            },
        )
        self.knowledge.add(entry)

        self.digest.record(
            summary=f"Ingested idea: {title} — {len(parsed.get('claims', []))} claims, {len(parsed.get('search_queries', []))} queries",
            source="pipeline",
            audience="research",
            tags=["idea", "ingested"],
            metadata={"entry_id": entry_id},
        )

        logger.info("Ingested idea %s: %s", entry_id, title)
        return entry_id

    async def research_idea(
        self, idea_id: str, max_papers: int = 10, auto_distill: bool = True
    ) -> Dict[str, Any]:
        """Search for papers backing an idea's claims, fetch and optionally distill.

        Returns stats: queries_run, papers_found, papers_new, papers_distilled.
        """
        from researcher.search_engines import search_papers

        entry = self.knowledge.get(idea_id)
        if not entry:
            return {"error": f"Idea {idea_id} not found"}

        queries = entry.metadata.get("search_queries", [])
        if not queries:
            return {"error": "No search queries in idea metadata"}

        stats = {"queries_run": 0, "papers_found": 0, "papers_new": 0, "papers_distilled": 0}
        seen_urls: set = set()
        new_entry_ids: list = []

        # Search all queries in parallel
        search_tasks = [search_papers(q, max_results=max_papers // max(len(queries), 1)) for q in queries]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for i, results in enumerate(search_results):
            stats["queries_run"] += 1
            if isinstance(results, Exception):
                logger.warning("Search failed for query %d: %s", i, results)
                continue
            for r in results:
                if r.url in seen_urls or r.url in self._url_index:
                    continue
                seen_urls.add(r.url)
                stats["papers_found"] += 1

        # Fetch new papers (deduplicated)
        distilled_ids = {e.id for e in self.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED)}
        for url in seen_urls:
            if stats["papers_new"] >= max_papers:
                break
            try:
                eid = await self.ingest_paper(url)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", url, e)
                continue
            if eid and eid not in distilled_ids:
                new_entry_ids.append(eid)
                stats["papers_new"] += 1
                # Tag paper as linked to this idea
                paper = self.knowledge.get(eid)
                if paper:
                    tags = paper.tags or []
                    tags.append(f"idea:{idea_id}")
                    paper.tags = tags
                    self.knowledge.add(paper)

        # Distill new papers if requested
        if auto_distill:
            for eid in new_entry_ids:
                paper = self.knowledge.get(eid)
                if paper and paper.status == EntryStatus.INGESTED:
                    result = await self.distill(eid)
                    if result.success:
                        stats["papers_distilled"] += 1

        # Mark idea as researched
        entry.metadata["papers_linked"] = new_entry_ids
        self.knowledge.add(entry)
        self.knowledge.set_status(idea_id, EntryStatus.DISTILLED)

        self.digest.record(
            summary=f"Researched idea: {entry.title} — {stats['papers_new']} new papers, {stats['papers_distilled']} distilled",
            source="pipeline",
            audience="research",
            tags=["idea", "researched"],
            metadata={"idea_id": idea_id, **stats},
        )

        return stats

    async def brief_idea(self, idea_id: str) -> str:
        """Synthesize a brief evaluating an idea's claims against found literature."""
        from researcher.synthesizer import Synthesizer

        entry = self.knowledge.get(idea_id)
        if not entry:
            return f"Idea {idea_id} not found."

        claims = entry.metadata.get("claims", [])
        paper_ids = entry.metadata.get("papers_linked", [])

        # Gather distilled summaries for linked papers
        summaries = []
        for pid in paper_ids:
            summary_entry = self.knowledge.get(f"{pid}_summary")
            if summary_entry:
                try:
                    summaries.append(json.loads(summary_entry.content))
                except json.JSONDecodeError:
                    pass

        if not summaries:
            return f"No distilled papers linked to idea {idea_id}. Run research_idea first."

        synth = Synthesizer(self.knowledge, self.triples, self.pool)
        result = await synth.idea_brief(entry.content, claims, summaries)

        if result.success:
            # Store as Tier 3
            brief_entry = KnowledgeEntry(
                id=f"{idea_id}_brief",
                tier=Tier.DERIVED,
                title=f"Brief: {entry.title}",
                content=result.content,
                source=idea_id,
                scope="research",
                tags=["brief", "idea"],
                status=EntryStatus.DISTILLED,
                metadata={"idea_id": idea_id, "paper_count": result.paper_count},
            )
            self.knowledge.add(brief_entry)

            self.digest.record(
                summary=f"Briefed idea: {entry.title} — {result.paper_count} papers analyzed",
                source="pipeline",
                audience="research",
                tags=["idea", "briefed"],
                metadata={"idea_id": idea_id},
            )

        return result.content if result.success else f"Brief generation failed: {result.content}"

    # ------------------------------------------------------------------
    # Synergize
    # ------------------------------------------------------------------

    async def synergize(
        self, min_score: float = 0.5, max_concepts: int = 10
    ) -> Dict[str, Any]:
        """Classify concepts and generate targeted FRs across the ecosystem.

        Returns parsed JSON with concept classifications and feature requests,
        or raw content if JSON parsing fails.
        """
        from researcher.synthesizer import Synthesizer

        synth = Synthesizer(self.knowledge, self.triples, self.pool)
        projects = self.config.get("projects", {})
        n_samples = self.config.get("synergize_samples", 1)

        result = await synth.synergize(
            projects=projects,
            min_score=min_score,
            max_concepts=max_concepts,
            n_samples=n_samples,
        )

        if not result.success:
            return {"error": result.content, "raw": None}

        # Try to parse JSON from LLM output
        content = result.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])

        try:
            classifications = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "LLM returned non-JSON", "raw": result.content}

        # Gather existing capabilities for post-generation filtering
        existing_caps = set()
        for entry in self.knowledge.get_by_tier(Tier.DERIVED):
            tags = entry.tags or []
            if "capability" in tags and (entry.metadata or {}).get("capability_status") in ("exists", "planned"):
                existing_caps.add(entry.title.lower())

        # Embed existing caps for fuzzy matching
        cap_embeddings = {}
        if existing_caps and self.relevance:
            try:
                for cap in existing_caps:
                    emb = await self.relevance._embed(cap)
                    if emb:
                        cap_embeddings[cap] = emb
            except Exception:
                pass  # Fall back to substring matching only

        # Store FRs as Tier 3 knowledge entries, filtering out already-built
        fr_count = 0
        skipped = 0  # capability matches
        fr_existed = 0  # already in knowledge store
        for item in classifications:
            for fr in item.get("feature_requests", []):
                fr_title_lower = fr.get("title", "").lower()

                # Skip FRs that match existing capabilities (substring)
                if any(cap in fr_title_lower or fr_title_lower in cap for cap in existing_caps):
                    skipped += 1
                    continue

                # Skip FRs that match via embedding similarity
                if cap_embeddings:
                    try:
                        from khonliang_researcher import cosine_similarity
                        fr_emb = await self.relevance._embed(fr_title_lower)
                        if fr_emb and any(
                            cosine_similarity(fr_emb, ce) > 0.85
                            for ce in cap_embeddings.values()
                        ):
                            skipped += 1
                            continue
                    except Exception:
                        pass

                import hashlib
                # Hash target + title + concept for stable, collision-resistant IDs
                fr_hash_input = f"{fr['target']}:{fr['title']}:{item.get('concept', '')}"
                fr_id = f"fr_{fr['target']}_{hashlib.sha256(fr_hash_input.encode()).hexdigest()[:8]}"

                # Don't overwrite existing FRs (preserves reviews, deps, status history)
                existing = self.knowledge.get(fr_id)
                if existing:
                    fr_existed += 1
                    continue

                fr_entry = KnowledgeEntry(
                    id=fr_id,
                    tier=Tier.DERIVED,
                    title=fr["title"],
                    content=json.dumps(fr, indent=2),
                    source="synergize",
                    scope="research",
                    tags=["fr", f"target:{fr['target']}", item.get("classification", "")],
                    status=EntryStatus.DISTILLED,
                    metadata={
                        "concept": item.get("concept", ""),
                        "classification": item.get("classification", ""),
                        "target": fr["target"],
                        "priority": fr.get("priority", "medium"),
                        "backing_papers": fr.get("backing_papers", []),
                        "synergies": [item["synergies"]] if isinstance(item.get("synergies"), str) else item.get("synergies", []),
                    },
                )
                self.knowledge.add(fr_entry)
                fr_count += 1

        if skipped:
            logger.info("Synergize: skipped %d FRs matching existing capabilities", skipped)
        if fr_existed:
            logger.info("Synergize: skipped %d FRs already in knowledge store", fr_existed)

        self.digest.record(
            summary=f"Synergize: {len(classifications)} concepts classified, {fr_count} FRs generated",
            source="pipeline",
            audience="research",
            tags=["synergize"],
            metadata={"concept_count": len(classifications), "fr_count": fr_count},
        )

        return {
            "classifications": classifications,
            "fr_count": fr_count,
            "concept_count": len(classifications),
        }

    async def synergize_compare(
        self, min_score: float = 0.5, max_concepts: int = 10
    ) -> Dict[str, Any]:
        """Run synergize in compare mode: return all candidates with diversity metrics.

        Generates N candidates and returns them all with the model's selection
        and concept overlap statistics for evaluating self-distillation quality.
        """
        from researcher.synthesizer import Synthesizer

        synth = Synthesizer(self.knowledge, self.triples, self.pool)
        projects = self.config.get("projects", {})
        n_samples = self.config.get("synergize_samples", 3)

        result = await synth.synergize(
            projects=projects,
            min_score=min_score,
            max_concepts=max_concepts,
            n_samples=max(n_samples, 2),
            compare=True,
        )

        if not result.success:
            return {"error": result.content}

        # Parse the compare envelope
        content = result.content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])

        try:
            envelope = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Could not parse compare output", "raw": result.content}

        selected = envelope.get("selected", 1)
        candidates = envelope.get("candidates", [])

        # Parse each candidate and compute diversity metrics
        parsed = []
        all_concepts: list[set] = []
        all_fr_titles: list[set] = []
        for i, raw in enumerate(candidates):
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])
            try:
                items = json.loads(raw)
                concepts = set()
                fr_titles = set()
                for item in items:
                    concept = str(item.get("concept") or "").strip().lower()
                    if concept:
                        concepts.add(concept)
                    for fr in item.get("feature_requests", []):
                        title = str(fr.get("title") or "").strip().lower()
                        if title:
                            fr_titles.add(title)
                all_concepts.append(concepts)
                all_fr_titles.append(fr_titles)
                parsed.append({
                    "candidate": i + 1,
                    "concepts": len(concepts),
                    "frs": len(fr_titles),
                    "valid": True,
                })
            except json.JSONDecodeError:
                all_concepts.append(set())
                all_fr_titles.append(set())
                parsed.append({"candidate": i + 1, "valid": False})

        # Compute global concept and FR overlap across all candidates
        union_concepts = set().union(*all_concepts) if all_concepts else set()
        intersection_concepts = all_concepts[0].copy() if all_concepts else set()
        for s in all_concepts[1:]:
            intersection_concepts &= s

        union_frs = set().union(*all_fr_titles) if all_fr_titles else set()
        intersection_frs = all_fr_titles[0].copy() if all_fr_titles else set()
        for s in all_fr_titles[1:]:
            intersection_frs &= s

        return {
            "selected": selected,
            "n_candidates": len(candidates),
            "candidates": parsed,
            "diversity": {
                "unique_concepts": len(union_concepts),
                "shared_concepts": len(intersection_concepts),
                "concept_overlap": len(intersection_concepts) / max(len(union_concepts), 1),
                "unique_frs": len(union_frs),
                "shared_frs": len(intersection_frs),
                "fr_overlap": len(intersection_frs) / max(len(union_frs), 1),
            },
        }

    _README_CLAIMS_PROMPT = """\
Extract capability claims from the following README. List only concrete, specific \
features that this project claims to implement. Do not list generic descriptions. \
Each capability name should be 3-8 words describing what the project does.

README:
{readme_text}

Respond with JSON only. The "claims" array must contain capabilities found in the README above.
{{"claims": ["capability from readme", ...]}}
"""

    async def _extract_readme_claims(self, readme: str) -> list:
        """Extract capability claims from README text using the extractor model."""
        client = self.pool.get_client("extractor")
        prompt = self._README_CLAIMS_PROMPT.format(readme_text=readme[:2000])
        try:
            raw = await client.generate(
                prompt=prompt,
                system="Extract capabilities. Output only JSON.",
                temperature=0.1,
                max_tokens=1000,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])
            return json.loads(raw).get("claims", [])
        except Exception:
            return []

    @staticmethod
    def _extract_package_metadata(repo_path: Path) -> Dict[str, Any]:
        """Extract project metadata from package config files."""
        meta: Dict[str, Any] = {
            "project_name": "", "description": "", "dependencies": [],
            "entry_points": [], "mcp_tools": [], "ecosystem": "unknown",
        }

        # pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            import tomllib
            try:
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                proj = data.get("project", {})
                meta["project_name"] = proj.get("name", "")
                meta["description"] = proj.get("description", "")
                meta["dependencies"] = [
                    d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
                    for d in proj.get("dependencies", [])
                ]
                meta["entry_points"] = list(proj.get("scripts", {}).keys())
                meta["ecosystem"] = "python"
            except Exception:
                pass

        # package.json
        pkg_json = repo_path / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text(errors="replace"))
                meta["project_name"] = meta["project_name"] or data.get("name", "")
                meta["description"] = meta["description"] or data.get("description", "")
                meta["dependencies"] = meta["dependencies"] or list(data.get("dependencies", {}).keys())
                if not meta["entry_points"]:
                    bin_data = data.get("bin", {})
                    if isinstance(bin_data, dict):
                        meta["entry_points"] = list(bin_data.keys())
                    elif isinstance(bin_data, str):
                        package_name = data.get("name", "") or meta["project_name"]
                        meta["entry_points"] = [package_name] if package_name else []
                if meta["ecosystem"] == "unknown":
                    meta["ecosystem"] = "node"
            except (json.JSONDecodeError, Exception):
                pass

        # setup.py fallback
        if meta["ecosystem"] == "unknown":
            setup_py = repo_path / "setup.py"
            if setup_py.exists():
                import re
                meta["ecosystem"] = "python"
                text = setup_py.read_text(errors="replace")[:3000]
                m = re.search(r'name\s*=\s*["\']([^"\']+)', text)
                if m:
                    meta["project_name"] = m.group(1)
                m = re.search(r'description\s*=\s*["\']([^"\']+)', text)
                if m:
                    meta["description"] = m.group(1)

        # MCP config files
        for mcp_file in ["mcp.json", ".mcp", "claude_desktop_config.json"]:
            p = repo_path / mcp_file
            if p.exists():
                try:
                    data = json.loads(p.read_text(errors="replace"))
                    meta["mcp_tools"] = list(data.get("tools", {}).keys())
                except Exception:
                    pass

        return meta

    async def ingest_github_repo(
        self, repo_url: str, label: str = "", depth: str = "readme+code",
    ) -> Dict[str, Any]:
        """Cleanroom ingest a GitHub repo: clone, AST-scan, store concepts, delete clone.

        Stores distilled capabilities and architecture as concepts — no code is retained.
        depth: "readme" (README only), "readme+code" (AST scan), "full" (README + code + docs)
        """
        import hashlib
        import shutil
        import subprocess
        import tempfile
        from researcher.synthesizer import Synthesizer

        # Parse repo URL
        repo_url = repo_url.rstrip("/")
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
        if len(parts) < 2:
            return {"error": f"Invalid GitHub URL: {repo_url}"}
        owner, repo_name = parts[0], parts[1]
        repo_key = f"{owner}/{repo_name}"
        entry_id = f"ghrepo_{hashlib.sha256(repo_key.encode()).hexdigest()[:12]}"

        # Clone to temp dir (shallow)
        tmp_dir = tempfile.mkdtemp(prefix="researcher_gh_")
        try:
            proc = subprocess.run(
                ["git", "clone", "--depth=1", f"https://github.com/{repo_key}.git", tmp_dir],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                return {"error": f"Clone failed: {proc.stderr[:200]}"}

            repo_path = Path(tmp_dir)

            # Extract package metadata
            pkg_meta = self._extract_package_metadata(repo_path)
            if not label and pkg_meta["description"]:
                label = pkg_meta["description"]

            # Read README
            readme = ""
            for name in ["README.md", "readme.md", "README.rst", "README"]:
                p = repo_path / name
                if p.exists():
                    readme = p.read_text(errors="replace")[:4000]
                    break

            code_capabilities = []
            readme_claims = []
            imports_from = {}
            architecture = ""

            # Extract README claims via LLM
            if readme:
                readme_claims = await self._extract_readme_claims(readme)

            if depth in ("readme+code", "full"):
                # AST scan — same as scan_codebase but on temp dir
                synth = Synthesizer(self.knowledge, self.triples, self.pool)
                result = await synth.scan_codebase(
                    project_name=repo_key,
                    repo_path=str(repo_path),
                    description=label or readme[:500],
                    dependencies=", ".join(pkg_meta["dependencies"][:15]),
                )
                if result.success:
                    content = result.content.strip()
                    if content.startswith("```"):
                        content = "\n".join(content.split("\n")[1:])
                    if content.endswith("```"):
                        content = "\n".join(content.split("\n")[:-1])
                    try:
                        scan_data = json.loads(content)
                        code_capabilities = scan_data.get("capabilities", [])
                        imports_from = scan_data.get("imports_from", {})
                        architecture = scan_data.get("architecture", "")
                    except json.JSONDecodeError:
                        pass

            # Scan documentation files for full depth
            doc_summary = ""
            if depth == "full":
                doc_files = []
                for doc_name in ["ARCHITECTURE.md", "CONTRIBUTING.md", "API.md", "DESIGN.md"]:
                    p = repo_path / doc_name
                    if p.exists():
                        doc_files.append((doc_name, p.read_text(errors="replace")[:3000]))
                docs_dir = repo_path / "docs"
                if docs_dir.is_dir():
                    for doc_path in sorted(docs_dir.rglob("*.md"))[:10]:
                        rel = doc_path.relative_to(repo_path)
                        doc_files.append((str(rel), doc_path.read_text(errors="replace")[:2000]))
                if doc_files:
                    doc_texts = [f"--- {name} ---\n{content}" for name, content in doc_files[:8]]
                    combined = "\n\n".join(doc_texts)[:8000]
                    client = self.pool.get_client("extractor")
                    try:
                        doc_summary = await client.generate(
                            prompt=f"Summarize the key technical decisions, patterns, and capabilities described in these project documents:\n\n{combined}",
                            system="Summarize technical documentation concisely. Focus on architecture, APIs, and design decisions.",
                            temperature=0.1,
                            max_tokens=1000,
                        )
                        doc_summary = doc_summary.strip()
                    except Exception:
                        pass

            # Merge capabilities: code-verified first, then README-only claims
            code_set = {c.lower() for c in code_capabilities}
            readme_only = [c for c in readme_claims if c.lower() not in code_set]
            capabilities = code_capabilities + readme_only

            # Build concept summary (no code stored)
            summary_parts = [f"GitHub: {repo_key}"]
            if label:
                summary_parts.append(f"Description: {label}")
            if readme:
                # Distill README to first paragraph
                first_para = readme.split("\n\n")[0][:500] if readme else ""
                summary_parts.append(f"README: {first_para}")
            if capabilities:
                summary_parts.append(f"Capabilities: {', '.join(capabilities)}")
            if architecture:
                summary_parts.append(f"Architecture: {architecture}")
            if pkg_meta["entry_points"]:
                summary_parts.append(f"Entry points: {', '.join(pkg_meta['entry_points'])}")
            if pkg_meta["mcp_tools"]:
                summary_parts.append(f"MCP tools: {', '.join(pkg_meta['mcp_tools'])}")
            if doc_summary:
                summary_parts.append(f"Documentation: {doc_summary[:1000]}")

            # Store as knowledge entry
            entry = KnowledgeEntry(
                id=entry_id,
                tier=Tier.DERIVED,
                title=f"GitHub: {repo_key}",
                content="\n".join(summary_parts),
                source=repo_url,
                scope="external",
                tags=["github", "external", "concepts"],
                status=EntryStatus.DISTILLED,
                metadata={
                    "repo": repo_key,
                    "url": repo_url,
                    "capabilities": capabilities,
                    "code_capabilities": code_capabilities,
                    "readme_claims": readme_claims,
                    "readme_only_claims": readme_only,
                    "imports_from": imports_from,
                    "architecture": architecture,
                    "package": pkg_meta,
                    "doc_summary": doc_summary or None,
                    "depth": depth,
                },
            )
            self.knowledge.add(entry)

            # Score relevance against configured projects
            scores = await self.score_relevance(entry_id)

            # Store triples — differentiate code-verified vs README claims
            readme_set = {c.lower() for c in readme_claims}
            for cap in code_capabilities:
                self.triples.add(
                    subject=repo_key,
                    predicate="implements",
                    obj=cap,
                    confidence=0.95 if cap.lower() in readme_set else 0.85,
                    source=f"github:{repo_key}",
                )
            for claim in readme_only:
                self.triples.add(
                    subject=repo_key,
                    predicate="claims",
                    obj=claim,
                    confidence=0.6,
                    source=f"github:{repo_key}",
                )

            self.digest.record(
                summary=f"Ingested GitHub repo {repo_key}: {len(capabilities)} capabilities extracted",
                source="pipeline",
                audience="research",
                tags=["github", "ingest"],
                metadata={"repo": repo_key, "cap_count": len(capabilities)},
            )

            return {
                "repo": repo_key,
                "entry_id": entry_id,
                "capabilities": capabilities,
                "code_capabilities": code_capabilities,
                "readme_claims": readme_claims,
                "readme_only_claims": readme_only,
                "imports_from": imports_from,
                "architecture": architecture,
                "depth": depth,
                "relevance_scores": scores,
            }

        finally:
            # Always delete the clone — cleanroom
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def scan_codebase(self, project: str) -> Dict[str, Any]:
        """Scan a project's codebase and store discovered capabilities.

        Reads repo from config, extracts capabilities via LLM, stores them
        as 'exists' capability entries in the knowledge store.
        Returns {capabilities, imports_from, stored}.
        """
        from researcher.synthesizer import Synthesizer

        projects = self.config.get("projects", {})
        if project not in projects:
            return {"error": f"Unknown project: {project}"}

        cfg = projects.get(project, {})

        # Check DB registry first, fall back to config
        repo_entry = self.knowledge.get(f"repo_{project}")
        if repo_entry and repo_entry.metadata:
            repo_path = repo_entry.metadata.get("repo_path", "")
        else:
            repo_path = cfg.get("repo", "")
        if not repo_path:
            return {"error": f"No repo path for {project}. Use register_repo() first."}

        synth = Synthesizer(self.knowledge, self.triples, self.pool)
        result = await synth.scan_codebase(
            project_name=project,
            repo_path=repo_path,
            description=cfg.get("description", ""),
            dependencies=", ".join(cfg.get("depends_on", [])),
        )

        if not result.success:
            return {"error": result.content}

        # Parse JSON
        content = result.content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])

        try:
            scan_data = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "LLM returned non-JSON", "raw": result.content}

        # Store capabilities
        import hashlib
        stored = 0
        for cap_name in scan_data.get("capabilities", []):
            cap_id = f"cap_{project}_{hashlib.sha256(cap_name.encode()).hexdigest()[:8]}"
            existing = self.knowledge.get(cap_id)
            if existing:
                continue  # Don't overwrite
            entry = KnowledgeEntry(
                id=cap_id,
                tier=Tier.DERIVED,
                title=cap_name,
                content=f"exists: {cap_name}",
                source="codebase_scan",
                scope="capability",
                tags=["capability", f"cap:{project}", "cap:exists"],
                status=EntryStatus.DISTILLED,
                metadata={
                    "target": project,
                    "concept": cap_name.lower(),
                    "capability_status": "exists",
                    "source": "codebase_scan",
                },
            )
            self.knowledge.add(entry)
            stored += 1

        # Store cross-dependencies as triples
        for dep_project, usages in scan_data.get("imports_from", {}).items():
            for usage in usages:
                self.triples.add(
                    subject=project,
                    predicate="imports_from",
                    obj=f"{dep_project}:{usage}",
                    confidence=0.9,
                    source=f"scan:{project}",
                )

        self.digest.record(
            summary=f"Scanned {project}: {len(scan_data.get('capabilities', []))} capabilities found, {stored} new stored",
            source="pipeline",
            audience="research",
            tags=["scan", "capabilities"],
            metadata={"project": project, "stored": stored},
        )

        return {
            "capabilities": scan_data.get("capabilities", []),
            "imports_from": scan_data.get("imports_from", {}),
            "stored": stored,
        }

    async def research_from_capabilities(
        self, project: Optional[str] = None, num_queries: int = 5,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Generate research queries from scanned capabilities and search for papers.

        Reads stored capabilities, clusters them into search themes via LLM,
        then searches arxiv/semantic scholar for each theme.
        """
        from researcher.synthesizer import Synthesizer
        from researcher.search_engines import search_papers

        # Load capabilities
        caps_by_project: Dict[str, List[str]] = {}
        for entry in self.knowledge.get_by_tier(Tier.DERIVED):
            if not entry.tags or "capability" not in entry.tags:
                continue
            if "cap:exists" not in entry.tags:
                continue
            for tag in entry.tags:
                if tag.startswith("cap:") and tag not in ("cap:exists", "cap:planned"):
                    proj = tag.replace("cap:", "")
                    if project and proj != project:
                        continue
                    caps_by_project.setdefault(proj, []).append(entry.title)

        if not caps_by_project:
            return {"error": f"No capabilities found{f' for {project}' if project else ''}. Run scan_codebase first."}

        synth = Synthesizer(self.knowledge, self.triples, self.pool)
        projects_cfg = self.config.get("projects", {})
        all_results = []

        for proj, caps in caps_by_project.items():
            desc = projects_cfg.get(proj, {}).get("description", "")
            queries = await synth.generate_research_queries(
                project_name=proj,
                capabilities=caps,
                description=desc,
                num_queries=num_queries,
            )

            for q in queries:
                query_str = q.get("query", "")
                if not query_str:
                    continue
                try:
                    results = await search_papers(
                        query_str, max_results=max_results,
                    )
                    all_results.append({
                        "project": proj,
                        "query": query_str,
                        "targets": q.get("targets", ""),
                        "rationale": q.get("rationale", ""),
                        "papers": [
                            {"title": r.title, "url": r.url, "source": r.source}
                            for r in results[:max_results]
                        ],
                    })
                except Exception as e:
                    logger.warning("Search failed for '%s': %s", query_str, e)

        self.digest.record(
            summary=f"Research from capabilities: {len(all_results)} queries across {list(caps_by_project.keys())}",
            source="pipeline",
            audience="research",
            tags=["research", "capabilities"],
        )

        return {"queries": all_results, "project_count": len(caps_by_project)}

    async def evaluate_capability(self, capability_description: str) -> Dict[str, Any]:
        """Evaluate whether a new khonliang capability could improve the researcher.

        Returns parsed JSON assessment or raw content if parsing fails.
        """
        from researcher.synthesizer import Synthesizer

        synth = Synthesizer(self.knowledge, self.triples, self.pool)
        result = await synth.evaluate_capability(capability_description)

        if not result.success:
            return {"error": result.content}

        content = result.content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])

        try:
            evaluation = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "LLM returned non-JSON", "raw": result.content}

        self.digest.record(
            summary=f"Evaluated capability: {evaluation.get('summary', capability_description[:80])}",
            source="pipeline",
            audience="research",
            tags=["evaluate", "capability"],
            metadata={"score": evaluation.get("score", 0), "applicable": evaluation.get("applicable", False)},
        )

        return evaluation

    async def review_frs(
        self, target: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Review stored FRs with the large reviewer model.

        Updates each FR entry with review verdict and metadata.
        Returns list of {fr_id, verdict, reasoning, ...} dicts.
        """
        from researcher.synthesizer import Synthesizer

        synth = Synthesizer(self.knowledge, self.triples, self.pool)
        projects = self.config.get("projects", {})
        frs = self.get_feature_requests(target=target)

        if not frs:
            return []

        results = []
        for fr in frs:
            fr_target = fr.get("target", "")
            project_cfg = projects.get(fr_target, {})
            if not project_cfg:
                continue

            logger.info("Reviewing FR: %s → %s", fr.get("title", "?"), fr_target)

            review = await synth.review_fr(
                fr_data=fr,
                project_config=project_cfg,
                project_name=fr_target,
            )

            # Update the FR entry with review results
            entry = self.knowledge.get(fr["id"])
            if entry:
                entry.metadata["review"] = review
                entry.metadata["review_verdict"] = review.get("verdict", "error")
                # If revised, update the FR content
                if review.get("verdict") == "revise":
                    fr_content = json.loads(entry.content)
                    if review.get("revised_title"):
                        fr_content["original_title"] = fr_content.get("title", "")
                        fr_content["title"] = review["revised_title"]
                    if review.get("revised_description"):
                        fr_content["original_description"] = fr_content.get("description", "")
                        fr_content["description"] = review["revised_description"]
                    if review.get("revised_priority"):
                        fr_content["priority"] = review["revised_priority"]
                    entry.content = json.dumps(fr_content, indent=2)
                    entry.title = review.get("revised_title", entry.title)
                self.knowledge.add(entry)

            results.append({
                "fr_id": fr["id"],
                "title": fr.get("title", ""),
                "target": fr_target,
                **review,
            })

        accepted = sum(1 for r in results if r.get("verdict") == "accept")
        revised = sum(1 for r in results if r.get("verdict") == "revise")
        rejected = sum(1 for r in results if r.get("verdict") == "reject")

        self.digest.record(
            summary=f"FR review: {accepted} accepted, {revised} revised, {rejected} rejected",
            source="pipeline",
            audience="research",
            tags=["review", "fr"],
            metadata={"accepted": accepted, "revised": revised, "rejected": rejected},
        )

        return results

    def get_feature_requests(
        self, target: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query stored feature requests, optionally filtered by target project."""
        frs = []
        for entry in self.knowledge.get_by_tier(Tier.DERIVED):
            tags = entry.tags or []
            if "fr" not in tags:
                continue
            if "fr:archived" in tags or "fr:completed" in tags:
                continue
            status = (entry.metadata or {}).get("fr_status", "open")
            if status in ("completed", "archived"):
                continue
            if target and f"target:{target}" not in tags:
                continue
            try:
                fr_data = json.loads(entry.content)
            except json.JSONDecodeError:
                fr_data = {"title": entry.title}
            frs.append({
                "id": entry.id,
                "concept": entry.metadata.get("concept", ""),
                "classification": entry.metadata.get("classification", ""),
                "target": entry.metadata.get("target", ""),
                "priority": entry.metadata.get("priority", "medium"),
                "review_verdict": entry.metadata.get("review_verdict", ""),
                "review": entry.metadata.get("review", {}),
                **fr_data,
            })
        return frs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> List[KnowledgeEntry]:
        """Search papers and summaries."""
        return self.knowledge.search(query, scope="research", limit=limit)

    def get_reading_list(self) -> Dict[str, List[Dict]]:
        """Return papers grouped by status."""
        def _info(entry):
            return {
                "entry_id": entry.id,
                "title": entry.title,
                "url": entry.metadata.get("url", ""),
                "status": entry.status,
                "tags": entry.tags,
            }

        pending = [_info(e) for e in self.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED)]
        processing = [_info(e) for e in self.knowledge.get_by_status(EntryStatus.PROCESSING, tier=Tier.IMPORTED)]
        distilled = [_info(e) for e in self.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED)]
        failed = [_info(e) for e in self.knowledge.get_by_status(EntryStatus.FAILED, tier=Tier.IMPORTED)]
        skipped = [_info(e) for e in self.knowledge.get_by_status(EntryStatus.SKIPPED, tier=Tier.IMPORTED)]

        return {"pending": pending + processing, "distilled": distilled, "failed": failed, "skipped": skipped}

    def get_paper_context(self, query: str) -> str:
        """Build prompt context from relevant papers and triples."""
        kb_context = self.knowledge.build_context(
            query, scope="research", max_chars=4000
        )
        triple_context = self.triples.build_context(
            subjects=None, max_triples=15, min_confidence=0.3
        )
        parts = []
        if kb_context:
            parts.append(f"## Relevant Papers\n{kb_context}")
        if triple_context:
            parts.append(f"## Known Relationships\n{triple_context}")
        return "\n\n".join(parts)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def create_pipeline(config_path: str = "config.yaml") -> ResearchPipeline:
    """Factory: create a fully wired pipeline from config."""
    config = load_config(config_path)

    # Resolve relative paths against config file's directory so the server
    # works regardless of the caller's cwd.
    config_dir = Path(config_path).resolve().parent

    db_path = config.get("db_path", "data/researcher.db")
    if not Path(db_path).is_absolute():
        db_path = str(config_dir / db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    knowledge = KnowledgeStore(db_path)
    predicate_aliases = config.get("predicate_aliases", {})
    triples = TripleStore(db_path, predicate_aliases=predicate_aliases)
    digest = DigestStore(db_path)

    role_model_map = {
        "summarizer": config.get("models", {}).get("summarizer", "qwen2.5:7b"),
        "extractor": config.get("models", {}).get("extractor", "llama3.2:3b"),
        "assessor": config.get("models", {}).get("assessor", "llama3.2:3b"),
        "idea_parser": config.get("models", {}).get("idea_parser", "llama3.2:3b"),
        "reviewer": config.get("models", {}).get("reviewer", "qwen2.5:32b"),
    }
    ollama_url = config.get("ollama_url", "http://localhost:11434")
    model_timeouts = config.get("model_timeouts", {})
    pool = ModelPool(
        role_model_map,
        base_url=ollama_url,
        model_timeouts=model_timeouts or None,
    )

    return ResearchPipeline(
        knowledge=knowledge,
        triples=triples,
        digest=digest,
        pool=pool,
        config=config,
    )
