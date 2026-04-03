"""Research pipeline: fetch → ingest → distill → extract triples.

Orchestrates khonliang components:
  - ResearchPool for threaded fetch/parse
  - KnowledgeStore for paper storage (Tier 2 raw, Tier 3 summaries)
  - TripleStore for relationship extraction
  - DigestStore for activity tracking
  - Roles for LLM distillation
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from khonliang.knowledge.store import KnowledgeStore, KnowledgeEntry, Tier
from khonliang.knowledge.triples import TripleStore
from khonliang.digest.store import DigestStore
from khonliang.pool import ModelPool
from khonliang.research.pool import ResearchPool
from khonliang.research.models import ResearchTask

from researcher.fetcher import extract_arxiv_id, fetch_arxiv, fetch_url
from researcher.parser import parse_paper_list, PaperReference
from researcher.queue import PaperFetcher, ListParser
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

        # Research pool for threaded fetching
        self.research_pool = ResearchPool()
        self.research_pool.register(PaperFetcher())

        parser_client = pool.get_client("extractor")
        self.research_pool.register(ListParser(llm_client=parser_client))

        # URL dedup: check knowledge store for existing entries
        self._url_index: Dict[str, str] = {}  # url -> entry_id
        self._build_url_index()

    def _build_url_index(self):
        """Build URL index from existing knowledge entries."""
        for entry in self.knowledge.get_by_tier(Tier.IMPORTED):
            url = entry.metadata.get("url", "")
            if url:
                self._url_index[url] = entry.id

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

        # Normalize arxiv URLs
        arxiv_id = extract_arxiv_id(url)
        if arxiv_id:
            result = await fetch_arxiv(url)
        else:
            result = await fetch_url(url)

        if not result.content.strip():
            logger.warning("Empty content from %s", url)
            return None

        # Store as Tier 2
        import hashlib
        entry_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        entry = KnowledgeEntry(
            id=entry_id,
            tier=Tier.IMPORTED,
            title=result.title or url,
            content=result.content,
            source=url,
            scope="research",
            tags=["paper", "undistilled"],
            metadata={
                "url": url,
                "fetched_at": result.fetched_at,
                **result.metadata,
            },
        )
        self.knowledge.add(entry)
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
    # Distillation
    # ------------------------------------------------------------------

    async def distill(self, entry_id: str) -> DistillResult:
        """Run full distillation pipeline on a stored paper."""
        entry = self.knowledge.get(entry_id)
        if not entry:
            return DistillResult(entry_id=entry_id, title="NOT FOUND")

        result = DistillResult(entry_id=entry_id, title=entry.title)

        # Step 1: Summarize
        summary_resp = await self.summarizer.handle(entry.content)
        if summary_resp.get("success"):
            result.summary = summary_resp["summary"]
        else:
            logger.warning("Summarization failed for %s", entry_id)
            return result

        # Step 2: Extract triples from summary
        summary_text = json.dumps(result.summary, indent=2)
        extract_resp = await self.extractor.handle(summary_text)
        if extract_resp.get("success"):
            result.triples = extract_resp["triples"]

        # Step 3: Assess applicability per project
        projects = self.config.get("projects", {})
        for project_name, project_cfg in projects.items():
            assess_resp = await self.assessor.handle(
                summary_text,
                context={"project_description": project_cfg.get("description", "")},
            )
            if assess_resp.get("success"):
                result.assessments[project_name] = assess_resp["assessment"]

        # Store results
        self._store_distillation(entry, result)
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
                tags=["summary", "distilled"],
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

        # Update original entry tags
        tags = entry.tags or []
        if "undistilled" in tags:
            tags.remove("undistilled")
        tags.append("distilled")
        # Re-add with updated tags
        entry.tags = tags
        self.knowledge.add(entry)

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
        for entry in self.knowledge.get_by_tier(Tier.IMPORTED):
            if "undistilled" in (entry.tags or []):
                result = await self.distill(entry.id)
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> List[KnowledgeEntry]:
        """Search papers and summaries."""
        return self.knowledge.search(query, scope="research", limit=limit)

    def get_reading_list(self) -> Dict[str, List[Dict]]:
        """Return papers grouped by status."""
        undistilled = []
        distilled = []

        for entry in self.knowledge.get_by_tier(Tier.IMPORTED):
            info = {
                "entry_id": entry.id,
                "title": entry.title,
                "url": entry.metadata.get("url", ""),
                "tags": entry.tags,
            }
            if "undistilled" in (entry.tags or []):
                undistilled.append(info)
            else:
                distilled.append(info)

        return {"pending": undistilled, "distilled": distilled}

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

    db_path = config.get("db_path", "data/researcher.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    knowledge = KnowledgeStore(db_path)
    triples = TripleStore(db_path)
    digest = DigestStore(db_path)

    role_model_map = {
        "summarizer": config.get("models", {}).get("summarizer", "qwen2.5:7b"),
        "extractor": config.get("models", {}).get("extractor", "llama3.2:3b"),
        "assessor": config.get("models", {}).get("assessor", "llama3.2:3b"),
    }
    ollama_url = config.get("ollama_url", "http://localhost:11434")
    pool = ModelPool(role_model_map, base_url=ollama_url)

    return ResearchPipeline(
        knowledge=knowledge,
        triples=triples,
        digest=digest,
        pool=pool,
        config=config,
    )
