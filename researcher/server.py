"""MCP server for khonliang-researcher.

Extends KhonliangMCPServer with custom research tools.
Standard knowledge_search, triple_query, etc. come free from khonliang.

Usage:
    python -m researcher.server
    python -m researcher.server --db data/researcher.db --config config.yaml
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from khonliang.knowledge.store import Tier, EntryStatus
from khonliang.mcp import KhonliangMCPServer, compact_list, compact_entry, truncate, brief_or_full, format_response

from researcher.pipeline import create_pipeline, ResearchPipeline

logger = logging.getLogger(__name__)


def _compact_field(value: Any, limit: int = 80) -> str:
    """Sanitize a value for inclusion in pipe-delimited compact output.

    Replaces newlines and pipe characters that would otherwise corrupt
    downstream parsing, then truncates to ``limit`` chars.
    """
    if value is None:
        return "?"
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "/")
    return truncate(text.strip(), limit)


def create_research_server(pipeline: ResearchPipeline):
    """Create MCP server with khonliang base tools + custom research tools."""
    from researcher.synthesizer import Synthesizer
    from researcher.worker import DistillWorker

    base = KhonliangMCPServer(
        knowledge_store=pipeline.knowledge,
        triple_store=pipeline.triples,
    )
    base.add_guide("research_guide", "how to discover, distill, and use research papers")
    mcp = base.create_app()

    _RESEARCH_GUIDE = """\
# Research Pipeline Guide

## Quick start
1. `find_papers(query)` — search arxiv + semantic scholar
2. `fetch_paper(url)` — ingest a paper into the knowledge store
3. `start_distillation()` — summarize + extract triples + score relevance
4. `feature_requests(target="your_project")` — see what emerged

## Discovery
- `find_papers(query, engines="arxiv,semantic_scholar")` — search by keyword
- `browse_feeds(query, relevant_only=True)` — scan RSS feeds for new posts
- `research_capabilities(project)` — generate queries from scanned code capabilities
- `scan_codebase(project)` — AST-scan a project to discover what it already does
- `ingest_github(repo_url)` — cleanroom-extract concepts from a GitHub repo

## Ingestion
- `fetch_paper(url)` — single paper (arxiv, HTML, PDF)
- `fetch_papers_batch(urls)` — comma-separated URLs, concurrent fetch
- `fetch_paper_list(url)` — parse an awesome-list or bibliography page
- `ingest_file(path)` — local file (PDF, HTML, Markdown, text)
- `ingest_idea(text)` — decompose informal text into researchable claims

## Distillation
- `start_distillation(batch_size)` — process pending queue (0 = all)
- `distill_paper(entry_id)` — distill a single paper
- `worker_status()` — check queue depth and progress
- `reading_list(detail)` — see pending/distilled/failed/skipped counts

Papers below the relevance threshold are auto-skipped.
Each distilled paper produces: structured summary, relationship triples,
and per-project applicability scores.

## Exploration
- `knowledge_search(query)` — full-text search across all stored content
- `find_relevant(query, project)` — search filtered by project relevance
- `concept_tree(concept)` — trace a concept's connections as a tree
- `concept_path(start, end)` — find how two concepts connect
- `concept_matrix(min_connections)` — entity × document coverage matrix
- `concepts_for_project(project)` — concepts ranked by project relevance
- `paper_context(query)` — build rich context for prompt injection
- `triple_query(subject)` — query the relationship graph directly

## Synthesis
- `synthesize_topic(topic)` — cross-paper analysis of a theme
- `synthesize_project(project)` — applicability brief for a project
- `synthesize_landscape()` — map major directions, trends, gaps
- `project_landscape(project)` — per-project view with concepts, FRs, capabilities

## Feature Requests
- `synergize()` — classify concepts and auto-generate FRs
- `feature_requests(target)` — list FRs for a project
- `next_fr(target)` — highest priority unblocked FR
- `review_feature_requests(target)` — deep-review with 32B model
- `promote_fr(...)` — manually create a vetted FR
- `fr_workflow()` — full lifecycle protocol (open → planned → completed)
- `project_capabilities(target)` — what exists vs what's planned

## Ideas Pipeline
- `ingest_idea(text)` — parse claims + generate search queries
- `research_idea(idea_id)` — find papers for each claim
- `brief_idea(idea_id)` — synthesize claim-by-claim assessment

## Evaluation
- `evaluate_capability(capability)` — assess if a khonliang feature helps researcher
- `score_relevance(entry_id)` — score a paper against all projects
- `health_check()` — verify Ollama, models, DB, disk

## detail parameter
Most tools accept detail="compact|brief|full":
- compact: key=value pairs for agent loops
- brief: structured one-line-per-item (default)
- full: rich detail with context"""

    @mcp.tool()
    async def research_guide() -> str:
        """how to discover, distill, and use research papers"""
        return _RESEARCH_GUIDE

    synthesizer = Synthesizer(pipeline.knowledge, pipeline.triples, pipeline.pool)
    worker = DistillWorker(pipeline)

    # ------------------------------------------------------------------
    # Capability tracking helpers
    # ------------------------------------------------------------------

    def _update_capability_status(target: str, title: str, concept: str, status: str, fr_id: str = ""):
        """Track what exists/is planned per project. Auto-updated on FR status changes."""
        from researcher.pipeline import update_capability_status
        update_capability_status(pipeline.knowledge, target, title, concept, status, fr_id)

    # ------------------------------------------------------------------
    # Custom research tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def fetch_paper(url: str) -> str:
        """Fetch and store a research paper by URL or arxiv ID.

        Supports arxiv HTML pages, abstract pages, and generic URLs.
        Returns the entry ID for later distillation.
        """
        try:
            entry_id = await pipeline.ingest_paper(url)
            if entry_id:
                entry = pipeline.knowledge.get(entry_id)
                title = entry.title if entry else url
                size = len(entry.content) if entry and entry.content else 0
                if size < 500:
                    return f"WARNING: {title}\nEntry ID: {entry_id}\nContent: {size} chars — likely abstract-only or fetch failed"
                return f"Ingested: {title}\nEntry ID: {entry_id}\nContent: {size:,} chars"
            return f"Failed to ingest: {url}"
        except Exception as e:
            return f"Error fetching {url}: {e}"

    @mcp.tool()
    async def ingest_file(path: str) -> str:
        """Ingest a local file (PDF, HTML, Markdown, or text).

        Reads the file, auto-detects format, extracts text, and stores it.
        Returns the entry ID for later distillation.
        """
        from researcher.fetcher import fetch_file
        try:
            result = await fetch_file(path)
            if not result.content.strip():
                return f"No text content extracted from: {path}"

            import hashlib
            from khonliang.knowledge.store import KnowledgeEntry, Tier, EntryStatus

            entry_id = hashlib.sha256(path.encode()).hexdigest()[:16]
            entry = KnowledgeEntry(
                id=entry_id,
                tier=Tier.IMPORTED,
                title=result.title or path,
                content=result.content,
                source=result.url,
                scope="research",
                tags=["paper", f"format:{result.format.value}"],
                status=EntryStatus.INGESTED,
                metadata={
                    "url": result.url,
                    "format": result.format.value,
                    "fetched_at": result.fetched_at,
                    **result.metadata,
                },
            )
            pipeline.knowledge.add(entry)
            return f"Ingested: {result.title} ({result.format.value})\nEntry ID: {entry_id}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading {path}: {e}"

    @mcp.tool()
    async def fetch_paper_list(url: str) -> str:
        """Parse a URL containing a list of papers (awesome-list, bibliography, etc.).

        Uses LLM to understand the list format and extract paper references.
        Returns the discovered papers with titles, URLs, and categories.
        """
        try:
            papers = await pipeline.ingest_paper_list(url)
            if not papers:
                return "No papers found in the list."

            lines = [f"Found {len(papers)} papers:\n"]
            current_cat = ""
            for p in papers:
                if p.category != current_cat:
                    current_cat = p.category
                    lines.append(f"\n## {current_cat}")
                lines.append(f"- {p.title or '(untitled)'}: {p.url}")

            lines.append(
                f"\nUse fetch_paper(url) to ingest individual papers, "
                f"or fetch_papers_batch(urls) to fetch multiple."
            )
            return "\n".join(lines)
        except Exception as e:
            return f"Error parsing list from {url}: {e}"

    @mcp.tool()
    async def fetch_papers_batch(urls: str) -> str:
        """Fetch multiple papers concurrently. Pass comma-separated URLs."""
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        if not url_list:
            return "No URLs provided."

        from researcher.parser import PaperReference
        refs = [PaperReference(title="", url=u) for u in url_list]
        entry_ids = await pipeline.ingest_papers_from_list(refs)
        return f"Ingested {len(entry_ids)} of {len(url_list)} papers.\nEntry IDs: {', '.join(entry_ids)}"

    @mcp.tool()
    async def distill_paper(entry_id: str) -> str:
        """Run LLM distillation on a stored paper.

        Produces: structured summary, relationship triples,
        and applicability assessments for configured projects.
        """
        result = await pipeline.distill(entry_id)
        if not result.success:
            return f"Distillation failed for {entry_id}: {result.title}"

        parts = [f"# {result.title}\n"]

        if result.summary:
            parts.append("## Summary")
            parts.append(json.dumps(result.summary, indent=2))

        if result.triples:
            parts.append(f"\n## Relationships ({len(result.triples)} triples)")
            for t in result.triples:
                conf = t.get("confidence", 0)
                parts.append(
                    f"- {t.get('subject', '?')} "
                    f"—[{t.get('predicate', '?')}]→ "
                    f"{t.get('object', '?')} ({conf:.0%})"
                )

        if result.assessments:
            parts.append("\n## Applicability")
            for project, assessment in result.assessments.items():
                if isinstance(assessment, dict):
                    score = assessment.get("score", 0)
                    reasoning = assessment.get("reasoning", "")
                    parts.append(f"\n### {project} (score: {score:.0%})")
                    parts.append(reasoning)
                    ideas = assessment.get("implementation_ideas", [])
                    if ideas:
                        parts.append("Implementation ideas:")
                        for idea in ideas:
                            parts.append(f"  - {idea}")

        return "\n".join(parts)

    @mcp.tool()
    async def distill_pending() -> str:
        """Distill all papers that haven't been processed yet."""
        results = await pipeline.distill_all_pending()
        if not results:
            return "No pending papers to distill."

        succeeded = sum(1 for r in results if r.success)
        lines = [f"Distilled {succeeded}/{len(results)} papers:\n"]
        for r in results:
            status = "ok" if r.success else "FAILED"
            lines.append(f"  [{status}] {r.title}")
        return "\n".join(lines)

    @mcp.tool()
    async def research_capabilities(
        project: str = "", num_queries: int = 5, max_results: int = 5,
    ) -> str:
        """Search for papers that could improve a project based on its scanned capabilities.

        Uses AST-scanned capabilities to generate targeted search queries,
        then searches arxiv/semantic scholar. Run scan_codebase first.
        """
        result = await pipeline.research_from_capabilities(
            project=project or None,
            num_queries=num_queries,
            max_results=max_results,
        )
        if "error" in result:
            return result["error"]

        lines = []
        for q in result.get("queries", []):
            lines.append(f"## [{q['project']}] {q['query']}")
            if q.get("rationale"):
                lines.append(f"  Why: {q['rationale']}")
            for p in q.get("papers", []):
                lines.append(f"  - {p['title']} [{p['source']}]")
                lines.append(f"    {p['url']}")
            lines.append("")

        if not lines:
            return "No queries generated. Check capabilities with project_capabilities()."
        return "\n".join(lines)

    @mcp.tool()
    async def find_papers(query: str, max_results: int = 20, engines: str = "") -> str:
        """Search for research papers across multiple sources.

        Searches arxiv and Semantic Scholar in parallel, deduplicates results.
        Pass comma-separated engine names to filter: "arxiv", "semantic_scholar"
        Use fetch_paper(url) to ingest any interesting results.
        """
        from researcher.search_engines import search_papers

        engine_list = [e.strip() for e in engines.split(",") if e.strip()] or None
        try:
            results = await search_papers(query, engines=engine_list, max_results=max_results)
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return f"No papers found for: {query}"

        lines = [f"Found {len(results)} papers for '{query}':\n"]
        for i, r in enumerate(results, 1):
            authors = ", ".join(r.metadata.get("authors", [])[:3])
            if len(r.metadata.get("authors", [])) > 3:
                authors += f" +{len(r.metadata['authors']) - 3} more"
            lines.append(f"{i}. **{r.title}** [{r.source}]")
            if authors:
                lines.append(f"   {authors}")
            if r.content:
                lines.append(f"   {r.content[:200]}...")
            lines.append(f"   {r.url}")
            lines.append("")

        return "\n".join(lines)

    @mcp.tool()
    async def browse_feeds(query: str = "", feeds: str = "", relevant_only: bool = False) -> str:
        """Browse RSS feeds. query filters by keyword. relevant_only=true
        pre-filters by embedding relevance to configured projects.
        """
        from researcher.rss import fetch_all_feeds

        feed_list = [f.strip() for f in feeds.split(",") if f.strip()] or None
        entries = await fetch_all_feeds(feed_list)

        if query:
            keywords = query.lower().split()
            entries = [
                e for e in entries
                if any(kw in f"{e.title} {e.content}".lower() for kw in keywords)
            ]

        # RR-9: Relevance pre-filtering
        if relevant_only and entries:
            scored = []
            for e in entries[:100]:  # Cap embedding calls
                text = f"{e.title}\n{e.content[:200]}"
                is_rel, scores = await pipeline.relevance.is_relevant(e.title, text)
                if is_rel:
                    max_score = max(scores.values()) if scores else 0
                    scored.append((e, max_score))
            scored.sort(key=lambda x: -x[1])
            entries = [e for e, _ in scored]

        if not entries:
            return f"No posts{f' for: {query}' if query else ''}."

        lines = [f"{len(entries)} posts{f' matching \"{query}\"' if query else ''}"]
        for i, e in enumerate(entries[:30], 1):
            pub = e.metadata.get("published", "")[:10]
            lines.append(f"{i}. [{e.source}] {truncate(e.title, 80)} {pub}")
            lines.append(f"   {e.url}")

        if len(entries) > 30:
            lines.append(f"... +{len(entries) - 30} more")

        return "\n".join(lines)

    @mcp.tool()
    def find_relevant(query: str, project: str = "", detail: str = "brief") -> str:
        """Search papers by topic. detail: compact|brief|full.

        When ``project`` is set, results are filtered to entries scored
        above the pipeline's relevance threshold for that project (using
        ``entry.metadata['relevance_scores']``).
        """
        results = pipeline.search(query, limit=10)
        if project and results:
            threshold = pipeline.relevance.threshold
            results = [
                e for e in results
                if (e.metadata or {}).get("relevance_scores", {}).get(project, 0) >= threshold
            ]
        if not results:
            return f"No papers found for: {query}"

        def compact():
            return "\n".join(
                f"id={_compact_field(e.id, 40)}|title={_compact_field(e.title, 60)}"
                for e in results
            )

        def brief():
            lines = [f"{len(results)} results for '{query}'"]
            for entry in results:
                tags = ", ".join(entry.tags or [])
                lines.append(f"[{entry.id}] {entry.title}")
                if tags:
                    lines.append(f"  {tags}")
            return "\n".join(lines)

        def full():
            lines = [f"Found {len(results)} results for '{query}':\n"]
            for entry in results:
                lines.append(f"[{entry.id}] {entry.title}")
                lines.append(f"  Tags: {', '.join(entry.tags or [])}")
                preview = entry.content[:150].replace("\n", " ")
                lines.append(f"  {preview}...")
                lines.append("")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    def reading_list(detail: str = "brief") -> str:
        """Show ingested papers. detail='brief' (counts only) or 'full' (all titles)."""
        rl = pipeline.get_reading_list()
        pending = rl.get("pending", [])
        distilled = rl.get("distilled", [])
        failed = rl.get("failed", [])
        skipped = rl.get("skipped", [])

        def brief():
            parts = [f"pending:{len(pending)} distilled:{len(distilled)} failed:{len(failed)} skipped:{len(skipped)}"]
            if pending:
                parts.append("PENDING:")
                for p in pending:
                    parts.append(f"  {p['entry_id']} | {truncate(p['title'], 70)}")
            return "\n".join(parts)

        def full():
            lines = []
            for label, items in [("PENDING", pending), ("DISTILLED", distilled), ("SKIPPED", skipped), ("FAILED", failed)]:
                if items:
                    lines.append(f"{label} ({len(items)})")
                    for p in items:
                        lines.append(f"  {p['entry_id']} | {truncate(p['title'], 70)}")
            return "\n".join(lines) if lines else "No papers ingested."

        return brief_or_full(brief, full, detail=detail)

    @mcp.tool()
    def paper_context(query: str, detail: str = "brief") -> str:
        """Paper context for prompt injection. detail: compact|brief|full."""
        raw = pipeline.get_paper_context(query)

        def compact():
            # Extract just paper titles from the raw context
            lines = []
            for line in raw.split("\n"):
                line = line.strip()
                if line.startswith("[") and "]" in line:
                    lines.append(truncate(line, 80))
                if len(lines) >= 3:
                    break
            return "\n".join(lines) if lines else truncate(raw, 100)

        def brief():
            return truncate(raw, 2000)

        def full():
            return raw

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    def paper_digest(hours: float = 24) -> str:
        """Show recent research activity digest."""
        entries = pipeline.digest.get_since(hours=hours)
        if not entries:
            return f"No research activity in the last {hours} hours."

        lines = [f"Research activity (last {hours}h): {len(entries)} events\n"]
        for entry in entries:
            lines.append(f"- {entry.summary}")
            if entry.tags:
                lines.append(f"  Tags: {', '.join(entry.tags)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Worker tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def start_distillation(batch_size: int = 0) -> str:
        """Start processing the distillation queue.

        Processes pending papers sequentially (summarize + extract + assess).
        batch_size=0 means process all pending. Otherwise process N papers.
        Returns progress when done.
        """
        pending = worker.count_pending()
        if pending == 0:
            return "No pending papers to distill."

        limit = batch_size if batch_size > 0 else None
        target = min(pending, batch_size) if batch_size > 0 else pending

        stats = await worker.run_batch(limit=limit)
        return (
            f"Distillation complete.\n"
            f"  Processed: {stats['processed']}\n"
            f"  Failed: {stats['failed']}\n"
            f"  Remaining: {worker.count_pending()}"
        )

    @mcp.tool()
    def worker_status() -> str:
        """Check distillation worker status and queue depth."""
        s = worker.stats
        return (
            f"Worker running: {s['running']}\n"
            f"Pending: {s['pending']}\n"
            f"Processed: {s['processed']}\n"
            f"Failed: {s['failed']}"
        )

    # ------------------------------------------------------------------
    # FR promotion
    # ------------------------------------------------------------------

    @mcp.tool()
    def promote_fr(
        target: str,
        title: str,
        description: str,
        priority: str = "medium",
        concept: str = "",
        classification: str = "app",
        backing_papers: str = "",
    ) -> str:
        """Promote a vetted feature request into the FR store.

        Use this after discussing and refining a concept — this stores it
        as a proper FR that target project Claudes can pick up via
        feature_requests(target=...).

        Args:
            target: Project name (e.g. khonliang, developer, genealogy — must be a key in config.yaml `projects`)
            title: Short FR title
            description: Full FR description with design, motivation, acceptance criteria
            priority: high, medium, or low
            concept: Research concept this FR derives from
            classification: library, library+app, or app
            backing_papers: Comma-separated paper titles
        """
        import hashlib
        from khonliang.knowledge.store import KnowledgeEntry, Tier, EntryStatus

        fr_id = f"fr_{target}_{hashlib.sha256(title.encode()).hexdigest()[:8]}"
        papers = [p.strip() for p in backing_papers.split(",") if p.strip()] if backing_papers else []

        fr_data = {
            "target": target,
            "title": title,
            "description": description,
            "priority": priority,
            "backing_papers": papers,
        }

        entry = KnowledgeEntry(
            id=fr_id,
            tier=Tier.DERIVED,
            title=title,
            content=json.dumps(fr_data, indent=2),
            source="promoted",
            scope="research",
            tags=["fr", f"target:{target}", classification],
            status=EntryStatus.DISTILLED,
            metadata={
                "concept": concept,
                "classification": classification,
                "target": target,
                "priority": priority,
                "backing_papers": papers,
                "promoted": True,
            },
        )
        pipeline.knowledge.add(entry)

        pipeline.digest.record(
            summary=f"Promoted FR: {title} → {target}",
            source="pipeline",
            audience="research",
            tags=["fr", "promoted"],
            metadata={"fr_id": fr_id, "target": target},
        )

        return f"FR promoted: {title}\nID: {fr_id}\nTarget: {target}\nPriority: {priority}\n\nProject Claudes can now pick this up via feature_requests(target='{target}')"

    @mcp.tool()
    def fr_workflow() -> str:
        """Get the FR workflow protocol for project Claudes.

        Returns the standard process for discovering, claiming, implementing,
        and completing feature requests from the researcher.
        """
        return """# Feature Request Workflow

## Discovery
1. `feature_requests(target="your_project")` — see all FRs for your project with status, deps, priority
2. `next_fr(target="your_project")` — get the highest priority FR with all dependencies met

## Claiming
3. `update_fr_status(fr_id, "planned", notes="planning implementation")` — claim the FR
4. Review the FR description, backing papers, and dependencies
5. If you need research context: `paper_context(query)` or `knowledge_search(query)`

## Implementation
6. `update_fr_status(fr_id, "in_progress", branch="fr/fr_id_short")` — start work
7. Create a worktree or branch for the FR
8. Implement, referencing only your project and its dependencies (e.g. khonliang)
9. Do NOT reference sibling projects (developer cannot mention genealogy)

## Completion
10. `update_fr_status(fr_id, "completed", notes="PR #N merged")` — mark done, unblocks dependents

## Managing FRs
- `set_fr_dependency(fr_id, depends_on)` — link FRs (e.g. app FR depends on library FR)
- `fr_overlaps(target)` — find duplicate/overlapping FRs
- `merge_frs(keep_id, merge_ids)` — combine overlapping FRs
- `promote_fr(...)` — create a new FR from research findings

## Capability Tracking
- `project_capabilities(target)` — see what exists vs what's planned per project
- Capabilities are tracked automatically:
  - FR → planned/in_progress: recorded as **planned** capability
  - FR → completed: recorded as **exists** capability
- Synergize uses this to avoid proposing features that already exist or are planned
- `evaluate_capability(description)` — assess if a new khonliang feature benefits the researcher

## Dependency Rules
- Library (khonliang) FRs should be completed before app FRs that depend on them
- `next_fr()` automatically respects this — it won't surface blocked FRs
- When a library FR is completed, dependent app FRs become unblocked

## Status Progression
open → planned → in_progress → completed

Each status update is recorded with timestamp and optional notes for audit trail.
Completing an FR automatically records the capability as "exists" for the target project.
"""

    @mcp.tool()
    def update_fr_status(fr_id: str, status: str, branch: str = "", notes: str = "") -> str:
        """Update a feature request's lifecycle status.

        Status progression: open → planned → in_progress → completed
        Project Claudes call this as they work through FRs.

        Args:
            fr_id: The FR ID
            status: open, planned, in_progress, or completed
            branch: Git branch name if in_progress (e.g. fr/khonliang_dac7335a)
            notes: Optional notes (e.g. "PR #42 opened", "blocked on X")
        """
        valid_statuses = ["open", "planned", "in_progress", "completed"]
        if status not in valid_statuses:
            return f"Invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}"

        entry = pipeline.knowledge.get(fr_id)
        if not entry:
            return f"FR {fr_id} not found."

        prev_status = entry.metadata.get("fr_status", "open")
        entry.metadata["fr_status"] = status
        if branch:
            entry.metadata["branch"] = branch
        if notes:
            history = entry.metadata.get("status_history", [])
            import time
            history.append({"status": status, "notes": notes, "at": time.strftime("%Y-%m-%d %H:%M")})
            entry.metadata["status_history"] = history

        if status == "completed":
            from khonliang.knowledge.store import EntryStatus
            entry.status = EntryStatus.ARCHIVED
            entry.tags = [t for t in (entry.tags or []) if t != "fr"] + ["fr:completed"]

            # Record capability as "exists" for the target project
            target = entry.metadata.get("target", "")
            concept = entry.metadata.get("concept", "")
            if target:
                _update_capability_status(target, entry.title, concept, "exists", fr_id)

            # Find unblocked FRs
            unblocked = []
            for e in pipeline.knowledge.get_by_tier(Tier.DERIVED):
                if "fr" not in (e.tags or []):
                    continue
                deps = e.metadata.get("depends_on", [])
                if fr_id in deps:
                    unblocked.append(e.title)

            pipeline.knowledge.add(entry)

            pipeline.digest.record(
                summary=f"FR completed: {entry.title}",
                source="pipeline",
                audience="research",
                tags=["fr", "completed"],
                metadata={"fr_id": fr_id},
            )

            result = f"Completed: {entry.title}"
            if unblocked:
                result += f"\n\nUnblocked {len(unblocked)} FR(s):"
                for title in unblocked:
                    result += f"\n  - {title}"
            return result

        # Track capability status transitions
        target = entry.metadata.get("target", "")
        concept = entry.metadata.get("concept", "")
        if target and status in ("planned", "in_progress"):
            _update_capability_status(target, entry.title, concept, "planned", fr_id)

        pipeline.knowledge.add(entry)

        pipeline.digest.record(
            summary=f"FR {prev_status} → {status}: {entry.title}",
            source="pipeline",
            audience="research",
            tags=["fr", status],
            metadata={"fr_id": fr_id, "branch": branch},
        )

        result = f"{entry.title}: {prev_status} → {status}"
        if branch:
            result += f"\nBranch: {branch}"
        if notes:
            result += f"\nNotes: {notes}"
        return result

    @mcp.tool()
    def set_fr_dependency(fr_id: str, depends_on: str) -> str:
        """Set dependencies between feature requests.

        Args:
            fr_id: The FR that has dependencies
            depends_on: Comma-separated FR IDs that must be completed first
        """
        entry = pipeline.knowledge.get(fr_id)
        if not entry:
            return f"FR {fr_id} not found."

        dep_ids = [d.strip() for d in depends_on.split(",") if d.strip()]

        # Validate deps exist
        for dep_id in dep_ids:
            dep = pipeline.knowledge.get(dep_id)
            if not dep:
                return f"Dependency {dep_id} not found."

        entry.metadata["depends_on"] = dep_ids
        pipeline.knowledge.add(entry)

        dep_titles = []
        for dep_id in dep_ids:
            dep = pipeline.knowledge.get(dep_id)
            dep_titles.append(f"  {dep_id}: {dep.title}" if dep else f"  {dep_id}: ?")

        return f"Dependencies set for {entry.title}:\n" + "\n".join(dep_titles)

    @mcp.tool()
    def next_fr(target: str = "", detail: str = "brief") -> str:
        """Next FR to work on (highest priority, deps met). detail: compact|brief|full."""
        from khonliang.knowledge.store import EntryStatus

        frs = pipeline.get_feature_requests(target=target or None)
        if not frs:
            return f"No FRs found{f' for {target}' if target else ''}."

        # Build a set of completed FR IDs (archived = done)
        all_entries = list(pipeline.knowledge.get_by_tier(Tier.DERIVED))
        completed_ids = set()
        for e in all_entries:
            if "fr:archived" in (e.tags or []) or "fr:completed" in (e.tags or []):
                completed_ids.add(e.id)
            if e.status == EntryStatus.ARCHIVED and "fr" in str(e.tags):
                completed_ids.add(e.id)

        # Score and filter FRs
        priority_score = {"high": 3, "medium": 2, "low": 1}
        class_score = {"library": 3, "library+app": 2, "app": 1}

        candidates = []
        blocked = []
        for fr in frs:
            # Skip FRs already being worked on or done
            entry = pipeline.knowledge.get(fr["id"])
            fr_status = entry.metadata.get("fr_status", "open") if entry else "open"
            if fr_status in ("planned", "in_progress", "completed"):
                continue
            deps = fr.get("depends_on", [])
            if not deps:
                # Also check entry metadata directly
                entry = pipeline.knowledge.get(fr["id"])
                if entry:
                    deps = entry.metadata.get("depends_on", [])

            unmet = [d for d in deps if d not in completed_ids]
            if unmet:
                blocked.append((fr, unmet))
                continue

            score = (
                priority_score.get(fr.get("priority", "medium"), 2) * 10
                + class_score.get(fr.get("classification", "app"), 1)
            )
            candidates.append((score, fr))

        candidates.sort(key=lambda x: -x[0])

        if not candidates:
            lines = ["All FRs are blocked on dependencies:\n"]
            for fr, unmet in blocked:
                unmet_titles = []
                for uid in unmet:
                    dep = pipeline.knowledge.get(uid)
                    unmet_titles.append(dep.title if dep else uid)
                lines.append(f"- {fr.get('title', '?')} → {fr.get('target', '?')}")
                lines.append(f"  Blocked by: {', '.join(unmet_titles)}")
            return "\n".join(lines)

        # Return top candidate with details
        _, top = candidates[0]
        entry = pipeline.knowledge.get(top["id"])
        fr_content = entry.content if entry else ""

        def compact():
            return (
                f"id={_compact_field(top['id'], 40)}|"
                f"title={_compact_field(top.get('title', '?'), 80)}|"
                f"target={_compact_field(top.get('target', '?'), 30)}|"
                f"priority={_compact_field(top.get('priority', '?'), 20)}"
            )

        def brief():
            lines = [
                f"{top.get('title', '?')}",
                f"id={top['id']} target={top.get('target','?')} priority={top.get('priority','?')} class={top.get('classification','?')}",
            ]
            desc = top.get("description", fr_content)
            if desc:
                lines.append(truncate(desc, 200))
            more = len(candidates) - 1
            if more:
                lines.append(f"+{more} more ready")
            return "\n".join(lines)

        def full():
            lines = [
                f"# Next FR: {top.get('title', '?')}",
                f"ID: {top['id']}",
                f"Target: {top.get('target', '?')}",
                f"Priority: {top.get('priority', '?')}",
                f"Classification: {top.get('classification', '?')}",
                f"Concept: {top.get('concept', '?')}",
            ]
            deps = top.get("depends_on", [])
            if not deps and entry:
                deps = entry.metadata.get("depends_on", [])
            if deps:
                lines.append(f"\nDependencies (all met):")
                for d in deps:
                    dep = pipeline.knowledge.get(d)
                    lines.append(f"  {d}: {dep.title if dep else '?'}")
            lines.append(f"\n## Description\n{top.get('description', fr_content)}")
            papers = top.get("backing_papers", [])
            if papers:
                lines.append(f"\n## Backing Papers")
                for p in papers:
                    lines.append(f"  - {p}")
            if len(candidates) > 1:
                lines.append(f"\n---\n{len(candidates) - 1} more FR(s) ready to work on.")
            if blocked:
                lines.append(f"{len(blocked)} FR(s) blocked on dependencies.")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    async def fr_overlaps(target: str = "", threshold: float = 0.75, detail: str = "brief") -> str:
        """Find overlapping FRs. detail: compact|brief|full."""
        frs = pipeline.get_feature_requests(target=target or None)
        if len(frs) < 2:
            return "Need at least 2 FRs to check for overlaps."

        # Embed each FR (title + description)
        embeddings = {}
        for fr in frs:
            text = f"{fr.get('title', '')}\n{fr.get('description', '')[:500]}"
            emb = await pipeline.relevance._embed(text)
            if emb:
                embeddings[fr["id"]] = (fr, emb)

        if len(embeddings) < 2:
            return "Could not embed enough FRs to compare."

        # Compare all pairs
        from khonliang_researcher import cosine_similarity
        pairs = []
        ids = list(embeddings.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                fr_a, emb_a = embeddings[ids[i]]
                fr_b, emb_b = embeddings[ids[j]]
                sim = cosine_similarity(emb_a, emb_b)
                if sim >= threshold:
                    pairs.append((sim, fr_a, fr_b))

        pairs.sort(key=lambda x: -x[0])

        if not pairs:
            return f"No overlapping FRs above {threshold:.0%}."

        BRIEF_LIMIT = 10

        def compact():
            return f"overlaps={len(pairs)}|threshold={threshold:.0%}"

        def brief():
            lines = [f"{len(pairs)} overlaps above {threshold:.0%}"]
            shown = pairs[:BRIEF_LIMIT]
            for sim, fr_a, fr_b in shown:
                lines.append(
                    f"  {sim:.0%}: {truncate(fr_a.get('title','?'),40)} <> {truncate(fr_b.get('title','?'),40)}"
                )
            remaining = len(pairs) - len(shown)
            if remaining > 0:
                lines.append(f"  +{remaining} more (use detail='full' to see all)")
            return "\n".join(lines)

        def full():
            lines = [f"# FR Overlaps ({len(pairs)} pairs above {threshold:.0%})\n"]
            for sim, fr_a, fr_b in pairs:
                lines.append(f"**{sim:.0%} similar:**")
                lines.append(f"  A: [{fr_a['id']}] {fr_a.get('title', '?')} → {fr_a.get('target', '?')}")
                lines.append(f"  B: [{fr_b['id']}] {fr_b.get('title', '?')} → {fr_b.get('target', '?')}")
                if fr_a.get('target') != fr_b.get('target'):
                    lines.append(f"  Different targets — may be intentional (library+app split)")
                if fr_a.get('concept') == fr_b.get('concept'):
                    lines.append(f"  Same concept: {fr_a.get('concept', '?')}")
                lines.append("")
            lines.append("Use merge_frs(keep_id, merge_ids) to combine overlapping FRs.")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    def merge_frs(keep_id: str, merge_ids: str, merged_title: str = "", merged_description: str = "") -> str:
        """Merge overlapping FRs into one. Keeps one, archives the rest.

        Args:
            keep_id: FR ID to keep as the primary
            merge_ids: Comma-separated FR IDs to merge into the primary
            merged_title: Optional new title for the merged FR
            merged_description: Optional new description incorporating both
        """
        from khonliang.knowledge.store import EntryStatus

        keep_entry = pipeline.knowledge.get(keep_id)
        if not keep_entry:
            return f"FR {keep_id} not found."

        ids_to_merge = [mid.strip() for mid in merge_ids.split(",") if mid.strip()]
        merged_entries = []
        for mid in ids_to_merge:
            entry = pipeline.knowledge.get(mid)
            if not entry:
                return f"FR {mid} not found."
            merged_entries.append(entry)

        # Update the kept FR
        try:
            keep_data = json.loads(keep_entry.content)
        except json.JSONDecodeError:
            keep_data = {}

        # Collect backing papers from all
        all_papers = set(keep_entry.metadata.get("backing_papers", []))
        merged_concepts = [keep_entry.metadata.get("concept", "")]
        for entry in merged_entries:
            all_papers.update(entry.metadata.get("backing_papers", []))
            concept = entry.metadata.get("concept", "")
            if concept and concept not in merged_concepts:
                merged_concepts.append(concept)

        if merged_title:
            keep_entry.title = merged_title
            keep_data["title"] = merged_title
        if merged_description:
            keep_data["description"] = merged_description
        keep_data["backing_papers"] = list(all_papers)
        keep_data["merged_from"] = ids_to_merge
        keep_entry.content = json.dumps(keep_data, indent=2)
        keep_entry.metadata["backing_papers"] = list(all_papers)
        keep_entry.metadata["merged_from"] = ids_to_merge
        keep_entry.metadata["merged_concepts"] = merged_concepts
        pipeline.knowledge.add(keep_entry)

        # Archive the merged FRs
        for entry in merged_entries:
            entry.status = EntryStatus.ARCHIVED
            entry.tags = [t for t in (entry.tags or []) if t != "fr"] + ["fr:archived", f"merged_into:{keep_id}"]
            pipeline.knowledge.add(entry)

        pipeline.digest.record(
            summary=f"Merged FRs: {', '.join(ids_to_merge)} into {keep_id}",
            source="pipeline",
            audience="research",
            tags=["fr", "merged"],
            metadata={"keep_id": keep_id, "merged_ids": ids_to_merge},
        )

        return (
            f"Merged {len(ids_to_merge)} FR(s) into {keep_id}\n"
            f"Title: {keep_entry.title}\n"
            f"Backing papers: {len(all_papers)}\n"
            f"Concepts: {', '.join(merged_concepts)}\n"
            f"Archived: {', '.join(ids_to_merge)}"
        )

    # ------------------------------------------------------------------
    # FR pipeline improvements
    # ------------------------------------------------------------------

    @mcp.tool()
    async def auto_deduplicate_frs(
        target: str = "",
        threshold: float = 0.85,
        dry_run: bool = True,
        detail: str = "brief",
    ) -> str:
        """Auto-deduplicate FRs by merging high-similarity pairs.

        Finds all FR pairs above the similarity threshold and merges them,
        keeping the FR with the higher priority (or the one with more
        backing papers as tiebreaker). Run with dry_run=true first to
        preview what would be merged.

        Args:
            target: Filter to FRs for this target (empty = all)
            threshold: Similarity threshold (default 0.85 — higher than
                       fr_overlaps default to avoid false merges)
            dry_run: If true, just report what would be merged
            detail: compact|brief|full
        """
        frs = pipeline.get_feature_requests(target=target or None)
        if len(frs) < 2:
            return "Need at least 2 FRs to deduplicate."

        # Embed all FRs
        embeddings = {}
        for fr in frs:
            text = f"{fr.get('title', '')}\n{fr.get('description', '')[:500]}"
            emb = await pipeline.relevance._embed(text)
            if emb:
                embeddings[fr["id"]] = (fr, emb)

        if len(embeddings) < 2:
            return "Could not embed enough FRs to compare."

        # Find overlapping pairs
        from khonliang_researcher import cosine_similarity
        pairs = []
        ids = list(embeddings.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                fr_a, emb_a = embeddings[ids[i]]
                fr_b, emb_b = embeddings[ids[j]]
                sim = cosine_similarity(emb_a, emb_b)
                if sim >= threshold:
                    pairs.append((sim, fr_a, fr_b))

        pairs.sort(key=lambda x: -x[0])

        if not pairs:
            return f"No duplicates above {threshold:.0%}."

        # Decide which to keep in each pair (higher priority wins,
        # then more backing papers, then alphabetical ID)
        PRIORITY_ORDER = {"high": 3, "medium": 2, "low": 1}
        merges = []
        already_merged = set()

        for sim, fr_a, fr_b in pairs:
            if fr_a["id"] in already_merged or fr_b["id"] in already_merged:
                continue

            score_a = PRIORITY_ORDER.get(fr_a.get("priority", "medium"), 2)
            score_b = PRIORITY_ORDER.get(fr_b.get("priority", "medium"), 2)
            papers_a = len(fr_a.get("backing_papers", []))
            papers_b = len(fr_b.get("backing_papers", []))

            if score_a > score_b or (score_a == score_b and papers_a >= papers_b):
                keep, merge = fr_a, fr_b
            else:
                keep, merge = fr_b, fr_a

            merges.append({"keep": keep["id"], "merge": merge["id"], "similarity": sim,
                           "keep_title": keep.get("title", "?"), "merge_title": merge.get("title", "?")})
            already_merged.add(merge["id"])

        if not dry_run:
            from khonliang.knowledge.store import EntryStatus
            merged_count = 0
            for m in merges:
                keep_entry = pipeline.knowledge.get(m["keep"])
                merge_entry = pipeline.knowledge.get(m["merge"])
                if not keep_entry or not merge_entry:
                    continue

                # Collect backing papers
                keep_papers = set(keep_entry.metadata.get("backing_papers", []))
                keep_papers.update(merge_entry.metadata.get("backing_papers", []))
                keep_entry.metadata["backing_papers"] = list(keep_papers)
                keep_entry.metadata.setdefault("merged_from", []).append(m["merge"])
                pipeline.knowledge.add(keep_entry)

                # Archive the merged FR
                merge_entry.status = EntryStatus.ARCHIVED
                merge_entry.tags = [t for t in (merge_entry.tags or []) if t != "fr"] + ["fr:archived", f"merged_into:{m['keep']}"]
                pipeline.knowledge.add(merge_entry)
                merged_count += 1

            pipeline.digest.record(
                summary=f"Auto-deduplicated: {merged_count} FRs merged (threshold={threshold:.0%})",
                source="pipeline", audience="research",
                tags=["fr", "dedup"],
            )

        def compact():
            return f"duplicates={len(merges)}|threshold={threshold:.0%}|dry_run={dry_run}"

        def brief():
            mode = "DRY RUN — " if dry_run else ""
            lines = [f"{mode}{len(merges)} merges at {threshold:.0%} threshold"]
            for m in merges[:15]:
                lines.append(f"  {m['similarity']:.0%}: keep [{m['keep'][:20]}] merge [{m['merge'][:20]}]")
            if len(merges) > 15:
                lines.append(f"  +{len(merges)-15} more")
            if dry_run:
                lines.append("\nRun with dry_run=false to execute.")
            return "\n".join(lines)

        def full():
            mode = "DRY RUN\n\n" if dry_run else "EXECUTED\n\n"
            lines = [f"{mode}{len(merges)} merges at {threshold:.0%} threshold\n"]
            for m in merges:
                lines.append(f"**{m['similarity']:.0%}:** keep {m['keep']}")
                lines.append(f"  → {m['keep_title']}")
                lines.append(f"  merge {m['merge']}")
                lines.append(f"  → {m['merge_title']}")
                lines.append("")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    async def cluster_frs(
        target: str = "",
        min_cluster_size: int = 2,
        threshold: float = 0.7,
        detail: str = "brief",
    ) -> str:
        """Group related FRs into clusters via embedding similarity.

        Unlike fr_overlaps (pairwise), this groups FRs into neighborhoods
        where all members are above the threshold with at least one other
        member. Useful for seeing which FRs should be bundled into a single
        work unit or milestone.

        Args:
            target: Filter to FRs for this target (empty = all)
            min_cluster_size: Minimum FRs in a cluster to report
            threshold: Similarity threshold for cluster membership
            detail: compact|brief|full
        """
        frs = pipeline.get_feature_requests(target=target or None)
        if len(frs) < 2:
            return "Need at least 2 FRs to cluster."

        # Embed all FRs
        embeddings = {}
        for fr in frs:
            text = f"{fr.get('title', '')}\n{fr.get('description', '')[:500]}"
            emb = await pipeline.relevance._embed(text)
            if emb:
                embeddings[fr["id"]] = (fr, emb)

        if len(embeddings) < 2:
            return "Could not embed enough FRs."

        # Build adjacency: which FRs are similar to which
        from khonliang_researcher import cosine_similarity
        ids = list(embeddings.keys())
        neighbors: dict[str, set[str]] = {fid: set() for fid in ids}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                _, emb_a = embeddings[ids[i]]
                _, emb_b = embeddings[ids[j]]
                sim = cosine_similarity(emb_a, emb_b)
                if sim >= threshold:
                    neighbors[ids[i]].add(ids[j])
                    neighbors[ids[j]].add(ids[i])

        # Connected components (simple BFS)
        from collections import deque
        visited = set()
        clusters = []
        for fid in ids:
            if fid in visited or not neighbors[fid]:
                continue
            cluster = []
            queue = deque([fid])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for n in neighbors[current]:
                    if n not in visited:
                        queue.append(n)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        if not clusters:
            return f"No clusters of {min_cluster_size}+ FRs at {threshold:.0%} threshold."

        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)

        def compact():
            return f"clusters={len(clusters)}|total_frs={sum(len(c) for c in clusters)}|threshold={threshold:.0%}"

        def brief():
            lines = [f"{len(clusters)} clusters ({sum(len(c) for c in clusters)} FRs) at {threshold:.0%}"]
            for i, cluster in enumerate(clusters, 1):
                fr_titles = [truncate(embeddings[fid][0].get("title", "?"), 50) for fid in cluster]
                targets = set(embeddings[fid][0].get("target", "?") for fid in cluster)
                lines.append(f"\n  Cluster {i} ({len(cluster)} FRs, targets: {','.join(targets)}):")
                for title in fr_titles[:5]:
                    lines.append(f"    - {title}")
                if len(fr_titles) > 5:
                    lines.append(f"    +{len(fr_titles)-5} more")
            return "\n".join(lines)

        def full():
            lines = [f"# FR Clusters ({len(clusters)} clusters, {sum(len(c) for c in clusters)} FRs)\n"]
            for i, cluster in enumerate(clusters, 1):
                targets = set(embeddings[fid][0].get("target", "?") for fid in cluster)
                lines.append(f"## Cluster {i} ({len(cluster)} FRs, targets: {','.join(targets)})")
                for fid in cluster:
                    fr = embeddings[fid][0]
                    lines.append(f"  [{fid}] {fr.get('title', '?')} → {fr.get('target', '?')} [{fr.get('priority', '?')}]")
                lines.append("")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    def update_fr(
        fr_id: str,
        title: str = "",
        description: str = "",
        priority: str = "",
        target: str = "",
        notes: str = "",
    ) -> str:
        """Modify an existing FR in-place.

        Updates the specified fields without creating a new FR. Unchanged
        fields are left as-is. Records the modification in the digest.

        Args:
            fr_id: The FR to modify
            title: New title (empty = keep current)
            description: New description (empty = keep current)
            priority: New priority: high, medium, low (empty = keep current)
            target: New target project (empty = keep current)
            notes: Reason for the change (recorded in digest)
        """
        entry = pipeline.knowledge.get(fr_id)
        if not entry:
            return f"FR {fr_id} not found."

        try:
            fr_data = json.loads(entry.content)
        except (json.JSONDecodeError, TypeError):
            return f"FR {fr_id} has malformed content — cannot modify safely."

        if priority and priority not in ("high", "medium", "low"):
            return f"Invalid priority '{priority}'. Must be high, medium, or low."

        changes = []
        if title:
            entry.title = title
            fr_data["title"] = title
            changes.append(f"title → {title}")
        if description:
            fr_data["description"] = description
            changes.append("description updated")
        if priority:
            fr_data["priority"] = priority
            entry.metadata["priority"] = priority
            changes.append(f"priority → {priority}")
        if target:
            fr_data["target"] = target
            entry.metadata["target"] = target
            # Update tags
            entry.tags = [t for t in (entry.tags or []) if not t.startswith("target:")] + [f"target:{target}"]
            changes.append(f"target → {target}")

        if not changes:
            return f"No changes specified for {fr_id}."

        entry.content = json.dumps(fr_data, indent=2)
        pipeline.knowledge.add(entry)

        pipeline.digest.record(
            summary=f"FR modified: {fr_id} — {', '.join(changes)}",
            source="pipeline",
            audience="research",
            tags=["fr", "modified"],
            metadata={"fr_id": fr_id, "changes": changes, "notes": notes},
        )

        return (
            f"Updated {fr_id}:\n"
            f"  Changes: {', '.join(changes)}\n"
            f"  Title: {entry.title}\n"
            + (f"  Notes: {notes}\n" if notes else "")
        )

    # ------------------------------------------------------------------
    # Relevance scoring tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def score_relevance(entry_id: str, detail: str = "brief") -> str:
        """Score paper relevance to projects. detail: compact|brief|full."""
        scores = await pipeline.score_relevance(entry_id)
        if not scores:
            return f"Could not score {entry_id}. Check embedding model is available."

        entry = pipeline.knowledge.get(entry_id)
        title = entry.title if entry else entry_id
        threshold = pipeline.relevance.threshold
        max_score = max(scores.values())
        above = max_score >= threshold

        def compact():
            score_pairs = "|".join(
                f"{_compact_field(p, 30)}={s:.2f}"
                for p, s in sorted(scores.items(), key=lambda x: -x[1])
            )
            return f"{score_pairs}|status={'above' if above else 'below'}|threshold={threshold:.2f}"

        def brief():
            lines = [f"{title}"]
            for project, score in sorted(scores.items(), key=lambda x: -x[1]):
                lines.append(f"  {project}: {score:.2f}")
            lines.append(f"threshold={threshold:.2f} {'above' if above else 'below'}")
            return "\n".join(lines)

        def full():
            lines = [f"# Relevance: {title}\n"]
            for project, score in sorted(scores.items(), key=lambda x: -x[1]):
                bar = "=" * int(score * 20)
                lines.append(f"  {project:15s} {score:.2f} |{bar}")
            if above:
                lines.append(f"\nAbove threshold ({threshold:.2f}) — will be distilled.")
            else:
                lines.append(f"\nBelow threshold ({threshold:.2f}) — would be skipped by worker.")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    # ------------------------------------------------------------------
    # Idea tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def ingest_idea(text: str, source_label: str = "") -> str:
        """Ingest informal text (LinkedIn post, tweet, blog snippet, thought).

        Decomposes the text into researchable claims and search queries.
        Returns the idea ID for use with research_idea and brief_idea.
        """
        try:
            idea_id = await pipeline.ingest_idea(text, source_label)
        except RuntimeError as e:
            return f"Failed: {e}"

        entry = pipeline.knowledge.get(idea_id)
        if not entry:
            return f"Idea stored as {idea_id} but could not reload."

        claims = entry.metadata.get("claims", [])
        queries = entry.metadata.get("search_queries", [])

        lines = [
            f"# Idea: {entry.title}",
            f"ID: {idea_id}",
            f"\n## Claims ({len(claims)})",
        ]
        for c in claims:
            lines.append(f"- {c}")
        lines.append(f"\n## Search Queries ({len(queries)})")
        for q in queries:
            lines.append(f"- {q}")
        lines.append(f"\nNext: call research_idea('{idea_id}') to find papers.")
        return "\n".join(lines)

    @mcp.tool()
    async def research_idea(idea_id: str, max_papers: int = 10, auto_distill: bool = True) -> str:
        """Search for papers backing an idea's claims, fetch and distill them.

        Runs all search queries from the idea in parallel, deduplicates,
        fetches new papers, and optionally distills them.
        """
        stats = await pipeline.research_idea(idea_id, max_papers, auto_distill)
        if "error" in stats:
            return f"Error: {stats['error']}"

        return (
            f"Research complete for idea {idea_id}:\n"
            f"  Queries run: {stats['queries_run']}\n"
            f"  Papers found: {stats['papers_found']}\n"
            f"  New papers ingested: {stats['papers_new']}\n"
            f"  Papers distilled: {stats['papers_distilled']}\n"
            f"\nNext: call brief_idea('{idea_id}') to synthesize findings."
        )

    @mcp.tool()
    async def brief_idea(idea_id: str) -> str:
        """Synthesize a brief evaluating an idea's claims against found literature.

        Analyzes linked papers and produces a claim-by-claim assessment:
        which claims are supported, contradicted, or unaddressed.
        """
        return await pipeline.brief_idea(idea_id)

    # ------------------------------------------------------------------
    # Graph + Matrix tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def concept_matrix(min_connections: int = 2, max_concepts: int = 30) -> str:
        """Show a concept × paper matrix.

        Rows are concepts/methods, columns are papers. Shows which papers
        cover which concepts, with relationship types and confidence scores.
        Good for finding coverage gaps and seeing concept density.
        """
        from khonliang_researcher import build_concept_matrix, format_matrix

        matrix_data = build_concept_matrix(
            pipeline.triples,
            min_connections=min_connections,
            max_entities=max_concepts,
        )
        if not matrix_data["entities"]:
            return "No concept matrix data. Distill some papers first."
        return format_matrix(matrix_data, pipeline.knowledge, pipeline.triples)

    @mcp.tool()
    def concept_tree(concept: str, depth: int = 4, branches: int = 3) -> str:
        """Trace a concept's connections through the knowledge graph.

        Shows a tree of related concepts connected through papers:
            GRPO
            ├── improved_by → MAGRPO
            │   ├── applied_to → LLM Collaboration
            │   └── extends → C3
            └── used_by → ConsensusEngine

        Like a LinkedIn connection graph for research concepts.
        """
        from khonliang_researcher import build_concept_graph, trace_chain

        graph = build_concept_graph(pipeline.triples, knowledge=pipeline.knowledge)
        return trace_chain(graph, concept, max_depth=depth, max_branches=branches)

    @mcp.tool()
    def concept_path(start: str, end: str) -> str:
        """Find how two concepts connect through the knowledge graph.

        Traces paths from start → end through intermediate concepts.
        Shows the chain of papers and relationships connecting them.
        """
        from khonliang_researcher import build_concept_graph, find_paths

        graph = build_concept_graph(pipeline.triples, knowledge=pipeline.knowledge)
        paths = find_paths(graph, start, end)
        if not paths:
            return f"No path found from '{start}' to '{end}'."

        lines = [f"Found {len(paths)} path(s) from '{start}' to '{end}':\n"]
        for i, path in enumerate(paths[:5], 1):
            chain = " → ".join(
                f"{node} —[{pred}]→ {target}"
                for node, pred, target in path
            )
            lines.append(f"  {i}. {chain}")
        return "\n".join(lines)

    @mcp.tool()
    def concepts_for_project(project: str, min_score: float = 0.4, limit: int = 30, detail: str = "brief") -> str:
        """Concepts relevant to a project. detail: compact|brief|full."""
        from khonliang_researcher import build_project_scores

        scores = build_project_scores(pipeline.knowledge, pipeline.triples)
        ranked = []
        for concept, proj_scores in scores.items():
            score = proj_scores.get(project, 0)
            if score >= min_score:
                ranked.append((concept, score, proj_scores))

        if not ranked:
            return f"No concepts for '{project}' above {min_score:.0%}."

        ranked.sort(key=lambda x: -x[1])
        ranked = ranked[:limit]

        def compact():
            top = ranked[:5]
            return "\n".join(f"{c}:{s:.0%}" for c, s, _ in top)

        def brief():
            top = ranked[:15]
            lines = [f"{project} concepts ({len(ranked)})"]
            for concept, score, _ in top:
                lines.append(f"  {concept}: {score:.0%}")
            return "\n".join(lines)

        def full():
            lines = [f"## Concepts for {project} ({len(ranked)} above {min_score:.0%})\n"]
            for concept, score, all_scores in ranked:
                other = ", ".join(
                    f"{p}:{s:.0%}" for p, s in sorted(all_scores.items(), key=lambda x: -x[1])
                    if p != project
                )
                lines.append(f"- **{concept}** ({score:.0%})")
                if other:
                    lines.append(f"  Also: {other}")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    # ------------------------------------------------------------------
    # Synergize tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def synergize(min_score: float = 0.5, max_concepts: int = 10, detail: str = "brief") -> str:
        """Classify concepts and generate FRs. detail: compact|brief|full."""
        result = await pipeline.synergize(min_score=min_score, max_concepts=max_concepts)
        if "error" in result:
            raw = result.get("raw", "")
            return f"Error: {result['error']}\n\n{raw}" if raw else f"Error: {result['error']}"

        concepts = result["concept_count"]
        fr_count = result["fr_count"]
        classifications = result["classifications"]

        def compact():
            fr_titles = []
            for item in classifications:
                for fr in item.get("feature_requests", []):
                    fr_titles.append(truncate(fr.get("title", "?"), 50))
            return f"concepts={concepts}|frs={fr_count}|" + "; ".join(fr_titles[:5])

        def brief():
            lines = [f"{concepts} concepts, {fr_count} FRs"]
            for item in classifications:
                concept = item.get("concept", "?")
                cls = item.get("classification", "?")
                targets = ",".join(item.get("targets", []))
                lines.append(f"\n{concept} -> {cls} [{targets}]")
                for fr in item.get("feature_requests", []):
                    lines.append(f"  [{fr.get('priority','med')}] {fr.get('title','?')} -> {fr.get('target','?')}")
            return "\n".join(lines)

        def full():
            lines = [f"{concepts} concepts, {fr_count} FRs"]
            for item in classifications:
                concept = item.get("concept", "?")
                cls = item.get("classification", "?")
                targets = ",".join(item.get("targets", []))
                lines.append(f"\n{concept} -> {cls} [{targets}]")
                for fr in item.get("feature_requests", []):
                    lines.append(f"  [{fr.get('priority','med')}] {fr.get('title','?')} -> {fr.get('target','?')}")
                    if fr.get("description"):
                        lines.append(f"    {truncate(fr['description'], 200)}")
                papers = item.get("backing_papers", [])
                if papers:
                    lines.append(f"  papers: {', '.join(str(p) for p in papers)}")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    async def synergize_compare(min_score: float = 0.5, max_concepts: int = 10) -> str:
        """Compare self-distillation candidates. Shows per-candidate concept/FR counts and
        diversity metrics (overlap ratios) across N synergize outputs.

        Generates N candidates in parallel, selects the best, and reports
        summary statistics. Does not return full candidate texts (use synergize
        directly for the selected output).
        """
        result = await pipeline.synergize_compare(min_score=min_score, max_concepts=max_concepts)
        if "error" in result:
            raw = result.get("raw", "")
            return f"Error: {result['error']}\n{raw[:500]}" if raw else f"Error: {result['error']}"

        lines = [f"Selected candidate {result['selected']} of {result['n_candidates']}"]
        for c in result["candidates"]:
            status = f"{c['concepts']} concepts, {c['frs']} FRs" if c["valid"] else "parse error"
            marker = " <-" if c["candidate"] == result["selected"] else ""
            lines.append(f"  #{c['candidate']}: {status}{marker}")

        d = result["diversity"]
        lines.append(f"\nDiversity:")
        lines.append(f"  concepts: {d['unique_concepts']} unique, {d['shared_concepts']} shared ({d['concept_overlap']:.0%} overlap)")
        lines.append(f"  FRs: {d['unique_frs']} unique, {d['shared_frs']} shared ({d['fr_overlap']:.0%} overlap)")

        return "\n".join(lines)

    @mcp.tool()
    async def review_feature_requests(target: str = "", detail: str = "brief") -> str:
        """Review FRs with 32B model. detail='brief' or 'full'."""
        results = await pipeline.review_frs(target=target or None)
        if not results:
            return "No FRs to review."

        accepted = sum(1 for r in results if r.get("verdict") == "accept")
        revised = sum(1 for r in results if r.get("verdict") == "revise")
        rejected = sum(1 for r in results if r.get("verdict") == "reject")

        def fmt_brief(r):
            v = r.get("verdict", "?").upper()
            c = r.get("confidence", 0)
            revised_note = f" => {r['revised_title']}" if r.get("revised_title") else ""
            return f"[{v} {c:.0%}] {r.get('title','?')} -> {r.get('target','?')}{revised_note}"

        def fmt_full(r):
            lines = [fmt_brief(r)]
            lines.append(f"  {truncate(r.get('reasoning', ''), 200)}")
            for c in r.get("concerns", []):
                lines.append(f"  - {truncate(c, 100)}")
            return "\n".join(lines)

        summary = f"reviewed:{len(results)} accept:{accepted} revise:{revised} reject:{rejected}"

        def brief():
            return compact_list(results, fmt_brief, header=summary, limit=50)

        def full():
            return compact_list(results, fmt_full, header=summary, limit=50)

        return brief_or_full(brief, full, detail=detail)

    @mcp.tool()
    async def evaluate_capability(capability: str, detail: str = "brief") -> str:
        """Evaluate if a khonliang feature benefits researcher. detail: compact|brief|full."""
        result = await pipeline.evaluate_capability(capability)

        if "error" in result:
            return f"Error: {result['error']}\n{result.get('raw', '')}"

        applicable = result.get("applicable", "?")
        score = result.get("score", 0)

        def compact():
            return f"applicable={applicable}|score={score:.0%}|frs={len(result.get('suggested_frs', []))}"

        def brief():
            lines = [f"applicable:{applicable} score:{score:.0%} | {result.get('summary','')}"]
            for fr in result.get("suggested_frs", []):
                dep = f" (needs {fr['depends_on']})" if fr.get("depends_on") else ""
                lines.append(f"  FR [{fr.get('priority','?')}] {fr.get('title','?')}{dep}")
            return "\n".join(lines)

        def full():
            lines = [f"applicable:{applicable} score:{score:.0%} | {result.get('summary','')}"]
            for label, key in [("uses", "direct_uses"), ("improves", "improvements"), ("unlocks", "new_features")]:
                items = result.get(key, [])
                if items:
                    lines.append(f"{label}: {'; '.join(truncate(i, 80) for i in items)}")
            if result.get("integration_notes"):
                lines.append(f"integration: {truncate(result['integration_notes'], 150)}")
            for fr in result.get("suggested_frs", []):
                dep = f" (needs {fr['depends_on']})" if fr.get("depends_on") else ""
                lines.append(f"  FR [{fr.get('priority','?')}] {fr.get('title','?')}{dep}")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    def feature_requests(target: str = "", detail: str = "brief") -> str:
        """List feature requests. detail='brief' (default) or 'full'."""
        frs = pipeline.get_feature_requests(target=target or None)
        if not frs:
            return f"No FRs{f' for {target}' if target else ''}."

        priority_order = {"high": 0, "medium": 1, "low": 2}
        frs.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))

        def fmt_brief(fr):
            entry = pipeline.knowledge.get(fr["id"])
            status = (entry.metadata.get("fr_status", "open") if entry else "open")
            status_str = f" ({status})" if status != "open" else ""
            return f"{fr['id']} | [{fr.get('priority','med')}] {fr.get('title','?')} -> {fr.get('target','?')}{status_str}"

        def fmt_full(fr):
            entry = pipeline.knowledge.get(fr["id"])
            status = (entry.metadata.get("fr_status", "open") if entry else "open")
            branch = (entry.metadata.get("branch", "") if entry else "")
            deps = (entry.metadata.get("depends_on", []) if entry else [])
            lines = [fmt_brief(fr)]
            if branch:
                lines.append(f"  branch: {branch}")
            lines.append(f"  concept: {fr.get('concept', '?')} | class: {fr.get('classification', '?')}")
            if deps:
                lines.append(f"  depends: {', '.join(deps)}")
            desc = fr.get("description", "")
            if desc:
                lines.append(f"  {truncate(desc, 150)}")
            return "\n".join(lines)

        def brief():
            return compact_list(frs, fmt_brief,
                header=f"FRs ({len(frs)}){f' for {target}' if target else ''}",
                limit=50, empty_msg="None.")

        def full():
            return compact_list(frs, fmt_full,
                header=f"FRs ({len(frs)}){f' for {target}' if target else ''}",
                limit=50, empty_msg="None.")

        return brief_or_full(brief, full, detail=detail)

    # ------------------------------------------------------------------
    # Synthesis tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def synthesize_topic(topic: str, detail: str = "brief") -> str:
        """Synthesize papers on a topic. detail: compact|brief|full."""
        result = await synthesizer.topic_summary(topic)
        if not result.success:
            return result.content

        content = result.content
        count = result.paper_count

        def compact():
            # Extract key lines: count + first substantive bullet points
            lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
            top = [_compact_field(l.lstrip("- "), 80) for l in lines[:5]]
            return f"topic={_compact_field(topic, 60)}|papers={count}|highlights={'; '.join(top)}"

        def brief():
            # Structured bullets, no narrative
            lines = [f"{topic} ({count} papers)"]
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(truncate(line, 120))
                if len(lines) > 15:
                    break
            return "\n".join(lines)

        def full():
            return f"# Topic: {topic} ({count} papers)\n\n{content}"

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    async def synthesize_project(project: str = "khonliang", detail: str = "brief") -> str:
        """Project applicability brief. detail: compact|brief|full."""
        projects = pipeline.config.get("projects", {})
        if project not in projects:
            available = ", ".join(projects.keys()) or "none configured"
            return f"Project '{project}' not found. Available: {available}"

        cfg = projects[project]
        result = await synthesizer.project_brief(
            project, cfg.get("description", "")
        )
        if not result.success:
            return result.content

        content = result.content
        count = result.paper_count

        def compact():
            lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
            top = [_compact_field(l.lstrip("- "), 80) for l in lines[:5]]
            return f"project={_compact_field(project, 30)}|papers={count}|highlights={'; '.join(top)}"

        def brief():
            lines = [f"{project} ({count} papers)"]
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(truncate(line, 120))
                if len(lines) > 15:
                    break
            return "\n".join(lines)

        def full():
            return f"# {project} Brief ({count} papers)\n\n{content}"

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    async def synthesize_landscape(detail: str = "brief") -> str:
        """Research landscape overview. detail: compact|brief|full."""
        result = await synthesizer.landscape()
        if not result.success:
            return result.content

        content = result.content
        count = result.paper_count

        def compact():
            lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
            top = [_compact_field(l.lstrip("- "), 80) for l in lines[:5]]
            return f"scope=landscape|papers={count}|highlights={'; '.join(top)}"

        def brief():
            lines = [f"Landscape ({count} papers)"]
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(truncate(line, 120))
                if len(lines) > 15:
                    break
            return "\n".join(lines)

        def full():
            return f"# Research Landscape ({count} papers)\n\n{content}"

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    def project_capabilities(target: str = "") -> str:
        """Show what exists vs what's planned for each project.

        Automatically maintained as FRs move through their lifecycle.
        Synergize uses this to avoid proposing features that already exist.
        """
        caps = []
        for entry in pipeline.knowledge.get_by_tier(Tier.DERIVED):
            tags = entry.tags or []
            if "capability" not in tags:
                continue
            if target and f"cap:{target}" not in tags:
                continue
            caps.append(entry)

        if not caps:
            return "No capabilities tracked yet. Capabilities are recorded automatically when FRs are completed."

        # Group by target then status
        from collections import defaultdict
        grouped = defaultdict(lambda: defaultdict(list))
        for c in caps:
            t = (c.metadata or {}).get("target", "unknown")
            s = (c.metadata or {}).get("capability_status", "unknown")
            grouped[t][s].append(c.title)

        lines = ["# Project Capabilities\n"]
        for proj in sorted(grouped.keys()):
            lines.append(f"## {proj}")
            for status in ["exists", "planned", "exploring"]:
                items = grouped[proj].get(status, [])
                if items:
                    lines.append(f"\n### {status.upper()} ({len(items)})")
                    for item in items:
                        lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Codebase scanning and repo registry
    # ------------------------------------------------------------------

    @mcp.tool()
    async def ingest_github(repo_url: str, label: str = "", depth: str = "readme+code") -> str:
        """Cleanroom ingest a GitHub repo: clone, AST-scan concepts, delete clone.

        No code is retained — only distilled capabilities and architecture patterns.
        depth: "readme" (README only), "readme+code" (AST scan), "full" (all docs)
        """
        result = await pipeline.ingest_github_repo(repo_url, label=label, depth=depth)
        if "error" in result:
            return result["error"]

        arch = result.get("architecture", "")
        header = f"{result['repo']} | {len(result['capabilities'])} capabilities | depth={result['depth']}"
        if arch:
            header += f" | {arch}"
        lines = [header]
        for cap in result.get("code_capabilities", []):
            lines.append(f"  [code] {cap}")
        for claim in result.get("readme_only_claims", []):
            lines.append(f"  [readme] {claim}")
        if result.get("imports_from"):
            for dep, items in result["imports_from"].items():
                lines.append(f"  imports {dep}: {', '.join(items[:3])}")
        if result.get("relevance_scores"):
            for proj, score in sorted(result["relevance_scores"].items(), key=lambda x: -x[1]):
                lines.append(f"  relevance({proj}): {score:.2f}")
        return "\n".join(lines)

    @mcp.tool()
    async def scan_codebase(project: str) -> str:
        """Scan a project's codebase to discover and store capabilities.

        Reads the repo, extracts code signatures, uses LLM to identify
        implemented features. Stores results as capability entries.
        """
        result = await pipeline.scan_codebase(project)
        if "error" in result:
            return f"Error: {result['error']}\n{result.get('raw', '')}"

        caps = result.get("capabilities", [])
        imports = result.get("imports_from", {})
        stored = result.get("stored", 0)

        lines = [f"{project}: {len(caps)} capabilities found, {stored} new stored"]
        for c in caps:
            lines.append(f"  {c}")
        if imports:
            for dep, usages in imports.items():
                lines.append(f"  imports from {dep}: {', '.join(usages)}")

        return "\n".join(lines)

    @mcp.tool()
    def register_repo(project: str, repo_path: str, description: str = "", depends_on: str = "", scope: str = "") -> str:
        """Register or update a project repo in the DB.

        Stores repo location, description, scope, and dependencies
        so scan_codebase and synergize can find it.
        """
        from khonliang.knowledge.store import KnowledgeEntry

        entry_id = f"repo_{project}"
        deps = [d.strip() for d in depends_on.split(",") if d.strip()]

        entry = KnowledgeEntry(
            id=entry_id,
            tier=Tier.DERIVED,
            title=f"Repo: {project}",
            content=description or pipeline.config.get("projects", {}).get(project, {}).get("description", ""),
            source="registry",
            scope="registry",
            tags=["repo", f"repo:{project}"],
            status=EntryStatus.DISTILLED,
            metadata={
                "project": project,
                "repo_path": repo_path,
                "scope": scope or pipeline.config.get("projects", {}).get(project, {}).get("scope", ""),
                "depends_on": deps or pipeline.config.get("projects", {}).get(project, {}).get("depends_on", []),
            },
        )
        pipeline.knowledge.add(entry)
        return f"Registered {project} at {repo_path}"

    @mcp.tool()
    def list_repos() -> str:
        """List all registered project repos."""
        repos = []
        for entry in pipeline.knowledge.get_by_tier(Tier.DERIVED):
            if "repo" not in (entry.tags or []):
                continue
            meta = entry.metadata or {}
            deps = ", ".join(meta.get("depends_on", [])) or "none"
            repos.append(f"{meta.get('project','?')} | {meta.get('repo_path','?')} | scope:{meta.get('scope','?')} | deps:{deps}")

        if not repos:
            # Fall back to config
            for name, cfg in pipeline.config.get("projects", {}).items():
                repo = cfg.get("repo", "not set")
                deps = ", ".join(cfg.get("depends_on", [])) or "none"
                repos.append(f"{name} | {repo} | scope:{cfg.get('scope','?')} | deps:{deps}")

        return "\n".join(repos) if repos else "No repos registered."

    # ------------------------------------------------------------------
    # RR-8: Per-Project Landscape Reports
    # ------------------------------------------------------------------

    @mcp.tool()
    async def project_landscape(project: str, detail: str = "brief") -> str:
        """Research landscape for a specific project. Shows relevant concepts,
        paper counts, capability gaps, and open FRs. detail='brief' or 'full'."""
        projects = pipeline.config.get("projects", {})
        if project not in projects:
            return f"Unknown project. Available: {', '.join(projects.keys())}"

        from khonliang_researcher import build_project_scores

        # Concepts scored for this project
        scores = build_project_scores(pipeline.knowledge, pipeline.triples)
        proj_concepts = {
            c: s[project] for c, s in scores.items()
            if project in s and s[project] >= 0.4
        }
        top_concepts = sorted(proj_concepts.items(), key=lambda x: -x[1])

        # Paper counts
        distilled = len(pipeline.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED))
        total = len(list(pipeline.knowledge.get_by_tier(Tier.IMPORTED)))

        # Open FRs for this project
        frs = pipeline.get_feature_requests(target=project)

        # Capabilities
        caps_exist = []
        caps_planned = []
        for entry in pipeline.knowledge.get_by_tier(Tier.DERIVED):
            tags = entry.tags or []
            if "capability" not in tags or f"cap:{project}" not in tags:
                continue
            status = (entry.metadata or {}).get("capability_status", "")
            if status == "exists":
                caps_exist.append(entry.title)
            elif status == "planned":
                caps_planned.append(entry.title)

        def brief():
            lines = [f"{project} | concepts:{len(proj_concepts)} papers:{distilled}/{total} frs:{len(frs)} exists:{len(caps_exist)} planned:{len(caps_planned)}"]
            if top_concepts:
                lines.append("top concepts: " + ", ".join(f"{c}({s:.0%})" for c, s in top_concepts[:8]))
            if frs:
                lines.append("open FRs: " + ", ".join(truncate(f.get("title", "?"), 40) for f in frs[:5]))
            return "\n".join(lines)

        def full():
            lines = [f"{project} landscape"]
            lines.append(f"papers: {distilled} distilled / {total} total")
            lines.append(f"concepts ({len(proj_concepts)}):")
            for c, s in top_concepts[:15]:
                lines.append(f"  {c} ({s:.0%})")
            if caps_exist:
                lines.append(f"exists ({len(caps_exist)}):")
                for c in caps_exist:
                    lines.append(f"  {c}")
            if caps_planned:
                lines.append(f"planned ({len(caps_planned)}):")
                for c in caps_planned:
                    lines.append(f"  {c}")
            if frs:
                lines.append(f"open FRs ({len(frs)}):")
                for f in frs:
                    lines.append(f"  [{f.get('priority','?')}] {f.get('title','?')}")
            return "\n".join(lines)

        return brief_or_full(brief, full, detail=detail)

    # ------------------------------------------------------------------
    # RR-10: Environment Health Checks
    # ------------------------------------------------------------------

    @mcp.tool()
    async def health_check() -> str:
        """Check Ollama version, model availability, and system health."""
        import subprocess
        import shutil

        lines = []

        # Ollama version
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
            version = result.stdout.strip().replace("ollama version is ", "")
            lines.append(f"ollama: {version}")
        except Exception as e:
            lines.append(f"ollama: ERROR ({e})")

        # Check required models
        models_config = pipeline.config.get("models", {})
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            installed = {line.split()[0].split(":")[0] for line in result.stdout.strip().split("\n")[1:] if line.strip()}
            for role, model in models_config.items():
                model_base = model.split(":")[0]
                status = "ok" if model_base in installed else "MISSING"
                lines.append(f"  {role}: {model} [{status}]")
        except Exception as e:
            lines.append(f"  models: ERROR ({e})")

        # DB size
        from pathlib import Path
        db_path = pipeline.config.get("db_path", "data/researcher.db")
        if Path(db_path).exists():
            size_mb = Path(db_path).stat().st_size / (1024 * 1024)
            lines.append(f"db: {size_mb:.1f}MB")

        # Disk space
        disk = shutil.disk_usage("/")
        free_gb = disk.free / (1024**3)
        lines.append(f"disk: {free_gb:.1f}GB free")

        # Paper stats
        distilled = len(pipeline.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED))
        pending = len(pipeline.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED))
        lines.append(f"papers: {distilled} distilled, {pending} pending")

        return "\n".join(lines)

    return mcp


def main():
    parser = argparse.ArgumentParser(description="khonliang-researcher MCP server")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--transport", default="stdio", choices=["stdio"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Override db_path if provided
    import yaml
    config_path = Path(args.config)
    if args.db and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        config["db_path"] = args.db
        # Write temp config (or just pass directly)

    pipeline = create_pipeline(args.config)
    mcp = create_research_server(pipeline)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
