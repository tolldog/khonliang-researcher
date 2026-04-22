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


def _filter_taxonomy(taxonomy: dict[str, Any], audience: str = "") -> tuple[list[dict], list[dict]]:
    groups = list(taxonomy.get("groups", []))
    relationships = list(taxonomy.get("relationships", []))
    audience = str(audience or "").strip()
    if not audience:
        return groups, relationships

    selected_codes = {g["code"] for g in groups if g.get("audience") == audience}
    parent_codes = {
        rel["target"]
        for rel in relationships
        if rel.get("source") in selected_codes and rel.get("predicate") == "specializes"
    }
    selected_codes |= parent_codes
    return (
        [g for g in groups if g.get("code") in selected_codes],
        [
            rel for rel in relationships
            if rel.get("source") in selected_codes and rel.get("target") in selected_codes
        ],
    )


def _format_concept_taxonomy_limited(
    taxonomy: dict[str, Any],
    *,
    audience: str = "",
    detail: str = "brief",
    limit: int = 50,
) -> str:
    groups, relationships = _filter_taxonomy(taxonomy, audience)
    audience = str(audience or "").strip()
    if not groups:
        suffix = f" for audience '{audience}'" if audience else ""
        return f"No taxonomy groups{suffix}. Distill some papers first."

    groups = sorted(groups, key=lambda g: (g.get("audience", ""), g.get("code", "")))
    relationships = sorted(relationships, key=lambda r: (r.get("source", ""), r.get("target", "")))
    total_groups = len(groups)
    max_groups = max(1, int(limit))
    shown_groups = groups[:max_groups]
    shown_group_codes = {g.get("code") for g in shown_groups}
    relationships = [
        rel
        for rel in relationships
        if rel.get("source") in shown_group_codes and rel.get("target") in shown_group_codes
    ]

    def compact():
        return "\n".join(
            f"{g['code']}|{_compact_field(g.get('audience'), 32)}|"
            f"{_compact_field(g.get('label'), 60)}|entities={len(g.get('entities', []))}"
            for g in shown_groups
        )

    def brief():
        lines = [
            f"Concept taxonomy: {total_groups} groups, showing {len(shown_groups)}, "
            f"{len(relationships)} relationships"
        ]
        for group in shown_groups:
            entities = ", ".join(group.get("entities", [])[:3])
            if len(group.get("entities", [])) > 3:
                entities += f", +{len(group['entities']) - 3} more"
            lines.append(
                f"{group['code']} [{group.get('audience', 'general')}] "
                f"{group.get('label', '?')} ({len(group.get('entities', []))})"
            )
            if entities:
                lines.append(f"  {entities}")
        return "\n".join(lines)

    def full():
        lines = [brief()]
        if relationships:
            lines.append("\nRelationships:")
            for rel in relationships[:80]:
                lines.append(f"  {rel['source']} -[{rel['predicate']}]-> {rel['target']}")
        return "\n".join(lines)

    return format_response(compact, brief, full, detail)


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
4. `synergize_concepts()` — find concept bundles for developer to turn into FRs

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
- `project_landscape(project)` — per-project view with concepts and capabilities
- `brief_on(topic, in_context_of)` — single-shot topic-in-context brief
  (multi-query recall; reuses stored distills, no new LLM synth call)

## Feature Requests
Researcher no longer owns active FR lifecycle. Use developer for
`promote_fr`, `feature_requests`, `next_fr`, status updates, dependencies,
milestones, and work units. Researcher keeps historical FR records only so
paper/evidence references remain resolvable by ID.

- `synergize_concepts()` — concept bundles for developer to evaluate
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
        from khonliang_researcher import (
            build_concept_graph,
            find_paths,
            format_entity_suggestions,
            suggest_entities,
        )

        graph = build_concept_graph(pipeline.triples, knowledge=pipeline.knowledge)
        paths = find_paths(graph, start, end)
        if not paths:
            suggestion_lines = []
            if start not in graph:
                start_suggestions = format_entity_suggestions(suggest_entities(graph, start))
                if start_suggestions:
                    suggestion_lines.append(f"Start {start_suggestions}")
            if end not in graph:
                end_suggestions = format_entity_suggestions(suggest_entities(graph, end))
                if end_suggestions:
                    suggestion_lines.append(f"End {end_suggestions}")
            suffix = "\n" + "\n".join(suggestion_lines) if suggestion_lines else ""
            return f"No path found from '{start}' to '{end}'." + suffix

        lines = [f"Found {len(paths)} path(s) from '{start}' to '{end}':\n"]
        for i, path in enumerate(paths[:5], 1):
            chain = " → ".join(
                f"{node} —[{pred}]→ {target}"
                for node, pred, target in path
            )
            lines.append(f"  {i}. {chain}")
        return "\n".join(lines)

    @mcp.tool()
    def concept_taxonomy(
        audience: str = "",
        universal_concepts: str = "",
        limit: int = 50,
        detail: str = "brief",
    ) -> str:
        """Audience-scoped concept taxonomy. detail: compact|brief|full.

        Groups existing graph nodes into deterministic taxonomy buckets with
        stable codes, and links domain-specific concepts to universal parents
        when ``universal_concepts`` names those parent patterns.
        """
        from khonliang_researcher import build_concept_graph, build_concept_taxonomy
        from researcher.util import split_csv

        graph = build_concept_graph(pipeline.triples, knowledge=pipeline.knowledge)
        taxonomy = build_concept_taxonomy(
            graph,
            universal_concepts=split_csv(universal_concepts),
        )
        return _format_concept_taxonomy_limited(
            taxonomy,
            audience=audience,
            detail=detail,
            limit=limit,
        )

    @mcp.tool()
    def investigation_workspace(
        seeds: str,
        label: str = "",
        branches: str = "",
        depth: int = 2,
        branches_per_node: int = 4,
        detail: str = "brief",
    ) -> str:
        """Create a temporary branchable evidence workspace. detail: compact|brief|full.

        Branch specs use ``label:seed one,seed two`` and may be separated by
        semicolons. The workspace references corpus papers one-way so the
        long-lived library graph is not polluted by exploratory labels.
        """
        from khonliang_researcher import (
            build_investigation_workspace,
            format_investigation_workspace,
        )
        from researcher.util import parse_branch_specs

        workspace = build_investigation_workspace(
            pipeline.triples,
            seeds=seeds,
            label=label,
            branch_specs=parse_branch_specs(branches),
            knowledge=pipeline.knowledge,
            max_depth=depth,
            max_branches=branches_per_node,
        )
        return format_investigation_workspace(workspace, detail=detail)

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

    @mcp.tool()
    def concept_map_freshness(detail: str = "brief") -> str:
        """Freshness signal for the concept graph relative to distilled papers.

        Cheap — aggregate queries only, no corpus scan. Consumers (e.g.
        developer's milestone planner) use this to decide whether current
        concept neighborhoods are fresh enough to drive sequencing, or
        whether to trigger a re-ingestion first.

        detail=compact  pipe-delimited key=value, for agent loops
        detail=brief    structured summary (default)
        detail=full     same summary plus raw timestamps
        """
        f = pipeline.concept_map_freshness()

        def compact():
            return (
                f"fresh={str(f['fresh']).lower()}|"
                f"pending={f['pending_distilled']}|"
                f"triples={f['totals']['triples']}|"
                f"distilled={f['totals']['distilled_papers']}"
            )

        def brief():
            lag = f["lag_seconds"]
            activity_txt = f"{lag / 3600:.1f}h ago" if lag is not None else "none yet"
            status = "fresh" if f["fresh"] else f"stale ({f['pending_distilled']} pending)"
            return (
                f"Concept map: {status}\n"
                f"  triples: {f['totals']['triples']}\n"
                f"  distilled papers: {f['totals']['distilled_papers']}\n"
                f"  last triple activity: {activity_txt}"
            )

        def full():
            return json.dumps(f, indent=2, default=str)

        return format_response(compact, brief, full, detail)

    # ------------------------------------------------------------------
    # Synergize tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def synergize_concepts(min_score: float = 0.5, max_concepts: int = 10, detail: str = "brief") -> str:
        """Find conceptual connections, return bundles (no FRs). detail: compact|brief|full.

        Generic concept bundling — groups related concepts based on shared
        themes, methods, or complementary findings. What to DO with the
        bundles (FRs, research leads) is the caller's decision.
        """
        result = await pipeline.synergize_concepts(min_score=min_score, max_concepts=max_concepts)
        if "error" in result:
            raw = result.get("raw", "")
            return f"Error: {result['error']}\n{raw}" if raw else f"Error: {result['error']}"

        bundles = result.get("bundles", [])
        count = result.get("concept_count", 0)
        papers = result.get("paper_count", 0)

        def compact():
            names = [b.get("name", "?") for b in bundles] if isinstance(bundles, list) else []
            return f"bundles={count}|papers={papers}|" + "; ".join(names[:5])

        def brief():
            if not isinstance(bundles, list):
                return str(bundles)
            lines = [f"{count} concept bundles from {papers} papers"]
            for b in bundles:
                strength = b.get("strength", 0)
                lines.append(f"\n{b.get('name', '?')} (strength: {strength:.0%})")
                lines.append(f"  concepts: {', '.join(b.get('concepts', []))}")
                lines.append(f"  {b.get('summary', '')}")
            return "\n".join(lines)

        def full():
            if not isinstance(bundles, list):
                return str(bundles)
            lines = [f"{count} concept bundles from {papers} papers"]
            for b in bundles:
                strength = b.get("strength", 0)
                lines.append(f"\n## {b.get('name', '?')} (strength: {strength:.0%})")
                lines.append(f"Concepts: {', '.join(b.get('concepts', []))}")
                lines.append(f"Connection: {b.get('connection', '?')}")
                lines.append(f"Summary: {b.get('summary', '')}")
                papers_list = b.get("papers", [])
                if papers_list:
                    lines.append(f"Papers: {', '.join(papers_list)}")
            return "\n".join(lines)

        return format_response(compact, brief, full, detail)

    @mcp.tool()
    async def synergize(min_score: float = 0.5, max_concepts: int = 10, detail: str = "brief") -> str:
        """[DEPRECATED] Classify concepts and return candidate FRs. detail: compact|brief|full.

        Researcher is the knowledge layer; FR generation belongs to developer.
        Use `synergize_concepts` for concept bundles. This compatibility
        function no longer writes researcher-owned FR records.
        """
        result = await pipeline.synergize(min_score=min_score, max_concepts=max_concepts)

        # Visible deprecation notice for MCP consumers — Python warnings don't
        # cross the bus. Prepended to every response so callers see it at use.
        _DEPRECATION_NOTICE = (
            "[deprecated] researcher.synergize returns candidate FR text only "
            "and no longer writes researcher FR records. Use synergize_concepts "
            "for bundles; developer owns accepted FRs.\n\n"
        )

        if "error" in result:
            raw = result.get("raw", "")
            error_txt = f"Error: {result['error']}\n\n{raw}" if raw else f"Error: {result['error']}"
            return _DEPRECATION_NOTICE + error_txt

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

        return _DEPRECATION_NOTICE + format_response(compact, brief, full, detail)

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
    def brief_on(
        topic: str,
        in_context_of: str = "",
        project: str = "",
        detail: str = "brief",
        top_k: int = 5,
    ) -> str:
        """Topic-in-context brief over the corpus.

        Single-shot alternative to synthesize_topic that (a) accepts an
        ``in_context_of`` scoping phrase and (b) expands recall via
        multi-query retrieval (topic / topic+context / context) and then
        bundles per-paper summaries under the detail-level budget.

        Does not spin up a new distillation pipeline — it reuses whatever
        summary ``distill_paper`` already produced for each selected
        entry, falling back to the entry's own content head if no
        summary is stored.

        Returns JSON:
          {brief, source_ids, retrieval_diagnostics: {queries_run,
           total_hits, top_k_chosen, per_query_hits}}

        detail: compact (TL;DR + bare id list), brief (TL;DR + one-line
        per source, target <=2000 chars), full (per-source paragraph).
        """
        topic = (topic or "").strip()
        ctx = (in_context_of or "").strip()
        if not topic:
            return json.dumps({
                "brief": "brief_on requires a non-empty topic.",
                "source_ids": [],
                "retrieval_diagnostics": {
                    "queries_run": [],
                    "total_hits": 0,
                    "top_k_chosen": 0,
                },
            })

        # 1. Multi-query expansion. Union + dedup by entry id; track which
        #    queries surfaced each entry so we can boost entries that
        #    appear in BOTH topic and context queries (per FR ranking
        #    guidance).
        queries: list[str] = [topic]
        if ctx:
            queries.append(f"{topic} {ctx}")
            queries.append(ctx)

        hits_per_query: dict[str, list[str]] = {}
        entries_by_id: dict[str, Any] = {}
        rank_reciprocal_sum: dict[str, float] = {}
        query_hit_counts: dict[str, int] = {}

        threshold = pipeline.relevance.threshold if project else 0.0
        for q in queries:
            results = pipeline.search(q, limit=10)
            if project:
                results = [
                    e for e in results
                    if (e.metadata or {}).get("relevance_scores", {})
                    .get(project, 0) >= threshold
                ]
            hits_per_query[q] = [e.id for e in results]
            for rank, entry in enumerate(results):
                entries_by_id.setdefault(entry.id, entry)
                rank_reciprocal_sum[entry.id] = (
                    rank_reciprocal_sum.get(entry.id, 0.0) + 1.0 / (rank + 1)
                )
                query_hit_counts[entry.id] = query_hit_counts.get(entry.id, 0) + 1

        total_hits = len(entries_by_id)

        if not entries_by_id:
            return json.dumps({
                "brief": (
                    f"No corpus matches for topic='{topic}'"
                    + (f" in_context_of='{ctx}'" if ctx else "")
                    + "."
                ),
                "source_ids": [],
                "retrieval_diagnostics": {
                    "queries_run": queries,
                    "total_hits": 0,
                    "top_k_chosen": 0,
                    "per_query_hits": {q: len(ids) for q, ids in hits_per_query.items()},
                },
            })

        # 2. Rank: (query_hit_count desc, reciprocal sum desc). Entries
        #    hit by both topic and context get strict preference.
        def _score(eid: str) -> tuple[int, float]:
            return (query_hit_counts[eid], rank_reciprocal_sum[eid])

        ranked_ids = sorted(entries_by_id.keys(), key=_score, reverse=True)
        top_k = max(1, int(top_k))
        chosen_ids = ranked_ids[:top_k]

        # 3. For each chosen entry, load its already-distilled summary if
        #    one exists — reusing distill_paper's output keeps the
        #    "10x-outlier survives unchanged" distill invariant intact.
        #    Fall back to the entry's own content head when no summary
        #    has been produced yet.
        per_source: list[dict[str, Any]] = []
        for eid in chosen_ids:
            entry = entries_by_id[eid]
            key_claim = ""
            summary_entry = pipeline.knowledge.get(f"{eid}_summary")
            if summary_entry is not None:
                try:
                    summary_data = json.loads(summary_entry.content)
                except (json.JSONDecodeError, TypeError):
                    summary_data = None
                if isinstance(summary_data, dict):
                    findings = summary_data.get("key_findings") or []
                    if findings and isinstance(findings[0], str):
                        key_claim = findings[0]
                    if not key_claim:
                        key_claim = summary_data.get("abstract", "") or ""
            if not key_claim:
                key_claim = (entry.content or "").strip().split("\n", 1)[0]
            per_source.append({
                "id": eid,
                "title": entry.title,
                "key_claim": truncate(key_claim, 220),
            })

        # 4. Bundle under the detail budget. No new LLM synth call —
        #    brief_on is intentionally a cheap retrieval+reuse primitive,
        #    distinct from synthesize_topic which does run a synth LLM.
        header = f"{topic}"
        if ctx:
            header += f" (context: {ctx})"
        header += f" - {len(chosen_ids)} of {total_hits} matching entries"

        def compact() -> str:
            ids = ",".join(chosen_ids)
            return f"topic={_compact_field(topic, 60)}|context={_compact_field(ctx, 60)}|ids={ids}"

        def brief() -> str:
            lines = [header]
            for s in per_source:
                line = f"[{s['id']}] {truncate(s['title'], 80)} - {s['key_claim']}"
                lines.append(truncate(line, 300))
            out = "\n".join(lines)
            if len(out) > 2000:
                out = out[:1997] + "..."
            return out

        def full() -> str:
            lines = [f"# {header}\n"]
            for s in per_source:
                lines.append(f"## [{s['id']}] {s['title']}")
                if s["key_claim"]:
                    lines.append(s["key_claim"])
                lines.append("")
            return "\n".join(lines)

        brief_text = format_response(compact, brief, full, detail)

        return json.dumps({
            "brief": brief_text,
            "source_ids": chosen_ids,
            "retrieval_diagnostics": {
                "queries_run": queries,
                "total_hits": total_hits,
                "top_k_chosen": len(chosen_ids),
                "per_query_hits": {q: len(ids) for q, ids in hits_per_query.items()},
            },
        })

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
        paper counts, and capabilities. detail='brief' or 'full'."""
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
            lines = [f"{project} | concepts:{len(proj_concepts)} papers:{distilled}/{total} exists:{len(caps_exist)} planned:{len(caps_planned)}"]
            if top_concepts:
                lines.append("top concepts: " + ", ".join(f"{c}({s:.0%})" for c, s in top_concepts[:8]))
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
