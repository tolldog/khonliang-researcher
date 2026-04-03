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

from khonliang.mcp import KhonliangMCPServer

from researcher.pipeline import create_pipeline, ResearchPipeline

logger = logging.getLogger(__name__)


def create_research_server(pipeline: ResearchPipeline):
    """Create MCP server with khonliang base tools + custom research tools."""
    from researcher.synthesizer import Synthesizer
    from researcher.worker import DistillWorker

    base = KhonliangMCPServer(
        knowledge_store=pipeline.knowledge,
        triple_store=pipeline.triples,
    )
    mcp = base.create_app()

    synthesizer = Synthesizer(pipeline.knowledge, pipeline.triples, pipeline.pool)
    worker = DistillWorker(pipeline)

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
                return f"Ingested: {title}\nEntry ID: {entry_id}"
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
            from khonliang.knowledge.store import KnowledgeEntry, Tier

            entry_id = hashlib.sha256(path.encode()).hexdigest()[:16]
            entry = KnowledgeEntry(
                id=entry_id,
                tier=Tier.IMPORTED,
                title=result.title or path,
                content=result.content,
                source=result.url,
                scope="research",
                tags=["paper", "undistilled", f"format:{result.format.value}"],
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
    async def browse_feeds(query: str = "", feeds: str = "") -> str:
        """Browse AI research blog RSS feeds for recent posts.

        Fetches RSS/Atom feeds from: Anthropic, OpenAI, DeepMind, HuggingFace,
        Ollama, LangChain, LlamaIndex, Cohere, Mistral.

        If query is provided, filters entries by keyword match.
        Pass comma-separated feed names to limit sources.
        Use fetch_paper(url) to ingest any interesting posts.
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

        if not entries:
            return f"No feed entries found{f' for: {query}' if query else ''}."

        lines = [f"Found {len(entries)} posts{f' matching \"{query}\"' if query else ''}:\n"]
        for i, e in enumerate(entries[:30], 1):
            pub = e.metadata.get("published", "")[:10]
            lines.append(f"{i}. [{e.source}] {e.title}")
            if pub:
                lines.append(f"   {pub}")
            if e.content:
                lines.append(f"   {e.content[:150]}...")
            lines.append(f"   {e.url}")
            lines.append("")

        if len(entries) > 30:
            lines.append(f"... and {len(entries) - 30} more")

        return "\n".join(lines)

    @mcp.tool()
    def find_relevant(query: str, project: str = "") -> str:
        """Search papers by topic. Optionally filter by project relevance.

        Returns matching papers from the knowledge store with their summaries.
        """
        results = pipeline.search(query, limit=10)
        if not results:
            return f"No papers found for: {query}"

        lines = [f"Found {len(results)} results for '{query}':\n"]
        for entry in results:
            lines.append(f"[{entry.id}] {entry.title}")
            lines.append(f"  Tags: {', '.join(entry.tags or [])}")
            # Show first 150 chars of content
            preview = entry.content[:150].replace("\n", " ")
            lines.append(f"  {preview}...")
            lines.append("")

        return "\n".join(lines)

    @mcp.tool()
    def reading_list() -> str:
        """Show all ingested papers and their distillation status."""
        rl = pipeline.get_reading_list()
        pending = rl.get("pending", [])
        distilled = rl.get("distilled", [])

        lines = []
        if pending:
            lines.append(f"## Pending distillation ({len(pending)})\n")
            for p in pending:
                lines.append(f"- [{p['entry_id']}] {p['title']}")
                if p.get("url"):
                    lines.append(f"  {p['url']}")

        if distilled:
            lines.append(f"\n## Distilled ({len(distilled)})\n")
            for p in distilled:
                lines.append(f"- [{p['entry_id']}] {p['title']}")

        if not pending and not distilled:
            lines.append("No papers ingested yet. Use fetch_paper(url) to start.")

        return "\n".join(lines)

    @mcp.tool()
    def paper_context(query: str) -> str:
        """Build rich context from papers and triples for prompt injection.

        Combines relevant paper summaries with relationship triples.
        """
        return pipeline.get_paper_context(query)

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
        pending = worker._count_pending()
        if pending == 0:
            return "No pending papers to distill."

        limit = batch_size if batch_size > 0 else None
        target = min(pending, batch_size) if batch_size > 0 else pending

        stats = await worker.run_batch(limit=limit)
        return (
            f"Distillation complete.\n"
            f"  Processed: {stats['processed']}\n"
            f"  Failed: {stats['failed']}\n"
            f"  Remaining: {worker._count_pending()}"
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
    # Synthesis tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def synthesize_topic(topic: str) -> str:
        """Generate a combined summary of papers related to a topic.

        Searches distilled papers matching the topic and produces a
        cross-paper analysis covering themes, methods, gaps, and connections.
        """
        result = await synthesizer.topic_summary(topic)
        if not result.success:
            return result.content
        return f"# Topic: {topic} ({result.paper_count} papers)\n\n{result.content}"

    @mcp.tool()
    async def synthesize_project(project: str = "autostock") -> str:
        """Generate an applicability brief for a project.

        Analyzes all distilled papers and ranks them by relevance to the
        specified project (from config.yaml). Returns prioritized recommendations.
        """
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
        return f"# {project} Brief ({result.paper_count} papers)\n\n{result.content}"

    @mcp.tool()
    async def synthesize_landscape() -> str:
        """Generate a research landscape overview across all distilled papers.

        Maps major directions, emerging trends, consensus views,
        contested areas, and gaps in the literature.
        """
        result = await synthesizer.landscape()
        if not result.success:
            return result.content
        return f"# Research Landscape ({result.paper_count} papers)\n\n{result.content}"

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
