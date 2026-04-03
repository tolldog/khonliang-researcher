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
    base = KhonliangMCPServer(
        knowledge_store=pipeline.knowledge,
        triple_store=pipeline.triples,
    )
    mcp = base.create_app()
    mcp.name = "khonliang-researcher"

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
