"""CLI for khonliang-researcher.

Usage:
    khonliang-researcher fetch URL
    khonliang-researcher fetch-list URL
    khonliang-researcher distill ENTRY_ID
    khonliang-researcher distill --all
    khonliang-researcher search QUERY
    khonliang-researcher list
    khonliang-researcher serve
"""

import asyncio
import json
import sys

import click

from researcher.pipeline import create_pipeline


def _run(coro):
    """Run an async coroutine."""
    return asyncio.run(coro)


@click.group()
@click.option("--config", default="config.yaml", help="Config file path")
@click.pass_context
def cli(ctx, config):
    """khonliang-researcher — Research paper ingestion and distillation."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


def _get_pipeline(ctx):
    if "pipeline" not in ctx.obj:
        ctx.obj["pipeline"] = create_pipeline(ctx.obj["config"])
    return ctx.obj["pipeline"]


@cli.command()
@click.argument("url")
@click.pass_context
def fetch(ctx, url):
    """Fetch and store a research paper."""
    pipeline = _get_pipeline(ctx)

    async def _fetch():
        entry_id = await pipeline.ingest_paper(url)
        if entry_id:
            entry = pipeline.knowledge.get(entry_id)
            title = entry.title if entry else url
            click.echo(f"Ingested: {title}")
            click.echo(f"Entry ID: {entry_id}")
        else:
            click.echo(f"Failed to ingest: {url}", err=True)
            sys.exit(1)

    _run(_fetch())


@cli.command("fetch-list")
@click.argument("url")
@click.option("--auto-fetch", is_flag=True, help="Auto-fetch all discovered papers")
@click.option("--max", "max_papers", default=50, help="Max papers to auto-fetch")
@click.pass_context
def fetch_list(ctx, url, auto_fetch, max_papers):
    """Parse a paper list URL and show/fetch discovered papers."""
    pipeline = _get_pipeline(ctx)

    async def _parse():
        papers = await pipeline.ingest_paper_list(url)
        if not papers:
            click.echo("No papers found.")
            return

        click.echo(f"Found {len(papers)} papers:\n")
        current_cat = ""
        for i, p in enumerate(papers):
            if p.category != current_cat:
                current_cat = p.category
                click.echo(f"\n  {current_cat}")
            click.echo(f"  {i+1:3d}. {p.title or '(untitled)'}")
            click.echo(f"       {p.url}")

        if auto_fetch:
            to_fetch = papers[:max_papers]
            click.echo(f"\nFetching {len(to_fetch)} papers...")
            entry_ids = await pipeline.ingest_papers_from_list(to_fetch)
            click.echo(f"Ingested {len(entry_ids)} papers.")

    _run(_parse())


@cli.command()
@click.argument("entry_id", required=False)
@click.option("--all", "distill_all", is_flag=True, help="Distill all pending papers")
@click.pass_context
def distill(ctx, entry_id, distill_all):
    """Distill a paper (or all pending papers)."""
    pipeline = _get_pipeline(ctx)

    async def _distill():
        if distill_all:
            results = await pipeline.distill_all_pending()
            if not results:
                click.echo("No pending papers to distill.")
                return
            for r in results:
                status = "ok" if r.success else "FAILED"
                click.echo(f"  [{status}] {r.title}")
                if r.success and r.triples:
                    click.echo(f"         {len(r.triples)} triples extracted")
        elif entry_id:
            result = await pipeline.distill(entry_id)
            if result.success:
                click.echo(f"Distilled: {result.title}")
                if result.summary:
                    click.echo(json.dumps(result.summary, indent=2))
                if result.triples:
                    click.echo(f"\n{len(result.triples)} triples:")
                    for t in result.triples:
                        click.echo(
                            f"  {t.get('subject')} "
                            f"—[{t.get('predicate')}]→ "
                            f"{t.get('object')} ({t.get('confidence', 0):.0%})"
                        )
                if result.assessments:
                    click.echo("\nApplicability:")
                    for proj, assess in result.assessments.items():
                        if isinstance(assess, dict):
                            click.echo(f"  {proj}: {assess.get('score', 0):.0%}")
            else:
                click.echo(f"Distillation failed: {result.title}", err=True)
        else:
            click.echo("Provide an ENTRY_ID or use --all", err=True)

    _run(_distill())


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, help="Max results")
@click.pass_context
def search(ctx, query, limit):
    """Search papers and summaries."""
    pipeline = _get_pipeline(ctx)
    results = pipeline.search(query, limit=limit)

    if not results:
        click.echo(f"No results for: {query}")
        return

    click.echo(f"Found {len(results)} results:\n")
    for entry in results:
        tier = entry.tier.value if hasattr(entry.tier, "value") else entry.tier
        click.echo(f"[{entry.id}] Tier {tier} — {entry.title}")
        click.echo(f"  Tags: {', '.join(entry.tags or [])}")
        preview = entry.content[:120].replace("\n", " ")
        click.echo(f"  {preview}...")
        click.echo()


@cli.command("find")
@click.argument("query")
@click.option("--max", "max_results", default=20, help="Max results")
@click.option("--sort", "sort_by", default="relevance", help="Sort: relevance, lastUpdatedDate, submittedDate")
@click.pass_context
def find_papers(ctx, query, max_results, sort_by):
    """Search arxiv for papers matching keywords."""
    from researcher.fetcher import search_arxiv

    async def _search():
        results = await search_arxiv(query, max_results=max_results, sort_by=sort_by)
        if not results:
            click.echo(f"No papers found for: {query}")
            return

        click.echo(f"Found {len(results)} papers:\n")
        for i, r in enumerate(results, 1):
            authors = ", ".join(r.authors[:3])
            click.echo(f"  {i:3d}. {r.title}")
            click.echo(f"       {authors}")
            click.echo(f"       {r.url}")
            click.echo()

    _run(_search())


@cli.command("list")
@click.pass_context
def reading_list(ctx):
    """Show ingested papers and their status."""
    pipeline = _get_pipeline(ctx)
    rl = pipeline.get_reading_list()

    pending = rl.get("pending", [])
    distilled = rl.get("distilled", [])

    if pending:
        click.echo(f"Pending distillation ({len(pending)}):\n")
        for p in pending:
            click.echo(f"  [{p['entry_id']}] {p['title']}")

    if distilled:
        click.echo(f"\nDistilled ({len(distilled)}):\n")
        for p in distilled:
            click.echo(f"  [{p['entry_id']}] {p['title']}")

    if not pending and not distilled:
        click.echo("No papers yet. Use 'fetch' to start.")


@cli.command()
@click.option("--batch", type=int, default=0, help="Process N papers (0=all)")
@click.option("--pause", type=float, default=2.0, help="Seconds between papers")
@click.pass_context
def worker(ctx, batch, pause):
    """Run the distillation worker on pending papers."""
    from researcher.worker import DistillWorker

    pipeline = _get_pipeline(ctx)
    w = DistillWorker(pipeline, pause_between=pause)

    pending = w._count_pending()
    if pending == 0:
        click.echo("No pending papers.")
        return

    target = min(pending, batch) if batch > 0 else pending
    click.echo(f"Processing {target} papers...\n")

    async def _run_worker():
        limit = batch if batch > 0 else None
        return await w.run_batch(limit=limit)

    stats = _run(_run_worker())
    click.echo(f"\nDone: {stats['processed']} processed, {stats['failed']} failed")
    click.echo(f"Remaining: {w._count_pending()}")


@cli.group()
@click.pass_context
def synthesize(ctx):
    """Generate combined summaries across papers."""
    pass


@synthesize.command("topic")
@click.argument("topic")
@click.pass_context
def synth_topic(ctx, topic):
    """Synthesize papers around a topic."""
    from researcher.synthesizer import Synthesizer

    pipeline = _get_pipeline(ctx)
    s = Synthesizer(pipeline.knowledge, pipeline.triples, pipeline.pool)

    async def _synth():
        result = await s.topic_summary(topic)
        if result.success:
            click.echo(f"# {topic} ({result.paper_count} papers)\n")
            click.echo(result.content)
        else:
            click.echo(result.content, err=True)

    _run(_synth())


@synthesize.command("project")
@click.argument("project", default="autostock")
@click.pass_context
def synth_project(ctx, project):
    """Generate applicability brief for a project."""
    from researcher.synthesizer import Synthesizer

    pipeline = _get_pipeline(ctx)
    projects = pipeline.config.get("projects", {})

    if project not in projects:
        click.echo(f"Project '{project}' not in config. Available: {', '.join(projects.keys())}", err=True)
        return

    s = Synthesizer(pipeline.knowledge, pipeline.triples, pipeline.pool)
    cfg = projects[project]

    async def _synth():
        result = await s.project_brief(project, cfg.get("description", ""))
        if result.success:
            click.echo(f"# {project} Brief ({result.paper_count} papers)\n")
            click.echo(result.content)
        else:
            click.echo(result.content, err=True)

    _run(_synth())


@synthesize.command("landscape")
@click.pass_context
def synth_landscape(ctx):
    """Map the research landscape across all distilled papers."""
    from researcher.synthesizer import Synthesizer

    pipeline = _get_pipeline(ctx)
    s = Synthesizer(pipeline.knowledge, pipeline.triples, pipeline.pool)

    async def _synth():
        result = await s.landscape()
        if result.success:
            click.echo(f"# Research Landscape ({result.paper_count} papers)\n")
            click.echo(result.content)
        else:
            click.echo(result.content, err=True)

    _run(_synth())


@cli.command()
@click.option("--transport", default="stdio", help="MCP transport")
@click.pass_context
def serve(ctx, transport):
    """Run the MCP server."""
    from researcher.server import create_research_server

    pipeline = _get_pipeline(ctx)
    mcp = create_research_server(pipeline)
    click.echo("Starting khonliang-researcher MCP server...", err=True)
    mcp.run(transport=transport)


def main():
    cli()


if __name__ == "__main__":
    main()
