"""CLI for khonliang-researcher.

Usage:
    khonliang-researcher fetch URL
    khonliang-researcher fetch-list URL
    khonliang-researcher distill ENTRY_ID
    khonliang-researcher distill --all
    khonliang-researcher search QUERY
    khonliang-researcher list
    khonliang-researcher worker
    khonliang-researcher health
    khonliang-researcher scan PROJECT
    khonliang-researcher ingest-github REPO_URL
    khonliang-researcher research-caps [PROJECT]
    khonliang-researcher graph tree|matrix|path|project
    khonliang-researcher project landscape|capabilities
    khonliang-researcher repo register|list
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


# ------------------------------------------------------------------
# Paper ingestion
# ------------------------------------------------------------------

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


@cli.command("ingest-file")
@click.argument("path")
@click.pass_context
def ingest_file(ctx, path):
    """Ingest a local file (PDF, HTML, Markdown, or text)."""
    from researcher.fetcher import fetch_file

    pipeline = _get_pipeline(ctx)

    async def _ingest():
        import hashlib
        from khonliang.knowledge.store import KnowledgeEntry, Tier, EntryStatus

        result = await fetch_file(path)
        if not result.content.strip():
            click.echo(f"No text content extracted from: {path}", err=True)
            sys.exit(1)

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
        click.echo(f"Ingested: {result.title} ({result.format.value})")
        click.echo(f"Entry ID: {entry_id}")

    _run(_ingest())


# ------------------------------------------------------------------
# Distillation
# ------------------------------------------------------------------

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
@click.option("--batch", type=int, default=0, help="Process N papers (0=all)")
@click.option("--pause", type=float, default=2.0, help="Seconds between papers")
@click.pass_context
def worker(ctx, batch, pause):
    """Run the distillation worker on pending papers."""
    from researcher.worker import DistillWorker

    pipeline = _get_pipeline(ctx)
    w = DistillWorker(pipeline, pause_between=pause)

    pending = w.count_pending()
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
    click.echo(f"Remaining: {w.count_pending()}")


# ------------------------------------------------------------------
# Search and discovery
# ------------------------------------------------------------------

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


@cli.command("feeds")
@click.option("--query", "-q", default="", help="Filter by keywords")
@click.option("--feeds", "-f", default="", help="Comma-separated feed names")
@click.option("--ingest", is_flag=True, help="Auto-ingest all matching posts")
@click.pass_context
def browse_feeds(ctx, query, feeds, ingest):
    """Browse AI research blog RSS feeds."""
    from researcher.rss import fetch_all_feeds, DEFAULT_FEEDS

    pipeline = _get_pipeline(ctx)
    feed_list = [f.strip() for f in feeds.split(",") if f.strip()] or None

    async def _browse():
        entries = await fetch_all_feeds(feed_list)

        if query:
            keywords = query.lower().split()
            entries = [
                e for e in entries
                if any(kw in f"{e.title} {e.content}".lower() for kw in keywords)
            ]

        if not entries:
            click.echo("No entries found.")
            return

        click.echo(f"Found {len(entries)} posts:\n")
        for i, e in enumerate(entries[:50], 1):
            pub = e.metadata.get("published", "")[:10]
            click.echo(f"  {i:3d}. [{e.source}] {e.title}")
            if pub:
                click.echo(f"       {pub}")
            click.echo(f"       {e.url}")

        if ingest:
            click.echo(f"\nIngesting {len(entries)} posts...")
            from researcher.parser import PaperReference
            refs = [PaperReference(title=e.title, url=e.url) for e in entries if e.url]
            ids = await pipeline.ingest_papers_from_list(refs, max_concurrent=3)
            click.echo(f"Ingested {len(ids)} posts.")

    _run(_browse())


@cli.command("context")
@click.argument("query")
@click.pass_context
def paper_context(ctx, query):
    """Build rich context from papers and triples for a query."""
    pipeline = _get_pipeline(ctx)
    result = pipeline.get_paper_context(query)
    if result:
        click.echo(result)
    else:
        click.echo(f"No context found for: {query}")


@cli.command("digest")
@click.option("--hours", default=24.0, help="Hours of activity to show")
@click.pass_context
def paper_digest(ctx, hours):
    """Show recent research activity digest."""
    pipeline = _get_pipeline(ctx)
    entries = pipeline.digest.get_since(hours=hours)
    if not entries:
        click.echo(f"No research activity in the last {hours} hours.")
        return

    click.echo(f"Research activity (last {hours}h): {len(entries)} events\n")
    for entry in entries:
        click.echo(f"- {entry.summary}")
        if entry.tags:
            click.echo(f"  Tags: {', '.join(entry.tags)}")


# ------------------------------------------------------------------
# Reading list and scoring
# ------------------------------------------------------------------

@cli.command("list")
@click.pass_context
def reading_list(ctx):
    """Show ingested papers and their status."""
    pipeline = _get_pipeline(ctx)
    rl = pipeline.get_reading_list()

    pending = rl.get("pending", [])
    distilled = rl.get("distilled", [])
    skipped = rl.get("skipped", [])
    failed = rl.get("failed", [])

    if pending:
        click.echo(f"Pending distillation ({len(pending)}):\n")
        for p in pending:
            click.echo(f"  [{p['entry_id']}] {p['title']}")

    if distilled:
        click.echo(f"\nDistilled ({len(distilled)}):\n")
        for p in distilled:
            click.echo(f"  [{p['entry_id']}] {p['title']}")

    if skipped:
        click.echo(f"\nSkipped — low relevance ({len(skipped)}):\n")
        for p in skipped:
            click.echo(f"  [{p['entry_id']}] {p['title']}")

    if failed:
        click.echo(f"\nFailed ({len(failed)}):\n")
        for p in failed:
            click.echo(f"  [{p['entry_id']}] {p['title']}")

    if not pending and not distilled:
        click.echo("No papers yet. Use 'fetch' to start.")


@cli.command("score")
@click.argument("entry_id")
@click.pass_context
def score_paper(ctx, entry_id):
    """Score a paper's relevance to all projects."""
    pipeline = _get_pipeline(ctx)

    async def _score():
        scores = await pipeline.score_relevance(entry_id)
        if not scores:
            click.echo(f"Could not score {entry_id}.")
            return

        entry = pipeline.knowledge.get(entry_id)
        click.echo(f"Relevance: {entry.title if entry else entry_id}\n")
        for project, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "=" * int(score * 40)
            click.echo(f"  {project:15s} {score:.3f} |{bar}")

        threshold = pipeline.relevance.threshold
        max_score = max(scores.values())
        if max_score < threshold:
            click.echo(f"\nBelow threshold ({threshold}) — would be skipped.")
        else:
            click.echo(f"\nAbove threshold ({threshold}) — will be distilled.")

    _run(_score())


# ------------------------------------------------------------------
# Synthesis
# ------------------------------------------------------------------

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
@click.argument("project", default="khonliang")
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


# ------------------------------------------------------------------
# Synergize and evaluate
# ------------------------------------------------------------------

@click.option("--min-score", default=0.5, help="Minimum concept score threshold")
@click.option("--max-concepts", default=30, help="Max concepts to analyze")
@click.pass_context
def synergize_cmd(ctx, min_score, max_concepts):
    """[DEPRECATED] Classify concepts and return candidate FR text.

    FR generation is moving from researcher to developer. No CLI equivalent
    ships here; use the `synergize_concepts` MCP tool or
    `ResearchPipeline.synergize_concepts()` Python API for concept bundles.
    This compatibility helper no longer writes researcher-owned FR records.
    """
    pipeline = _get_pipeline(ctx)

    async def _synergize():
        click.echo(
            "[deprecated] researcher.synergize emits FRs — that role moves to "
            "developer. For concept bundles, use the `synergize_concepts` MCP "
            "tool or `ResearchPipeline.synergize_concepts()` (no CLI equivalent).",
            err=True,
        )
        click.echo("Analyzing concepts across projects...")
        result = await pipeline.synergize(min_score=min_score, max_concepts=max_concepts)
        if "error" in result:
            click.echo(f"Error: {result['error']}", err=True)
            if result.get("raw"):
                click.echo(result["raw"])
            return

        click.echo(f"\n{result['concept_count']} concepts classified, {result['fr_count']} candidate FRs returned\n")
        for item in result["classifications"]:
            concept = item.get("concept", "?")
            cls = item.get("classification", "?")
            targets = ", ".join(item.get("targets", []))
            click.echo(f"  {concept} → {cls} [{targets}]")
            for fr in item.get("feature_requests", []):
                click.echo(f"    [{fr.get('priority', 'med')}] {fr.get('title', '?')} → {fr.get('target', '?')}")

    _run(_synergize())


@cli.command("evaluate")
@click.argument("capability")
@click.pass_context
def evaluate_cmd(ctx, capability):
    """Evaluate whether a khonliang capability could improve the researcher."""
    pipeline = _get_pipeline(ctx)

    async def _evaluate():
        result = await pipeline.evaluate_capability(capability)
        if "error" in result:
            click.echo(f"Error: {result['error']}", err=True)
            if result.get("raw"):
                click.echo(result["raw"])
            return

        applicable = result.get("applicable", "?")
        score = result.get("score", 0)
        click.echo(f"Applicable: {applicable}  Score: {score:.0%}")
        click.echo(f"Summary: {result.get('summary', '')}\n")

        for label, key in [("Direct uses", "direct_uses"), ("Improvements", "improvements"), ("New features", "new_features")]:
            items = result.get(key, [])
            if items:
                click.echo(f"{label}:")
                for item in items:
                    click.echo(f"  - {item}")

        if result.get("integration_notes"):
            click.echo(f"\nIntegration: {result['integration_notes']}")

        for fr in result.get("suggested_frs", []):
            dep = f" (needs {fr['depends_on']})" if fr.get("depends_on") else ""
            click.echo(f"\n  FR [{fr.get('priority', '?')}] {fr.get('title', '?')}{dep}")

    _run(_evaluate())


# ------------------------------------------------------------------
# Concept graph
# ------------------------------------------------------------------

@cli.group()
@click.pass_context
def graph(ctx):
    """Concept graph exploration."""
    pass


@graph.command("matrix")
@click.option("--min-connections", default=2, help="Min connections to include a concept")
@click.option("--max-concepts", default=30, help="Max concepts in matrix")
@click.pass_context
def graph_matrix(ctx, min_connections, max_concepts):
    """Show concept × paper matrix."""
    from khonliang_researcher import build_concept_matrix, format_matrix

    pipeline = _get_pipeline(ctx)
    matrix_data = build_concept_matrix(
        pipeline.triples,
        min_connections=min_connections,
        max_entities=max_concepts,
    )
    if not matrix_data["entities"]:
        click.echo("No concept matrix data. Distill some papers first.")
        return
    click.echo(format_matrix(matrix_data, pipeline.knowledge, pipeline.triples))


@graph.command("tree")
@click.argument("concept")
@click.option("--depth", default=4, help="Max traversal depth")
@click.option("--branches", default=3, help="Max branches per node")
@click.pass_context
def graph_tree(ctx, concept, depth, branches):
    """Trace a concept's connections through the knowledge graph."""
    from khonliang_researcher import build_concept_graph, trace_chain

    pipeline = _get_pipeline(ctx)
    g = build_concept_graph(pipeline.triples, knowledge=pipeline.knowledge)
    result = trace_chain(g, concept, max_depth=depth, max_branches=branches)
    click.echo(result)


@graph.command("path")
@click.argument("start")
@click.argument("end")
@click.pass_context
def graph_path(ctx, start, end):
    """Find how two concepts connect through the knowledge graph."""
    from khonliang_researcher import (
        build_concept_graph,
        find_paths,
        format_entity_suggestions,
        suggest_entities,
    )

    pipeline = _get_pipeline(ctx)
    g = build_concept_graph(pipeline.triples, knowledge=pipeline.knowledge)
    paths = find_paths(g, start, end)
    if not paths:
        click.echo(f"No path found from '{start}' to '{end}'.")
        if start not in g:
            start_suggestions = format_entity_suggestions(suggest_entities(g, start))
            if start_suggestions:
                click.echo(f"Start {start_suggestions}")
        if end not in g:
            end_suggestions = format_entity_suggestions(suggest_entities(g, end))
            if end_suggestions:
                click.echo(f"End {end_suggestions}")
        return

    click.echo(f"Found {len(paths)} path(s) from '{start}' to '{end}':\n")
    for i, path in enumerate(paths[:5], 1):
        chain = " → ".join(
            f"{node} —[{pred}]→ {target}"
            for node, pred, target in path
        )
        click.echo(f"  {i}. {chain}")


@graph.command("project")
@click.argument("project")
@click.option("--min-score", default=0.4, help="Minimum concept score")
@click.option("--limit", default=30, help="Max concepts to show")
@click.pass_context
def graph_project(ctx, project, min_score, limit):
    """Show concepts most relevant to a project."""
    from khonliang_researcher import build_project_scores

    pipeline = _get_pipeline(ctx)
    scores = build_project_scores(pipeline.knowledge, pipeline.triples)

    ranked = []
    for concept, proj_scores in scores.items():
        score = proj_scores.get(project, 0)
        if score >= min_score:
            ranked.append((concept, score, proj_scores))

    if not ranked:
        click.echo(f"No concepts found for '{project}' above {min_score:.0%}.")
        return

    ranked.sort(key=lambda x: -x[1])
    ranked = ranked[:limit]

    click.echo(f"Concepts for {project} ({len(ranked)} above {min_score:.0%}):\n")
    for concept, score, all_scores in ranked:
        other = ", ".join(
            f"{p}:{s:.0%}" for p, s in sorted(all_scores.items(), key=lambda x: -x[1])
            if p != project
        )
        click.echo(f"  {concept} ({score:.0%})")
        if other:
            click.echo(f"    Also: {other}")


@graph.command("taxonomy")
@click.option("--audience", default="", help="Filter by audience, e.g. developer-researcher")
@click.option("--universal", default="", help="Comma-separated universal parent concepts")
@click.option("--limit", default=50, help="Maximum groups to print")
@click.pass_context
def graph_taxonomy(ctx, audience, universal, limit):
    """Show audience-scoped concept taxonomy groups."""
    from khonliang_researcher import build_concept_graph, build_concept_taxonomy
    from researcher.util import split_csv

    pipeline = _get_pipeline(ctx)
    graph_data = build_concept_graph(pipeline.triples, knowledge=pipeline.knowledge)
    taxonomy = build_concept_taxonomy(
        graph_data,
        universal_concepts=split_csv(universal),
    )
    click.echo(_format_taxonomy(taxonomy, audience=audience, limit=limit))


def _format_taxonomy(taxonomy, *, audience="", limit=50):
    groups = taxonomy.get("groups", [])
    relationships = taxonomy.get("relationships", [])
    audience = str(audience or "").strip()

    if audience:
        selected_codes = {g["code"] for g in groups if g.get("audience") == audience}
        parent_codes = {
            rel["target"]
            for rel in relationships
            if rel.get("source") in selected_codes and rel.get("predicate") == "specializes"
        }
        selected_codes |= parent_codes
        groups = [g for g in groups if g["code"] in selected_codes]
        relationships = [
            rel for rel in relationships
            if rel.get("source") in selected_codes and rel.get("target") in selected_codes
        ]

    if not groups:
        suffix = f" for audience '{audience}'" if audience else ""
        return f"No taxonomy groups{suffix}. Distill some papers first."

    total_groups = len(groups)
    groups = sorted(groups, key=lambda g: (g.get("audience", ""), g.get("code", "")))
    groups = groups[:max(1, int(limit))]
    displayed_codes = {group["code"] for group in groups}
    relationships = [
        rel for rel in relationships
        if rel.get("source") in displayed_codes and rel.get("target") in displayed_codes
    ]

    lines = [f"Concept taxonomy ({total_groups} groups, showing {len(groups)})"]
    current_audience = ""
    for group in groups:
        group_audience = group.get("audience", "general")
        if group_audience != current_audience:
            current_audience = group_audience
            lines.append(f"\n{group_audience}:")
        entities = ", ".join(group.get("entities", [])[:5])
        if len(group.get("entities", [])) > 5:
            entities += f", +{len(group['entities']) - 5} more"
        lines.append(f"  {group['code']} {group['label']} ({len(group.get('entities', []))})")
        if entities:
            lines.append(f"    {entities}")

    if relationships:
        lines.append("\nRelationships:")
        for rel in sorted(relationships, key=lambda r: (r.get("source", ""), r.get("target", "")))[:50]:
            lines.append(f"  {rel['source']} -[{rel['predicate']}]-> {rel['target']}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Ideas
# ------------------------------------------------------------------

@cli.group()
@click.pass_context
def idea(ctx):
    """Ingest and research informal ideas (LinkedIn posts, tweets, etc.)."""
    pass


@idea.command("ingest")
@click.argument("text")
@click.option("--source", default="", help="Source label (e.g. 'linkedin', 'twitter')")
@click.pass_context
def idea_ingest(ctx, text, source):
    """Parse informal text into researchable claims."""
    pipeline = _get_pipeline(ctx)

    async def _ingest():
        idea_id = await pipeline.ingest_idea(text, source)
        entry = pipeline.knowledge.get(idea_id)
        click.echo(f"Idea: {entry.title}")
        click.echo(f"ID: {idea_id}")
        claims = entry.metadata.get("claims", [])
        queries = entry.metadata.get("search_queries", [])
        click.echo(f"\nClaims ({len(claims)}):")
        for c in claims:
            click.echo(f"  - {c}")
        click.echo(f"\nSearch queries ({len(queries)}):")
        for q in queries:
            click.echo(f"  - {q}")

    _run(_ingest())


@idea.command("research")
@click.argument("idea_id")
@click.option("--max-papers", default=10, help="Max papers to fetch")
@click.option("--no-distill", is_flag=True, help="Skip auto-distillation")
@click.pass_context
def idea_research(ctx, idea_id, max_papers, no_distill):
    """Search for papers backing an idea's claims."""
    pipeline = _get_pipeline(ctx)

    async def _research():
        stats = await pipeline.research_idea(idea_id, max_papers, auto_distill=not no_distill)
        if "error" in stats:
            click.echo(f"Error: {stats['error']}", err=True)
            return
        click.echo(f"Queries run: {stats['queries_run']}")
        click.echo(f"Papers found: {stats['papers_found']}")
        click.echo(f"New papers: {stats['papers_new']}")
        click.echo(f"Distilled: {stats['papers_distilled']}")

    _run(_research())


@idea.command("brief")
@click.argument("idea_id")
@click.pass_context
def idea_brief(ctx, idea_id):
    """Synthesize a brief evaluating claims against literature."""
    pipeline = _get_pipeline(ctx)

    async def _brief():
        content = await pipeline.brief_idea(idea_id)
        click.echo(content)

    _run(_brief())


@idea.command("full")
@click.argument("text")
@click.option("--source", default="", help="Source label")
@click.option("--max-papers", default=10, help="Max papers to fetch")
@click.pass_context
def idea_full(ctx, text, source, max_papers):
    """All-in-one: ingest → research → brief."""
    pipeline = _get_pipeline(ctx)

    async def _full():
        click.echo("Parsing idea...")
        idea_id = await pipeline.ingest_idea(text, source)
        entry = pipeline.knowledge.get(idea_id)
        click.echo(f"Idea: {entry.title} ({idea_id})")

        claims = entry.metadata.get("claims", [])
        click.echo(f"Claims: {len(claims)}")
        for c in claims:
            click.echo(f"  - {c}")

        click.echo("\nSearching for papers...")
        stats = await pipeline.research_idea(idea_id, max_papers)
        click.echo(f"Found {stats['papers_new']} new papers, distilled {stats['papers_distilled']}")

        if stats["papers_new"] > 0:
            click.echo("\nGenerating brief...")
            content = await pipeline.brief_idea(idea_id)
            click.echo(f"\n{content}")
        else:
            click.echo("\nNo new papers found — skipping brief.")

    _run(_full())


# ------------------------------------------------------------------
# Codebase scanning and GitHub ingestion
# ------------------------------------------------------------------

@cli.command("scan")
@click.argument("project")
@click.pass_context
def scan_codebase(ctx, project):
    """Scan a project's codebase to discover capabilities."""
    pipeline = _get_pipeline(ctx)

    async def _scan():
        click.echo(f"Scanning {project}...")
        result = await pipeline.scan_codebase(project)
        if "error" in result:
            click.echo(f"Error: {result['error']}", err=True)
            if result.get("raw"):
                click.echo(result["raw"])
            return

        caps = result.get("capabilities", [])
        imports = result.get("imports_from", {})
        stored = result.get("stored", 0)

        click.echo(f"\n{project}: {len(caps)} capabilities found, {stored} new stored\n")
        for c in caps:
            click.echo(f"  {c}")
        if imports:
            click.echo()
            for dep, usages in imports.items():
                click.echo(f"  Imports from {dep}: {', '.join(usages)}")

    _run(_scan())


@cli.command("ingest-github")
@click.argument("repo_url")
@click.option("--label", default="", help="Custom label for the repo")
@click.option("--depth", default="readme+code", type=click.Choice(["readme", "readme+code", "full"]))
@click.pass_context
def ingest_github(ctx, repo_url, label, depth):
    """Cleanroom ingest a GitHub repo — extract concepts, delete clone."""
    pipeline = _get_pipeline(ctx)

    async def _ingest():
        click.echo(f"Ingesting {repo_url} (depth={depth})...")
        result = await pipeline.ingest_github_repo(repo_url, label=label, depth=depth)
        if "error" in result:
            click.echo(f"Error: {result['error']}", err=True)
            return

        click.echo(f"\n{result['repo']}: {len(result['capabilities'])} capabilities\n")
        for cap in result.get("code_capabilities", []):
            click.echo(f"  [code] {cap}")
        for claim in result.get("readme_only_claims", []):
            click.echo(f"  [readme] {claim}")
        if result.get("imports_from"):
            click.echo()
            for dep, items in result["imports_from"].items():
                click.echo(f"  Imports {dep}: {', '.join(items[:5])}")
        if result.get("relevance_scores"):
            click.echo()
            for proj, score in sorted(result["relevance_scores"].items(), key=lambda x: -x[1]):
                click.echo(f"  Relevance({proj}): {score:.2f}")

    _run(_ingest())


@cli.command("research-caps")
@click.argument("project", required=False)
@click.pass_context
def research_caps(ctx, project):
    """Generate academic search queries from scanned capabilities."""
    pipeline = _get_pipeline(ctx)

    async def _research():
        click.echo("Generating research queries from capabilities...")
        result = await pipeline.research_from_capabilities(project)

        queries = result.get("queries", [])
        if not queries:
            click.echo("No queries generated. Scan a codebase first.")
            return

        click.echo(f"\n{len(queries)} queries across {result.get('project_count', 0)} project(s):\n")
        for q in queries:
            click.echo(f"  Query: {q['query']}")
            click.echo(f"  Papers: {len(q.get('papers', []))}")
            for p in q.get("papers", [])[:5]:
                click.echo(f"    - {p.get('title', '?')}")
            click.echo()

    _run(_research())


# ------------------------------------------------------------------
# Project management
# ------------------------------------------------------------------

@cli.group()
@click.pass_context
def project(ctx):
    """Project landscape and capabilities."""
    pass


@project.command("landscape")
@click.argument("name")
@click.pass_context
def project_landscape(ctx, name):
    """Show research landscape for a project."""
    from khonliang.knowledge.store import Tier, EntryStatus
    from khonliang_researcher import build_project_scores

    pipeline = _get_pipeline(ctx)
    projects = pipeline.config.get("projects", {})
    if name not in projects:
        click.echo(f"Unknown project. Available: {', '.join(projects.keys())}", err=True)
        return

    scores = build_project_scores(pipeline.knowledge, pipeline.triples)
    proj_concepts = {
        c: s[name] for c, s in scores.items()
        if name in s and s[name] >= 0.4
    }
    top_concepts = sorted(proj_concepts.items(), key=lambda x: -x[1])

    distilled = len(pipeline.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED))
    total = len(list(pipeline.knowledge.get_by_tier(Tier.IMPORTED)))

    frs = pipeline.get_historical_feature_requests(target=name)

    caps_exist = []
    caps_planned = []
    for entry in pipeline.knowledge.get_by_tier(Tier.DERIVED):
        tags = entry.tags or []
        if "capability" not in tags or f"cap:{name}" not in tags:
            continue
        status = (entry.metadata or {}).get("capability_status", "")
        if status == "exists":
            caps_exist.append(entry.title)
        elif status == "planned":
            caps_planned.append(entry.title)

    click.echo(f"{name} landscape")
    click.echo(f"Papers: {distilled} distilled / {total} total")

    if top_concepts:
        click.echo(f"\nConcepts ({len(proj_concepts)}):")
        for c, s in top_concepts[:15]:
            click.echo(f"  {c} ({s:.0%})")

    if caps_exist:
        click.echo(f"\nExists ({len(caps_exist)}):")
        for c in caps_exist:
            click.echo(f"  {c}")

    if caps_planned:
        click.echo(f"\nPlanned ({len(caps_planned)}):")
        for c in caps_planned:
            click.echo(f"  {c}")

    if frs:
        click.echo(f"\nHistorical researcher FRs ({len(frs)}):")
        for f in frs:
            click.echo(f"  [{f.get('priority', '?')}] {f.get('title', '?')}")


@project.command("capabilities")
@click.option("--target", "-t", default="", help="Filter by project")
@click.pass_context
def project_capabilities(ctx, target):
    """Show what exists vs what's planned for each project."""
    from khonliang.knowledge.store import Tier
    from collections import defaultdict

    pipeline = _get_pipeline(ctx)
    caps = []
    for entry in pipeline.knowledge.get_by_tier(Tier.DERIVED):
        tags = entry.tags or []
        if "capability" not in tags:
            continue
        if target and f"cap:{target}" not in tags:
            continue
        caps.append(entry)

    if not caps:
        click.echo("No capabilities tracked yet. Complete some FRs first.")
        return

    grouped = defaultdict(lambda: defaultdict(list))
    for c in caps:
        t = (c.metadata or {}).get("target", "unknown")
        s = (c.metadata or {}).get("capability_status", "unknown")
        grouped[t][s].append(c.title)

    for proj in sorted(grouped.keys()):
        click.echo(f"\n{proj}:")
        for status in ["exists", "planned", "exploring"]:
            items = grouped[proj].get(status, [])
            if items:
                click.echo(f"  {status.upper()} ({len(items)}):")
                for item in items:
                    click.echo(f"    {item}")


# ------------------------------------------------------------------
# Repo registry
# ------------------------------------------------------------------

@cli.group()
@click.pass_context
def repo(ctx):
    """Manage project repo registry."""
    pass


@repo.command("register")
@click.argument("project_name")
@click.argument("repo_path")
@click.option("--description", "-d", default="", help="Project description")
@click.option("--depends-on", default="", help="Comma-separated dependency projects")
@click.option("--scope", default="", help="Project scope")
@click.pass_context
def repo_register(ctx, project_name, repo_path, description, depends_on, scope):
    """Register a project repo for scanning."""
    from khonliang.knowledge.store import KnowledgeEntry, Tier, EntryStatus

    pipeline = _get_pipeline(ctx)
    entry_id = f"repo_{project_name}"
    deps = [d.strip() for d in depends_on.split(",") if d.strip()]

    entry = KnowledgeEntry(
        id=entry_id,
        tier=Tier.DERIVED,
        title=f"Repo: {project_name}",
        content=description or pipeline.config.get("projects", {}).get(project_name, {}).get("description", ""),
        source="registry",
        scope="registry",
        tags=["repo", f"repo:{project_name}"],
        status=EntryStatus.DISTILLED,
        metadata={
            "project": project_name,
            "repo_path": repo_path,
            "scope": scope or pipeline.config.get("projects", {}).get(project_name, {}).get("scope", ""),
            "depends_on": deps or pipeline.config.get("projects", {}).get(project_name, {}).get("depends_on", []),
        },
    )
    pipeline.knowledge.add(entry)
    click.echo(f"Registered {project_name} at {repo_path}")


@repo.command("list")
@click.pass_context
def repo_list(ctx):
    """List all registered project repos."""
    from khonliang.knowledge.store import Tier

    pipeline = _get_pipeline(ctx)
    repos = []
    for entry in pipeline.knowledge.get_by_tier(Tier.DERIVED):
        if "repo" not in (entry.tags or []):
            continue
        meta = entry.metadata or {}
        deps = ", ".join(meta.get("depends_on", [])) or "none"
        repos.append((meta.get("project", "?"), meta.get("repo_path", "?"), meta.get("scope", "?"), deps))

    if not repos:
        # Fall back to config
        for name, cfg in pipeline.config.get("projects", {}).items():
            repo_path = cfg.get("repo", "not set")
            deps = ", ".join(cfg.get("depends_on", [])) or "none"
            repos.append((name, repo_path, cfg.get("scope", "?"), deps))

    if not repos:
        click.echo("No repos registered.")
        return

    for name, path, scope, deps in repos:
        click.echo(f"  {name:15s} {path}")
        click.echo(f"                  scope: {scope}  deps: {deps}")


# ------------------------------------------------------------------
# Health and status
# ------------------------------------------------------------------

@cli.command("health")
@click.pass_context
def health_check(ctx):
    """Check Ollama, models, DB, and system health."""
    import subprocess
    import shutil
    from pathlib import Path
    from khonliang.knowledge.store import Tier, EntryStatus

    pipeline = _get_pipeline(ctx)

    # Ollama version
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip().replace("ollama version is ", "")
        click.echo(f"Ollama: {version}")
    except Exception as e:
        click.echo(f"Ollama: ERROR ({e})")

    # Required models
    models_config = pipeline.config.get("models", {})
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        installed = {line.split()[0].split(":")[0] for line in result.stdout.strip().split("\n")[1:] if line.strip()}
        for role, model in models_config.items():
            model_base = model.split(":")[0]
            status = "ok" if model_base in installed else "MISSING"
            click.echo(f"  {role}: {model} [{status}]")
    except Exception as e:
        click.echo(f"  Models: ERROR ({e})")

    # DB size
    db_path = pipeline.config.get("db_path", "data/researcher.db")
    if Path(db_path).exists():
        size_mb = Path(db_path).stat().st_size / (1024 * 1024)
        click.echo(f"DB: {size_mb:.1f}MB")

    # Disk space
    disk = shutil.disk_usage("/")
    free_gb = disk.free / (1024**3)
    click.echo(f"Disk: {free_gb:.1f}GB free")

    # Paper stats
    distilled = len(pipeline.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED))
    pending = len(pipeline.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED))
    click.echo(f"Papers: {distilled} distilled, {pending} pending")


@cli.command("status")
@click.pass_context
def worker_status(ctx):
    """Show worker queue depth and paper counts."""
    from khonliang.knowledge.store import Tier, EntryStatus

    pipeline = _get_pipeline(ctx)

    pending = len(pipeline.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED))
    processing = len(pipeline.knowledge.get_by_status(EntryStatus.PROCESSING, tier=Tier.IMPORTED))
    distilled = len(pipeline.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED))
    failed = len(pipeline.knowledge.get_by_status(EntryStatus.FAILED, tier=Tier.IMPORTED))
    skipped = len(pipeline.knowledge.get_by_status(EntryStatus.SKIPPED, tier=Tier.IMPORTED))

    click.echo(f"Pending:    {pending}")
    click.echo(f"Processing: {processing}")
    click.echo(f"Distilled:  {distilled}")
    click.echo(f"Failed:     {failed}")
    click.echo(f"Skipped:    {skipped}")


# ------------------------------------------------------------------
# Server
# ------------------------------------------------------------------

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
