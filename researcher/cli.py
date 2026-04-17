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

from researcher.pipeline import create_pipeline, update_capability_status


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
    """[DEPRECATED] Classify concepts and generate FRs across the ecosystem.

    FR generation is moving from researcher to developer. No CLI equivalent
    ships here; use the `synergize_concepts` MCP tool or
    `ResearchPipeline.synergize_concepts()` Python API for concept bundles.
    FR generation via developer is tracked as fr_developer_4724d49d.
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

        click.echo(f"\n{result['concept_count']} concepts classified, {result['fr_count']} FRs generated\n")
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
# Feature requests
# ------------------------------------------------------------------

@click.pass_context
def fr(ctx):
    """Legacy local FR helpers.

    Not registered as a CLI command. Developer owns active FR lifecycle.
    """
    click.echo("Researcher no longer owns active FR lifecycle. Use developer.")


@click.option("--target", "-t", default="", help="Filter by target project")
@click.option("--reviewed", is_flag=True, help="Only show reviewed FRs")
@click.pass_context
def fr_list(ctx, target, reviewed):
    """List generated feature requests."""
    pipeline = _get_pipeline(ctx)
    frs = pipeline.get_feature_requests(target=target or None)

    if not frs:
        click.echo(f"No FRs found{f' for {target}' if target else ''}. Run synergize first.")
        return

    if reviewed:
        frs = [f for f in frs if f.get("review_verdict")]

    priority_order = {"high": 0, "medium": 1, "low": 2}
    frs.sort(key=lambda x: (priority_order.get(x.get("priority", "medium"), 1), x.get("target", "")))

    current_cls = ""
    for f in frs:
        cls = f.get("classification", "unknown")
        if cls != current_cls:
            current_cls = cls
            click.echo(f"\n{cls.upper()}:")

        entry = pipeline.knowledge.get(f["id"])
        status = (entry.metadata.get("fr_status", "open") if entry else "open")
        status_str = f" ({status})" if status != "open" else ""

        verdict = ""
        if entry and entry.metadata.get("review_verdict"):
            v = entry.metadata["review_verdict"].upper()
            conf = entry.metadata.get("review", {}).get("confidence", 0)
            verdict = f" [{v} {conf:.0%}]"

        click.echo(f"  [{f.get('priority', 'med'):6s}] {f.get('title', '?')} → {f.get('target', '?')}{status_str}{verdict}")
        click.echo(f"          ID: {f['id']}  Concept: {f.get('concept', '?')}")


def fr_workflow():
    """Show the FR workflow protocol for project Claudes."""
    click.echo("""FR Workflow: open → planned → in_progress → completed

Discovery:
  fr list -t PROJECT          List FRs for a project
  fr next -t PROJECT          Get highest priority unblocked FR

Claiming:
  fr update FR_ID planned     Claim an FR
  fr update FR_ID in_progress --branch fr/short_id   Start work

Completion:
  fr update FR_ID completed --notes "PR #N merged"

Management:
  fr deps FR_ID DEP1,DEP2     Set dependencies
  fr overlaps                  Find duplicate FRs
  fr merge KEEP_ID MERGE_IDS  Combine overlapping FRs
  fr promote -t TARGET TITLE  Create a new FR""")


@click.option("--target", "-t", default="", help="Filter by target project")
@click.pass_context
def fr_next(ctx, target):
    """Get the next FR to work on — highest priority, all deps met."""
    from khonliang.knowledge.store import Tier, EntryStatus

    pipeline = _get_pipeline(ctx)
    frs = pipeline.get_feature_requests(target=target or None)
    if not frs:
        click.echo(f"No FRs found{f' for {target}' if target else ''}.")
        return

    all_entries = list(pipeline.knowledge.get_by_tier(Tier.DERIVED))
    completed_ids = set()
    for e in all_entries:
        if "fr:archived" in (e.tags or []) or "fr:completed" in (e.tags or []):
            completed_ids.add(e.id)
        if e.status == EntryStatus.ARCHIVED and "fr" in str(e.tags):
            completed_ids.add(e.id)

    priority_score = {"high": 3, "medium": 2, "low": 1}
    class_score = {"library": 3, "library+app": 2, "app": 1}

    candidates = []
    blocked = []
    for f in frs:
        entry = pipeline.knowledge.get(f["id"])
        fr_status = entry.metadata.get("fr_status", "open") if entry else "open"
        if fr_status in ("planned", "in_progress", "completed"):
            continue

        deps = f.get("depends_on", [])
        if not deps and entry:
            deps = entry.metadata.get("depends_on", [])

        unmet = [d for d in deps if d not in completed_ids]
        if unmet:
            blocked.append((f, unmet))
            continue

        score = (
            priority_score.get(f.get("priority", "medium"), 2) * 10
            + class_score.get(f.get("classification", "app"), 1)
        )
        candidates.append((score, f))

    candidates.sort(key=lambda x: -x[0])

    if not candidates:
        click.echo("All FRs are blocked on dependencies:\n")
        for f, unmet in blocked:
            unmet_titles = []
            for uid in unmet:
                dep = pipeline.knowledge.get(uid)
                unmet_titles.append(dep.title if dep else uid)
            click.echo(f"  {f.get('title', '?')} → {f.get('target', '?')}")
            click.echo(f"    Blocked by: {', '.join(unmet_titles)}")
        return

    _, top = candidates[0]
    entry = pipeline.knowledge.get(top["id"])

    click.echo(f"Next FR: {top.get('title', '?')}")
    click.echo(f"ID: {top['id']}")
    click.echo(f"Target: {top.get('target', '?')}")
    click.echo(f"Priority: {top.get('priority', '?')}")
    click.echo(f"Classification: {top.get('classification', '?')}")
    click.echo(f"Concept: {top.get('concept', '?')}")

    deps = top.get("depends_on", [])
    if not deps and entry:
        deps = entry.metadata.get("depends_on", [])
    if deps:
        click.echo(f"\nDependencies (all met):")
        for d in deps:
            dep = pipeline.knowledge.get(d)
            click.echo(f"  {d}: {dep.title if dep else '?'}")

    desc = top.get("description", entry.content if entry else "")
    if desc:
        click.echo(f"\n{desc}")

    papers = top.get("backing_papers", [])
    if papers:
        click.echo(f"\nBacking papers:")
        for p in papers:
            click.echo(f"  - {p}")

    if len(candidates) > 1:
        click.echo(f"\n{len(candidates) - 1} more FR(s) ready.")
    if blocked:
        click.echo(f"{len(blocked)} FR(s) blocked.")


@click.argument("fr_id")
@click.argument("status", type=click.Choice(["open", "planned", "in_progress", "completed"]))
@click.option("--branch", default="", help="Git branch name (for in_progress)")
@click.option("--notes", default="", help="Status notes")
@click.pass_context
def fr_update(ctx, fr_id, status, branch, notes):
    """Update a feature request's lifecycle status."""
    import time
    from khonliang.knowledge.store import Tier, EntryStatus

    pipeline = _get_pipeline(ctx)
    entry = pipeline.knowledge.get(fr_id)
    if not entry:
        click.echo(f"FR {fr_id} not found.", err=True)
        sys.exit(1)

    prev_status = entry.metadata.get("fr_status", "open")
    entry.metadata["fr_status"] = status
    if branch:
        entry.metadata["branch"] = branch
    if notes:
        history = entry.metadata.get("status_history", [])
        history.append({"status": status, "notes": notes, "at": time.strftime("%Y-%m-%d %H:%M")})
        entry.metadata["status_history"] = history

    target = entry.metadata.get("target", "")
    concept = entry.metadata.get("concept", "")

    if status == "completed":
        entry.status = EntryStatus.ARCHIVED
        entry.tags = [t for t in (entry.tags or []) if t != "fr"] + ["fr:completed"]

        if target:
            update_capability_status(pipeline.knowledge, target, entry.title, concept, "exists", fr_id)
            click.echo(f"Completed: {entry.title}")
            # Check for unblocked FRs
            unblocked = []
            for e in pipeline.knowledge.get_by_tier(Tier.DERIVED):
                if "fr" not in (e.tags or []):
                    continue
                deps = e.metadata.get("depends_on", [])
                if fr_id in deps:
                    unblocked.append(e.title)
            if unblocked:
                click.echo(f"\nUnblocked {len(unblocked)} FR(s):")
                for title in unblocked:
                    click.echo(f"  - {title}")
    else:
        if target and status in ("planned", "in_progress"):
            update_capability_status(pipeline.knowledge, target, entry.title, concept, "planned", fr_id)
        click.echo(f"{entry.title}: {prev_status} → {status}")
        if branch:
            click.echo(f"Branch: {branch}")

    pipeline.knowledge.add(entry)

    pipeline.digest.record(
        summary=f"FR {prev_status} → {status}: {entry.title}",
        source="pipeline",
        audience="research",
        tags=["fr", status],
        metadata={"fr_id": fr_id, "branch": branch},
    )


@click.argument("title")
@click.option("--target", "-t", required=True, help="Target project")
@click.option("--description", "-d", default="", help="Full FR description")
@click.option("--priority", "-p", default="medium", type=click.Choice(["high", "medium", "low"]))
@click.option("--concept", default="", help="Research concept this derives from")
@click.option("--classification", default="app", type=click.Choice(["library", "library+app", "app"]))
@click.option("--papers", default="", help="Comma-separated backing paper titles")
@click.pass_context
def fr_promote(ctx, title, target, description, priority, concept, classification, papers):
    """Promote a vetted feature request into the FR store."""
    import hashlib
    from khonliang.knowledge.store import KnowledgeEntry, Tier, EntryStatus

    pipeline = _get_pipeline(ctx)

    fr_id = f"fr_{target}_{hashlib.sha256(title.encode()).hexdigest()[:8]}"
    paper_list = [p.strip() for p in papers.split(",") if p.strip()] if papers else []

    fr_data = {
        "target": target,
        "title": title,
        "description": description,
        "priority": priority,
        "backing_papers": paper_list,
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
            "backing_papers": paper_list,
            "promoted": True,
        },
    )
    pipeline.knowledge.add(entry)
    click.echo(f"FR promoted: {title}")
    click.echo(f"ID: {fr_id}")
    click.echo(f"Target: {target}  Priority: {priority}")


@click.argument("fr_id")
@click.argument("depends_on")
@click.pass_context
def fr_deps(ctx, fr_id, depends_on):
    """Set dependencies between FRs. DEPENDS_ON is comma-separated FR IDs."""
    pipeline = _get_pipeline(ctx)
    entry = pipeline.knowledge.get(fr_id)
    if not entry:
        click.echo(f"FR {fr_id} not found.", err=True)
        sys.exit(1)

    dep_ids = [d.strip() for d in depends_on.split(",") if d.strip()]
    for dep_id in dep_ids:
        dep = pipeline.knowledge.get(dep_id)
        if not dep:
            click.echo(f"Dependency {dep_id} not found.", err=True)
            sys.exit(1)

    entry.metadata["depends_on"] = dep_ids
    pipeline.knowledge.add(entry)

    click.echo(f"Dependencies set for {entry.title}:")
    for dep_id in dep_ids:
        dep = pipeline.knowledge.get(dep_id)
        click.echo(f"  {dep_id}: {dep.title if dep else '?'}")


@click.option("--target", "-t", default="", help="Filter by target project")
@click.option("--threshold", default=0.75, help="Similarity threshold")
@click.pass_context
def fr_overlaps(ctx, target, threshold):
    """Find overlapping FRs that may need merging."""
    pipeline = _get_pipeline(ctx)

    async def _overlaps():
        frs = pipeline.get_feature_requests(target=target or None)
        if len(frs) < 2:
            click.echo("Need at least 2 FRs to check for overlaps.")
            return

        from khonliang_researcher import cosine_similarity

        embeddings = {}
        for f in frs:
            text = f"{f.get('title', '')}\n{f.get('description', '')[:500]}"
            emb = await pipeline.relevance._embed(text)
            if emb:
                embeddings[f["id"]] = (f, emb)

        if len(embeddings) < 2:
            click.echo("Could not embed enough FRs to compare.")
            return

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
            click.echo(f"No overlapping FRs above {threshold:.0%} similarity.")
            return

        click.echo(f"FR Overlaps ({len(pairs)} pairs above {threshold:.0%}):\n")
        for sim, fr_a, fr_b in pairs:
            click.echo(f"  {sim:.0%} similar:")
            click.echo(f"    A: [{fr_a['id']}] {fr_a.get('title', '?')} → {fr_a.get('target', '?')}")
            click.echo(f"    B: [{fr_b['id']}] {fr_b.get('title', '?')} → {fr_b.get('target', '?')}")
            if fr_a.get('target') != fr_b.get('target'):
                click.echo(f"    Different targets — may be intentional")
            click.echo()

    _run(_overlaps())


@click.argument("keep_id")
@click.argument("merge_ids")
@click.option("--title", default="", help="New merged title")
@click.option("--description", default="", help="New merged description")
@click.pass_context
def fr_merge(ctx, keep_id, merge_ids, title, description):
    """Merge overlapping FRs. MERGE_IDS is comma-separated."""
    from khonliang.knowledge.store import EntryStatus

    pipeline = _get_pipeline(ctx)
    keep_entry = pipeline.knowledge.get(keep_id)
    if not keep_entry:
        click.echo(f"FR {keep_id} not found.", err=True)
        sys.exit(1)

    ids_to_merge = [mid.strip() for mid in merge_ids.split(",") if mid.strip()]
    merged_entries = []
    for mid in ids_to_merge:
        entry = pipeline.knowledge.get(mid)
        if not entry:
            click.echo(f"FR {mid} not found.", err=True)
            sys.exit(1)
        merged_entries.append(entry)

    try:
        keep_data = json.loads(keep_entry.content)
    except json.JSONDecodeError:
        keep_data = {}

    all_papers = set(keep_entry.metadata.get("backing_papers", []))
    merged_concepts = [keep_entry.metadata.get("concept", "")]
    for entry in merged_entries:
        all_papers.update(entry.metadata.get("backing_papers", []))
        concept = entry.metadata.get("concept", "")
        if concept and concept not in merged_concepts:
            merged_concepts.append(concept)

    if title:
        keep_entry.title = title
        keep_data["title"] = title
    if description:
        keep_data["description"] = description
    keep_data["backing_papers"] = list(all_papers)
    keep_data["merged_from"] = ids_to_merge
    keep_entry.content = json.dumps(keep_data, indent=2)
    keep_entry.metadata["backing_papers"] = list(all_papers)
    keep_entry.metadata["merged_from"] = ids_to_merge
    keep_entry.metadata["merged_concepts"] = merged_concepts
    pipeline.knowledge.add(keep_entry)

    for entry in merged_entries:
        entry.status = EntryStatus.ARCHIVED
        entry.tags = [t for t in (entry.tags or []) if t != "fr"] + ["fr:archived", f"merged_into:{keep_id}"]
        pipeline.knowledge.add(entry)

    click.echo(f"Merged {len(ids_to_merge)} FR(s) into {keep_id}")
    click.echo(f"Title: {keep_entry.title}")
    click.echo(f"Backing papers: {len(all_papers)}")
    click.echo(f"Concepts: {', '.join(merged_concepts)}")
    click.echo(f"Archived: {', '.join(ids_to_merge)}")


@click.option("--target", "-t", default="", help="Filter by target project")
@click.pass_context
def fr_review(ctx, target):
    """Deep-review FRs using the largest model."""
    pipeline = _get_pipeline(ctx)

    async def _review():
        frs = pipeline.get_feature_requests(target=target or None)
        if not frs:
            click.echo("No FRs to review. Run synergize first.")
            return

        click.echo(f"Reviewing {len(frs)} FRs with {pipeline.config.get('models', {}).get('reviewer', 'qwen2.5:32b')}...\n")
        results = await pipeline.review_frs(target=target or None)

        for r in results:
            verdict = r.get("verdict", "?").upper()
            confidence = r.get("confidence", 0)
            click.echo(f"  [{verdict} {confidence:.0%}] {r.get('title', '?')} → {r.get('target', '?')}")
            click.echo(f"    {r.get('reasoning', '')}")
            if r.get("revised_title"):
                click.echo(f"    Revised: {r['revised_title']}")
            for c in r.get("concerns", []):
                click.echo(f"    - {c}")
            click.echo()

        accepted = sum(1 for r in results if r.get("verdict") == "accept")
        revised = sum(1 for r in results if r.get("verdict") == "revise")
        rejected = sum(1 for r in results if r.get("verdict") == "reject")
        click.echo(f"Summary: {accepted} accepted, {revised} revised, {rejected} rejected")

    _run(_review())


# Keep old "frs" and "review" commands as aliases for backwards compat
@cli.command("frs", hidden=True)
@click.option("--target", "-t", default="")
@click.option("--reviewed", is_flag=True)
@click.pass_context
def frs_compat(ctx, target, reviewed):
    """Alias for 'fr list'."""
    ctx.invoke(fr_list, target=target, reviewed=reviewed)


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

    frs = pipeline.get_feature_requests(target=name)

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
        click.echo(f"\nOpen FRs ({len(frs)}):")
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
