# khonliang-researcher

Research ingestion and evidence service for the khonliang workspace.

Researcher discovers papers and other source material, ingests local documents, distills content into summaries and triples, builds concept views, and exposes compact evidence-oriented skills through the bus/MCP surface.

## Current Role

Researcher is the evidence layer:

- fetch papers, PDFs, HTML, Markdown, and informal ideas
- distill sources into summaries, concept triples, and project relevance scores
- build concept trees, paths, matrices, taxonomies, and project applicability briefs
- keep paper and evidence references resolvable

Researcher is not the active feature-request lifecycle owner. Use `khonliang-developer` for FR promotion, dependencies, status, milestones, specs, work units, git/GitHub workflow, and repo hygiene.

## Runtime Shape

The active agent path is:

1. `khonliang-bus` runs the shared bus and MCP adapter.
2. Researcher starts as an agent and registers skills with the bus.
3. Developer and other sessions request researcher skills through the bus.
4. Researcher returns compact responses by default; callers ask for `detail=full` only when needed.

Direct MCP entrypoints remain available for local debugging, but the bus is the preferred integration surface.

## Setup

Create a virtual environment and install the package:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e .
```

Copy and edit the config:

```sh
cp config.example.yaml config.yaml
```

`config.yaml` is local-only. It may contain machine paths, local database locations, and private project mappings. Prefer GitHub repository URLs in project config when the agent should be portable across sessions.

## Running

Start the researcher server directly for local debugging:

```sh
.venv/bin/python -m researcher.server --config /absolute/path/to/config.yaml
```

For normal khonliang usage, start researcher through the bus/developer workflow so its skills are registered and discoverable by other agents.

## Common Workflows

Ingest a local document:

```sh
.venv/bin/python -m researcher.cli --config config.yaml ingest-file /path/to/document.pdf
```

Distill queued sources:

```sh
.venv/bin/python -m researcher.cli --config config.yaml distill --all
```

Ask the running agent for compact evidence:

```text
paper_context(query="multi-agent code review", detail="brief")
concept_path(start="multi-agent review", end="code review")
project_landscape(project="developer", detail="brief")
```

## Repository Boundaries

- `khonliang-researcher`: application logic for ingestion, distillation, concept exploration, and researcher agent skills
- `khonliang-researcher-lib`: reusable research primitives shared by researcher and developer
- `khonliang-developer`: FR lifecycle, work bundles, specs, milestones, repo hygiene, git/GitHub automation
- `khonliang-bus` / `khonliang-bus-lib`: service registry, skill contracts, transport, and agent communication

When logic is useful outside this application, put it in `khonliang-researcher-lib`. When logic manages implementation work, put it in `khonliang-developer`.

## Validation

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall researcher
```

The repo hygiene baseline is tracked in `docs/repo-hygiene-audit.md`.
