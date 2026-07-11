# khonliang-researcher

Automated research pipeline: RSS/search -> fetch -> distill -> concept graph -> concept bundles and evidence briefs. Served over MCP for consumption by project Claudes (khonliang, developer, genealogy).

## Stack

- Python, async throughout
- Local LLMs via Ollama (qwen2.5:7b summarizer, llama3.2:3b extractor, qwen2.5:32b reviewer, nomic-embed-text embedder)
- SQLite-backed stores: KnowledgeStore, TripleStore, DigestStore (from khonliang)
- MCP server extending khonliang's KhonliangMCPServer

## MCP Tool Response Convention

All MCP tool responses must be token-efficient. External agents pay per token — verbose output wastes context and money.

**Rules:**
- No preamble ("Here are the results:", "I found the following:")
- Data only: `id | title | score` not paragraphs
- Default to brief — agent asks for detail when needed
- Every word must earn its place — if removing it doesn't lose information, remove it

## Architecture Boundary

- **khonliang** = library. Agent roles, routing, consensus, stores, MCP transport, generic capabilities.
- **researcher** = business logic. Paper discovery, distillation strategy, concept extraction, relevance scoring, concept bundling, report generation.
- **developer** = active FR lifecycle. Promotion, status, dependencies, milestones, specs, and work units live there.
- New features go in researcher unless they are generic multi-agent primitives. When in doubt, it's researcher business logic.

## Capability Tracking

Projects have `exists` / `planned` / `exploring` capability tags. Researcher
reads these for landscape and capability reports, while developer owns active
FR lifecycle updates.

## Key Files

- `researcher/server.py` — MCP tools (40+ tools)
- `researcher/pipeline.py` — orchestration layer
- `researcher/synthesizer.py` — LLM synthesis (synergize, review, briefs, evaluate_capability)
- `researcher/graph.py` — concept graph, project score propagation
- `researcher/relevance.py` — embedding-based relevance scoring
- `researcher/fetcher.py` — paper/URL fetching with browser headers
- `researcher/worker.py` — batch distillation worker
- `researcher/librarian_client.py` — bus client for calling librarian-primary's skills (see Librarian below)
- `config.yaml` — models, projects, thresholds

## Librarian (bus service, not co-resident)

librarian used to run in-process here (`researcher/librarian_agent.py`); it
has split into its own repo/agent (`khonliang-librarian`, `librarian-primary`,
fr_librarian_bc0a06d7). Researcher now only *consumes* librarian's 7 skills
(classify_paper, taxonomy_report, rebuild_neighborhoods,
suggest_missing_nodes, promote_investigation, identify_gaps, library_health)
over the bus (fr_researcher_11e9524a):

- `researcher/librarian_client.py` — thin wrapper over
  `agent.request(agent_type="librarian", operation=..., args=...)` (same
  pattern as `agent.py`'s `stage_payload`/`ingest_from_artifact` calling
  `agent_type="store"`). Every call is best-effort: failures (timeout,
  connection error, librarian not registered) return
  `{"available": False, "reason": ...}` rather than raising — librarian
  absence degrades quality, not function (optional-coordinator principle).
- `researcher-primary`'s `ask_librarian` bus skill is a thin proxy over this
  client for callers that only address researcher.
- The `ingest.url_distilled` / `ingest.queue_drained` bus events researcher
  publishes (`researcher/ingest_watcher.py`) remain the sanctioned
  fire-and-forget path librarian's own watcher subscribes to — no blocking
  call was added to the ingest hot path for this FR.

## Running

```
.venv/bin/python -m researcher.server --config /path/to/khonliang-researcher/config.yaml
```

Config path must be absolute for cross-session MCP launches. Copy
`config.example.yaml` to `config.yaml` and edit the paths for your
environment — `config.yaml` is git-ignored because it contains
machine-specific paths.
