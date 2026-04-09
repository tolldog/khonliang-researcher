# khonliang-researcher

Automated research pipeline: RSS/search -> fetch -> distill -> concept graph -> synergize -> FR generation. Served over MCP for consumption by project Claudes (khonliang, developer, genealogy).

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
- **researcher** = business logic. Paper discovery, distillation strategy, concept extraction, relevance scoring, FR generation heuristics, report generation.
- New features go in researcher unless they are generic multi-agent primitives. When in doubt, it's researcher business logic.

## Capability Tracking

Projects have `exists` / `planned` / `exploring` capability tags, auto-maintained via FR lifecycle:
- `update_fr_status("completed")` -> records capability as `exists`
- `update_fr_status("planned")` -> records as `planned`
- `synergize()` reads these to avoid proposing features that already exist

## Key Files

- `researcher/server.py` — MCP tools (40+ tools)
- `researcher/pipeline.py` — orchestration layer
- `researcher/synthesizer.py` — LLM synthesis (synergize, review, briefs, evaluate_capability)
- `researcher/graph.py` — concept graph, project score propagation
- `researcher/relevance.py` — embedding-based relevance scoring
- `researcher/fetcher.py` — paper/URL fetching with browser headers
- `researcher/worker.py` — batch distillation worker
- `config.yaml` — models, projects, thresholds

## Running

```
.venv/bin/python -m researcher.server --config /mnt/dev/ttoll/dev/khonliang-researcher/config.yaml
```

Config path must be absolute for cross-session MCP launches.
