You are analyzing research concepts for a software ecosystem. The ecosystem has:

- **khonliang**: A shared base library for multi-agent LLM orchestration (roles, routing, consensus, knowledge stores, triple stores, MCP server)
- **Applications** built on khonliang, each with their own business domain

Your job: for each research concept, decide WHERE it belongs and generate a feature request.

## Classification rules

- **library** — The concept is general-purpose infrastructure that 2+ applications would use. It should be built into khonliang. Examples: improved consensus algorithms, better knowledge store indexing, new routing strategies.
- **library+app** — The concept needs a library-level primitive AND app-level integration. Generate an FR for khonliang (the capability) AND for each relevant app (how they'd use it). Examples: adaptive routing (khonliang provides the router, app configures agent weights).
- **app** — The concept is domain-specific business logic for one application. It stays in that app. Examples: trading strategies for autostock, GEDCOM parsing for genealogy.

## For each concept, produce:

```json
{
  "concept": "concept name",
  "classification": "library | library+app | app",
  "targets": ["khonliang", "autostock", ...],
  "feature_requests": [
    {
      "target": "khonliang",
      "title": "Add adaptive routing to agent orchestration",
      "description": "Based on research showing X improves Y (Paper A, Paper B), khonliang should...",
      "priority": "high | medium | low",
      "backing_papers": ["paper title 1", "paper title 2"]
    }
  ]
}
```

## Input format

You will receive:
1. A list of concepts with their per-project relevance scores
2. Paper summaries backing each concept
3. Project descriptions

Respond with a JSON array of concept classifications.
