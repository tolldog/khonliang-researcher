# Khonliang Feedback

Issues and feature requests discovered while building khonliang-researcher.
Each validated entry becomes a khonliang issue/PR.

## Pending

- **generate_json reliability**: `OllamaClient.generate_json()` fails frequently on math-heavy content. The retry-based parsing isn't enough — needs `constrained=True` (Ollama native JSON mode) as a parameter, and text pre-cleaning as a documented pattern. Discovered: 2026-04-03

- **generate_json constrained param**: The `constrained` parameter exists in the signature but we had to discover it by reading source. Should be documented and defaulted to True for `generate_json`. Discovered: 2026-04-03

- **Model override on generate**: `generate_json(model=model)` works for selecting a different model than the pool default, but only if the model is already pulled. Should auto-pull or give a clear error. Discovered: 2026-04-03

- **KnowledgeStore needs status field**: Entries only have `tags` for status tracking (we use "undistilled"/"distilled" tags). A proper `status` enum field would be cleaner. Discovered: 2026-04-03

- **KnowledgeStore URL-aware dedup**: Current dedup is SHA256 of content. We need dedup by URL/source_id too — same paper fetched twice from different formats should be detected. Discovered: 2026-04-03

- **TripleStore predicate normalization**: We got `applies_to`, `is_applicable_to`, `is applicable to` as separate predicates meaning the same thing. Need a normalizer or alias system. Discovered: 2026-04-03

- **Blackboard not persistent**: When the MCP server restarts, all blackboard state is lost. Optional SQLite persistence would help (proposed as KH-5). Discovered: 2026-04-03

- **DigestSynthesizer prompt override**: The default synthesis prompt is generic. We needed domain-specific prompts for research distillation. Should accept custom prompt templates. Discovered: 2026-04-03

- **BaseRole context budget**: The `enforce_budget()` truncation is crude (chars/4 heuristic). For long papers, a smarter approach would extract abstract + intro + conclusion rather than just truncating from the start. Discovered: 2026-04-03

- **MCP server tool descriptions**: Can't customize tool descriptions after registration. The auto-generated descriptions from function docstrings are sometimes too verbose. Discovered: 2026-04-03

## Resolved

<!-- Move entries here when addressed in a khonliang release -->
