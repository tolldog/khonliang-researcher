"""Synthesize combined summaries across multiple distilled papers.

Uses khonliang's OllamaClient to generate cross-paper analysis:
  - Topic summaries: "What do these papers say about consensus?"
  - Project briefs: "What's applicable to the developer project?"
  - Landscape overviews: "What are the main themes and gaps?"
  - Idea briefs: "Do these papers support or contradict this idea?"
  - Relationship maps: "How do these papers connect?"
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from khonliang.knowledge.store import KnowledgeStore, Tier
from khonliang.knowledge.triples import TripleStore
from khonliang.pool import ModelPool
from khonliang_researcher import select_best_of_n, serialize_candidates

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM = """\
You are a research synthesis assistant. Given summaries of multiple papers,
produce a combined analysis. Be specific — cite paper titles and findings.
Do not invent information not present in the summaries.
"""

_TOPIC_PROMPT = """\
Synthesize these {count} paper summaries into a coherent overview of the topic: "{topic}"

For each major finding or method, cite which paper(s) it comes from.
Structure your response as:

## Key Themes
- Theme 1: description (Paper A, Paper B)

## Methods & Approaches
- Approach 1: description (Paper A)

## Open Questions & Gaps
- Gap 1: description

## Connections Between Papers
- Paper A extends Paper B's work by...

Paper summaries:
{summaries}
"""

_PROJECT_PROMPT = """\
You are analyzing {count} research papers for applicability to a specific project.

PROJECT: {project_name}
DESCRIPTION: {project_description}

For EACH applicable paper, explain SPECIFICALLY how its findings could be implemented in this project. Be concrete — name specific components, methods, or code changes.

Structure EXACTLY as:

## Implement Now
For each: Paper title → what to build/change → expected benefit

## Worth Exploring
For each: Paper title → idea → what needs validation first

## Background Only
Papers that inform thinking but don't have direct implementation paths

Keep each entry to 2-3 sentences max. Do not repeat papers.

PAPERS:
{summaries}
"""

_IDEA_PROMPT = """\
Someone shared the following idea or claim:

--- ORIGINAL TEXT ---
{idea_text}
--- END ---

Claims extracted from this text:
{claims}

Below are summaries of {count} research papers found by searching for literature related to these claims.

For EACH claim, evaluate:
1. **Supported** — which papers provide evidence for it, and how strong is the evidence?
2. **Contradicted** — which papers present conflicting findings?
3. **Unaddressed** — claims with no relevant literature found

Structure your response as:

## Claim Analysis
For each claim:
### "[claim text]"
**Verdict**: Supported / Contradicted / Partially Supported / Unaddressed
**Evidence**: cite specific papers and findings
**Nuance**: important caveats or conditions

## Overall Assessment
- How well-grounded is the original idea?
- What's missing from the literature?
- Suggested next steps for deeper investigation

Papers:
{summaries}
"""

_SYNERGIZE_CONCEPTS_PROMPT = """\
You are analyzing research concepts from a paper corpus. Your job is to find
what goes together conceptually — group related concepts into bundles based
on shared themes, methods, or complementary findings.

CONCEPTS WITH SCORES:
{concepts}

BACKING PAPER SUMMARIES:
{summaries}

Group these concepts into bundles of related ideas. For each bundle:

1. **Name**: A short label for the bundle (2-5 words)
2. **Concepts**: Which concepts belong in this bundle
3. **Connection**: WHY these concepts go together (shared method, complementary findings, same problem from different angles)
4. **Strength**: How strong is the evidence for this grouping (number of independent papers, consistency of findings)
5. **Summary**: One sentence describing what this bundle represents
6. **Papers**: Key papers backing this bundle

Do NOT generate feature requests, priorities, targets, or classifications.
Just group what goes together and explain why.

Respond with a JSON array:
[
  {{
    "name": "short bundle name",
    "concepts": ["concept1", "concept2"],
    "connection": "why these go together",
    "strength": 0.0-1.0,
    "summary": "one line describing the bundle",
    "papers": ["paper title 1", "paper title 2"]
  }}
]

Only output the JSON array. No other text.
"""

_SYNERGIZE_PROMPT = """\
You are analyzing research concepts for a software ecosystem with a shared library and multiple applications.

SHARED LIBRARY: khonliang
{library_description}

APPLICATIONS:
{app_descriptions}

DEPENDENCY GRAPH:
{dependency_graph}

ALREADY BUILT (do NOT propose these — they exist):
{existing_capabilities}

PLANNED (in progress or queued — do not duplicate):
{planned_capabilities}

CONCEPTS WITH SCORES:
{concepts}

BACKING PAPER SUMMARIES:
{summaries}

For each concept, classify WHERE it belongs and generate feature requests:

- **library**: General infrastructure useful to 2+ apps. Build into khonliang.
- **library+app**: Needs a library primitive AND app-level wiring. FR for both.
- **app**: Domain-specific to one application. Stays in that app only.

CRITICAL SCOPING RULES for feature requests:
- An FR for a project may ONLY reference that project and its dependencies.
- All apps depend on khonliang, so app FRs may reference khonliang.
- App FRs must NEVER reference other apps (no cross-app mentions).
- khonliang FRs describe the capability generically, not how specific apps use it.
- If two apps would benefit from the same concept independently, note it as a "synergy" but write separate, self-contained FRs for each.
- Do NOT propose features that already exist in the project descriptions above. Only propose genuinely NEW capabilities not already built.
- Focus on the most impactful and novel concepts. Quality over quantity — 3-5 well-scoped FRs are better than 15 vague ones.

Respond with a JSON array. Each element:
{{
  "concept": "name",
  "classification": "library | library+app | app",
  "targets": ["khonliang", "appname", ...],
  "synergies": ["optional: note if 2+ apps benefit independently from this concept"],
  "feature_requests": [
    {{
      "target": "khonliang or appname",
      "title": "short FR title",
      "description": "what to build and why, citing papers. ONLY reference this target and its dependencies.",
      "priority": "high | medium | low",
      "backing_papers": ["paper title"]
    }}
  ]
}}

Only output the JSON array. No other text.
"""

_REVIEW_PROMPT = """\
You are a senior software architect reviewing a feature request for a project.

PROJECT: {project_name}
DESCRIPTION: {project_description}
DEPENDENCIES: {dependencies}

FEATURE REQUEST:
Title: {fr_title}
Description: {fr_description}
Classification: {classification}
Concept: {concept}
Backing papers: {backing_papers}

RELATED RESEARCH:
{paper_context}

Evaluate this FR critically. Consider:

1. **Problem fit**: Does this solve an actual problem the project has, or is it just topically related? A trading system doesn't need every multi-agent technique.
2. **Architectural fit**: Is this compatible with the project's current design? Would it require fundamental restructuring?
3. **Value vs effort**: Is the expected benefit worth the implementation cost?
4. **Alternatives**: Could the project achieve the same goal more simply without this research concept?
5. **Readiness**: Is the research mature enough to implement, or still experimental?

Respond with JSON:
{{
  "verdict": "accept | revise | reject",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentences explaining your verdict",
  "revised_title": "if revise: improved FR title that better fits the project",
  "revised_description": "if revise: improved description scoped to what actually makes sense",
  "revised_priority": "high | medium | low",
  "concerns": ["list of specific concerns or risks"]
}}

Only output JSON. No other text.
"""

_EVALUATE_CAPABILITY_PROMPT = """\
You are evaluating whether a new library capability could improve a downstream application.

LIBRARY: khonliang
NEW CAPABILITY:
{capability_description}

APPLICATION: researcher
DESCRIPTION: Automated research pipeline that discovers papers via RSS/search, fetches and
distills them using local LLMs, extracts concept graphs and triples, scores relevance to
configured projects, generates feature requests via synergize, and serves everything over MCP
so project Claudes can consume research insights.

CURRENT PIPELINE COMPONENTS:
- Paper fetching (RSS, arxiv, web)
- LLM distillation (summarize, extract triples, assess per-project)
- Embedding-based relevance scoring
- Concept graph with project score propagation
- Synergize: concept classification and FR generation
- FR lifecycle: promote, review, merge, dependencies, status tracking
- Idea ingestion: parse informal text into researchable claims
- Knowledge store, triple store, digest store (all SQLite-backed)

EXISTING RESEARCHER FRs (open):
{existing_frs}

Evaluate:
1. **Direct applicability**: Could the researcher pipeline use this capability directly? Where?
2. **Improvement potential**: Would it make existing features better (faster, more accurate, etc.)?
3. **New features unlocked**: Does it enable something the researcher can't do today?
4. **Integration effort**: How much work to adopt? Does it replace existing code or add to it?

Respond with JSON:
{{
  "applicable": true/false,
  "score": 0.0-1.0,
  "summary": "one-line assessment",
  "direct_uses": ["where this capability applies in the researcher pipeline"],
  "improvements": ["existing features it would improve"],
  "new_features": ["new capabilities it would unlock"],
  "integration_notes": "brief assessment of effort and approach",
  "suggested_frs": [
    {{
      "title": "FR title for researcher",
      "description": "what to build",
      "priority": "high | medium | low",
      "depends_on": "KH-X if applicable"
    }}
  ]
}}

Only output JSON. No other text.
"""

_RESEARCH_QUERIES_PROMPT = """\
A software project has these implemented capabilities:

PROJECT: {project_name}
DESCRIPTION: {project_description}

CAPABILITIES:
{capabilities}

Generate {num_queries} arxiv search queries to find papers that advance these capabilities.

IMPORTANT:
- Translate capability names into academic terminology
- BAD: "per-project landscape reports" (too product-specific)
- GOOD: "multi-document summarization for software projects"
- BAD: "automated environment health checks"
- GOOD: "anomaly detection in distributed system monitoring"
- Each query should be 4-8 words of arxiv-style keywords

Respond with JSON:
{{
  "queries": [
    {{
      "query": "arxiv search terms",
      "targets": "which capabilities this improves",
      "rationale": "one sentence why"
    }}
  ]
}}

Only output JSON. No other text.
"""

_SCAN_PROJECT_PROMPT = """\
Given the AST-extracted structure of a Python project, identify its capabilities.

PROJECT: {project_name}
DESCRIPTION: {project_description}
DEPENDS ON: {dependencies}

MODULE MAP:
{module_map}

Extract capabilities. Each must be:
- Concrete and specific (not "data processing" but "GEDCOM file parsing")
- Actually implemented (classes/methods exist), not aspirational
- Named in 3-8 words

Also identify what this project imports from its dependencies (not stdlib/pip).
Identify the primary architecture pattern (e.g., "plugin system", "pipeline", "event-driven", "CLI tool", "MCP server", "library") in 3-8 words.

Respond with JSON:
{{
  "capabilities": ["Capability Name 1", "Capability Name 2", ...],
  "imports_from": {{"dependency_name": ["what it uses"]}},
  "architecture": "primary pattern in 3-8 words"
}}

Only output JSON. No other text.
"""

_LANDSCAPE_PROMPT = """\
Analyze these {count} paper summaries to map the research landscape.

Structure your response as:

## Major Research Directions
- Direction 1: description, key papers, maturity level

## Emerging Trends
- Trend 1: description, earliest/latest papers

## Consensus Views
- What most papers agree on

## Contested Areas
- Where papers disagree or take different approaches

## Gaps in the Literature
- What's not being studied but should be

Paper summaries:
{summaries}
"""


_SELECTION_PROMPT = """\
You are evaluating {n} candidate outputs for the same synthesis task.
Select the BEST candidate based on:
1. Concept diversity — covers distinct, non-overlapping ideas
2. FR quality — specific, well-scoped, actionable feature requests
3. Correct scoping — library vs app classification is accurate
4. Novelty — proposes genuinely new capabilities, not restatements

CANDIDATES:
{candidates}

Respond with ONLY the number of the best candidate (1-{n}). No other text.
"""


@dataclass
class SynthesisResult:
    query: str
    synthesis_type: str  # topic, project, landscape
    content: str
    paper_count: int
    paper_ids: List[str] = field(default_factory=list)
    success: bool = False


class Synthesizer:
    """Generate combined summaries across multiple papers."""

    def __init__(self, knowledge: KnowledgeStore, triples: TripleStore, pool: ModelPool):
        self.knowledge = knowledge
        self.triples = triples
        self.pool = pool

    def _get_distilled_summaries(
        self, query: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get distilled paper summaries, optionally filtered by search query."""
        if query:
            entries = self.knowledge.search(query, scope="research", limit=limit or 500)
            # Filter to Tier 3 summaries only
            entries = [e for e in entries if e.tier == Tier.DERIVED and "summary" in (e.tags or [])]
        else:
            entries = [
                e
                for e in self.knowledge.get_by_tier(Tier.DERIVED)
                if "summary" in (e.tags or [])
            ]
            if limit:
                entries = entries[:limit]

        summaries = []
        for entry in entries:
            try:
                data = json.loads(entry.content)
            except (json.JSONDecodeError, TypeError):
                data = {"raw": entry.content}

            summaries.append({
                "entry_id": entry.id,
                "parent_id": entry.metadata.get("parent_id", ""),
                "title": data.get("title", entry.title),
                "summary": data,
                "assessments": entry.metadata.get("assessments", {}),
            })

        return summaries

    def _format_summaries(self, summaries: List[Dict], max_chars: int = 10000) -> str:
        """Format summaries compactly for prompt injection.

        Keeps total under max_chars to avoid exceeding model context.
        """
        parts = []
        total = 0
        for i, s in enumerate(summaries, 1):
            data = s["summary"]
            title = data.get("title", s["title"])
            abstract = data.get("abstract", "")
            findings = data.get("key_findings", [])
            methods = data.get("methods", [])

            # Compact format: one block per paper
            lines = [f"{i}. {title}"]
            if abstract:
                lines.append(f"   {abstract[:150]}")
            if findings:
                for f in findings[:3]:
                    lines.append(f"   - {f[:100]}")
            if methods:
                lines.append(f"   Methods: {', '.join(methods)}")

            block = "\n".join(lines)
            if total + len(block) > max_chars:
                parts.append(f"... and {len(summaries) - i + 1} more papers")
                break
            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    async def _generate(
        self, prompt: str, n_samples: int = 1, compare: bool = False,
    ) -> str:
        """Run LLM generation via the summarizer model.

        Always delegates to select_best_of_n. For ordinary single-sample
        generation it uses n=1 (which short-circuits to one client.generate
        call inside the helper). For multi-candidate runs it uses n_samples
        (or max(n_samples, 2) when compare=True so the returned payload can
        carry candidate + selection metadata for quality evaluation).

        Selection uses researcher's FR-aware _SELECTION_PROMPT.
        """
        client = self.pool.get_client("summarizer")
        n = max(n_samples, 2) if compare else n_samples

        result = await select_best_of_n(
            client,
            prompt,
            n=n,
            system=_SYNTHESIS_SYSTEM,
            temperature=0.3 if n <= 1 else 0.7,
            selection_temperature=0.1,
            max_tokens=6000,
            selection_prompt_template=_SELECTION_PROMPT,
            return_candidates=compare,
        )

        if compare:
            return serialize_candidates(result)
        return result

    async def topic_summary(
        self, topic: str, limit: int = 30
    ) -> SynthesisResult:
        """Synthesize papers around a topic/query."""
        summaries = self._get_distilled_summaries(query=topic, limit=limit)
        if not summaries:
            return SynthesisResult(
                query=topic, synthesis_type="topic", content="No distilled papers found.",
                paper_count=0, success=False,
            )

        formatted = self._format_summaries(summaries)
        prompt = _TOPIC_PROMPT.format(
            count=len(summaries), topic=topic, summaries=formatted,
        )

        content = await self._generate(prompt)
        return SynthesisResult(
            query=topic,
            synthesis_type="topic",
            content=content,
            paper_count=len(summaries),
            paper_ids=[s["entry_id"] for s in summaries],
            success=True,
        )

    async def project_brief(
        self, project_name: str, project_description: str, limit: int = 20
    ) -> SynthesisResult:
        """Generate applicability brief for a specific project."""
        summaries = self._get_distilled_summaries(limit=None)  # fetch all, filter by score below
        # Filter to summaries that scored for this project
        summaries = [
            s for s in summaries
            if project_name in s.get("assessments", {})
            and isinstance(s["assessments"][project_name], dict)
            and float(s["assessments"][project_name].get("score", 0)) > 0.3
        ]
        # Sort by score descending, take top N
        summaries.sort(
            key=lambda s: float(s.get("assessments", {}).get(project_name, {}).get("score", 0)),
            reverse=True,
        )
        summaries = summaries[:limit]
        if not summaries:
            return SynthesisResult(
                query=project_name, synthesis_type="project",
                content="No distilled papers found.", paper_count=0, success=False,
            )

        formatted = self._format_summaries(summaries)
        prompt = _PROJECT_PROMPT.format(
            count=len(summaries),
            project_name=project_name,
            project_description=project_description,
            summaries=formatted,
        )

        content = await self._generate(prompt)
        return SynthesisResult(
            query=project_name,
            synthesis_type="project",
            content=content,
            paper_count=len(summaries),
            paper_ids=[s["entry_id"] for s in summaries],
            success=True,
        )

    async def landscape(self, limit: int = 50) -> SynthesisResult:
        """Generate a research landscape overview across all papers."""
        summaries = self._get_distilled_summaries(limit=limit)
        if not summaries:
            return SynthesisResult(
                query="landscape", synthesis_type="landscape",
                content="No distilled papers found.", paper_count=0, success=False,
            )

        formatted = self._format_summaries(summaries)
        prompt = _LANDSCAPE_PROMPT.format(
            count=len(summaries), summaries=formatted,
        )

        content = await self._generate(prompt)

        # Also include triple-store relationships
        triple_ctx = self.triples.build_context(max_triples=30, min_confidence=0.5)
        if triple_ctx:
            content += f"\n\n## Known Relationships (from triple store)\n{triple_ctx}"

        return SynthesisResult(
            query="landscape",
            synthesis_type="landscape",
            content=content,
            paper_count=len(summaries),
            paper_ids=[s["entry_id"] for s in summaries],
            success=True,
        )

    def _get_existing_fr_concepts(self) -> set:
        """Get lowercase concept names from existing open FRs."""
        concepts = set()
        for entry in self.knowledge.get_by_tier(Tier.DERIVED):
            if not entry.tags or "fr" not in entry.tags:
                continue
            status = (entry.metadata or {}).get("fr_status", "open")
            if status in ("completed", "archived"):
                continue
            concept = (entry.metadata or {}).get("concept", "")
            if concept:
                concepts.add(concept.lower())
        return concepts

    async def _deduplicate_concepts(
        self,
        ranked: List,
        max_per_cluster: int = 2,
        similarity_threshold: float = 0.70,
    ) -> List:
        """Cluster similar concepts by embedding similarity, keep top N per cluster.

        This ensures synergize surfaces diverse concepts instead of N variations
        of the same theme (e.g., 'GRA', 'collaborative framework GRA', 'small LLMs
        in data synthesis' all from one paper).
        """
        if len(ranked) <= max_per_cluster:
            return ranked

        from khonliang_researcher import cosine_similarity

        # Embed all concept names
        client = self.pool.get_client("summarizer")
        ollama_url = client.base_url if hasattr(client, "base_url") else "http://localhost:11434"
        embed_model = "nomic-embed-text"

        import aiohttp
        embeddings = {}
        try:
            async with aiohttp.ClientSession() as session:
                for concept, _ in ranked:
                    async with session.post(
                        f"{ollama_url}/api/embed",
                        json={"model": embed_model, "input": concept},
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            emb = data.get("embeddings", [[]])[0]
                            if emb:
                                embeddings[concept] = emb
        except Exception as e:
            logger.warning("Concept dedup embedding failed: %s — skipping dedup", e)
            return ranked

        # Greedy clustering: assign each concept to first cluster it's similar to
        clusters: List[List] = []
        for concept, scores in ranked:
            if concept not in embeddings:
                clusters.append([(concept, scores)])
                continue
            placed = False
            for cluster in clusters:
                rep = cluster[0][0]
                if rep in embeddings:
                    sim = cosine_similarity(embeddings[concept], embeddings[rep])
                    if sim >= similarity_threshold:
                        cluster.append((concept, scores))
                        placed = True
                        break
            if not placed:
                clusters.append([(concept, scores)])

        # Pick top max_per_cluster from each cluster
        diverse = []
        for cluster in clusters:
            diverse.extend(cluster[:max_per_cluster])

        # Re-sort by total score
        diverse.sort(key=lambda x: sum(x[1].values()), reverse=True)
        logger.info(
            "Concept dedup: %d → %d concepts (%d clusters)",
            len(ranked), len(diverse), len(clusters),
        )
        return diverse

    async def synergize_concepts(
        self,
        min_score: float = 0.5,
        max_concepts: int = 10,
        n_samples: int = 1,
        compare: bool = False,
    ) -> SynthesisResult:
        """Find conceptual connections across the corpus. Returns bundles, NOT FRs.

        This is the generic, domain-agnostic version of synergize. It groups
        related concepts based on shared themes, methods, or complementary
        findings. What to DO with the bundles (FRs, research leads, evidence
        reports) is the caller's decision — not the researcher's.

        The FR-generating ``synergize()`` method is the developer-specific
        version that adds classification, targeting, and FR output. It stays
        for backward compatibility but the clean path is:
          researcher.synergize_concepts() → developer.process_bundles()
        """
        from khonliang_researcher import build_project_scores

        concept_scores = build_project_scores(self.knowledge, self.triples, min_score=min_score)
        if not concept_scores:
            return SynthesisResult(
                query="synergize_concepts", synthesis_type="concept_bundles",
                content="No concept scores available. Distill papers first.",
                paper_count=0, success=False,
            )

        # Filter and rank
        qualified = {
            concept: scores
            for concept, scores in concept_scores.items()
            if max(scores.values()) >= min_score
        }

        ranked = sorted(
            qualified.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True,
        )

        ranked = await self._deduplicate_concepts(ranked, max_per_cluster=2)
        ranked = ranked[:max_concepts]

        if not ranked:
            return SynthesisResult(
                query="synergize_concepts", synthesis_type="concept_bundles",
                content="No concepts meet the score threshold.",
                paper_count=0, success=False,
            )

        # Find backing papers via triples
        concept_papers: Dict[str, List[str]] = defaultdict(list)
        all_triples = self.triples.get(min_confidence=0.3, limit=5000)
        for t in all_triples:
            source = t.source or ""
            if source.startswith("paper:"):
                paper_id = source.replace("paper:", "")
                if t.subject in qualified:
                    concept_papers[t.subject].append(paper_id)
                if t.object in qualified:
                    concept_papers[t.object].append(paper_id)

        # Gather paper summaries
        paper_ids_needed = set()
        for ids in concept_papers.values():
            paper_ids_needed.update(ids)

        summaries = self._get_distilled_summaries(limit=100)
        summary_by_parent = {}
        for s in summaries:
            pid = s["parent_id"]
            if pid in paper_ids_needed:
                summary_by_parent[pid] = s

        # Format concepts for prompt
        concept_lines = []
        for concept, scores in ranked:
            score_str = ", ".join(f"{p}: {s:.0%}" for p, s in sorted(scores.items(), key=lambda x: -x[1]))
            papers = concept_papers.get(concept, [])
            paper_titles = [summary_by_parent[pid]["title"] for pid in papers[:3] if pid in summary_by_parent]
            papers_str = "; ".join(paper_titles) if paper_titles else "no direct papers"
            concept_lines.append(f"- {concept} [{score_str}] — papers: {papers_str}")

        formatted_summaries = self._format_summaries(
            [summary_by_parent[pid] for pid in list(summary_by_parent.keys())[:20]],
            max_chars=6000,
        )

        prompt = _SYNERGIZE_CONCEPTS_PROMPT.format(
            concepts="\n".join(concept_lines),
            summaries=formatted_summaries,
        )

        content = await self._generate(prompt, n_samples=n_samples, compare=compare)

        return SynthesisResult(
            query="synergize_concepts",
            synthesis_type="concept_bundles",
            content=content,
            paper_count=len(summary_by_parent),
            paper_ids=list(summary_by_parent.keys()),
            success=True,
        )

    async def synergize(
        self,
        projects: Dict[str, Dict[str, Any]],
        min_score: float = 0.5,
        min_projects: int = 1,
        max_concepts: int = 10,
        n_samples: int = 1,
        compare: bool = False,
    ) -> SynthesisResult:
        """Classify research concepts and generate targeted FRs.

        Analyzes concept scores across projects to determine whether each
        concept should become a library feature, app feature, or both.
        Deduplicates similar concepts via embeddings and excludes concepts
        that already have open FRs.

        Returns SynthesisResult with JSON content containing classified
        concepts and generated feature requests.
        """
        from khonliang_researcher import build_project_scores

        # Step 1: Get concept scores across all projects
        concept_scores = build_project_scores(self.knowledge, self.triples, min_score=min_score)
        if not concept_scores:
            return SynthesisResult(
                query="synergize", synthesis_type="synergize",
                content="No concept scores available. Distill papers first.",
                paper_count=0, success=False,
            )

        # Step 2: Filter to concepts that score for at least min_projects
        qualified = {
            concept: scores
            for concept, scores in concept_scores.items()
            if len([s for s in scores.values() if s >= min_score]) >= min_projects
        }

        # Step 2b: Exclude concepts already covered by open FRs
        existing_fr_concepts = self._get_existing_fr_concepts()
        if existing_fr_concepts:
            before = len(qualified)
            qualified = {
                c: s for c, s in qualified.items()
                if c.lower() not in existing_fr_concepts
            }
            excluded = before - len(qualified)
            if excluded:
                logger.info("Excluded %d concepts already covered by FRs", excluded)

        # Sort by total score across projects (most cross-cutting first)
        ranked = sorted(
            qualified.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True,
        )

        # Step 2c: Deduplicate similar concepts via embeddings
        ranked = await self._deduplicate_concepts(ranked, max_per_cluster=2)
        ranked = ranked[:max_concepts]

        if not ranked:
            return SynthesisResult(
                query="synergize", synthesis_type="synergize",
                content="No concepts meet the score threshold.",
                paper_count=0, success=False,
            )

        # Step 3: Find backing papers for each concept via triples
        concept_papers: Dict[str, List[str]] = defaultdict(list)
        all_triples = self.triples.get(min_confidence=0.3, limit=5000)
        for t in all_triples:
            source = t.source or ""
            if source.startswith("paper:"):
                paper_id = source.replace("paper:", "")
                if t.subject in qualified:
                    concept_papers[t.subject].append(paper_id)
                if t.object in qualified:
                    concept_papers[t.object].append(paper_id)

        # Step 4: Gather paper summaries
        paper_ids_needed = set()
        for ids in concept_papers.values():
            paper_ids_needed.update(ids)

        summaries = self._get_distilled_summaries(limit=100)
        summary_by_parent = {}
        for s in summaries:
            pid = s["parent_id"]
            if pid in paper_ids_needed:
                summary_by_parent[pid] = s

        # Step 5: Format concepts for prompt
        concept_lines = []
        for concept, scores in ranked:
            score_str = ", ".join(f"{p}: {s:.0%}" for p, s in sorted(scores.items(), key=lambda x: -x[1]))
            papers = concept_papers.get(concept, [])
            paper_titles = []
            for pid in papers[:3]:
                if pid in summary_by_parent:
                    paper_titles.append(summary_by_parent[pid]["title"])
            papers_str = "; ".join(paper_titles) if paper_titles else "no direct papers"
            concept_lines.append(f"- {concept} [{score_str}] — papers: {papers_str}")

        # Step 6: Format project descriptions
        library_desc = ""
        app_descs = []
        for name, cfg in projects.items():
            desc = cfg.get("description", "").strip()
            scope = cfg.get("scope", "")
            if scope == "library":
                library_desc = f"{name}: {desc}"
            else:
                app_descs.append(f"- {name} ({scope}): {desc}")

        # Step 7: Build dependency graph text
        dep_lines = []
        for name, cfg in projects.items():
            deps = cfg.get("depends_on", [])
            if deps:
                dep_lines.append(f"- {name} depends on: {', '.join(deps)}")
            else:
                dep_lines.append(f"- {name}: base library (no dependencies)")
        dep_lines.append("- Apps do NOT depend on each other. No cross-app references in FRs.")

        # Step 8: Gather existing/planned capabilities from knowledge store
        exists_lines = []
        planned_lines = []
        for entry in self.knowledge.get_by_tier(Tier.DERIVED):
            tags = entry.tags or []
            if "capability" not in tags:
                continue
            cap_target = (entry.metadata or {}).get("target", "")
            cap_status = (entry.metadata or {}).get("capability_status", "")
            if cap_status == "exists":
                exists_lines.append(f"- [{cap_target}] {entry.title}")
            elif cap_status == "planned":
                planned_lines.append(f"- [{cap_target}] {entry.title}")

        existing_text = "\n".join(exists_lines) if exists_lines else "None tracked yet — check project descriptions above for what already exists."
        planned_text = "\n".join(planned_lines) if planned_lines else "None"

        # Step 9: Format paper summaries (compact)
        formatted_summaries = self._format_summaries(
            [summary_by_parent[pid] for pid in list(summary_by_parent.keys())[:20]],
            max_chars=6000,
        )

        # Step 10: Generate
        prompt = _SYNERGIZE_PROMPT.format(
            library_description=library_desc or "Multi-agent LLM orchestration library",
            app_descriptions="\n".join(app_descs) or "No app descriptions available",
            dependency_graph="\n".join(dep_lines),
            existing_capabilities=existing_text,
            planned_capabilities=planned_text,
            concepts="\n".join(concept_lines),
            summaries=formatted_summaries,
        )

        content = await self._generate(prompt, n_samples=n_samples, compare=compare)

        return SynthesisResult(
            query="synergize",
            synthesis_type="synergize",
            content=content,
            paper_count=len(summary_by_parent),
            paper_ids=list(summary_by_parent.keys()),
            success=True,
        )

    async def review_fr(
        self,
        fr_data: Dict[str, Any],
        project_config: Dict[str, Any],
        project_name: str,
    ) -> Dict[str, Any]:
        """Deep review of a single FR using the largest available model.

        Returns review verdict with reasoning, revisions, and concerns.
        """
        # Get paper context for the backing papers
        backing = fr_data.get("backing_papers", [])
        paper_context = ""
        if backing:
            summaries = self._get_distilled_summaries(limit=50)
            relevant = [s for s in summaries if s["title"] in backing]
            if relevant:
                paper_context = self._format_summaries(relevant, max_chars=3000)

        if not paper_context:
            # Fallback: search for the concept
            concept = fr_data.get("concept", "")
            if concept:
                summaries = self._get_distilled_summaries(query=concept, limit=5)
                if summaries:
                    paper_context = self._format_summaries(summaries, max_chars=3000)

        deps = project_config.get("depends_on", [])
        prompt = _REVIEW_PROMPT.format(
            project_name=project_name,
            project_description=project_config.get("description", "").strip(),
            dependencies=", ".join(deps) if deps else "none (base library)",
            fr_title=fr_data.get("title", ""),
            fr_description=fr_data.get("description", ""),
            classification=fr_data.get("classification", ""),
            concept=fr_data.get("concept", ""),
            backing_papers=", ".join(backing) if backing else "none cited",
            paper_context=paper_context or "No paper summaries available.",
        )

        # Use reviewer model (largest available)
        client = self.pool.get_client("reviewer")
        content = await client.generate(
            prompt=prompt,
            system="You are a senior software architect. Be critical and practical. Reject FRs that are vague, redundant, or don't solve a real problem.",
            temperature=0.2,
            max_tokens=2000,
        )

        # Parse response
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "verdict": "error",
                "confidence": 0,
                "reasoning": f"Review model returned non-JSON: {content[:200]}",
                "concerns": [],
            }

    async def review_all_frs(
        self,
        projects: Dict[str, Dict[str, Any]],
        target: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Review all stored FRs using the large model.

        Returns list of {fr, review} dicts.
        """
        from researcher.pipeline import create_pipeline
        # We need the pipeline's get_feature_requests �� but we're called from it
        # So accept frs as parameter instead
        raise NotImplementedError("Call pipeline.review_frs() instead")

    async def idea_brief(
        self, idea_text: str, claims: List[str], paper_summaries: List[Dict]
    ) -> SynthesisResult:
        """Evaluate claims from an idea against found literature."""
        if not paper_summaries:
            return SynthesisResult(
                query="idea", synthesis_type="idea",
                content="No paper summaries to evaluate against.",
                paper_count=0, success=False,
            )

        # Format summaries compactly
        parts = []
        for i, s in enumerate(paper_summaries, 1):
            title = s.get("title", "Untitled")
            abstract = s.get("abstract", "")
            findings = s.get("key_findings", [])
            lines = [f"{i}. {title}"]
            if abstract:
                lines.append(f"   {abstract[:200]}")
            for f in findings[:3]:
                lines.append(f"   - {f[:120]}")
            parts.append("\n".join(lines))

        formatted = "\n\n".join(parts)
        claims_text = "\n".join(f"- {c}" for c in claims)

        prompt = _IDEA_PROMPT.format(
            idea_text=idea_text[:3000],
            claims=claims_text,
            count=len(paper_summaries),
            summaries=formatted,
        )

        content = await self._generate(prompt)
        return SynthesisResult(
            query="idea",
            synthesis_type="idea",
            content=content,
            paper_count=len(paper_summaries),
            success=True,
        )

    async def scan_codebase(
        self,
        project_name: str,
        repo_path: str,
        description: str = "",
        dependencies: str = "",
    ) -> SynthesisResult:
        """AST-based codebase scan.

        Phase 1: ast module extracts classes, methods, docstrings, imports.
        Phase 2: Single 7B call synthesizes capabilities from the structured map.
        """
        import ast
        import os
        from pathlib import Path
        from researcher.util import repo_tree

        try:
            tree_ctx = repo_tree(repo_path, prefix="researcher_scan_")
            repo_context = tree_ctx.__enter__()
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            return SynthesisResult(
                query="scan", synthesis_type="scan",
                content=str(e),
                paper_count=0, success=False,
            )

        try:
            repo = Path(repo_context)
            # ---- Phase 1: AST extraction ----
            skip_dirs = {"__pycache__", ".venv", "venv", "node_modules", ".git",
                          "dist", "build", ".eggs", "tests", "test"}
            module_map = []

            for root, dirs, files in os.walk(repo):
                dirs[:] = [d for d in dirs
                           if not d.startswith(".") and d not in skip_dirs
                           and not d.endswith(".egg-info")]
                for f in sorted(files):
                    if not f.endswith(".py"):
                        continue
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, repo)
                    try:
                        source = Path(abs_path).read_text(errors="replace")
                        ast_tree = ast.parse(source)
                    except (SyntaxError, ValueError):
                        continue

                    info = self._extract_ast_info(ast_tree)
                    if not info["classes"] and not info["functions"] and not info["imports"]:
                        continue
                    module_map.append((rel_path, info))

            # Build set of all project-defined names for call graph filtering
            project_names = set()
            for _, info in module_map:
                for cls in info["classes"]:
                    project_names.add(cls["name"])
                    for m in cls["methods"]:
                        project_names.add(m["name"])
                for func in info["functions"]:
                    project_names.add(func["name"])

            _STDLIB_PREFIXES = ("os", "sys", "json", "logging", "typing",
                                "dataclasses", "pathlib", "collections", "abc",
                                "enum", "time", "datetime", "re", "hashlib",
                                "math", "functools", "asyncio", "uuid",
                                "sqlite3", "inspect", "textwrap", "contextlib",
                                "copy", "io", "threading", "concurrent")

            # Format module map: compact one line per module.
            lines = []
            for rel_path, info in module_map:
                doc = info.get("docstring", "")
                parts = [rel_path]
                if doc:
                    parts.append(f'"{doc}"')
                for cls in info["classes"]:
                    bases = f"({', '.join(cls['bases'])})" if cls["bases"] else ""
                    methods = ", ".join(m["name"] for m in cls["methods"])
                    parts.append(f"class {cls['name']}{bases} [{methods}]" if methods
                                 else f"class {cls['name']}{bases}")
                for func in info["functions"]:
                    parts.append(f"{'async ' if func.get('is_async') else ''}def {func['name']}")
                # External imports only (skip stdlib)
                ext_imports = [i for i in info["imports"]
                               if not i.startswith(_STDLIB_PREFIXES)]
                if ext_imports:
                    parts.append(f"imports: {', '.join(ext_imports[:3])}")
                # Intra-project calls (intersection of callee names with project definitions)
                # Filter out generic names that don't convey architecture
                _GENERIC = {"__init__", "handle", "run", "start", "stop", "close",
                            "get", "set", "add", "remove", "update", "delete",
                            "read", "write", "load", "save", "open", "parse",
                            "format", "build", "create", "make", "init", "setup"}
                project_calls = sorted(
                    (info.get("calls", set()) & project_names) - _GENERIC
                )
                if project_calls:
                    parts.append(f"calls: {', '.join(project_calls[:8])}")
                lines.append(" | ".join(parts))

            logger.info("AST scan: %d modules, %d chars for %s",
                         len(module_map), sum(len(l) for l in lines), project_name)

            # ---- Phase 2: Chunked LLM calls ----
            # Split modules into chunks to avoid attention degradation on long inputs.
            chunk_size = 20
            all_capabilities = []
            all_imports: Dict[str, list] = {}
            all_architectures: list[str] = []
            reviewer = self.pool.get_client("reviewer")

            for i in range(0, len(lines), chunk_size):
                chunk_text = "\n".join(lines[i:i + chunk_size])
                prompt = _SCAN_PROJECT_PROMPT.format(
                    project_name=project_name,
                    project_description=description or "No description",
                    dependencies=dependencies or "None",
                    module_map=chunk_text,
                )
                try:
                    raw = await reviewer.generate(
                        prompt=prompt,
                        system="You extract capabilities from code structure. Output only JSON.",
                        temperature=0.1,
                        max_tokens=2000,
                    )
                    sj = raw.strip()
                    if sj.startswith("```"):
                        sj = "\n".join(sj.split("\n")[1:])
                    if sj.endswith("```"):
                        sj = "\n".join(sj.split("\n")[:-1])
                    data = json.loads(sj)
                    for cap in data.get("capabilities", []):
                        if cap not in all_capabilities:
                            all_capabilities.append(cap)
                    for dep, items in data.get("imports_from", {}).items():
                        all_imports.setdefault(dep, []).extend(
                            i for i in items if i not in all_imports.get(dep, [])
                        )
                    arch = data.get("architecture", "")
                    if arch and arch not in all_architectures:
                        all_architectures.append(arch)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning("Scan chunk %d failed for %s: %s",
                                   i // chunk_size, project_name, e
                    )
                    continue

            content = json.dumps({
                "capabilities": all_capabilities,
                "imports_from": all_imports,
                "architecture": all_architectures[0] if all_architectures else "",
            })
            return SynthesisResult(
                query="scan",
                synthesis_type="scan",
                content=content,
                paper_count=len(module_map),
                success=True,
            )
        finally:
            tree_ctx.__exit__(None, None, None)

    @staticmethod
    def _extract_ast_info(tree: "ast.Module") -> dict:
        """Extract structured info from an AST: classes, functions, imports, calls."""
        import ast

        info: dict = {
            "docstring": ast.get_docstring(tree) or "",
            "imports": [],
            "classes": [],
            "functions": [],
            "calls": set(),  # callee names found in function bodies
        }
        if len(info["docstring"]) > 120:
            info["docstring"] = info["docstring"][:120] + "..."

        def _collect_calls(body_nodes):
            """Walk function body and collect call target names."""
            for node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
                if isinstance(node, ast.Call):
                    # self.foo(...) -> "foo"
                    if isinstance(node.func, ast.Attribute):
                        info["calls"].add(node.func.attr)
                    # foo(...) -> "foo"
                    elif isinstance(node.func, ast.Name):
                        info["calls"].add(node.func.id)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    info["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [a.name for a in node.names[:5]]
                if len(node.names) > 5:
                    names.append(f"...+{len(node.names) - 5}")
                info["imports"].append(f"{module}: {', '.join(names)}")

            elif isinstance(node, ast.ClassDef):
                cls = {
                    "name": node.name,
                    "bases": [ast.unparse(b) for b in node.bases],
                    "docstring": (ast.get_docstring(node) or "")[:120],
                    "methods": [],
                }
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("_") and item.name != "__init__":
                            continue
                        args = [a.arg for a in item.args.args if a.arg != "self"][:4]
                        cls["methods"].append({
                            "name": item.name,
                            "args": args,
                            "docstring": (ast.get_docstring(item) or "")[:80],
                        })
                        _collect_calls(item.body)
                info["classes"].append(cls)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                args = [a.arg for a in node.args.args if a.arg != "self"][:4]
                info["functions"].append({
                    "name": node.name,
                    "args": args,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "docstring": (ast.get_docstring(node) or "")[:80],
                })
                _collect_calls(node.body)

        return info

    async def generate_research_queries(
        self,
        project_name: str,
        capabilities: List[str],
        description: str = "",
        num_queries: int = 5,
    ) -> List[Dict[str, str]]:
        """Turn project capabilities into academic search queries."""
        # Cap at 20 to keep within 3B context
        caps = capabilities[:20] if len(capabilities) > 20 else capabilities
        cap_text = "\n".join(f"- {c}" for c in caps)
        prompt = _RESEARCH_QUERIES_PROMPT.format(
            project_name=project_name,
            project_description=description or "No description",
            capabilities=cap_text,
            num_queries=num_queries,
        )

        # Use extractor (3B) — cheap, just clustering and rephrasing
        extractor = self.pool.get_client("extractor")
        raw = await extractor.generate(
            prompt=prompt,
            system="You generate research search queries. Output only JSON.",
            temperature=0.3,
            max_tokens=2000,
        )

        sj = raw.strip()
        if sj.startswith("```"):
            sj = "\n".join(sj.split("\n")[1:])
        if sj.endswith("```"):
            sj = "\n".join(sj.split("\n")[:-1])

        try:
            data = json.loads(sj)
            return data.get("queries", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse research queries: %s", raw[:200])
            return []

    async def evaluate_capability(
        self, capability_description: str
    ) -> SynthesisResult:
        """Evaluate whether a new khonliang capability could improve the researcher.

        Takes a description of a new/updated khonliang feature and returns
        an assessment of how the researcher pipeline could leverage it.
        """
        # Gather existing researcher FRs for context
        existing_frs = []
        for entry in self.knowledge.get_by_tier(Tier.DERIVED):
            if not entry.tags or "fr" not in entry.tags:
                continue
            if "target:researcher" not in (entry.tags or []):
                continue
            status = (entry.metadata or {}).get("fr_status", "open")
            if status in ("completed", "archived"):
                continue
            existing_frs.append(f"- [{(entry.metadata or {}).get('priority', '?')}] {entry.title}")

        frs_text = "\n".join(existing_frs) if existing_frs else "None"

        prompt = _EVALUATE_CAPABILITY_PROMPT.format(
            capability_description=capability_description,
            existing_frs=frs_text,
        )

        content = await self._generate(prompt)
        return SynthesisResult(
            query="evaluate_capability",
            synthesis_type="evaluate_capability",
            content=content,
            paper_count=0,
            success=True,
        )
