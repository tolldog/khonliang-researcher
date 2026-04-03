"""Synthesize combined summaries across multiple distilled papers.

Uses khonliang's OllamaClient to generate cross-paper analysis:
  - Topic summaries: "What do these papers say about consensus?"
  - Project briefs: "What's applicable to autostock?"
  - Landscape overviews: "What are the main themes and gaps?"
  - Relationship maps: "How do these papers connect?"
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from khonliang.knowledge.store import KnowledgeStore, Tier
from khonliang.knowledge.triples import TripleStore
from khonliang.pool import ModelPool

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
        self, query: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get distilled paper summaries, optionally filtered by search query."""
        if query:
            entries = self.knowledge.search(query, scope="research", limit=limit)
            # Filter to Tier 3 summaries only
            entries = [e for e in entries if e.tier == Tier.DERIVED and "summary" in (e.tags or [])]
        else:
            entries = [
                e
                for e in self.knowledge.get_by_tier(Tier.DERIVED)
                if "summary" in (e.tags or [])
            ][:limit]

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

    async def _generate(self, prompt: str) -> str:
        """Run LLM generation via the summarizer model."""
        client = self.pool.get_client("summarizer")
        return await client.generate(
            prompt=prompt,
            system=_SYNTHESIS_SYSTEM,
            temperature=0.3,
            max_tokens=4000,
        )

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
        summaries = self._get_distilled_summaries(limit=limit)
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
