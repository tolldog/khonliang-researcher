"""LLM distillation roles for research papers.

Three BaseRole subclasses:
  - SummarizerRole: structured paper summary
  - ExtractorRole: relationship triple extraction
  - AssessorRole: project applicability scoring
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from khonliang.roles.base import BaseRole

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class SummarizerRole(BaseRole):
    """Produce a structured JSON summary of a research paper."""

    def __init__(self, model_pool, **kwargs):
        super().__init__(
            role="summarizer",
            model_pool=model_pool,
            system_prompt=_load_prompt("summarizer.md"),
            **kwargs,
        )

    async def handle(
        self, message: str, session_id: str = "", context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Truncate to fit model context (keep ~12k chars for 7B model)
        text = message[:12000] if len(message) > 12000 else message
        prompt = f"Summarize this research paper:\n\n{text}"
        try:
            result = await self.client.generate_json(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.2,
                max_tokens=4000,
            )
            return {"summary": result, "success": True}
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return {"summary": None, "success": False, "error": str(e)}


class ExtractorRole(BaseRole):
    """Extract semantic triples from a paper summary."""

    def __init__(self, model_pool, **kwargs):
        super().__init__(
            role="extractor",
            model_pool=model_pool,
            system_prompt=_load_prompt("extractor.md"),
            **kwargs,
        )

    async def handle(
        self, message: str, session_id: str = "", context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        prompt = f"Extract relationship triples from this paper summary:\n\n{message}"
        try:
            result = await self.client.generate_json(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.1,
                max_tokens=3000,
            )
            # Normalize: might be a list or {"triples": [...]}
            if isinstance(result, dict) and "triples" in result:
                triples = result["triples"]
            elif isinstance(result, list):
                triples = result
            else:
                triples = []
            return {"triples": triples, "success": True}
        except Exception as e:
            logger.error("Extraction failed: %s", e)
            return {"triples": [], "success": False, "error": str(e)}


class AssessorRole(BaseRole):
    """Score a paper's applicability to a specific project."""

    def __init__(self, model_pool, **kwargs):
        super().__init__(
            role="assessor",
            model_pool=model_pool,
            system_prompt=_load_prompt("assessor.md"),
            **kwargs,
        )

    async def handle(
        self, message: str, session_id: str = "", context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        project_desc = ""
        if context and "project_description" in context:
            project_desc = context["project_description"]

        prompt = f"Paper summary:\n{message}"
        if project_desc:
            prompt += f"\n\nProject description:\n{project_desc}"
        else:
            prompt += "\n\nAssess general applicability to multi-agent LLM systems."

        try:
            result = await self.client.generate_json(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.2,
                max_tokens=2000,
            )
            return {"assessment": result, "success": True}
        except Exception as e:
            logger.error("Assessment failed: %s", e)
            return {"assessment": None, "success": False, "error": str(e)}


def _load_prompt(filename: str) -> str:
    """Load a prompt file from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text().strip()
    logger.warning("Prompt file not found: %s", path)
    return f"You are a research paper {filename.replace('.md', '')}."
