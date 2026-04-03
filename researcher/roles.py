"""LLM distillation roles for research papers.

Three BaseRole subclasses:
  - SummarizerRole: structured paper summary (model selected by paper size)
  - ExtractorRole: relationship triple extraction
  - AssessorRole: project applicability scoring

Model selection for summarization:
  tiny/small  (<15k chars): llama3.2:3b  — fast, fits in context, no truncation
  medium      (15k-50k):    qwen2.5:7b   — more capacity, truncated to 12k
  large/huge  (50k+):       qwen2.5:7b   — same, same truncation
  retry:                    deepseek-r1:14b — fallback for papers that fail on 7B
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from khonliang.roles.base import BaseRole

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Size thresholds and model tiers for summarization
MODEL_TIERS: List[Tuple[int, str, int]] = [
    # (max_chars, model, context_limit)
    (15_000, "llama3.2:3b", 15_000),    # tiny/small: fast model, no truncation
    (50_000, "qwen2.5:7b", 12_000),     # medium: 7B with truncation
    (999_999_999, "qwen2.5:7b", 12_000), # large/huge: same
]

# Fallback model for retries after failure
FALLBACK_MODEL = "deepseek-r1:14b"
FALLBACK_CONTEXT_LIMIT = 16_000


def _select_model(content_length: int) -> Tuple[str, int]:
    """Select model and context limit based on paper size."""
    for max_chars, model, ctx_limit in MODEL_TIERS:
        if content_length <= max_chars:
            return model, ctx_limit
    return MODEL_TIERS[-1][1], MODEL_TIERS[-1][2]


class SummarizerRole(BaseRole):
    """Produce a structured JSON summary of a research paper.

    Model is selected based on paper size. Falls back to larger model on failure.
    """

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
        content_length = len(message)
        model, ctx_limit = _select_model(content_length)
        is_retry = context.get("retry", False) if context else False

        if is_retry:
            model = FALLBACK_MODEL
            ctx_limit = FALLBACK_CONTEXT_LIMIT
            logger.info("Retry with fallback model %s", model)

        text = message[:ctx_limit] if content_length > ctx_limit else message
        text = _clean_for_json(text)
        prompt = f"Summarize this research paper:\n\n{text}"

        logger.debug(
            "Summarizing %d chars with %s (truncated to %d)",
            content_length, model, min(content_length, ctx_limit),
        )

        try:
            result = await self.client.generate_json(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.2,
                max_tokens=4000,
                model=model,
                constrained=is_retry,  # Use Ollama native JSON mode on retry
            )
            return {"summary": result, "success": True, "model_used": model}
        except Exception as e:
            logger.error("Summarization failed with %s: %s", model, e)

            # Auto-retry with fallback model + constrained JSON if not already retrying
            if not is_retry and model != FALLBACK_MODEL:
                logger.info("Auto-retrying with %s", FALLBACK_MODEL)
                return await self.handle(
                    message, session_id,
                    context={**(context or {}), "retry": True},
                )

            return {"summary": None, "success": False, "error": str(e), "model_used": model}


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


def _clean_for_json(text: str) -> str:
    """Strip content that confuses LLM JSON generation.

    Removes: math notation, LaTeX, unicode math symbols, excessive whitespace.
    Keeps: readable English text, numbers, basic punctuation.
    """
    import re
    # Remove LaTeX math blocks
    text = re.sub(r"\$\$.*?\$\$", "[math]", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", "[math]", text)
    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove unicode math symbols that break JSON encoding
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_prompt(filename: str) -> str:
    """Load a prompt file from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text().strip()
    logger.warning("Prompt file not found: %s", path)
    return f"You are a research paper {filename.replace('.md', '')}."
