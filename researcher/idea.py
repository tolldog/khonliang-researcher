"""LLM role for decomposing informal text into researchable components.

Takes a LinkedIn post, tweet, blog snippet, or freeform thought and
produces structured claims + search queries for the research pipeline.
"""

import logging
from typing import Any, Dict, Optional

from khonliang.roles.base import BaseRole

from researcher.roles import _clean_for_json, _load_prompt

logger = logging.getLogger(__name__)


class IdeaParserRole(BaseRole):
    """Decompose informal text into claims and search queries."""

    def __init__(self, model_pool, **kwargs):
        super().__init__(
            role="idea_parser",
            model_pool=model_pool,
            system_prompt=_load_prompt("idea_parser.md"),
            **kwargs,
        )

    async def handle(
        self, message: str, session_id: str = "", context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        text = _clean_for_json(message[:15_000])
        model = "qwen2.5:7b" if len(message) > 2_000 else "llama3.2:3b"

        prompt = f"Decompose this into researchable claims and search queries:\n\n{text}"

        try:
            result = await self.client.generate_json(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.3,
                max_tokens=2000,
                model=model,
            )
            # Normalize output
            if isinstance(result, dict):
                return {
                    "title": result.get("title", "Untitled idea"),
                    "source_type": result.get("source_type", "freeform"),
                    "claims": result.get("claims", []),
                    "search_queries": result.get("search_queries", []),
                    "keywords": result.get("keywords", []),
                    "success": True,
                }
            logger.warning("Unexpected idea parser output: %s", type(result))
            return {"success": False, "error": "Unexpected output format"}
        except Exception as e:
            logger.error("Idea parsing failed with %s: %s", model, e)
            return {"success": False, "error": str(e)}
