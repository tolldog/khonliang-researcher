"""Research workers that plug into khonliang's ResearchPool.

Two researchers:
  - PaperFetcher: fetches and ingests individual papers
  - ListParser: fetches awesome-lists and extracts paper references

Both run in the ResearchPool's managed thread pool with per-researcher
concurrency limits and rate limiting.
"""

import logging
from typing import Any, Dict, List

from khonliang.research.base import BaseResearcher
from khonliang.research.models import ResearchResult, ResearchTask

from researcher.fetcher import fetch_arxiv, fetch_url, extract_arxiv_id
from researcher.parser import parse_paper_list

logger = logging.getLogger(__name__)


class PaperFetcher(BaseResearcher):
    """Fetch a paper by URL, convert to text, return content for ingestion."""

    name = "paper_fetcher"
    capabilities = ["fetch_paper"]
    max_concurrent = 5

    async def research(self, task: ResearchTask) -> ResearchResult:
        url = task.query

        # Use arxiv-specific fetcher if applicable
        arxiv_id = extract_arxiv_id(url)
        if arxiv_id:
            result = await fetch_arxiv(url)
        else:
            result = await fetch_url(url)

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=result.title,
            content=result.content,
            sources=[result.url],
            metadata=result.metadata,
            scope=task.scope,
        )


class ListParser(BaseResearcher):
    """Fetch an awesome-list URL, use LLM to extract paper references."""

    name = "list_parser"
    capabilities = ["parse_list"]
    max_concurrent = 2

    def __init__(self, llm_client=None):
        super().__init__()
        self._client = llm_client

    async def research(self, task: ResearchTask) -> ResearchResult:
        from researcher.fetcher import fetch_raw

        url = task.query
        raw_text = await fetch_raw(url)

        if self._client:
            papers = await parse_paper_list(raw_text, self._client)
        else:
            from researcher.parser import _regex_fallback
            papers = _regex_fallback(raw_text)

        # Return structured list as content
        import json
        paper_dicts = [
            {"title": p.title, "url": p.url, "category": p.category}
            for p in papers
        ]

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Paper list from {url}",
            content=json.dumps(paper_dicts, indent=2),
            sources=[url],
            metadata={"paper_count": len(papers)},
            scope=task.scope,
        )
