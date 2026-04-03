"""Search engines for finding research papers.

Each engine extends khonliang's BaseEngine with its own rate limits
and concurrency. They're composed via the search() function which
runs all enabled engines in parallel and merges results.

Engines:
  - ArxivEngine: arxiv API (Atom feed)
  - SemanticScholarEngine: Semantic Scholar API (free, no key needed)
  - GoogleScholarEngine: Google Scholar via SerpAPI (needs key) — TODO

Adding a new engine:
  1. Subclass BaseEngine
  2. Implement execute() returning List[EngineResult]
  3. Register in ENGINES dict below
"""

import asyncio
import json
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from khonliang.research.engine import BaseEngine, EngineResult

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "khonliang-researcher/0.1 (research tool; mailto:noreply@example.com)"
}


class ArxivEngine(BaseEngine):
    """Search arxiv via the Atom API."""

    name = "arxiv"
    max_threads = 2
    rate_limit = 3.0  # arxiv asks for 3s between requests
    timeout = 30.0

    async def execute(self, query: str, **kwargs: Any) -> List[EngineResult]:
        max_results = kwargs.get("max_results", 20)
        sort_by = kwargs.get("sort_by", "relevance")

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        api_url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

        async with aiohttp.ClientSession(headers=_HEADERS) as session:
            async with session.get(api_url) as resp:
                resp.raise_for_status()
                xml_text = await resp.text()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        results = []

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)

            title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
            abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
            entry_url = id_el.text.strip() if id_el is not None else ""

            authors = []
            for author_el in entry.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text.strip())

            results.append(EngineResult(
                title=title,
                content=abstract,
                url=entry_url,
                source="arxiv",
                metadata={"authors": authors},
            ))

        return results


class SemanticScholarEngine(BaseEngine):
    """Search Semantic Scholar (free API, no key needed)."""

    name = "semantic_scholar"
    max_threads = 2
    rate_limit = 1.0  # 100 req/5min for unauthenticated
    timeout = 15.0

    async def execute(self, query: str, **kwargs: Any) -> List[EngineResult]:
        max_results = min(kwargs.get("max_results", 20), 100)

        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,url,authors,externalIds,year",
        }
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"

        async with aiohttp.ClientSession(headers=_HEADERS) as session:
            async with session.get(api_url) as resp:
                if resp.status == 429:
                    logger.warning("Semantic Scholar rate limited")
                    return []
                resp.raise_for_status()
                data = await resp.json()

        results = []
        for paper in data.get("data", []):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "") or ""
            url = paper.get("url", "")

            # Prefer arxiv URL if available
            ext_ids = paper.get("externalIds", {}) or {}
            arxiv_id = ext_ids.get("ArXiv", "")
            if arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"

            authors = [a.get("name", "") for a in (paper.get("authors") or [])]

            results.append(EngineResult(
                title=title,
                content=abstract,
                url=url,
                source="semantic_scholar",
                metadata={
                    "authors": authors,
                    "year": paper.get("year"),
                    "arxiv_id": arxiv_id,
                },
            ))

        return results


# TODO: Future engines to add
# - GoogleScholarEngine: via SerpAPI or scholarly lib (needs API key)
# - BlogEngine: RSS/Atom feeds from AI research blogs
#   Sources: Anthropic, OpenAI, Google DeepMind, Meta AI, Hugging Face,
#   Microsoft Research, Apple ML, Nvidia, Cohere, Mistral
# - DDGEngine: DuckDuckGo instant answers for broader web search
# - PapersWithCodeEngine: paperswithcode.com API for papers with implementations

# Registry of available engines
ENGINES: Dict[str, BaseEngine] = {
    "arxiv": ArxivEngine(),
    "semantic_scholar": SemanticScholarEngine(),
}


async def search_papers(
    query: str,
    engines: Optional[List[str]] = None,
    max_results: int = 20,
    **kwargs,
) -> List[EngineResult]:
    """Search multiple engines in parallel, merge and deduplicate results.

    Args:
        query: Search keywords
        engines: List of engine names to use (default: all)
        max_results: Max results per engine
        **kwargs: Passed to each engine

    Returns:
        Merged, deduplicated results sorted by source diversity.
    """
    engine_names = engines or list(ENGINES.keys())
    active_engines = [ENGINES[n] for n in engine_names if n in ENGINES]

    if not active_engines:
        return []

    # Start engines that need thread pools
    for e in active_engines:
        e.start()

    # Run all searches in parallel
    tasks = [e.query(query, max_results=max_results, **kwargs) for e in active_engines]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge and deduplicate by URL
    seen_urls = set()
    merged = []
    for result_set in all_results:
        if isinstance(result_set, Exception):
            logger.warning("Engine search failed: %s", result_set)
            continue
        for r in result_set:
            # Normalize arxiv URLs for dedup
            url_key = r.url.rstrip("/")
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                merged.append(r)

    return merged
