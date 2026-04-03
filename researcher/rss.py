"""RSS/Atom feed reader for research blogs.

Fetches and parses RSS/Atom feeds from AI research blogs, returns
entries as EngineResults for integration with the search pipeline.

Pre-configured feeds for major AI companies. Add custom feeds via config.

Usage:
    # Fetch all configured feeds
    entries = await fetch_all_feeds()

    # Fetch specific feeds
    entries = await fetch_feeds(["anthropic", "ollama"])

    # Search across cached feed entries
    results = search_feed_cache("multi-agent tool use")
"""

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from khonliang.research.engine import BaseEngine, EngineResult

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "khonliang-researcher/0.1 (research tool; mailto:noreply@example.com)"
}


@dataclass
class FeedConfig:
    name: str
    url: str
    source: str  # tag for EngineResult.source


# Pre-configured feeds for AI research blogs
DEFAULT_FEEDS: Dict[str, FeedConfig] = {
    # Core Agent + LLM
    "anthropic": FeedConfig(
        name="Anthropic Engineering",
        url="https://www.anthropic.com/engineering/rss.xml",
        source="anthropic",
    ),
    "openai": FeedConfig(
        name="OpenAI Blog",
        url="https://openai.com/blog/rss.xml",
        source="openai",
    ),
    "openai_dev": FeedConfig(
        name="OpenAI Developers",
        url="https://developers.openai.com/blog/rss.xml",
        source="openai",
    ),
    # MCP + Agent Protocols
    "mcp": FeedConfig(
        name="Model Context Protocol Blog",
        url="https://blog.modelcontextprotocol.io/rss.xml",
        source="mcp",
    ),
    # Local / Open Model Ecosystem
    "ollama": FeedConfig(
        name="Ollama Blog",
        url="https://ollama.com/blog/rss.xml",
        source="ollama",
    ),
    "huggingface": FeedConfig(
        name="Hugging Face Blog",
        url="https://huggingface.co/blog/feed.xml",
        source="huggingface",
    ),
    # Agent Frameworks / Orchestration
    "langchain": FeedConfig(
        name="LangChain Blog",
        url="https://blog.langchain.com/rss/",
        source="langchain",
    ),
    # Inference + Serving
    "vllm": FeedConfig(
        name="vLLM Blog",
        url="https://vllm.ai/blog/rss.xml",
        source="vllm",
    ),
    # Enterprise / Applied Agents
    "microsoft": FeedConfig(
        name="Microsoft AI / Agent Dev Blog",
        url="https://devblogs.microsoft.com/feed/",
        source="microsoft",
    ),
    # Infra + Scaling
    "nvidia": FeedConfig(
        name="NVIDIA Developer Blog",
        url="https://developer.nvidia.com/blog/feed/",
        source="nvidia",
    ),
}


def load_opml(path: str) -> Dict[str, FeedConfig]:
    """Load feeds from an OPML file. Handles nested category outlines."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}

    tree = ET.parse(p)
    root = tree.getroot()
    feeds = {}

    for outline in root.iter("outline"):
        url = outline.get("xmlUrl", "")
        name = outline.get("text", "")
        if url and name:
            # Find parent category if nested
            category = ""
            for parent in root.iter("outline"):
                if outline in list(parent):
                    cat = parent.get("text", "")
                    if cat and not parent.get("xmlUrl"):
                        category = cat
                        break

            key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30]
            feeds[key] = FeedConfig(name=name, url=url, source=key)

    return feeds


def _parse_feed(xml_text: str, source: str) -> List[EngineResult]:
    """Parse RSS or Atom feed XML into EngineResults."""
    results = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("Failed to parse feed XML for %s", source)
        return []

    # Detect feed type
    ns = {}
    if root.tag.startswith("{http://www.w3.org/2005/Atom"):
        # Atom feed
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        for entry in entries:
            title = _text(entry.find("atom:title", ns))
            summary = _text(entry.find("atom:summary", ns)) or _text(entry.find("atom:content", ns))
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            if link_el is None:
                link_el = entry.find("atom:link", ns)
            url = link_el.get("href", "") if link_el is not None else ""
            published = _text(entry.find("atom:published", ns)) or _text(entry.find("atom:updated", ns))

            results.append(EngineResult(
                title=title,
                content=_strip_html(summary)[:500] if summary else "",
                url=url,
                source=source,
                metadata={"published": published},
            ))
    else:
        # RSS 2.0 feed
        channel = root.find("channel")
        if channel is None:
            channel = root
        for item in channel.findall("item"):
            title = _text(item.find("title"))
            desc = _text(item.find("description"))
            url = _text(item.find("link"))
            pub_date = _text(item.find("pubDate"))

            results.append(EngineResult(
                title=title,
                content=_strip_html(desc)[:500] if desc else "",
                url=url,
                source=source,
                metadata={"published": pub_date},
            ))

    return results


def _text(el) -> str:
    """Extract text from an XML element."""
    if el is None:
        return ""
    return (el.text or "").strip()


def _strip_html(text: str) -> str:
    """Rough HTML tag removal for feed descriptions."""
    return re.sub(r"<[^>]+>", "", text).strip()


class RSSEngine(BaseEngine):
    """Search engine that fetches and searches RSS/Atom feeds.

    Loads feeds from OPML file if available, falls back to DEFAULT_FEEDS.
    """

    name = "rss"
    max_threads = 2
    rate_limit = 1.0
    timeout = 15.0

    def __init__(self, feeds: Optional[Dict[str, FeedConfig]] = None, opml_path: Optional[str] = None):
        super().__init__()
        if feeds:
            self.feeds = feeds
        elif opml_path:
            self.feeds = load_opml(opml_path) or DEFAULT_FEEDS
        else:
            self.feeds = DEFAULT_FEEDS
        self._cache: List[EngineResult] = []
        self._cache_time: float = 0
        self._cache_ttl: float = 3600  # 1 hour

    async def execute(self, query: str, **kwargs: Any) -> List[EngineResult]:
        """Search RSS feeds for entries matching the query.

        Fetches feeds if cache is stale, then filters by keyword match.
        """
        feed_names = kwargs.get("feeds")  # Optional: only search specific feeds
        max_results = kwargs.get("max_results", 20)

        # Refresh cache if stale
        if time.monotonic() - self._cache_time > self._cache_ttl:
            await self._refresh_cache(feed_names)

        # Simple keyword search across cached entries
        query_lower = query.lower()
        keywords = query_lower.split()
        results = []

        for entry in self._cache:
            text = f"{entry.title} {entry.content}".lower()
            score = sum(1 for kw in keywords if kw in text) / len(keywords)
            if score > 0.3:  # At least 30% of keywords match
                entry.score = score
                results.append(entry)

        results.sort(key=lambda r: -r.score)
        return results[:max_results]

    async def _refresh_cache(self, feed_names: Optional[List[str]] = None):
        """Fetch all configured feeds and update cache."""
        feeds_to_fetch = self.feeds
        if feed_names:
            feeds_to_fetch = {k: v for k, v in self.feeds.items() if k in feed_names}

        all_entries = []
        async with aiohttp.ClientSession(headers=_HEADERS) as session:
            tasks = [
                self._fetch_feed(session, cfg)
                for cfg in feeds_to_fetch.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    all_entries.extend(result)
                elif isinstance(result, Exception):
                    logger.warning("Feed fetch failed: %s", result)

        self._cache = all_entries
        self._cache_time = time.monotonic()
        logger.info("RSS cache refreshed: %d entries from %d feeds",
                     len(all_entries), len(feeds_to_fetch))

    async def _fetch_feed(self, session: aiohttp.ClientSession, cfg: FeedConfig) -> List[EngineResult]:
        """Fetch and parse a single feed."""
        try:
            async with session.get(cfg.url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning("Feed %s returned %d", cfg.name, resp.status)
                    return []
                xml_text = await resp.text()
                entries = _parse_feed(xml_text, cfg.source)
                logger.debug("Feed %s: %d entries", cfg.name, len(entries))
                return entries
        except Exception as e:
            logger.warning("Feed %s error: %s", cfg.name, e)
            return []


async def fetch_all_feeds(
    feeds: Optional[List[str]] = None,
    opml_path: Optional[str] = None,
) -> List[EngineResult]:
    """Fetch and return all entries from configured RSS feeds.

    Args:
        feeds: Optional list of feed names to fetch (default: all)
        opml_path: Optional path to OPML file (default: feeds.opml)

    Returns:
        All feed entries as EngineResults.
    """
    if opml_path is None:
        from pathlib import Path
        default_opml = Path("feeds.opml")
        if default_opml.exists():
            opml_path = str(default_opml)

    engine = RSSEngine(opml_path=opml_path)
    await engine._refresh_cache(feeds)
    return engine._cache


async def fetch_feed_urls(feeds: Optional[List[str]] = None) -> List[str]:
    """Fetch feeds and return just the URLs for batch ingestion."""
    entries = await fetch_all_feeds(feeds)
    return [e.url for e in entries if e.url]
