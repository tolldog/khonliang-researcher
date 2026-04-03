"""Fetch research papers and web content, convert HTML to clean text."""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup


@dataclass
class FetchResult:
    url: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)
    fetched_at: float = field(default_factory=time.time)


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

_HEADERS = {
    "User-Agent": "khonliang-researcher/0.1 (research tool; mailto:noreply@example.com)"
}


def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """Pull an arxiv ID from a URL or bare ID string."""
    m = _ARXIV_ID_RE.search(url_or_id)
    return m.group(0) if m else None


def _html_to_text(html: str) -> tuple[str, str]:
    """Convert HTML to clean text. Returns (title, body_text)."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # For arxiv, try to get the article body specifically
    article = soup.find("article") or soup.find("main") or soup.body or soup
    text = article.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return title, text


def _extract_arxiv_metadata(soup: BeautifulSoup, arxiv_id: str) -> dict:
    """Extract metadata from arxiv HTML page."""
    meta = {"arxiv_id": arxiv_id, "source": "arxiv"}

    # Authors from meta tags
    authors = []
    for tag in soup.find_all("meta", attrs={"name": "citation_author"}):
        if tag.get("content"):
            authors.append(tag["content"])
    if authors:
        meta["authors"] = authors

    # Date
    date_tag = soup.find("meta", attrs={"name": "citation_date"})
    if date_tag and date_tag.get("content"):
        meta["date"] = date_tag["content"]

    return meta


async def fetch_url(url: str, timeout: int = 30) -> FetchResult:
    """Fetch a URL and return clean text content."""
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            html = await resp.text()

    title, content = _html_to_text(html)

    return FetchResult(
        url=url,
        title=title,
        content=content,
        metadata={"source": "web"},
    )


async def fetch_arxiv(arxiv_id_or_url: str) -> FetchResult:
    """Fetch an arxiv paper by ID or URL. Prefers HTML version."""
    arxiv_id = extract_arxiv_id(arxiv_id_or_url)
    if not arxiv_id:
        raise ValueError(f"Cannot extract arxiv ID from: {arxiv_id_or_url}")

    url = f"https://arxiv.org/html/{arxiv_id}"

    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 404:
                # Fall back to abstract page
                url = f"https://arxiv.org/abs/{arxiv_id}"
                async with session.get(url) as abs_resp:
                    abs_resp.raise_for_status()
                    html = await abs_resp.text()
            else:
                resp.raise_for_status()
                html = await resp.text()

    soup = BeautifulSoup(html, "html.parser")
    title, content = _html_to_text(html)
    metadata = _extract_arxiv_metadata(soup, arxiv_id)

    return FetchResult(
        url=url,
        title=title,
        content=content,
        metadata=metadata,
    )


@dataclass
class SearchResult:
    arxiv_id: str
    title: str
    authors: list
    abstract: str
    url: str


async def search_arxiv(
    query: str, max_results: int = 20, sort_by: str = "relevance"
) -> list[SearchResult]:
    """Search arxiv for papers matching keywords.

    Uses the arxiv API (Atom feed). Returns structured results.
    sort_by: "relevance" or "lastUpdatedDate" or "submittedDate"
    """
    import urllib.parse
    import xml.etree.ElementTree as ET

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    api_url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
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

        arxiv_id = extract_arxiv_id(entry_url) or ""

        results.append(SearchResult(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry_url,
        ))

    return results


async def fetch_raw(url: str, timeout: int = 30) -> str:
    """Fetch raw text content from a URL (for markdown files, READMEs, etc.)."""
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            return await resp.text()
