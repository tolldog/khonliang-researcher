"""Fetch research papers and web content in multiple formats.

Supported formats:
  - HTML: stripped to clean text via BeautifulSoup
  - PDF: text extraction via PyMuPDF (fitz)
  - Markdown: preserved as-is (already text)
  - Plain text: passed through

Format is auto-detected from Content-Type header and URL extension.
"""

import io
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ContentFormat(str, Enum):
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class FetchResult:
    url: str
    title: str
    content: str
    format: ContentFormat = ContentFormat.UNKNOWN
    metadata: dict = field(default_factory=dict)
    fetched_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    arxiv_id: str
    title: str
    authors: list
    abstract: str
    url: str


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    # Modern Chrome client-hints + Sec-Fetch-* set. Some Cloudflare-
    # fronted hosts (Substack and others) return 403 to UA strings
    # that aren't accompanied by these headers — the fingerprint
    # mismatch flags the request as scripted (bug_researcher_06622a22).
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Linux"',
    "Connection": "keep-alive",
}


# Hosts known to consistently 403 even with full browser-fingerprint
# headers (Cloudflare interactive challenge, custom anti-bot, etc.).
# Surfacing a hint at fetch time saves the caller from retrying with
# the same request shape and points them at the WebFetch fallback the
# dogfood that filed bug_researcher_06622a22 used to work around it.
_KNOWN_BLOCKED_HOSTS = frozenset({
    "substack.com",
})


def _is_known_blocked_host(host: str) -> bool:
    """True if ``host`` (or its parent suffix) is a known anti-bot source."""
    h = host.lower()
    return any(h == known or h.endswith("." + known) for known in _KNOWN_BLOCKED_HOSTS)


class FetchBlockedError(RuntimeError):
    """The remote rejected the request despite a browser fingerprint.

    Raised on 403 (and analogous bot-challenge statuses) so callers can
    distinguish "site refuses scripted access" from generic transport
    failures and fall back to a browser-driven fetcher (WebFetch, etc.)
    piped into ``ingest_file``.
    """


_LINKEDIN_REDIRECT_PAGE_KEY = "d_shortlink_frontend_external_link_redirect_interstitial"
_MAX_SHORTLINK_REDIRECTS = 3


def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """Pull an arxiv ID from a URL or bare ID string."""
    m = _ARXIV_ID_RE.search(url_or_id)
    return m.group(0) if m else None


def _extract_linkedin_external_url(html: str, source_url: str = "") -> Optional[str]:
    """Extract the real target from LinkedIn's shortlink interstitial page."""
    soup = BeautifulSoup(html, "html.parser")
    source_host = urlparse(source_url).netloc.lower()
    is_linkedin_source = source_host in {"lnkd.in", "linkedin.com", "www.linkedin.com"}

    page_key = soup.find("meta", attrs={"name": "pageKey"})
    is_redirect = (
        page_key is not None
        and page_key.get("content") == _LINKEDIN_REDIRECT_PAGE_KEY
    )

    link = soup.find("a", attrs={"data-tracking-control-name": "external_url_click"})
    if not link or not link.get("href"):
        return None

    target = str(link["href"]).strip()
    parsed = urlparse(target)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    if is_redirect or is_linkedin_source:
        return target
    return None


def _record_shortlink_resolution(metadata: dict, source_url: str) -> None:
    """Record the first caller URL and any intermediate shortlink hops."""
    existing_chain = metadata.get("resolved_chain")
    if isinstance(existing_chain, list):
        chain = [str(item) for item in existing_chain]
    else:
        existing_from = metadata.get("resolved_from")
        chain = [str(existing_from)] if existing_from else []
    metadata["resolved_from"] = source_url
    metadata["resolved_chain"] = [source_url, *chain]
    metadata["shortlink_resolver"] = "linkedin_external_interstitial"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(url: str, content_type: str = "") -> ContentFormat:
    """Detect content format from URL extension and Content-Type header.

    URL extension wins for ambiguous content-types (e.g., text/plain for .md files).
    """
    ct = content_type.lower().split(";")[0].strip()
    url_path = url.lower().split("?")[0].rstrip("/")

    # URL extension is most reliable (servers often miscategorize)
    if url_path.endswith(".pdf"):
        return ContentFormat.PDF
    if url_path.endswith(".md"):
        return ContentFormat.MARKDOWN
    if url_path.endswith((".html", ".htm")) or "/html/" in url_path:
        return ContentFormat.HTML
    if url_path.endswith(".txt"):
        return ContentFormat.TEXT

    # Content-Type for URLs without clear extensions
    if "pdf" in ct:
        return ContentFormat.PDF
    if "html" in ct:
        return ContentFormat.HTML
    if "markdown" in ct or "text/x-markdown" in ct:
        return ContentFormat.MARKDOWN
    if ct == "text/plain":
        return ContentFormat.TEXT

    # Default to HTML (most web pages)
    return ContentFormat.HTML


# ---------------------------------------------------------------------------
# Format-specific converters
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> tuple[str, str]:
    """Convert HTML to clean text. Returns (title, body_text)."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    article = soup.find("article") or soup.find("main") or soup.body or soup
    text = article.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return title, text


def _pdf_to_text(pdf_bytes: bytes) -> tuple[str, str]:
    """Extract text from PDF bytes. Returns (title, body_text)."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    title = doc.metadata.get("title", "") or ""

    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()

    body = "\n\n".join(pages)
    body = re.sub(r"\n{3,}", "\n\n", body)

    # If no metadata title, try first line
    if not title and body:
        first_line = body.split("\n")[0].strip()
        if len(first_line) < 200:
            title = first_line

    return title, body


def _markdown_to_text(md: str) -> tuple[str, str]:
    """Extract title and return markdown as-is (it's already readable text)."""
    title = ""
    for line in md.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            break

    return title, md


def _extract_arxiv_metadata(html: str, arxiv_id: str) -> dict:
    """Extract metadata from arxiv HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    meta = {"arxiv_id": arxiv_id, "source": "arxiv"}

    authors = []
    for tag in soup.find_all("meta", attrs={"name": "citation_author"}):
        if tag.get("content"):
            authors.append(tag["content"])
    if authors:
        meta["authors"] = authors

    date_tag = soup.find("meta", attrs={"name": "citation_date"})
    if date_tag and date_tag.get("content"):
        meta["date"] = date_tag["content"]

    return meta


# ---------------------------------------------------------------------------
# Convert raw response to text based on format
# ---------------------------------------------------------------------------

def _convert(raw: bytes, text: str, fmt: ContentFormat) -> tuple[str, str]:
    """Convert fetched content to (title, text) based on detected format."""
    if fmt == ContentFormat.PDF:
        return _pdf_to_text(raw)
    elif fmt == ContentFormat.HTML:
        return _html_to_text(text)
    elif fmt == ContentFormat.MARKDOWN:
        return _markdown_to_text(text)
    else:
        # Plain text
        title = ""
        first_line = text.split("\n")[0].strip() if text else ""
        if len(first_line) < 200:
            title = first_line
        return title, text


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

async def fetch_url(
    url: str,
    timeout: int = 60,
    *,
    _shortlink_depth: int = 0,
) -> FetchResult:
    """Fetch any URL, auto-detect format, return clean text.

    Handles HTML, PDF, Markdown, and plain text automatically. Raises
    :class:`FetchBlockedError` on 403 (and on known-blocked hosts that
    return any non-2xx) so callers can fall back to a browser-driven
    fetcher rather than retrying the same request.
    """
    host = urlparse(url).netloc.lower()
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 403 or (
                _is_known_blocked_host(host) and resp.status >= 400
            ):
                raise FetchBlockedError(
                    f"{url} returned {resp.status} despite browser headers; "
                    "this host is known anti-bot. Fall back to WebFetch (or a "
                    "browser-driven fetcher) and pipe the result via ingest_file."
                )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            fmt = _detect_format(url, content_type)

            if fmt == ContentFormat.PDF:
                raw = await resp.read()
                text_content = ""
            else:
                raw = b""
                text_content = await resp.text()

    if fmt == ContentFormat.HTML and _shortlink_depth < _MAX_SHORTLINK_REDIRECTS:
        target_url = _extract_linkedin_external_url(text_content, source_url=url)
        if target_url and target_url != url:
            logger.info("Resolved LinkedIn shortlink %s -> %s", url, target_url)
            result = await fetch_url(
                target_url,
                timeout=timeout,
                _shortlink_depth=_shortlink_depth + 1,
            )
            _record_shortlink_resolution(result.metadata, url)
            return result

    title, content = _convert(raw, text_content, fmt)

    return FetchResult(
        url=url,
        title=title,
        content=content,
        format=fmt,
        metadata={"source": "web", "content_type": content_type},
    )


async def fetch_arxiv(arxiv_id_or_url: str) -> FetchResult:
    """Fetch an arxiv paper by ID or URL.

    Tries formats in order: HTML → PDF → abstract page.
    """
    arxiv_id = extract_arxiv_id(arxiv_id_or_url)
    if not arxiv_id:
        raise ValueError(f"Cannot extract arxiv ID from: {arxiv_id_or_url}")

    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        # Try HTML first
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        async with session.get(html_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                html = await resp.text()
                title, content = _html_to_text(html)
                metadata = _extract_arxiv_metadata(html, arxiv_id)
                metadata["format_used"] = "html"
                return FetchResult(
                    url=html_url, title=title, content=content,
                    format=ContentFormat.HTML, metadata=metadata,
                )

        # Fall back to PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        logger.info("No HTML for %s, trying PDF", arxiv_id)
        async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" in content_type.lower():
                    pdf_bytes = await resp.read()
                    title, content = _pdf_to_text(pdf_bytes)
                    return FetchResult(
                        url=pdf_url, title=title, content=content,
                        format=ContentFormat.PDF,
                        metadata={"arxiv_id": arxiv_id, "source": "arxiv", "format_used": "pdf"},
                    )

        # Last resort: abstract page
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        logger.info("No PDF for %s, falling back to abstract", arxiv_id)
        async with session.get(abs_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            html = await resp.text()
            title, content = _html_to_text(html)
            metadata = _extract_arxiv_metadata(html, arxiv_id)
            metadata["format_used"] = "abstract"
            return FetchResult(
                url=abs_url, title=title, content=content,
                format=ContentFormat.HTML, metadata=metadata,
            )


async def fetch_raw(url: str, timeout: int = 30) -> str:
    """Fetch raw text content from a URL (for markdown files, READMEs, etc.)."""
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            return await resp.text()


async def fetch_file(path: str) -> FetchResult:
    """Read a local file (PDF, markdown, text, HTML)."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = _detect_format(path, "")

    if fmt == ContentFormat.PDF:
        raw = p.read_bytes()
        text_content = ""
    else:
        raw = b""
        text_content = p.read_text(errors="replace")

    title, content = _convert(raw, text_content, fmt)

    return FetchResult(
        url=f"file://{p.resolve()}",
        title=title or p.stem,
        content=content,
        format=fmt,
        metadata={"source": "file", "path": str(p.resolve())},
    )


# ---------------------------------------------------------------------------
# Arxiv search (kept for backward compat)
# ---------------------------------------------------------------------------

async def search_arxiv(
    query: str, max_results: int = 20, sort_by: str = "relevance"
) -> list[SearchResult]:
    """Search arxiv for papers matching keywords."""
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
