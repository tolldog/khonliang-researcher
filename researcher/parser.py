"""LLM-powered parser for extracting paper references from arbitrary text.

Handles: markdown awesome-lists, plain URL lists, mixed text+URLs,
HTML pages with links, etc. Uses khonliang OllamaClient to understand
the content rather than brittle regex.
"""

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PARSER_SYSTEM_PROMPT = """\
You extract research paper references from text. The input may be:
- A GitHub awesome-list (markdown with [title](url) links)
- A plain list of URLs
- Mixed text with embedded URLs
- An HTML page with links
- Any other format containing paper references

Extract ALL paper references you can find. For each paper, provide:
- title: the paper title (infer from link text, URL, or surrounding context)
- url: the URL to the paper (prefer arxiv.org links)
- category: the section/topic it appeared under (if any)

Output ONLY a JSON array:
[{"title": "Paper Title", "url": "https://arxiv.org/abs/1234.56789", "category": "Section Name"}]

Rules:
- Include ALL papers, even if you're unsure about the title
- Prefer arxiv URLs when multiple URLs exist for the same paper
- If a URL is not arxiv but points to a paper, include it anyway
- Category should be the nearest section heading or topic grouping
- If no category is apparent, use "uncategorized"
- Do NOT include non-paper links (GitHub repos, tools, blog posts without papers)
"""


@dataclass
class PaperReference:
    title: str
    url: str
    category: str = "uncategorized"


async def parse_paper_list(text: str, client) -> list[PaperReference]:
    """Use LLM to extract paper references from arbitrary text.

    Args:
        text: Raw text content (markdown, HTML, plain text, etc.)
        client: khonliang LLMClient (OllamaClient or compatible)

    Returns:
        List of PaperReference with title, URL, and category.
    """
    # Truncate very long content to avoid overwhelming the model
    if len(text) > 50_000:
        text = text[:50_000] + "\n\n[TRUNCATED — content too long]"

    try:
        result = await client.generate_json(
            prompt=f"Extract paper references from the following text:\n\n{text}",
            system=_PARSER_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=8000,
        )
    except Exception:
        logger.exception("LLM parsing failed, falling back to regex")
        return _regex_fallback(text)

    if isinstance(result, list):
        refs = result
    elif isinstance(result, dict) and "papers" in result:
        refs = result["papers"]
    else:
        logger.warning("Unexpected LLM output format: %s", type(result))
        return _regex_fallback(text)

    papers = []
    for item in refs:
        if isinstance(item, dict) and item.get("url"):
            papers.append(PaperReference(
                title=item.get("title", ""),
                url=item["url"],
                category=item.get("category", "uncategorized"),
            ))

    logger.info("LLM extracted %d paper references", len(papers))
    return papers


def _regex_fallback(text: str) -> list[PaperReference]:
    """Best-effort regex extraction when LLM is unavailable."""
    papers = []
    seen_urls = set()

    # Markdown links: [title](url)
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text):
        title, url = m.group(1), m.group(2)
        if "arxiv.org" in url and url not in seen_urls:
            seen_urls.add(url)
            papers.append(PaperReference(title=title, url=url))

    # Bare arxiv URLs
    for m in re.finditer(r"(https?://arxiv\.org/(?:abs|html|pdf)/[\d.]+(?:v\d+)?)", text):
        url = m.group(1)
        if url not in seen_urls:
            seen_urls.add(url)
            papers.append(PaperReference(title="", url=url))

    logger.info("Regex fallback extracted %d paper references", len(papers))
    return papers
