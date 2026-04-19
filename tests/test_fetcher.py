import asyncio

import pytest

from researcher import fetcher
from researcher.fetcher import ContentFormat


LINKEDIN_INTERSTITIAL = """
<html>
  <head>
    <meta name="pageKey" content="d_shortlink_frontend_external_link_redirect_interstitial">
  </head>
  <body>
    <a data-tracking-control-name="external_url_click"
       href="https://arxiv.org/pdf/2511.19699">Continue</a>
  </body>
</html>
"""


def test_extract_linkedin_external_url():
    assert (
        fetcher._extract_linkedin_external_url(
            LINKEDIN_INTERSTITIAL,
            source_url="https://lnkd.in/guZ5SMq3",
        )
        == "https://arxiv.org/pdf/2511.19699"
    )


def test_non_linkedin_source_does_not_resolve_tracking_link_without_page_key():
    html = """
    <html>
      <body>
        <a data-tracking-control-name="external_url_click"
           href="https://example.com/not-a-shortlink">Continue</a>
      </body>
    </html>
    """

    assert (
        fetcher._extract_linkedin_external_url(
            html,
            source_url="https://example.org/page-with-lnkd.in-text",
        )
        is None
    )


@pytest.mark.parametrize("href", ["javascript:alert(1)", "/relative/path", "https://"])
def test_linkedin_external_url_rejects_invalid_hrefs(href):
    html = LINKEDIN_INTERSTITIAL.replace("https://arxiv.org/pdf/2511.19699", href)

    assert (
        fetcher._extract_linkedin_external_url(
            html,
            source_url="https://lnkd.in/guZ5SMq3",
        )
        is None
    )


def test_fetch_url_resolves_linkedin_shortlink(monkeypatch):
    requests = []
    responses = {
        "https://lnkd.in/guZ5SMq3": (
            "text/html; charset=utf-8",
            LINKEDIN_INTERSTITIAL,
        ),
        "https://arxiv.org/pdf/2511.19699": (
            "text/plain",
            "A Layered Protocol Architecture for the Internet of Agents\nBody",
        ),
    }

    class FakeResponse:
        def __init__(self, url):
            self.headers = {"Content-Type": responses[url][0]}
            self._text = responses[url][1]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def text(self):
            return self._text

        async def read(self):
            return self._text.encode()

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            requests.append(url)
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    result = asyncio.run(fetcher.fetch_url("https://lnkd.in/guZ5SMq3"))

    assert requests == [
        "https://lnkd.in/guZ5SMq3",
        "https://arxiv.org/pdf/2511.19699",
    ]
    assert result.url == "https://arxiv.org/pdf/2511.19699"
    assert result.format == ContentFormat.TEXT
    assert result.metadata["resolved_from"] == "https://lnkd.in/guZ5SMq3"
    assert result.metadata["resolved_chain"] == ["https://lnkd.in/guZ5SMq3"]
    assert result.metadata["shortlink_resolver"] == "linkedin_external_interstitial"


def test_fetch_url_preserves_multi_hop_resolution_chain(monkeypatch):
    requests = []
    first = LINKEDIN_INTERSTITIAL.replace(
        "https://arxiv.org/pdf/2511.19699",
        "https://lnkd.in/second",
    )
    second = LINKEDIN_INTERSTITIAL.replace(
        "https://arxiv.org/pdf/2511.19699",
        "https://example.com/final.txt",
    )
    responses = {
        "https://lnkd.in/first": ("text/html", first),
        "https://lnkd.in/second": ("text/html", second),
        "https://example.com/final.txt": ("text/plain", "Final paper\nBody"),
    }

    class FakeResponse:
        def __init__(self, url):
            self.headers = {"Content-Type": responses[url][0]}
            self._text = responses[url][1]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def text(self):
            return self._text

        async def read(self):
            return self._text.encode()

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            requests.append(url)
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    result = asyncio.run(fetcher.fetch_url("https://lnkd.in/first"))

    assert requests == [
        "https://lnkd.in/first",
        "https://lnkd.in/second",
        "https://example.com/final.txt",
    ]
    assert result.url == "https://example.com/final.txt"
    assert result.metadata["resolved_from"] == "https://lnkd.in/first"
    assert result.metadata["resolved_chain"] == [
        "https://lnkd.in/first",
        "https://lnkd.in/second",
    ]


def test_fetch_url_stops_at_shortlink_redirect_cap(monkeypatch):
    requests = []

    class FakeResponse:
        headers = {"Content-Type": "text/html"}

        def __init__(self, url):
            current = int(url.rsplit("/", 1)[1])
            self._html = LINKEDIN_INTERSTITIAL.replace(
                "https://arxiv.org/pdf/2511.19699",
                f"https://lnkd.in/{current + 1}",
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def text(self):
            return self._html

        async def read(self):
            return self._html.encode()

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            requests.append(url)
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    result = asyncio.run(fetcher.fetch_url("https://lnkd.in/0"))

    assert requests == [
        f"https://lnkd.in/{index}"
        for index in range(fetcher._MAX_SHORTLINK_REDIRECTS + 1)
    ]
    assert result.url == f"https://lnkd.in/{fetcher._MAX_SHORTLINK_REDIRECTS}"
    assert result.format == ContentFormat.HTML
