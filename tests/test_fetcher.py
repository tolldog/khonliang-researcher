import asyncio

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
        fetcher._extract_linkedin_external_url(LINKEDIN_INTERSTITIAL)
        == "https://arxiv.org/pdf/2511.19699"
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
    assert result.metadata["shortlink_resolver"] == "linkedin_external_interstitial"
