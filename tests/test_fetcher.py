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


@pytest.mark.asyncio
async def test_fetch_file_reads_local_file_off_the_loop(tmp_path):
    # fetch_file now runs the blocking read+convert via asyncio.to_thread
    # (fr_researcher_e32d9bb7); the end-to-end result must be unchanged.
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nthe body text")
    result = await fetcher.fetch_file(str(f))
    assert result.format == ContentFormat.MARKDOWN
    assert "the body text" in result.content
    assert result.url.startswith("file://")


@pytest.mark.asyncio
async def test_fetch_file_missing_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        await fetcher.fetch_file(str(tmp_path / "nope.md"))


def test_detect_format_pdf_url_served_as_html_is_html():
    # A .pdf URL that the server explicitly serves as HTML is an error/
    # challenge page, not a PDF — must not be routed to the PDF parser.
    assert (
        fetcher._detect_format("https://x/paper.pdf", "text/html; charset=utf-8")
        == ContentFormat.HTML
    )


def test_detect_format_pdf_extension_wins_for_ambiguous_content_type():
    # The legitimate "server miscategorizes a real PDF" case is preserved:
    # ambiguous/empty/text-plain Content-Type on a .pdf URL stays PDF.
    assert fetcher._detect_format("https://x/paper.pdf", "") == ContentFormat.PDF
    assert (
        fetcher._detect_format("https://x/paper.pdf", "text/plain")
        == ContentFormat.PDF
    )
    assert (
        fetcher._detect_format("https://x/paper.pdf", "application/pdf")
        == ContentFormat.PDF
    )


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
            self.status = 200
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
            self.status = 200
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
        status = 200

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


def test_fetch_url_raises_FetchBlockedError_on_403(monkeypatch):
    """A 403 with browser headers means the host fingerprinted us as a
    bot. Surface a typed error pointing at the WebFetch fallback so the
    caller doesn't retry the same shape and pollute logs.
    """

    class FakeResponse:
        def __init__(self, url):
            self.status = 403
            self.headers = {"Content-Type": "text/html"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):  # pragma: no cover - shouldn't reach
            raise AssertionError("FetchBlockedError must fire before raise_for_status")

        async def text(self):  # pragma: no cover
            return ""

        async def read(self):  # pragma: no cover
            return b""

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    with pytest.raises(fetcher.FetchBlockedError) as ei:
        asyncio.run(fetcher.fetch_url("https://example.com/something"))
    msg = str(ei.value)
    assert "WebFetch" in msg
    assert "403" in msg
    # Generic 403 (host not in known-blocked list) — message must NOT
    # claim the host is anti-bot, since 403 can also be a real ACL deny.
    assert "known-anti-bot list" not in msg
    assert "ACL deny" in msg or "bot challenge" in msg


def test_fetch_url_known_blocked_host_message_calls_out_anti_bot(monkeypatch):
    """The error message for a listed host says so explicitly, so the
    caller knows fingerprint headers won't help and skips the retry.
    """

    class FakeResponse:
        def __init__(self, url):
            self.status = 403
            self.headers = {"Content-Type": "text/html"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):  # pragma: no cover
            raise AssertionError("FetchBlockedError must fire first")

        async def text(self):  # pragma: no cover
            return ""

        async def read(self):  # pragma: no cover
            return b""

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    with pytest.raises(fetcher.FetchBlockedError) as ei:
        asyncio.run(fetcher.fetch_url("https://author.substack.com:443/p/x"))
    msg = str(ei.value)
    assert "known-anti-bot list" in msg
    assert "author.substack.com" in msg


def test_is_known_blocked_host_uses_hostname_not_netloc(monkeypatch):
    """fetch_url must extract host via .hostname so port AND userinfo
    in the URL don't break the suffix match (urlparse(url).netloc keeps
    both; .hostname strips both).
    """

    seen_hosts = []

    real_predicate = fetcher._is_known_blocked_host

    def spy(h):
        seen_hosts.append(h)
        return real_predicate(h)

    monkeypatch.setattr(fetcher, "_is_known_blocked_host", spy)

    class FakeResponse:
        def __init__(self, url):
            self.status = 200
            self.headers = {"Content-Type": "text/plain"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def text(self):
            return "ok"

        async def read(self):
            return b"ok"

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    asyncio.run(fetcher.fetch_url("https://author.substack.com:443/p/x"))
    asyncio.run(fetcher.fetch_url("https://user:pw@author.substack.com/p/x"))

    # Both calls must extract the bare host — port stripped, userinfo
    # stripped — so the substack suffix match still fires.
    assert seen_hosts == ["author.substack.com", "author.substack.com"]


def test_fetch_blocked_error_does_not_leak_userinfo_or_query(monkeypatch):
    """The FetchBlockedError message must not echo userinfo or query
    parameters. URLs with basic-auth or sensitive query strings would
    otherwise leak via any caller that logs the exception.
    """

    class FakeResponse:
        def __init__(self, url):
            self.status = 403
            self.headers = {"Content-Type": "text/html"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):  # pragma: no cover
            raise AssertionError("FetchBlockedError must fire first")

        async def text(self):  # pragma: no cover
            return ""

        async def read(self):  # pragma: no cover
            return b""

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    sensitive = "https://user:supersecret@example.com/path?token=abcd1234&id=42"
    with pytest.raises(fetcher.FetchBlockedError) as ei:
        asyncio.run(fetcher.fetch_url(sensitive))
    msg = str(ei.value)
    assert "supersecret" not in msg
    assert "user" not in msg.split("/")[2] if "://" in msg else True
    assert "abcd1234" not in msg
    assert "token=" not in msg
    # Sanity — host + path are still surfaced for debuggability.
    assert "example.com/path" in msg


def test_fetch_url_raises_for_known_blocked_host_on_any_4xx(monkeypatch):
    """Substack and other listed hosts surface FetchBlockedError on any
    4xx/5xx response — they have a track record of returning 5xx /
    generic 4xx pages from the bot challenge layer too, not always 403.
    """

    class FakeResponse:
        def __init__(self, url):
            self.status = 429
            self.headers = {"Content-Type": "text/html"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):  # pragma: no cover
            raise AssertionError("FetchBlockedError must fire before raise_for_status")

        async def text(self):  # pragma: no cover
            return ""

        async def read(self):  # pragma: no cover
            return b""

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, timeout):
            return FakeResponse(url)

    monkeypatch.setattr(fetcher.aiohttp, "ClientSession", FakeSession)

    with pytest.raises(fetcher.FetchBlockedError):
        asyncio.run(fetcher.fetch_url("https://author.substack.com/p/post"))


def test_is_known_blocked_host_matches_subdomain():
    assert fetcher._is_known_blocked_host("substack.com")
    assert fetcher._is_known_blocked_host("author.substack.com")
    assert not fetcher._is_known_blocked_host("example.com")
    assert not fetcher._is_known_blocked_host("substacky.example.com")


# ---------------------------------------------------------------------------
# Readability-proxy fallback (fr_researcher_22486af4)
# ---------------------------------------------------------------------------


def test_readability_proxy_url_gating():
    f = fetcher._readability_proxy_url
    # Disabled / malformed config -> None (fail closed, no external call).
    assert f(None, "https://x.substack.com/a") is None
    assert f({}, "https://x.substack.com/a") is None
    assert f({"proxy": "no-placeholder"}, "https://x.substack.com/a") is None
    # Host allowlist gates which URLs reach the proxy.
    cfg = {"proxy": "https://r.jina.ai/{url}", "hosts": ["substack.com"]}
    assert (
        f(cfg, "https://x.substack.com/a")
        == "https://r.jina.ai/https://x.substack.com/a"
    )
    assert f(cfg, "https://other.com/a") is None
    # Absent/empty hosts -> applies to any blocked host.
    cfg2 = {"proxy": "https://r.jina.ai/{url}"}
    assert (
        f(cfg2, "https://anything.com/a")
        == "https://r.jina.ai/https://anything.com/a"
    )
    # Embedded credentials -> fail closed (never ship basic-auth to the proxy).
    assert f(cfg, "https://user:pass@x.substack.com/a") is None
    # Malformed hosts (bare string / mixed-case / non-strings) are normalized,
    # not crashed on.
    messy = {"proxy": "https://r.jina.ai/{url}", "hosts": "SubStack.com"}
    assert (
        f(messy, "https://x.substack.com/a")
        == "https://r.jina.ai/https://x.substack.com/a"
    )
    messy2 = {"proxy": "https://r.jina.ai/{url}", "hosts": [".SubStack.com", None, 123]}
    assert (
        f(messy2, "https://x.substack.com/a")
        == "https://r.jina.ai/https://x.substack.com/a"
    )


@pytest.mark.asyncio
async def test_fetch_url_falls_back_to_readability_on_block(monkeypatch):
    calls = []

    async def fake_direct(u, timeout=60):
        calls.append(u)
        if u.startswith("https://r.jina.ai/"):
            return fetcher.FetchResult(
                url=u, title="T", content="proxied body",
                format=ContentFormat.MARKDOWN, metadata={"source": "web"},
            )
        raise fetcher.FetchBlockedError("blocked 403")

    monkeypatch.setattr(fetcher, "_fetch_url_direct", fake_direct)
    cfg = {"proxy": "https://r.jina.ai/{url}", "hosts": ["substack.com"]}

    result = await fetcher.fetch_url(
        "https://x.substack.com/p/a", readability_fallback=cfg,
    )
    assert result.content == "proxied body"
    # restamped to the ORIGINAL url, with the fallback path recorded.
    assert result.url == "https://x.substack.com/p/a"
    assert result.metadata["fetched_via"] == "readability_fallback"
    # Only the proxy HOST is persisted — not the full templated URL (which
    # embeds the original url + any query params/tokens).
    assert result.metadata["readability_proxy"] == "r.jina.ai"
    assert calls == [
        "https://x.substack.com/p/a",
        "https://r.jina.ai/https://x.substack.com/p/a",
    ]


@pytest.mark.asyncio
async def test_fetch_url_no_proxy_call_when_fallback_disabled(monkeypatch):
    calls = []

    async def fake_direct(u, timeout=60):
        calls.append(u)
        raise fetcher.FetchBlockedError("blocked 403")

    monkeypatch.setattr(fetcher, "_fetch_url_direct", fake_direct)
    with pytest.raises(fetcher.FetchBlockedError):
        await fetcher.fetch_url("https://x.substack.com/p/a", readability_fallback=None)
    assert calls == ["https://x.substack.com/p/a"]  # no external proxy call


@pytest.mark.asyncio
async def test_fetch_url_no_proxy_call_when_host_not_allowlisted(monkeypatch):
    calls = []

    async def fake_direct(u, timeout=60):
        calls.append(u)
        raise fetcher.FetchBlockedError("blocked 403")

    monkeypatch.setattr(fetcher, "_fetch_url_direct", fake_direct)
    cfg = {"proxy": "https://r.jina.ai/{url}", "hosts": ["substack.com"]}
    with pytest.raises(fetcher.FetchBlockedError):
        await fetcher.fetch_url("https://other.com/p/a", readability_fallback=cfg)
    assert calls == ["https://other.com/p/a"]  # host not allowlisted -> no proxy


@pytest.mark.asyncio
async def test_fetch_url_surfaces_both_failures_when_proxy_also_fails(monkeypatch):
    async def fake_direct(u, timeout=60):
        if u.startswith("https://r.jina.ai/"):
            raise RuntimeError("proxy 500")
        raise fetcher.FetchBlockedError("blocked 403")

    monkeypatch.setattr(fetcher, "_fetch_url_direct", fake_direct)
    cfg = {"proxy": "https://r.jina.ai/{url}", "hosts": ["substack.com"]}
    with pytest.raises(fetcher.FetchBlockedError, match="also failed") as exc:
        await fetcher.fetch_url("https://x.substack.com/p/a", readability_fallback=cfg)
    # Carries the blocked url so callers know which URL failed...
    assert exc.value.url == "https://x.substack.com/p/a"
    # ...and the proxy cause is NOT chained (its str can embed the proxied URL;
    # `from None` keeps it out of any upstream logger.exception traceback).
    assert exc.value.__cause__ is None


@pytest.mark.asyncio
async def test_fetch_url_fallback_keys_on_resolved_blocked_target(monkeypatch):
    """A shortlink that resolves to a blocked target: the fallback must key on
    the TARGET host (allowlisted), not the original shortlink, and proxy the
    target. FetchBlockedError.url carries the actually-blocked URL."""
    calls = []

    async def fake_direct(u, timeout=60):
        calls.append(u)
        if u.startswith("https://r.jina.ai/"):
            return fetcher.FetchResult(
                url=u, title="T", content="proxied",
                format=ContentFormat.MARKDOWN, metadata={"source": "web"},
            )
        # The original shortlink resolved (inside _fetch_url_direct) to a
        # blocked substack target; the error names that target.
        raise fetcher.FetchBlockedError("blocked 403", url="https://x.substack.com/p/a")

    monkeypatch.setattr(fetcher, "_fetch_url_direct", fake_direct)
    # Only substack is allowlisted — lnkd.in is NOT, so a fix that keyed on the
    # original would not fall back at all.
    cfg = {"proxy": "https://r.jina.ai/{url}", "hosts": ["substack.com"]}

    result = await fetcher.fetch_url("https://lnkd.in/abc", readability_fallback=cfg)
    assert result.content == "proxied"
    assert calls == [
        "https://lnkd.in/abc",
        "https://r.jina.ai/https://x.substack.com/p/a",  # proxied the TARGET
    ]
    # restamped to the resolved target (dedupe keys on the real resource)
    assert result.url == "https://x.substack.com/p/a"
