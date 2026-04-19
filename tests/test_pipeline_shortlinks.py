import asyncio

from researcher import pipeline
from researcher.fetcher import ContentFormat, FetchResult
from researcher.pipeline import ResearchPipeline


class FakeKnowledge:
    def __init__(self):
        self.entries = []

    def add(self, entry):
        self.entries.append(entry)


class FakeDigest:
    def __init__(self):
        self.records = []

    def record(self, **kwargs):
        self.records.append(kwargs)


def _pipeline_with_index(index=None):
    pipe = ResearchPipeline.__new__(ResearchPipeline)
    pipe._url_index = dict(index or {})
    pipe.knowledge = FakeKnowledge()
    pipe.digest = FakeDigest()
    return pipe


def test_ingest_paper_dedupes_resolved_arxiv_shortlink(monkeypatch):
    existing_id = "existing-entry"
    pipe = _pipeline_with_index({"https://arxiv.org/abs/2511.19699": existing_id})

    async def fake_fetch_url(url):
        return FetchResult(
            url="https://arxiv.org/pdf/2511.19699",
            title="A Layered Protocol Architecture for the Internet of Agents",
            content="Paper body",
            format=ContentFormat.PDF,
            metadata={"resolved_from": url},
        )

    monkeypatch.setattr(pipeline, "fetch_url", fake_fetch_url)

    entry_id = asyncio.run(pipe.ingest_paper("https://lnkd.in/guZ5SMq3"))

    assert entry_id == existing_id
    assert pipe._url_index["https://lnkd.in/guZ5SMq3"] == existing_id
    assert pipe.knowledge.entries == []


def test_ingest_paper_stores_resolved_source_and_provenance(monkeypatch):
    pipe = _pipeline_with_index()

    async def fake_fetch_url(url):
        return FetchResult(
            url="https://example.com/paper.txt",
            title="Example Paper",
            content="Paper body",
            format=ContentFormat.TEXT,
            metadata={
                "resolved_from": url,
                "resolved_chain": [url],
                "shortlink_resolver": "linkedin_external_interstitial",
            },
        )

    monkeypatch.setattr(pipeline, "fetch_url", fake_fetch_url)

    entry_id = asyncio.run(pipe.ingest_paper("https://lnkd.in/example"))

    assert pipe._url_index["https://example.com/paper.txt"] == entry_id
    assert pipe._url_index["https://lnkd.in/example"] == entry_id
    entry = pipe.knowledge.entries[0]
    assert entry.source == "https://example.com/paper.txt"
    assert entry.metadata["url"] == "https://example.com/paper.txt"
    assert entry.metadata["original_url"] == "https://lnkd.in/example"
    assert entry.metadata["resolved_from"] == "https://lnkd.in/example"
