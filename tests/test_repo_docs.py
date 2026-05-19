"""Tests for repo-docs distillation primitives (fr_researcher_86a810a3)."""

from __future__ import annotations

import pytest

from researcher.repo_docs import (
    MAX_CORPUS_CHARS,
    NORMATIVE_CLAIM_PROMPT_VERSION,
    cache_artifact_id,
    cache_fingerprint,
    compute_corpus_hash,
    distill_repo_docs,
    extract_normative_claims,
    fingerprint_matches,
    normalize_corpus,
)


def test_normalize_corpus_sorted_by_path():
    out = normalize_corpus({"b.md": "B", "a.md": "A"})
    assert out.index("# a.md") < out.index("# b.md")


def test_normalize_corpus_normalizes_line_endings_and_trailing_whitespace():
    a = normalize_corpus({"x.md": "line1  \r\nline2\t \r\n"})
    b = normalize_corpus({"x.md": "line1\nline2\n"})
    assert a == b


def test_normalize_corpus_rejects_non_dict():
    with pytest.raises(TypeError):
        normalize_corpus("not a dict")  # type: ignore[arg-type]


def test_normalize_corpus_rejects_empty_dict():
    with pytest.raises(ValueError, match="content is required"):
        normalize_corpus({})


def test_normalize_corpus_rejects_non_string_body():
    with pytest.raises(TypeError, match="must be a string"):
        normalize_corpus({"x.md": 123})  # type: ignore[dict-item]


def test_compute_corpus_hash_is_stable_across_dict_order_and_line_endings():
    h1 = compute_corpus_hash({"a.md": "x\r\n", "b.md": "y  \n"})
    h2 = compute_corpus_hash({"b.md": "y\n", "a.md": "x\n"})
    assert h1 == h2


def test_compute_corpus_hash_differs_on_content_change():
    h1 = compute_corpus_hash({"a.md": "rule one"})
    h2 = compute_corpus_hash({"a.md": "rule two"})
    assert h1 != h2


def test_compute_corpus_hash_is_64_hex_chars():
    h = compute_corpus_hash({"a.md": "anything"})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_cache_fingerprint_round_trips():
    fp = cache_fingerprint(
        source_sha256="deadbeef", model="qwen2.5:7b", prompt_version="v1",
    )
    assert fingerprint_matches(fp, fp)
    # Missing key -> no match
    assert not fingerprint_matches({"source_sha256": "deadbeef"}, fp)
    # Different value -> no match
    assert not fingerprint_matches(
        {**fp, "model": "qwen2.5:32b"}, fp,
    )


def test_fingerprint_matches_rejects_non_dict_metadata():
    fp = cache_fingerprint(source_sha256="x", model="y", prompt_version="v1")
    assert not fingerprint_matches(None, fp)
    assert not fingerprint_matches("string", fp)


# ---------------------------------------------------------------------------
# extract_normative_claims (LLM transform with a mocked pool)
# ---------------------------------------------------------------------------


class _FakeClient:
    """Captures call args; returns a fixed digest."""

    def __init__(self, response: str = "- [source:README.md] never panic", model: str = "qwen2.5:7b"):
        self.response = response
        self.model = model
        self.last_kwargs: dict | None = None

    async def generate(self, **kwargs) -> str:
        self.last_kwargs = kwargs
        return self.response


class _FakePool:
    def __init__(self, client: _FakeClient):
        self.client = client
        self.last_role: str | None = None

    def get_client(self, role: str) -> _FakeClient:
        self.last_role = role
        return self.client


@pytest.mark.asyncio
async def test_extract_normative_claims_calls_summarizer_with_corpus():
    client = _FakeClient(response="- [source:README.md] always use Result types\n")
    pool = _FakePool(client)
    result = await extract_normative_claims(
        {"README.md": "All errors MUST be returned as Result.", "docs/intro.md": "Welcome!"},
        pool,
    )
    assert result["digest"] == "- [source:README.md] always use Result types"
    assert result["model"] == "qwen2.5:7b"
    assert result["prompt_version"] == NORMATIVE_CLAIM_PROMPT_VERSION
    assert pool.last_role == "summarizer"
    # Corpus must be present in the prompt, both paths included.
    prompt = client.last_kwargs["prompt"]
    assert "# README.md" in prompt
    assert "# docs/intro.md" in prompt
    assert client.last_kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_extract_normative_claims_respects_model_role_override():
    client = _FakeClient()
    pool = _FakePool(client)
    await extract_normative_claims(
        {"x.md": "Never block on I/O."},
        pool,
        model_role="reviewer",
    )
    assert pool.last_role == "reviewer"


@pytest.mark.asyncio
async def test_extract_normative_claims_rejects_oversized_corpus():
    big = "x" * (MAX_CORPUS_CHARS + 1)
    pool = _FakePool(_FakeClient())
    with pytest.raises(ValueError, match="exceeds max_corpus_chars"):
        await extract_normative_claims({"big.md": big}, pool)


# ---------------------------------------------------------------------------
# cache_artifact_id + distill_repo_docs end-to-end with a fake store
# ---------------------------------------------------------------------------


def test_cache_artifact_id_is_deterministic_and_prefixed():
    a = cache_artifact_id("deadbeef", "qwen2.5:7b", "v1")
    b = cache_artifact_id("deadbeef", "qwen2.5:7b", "v1")
    c = cache_artifact_id("deadbeef", "qwen2.5:32b", "v1")
    d = cache_artifact_id("deadbeef", "qwen2.5:7b", "v2")
    assert a == b
    assert a != c
    assert a != d
    assert a.startswith("art_repodocs_")


class _FakeStore:
    """Records (operation, args) calls; returns scripted envelopes."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        # Artifacts keyed by id; populated on artifact_create, read on
        # artifact_metadata + artifact_get.
        self.artifacts: dict[str, dict] = {}

    async def __call__(self, operation: str, args: dict) -> dict:
        self.calls.append((operation, args))
        if operation == "artifact_metadata":
            existing = self.artifacts.get(args["id"])
            if existing is None:
                return {"result": {"error": "not found"}}
            return {"result": existing["meta"]}
        if operation == "artifact_get":
            existing = self.artifacts.get(args["id"])
            if existing is None:
                return {"result": {"error": "not found"}}
            return {"result": {"text": existing["content"]}}
        if operation == "artifact_create":
            aid = args["id"]
            self.artifacts[aid] = {
                "content": args["content"],
                "meta": {
                    "id": aid,
                    "kind": args["kind"],
                    "title": args["title"],
                    "metadata": args.get("metadata", {}),
                    "producer": args.get("producer", ""),
                },
            }
            return {"result": {"id": aid}}
        raise AssertionError(f"unexpected operation: {operation}")


@pytest.mark.asyncio
async def test_distill_repo_docs_cache_miss_then_hit():
    """First call runs LLM + stores; second call with identical content reuses."""
    client = _FakeClient(response="- [source:README.md] never panic")
    pool = _FakePool(client)
    store = _FakeStore()
    content = {"README.md": "All errors MUST return Result."}

    first = await distill_repo_docs(
        content=content, pool=pool, store_request=store,
        repo_name="myrepo", producer="researcher-test",
    )
    assert first["cache_hit"] is False
    assert first["digest"] == "- [source:README.md] never panic"
    assert first["artifact_id"].startswith("art_repodocs_")

    # Second call with same content -> hits cache.
    pool.client.last_kwargs = None  # reset so a second LLM call would be detectable
    second = await distill_repo_docs(
        content=content, pool=pool, store_request=store,
        repo_name="myrepo", producer="researcher-test",
    )
    assert second["cache_hit"] is True
    assert second["artifact_id"] == first["artifact_id"]
    assert second["digest"] == first["digest"]
    # Second call must not have invoked the LLM.
    assert pool.client.last_kwargs is None


@pytest.mark.asyncio
async def test_distill_repo_docs_changed_content_misses_cache():
    pool = _FakePool(_FakeClient(response="- [source:a.md] rule"))
    store = _FakeStore()

    a = await distill_repo_docs(
        content={"a.md": "Original."}, pool=pool, store_request=store,
    )
    b = await distill_repo_docs(
        content={"a.md": "Changed."}, pool=pool, store_request=store,
    )
    assert a["cache_hit"] is False
    assert b["cache_hit"] is False
    assert a["artifact_id"] != b["artifact_id"]


@pytest.mark.asyncio
async def test_distill_repo_docs_model_change_invalidates_cache():
    """Switching model role -> different effective model -> different cache key."""

    class _SwapClient:
        """Returns a different model name on each get_client call to simulate role swap."""

        def __init__(self):
            self.models = ["qwen2.5:7b", "qwen2.5:32b"]
            self.idx = 0
            self.model = self.models[0]
            self.last_kwargs = None

        async def generate(self, **kwargs):
            self.last_kwargs = kwargs
            return f"- [source:x.md] rule under {self.model}"

    swap_client = _SwapClient()

    class _SwapPool:
        def __init__(self, client):
            self.client = client

        def get_client(self, role):
            # advance to next model on each call (mocking summarizer -> reviewer swap)
            self.client.model = self.client.models[min(self.client.idx, len(self.client.models) - 1)]
            self.client.idx += 1
            return self.client

    pool = _SwapPool(swap_client)
    store = _FakeStore()
    content = {"x.md": "Always check for None before calling."}

    first = await distill_repo_docs(
        content=content, pool=pool, store_request=store, model_role="summarizer",
    )
    second = await distill_repo_docs(
        content=content, pool=pool, store_request=store, model_role="reviewer",
    )
    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert first["artifact_id"] != second["artifact_id"]
    assert first["model"] == "qwen2.5:7b"
    assert second["model"] == "qwen2.5:32b"


@pytest.mark.asyncio
async def test_distill_repo_docs_prompt_version_invalidates_cache():
    pool = _FakePool(_FakeClient(response="- [source:r.md] rule"))
    store = _FakeStore()
    content = {"r.md": "Never block on I/O."}

    a = await distill_repo_docs(
        content=content, pool=pool, store_request=store, prompt_version="v1",
    )
    b = await distill_repo_docs(
        content=content, pool=pool, store_request=store, prompt_version="v2-experimental",
    )
    assert a["artifact_id"] != b["artifact_id"]


@pytest.mark.asyncio
async def test_distill_repo_docs_store_create_error_surfaces():
    """Store-side failure during artifact_create returns the error envelope verbatim."""

    pool = _FakePool(_FakeClient())

    async def failing_store(operation, args):
        if operation == "artifact_metadata":
            return {"result": {"error": "not found"}}
        if operation == "artifact_create":
            return {"result": {"error": "disk full"}}
        raise AssertionError(operation)

    result = await distill_repo_docs(
        content={"x.md": "rule."}, pool=pool, store_request=failing_store,
    )
    assert result == {"error": "disk full"}


@pytest.mark.asyncio
async def test_distill_repo_docs_handler_validates_args_and_dispatches():
    """Agent-side handler: arg validation + store_request closure wiring."""
    from researcher.agent import distill_repo_docs_handler

    class _FakeAgent:
        agent_id = "researcher-test"

        def __init__(self, store):
            self._store = store

        async def request(self, *, agent_type, operation, args):
            assert agent_type == "store"
            return await self._store(operation, args)

    class _FakePipeline:
        def __init__(self, pool):
            self.pool = pool

    store = _FakeStore()
    pool = _FakePool(_FakeClient(response="- [source:README.md] rule"))
    agent = _FakeAgent(store)
    pipeline = _FakePipeline(pool)

    # Happy path
    ok = await distill_repo_docs_handler(
        agent, pipeline,
        {"content": {"README.md": "Always do X."}, "repo_name": "alpha"},
    )
    assert ok["cache_hit"] is False
    assert ok["repo_name"] == "alpha"
    assert ok["artifact_id"].startswith("art_repodocs_")
    # Verify the producer came from agent.agent_id
    create_call = next(c for c in store.calls if c[0] == "artifact_create")
    assert create_call[1]["producer"] == "researcher-test"

    # Bad: non-dict content
    bad = await distill_repo_docs_handler(agent, pipeline, {"content": "not a dict"})
    assert "error" in bad and "object mapping" in bad["error"]

    # Bad: empty content
    bad = await distill_repo_docs_handler(agent, pipeline, {"content": {}})
    assert "error" in bad and "required" in bad["error"]

    # Bad: missing content
    bad = await distill_repo_docs_handler(agent, pipeline, {})
    assert "error" in bad

    # Bad: non-string body
    bad = await distill_repo_docs_handler(
        agent, pipeline, {"content": {"a.md": 42}},
    )
    assert "error" in bad and "must be a string" in bad["error"]


@pytest.mark.asyncio
async def test_distill_repo_docs_records_metadata_fingerprint_on_create():
    pool = _FakePool(_FakeClient(response="- [source:a.md] r"))
    store = _FakeStore()
    await distill_repo_docs(
        content={"a.md": "rule"}, pool=pool, store_request=store,
        repo_name="alpha", producer="researcher-test",
    )
    # Find the artifact_create call.
    create_calls = [c for c in store.calls if c[0] == "artifact_create"]
    assert len(create_calls) == 1
    args = create_calls[0][1]
    md = args["metadata"]
    assert md["source_sha256"] == compute_corpus_hash({"a.md": "rule"})
    assert md["model"] == "qwen2.5:7b"
    assert md["prompt_version"] == NORMATIVE_CLAIM_PROMPT_VERSION
    assert md["repo_name"] == "alpha"
    assert md["file_count"] == 1
    assert args["kind"] == "researcher_distillation"
    assert args["producer"] == "researcher-test"
    assert args["id"].startswith("art_repodocs_")
