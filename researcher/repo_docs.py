"""Distill a repository's docs corpus into normative claims.

Companion to ``fr_researcher_86a810a3``. Extracts the imperative content from
documentation (READMEs, ARCHITECTURE.md, conventions.md, ...) into a compact
artifact suitable for prompt augmentation during code review. The intent is
to hand reviewers a project's "conventions" without forcing the original
10–20k tokens of narrative markdown into every review prompt.

Layering:

* Stateless helpers (``normalize_corpus``, ``compute_corpus_hash``,
  ``cache_fingerprint``, ``cache_artifact_id``) have no I/O at all.
* ``extract_normative_claims`` is also store-free, but it does call the
  LLM via the supplied ``pool``. Use it as the building block when you
  want the transform without the cache.
* ``distill_repo_docs`` is the cache-aware orchestrator. It calls the LLM
  AND routes store I/O through a ``store_request`` callable so the bus
  RPC mechanism (envelope unwrap, agent-type routing) lives in the
  agent-side handler. Tests inject a fake store_request to exercise
  hit/miss / error / truncated branches without bringing up the bus.
"""

from __future__ import annotations

import hashlib
from typing import Any


NORMATIVE_CLAIM_PROMPT_VERSION = "v1"

NORMATIVE_CLAIM_PROMPT = """\
You are extracting normative content from a project's documentation. Your
output will be used as context for code review.

INCLUDE:
- Imperative rules: "must", "must not", "always", "never", "do not"
- Invariants and constraints stated as facts about how the system works
- Architectural decisions and conventions (naming, error handling, layering)
- Explicit "do/don't" patterns and anti-patterns

EXCLUDE:
- Narrative, motivation, history
- Examples (unless they are stating a rule)
- Backstory or rationale paragraphs
- "Welcome to <project>" boilerplate

OUTPUT FORMAT:
- Flat bulleted list, one rule per line, starting with "- "
- Each line prefixed with `[source:<path>]` to preserve provenance
- Preserve surprising or project-specific rules verbatim; do NOT paraphrase,
  generalize, or cluster
- If a section has no normative content, skip it

DOCS:
{corpus}
"""


# Maximum input size sent to the LLM. Larger corpora must be filtered upstream.
# Sized to fit qwen2.5:7b's 32k-token context with room for the prompt
# scaffold + response. Tunable per-call.
MAX_CORPUS_CHARS = 80_000

# Read-side cap on cache-hit ``artifact_get`` calls. Matches the
# bus/store HARD_MAX_CHARS=20000 ceiling (see
# ``khonliang_bus.bus.artifacts.HARD_MAX_CHARS`` and
# ``khonliang_store.store.local_store.HARD_MAX_CHARS``) so the requested
# size matches what we'll actually receive. Distillations land well
# under this (``max_tokens=3000`` ≈ 12 KB of text), so a cache hit
# returning ``truncated=True`` is an integrity signal worth surfacing
# rather than something to silently work around.
_CACHE_GET_MAX_CHARS = 20_000


def normalize_corpus(content: dict[str, str]) -> str:
    """Produce a deterministic ``# <path>\\n<body>\\n\\n`` concatenation.

    Sorted by path. CRLF normalized to LF. Per-line trailing whitespace
    stripped. Same content → same string regardless of dict ordering, so
    ``compute_corpus_hash`` is stable across callers.
    """
    if not isinstance(content, dict):
        raise TypeError("content must be a dict of path -> body strings")
    if not content:
        raise ValueError("content is required")
    blocks = []
    for path in sorted(content.keys()):
        body = content[path]
        if not isinstance(body, str):
            raise TypeError(f"content[{path!r}] must be a string")
        normalized = "\n".join(
            line.rstrip() for line in body.replace("\r\n", "\n").split("\n")
        )
        blocks.append(f"# {path}\n{normalized}\n")
    return "\n".join(blocks)


def compute_corpus_hash(content: dict[str, str]) -> str:
    """SHA-256 of the normalized corpus.

    Stable across callers; insensitive to dict ordering, line-ending style,
    and trailing whitespace.
    """
    return hashlib.sha256(normalize_corpus(content).encode("utf-8")).hexdigest()


def cache_fingerprint(
    *,
    source_sha256: str,
    model: str,
    prompt_version: str,
) -> dict[str, str]:
    """Metadata shape stored on cached distillations; matched on read.

    The triple ``(source_sha256, model, prompt_version)`` is the cache key.
    Changing the prompt or swapping models invalidates prior caches.
    """
    return {
        "source_sha256": source_sha256,
        "model": model,
        "prompt_version": prompt_version,
    }


def fingerprint_matches(metadata: Any, fingerprint: dict[str, str]) -> bool:
    """True if an artifact's metadata carries the same cache fingerprint."""
    if not isinstance(metadata, dict):
        return False
    for key, value in fingerprint.items():
        if metadata.get(key) != value:
            return False
    return True


async def extract_normative_claims(
    content: dict[str, str],
    pool: Any,
    *,
    model_role: str = "summarizer",
    max_corpus_chars: int = MAX_CORPUS_CHARS,
    max_tokens: int = 3000,
) -> dict[str, Any]:
    """Run the LLM extraction over a normalized corpus.

    Pure transform: no caching, no artifact storage. Returns the extracted
    digest plus the effective model string (so callers can include it in
    the cache fingerprint).

    Raises ``ValueError`` if the normalized corpus exceeds
    ``max_corpus_chars``. Callers should pre-filter or chunk upstream;
    automatic chunking is out of scope for v1.
    """
    corpus = normalize_corpus(content)
    if len(corpus) > max_corpus_chars:
        raise ValueError(
            f"docs corpus is {len(corpus)} chars, exceeds max_corpus_chars="
            f"{max_corpus_chars}. Filter docs upstream (e.g. exclude generated "
            f"files or large fixtures) or call with a subset."
        )
    prompt = NORMATIVE_CLAIM_PROMPT.format(corpus=corpus)
    client = pool.get_client(model_role)
    raw = await client.generate(
        prompt=prompt,
        system=(
            "Extract normative content for code review. "
            "Output only the bulleted list."
        ),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    digest = (raw or "").strip()
    return {
        "digest": digest,
        "model": getattr(client, "model", model_role),
        "prompt_version": NORMATIVE_CLAIM_PROMPT_VERSION,
    }


def cache_artifact_id(source_sha256: str, model: str, prompt_version: str) -> str:
    """Deterministic ``art_repodocs_<24-hex>`` id derived from the cache key.

    Stable across processes — two callers asking for the same (source,
    model, prompt_version) produce the same id, so a simple
    ``artifact_metadata(id=...)`` is a cache lookup.
    """
    key = f"{source_sha256}|{model}|{prompt_version}".encode("utf-8")
    return f"art_repodocs_{hashlib.sha256(key).hexdigest()[:24]}"


async def distill_repo_docs(
    *,
    content: dict[str, str],
    pool: Any,
    store_request: Any,
    repo_name: str = "",
    model_role: str = "summarizer",
    prompt_version: str = NORMATIVE_CLAIM_PROMPT_VERSION,
    max_corpus_chars: int = MAX_CORPUS_CHARS,
    max_tokens: int = 3000,
    producer: str = "",
) -> dict[str, Any]:
    """End-to-end: compute hash, check cache, run LLM on miss, store result.

    ``store_request`` is an async callable ``(operation, args) -> envelope``
    that routes a single store skill call. Three operations are issued:

    * ``artifact_metadata`` (always, to probe the cache key)
    * ``artifact_get`` (on cache hit, to fetch the body)
    * ``artifact_create`` (on cache miss, after the LLM run)

    The agent-side handler wires this to
    ``agent.request(agent_type='store', operation=op, args=args)``; tests
    pass a fake to exercise the hit / miss / error / truncated branches
    without bringing up the bus.

    Returns the cache shape on success:
        {
            "artifact_id": str,
            "digest": str,
            "model": str,
            "prompt_version": str,
            "source_sha256": str,
            "cache_hit": bool,
            "repo_name": str,
        }
    Or an ``{"error": ...}`` envelope from store, surfaced verbatim. The
    ``artifact_metadata`` "not found" error is treated as a cache miss
    rather than surfaced; any other metadata error is a real failure and
    aborts the call.
    """
    source_sha256 = compute_corpus_hash(content)
    # Resolve the client exactly once: the effective model determines the
    # cache key, and the same client must run the LLM on miss so the stored
    # artifact's model matches the key it's filed under.
    client = pool.get_client(model_role)
    effective_model = getattr(client, "model", model_role)
    artifact_id = cache_artifact_id(source_sha256, effective_model, prompt_version)

    cached = await store_request(
        "artifact_metadata", {"id": artifact_id},
    )
    cached_meta = _unwrap(cached)
    # Fail fast on an unparseable response — a non-dict here means the
    # store contract is broken; better to abort than mistake it for a
    # cache miss and silently run the LLM.
    if not isinstance(cached_meta, dict):
        return {"error": "store returned unexpected response shape on metadata lookup"}
    if "error" in cached_meta:
        # "not found" is the documented cache-miss sentinel; everything
        # else (permission, network, integrity) is a real failure and
        # must surface — don't silently fall through to an LLM run.
        if not _is_not_found_error(cached_meta["error"]):
            return cached_meta
    else:
        body_envelope = await store_request(
            "artifact_get",
            {"id": artifact_id, "offset": 0, "max_chars": _CACHE_GET_MAX_CHARS},
        )
        body_payload = _unwrap(body_envelope)
        # Cache hit means metadata exists. Body retrieval failures should be
        # surfaced, not papered over with an empty-string digest — mirrors
        # ``ingest_from_artifact``'s handling so a store regression doesn't
        # silently leak partial content as ``cache_hit=True``.
        if not isinstance(body_payload, dict):
            return {"error": "store returned unexpected response shape on cache hit"}
        if "error" in body_payload:
            return body_payload
        if body_payload.get("truncated") is True:
            return {
                "error": (
                    f"cached distillation {artifact_id} exceeds "
                    f"{_CACHE_GET_MAX_CHARS}-char read cap; refusing to return "
                    "a partial digest"
                ),
            }
        digest = (
            body_payload.get("text")
            or body_payload.get("content")
            or body_payload.get("body")
            or ""
        )
        meta = cached_meta.get("artifact") if isinstance(cached_meta.get("artifact"), dict) else cached_meta
        return {
            "artifact_id": artifact_id,
            "digest": digest.strip() if isinstance(digest, str) else "",
            "model": meta.get("metadata", {}).get("model", effective_model),
            "prompt_version": meta.get("metadata", {}).get("prompt_version", prompt_version),
            "source_sha256": source_sha256,
            "cache_hit": True,
            "repo_name": meta.get("metadata", {}).get("repo_name", repo_name),
        }

    # Cache miss: run the LLM against the SAME client we keyed by.
    corpus = normalize_corpus(content)
    if len(corpus) > max_corpus_chars:
        raise ValueError(
            f"docs corpus is {len(corpus)} chars, exceeds max_corpus_chars="
            f"{max_corpus_chars}. Filter docs upstream (e.g. exclude generated "
            f"files or large fixtures) or call with a subset."
        )
    raw = await client.generate(
        prompt=NORMATIVE_CLAIM_PROMPT.format(corpus=corpus),
        system=(
            "Extract normative content for code review. "
            "Output only the bulleted list."
        ),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    digest = (raw or "").strip()
    metadata = {
        **cache_fingerprint(
            source_sha256=source_sha256,
            model=effective_model,
            prompt_version=prompt_version,
        ),
        "repo_name": repo_name,
        "file_count": len(content),
    }
    create_envelope = await store_request(
        "artifact_create",
        {
            "id": artifact_id,
            "kind": "researcher_distillation",
            "title": (
                f"docs distillation: {repo_name}"
                if repo_name else "docs distillation"
            ),
            "content": digest,
            "content_type": "text/markdown",
            "producer": producer,
            "metadata": metadata,
            "source_artifacts": [],
        },
    )
    create_payload = _unwrap(create_envelope)
    # Mirror stage_payload / ingest_from_artifact: require a dict payload.
    # A non-dict here would silently get treated as "success" without an
    # id, masking real store-shape regressions.
    if not isinstance(create_payload, dict):
        return {"error": "store returned unexpected response shape on artifact_create"}
    if "error" in create_payload:
        return create_payload
    stored_id = (
        create_payload.get("id")
        or (create_payload.get("artifact") or {}).get("id")
        or artifact_id
    )
    return {
        "artifact_id": stored_id,
        "digest": digest,
        "model": effective_model,
        "prompt_version": prompt_version,
        "source_sha256": source_sha256,
        "cache_hit": False,
        "repo_name": repo_name,
    }


def _unwrap(envelope: Any) -> Any:
    """Pull the inner payload from a bus request envelope, mirroring
    :func:`researcher.agent._unwrap_request_envelope`.

    Duplicated here so this module has no agent.py dependency (keeps the
    test surface small). The shape is the contract; the helper is incidental.
    """
    if isinstance(envelope, dict):
        return envelope.get("result", envelope)
    return envelope


def _is_not_found_error(message: Any) -> bool:
    """True if a store error envelope's ``error`` string means cache-miss.

    Store agents return errors as opaque strings; this is the conventional
    sentinel (``"not found"``, possibly with extra context). Anything else
    — permission denied, transport failure, integrity failure — must be
    surfaced rather than swallowed as a miss.
    """
    if not isinstance(message, str):
        return False
    return "not found" in message.lower()
