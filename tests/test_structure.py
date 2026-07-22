"""Tests for researcher.structure.StructureRole (fr_researcher_d813ad52).

Mocks the LLM client entirely (ScriptedClient below) — no live Ollama model
required, matching this repo's house style (see test_feed_store.py /
test_distill_lock.py for other mocked-collaborator examples).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, RootModel

from researcher.structure import SPEC_VERSION, StructureRole, StructureResult


class PersonRecord(BaseModel):
    name: str
    age: int


class ScriptedClient:
    """Fake OllamaClient: returns/raises a scripted sequence of responses.

    Each entry in ``responses`` is either a dict (returned as-is from
    ``generate_json``) or an ``Exception`` instance (raised instead).
    """

    def __init__(self, model_name: str, responses: List[Any]):
        self.model = model_name
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def generate_json(self, **kwargs) -> Dict[str, Any]:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedClient: no more scripted responses")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class ScriptedPool:
    """Fake ModelPool: role -> pre-built client, no lazy construction."""

    def __init__(self, hot: ScriptedClient, escalation: ScriptedClient):
        self._clients = {"structure": hot, "reviewer": escalation}

    def get_client(self, role: str):
        return self._clients[role]


def _pool(hot_responses: List[Any], esc_responses: Optional[List[Any]] = None) -> ScriptedPool:
    hot = ScriptedClient("qwen2.5:7b", hot_responses)
    esc = ScriptedClient("qwen2.5:32b", esc_responses or [])
    return ScriptedPool(hot, esc)


@pytest.mark.asyncio
async def test_success_on_first_attempt():
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    result = await role.structure(
        data="Ada Lovelace, 36 years old.",
        schema=PersonRecord,
        purpose="test extraction",
        project="proj-a",
    )

    assert result.success is True
    assert result.needs_curation is False
    assert isinstance(result.record, PersonRecord)
    assert result.record.name == "Ada"
    assert result.attempts == 1
    assert result.escalated is False
    assert result.model_used == "qwen2.5:7b"
    assert result.schema_name == "PersonRecord"
    assert result.purpose == "test extraction"
    assert result.project == "proj-a"


@pytest.mark.asyncio
async def test_retries_same_tier_before_escalating():
    # First hot-tier attempt is missing a required field; second succeeds.
    pool = _pool([{"name": "Ada"}, {"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    result = await role.structure(data="text", schema=PersonRecord, purpose="p")

    assert result.success is True
    assert result.attempts == 2
    assert result.escalated is False
    assert result.model_used == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_escalates_after_two_failed_hot_tier_attempts():
    pool = _pool(
        hot_responses=[{"name": "Ada"}, {"bogus": "field"}],
        esc_responses=[{"name": "Ada", "age": 36}],
    )
    role = StructureRole(pool)

    result = await role.structure(data="text", schema=PersonRecord, purpose="p")

    assert result.success is True
    assert result.attempts == 3
    assert result.escalated is True
    assert result.model_used == "qwen2.5:32b"


@pytest.mark.asyncio
async def test_needs_curation_after_escalation_still_fails():
    pool = _pool(
        hot_responses=[{"name": "Ada"}, {"name": "Ada"}],
        esc_responses=[{"name": "Ada"}],
    )
    role = StructureRole(pool)

    result = await role.structure(data="text", schema=PersonRecord, purpose="p")

    assert result.success is False
    assert result.needs_curation is True
    assert result.record is None
    assert result.escalated is True
    assert result.attempts == 3
    assert len(result.errors) == 3
    # Terminal failure is stamped, not raised or silently dropped.
    prov = result.provenance()
    assert prov["model"] == "qwen2.5:32b"
    assert prov["spec_version"] == SPEC_VERSION
    assert prov["schema"] == "PersonRecord"


@pytest.mark.asyncio
async def test_semantically_wrong_but_json_valid_output_is_rejected():
    """A model can emit valid JSON that's still schema-invalid (wrong type) —
    defense in depth even if the decoder's constrained mode is reliable."""
    pool = _pool(
        hot_responses=[{"name": "Ada", "age": "very old"}, {"name": "Ada", "age": "still old"}],
        esc_responses=[{"name": "Ada", "age": "ancient"}],
    )
    role = StructureRole(pool)

    result = await role.structure(data="text", schema=PersonRecord, purpose="p")

    assert result.success is False
    assert result.needs_curation is True
    assert all("schema validation failed" in e for e in result.errors)


@pytest.mark.asyncio
async def test_non_object_json_response_invalid_for_a_dict_schema():
    """A dict-shaped schema (PersonRecord) still rejects a non-dict payload —
    model_validate is the sole arbiter, and a list fails PersonRecord's
    validation the same as any other schema-nonconforming shape."""
    pool = _pool(
        hot_responses=[["not", "an", "object"], ["still", "not"]],
        esc_responses=[{"name": "Ada", "age": 36}],
    )
    role = StructureRole(pool)

    result = await role.structure(data="text", schema=PersonRecord, purpose="p")

    assert result.success is True
    assert result.escalated is True


@pytest.mark.asyncio
async def test_non_object_json_response_valid_for_a_root_model_schema():
    """A caller's schema may be a Pydantic RootModel wrapping a list/scalar
    (e.g. RootModel[list[str]]) — a non-dict payload is then perfectly
    valid, so structure() must not gate on isinstance(raw, dict) before
    validating (codex P2, PR #77 round 1)."""

    class Tags(RootModel[List[str]]):
        pass

    pool = _pool([["python", "async"]])
    role = StructureRole(pool)

    result = await role.structure(data="text", schema=Tags, purpose="p")

    assert result.success is True
    assert result.attempts == 1
    assert result.escalated is False
    assert result.record.root == ["python", "async"]


@pytest.mark.asyncio
async def test_generation_transport_failure_counts_as_a_failed_attempt():
    pool = _pool(
        hot_responses=[RuntimeError("ollama unavailable"), {"name": "Ada", "age": 36}],
    )
    role = StructureRole(pool)

    result = await role.structure(data="text", schema=PersonRecord, purpose="p")

    assert result.success is True
    assert result.attempts == 2
    assert result.escalated is False


@pytest.mark.asyncio
async def test_schema_is_passed_to_generate_json_for_every_attempt():
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(data="text", schema=PersonRecord, purpose="p")

    hot_client = pool.get_client("structure")
    assert len(hot_client.calls) == 1
    call = hot_client.calls[0]
    assert call["schema"] == PersonRecord.model_json_schema()
    assert call["model"] == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_input_is_truncated_to_max_chars_before_prompting():
    """Oversized input must not deterministically burn every attempt on a
    context-window overflow that has nothing to do with extraction quality
    (codex P2, PR #77 round 3)."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool, max_chars=100)

    result = await role.structure(data="x" * 5000, schema=PersonRecord, purpose="p")

    prompt = pool.get_client("structure").calls[0]["prompt"]
    # The prompt wraps the (truncated) data with purpose/project framing —
    # assert the truncated data run itself, not the whole prompt (which
    # also contains incidental "x"s in its own scaffolding text).
    assert "x" * 101 not in prompt
    assert "x" * 100 in prompt
    # Truncation is stamped, not silent: a schema-valid success built from
    # truncated input may still be an INCOMPLETE record (relevant fields
    # can live past the cutoff) — callers doing anything
    # completeness-sensitive must be able to see this (codex P2, round 5).
    assert result.truncated is True
    assert result.provenance()["truncated"] is True


@pytest.mark.asyncio
async def test_truncated_flag_false_when_input_fits():
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool, max_chars=10_000)

    result = await role.structure(data="short text", schema=PersonRecord, purpose="p")

    assert result.truncated is False


@pytest.mark.asyncio
async def test_latex_noise_is_stripped_before_prompting():
    """LaTeX math reliably breaks JSON generation — structure() strips it
    before prompting, same reasoning as SummarizerRole's _clean_for_json.
    Forces sanitize_latex=True: a single inline $...$ pair is below the
    auto-detection heuristic's threshold on its own (codex P2, PR #77
    round 10 -- auto-detect), so this test targets _strip_latex_noise's
    mechanics directly rather than depending on heuristic tuning."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"Ada's age is $\alpha$ years.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "$" not in prompt
    assert r"\alpha" not in prompt


@pytest.mark.asyncio
async def test_latex_command_braced_content_is_preserved():
    """A braced LaTeX command's CONTENT is a real field value in
    TeX/Markdown-derived text (e.g. \\textit{Ada Lovelace}) — only the
    command wrapper should be dropped, not the name/title inside it
    (codex P2, PR #77 round 5). Forces sanitize_latex=True for a
    deterministic test of _strip_latex_noise's mechanics rather than
    depending on heuristic tuning (this input WOULD also auto-trigger,
    since it has 2 braced commands, but explicit is clearer intent)."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"Author: \textit{Ada Lovelace}, age \textbf{36}.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "Ada Lovelace" in prompt
    assert "36" in prompt
    assert "\\textit" not in prompt
    assert "\\textbf" not in prompt


@pytest.mark.asyncio
async def test_non_ascii_text_survives_sanitization():
    """structure() promises exact typed fields (unlike lossy summarization),
    so accented/CJK characters in the source text must NOT be stripped —
    codex P2, PR #77 round 4: SummarizerRole's _clean_for_json strips ALL
    non-ASCII, which would silently corrupt a name/title into a schema-valid
    but wrong record without ever setting needs_curation. Forces
    sanitize_latex=True: this plain-prose input has no LaTeX signals at
    all and would NOT auto-trigger sanitization on its own -- this test
    specifically targets _strip_latex_noise's non-ASCII-preserving
    behavior, not the auto-detection heuristic."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data="Author:édouard É 中文",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "édouard" in prompt
    assert "中文" in prompt


@pytest.mark.asyncio
async def test_multi_arg_latex_commands_are_left_untouched():
    """Unwrapping only the FIRST brace-group of a multi-arg command
    corrupts it (\\href{url}{title} -> "url{title}", \\frac{1}{2} ->
    "1{2}") — unambiguous data loss the model never gets a chance to
    recover from. A command with 2+ brace-group arguments must be left
    completely untouched instead of guessed at; under-stripping is a much
    smaller quality hit than mangling (codex P2, PR #77 round 7). Forces
    sanitize_latex=True for a deterministic test of _strip_latex_noise's
    mechanics (this input would also auto-trigger, since it has 2
    command-with-args matches, but explicit is clearer intent)."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"See \href{https://example.com}{Paper Title} and \frac{1}{2}.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert r"\href{https://example.com}{Paper Title}" in prompt
    assert r"\frac{1}{2}" in prompt
    # And single-arg commands are still unwrapped as before.
    assert "url{title}" not in prompt
    assert "1{2}" not in prompt


@pytest.mark.asyncio
async def test_nested_braces_in_single_arg_command_do_not_truncate_early():
    """A regex like `\\{[^}]*\\}` can only ever match up to the FIRST
    closing brace, so a nested brace inside a single-arg command
    (\\textit{Ada {Byron}}) truncated the argument early and left a
    dangling "}" in the output — real corruption on a realistic input (a
    name/title with an internal aside). Brace-balanced scanning must
    track nesting depth so the argument ends at the correctly MATCHING
    closing brace (codex P2, PR #77 round 8). Forces sanitize_latex=True:
    this input has only one command-with-args match by regex count (the
    nested brace means the naive count under-counts it), below the
    auto-detection threshold on its own -- this test targets
    _strip_latex_noise's mechanics directly."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"Author: \textit{Ada {Byron}}, Countess.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    # The command wrapper is gone; the argument's inner literal braces
    # ("Ada {Byron}", a normal parenthetical-style aside once unwrapped)
    # are legitimately preserved -- what must NOT happen is a dangling,
    # UNMATCHED brace from an early-truncated argument scan.
    assert "\\textit" not in prompt
    assert "Ada {Byron}" in prompt
    assert prompt.count("{") == prompt.count("}")
    assert "Countess" in prompt


@pytest.mark.asyncio
async def test_nested_single_arg_command_is_recursively_unwrapped():
    """A single-arg command nested inside another single-arg command
    (\\textit{\\emph{Ada}}) should have BOTH layers unwrapped, not just
    the outer one. Forces sanitize_latex=True: nested commands collapse
    to a single regex match by count, below the auto-detection threshold
    on its own -- this test targets _strip_latex_noise's mechanics
    directly."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"Name: \textit{\emph{Ada}}.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "Ada" in prompt
    assert "\\textit" not in prompt
    assert "\\emph" not in prompt
    assert "{" not in prompt
    assert "}" not in prompt


@pytest.mark.asyncio
async def test_starred_command_unwraps_correctly():
    """A command's argument was assumed to start immediately with `{`, so
    a starred variant like \\section*{Biography} fell into the "no args"
    branch and was corrupted into a stray "*{Biography}" -- the command
    name lost, but the star wrongly left behind as if it were content
    (codex P2, PR #77 round 11). The `*` must be skipped (and dropped,
    not kept) before looking for the real argument."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"\section*{Biography} Ada Lovelace, born 1815.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "Biography" in prompt
    assert "\\section" not in prompt
    assert "*{" not in prompt
    assert "*" not in prompt


@pytest.mark.asyncio
async def test_optional_bracket_arg_command_unwraps_correctly():
    """\\section[Short]{Long}'s argument was assumed to start immediately
    with `{`, so the `[Short]` optional arg was wrongly left behind as if
    it were content, corrupting the command into a stray "[Short]{Long}"
    (codex P2, PR #77 round 11). The `[...]` optional arg must be skipped
    AND discarded (it's a short-form alternate, not the main value) —
    only the required `{Long}` argument is kept."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"\section[Short]{Long Biography} Ada Lovelace, born 1815.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "Long Biography" in prompt
    assert "Short" not in prompt
    assert "\\section" not in prompt
    assert "[" not in prompt
    assert "]" not in prompt


@pytest.mark.asyncio
async def test_multi_arg_command_with_optional_bracket_left_untouched():
    """A 2+-required-arg command with an optional bracket in front
    (\\href[opt]{url}{title}) still follows the "leave the whole span
    untouched" rule for 2+ args -- the bracket must be included in the
    untouched span, not dropped separately."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"\href[opt]{https://example.com}{Paper Title} end.",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert r"\href[opt]{https://example.com}{Paper Title}" in prompt


# --- sanitize_latex tri-state: default (None) auto-detects; explicit True/
# False overrides it either way (codex P2, PR #77 round 9 -> round 10) ---


@pytest.mark.asyncio
async def test_default_auto_detects_real_latex_source_and_sanitizes():
    """Real LaTeX-source-shaped input (structural markers: \\documentclass,
    \\begin{document}, \\section{...}) auto-triggers sanitization with NO
    explicit sanitize_latex argument -- the whole point of detect-then-
    sanitize is that a caller with genuinely TeX-derived input doesn't
    have to know to opt in."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    data = (
        r"\documentclass{article}" "\n"
        r"\begin{document}" "\n"
        r"\section{Biography}" "\n"
        r"\textit{Ada Lovelace} was born in 1815." "\n"
        r"\end{document}"
    )
    await role.structure(data=data, schema=PersonRecord, purpose="p")

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "\\documentclass" not in prompt
    assert "\\begin{document}" not in prompt
    assert "\\section" not in prompt
    assert "\\textit" not in prompt
    assert "Ada Lovelace" in prompt


@pytest.mark.asyncio
async def test_default_does_not_sanitize_plain_prose():
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    data = "Ada Lovelace was born in London in 1815 and worked with Charles Babbage."
    await role.structure(data=data, schema=PersonRecord, purpose="p")

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert data in prompt  # byte-for-byte untouched


@pytest.mark.asyncio
async def test_default_does_not_sanitize_windows_path():
    """A Windows path is all backslashes and no braces -- exactly the
    non-LaTeX content that got silently mangled when sanitization ran
    unconditionally (codex P2, PR #77 round 9). Must not auto-trigger."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    data = r"See C:\Users\foo\bar.txt and C:\Windows\System32\drivers\etc\hosts."
    await role.structure(data=data, schema=PersonRecord, purpose="p")

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert data in prompt


@pytest.mark.asyncio
async def test_default_does_not_sanitize_regex_heavy_text():
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    data = r"Pattern: \d+\s*\w+ matches identifiers like \b[A-Z]\w*\b in text."
    await role.structure(data=data, schema=PersonRecord, purpose="p")

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert data in prompt


@pytest.mark.asyncio
async def test_default_does_not_sanitize_light_inline_math():
    """A single incidental $...$ pair (e.g. a price range) is below the
    auto-detection threshold -- real TeX source has MANY such patterns,
    not one (codex P2, PR #77 round 9 finding: this exact shape of input
    was being corrupted by unconditional sanitization)."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    data = "The price is $5 today and $10 tomorrow."
    await role.structure(data=data, schema=PersonRecord, purpose="p")

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert data in prompt


@pytest.mark.asyncio
async def test_default_does_not_sanitize_multiple_currency_mentions():
    """Bare $...$ pairs are NOT counted as a weak LaTeX signal at all
    (codex P2, PR #77 round 11): ordinary prose mentioning TWO OR MORE
    dollar amounts in the same snippet forms multiple non-overlapping
    $...$-shaped spans by the same naive regex a real inline-math
    detector would use, which met the old >=2 threshold and triggered
    sanitization -- replacing legitimate currency values with "[math]".
    This must not happen regardless of how many $ amounts appear."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    data = "The item costs $5. Shipping is $10. Total varies between $15 and $20."
    await role.structure(data=data, schema=PersonRecord, purpose="p")

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert data in prompt
    assert "[math]" not in prompt


@pytest.mark.asyncio
async def test_explicit_true_overrides_heuristic_on_ambiguous_input():
    """A caller who knows better than the heuristic (e.g. short TeX
    snippet with only one command) can force sanitization on."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    await role.structure(
        data=r"\textit{Ada Lovelace}",
        schema=PersonRecord,
        purpose="p",
        sanitize_latex=True,
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert "\\textit" not in prompt
    assert "Ada Lovelace" in prompt


@pytest.mark.asyncio
async def test_explicit_false_overrides_heuristic_on_real_latex():
    """A caller who wants the raw text regardless (e.g. inspecting the
    literal source) can force sanitization off even on real LaTeX."""
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool)

    data = r"\begin{document}\textit{Ada Lovelace}\end{document}"
    await role.structure(
        data=data, schema=PersonRecord, purpose="p", sanitize_latex=False
    )

    prompt = pool.get_client("structure").calls[0]["prompt"]
    assert data in prompt


@pytest.mark.asyncio
async def test_max_tokens_scales_with_schema_field_count():
    """A fixed max_tokens cap deterministically truncates output for any
    legitimately larger schema — a many-field record or a list-heavy
    RootModel — turning a correctly-extractable input into needs_curation
    for reasons unrelated to extraction quality (codex P2, PR #77 round
    7). The response budget must scale with the target schema's shape."""

    class ManyFields(BaseModel):
        f01: str
        f02: str
        f03: str
        f04: str
        f05: str
        f06: str
        f07: str
        f08: str
        f09: str
        f10: str
        f11: str
        f12: str
        f13: str
        f14: str
        f15: str
        f16: str
        f17: str
        f18: str
        f19: str
        f20: str
        f21: str
        f22: str
        f23: str
        f24: str
        f25: str
        f26: str
        f27: str
        f28: str
        f29: str
        f30: str
        f31: str
        f32: str
        f33: str
        f34: str
        f35: str
        f36: str
        f37: str
        f38: str
        f39: str
        f40: str

    small_pool = _pool([{"name": "Ada", "age": 36}])
    small_role = StructureRole(small_pool)
    await small_role.structure(data="text", schema=PersonRecord, purpose="p")
    small_max_tokens = small_pool.get_client("structure").calls[0]["max_tokens"]

    big_response = {f"f{i:02d}": "x" for i in range(1, 41)}
    big_pool = _pool([big_response])
    big_role = StructureRole(big_pool)
    await big_role.structure(data="text", schema=ManyFields, purpose="p")
    big_max_tokens = big_pool.get_client("structure").calls[0]["max_tokens"]

    assert big_max_tokens > small_max_tokens


@pytest.mark.asyncio
async def test_max_tokens_override_is_respected():
    pool = _pool([{"name": "Ada", "age": 36}])
    role = StructureRole(pool, max_tokens=999)

    await role.structure(data="text", schema=PersonRecord, purpose="p")

    assert pool.get_client("structure").calls[0]["max_tokens"] == 999
