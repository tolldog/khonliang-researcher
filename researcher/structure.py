"""structure(): schema-constrained extraction, distill's typed sibling.

fr_researcher_d813ad52. ONE transform engine, many schema owners:
unstructured text in, a schema-conforming record out — no second LLM
stack. ``distill`` produces researcher's own fixed shapes (summary,
triples, assessment); ``structure`` is the general-purpose counterpart —
any caller-supplied Pydantic model is a valid target schema, so
librarian's ``brief()`` and other catalog consumers compose the same
primitive instead of hand-rolling their own extraction loop.

Every extraction is PROVENANCE-STAMPED (model id + spec version + schema
name) so "re-run structure over everything below quality X" is a routine
batch job once a better local model lands — group prior extractions by
``(model_used, spec_version)`` and re-target the ones below the current
bar. See ``StructureResult.provenance()``.

Mechanics — validate -> retry -> escalate -> needs_curation, never silent
junk:
  1. Hot tier (local, cheap) generates a candidate.
  2. Validate against the caller's schema. On failure, retry once more on
     the SAME tier (transient decode noise is common; a repeat call often
     lands on-schema without spending an escalation).
  3. Still invalid after two hot-tier attempts -> escalate to a stronger
     configured tier for one more attempt.
  4. Still invalid -> return ``needs_curation=True``. Never raises for a
     content failure and never returns an unvalidated record — a
     terminal failure is a stamped, inspectable result, not a crash or a
     silently-dropped row.

This mirrors the tier-ladder already established in ``researcher.roles``
(``SummarizerRole``'s ``MODEL_TIERS`` + ``FALLBACK_MODEL`` retry-on-failure
pattern) rather than ``khonliang.routing.strategies.CascadeStrategy`` —
Cascade's escalation signal is a text heuristic (response length, hedging
markers) meant for free-text confidence, which doesn't fit a binary
schema-valid/invalid gate, and it isn't wired into this repo today.

Constrained-decoding note (load-bearing, read before assuming otherwise):
Ollama >=0.30 supports true JSON-schema-constrained decoding via
``format=<schema-dict>`` (the model literally cannot emit an off-schema
token). The ``OllamaClient`` this repo depends on
(``khonliang.client.OllamaClient.generate_json``) does NOT yet expose
that path — passing ``schema=`` only appends the schema to the system
prompt as text and still requests loose ``format="json"`` mode. Until
that client is extended (tracked as a fast-follow in the ollama-khonliang
library; out of scope here — it's a separate repo/review cycle), this
module's guarantee ("never a persisted off-schema record") is delivered
by the validate/retry/escalate loop below, not by the decoder. That is a
cost difference (more escalations than grammar-level constraint would
need), not a correctness difference: nothing that fails
``schema.model_validate`` is ever returned as a success. Swapping in true
constrained decoding later is a small, additive change to this module
once the client passthrough lands.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

#: Conservative char cap applied to ``data`` before prompting, mirroring
#: SummarizerRole's medium-tier truncation (researcher/roles.py
#: MODEL_TIERS). Without this, a caller handing structure() a full paper
#: or long web page deterministically overflows the model's context
#: window, burning every retry/escalation attempt and returning
#: needs_curation for input that was never actually malformed (codex P2,
#: PR #77 round 3). Override via the StructureRole constructor for a
#: caller with a known larger/smaller budget.
DEFAULT_MAX_CHARS = 12_000

#: Floor for the per-attempt response token budget (``num_predict``).
#: Matches SummarizerRole's own ``max_tokens`` ceiling — a reasonable
#: minimum for a single small-to-medium record. ``_estimate_max_tokens``
#: below scales UP from this floor for schemas with many fields; it never
#: goes lower, so a tiny schema still gets a comfortable budget.
DEFAULT_MAX_TOKENS = 4000

#: Bump when the extraction contract (prompt shape, retry/escalation
#: policy) changes materially enough that prior extractions should be
#: considered lower-quality than ones made under the new contract —
#: mirrors researcher/self_catalog.py's SCHEMA_VERSION convention, but
#: versions the *extraction process* rather than the catalog record shape.
SPEC_VERSION = 1

_SYSTEM_PROMPT = (
    "You extract structured data from unstructured text. Respond with "
    "valid JSON only, conforming exactly to the given schema. No "
    "markdown, no explanations, no extra fields."
)


@dataclass
class StructureResult:
    """Outcome of a ``structure()`` call.

    ``record`` is populated (a validated instance of the caller's schema)
    iff ``success`` is True. ``needs_curation`` is the terminal-failure
    flag: a caller must branch on it and queue for human/stronger review
    rather than treat ``success=False`` as a silent no-op.
    """

    success: bool
    record: Optional[BaseModel] = None
    needs_curation: bool = False
    purpose: str = ""
    project: str = ""
    attempts: int = 0
    escalated: bool = False
    model_used: str = ""
    spec_version: int = SPEC_VERSION
    schema_name: str = ""
    errors: List[str] = field(default_factory=list)
    stamped_at: float = field(default_factory=time.time)
    # True when ``data`` exceeded ``max_chars`` and was truncated before
    # prompting. A schema-valid ``success=True`` result built from
    # truncated input may still be INCOMPLETE (relevant fields can live
    # past the cutoff in a long paper/page) — this is NOT folded into
    # ``needs_curation`` (the extraction mechanism itself didn't fail),
    # but it must not be silent either: a caller doing anything
    # completeness-sensitive should check this flag explicitly (codex P2,
    # PR #77 round 5).
    truncated: bool = False

    def provenance(self) -> Dict[str, Any]:
        """Provenance stamp: model id + spec version + schema identity.

        The unit a "re-run structure over everything below quality X"
        dispatcher batch job would group and filter on.
        """
        return {
            "model": self.model_used,
            "spec_version": self.spec_version,
            "schema": self.schema_name,
            "escalated": self.escalated,
            "truncated": self.truncated,
            "stamped_at": self.stamped_at,
        }


class StructureRole:
    """Schema-constrained extraction: unstructured text -> validated record.

    Not a ``khonliang.roles.base.BaseRole`` subclass — ``BaseRole.handle()``
    targets one fixed output shape per role instance, while ``structure()``
    targets a caller-supplied schema per call. It reuses the same
    ``ModelPool``/``OllamaClient`` primitives ``BaseRole`` subclasses use
    (no new LLM client/session infrastructure), and follows the same
    tier-ladder shape as ``researcher.roles.SummarizerRole``.

    Args:
        model_pool: khonliang ``ModelPool`` — same instance
            ``ResearchPipeline`` already builds for the other roles.
        hot_role: ModelPool role key for the cheap first-pass tier.
            Defaults to the ``"structure"`` role entry (config.yaml
            ``models.structure``, falls back to the summarizer's model
            when unset — see ``create_pipeline``).
        escalation_role: ModelPool role key for the stronger tail tier.
            Defaults to ``"reviewer"`` — already the heaviest local model
            configured in this repo (qwen2.5:32b by default), reused here
            rather than adding a second escalation-specific config key.
        max_chars: Char cap applied to ``data`` before prompting (see
            ``DEFAULT_MAX_CHARS``).
        max_tokens: Fixed response token budget override. When ``None``
            (the default), the budget is derived per call from the
            target schema's shape via ``_estimate_max_tokens`` — a
            hardcoded cap here would deterministically truncate any
            legitimately larger schema (a list-heavy ``RootModel``, a
            many-field nested record) on every attempt, landing in
            ``needs_curation`` for reasons that have nothing to do with
            extraction quality (codex P2, PR #77 round 7). Set this only
            to pin an exact budget for every call regardless of schema.
    """

    def __init__(
        self,
        model_pool: Any,
        hot_role: str = "structure",
        escalation_role: str = "reviewer",
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: Optional[int] = None,
    ):
        self._pool = model_pool
        self._hot_role = hot_role
        self._escalation_role = escalation_role
        self._max_chars = max_chars
        self._max_tokens = max_tokens

    async def structure(
        self,
        data: str,
        schema: Type[BaseModel],
        purpose: str,
        project: str = "",
        sanitize_latex: Optional[bool] = None,
    ) -> StructureResult:
        """Extract ``data`` into ``schema``, escalating through the tier ladder.

        Args:
            data: Unstructured source text.
            schema: Pydantic model the output must validate against — the
                SAME model a caller uses for validation-on-write, per the
                FR ("no second LLM stack, no second schema").
            purpose: Why this extraction is happening — frames the prompt
                and is stamped onto the result for audit trail.
            project: Project scope. Project-aware per the FR ("no
                cross-project few-shot examples") — this v1 injects no
                few-shot examples at all, so ``project`` is provenance/
                audit metadata for now, not yet a prompt-shaping input.
                Kept as an explicit param (not smuggled into ``purpose``
                text) so future few-shot support has a stable place to
                read it from.
            sanitize_latex: Tri-state control over LaTeX-only
                preprocessing (see ``_strip_latex_noise``):
                  - ``None`` (default) — AUTO-DETECT from ``data`` itself
                    via ``_looks_like_latex`` and only sanitize when real
                    LaTeX signals are present.
                  - ``True`` — force sanitization on regardless of the
                    heuristic (a caller who knows the source is
                    TeX-derived but the heuristic is unsure, e.g. very
                    short input).
                  - ``False`` — force it off; ``data`` goes to the model
                    completely unmodified.
                This started as a plain opt-in (default off) after an
                earlier version ran the equivalent cleanup
                unconditionally on every call and silently mangled
                non-LaTeX backslash content — a Windows path, a regex, an
                escape sequence — for callers who never had TeX-derived
                input (codex P2, PR #77 round 9). A plain opt-in flag
                pushed the content-type judgment entirely onto the
                caller, though, and LaTeX handling IS genuinely valuable
                for sources that actually use it (arXiv abstracts, LaTeX
                papers) — so detect-then-sanitize replaces "caller must
                know in advance" with a conservative content sniff,
                while keeping the explicit override for a caller who
                knows better than the heuristic either way.
        """
        schema_name = schema.__name__
        schema_json = schema.model_json_schema()

        should_sanitize = (
            _looks_like_latex(data) if sanitize_latex is None else sanitize_latex
        )

        # Truncate to a conservative char budget BEFORE prompting, not
        # after a failed attempt: an oversized input would otherwise burn
        # every hot-tier/escalation attempt and land in needs_curation for
        # reasons that have nothing to do with the model's extraction
        # quality (codex P2, PR #77 round 3). This always applies,
        # independent of ``sanitize_latex`` — context-window overflow
        # protection is unrelated to LaTeX and is not the thing that was
        # made opt-in/auto-detected.
        text = _strip_latex_noise(data) if should_sanitize else data
        truncated = len(text) > self._max_chars
        if truncated:
            text = text[: self._max_chars]
        prompt = _build_prompt(text, purpose, project)
        errors: List[str] = []

        max_tokens = self._max_tokens or _estimate_max_tokens(schema_json)

        hot_client = self._pool.get_client(self._hot_role)
        hot_model = getattr(hot_client, "model", self._hot_role)

        for attempt in range(1, 3):  # hot tier: initial attempt + one same-tier retry
            record, err = await self._try_extract(
                hot_client, hot_model, prompt, schema, schema_json, max_tokens
            )
            if record is not None:
                return StructureResult(
                    success=True,
                    record=record,
                    purpose=purpose,
                    project=project,
                    attempts=attempt,
                    escalated=False,
                    model_used=hot_model,
                    schema_name=schema_name,
                    truncated=truncated,
                )
            errors.append(f"attempt {attempt} ({hot_model}): {err}")

        esc_client = self._pool.get_client(self._escalation_role)
        esc_model = getattr(esc_client, "model", self._escalation_role)
        record, err = await self._try_extract(
            esc_client, esc_model, prompt, schema, schema_json, max_tokens
        )
        if record is not None:
            return StructureResult(
                success=True,
                record=record,
                purpose=purpose,
                project=project,
                attempts=3,
                escalated=True,
                model_used=esc_model,
                schema_name=schema_name,
                truncated=truncated,
            )
        errors.append(f"escalation ({esc_model}): {err}")

        logger.warning(
            "structure(): needs_curation for purpose=%r project=%r schema=%s — %s",
            purpose, project, schema_name, errors,
        )
        return StructureResult(
            success=False,
            needs_curation=True,
            purpose=purpose,
            project=project,
            attempts=3,
            escalated=True,
            model_used=esc_model,
            schema_name=schema_name,
            truncated=truncated,
            errors=errors,
        )

    async def _try_extract(
        self,
        client: Any,
        model_name: str,
        prompt: str,
        schema: Type[BaseModel],
        schema_json: Dict[str, Any],
        max_tokens: int,
    ) -> Tuple[Optional[BaseModel], Optional[str]]:
        """One generate+validate attempt. Never raises — errors come back as a string."""
        try:
            raw = await client.generate_json(
                prompt=prompt,
                schema=schema_json,
                system=_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=max_tokens,
                model=model_name,
                constrained=True,
            )
        except Exception as e:  # LLM transport failure (timeout, unavailable, etc.)
            return None, f"generation failed: {e}"

        # Don't gate on isinstance(raw, dict) before validating: a caller's
        # schema may be a Pydantic RootModel wrapping a list/scalar (e.g.
        # RootModel[list[Item]]), for which a non-dict payload is perfectly
        # valid. Let model_validate be the single source of truth for
        # "is this on-schema" (codex P2, PR #77 round 1) — non-dict/
        # non-schema-conforming payloads still fail here and fall through
        # to the same retry/escalate/needs_curation path.
        try:
            return schema.model_validate(raw), None
        except ValidationError as e:
            return None, f"schema validation failed: {e}"
        # Deliberately narrow to ValidationError only (codex P2, PR #77 round
        # 2): a non-ValidationError here means the caller's schema itself is
        # broken (e.g. a buggy custom validator raising TypeError/
        # AttributeError) — a real application bug, not a data-quality
        # problem with the model's output. Swallowing it into
        # needs_curation would misclassify a deterministic bug as "bad
        # extraction" and hide it in a curation queue across an entire
        # batch instead of surfacing immediately. Let it propagate.


# Structural LaTeX markers: essentially never appear outside real TeX
# source, so a SINGLE occurrence of any of these is sufficient evidence
# on its own (codex P2, PR #77 round 9 follow-up — auto-detection).
#
# ``\*?(?:\[[^\]]*\])?`` after the sectioning commands allows the
# starred (``\section*{``) and optional-arg (``\section[Short]{``) forms
# that ``_process_latex_commands`` already knows how to clean (codex P2,
# PR #77 round 13) — without it, real LaTeX using these common forms
# skipped auto-detection and sanitization never ran.
_LATEX_STRUCTURAL_MARKERS = re.compile(
    r"\\begin\{|\\end\{|\\documentclass|\\usepackage"
    r"|\\section\*?(?:\[[^\]]*\])?\{|\\subsection\*?(?:\[[^\]]*\])?\{|\\chapter\*?(?:\[[^\]]*\])?\{"
    r"|\\cite\{|\\ref\{|\\label\{"
    r"|\$\$|\\\[|\\\]"
)

# Weaker signal: a backslash-letters run immediately followed by a brace
# group (``\command{...}``). Rare outside LaTeX (a Windows path or a
# regex has no braces after the backslash run), but a SINGLE occurrence
# could still be incidental, so this signal only counts once it reaches
# `_LATEX_MIN_WEAK_SIGNALS` hits.
#
# Deliberately does NOT count bare ``$...$`` pairs as a weak signal
# (codex P2, PR #77 round 11): ordinary prose mentioning two dollar
# amounts in the same snippet ("costs $5, shipping is $10") forms two
# non-overlapping ``$...$``-shaped spans by the same naive regex a real
# inline-math detector would use, which met the >=2 threshold and
# triggered sanitization on ordinary currency text — the false positive
# this whole tri-state design exists to prevent. ``$$``/``\[``/``\]``
# DISPLAY math stay in `_LATEX_STRUCTURAL_MARKERS` above (those don't
# false-positive on currency), so real math-heavy LaTeX is still caught
# via display math or the command-density signal below; bare single-$
# inline math is simply not a reliable enough signal on its own to
# justify the false-positive risk.
_LATEX_COMMAND_WITH_ARGS = re.compile(r"\\[a-zA-Z]+\{[^}]*\}")
_LATEX_MIN_WEAK_SIGNALS = 2


def _looks_like_latex(text: str) -> bool:
    """Conservative content sniff: does ``text`` look like real LaTeX source?

    Used by ``structure(..., sanitize_latex=None)`` (the default) to
    auto-detect whether ``_strip_latex_noise`` should run at all, rather
    than requiring every caller to know their content type in advance.

    Checked a maintained library first rather than hand-rolling another
    regex (this exact file has had multiple review rounds of regex-based
    LaTeX-handling bugs — codex P2, PR #77 rounds 3/7/8/11):
      - ``python-magic``/libmagic: NOT installed in this repo or anywhere
        else in the khonliang ecosystem (only the system `libmagic1`
        shared library is present, not the Python binding or the
        `python-magic` package) — would be a new dependency for exactly
        one call site.
      - ``pygments`` (``guess_lexer``): present in this dev venv only as
        an UNDECLARED transitive dependency of ``pytest``
        (`pytest -> pygments>=2.7.2`) — not a production dependency of
        this package, and not guaranteed present in a deployed agent
        that doesn't install test tooling. Manually verified
        ``guess_lexer`` correctly tags real LaTeX source as ``"TeX"`` and
        (usefully) never mistags a Windows path or regex-heavy string as
        ``"TeX"`` either — but making it a real production dependency
        pulls in a full multi-language syntax-highlighting library for
        one lightweight content sniff, which is disproportionate for
        this single call site.
    Neither fit cleanly, so this stays a small, conservative, dependency-
    free heuristic — see module docstring notes on rounds 3/7/8/9/11 for
    why "conservative" (bias away from triggering) matters more here than
    catching every possible LaTeX shape: false negatives (falling back to
    raw, unsanitized text) are SAFE, a caller can still force
    ``sanitize_latex=True``; false positives (sanitizing a Windows path,
    a regex-heavy string, or ordinary prose mentioning currency) are
    exactly what caused the bugs this tri-state design exists to prevent.
    """
    if _LATEX_STRUCTURAL_MARKERS.search(text):
        return True
    return len(_LATEX_COMMAND_WITH_ARGS.findall(text)) >= _LATEX_MIN_WEAK_SIGNALS


def _strip_latex_noise(text: str) -> str:
    """Strip LaTeX markup that reliably breaks JSON generation, preserving content.

    ONLY called when ``structure()`` decides ``should_sanitize`` is True —
    either ``_looks_like_latex`` auto-detected real LaTeX signals (the
    default, ``sanitize_latex=None``) or a caller forced it on explicitly
    (``sanitize_latex=True``, codex P2, PR #77 rounds 9/10) — appropriate
    ONLY for genuinely TeX-derived source text (LaTeX papers, arXiv
    abstracts). Even the brace-balanced version here is not a safe
    default for arbitrary text: it still unconditionally treats
    ``$...$``/``$$...$$`` as math-mode delimiters and any ``\\<letters>``
    run as a LaTeX command — both are common in non-LaTeX text that has
    nothing to do with TeX (a price like ``$5`` next to another ``$10``,
    a Windows path ``C:\\Users\\foo``, a regex escape like ``\\d+``), and
    this function will still mangle those inputs if it's told to run on
    them. That's why this is gated behind detection/an explicit override
    rather than run unconditionally.

    Deliberately narrower than ``researcher.roles._clean_for_json`` in two
    ways, both because structure() promises exact typed fields (unlike
    summarization, which tolerates lossy compression):
      - Does NOT strip non-ASCII characters (a name/title with accents or
        CJK characters must survive unmodified).
      - A SINGLE-braced command like ``\\textit{Ada Lovelace}`` keeps its
        BRACED CONTENT and drops only the command wrapper — real field
        values routinely live inside formatting commands in
        TeX/Markdown-derived text (codex P2, PR #77 round 5). A command
        with TWO OR MORE brace-group arguments (``\\href{url}{title}``,
        ``\\frac{1}{2}``) is left COMPLETELY UNTOUCHED rather than
        guessed at (codex P2, PR #77 round 7). A bare, content-free
        command like ``\\alpha`` has nothing to preserve and is dropped
        outright. Argument scanning is brace-BALANCED (see
        ``_process_latex_commands``), so a nested brace inside a
        single-arg command (``\\textit{Ada {Byron}}``) does not truncate
        the argument early (codex P2, PR #77 round 8).
    """
    # Negative lookbehind on both delimiters: an escaped ``\$`` (a literal
    # currency sign in real TeX, e.g. ``\$5 ... \$10``) must NOT be
    # treated as a math-mode delimiter — without it, the span between two
    # escaped dollar signs was replaced wholesale with "[math]",
    # corrupting legitimate currency values in genuine LaTeX source
    # (codex P2, PR #77 round 13).
    text = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$", "[math]", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$[^$]+(?<!\\)\$", "[math]", text)
    text = _process_latex_commands(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _process_latex_commands(text: str) -> str:
    """Walk ``text``, stripping LaTeX commands with brace-BALANCED argument scanning.

    Deliberately NOT regex-based for argument matching (codex P2, PR #77
    round 8): a pattern like ``\\{[^}]*\\}`` can only ever match up to the
    FIRST ``}``, so it can never handle nesting by construction — that's
    the wrong tool, not a matter of degree. This instead walks the string
    tracking an explicit brace-depth counter per argument
    (``_read_balanced_group``), so a nested brace
    (``\\textit{Ada {Byron}}``) stays inside the argument instead of
    ending it early and leaving a dangling ``}`` in the output.

    Per command, after locating its name:
      - An optional ``*`` (starred variant, e.g. ``\\section*{...}``) and
        an optional bracket-balanced ``[...]`` (LaTeX's optional-argument
        syntax, e.g. ``\\section[Short]{Long}``) are SKIPPED — neither
        counts as a required brace-group argument, and neither is kept in
        the output — before looking for the real ``{...}`` argument(s)
        (codex P2, PR #77 round 11). Without this, a command's argument
        was assumed to start immediately with ``{``, so
        ``\\section*{Biography}``/``\\section[Short]{Long}`` fell into the
        "no args" branch below and were corrupted into a stray
        ``*{Biography}``/``[Short]{Long}`` (the command name lost, but
        the star/bracket wrongly left behind as if it were content).
      - 0 brace-group arguments (a bare command like ``\\alpha``) — drop
        the command name (and any star/bracket after it), keep scanning.
      - 1 argument — unwrap to its content, recursively processed (so a
        nested single-arg command inside it, e.g.
        ``\\textit{\\emph{X}}``, also gets unwrapped). Any skipped ``[...]``
        optional-argument content is discarded, not appended — it's a
        short-form alternate (e.g. a TOC entry), not the main value.
      - 2+ arguments (``\\href{url}{title}``) — leave the ENTIRE command
        (name + star/bracket + every argument, verbatim, including any
        nesting inside them) untouched rather than guessed at.
    An unbalanced/unterminated ``{`` or ``[`` (malformed input) stops
    argument collection at that point rather than scanning to the end of
    the string; whatever arguments were already validly collected are
    still used, and the unterminated remainder is left as ordinary
    literal text.

    Known limitations (documented, not chased further — PR #77 round 12):
    a command's ``*``/``[...]``/``{...}`` must follow the command name with
    no intervening whitespace (``\\textit {Ada}`` is not recognized as
    having an argument); a command with only a star/optional-bracket and
    no required brace argument (``\\item[Pros]``) leaks the bracket content
    back into the output instead of being discarded. LaTeX comments
    (``% ...``) and verbatim/listing blocks (``\\begin{verbatim}``,
    ``\\verb|...|``) are also not specially handled. All are real-LaTeX-
    fidelity gaps (occasional stray syntax fragments reach the model), not
    non-LaTeX-content-corruption bugs — the schema-validation retry/
    escalate/``needs_curation`` loop in ``StructureRole`` remains the
    correctness backstop regardless of sanitizer fidelity.
    """
    name_re = re.compile(r"[a-zA-Z]+")
    out: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "\\":
            out.append(text[i])
            i += 1
            continue

        m = name_re.match(text, i + 1)
        if not m:
            out.append(text[i])  # bare backslash, not a command -- literal
            i += 1
            continue

        after_name = m.end()
        k = after_name

        if k < n and text[k] == "*":  # starred variant, e.g. \section*{...}
            k += 1

        if k < n and text[k] == "[":  # optional arg, e.g. \section[Short]{...}
            _, bracket_end = _read_balanced_group(text, k, "[", "]")
            if bracket_end != k:  # only advance if the brackets balanced
                k = bracket_end

        args: List[str] = []
        while k < n and text[k] == "{":
            arg, end = _read_balanced_group(text, k, "{", "}")
            if arg is None:
                break  # unterminated -- stop collecting, keep what we have
            args.append(arg)
            k = end

        if not args:
            i = after_name  # drop the bare command name only
        elif len(args) == 1:
            out.append(_process_latex_commands(args[0]))
            i = k
        else:
            out.append(text[i:k])  # 2+ args -- leave the whole span untouched
            i = k

    return "".join(out)


def _read_balanced_group(
    text: str, start: int, open_ch: str, close_ch: str
) -> Tuple[Optional[str], int]:
    """Read one ``open_ch...close_ch``-balanced group starting at ``text[start] == open_ch``.

    Tracks a depth counter (+1 on ``open_ch``, -1 on ``close_ch``) so a
    NESTED occurrence of ``open_ch`` inside the group stays part of its
    content instead of ending the group at the first ``close_ch``
    encountered. Returns ``(content, end)`` with ``end`` one past the
    matching closing delimiter, or ``(None, start)`` if the delimiters
    never balance before the end of the string. Used for both LaTeX's
    required ``{...}`` arguments and its optional ``[...]`` arguments —
    same balancing logic, different delimiter pair.
    """
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == open_ch:
            depth += 1
        elif text[idx] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
    return None, start


def _count_schema_fields(node: Any) -> int:
    """Recursively count fields in a JSON schema (properties + array item
    fields + $defs/definitions), for ``_estimate_max_tokens``'s heuristic."""
    if not isinstance(node, dict):
        return 0
    count = 0
    props = node.get("properties")
    if isinstance(props, dict):
        count += len(props)
        for sub in props.values():
            count += _count_schema_fields(sub)
    items = node.get("items")
    if isinstance(items, dict):
        count += _count_schema_fields(items) or 1
    for key in ("$defs", "definitions"):
        defs = node.get(key)
        if isinstance(defs, dict):
            for sub in defs.values():
                count += _count_schema_fields(sub)
    return count


def _estimate_max_tokens(
    schema_json: Dict[str, Any], floor: int = DEFAULT_MAX_TOKENS
) -> int:
    """Rough per-attempt response token budget for a schema, with headroom.

    A fixed cap here silently truncates any legitimately larger schema on
    every attempt — deterministic ``needs_curation`` for reasons that have
    nothing to do with extraction quality (codex P2, PR #77 round 7). This
    is a heuristic, not an exact token count: count schema fields
    (recursively, including array-item and $defs/definitions shapes) and
    budget generously per field, with ``floor`` as a minimum so small
    schemas still get a comfortable budget.
    """
    fields = _count_schema_fields(schema_json) or 1
    return max(floor, fields * 120 + 256)


def _build_prompt(data: str, purpose: str, project: str) -> str:
    lines = [f"Purpose: {purpose}"]
    if project:
        lines.append(f"Project: {project}")
    lines.append("")
    lines.append("Extract the schema-conforming record from this text:")
    lines.append("")
    lines.append(data)
    return "\n".join(lines)
