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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

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
    """

    def __init__(
        self,
        model_pool: Any,
        hot_role: str = "structure",
        escalation_role: str = "reviewer",
    ):
        self._pool = model_pool
        self._hot_role = hot_role
        self._escalation_role = escalation_role

    async def structure(
        self,
        data: str,
        schema: Type[BaseModel],
        purpose: str,
        project: str = "",
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
        """
        schema_name = schema.__name__
        schema_json = schema.model_json_schema()
        prompt = _build_prompt(data, purpose, project)
        errors: List[str] = []

        hot_client = self._pool.get_client(self._hot_role)
        hot_model = getattr(hot_client, "model", self._hot_role)

        for attempt in range(1, 3):  # hot tier: initial attempt + one same-tier retry
            record, err = await self._try_extract(
                hot_client, hot_model, prompt, schema, schema_json
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
                )
            errors.append(f"attempt {attempt} ({hot_model}): {err}")

        esc_client = self._pool.get_client(self._escalation_role)
        esc_model = getattr(esc_client, "model", self._escalation_role)
        record, err = await self._try_extract(
            esc_client, esc_model, prompt, schema, schema_json
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
            errors=errors,
        )

    async def _try_extract(
        self,
        client: Any,
        model_name: str,
        prompt: str,
        schema: Type[BaseModel],
        schema_json: Dict[str, Any],
    ) -> Tuple[Optional[BaseModel], Optional[str]]:
        """One generate+validate attempt. Never raises — errors come back as a string."""
        try:
            raw = await client.generate_json(
                prompt=prompt,
                schema=schema_json,
                system=_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=2000,
                model=model_name,
                constrained=True,
            )
        except Exception as e:  # LLM transport failure (timeout, unavailable, etc.)
            return None, f"generation failed: {e}"

        if not isinstance(raw, dict):
            return None, f"non-object JSON response: {raw!r}"

        try:
            return schema.model_validate(raw), None
        except ValidationError as e:
            return None, f"schema validation failed: {e}"


def _build_prompt(data: str, purpose: str, project: str) -> str:
    lines = [f"Purpose: {purpose}"]
    if project:
        lines.append(f"Project: {project}")
    lines.append("")
    lines.append("Extract the schema-conforming record from this text:")
    lines.append("")
    lines.append(data)
    return "\n".join(lines)
