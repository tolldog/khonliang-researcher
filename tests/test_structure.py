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
