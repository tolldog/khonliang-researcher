"""Tests for researcher utility helpers."""

import pytest

from researcher.util import parse_branch_specs


def test_parse_branch_specs_accepts_repeated_and_semicolon_values():
    specs = parse_branch_specs([
        "cache:Prompt Caching,Token Cost",
        "sessions:Session Hygiene;llm:Prompt Hacking",
    ])

    assert specs == [
        {"label": "cache", "seeds": ["Prompt Caching", "Token Cost"]},
        {"label": "sessions", "seeds": ["Session Hygiene"]},
        {"label": "llm", "seeds": ["Prompt Hacking"]},
    ]


@pytest.mark.parametrize("spec", ["missing-colon", ":seed", "empty:"])
def test_parse_branch_specs_rejects_malformed_specs(spec):
    with pytest.raises(ValueError, match="Invalid branch spec"):
        parse_branch_specs(spec)
