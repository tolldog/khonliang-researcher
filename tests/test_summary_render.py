"""Tests for researcher.server._render_summary_markdown.

Covers the distill_paper formatting fix from fr_researcher_48ab86ae:
the summary block is now rendered as markdown sections instead of a
``json.dumps`` envelope.
"""

from __future__ import annotations

from researcher.server import _render_summary_markdown


def test_renders_full_schema_as_markdown_sections():
    summary = {
        "title": "Some Paper",
        "authors": ["Alice", "Bob"],
        "abstract": "We did the thing.",
        "key_findings": [
            "Finding one with detail.",
            "Finding two with numbers.",
        ],
        "methods": ["Method A", "Method B"],
        "results": ["98.7% accuracy", "2x speedup"],
        "limitations": ["Tested on a small corpus"],
        "domains": ["nlp", "multi-agent"],
        "keywords": ["GRPO", "Dec-POMDP"],
    }
    out = _render_summary_markdown(summary)

    # No raw JSON syntax leaking through.
    assert "{" not in out
    assert "}" not in out
    assert '"authors"' not in out

    # Section headers + bullets render.
    assert out.startswith("## Summary")
    assert "**Authors:** Alice, Bob" in out
    assert "**Abstract:** We did the thing." in out
    assert "### Key findings" in out
    assert "- Finding one with detail." in out
    assert "### Methods" in out
    assert "- Method A" in out
    assert "### Results" in out
    assert "- 98.7% accuracy" in out
    assert "### Limitations" in out
    assert "**Domains:** nlp, multi-agent" in out
    assert "**Keywords:** GRPO, Dec-POMDP" in out


def test_omits_empty_sections():
    """Empty / missing fields don't generate empty headers."""
    summary = {
        "abstract": "Just an abstract.",
    }
    out = _render_summary_markdown(summary)
    assert "**Abstract:** Just an abstract." in out
    assert "### Key findings" not in out
    assert "### Methods" not in out
    assert "**Authors:**" not in out
    assert "**Domains:**" not in out


def test_surfaces_unknown_schema_keys_under_other():
    """Schema additions shouldn't disappear silently."""
    summary = {
        "abstract": "x",
        "novel_field": "this should still surface",
        "another_extra": 42,
    }
    out = _render_summary_markdown(summary)
    assert "### Other" in out
    assert "**novel_field:** this should still surface" in out
    assert "**another_extra:** 42" in out


def test_handles_non_dict_input():
    assert _render_summary_markdown(None) == ""  # type: ignore[arg-type]
    assert _render_summary_markdown({}) == ""
    assert _render_summary_markdown("garbage") == ""  # type: ignore[arg-type]


def test_renders_more_compactly_than_json_dumps():
    """The whole point of this fix: markdown beats json.dumps on
    token economy *and* readability. Sanity-check the size win on a
    representative payload.
    """
    import json

    summary = {
        "title": "Some Paper",
        "authors": ["Alice", "Bob"],
        "abstract": "We did the thing and learned a result.",
        "key_findings": ["Finding A", "Finding B", "Finding C"],
        "methods": ["Method M"],
        "results": ["98% acc"],
        "limitations": ["Small corpus"],
        "domains": ["nlp"],
        "keywords": ["foo", "bar"],
    }
    md = _render_summary_markdown(summary)
    js = json.dumps(summary, indent=2)
    assert len(md) < len(js), (
        f"markdown ({len(md)} chars) should be shorter than json.dumps "
        f"({len(js)} chars); got {md!r} vs {js!r}"
    )
