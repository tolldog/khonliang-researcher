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

    # No raw JSON syntax leaking through. Quoted-key signature
    # ('"authors":') is the load-bearing assertion — bare braces
    # would also fail many legitimate contents (math/set notation
    # in an abstract), so we don't gate on those.
    assert '"authors":' not in out
    assert '"key_findings":' not in out

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
    assert _render_summary_markdown(None) == ""
    assert _render_summary_markdown({}) == ""
    assert _render_summary_markdown("garbage") == ""


def test_returns_empty_when_only_unrenderable_known_keys_present():
    """`title` is rendered by the surrounding distill_paper handler,
    not by this helper. A summary that contains only `title` (or
    other keys that don't produce body content) must not emit a
    bare "## Summary" header.
    """
    assert _render_summary_markdown({"title": "A Paper"}) == ""
    # Same for keys whose values are present but render-empty.
    assert _render_summary_markdown({"title": "A Paper", "authors": []}) == ""
    assert _render_summary_markdown({"abstract": "   "}) == ""


def test_filters_whitespace_only_list_items():
    """Blank / whitespace-only entries in list fields drop out
    instead of producing "- " bullets, and a section that has only
    blank entries doesn't emit its header at all.
    """
    summary = {
        "key_findings": ["", "  ", "real finding"],
        "methods": ["", "   "],
    }
    out = _render_summary_markdown(summary)
    assert "### Key findings" in out
    assert "- real finding" in out
    # No "- " on its own line (would indicate an empty bullet rendered).
    assert "\n- \n" not in out and not out.endswith("- ")
    # Methods section had only blank entries → no header.
    assert "### Methods" not in out


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
