"""Tests for FR 97eb7550 — ingest_github / scan_codebase architectural-intent
extraction.

Two slices:
  B. ``collapse_capability_families`` — roll up enumerated README capability
     families (e.g. 24x "Language Support for <lang>") so they don't dilute
     relevance scoring (dog_36ae942f).
  A. ``Synthesizer.scan_codebase`` whole-repo consolidation pass — replaces the
     first-chunk-wins ``all_architectures[0]`` selection with one LLM pass over
     the dependency graph + module names + description, naming the repo's intent
     and promoting the most representative capability (dog_6d734159).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from researcher.pipeline import collapse_capability_families


# ---------------------------------------------------------------- slice B ----

def test_collapse_rolls_up_large_enumerated_family():
    caps = [f"Language Support for {lang}" for lang in
            ("English", "French", "German", "Spanish", "Italian")]
    out = collapse_capability_families(caps)
    assert len(out) == 1
    assert out[0].startswith("Language Support for (5 variants:")
    assert "English" in out[0]


def test_collapse_preserves_distinct_capabilities_and_order():
    caps = [
        "Conversation memory persistence",            # distinct, kept
        "Language Support for English",
        "Language Support for French",
        "Language Support for German",
        "MCP tool integration",                        # distinct, kept
    ]
    out = collapse_capability_families(caps)
    assert out[0] == "Conversation memory persistence"
    assert out[-1] == "MCP tool integration"
    # the 3-member family collapsed to exactly one entry, at its first position
    assert sum(c.startswith("Language Support for (") for c in out) == 1
    assert len(out) == 3


def test_collapse_leaves_small_families_untouched():
    # Only 2 members — below _CAP_FAMILY_MIN (3): must stay verbatim.
    caps = ["Export to PDF", "Export to CSV"]
    assert collapse_capability_families(caps) == caps


def test_collapse_never_touches_short_capabilities():
    # Two-word capabilities have no >=2-word prefix + trailing token → never group.
    caps = ["GEDCOM parsing", "Relevance scoring", "Concept graphing"]
    assert collapse_capability_families(caps) == caps


def test_collapse_empty_is_noop():
    assert collapse_capability_families([]) == []


# ---------------------------------------------------------------- slice A ----

class _FakeReviewer:
    """Returns the consolidation answer when it sees the consolidation prompt,
    otherwise the per-chunk scan answer. Branch on the unique 'OVERARCHING'
    marker that only ``_SCAN_CONSOLIDATE_PROMPT`` carries."""

    def __init__(self, *, chunk_arch: str, chunk_caps, consolidate: dict | None):
        self.chunk_arch = chunk_arch
        self.chunk_caps = chunk_caps
        self.consolidate = consolidate

    async def generate(self, *, prompt, system, temperature, max_tokens):
        if "OVERARCHING" in prompt:
            if self.consolidate is None:
                return "not json — force fallback"
            return json.dumps(self.consolidate)
        return json.dumps({
            "capabilities": list(self.chunk_caps),
            "imports_from": {},
            "architecture": self.chunk_arch,
        })


class _FakePool:
    def __init__(self, reviewer):
        self._reviewer = reviewer

    def get_client(self, _role):
        return self._reviewer


def _make_synth(reviewer):
    from researcher.synthesizer import Synthesizer
    synth = Synthesizer.__new__(Synthesizer)
    synth.pool = _FakePool(reviewer)
    return synth


@asynccontextmanager
async def _yield_repo(repo_dir):
    yield repo_dir


async def _run_scan(tmp_path, reviewer):
    # One parseable module so the AST phase yields a non-empty module_map.
    (tmp_path / "proxy.py").write_text(
        "class ShellProxy:\n"
        "    def intercept(self, cmd):\n"
        "        return cmd\n"
        "    def compress(self, output):\n"
        "        return output[:10]\n"
    )
    synth = _make_synth(reviewer)
    with patch("researcher.util.async_repo_tree",
               lambda p, prefix="": _yield_repo(tmp_path)):
        result = await synth.scan_codebase(
            project_name="o/r",
            repo_path=str(tmp_path),
            description="sits between coding agents and the shell",
            dependencies="none",
        )
    assert result.success, result.content
    return json.loads(result.content)


@pytest.mark.asyncio
async def test_consolidation_overrides_first_chunk_architecture(tmp_path):
    """The misleading per-chunk architecture ('CLI tool') is replaced by the
    consolidated intent, and the consolidated top capability is promoted to the
    front of the capability list."""
    reviewer = _FakeReviewer(
        chunk_arch="CLI tool",
        chunk_caps=["Run benchmark sessions"],
        consolidate={
            "architecture": "shell-output compression proxy",
            "top_capability": "Compress shell command output",
        },
    )
    data = await _run_scan(tmp_path, reviewer)
    assert data["architecture"] == "shell-output compression proxy"
    assert data["capabilities"][0] == "Compress shell command output"
    # the original chunk capability is retained behind the promoted one
    assert "Run benchmark sessions" in data["capabilities"]


@pytest.mark.asyncio
async def test_consolidation_failure_falls_back_to_first_chunk(tmp_path):
    """If the consolidation call returns unparseable output, the legacy
    first-chunk architecture is preserved (no regression / no crash)."""
    reviewer = _FakeReviewer(
        chunk_arch="CLI tool",
        chunk_caps=["Run benchmark sessions"],
        consolidate=None,  # -> non-JSON -> fallback
    )
    data = await _run_scan(tmp_path, reviewer)
    assert data["architecture"] == "CLI tool"
    assert data["capabilities"] == ["Run benchmark sessions"]


@pytest.mark.asyncio
async def test_consolidation_promoted_capability_is_deduped(tmp_path):
    """When the consolidated top_capability already exists in the accumulated
    list, it is moved to the front rather than duplicated."""
    reviewer = _FakeReviewer(
        chunk_arch="CLI tool",
        chunk_caps=["Run benchmark sessions", "Compress shell command output"],
        consolidate={
            "architecture": "shell-output compression proxy",
            "top_capability": "Compress shell command output",
        },
    )
    data = await _run_scan(tmp_path, reviewer)
    assert data["capabilities"][0] == "Compress shell command output"
    assert data["capabilities"].count("Compress shell command output") == 1
