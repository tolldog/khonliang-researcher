"""Small shared utilities for researcher CLI/server surfaces."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator


class RepoTreeError(RuntimeError):
    """Raised when a remote repository cannot be materialized as a local tree."""


def split_csv(value: str) -> list[str]:
    """Return non-empty comma-separated values with surrounding whitespace removed."""
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def parse_branch_specs(value: str | list[str] | tuple[str, ...]) -> list[dict]:
    """Parse investigation branch specs.

    Specs use ``label:seed one,seed two``. Multiple specs may be provided as a
    semicolon-separated string or as repeated CLI option values.
    """
    if isinstance(value, str):
        raw_specs = [part.strip() for part in value.split(";") if part.strip()]
    else:
        raw_specs = []
        for item in value:
            raw_specs.extend(part.strip() for part in str(item).split(";") if part.strip())

    specs = []
    for raw in raw_specs:
        label, sep, seed_text = raw.partition(":")
        label = label.strip()
        seeds = split_csv(seed_text) if sep else []
        specs.append({
            "label": label,
            "seeds": seeds,
        })
    return specs


def _github_repo(source: str) -> tuple[str, str] | None:
    """Return (owner/repo, clone_url) for supported GitHub URL forms."""
    value = str(source or "").strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value in {"https://github.com", "http://github.com", "git@github.com:"}:
        raise ValueError(f"Invalid GitHub URL: {source}")
    if value.startswith(("https://github.com/", "http://github.com/")):
        parts = value.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            key = f"{parts[0]}/{parts[1]}"
            return key, f"https://github.com/{key}.git"
        raise ValueError(f"Invalid GitHub URL: {source}")
    if value.startswith("git@github.com:"):
        parts = value.removeprefix("git@github.com:").split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            key = f"{parts[0]}/{parts[1]}"
            return key, f"git@github.com:{key}.git"
        raise ValueError(f"Invalid GitHub URL: {source}")
    return None


def github_repo_key(source: str) -> str | None:
    """Return owner/repo for supported GitHub URL forms, or None for local paths.

    Raises ValueError when the input looks like a GitHub URL but does not name
    both an owner and repository.
    """
    parsed = _github_repo(source)
    return parsed[0] if parsed else None


@contextmanager
def repo_tree(source: str, *, prefix: str = "researcher_repo_") -> Iterator[Path]:
    """Yield a local repo tree for either a local path or a GitHub repo URL.

    GitHub sources are shallow-cloned into a temporary directory and cleaned up
    when the context exits. Local sources are yielded unchanged.
    """
    parsed = _github_repo(source)
    if parsed is None:
        path = Path(source)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {source}")
        yield path
        return

    _, clone_url = parsed
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth=1", clone_url, tmp_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RepoTreeError(f"Clone failed: {proc.stderr[:200]}")
        yield Path(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
