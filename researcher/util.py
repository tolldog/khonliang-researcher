"""Small shared utilities for researcher CLI/server surfaces."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator


def split_csv(value: str) -> list[str]:
    """Return non-empty comma-separated values with surrounding whitespace removed."""
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def github_repo_key(source: str) -> str | None:
    """Return owner/repo for supported GitHub URL forms, or None for local paths."""
    value = str(source or "").strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith(("https://github.com/", "http://github.com/")):
        parts = value.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"
        return ""
    if value.startswith("git@github.com:"):
        parts = value.removeprefix("git@github.com:").split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"
        return ""
    return None


@contextmanager
def repo_tree(source: str, *, prefix: str = "researcher_repo_") -> Iterator[Path]:
    """Yield a local repo tree for either a local path or a GitHub repo URL.

    GitHub sources are shallow-cloned into a temporary directory and cleaned up
    when the context exits. Local sources are yielded unchanged.
    """
    repo_key = github_repo_key(source)
    if repo_key == "":
        raise ValueError(f"Invalid GitHub URL: {source}")
    if repo_key is None:
        path = Path(source)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {source}")
        yield path
        return

    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth=1", f"https://github.com/{repo_key}.git", tmp_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Clone failed: {proc.stderr[:200]}")
        yield Path(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
