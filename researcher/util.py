"""Small shared utilities for researcher CLI/server surfaces."""

from __future__ import annotations


def split_csv(value: str) -> list[str]:
    """Return non-empty comma-separated values with surrounding whitespace removed."""
    return [part.strip() for part in str(value or "").split(",") if part.strip()]
