"""Persistent RSS feed registry.

Replaces the in-code ``researcher/rss.py:DEFAULT_FEEDS`` dict with a
SQLite-backed table so a feed can be added/removed at runtime without a
PR + reviewer + Copilot + restart cycle (fr_researcher_b8b5c008).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Any, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds (
    feed_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feeds_url ON feeds(url);
"""

_UPDATABLE_FIELDS = frozenset({"name", "url", "source", "enabled", "metadata"})


class FeedError(Exception):
    """Raised for feed-registry validation failures (e.g. duplicate URL)."""


def _dumps_metadata(metadata: dict[str, Any]) -> str:
    """``json.dumps`` a metadata dict, converting failures to ``FeedError``.

    A dict can be a dict of non-JSON-serializable values (a set, a custom
    object, ...); letting that raise ``TypeError`` uncaught would bypass
    the FeedError contract the server-side tools rely on (Copilot finding
    on PR #71 round 5).
    """
    try:
        return json.dumps(metadata)
    except TypeError as e:
        raise FeedError(f"metadata is not JSON-serializable: {e}") from e


class FeedStore:
    """SQLite persistence for the feed registry.

    Each connection is explicitly closed (``closing``): sqlite3's
    connection context manager only commits/rolls back, it does NOT
    close it — see IngestWatcherStore / bug_khonliang-developer_be840d83
    for the fd-exhaustion failure mode this avoids.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self._conn()) as conn, conn:
            conn.executescript(_SCHEMA)

    def register_feed(
        self, name: str, url: str, source: str, metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Insert a new feed. Raises ``FeedError`` if the URL already exists."""
        if not name or not url or not source:
            raise FeedError("name, url, and source are all required")
        now = time.time()
        feed_id = f"feed_{uuid.uuid4().hex[:12]}"
        try:
            with closing(self._conn()) as conn, conn:
                existing = conn.execute(
                    "SELECT feed_id FROM feeds WHERE url = ?", (url,)
                ).fetchone()
                if existing is not None:
                    raise FeedError(f"a feed with url {url!r} already exists (feed_id={existing['feed_id']})")
                conn.execute(
                    """
                    INSERT INTO feeds (feed_id, name, url, source, enabled, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (feed_id, name, url, source, now, now, _dumps_metadata(metadata or {})),
                )
        except sqlite3.IntegrityError as e:
            # The pre-check SELECT can't close a race in concurrent/
            # multi-process use — a second writer can insert the same url
            # between our SELECT and INSERT. Surface as FeedError so the
            # server-side tool's `except FeedError` returns a clean error
            # envelope instead of an uncaught crash.
            raise FeedError(f"a feed with url {url!r} already exists: {e}") from e
        return self.get_feed(feed_id)

    def list_feeds(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM feeds"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY name"
        with closing(self._conn()) as conn, conn:
            rows = conn.execute(query).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_feed(self, feed_id: str) -> Optional[dict[str, Any]]:
        with closing(self._conn()) as conn, conn:
            row = conn.execute("SELECT * FROM feeds WHERE feed_id = ?", (feed_id,)).fetchone()
        return _row_to_dict(row) if row is not None else None

    def update_feed(self, feed_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise FeedError(
                f"unsupported fields for update_feed: {', '.join(sorted(unknown))}. "
                f"updatable fields: {', '.join(sorted(_UPDATABLE_FIELDS))}."
            )
        if not fields:
            return self.get_feed(feed_id)
        if "metadata" in fields:
            if not isinstance(fields["metadata"], dict):
                # Anything else (list, int, ...) would reach sqlite3 as an
                # unsupported bind parameter type and raise
                # sqlite3.ProgrammingError — a different exception class
                # than the IntegrityError this method already converts, so
                # it would crash the server-side tool uncaught (Copilot
                # finding on PR #71 round 4). Reject it as FeedError instead.
                raise FeedError(f"metadata must be a dict, got {type(fields['metadata']).__name__}")
            fields = {**fields, "metadata": _dumps_metadata(fields["metadata"])}
        if "enabled" in fields:
            fields = {**fields, "enabled": 1 if fields["enabled"] else 0}
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [time.time(), feed_id]
        try:
            with closing(self._conn()) as conn, conn:
                cur = conn.execute(
                    f"UPDATE feeds SET {set_clause}, updated_at = ? WHERE feed_id = ?",
                    values,
                )
                if cur.rowcount == 0:
                    return None
        except sqlite3.IntegrityError as e:
            # e.g. updating url to one that collides with the unique index —
            # surface as FeedError so the server-side tool's `except FeedError`
            # returns a clean error envelope instead of an uncaught crash.
            raise FeedError(f"update rejected: {e}") from e
        return self.get_feed(feed_id)

    def disable_feed(self, feed_id: str) -> bool:
        """Soft-delete: sets enabled=0, preserves the row and its history."""
        with closing(self._conn()) as conn, conn:
            cur = conn.execute(
                "UPDATE feeds SET enabled = 0, updated_at = ? WHERE feed_id = ?",
                (time.time(), feed_id),
            )
        return cur.rowcount > 0

    def seed_if_empty(self, default_feeds: dict[str, Any]) -> int:
        """Seed any DEFAULT_FEEDS entries not already present, by URL.

        Deliberately NOT gated on "table is empty": if a caller registers a
        custom feed before this ever runs, the table has 1 row but none of
        the defaults yet — a whole-table-count gate would permanently skip
        seeding the defaults from then on (Copilot P2 finding on PR #71).
        Checking per-URL existence makes this safe to call unconditionally
        and idempotently regardless of call order.

        ``default_feeds`` maps slug -> object with ``.name``/``.url``/``.source``
        attributes (``researcher.rss.FeedConfig``).
        """
        now = time.time()
        seeded = 0
        with closing(self._conn()) as conn, conn:
            for slug, cfg in default_feeds.items():
                existing = conn.execute(
                    "SELECT 1 FROM feeds WHERE url = ?", (cfg.url,)
                ).fetchone()
                if existing is not None:
                    continue
                feed_id = f"feed_{uuid.uuid4().hex[:12]}"
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO feeds
                        (feed_id, name, url, source, enabled, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (feed_id, cfg.name, cfg.url, cfg.source, now, now, json.dumps({"seed_slug": slug})),
                )
                # OR IGNORE can still no-op here (e.g. a same-transaction race
                # on the unique url index) despite the pre-check SELECT above
                # finding nothing — only count rows actually inserted.
                if cur.rowcount > 0:
                    seeded += 1
            return seeded


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["enabled"] = bool(d["enabled"])
    try:
        parsed = json.loads(d["metadata"])
    except (TypeError, ValueError):
        parsed = {}
    # Valid-but-non-object JSON ("[]", "null", "5") must not leak through —
    # callers do row["metadata"].get(...) and a non-dict would crash there.
    d["metadata"] = parsed if isinstance(parsed, dict) else {}
    return d
