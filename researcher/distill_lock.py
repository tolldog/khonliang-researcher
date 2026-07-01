"""Crash-safe distill locks (bug abfe679b).

Stops two drainers from distilling the same paper AND reclaims papers whose
distiller died — without timers or heartbeats.

A lock is one row keyed by ``paper_id`` carrying an OWNER token of the form
``<YYYYMMDD-HHMMSS>_<pid>`` — the owning process's start time plus PID. That pair
uniquely fingerprints a process instance on this host (a recycled PID gets a new
start time), and the date half doubles as a human-readable "locked since". SQLite
is local, so every drainer can check another's PID directly.

The row is a MARKER, not a held lock: if a process vanishes it just sits there
harmlessly until any live worker notices the owner is gone and clears it — so a
lost process never strands a paper.

- ``claim``: atomically take the lock; fails only if a LIVE owner holds it (a
  dead owner's lock is stolen). SQLite ``BEGIN IMMEDIATE`` serialises claimers.
- ``release``: drop our lock.
- ``reclaim_dead``: drop every lock whose owner process is gone; safe to call
  anytime because it never touches a live owner's lock.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

_HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _boot_time() -> Optional[int]:
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime "):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def _proc_start_ticks(pid: int) -> Optional[int]:
    """``starttime`` (jiffies since boot) from /proc/<pid>/stat, or None if gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except (OSError, ValueError):
        return None
    # comm (field 2) is parenthesised and may contain spaces / parens — split
    # after the LAST ')'. The remainder starts at field 3 (state); starttime is
    # field 22 overall, i.e. index 19 in the remainder.
    rparen = data.rfind(")")
    if rparen == -1:
        return None
    rest = data[rparen + 2:].split()
    try:
        return int(rest[19])
    except (IndexError, ValueError):
        return None


def _start_epoch(pid: int) -> Optional[float]:
    """Process start time as an epoch, or None if the pid isn't resolvable.

    Prefers /proc (Linux, no deps); falls back to psutil when installed so
    non-/proc hosts (e.g. macOS dev machines) still get a reuse-safe start time.
    """
    ticks = _proc_start_ticks(pid)
    btime = _boot_time()
    if ticks is not None and btime is not None:
        return btime + ticks / _HZ
    try:  # optional — only used where /proc is unavailable
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def owner_token_for(pid: int) -> Optional[str]:
    """``<YYYYMMDD-HHMMSS>_<pid>`` for a live pid, or None if it isn't running."""
    start = _start_epoch(pid)
    if start is None:
        return None
    # Sub-second (%f) precision matters: a PID reused within the same second must
    # NOT produce the same token as the dead owner, or its stale lock looks live.
    stamp = datetime.fromtimestamp(start).strftime("%Y%m%d-%H%M%S.%f")
    return f"{stamp}_{pid}"


class DistillLockStore:
    """Per-paper distill ownership locks in the shared knowledge SQLite file."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Computed once — stable for this process's lifetime.
        self._owner = owner_token_for(os.getpid()) or f"nostart_{os.getpid()}"
        self._ensure_schema()

    @property
    def owner(self) -> str:
        return self._owner

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _ensure_schema(self) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS distill_locks ("
                "  paper_id   TEXT PRIMARY KEY,"
                "  owner      TEXT NOT NULL,"
                "  claimed_at REAL NOT NULL"
                ")"
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _owner_alive(owner: str) -> bool:
        """True unless the owning process is DEFINITIVELY gone or PID-reused.

        Errs toward "alive" whenever liveness can't be confirmed, so live work is
        never stolen; only a ProcessLookupError (pid gone) or a start-time
        mismatch (pid recycled by a newer process) counts as dead.
        """
        try:
            pid = int(owner.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return False  # malformed owner -> reclaimable
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False  # definitely gone
        except PermissionError:
            return True   # exists, owned by another user
        except OSError:
            return True   # can't tell -> do not steal
        current = owner_token_for(pid)
        if current is None:
            return True   # pid exists but start time unreadable -> assume alive
        return current == owner  # mismatch => pid reused, original process gone

    def claim(self, paper_id: str) -> bool:
        """Atomically take the lock. False if a LIVE owner already holds it."""
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")  # serialise claimers across processes
            row = conn.execute(
                "SELECT owner FROM distill_locks WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            if row is not None and self._owner_alive(row[0]):
                conn.rollback()
                return False
            conn.execute(
                "INSERT OR REPLACE INTO distill_locks (paper_id, owner, claimed_at) "
                "VALUES (?, ?, ?)",
                (paper_id, self._owner, time.time()),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def release(self, paper_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "DELETE FROM distill_locks WHERE paper_id = ? AND owner = ?",
                (paper_id, self._owner),
            )
            conn.commit()
        finally:
            conn.close()

    def is_locked_live(self, paper_id: str) -> bool:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT owner FROM distill_locks WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None and self._owner_alive(row[0])

    def reclaim_dead(self) -> List[str]:
        """Delete every lock whose owner process is gone; return freed paper_ids.

        Runs the snapshot + delete in one ``BEGIN IMMEDIATE`` transaction and
        predicates each DELETE on the ORIGINAL owner, so a lock that a live
        worker re-claims between snapshot and delete is not clobbered (its owner
        no longer matches the dead token we saw).
        """
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")  # block concurrent claimers
            rows = conn.execute("SELECT paper_id, owner FROM distill_locks").fetchall()
            dead = [(pid, owner) for pid, owner in rows if not self._owner_alive(owner)]
            for pid, owner in dead:
                conn.execute(
                    "DELETE FROM distill_locks WHERE paper_id = ? AND owner = ?",
                    (pid, owner),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return [pid for pid, _ in dead]
