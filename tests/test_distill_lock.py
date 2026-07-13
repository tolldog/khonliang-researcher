"""PID-ownership distill locks (bug abfe679b).

Two workers can't distill the same paper (atomic claim), and a paper whose
distiller died is reclaimed by the next live worker — detected by the owner PID
no longer running, no timers/heartbeat. The owner token is ``<start>_<pid>`` so a
recycled PID (new start time) isn't mistaken for the original.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from khonliang.knowledge.store import EntryStatus, KnowledgeEntry, Tier

from researcher.distill_lock import DistillLockStore, owner_token_for
from researcher.pipeline import create_pipeline
from researcher.server import create_research_server
from researcher.worker import DistillWorker

# A token whose PID is not running (well above any live pid) -> owner is "dead".
DEAD_OWNER = "20200101-000000_2147483646"


async def _fast_sleep(_seconds: float) -> None:
    """Drop-in for ``asyncio.sleep`` in tests exercising ``_safe_release_lock``'s
    retry backoff — skips the real delay so fault-injection tests stay fast."""
    return None


def _make_config(tmp_path):
    config = {"db_path": str(tmp_path / "researcher.db"), "models": {}, "projects": {}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _add(pipeline, entry_id, status):
    pipeline.knowledge.add(KnowledgeEntry(
        id=entry_id, tier=Tier.IMPORTED, title=entry_id, content="body",
        scope="research", tags=["paper"], status=status,
    ))


def _inject_lock(db_path, paper_id, owner):
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT OR REPLACE INTO distill_locks (paper_id, owner, claimed_at) "
                 "VALUES (?, ?, ?)", (paper_id, owner, 0.0))
    conn.commit()
    conn.close()


# ----------------------------------------------------------- lock store ----

def test_owner_token_is_date_pid():
    tok = owner_token_for(os.getpid())
    assert tok is not None
    assert tok.endswith(f"_{os.getpid()}")
    assert "-" in tok.rsplit("_", 1)[0]  # a date-time stamp precedes the pid


def test_owner_token_has_subsecond_precision_and_is_deterministic():
    tok = owner_token_for(os.getpid())
    assert "." in tok.rsplit("_", 1)[0]  # microseconds present (same-second reuse safe)
    assert owner_token_for(os.getpid()) == tok  # stable for the same process


def test_claim_release_roundtrip(tmp_path):
    store = DistillLockStore(str(tmp_path / "k.db"))
    assert store.claim("p1") is True
    assert store.is_locked_live("p1") is True
    store.release("p1")
    assert store.is_locked_live("p1") is False


def test_claim_fails_while_live_owner_holds(tmp_path):
    store = DistillLockStore(str(tmp_path / "k.db"))
    assert store.claim("p1") is True
    assert store.claim("p1") is False  # live owner still holds it


def test_dead_owner_lock_is_reclaimable_and_stealable(tmp_path):
    db = str(tmp_path / "k.db")
    store = DistillLockStore(db)
    _inject_lock(db, "p1", DEAD_OWNER)

    assert store.is_locked_live("p1") is False
    assert store.reclaim_dead() == ["p1"]
    assert store.claim("p1") is True  # now free to take


def test_reclaim_keeps_live_locks(tmp_path):
    store = DistillLockStore(str(tmp_path / "k.db"))
    store.claim("p1")  # our own live lock
    assert store.reclaim_dead() == []
    assert store.is_locked_live("p1") is True


def test_reused_pid_with_different_start_is_dead(tmp_path):
    # our real pid but a fabricated (wrong) start time -> token mismatch -> dead
    db = str(tmp_path / "k.db")
    store = DistillLockStore(db)
    _inject_lock(db, "p1", f"19990101-000000_{os.getpid()}")
    assert store.is_locked_live("p1") is False


# -------------------------------------------------------- pipeline recover ----

def test_recover_requeues_dead_owner_processing(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "orphan", EntryStatus.PROCESSING)
    _inject_lock(str(tmp_path / "researcher.db"), "orphan", DEAD_OWNER)

    assert pipeline.recover_stalled_processing() == 1
    assert pipeline.knowledge.get("orphan").status == EntryStatus.INGESTED


def test_recover_requeues_processing_with_no_lock(tmp_path):
    # defensive: PROCESSING with no lock row (crash between claim and status) is
    # an orphan too.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "nolock", EntryStatus.PROCESSING)
    assert pipeline.recover_stalled_processing() == 1
    assert pipeline.knowledge.get("nolock").status == EntryStatus.INGESTED


def test_recover_clears_partial_triples_before_requeue(tmp_path):
    # codex: a crash inside _store_distillation can leave partial paper:<id>
    # triples; recovery must clear them before requeue so the retry starts clean.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "orphan", EntryStatus.PROCESSING)
    pipeline.triples.add("A", "relates_to", "B", confidence=0.7, source="paper:orphan")

    assert pipeline.recover_stalled_processing() == 1
    assert pipeline.knowledge.get("orphan").status == EntryStatus.INGESTED
    leftover = [t for t in pipeline.triples.get(limit=100) if "paper:orphan" in t.sources]
    assert leftover == []  # partial triple cleared


def test_recover_clear_preserves_co_sourced_triples(tmp_path):
    # clearing orphan's provenance must not delete a fact another paper asserts.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "orphan", EntryStatus.PROCESSING)
    pipeline.triples.add("A", "relates_to", "B", confidence=0.7, source="paper:orphan")
    pipeline.triples.add("A", "relates_to", "B", confidence=0.9, source="paper:keep")

    pipeline.recover_stalled_processing()

    survivors = pipeline.triples.get(subject="A")
    assert len(survivors) == 1
    assert survivors[0].sources == ["paper:keep"]  # orphan token gone, other kept


def test_recover_skips_paper_finished_after_scan(tmp_path, monkeypatch):
    # codex P1: if the owner FINISHED (DISTILLED + released) between the
    # PROCESSING scan and our claim, the re-read under the claim must leave it
    # alone — not scrub the good result back to INGESTED.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.DISTILLED)  # already done, lock released
    pipeline.triples.add("A", "relates_to", "B", confidence=0.9, source="paper:p1")

    real = pipeline.knowledge.get_by_status
    def fake(status, tier=None):
        if status == EntryStatus.PROCESSING:
            return [pipeline.knowledge.get("p1")]  # simulate it was PROCESSING at scan
        return real(status, tier=tier)
    monkeypatch.setattr(pipeline.knowledge, "get_by_status", fake)

    assert pipeline.recover_stalled_processing() == 0
    assert pipeline.knowledge.get("p1").status == EntryStatus.DISTILLED  # untouched
    assert [t for t in pipeline.triples.get(limit=None) if "paper:p1" in t.sources]  # preserved


def test_recover_leaves_live_locked_processing(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "live", EntryStatus.PROCESSING)
    pipeline.locks.claim("live")  # a live owner holds it

    assert pipeline.recover_stalled_processing() == 0
    assert pipeline.knowledge.get("live").status == EntryStatus.PROCESSING


def test_get_next_skips_live_locked_papers(tmp_path):
    # codex: don't hand out a paper a sibling is actively distilling — pick a
    # different INGESTED one so a batch slot isn't spent on a contention skip.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "locked", EntryStatus.INGESTED)
    _add(pipeline, "free", EntryStatus.INGESTED)
    pipeline.locks.claim("locked")  # a live owner holds it
    worker = DistillWorker(pipeline)

    nxt = worker.get_next()

    assert nxt is not None and nxt.id == "free"  # the live-locked one is skipped


def test_worker_get_next_reclaims_dead_orphan(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "orphan", EntryStatus.PROCESSING)
    _inject_lock(str(tmp_path / "researcher.db"), "orphan", DEAD_OWNER)
    worker = DistillWorker(pipeline)

    nxt = worker.get_next()

    assert nxt is not None and nxt.id == "orphan"
    assert pipeline.knowledge.get("orphan").status == EntryStatus.INGESTED


# ------------------------------------------------------------ distill flow ----

async def _ok_summary(*a, **k):
    return {"success": True, "summary": {"title": "p1"}}
async def _ok_extract(*a, **k):
    return {"success": True, "triples": []}


@pytest.mark.asyncio
async def test_distill_releases_lock_on_success(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)

    result = await pipeline.distill("p1")

    assert result.success is True
    assert pipeline.knowledge.get("p1").status == EntryStatus.DISTILLED
    assert pipeline.locks.is_locked_live("p1") is False  # released


@pytest.mark.asyncio
async def test_distill_releases_lock_on_failure(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    async def boom(*a, **k):
        raise RuntimeError("summarizer exploded")
    pipeline.summarizer = SimpleNamespace(handle=boom)

    result = await pipeline.distill("p1")

    assert result.success is False
    assert pipeline.knowledge.get("p1").status == EntryStatus.FAILED
    assert pipeline.locks.is_locked_live("p1") is False  # released even on failure


@pytest.mark.asyncio
async def test_distill_failure_clears_partial_artifacts(tmp_path):
    # codex: a crash inside _store_distillation (partial writes) then FAILED must
    # not strand stale artifacts — recovery only touches PROCESSING rows.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)

    def store_boom(entry, result):
        pipeline.triples.add("A", "relates_to", "B", confidence=0.7,
                             source=f"paper:{entry.id}")  # partial write
        raise RuntimeError("store crashed mid-way")
    pipeline._store_distillation = store_boom

    result = await pipeline.distill("p1")

    assert result.success is False
    assert pipeline.knowledge.get("p1").status == EntryStatus.FAILED
    assert [t for t in pipeline.triples.get(limit=None) if "paper:p1" in t.sources] == []
    assert pipeline.locks.is_locked_live("p1") is False  # lock still released


@pytest.mark.asyncio
async def test_distill_skips_when_live_locked(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.locks.claim("p1")  # a live owner already holds it

    called = []
    async def spy(*a, **k):
        called.append(1)
        return {"success": True, "summary": {}}
    pipeline.summarizer = SimpleNamespace(handle=spy)

    result = await pipeline.distill("p1")

    assert result.skipped is True  # benign skip (success stays truthy)
    assert called == []  # never distilled — skipped
    assert pipeline.knowledge.get("p1").status == EntryStatus.INGESTED  # untouched


@pytest.mark.asyncio
async def test_distill_skip_sets_skipped_flag(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.locks.claim("p1")  # a live owner holds it
    result = await pipeline.distill("p1")
    # A skip is neither success nor failure: success stays False (not a distill),
    # skipped flags it for callers so they don't misreport it as an error.
    assert result.success is False and result.skipped is True


@pytest.mark.asyncio
async def test_worker_process_item_skip_returns_skip_sentinel(tmp_path):
    # A lock-contention skip returns the SKIP sentinel so BaseQueueWorker counts
    # it as `declined` — neither processed nor a retry failure (bug abfe679b).
    from khonliang_researcher.worker import SKIP
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.locks.claim("p1")  # live owner holds it

    async def not_irrelevant(entry_id):
        return False
    pipeline.filter_irrelevant = not_irrelevant
    worker = DistillWorker(pipeline)

    ok = await worker.process_item(pipeline.knowledge.get("p1"))

    assert ok is SKIP


@pytest.mark.asyncio
async def test_distill_success_survives_record_signal_failure(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)
    async def boom_signal(*a, **k):
        raise RuntimeError("telemetry down")
    pipeline.relevance = SimpleNamespace(record_signal=boom_signal)

    result = await pipeline.distill("p1")

    assert result.success is True
    assert pipeline.knowledge.get("p1").status == EntryStatus.DISTILLED


@pytest.mark.asyncio
async def test_distill_success_survives_digest_failure(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)
    def boom_record(*a, **k):
        raise RuntimeError("digest db down")
    pipeline.digest = SimpleNamespace(record=boom_record)

    result = await pipeline.distill("p1")

    assert result.success is True
    assert pipeline.knowledge.get("p1").status == EntryStatus.DISTILLED


# ---------------------------------------------- pre-LLM DB window (706df96b) ----
#
# bug_khonliang-researcher_706df96b: distill_paper crashed instantly with
# "OperationalError: unable to open database file" — an unhandled DB error in
# the pre-LLM window (entry lookup / lock claim / PROCESSING flip) propagated
# uncaught, crashing the ingest job at phase=error before any LLM call. These
# guard each step so a transient DB-open failure there becomes a graceful,
# retryable ``result.errored=True`` instead of an uncaught exception, and never
# burns the entry to FAILED (which would fall out of the worker's INGESTED scan
# and stop being retried).

@pytest.mark.asyncio
async def test_distill_entry_lookup_failure_is_graceful_not_raised(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_get(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.knowledge.get = boom_get

    result = await pipeline.distill("p1")  # must not raise

    assert result.errored is True
    assert result.success is False
    assert result.skipped is False
    # codex P3 round 11: the real entry_id must be identifiable from the
    # result, not a generic "ERROR" placeholder — batch callers render
    # `title` directly and need to know WHICH paper needs attention.
    assert result.entry_id == "p1"
    assert result.title != "ERROR"
    assert "p1" in result.title


@pytest.mark.asyncio
async def test_distill_lock_claim_failure_is_graceful_not_raised(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_claim(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.claim = boom_claim

    result = await pipeline.distill("p1")  # must not raise

    assert result.errored is True
    assert result.success is False
    assert result.skipped is False
    # Entry status untouched — still retryable by the next drain cycle.
    assert pipeline.knowledge.get("p1").status == EntryStatus.INGESTED


@pytest.mark.asyncio
async def test_distill_processing_flip_failure_releases_lock_and_is_graceful(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    real_set_status = pipeline.knowledge.set_status
    def flaky_set_status(entry_id, status):
        if status == EntryStatus.PROCESSING:
            raise sqlite3.OperationalError("unable to open database file")
        return real_set_status(entry_id, status)
    pipeline.knowledge.set_status = flaky_set_status

    result = await pipeline.distill("p1")  # must not raise

    assert result.errored is True
    assert result.success is False
    assert result.skipped is False
    assert result.stuck is False  # release succeeded — not the compounded-failure case
    # Lock claimed then released — not leaked/stuck to this still-live process.
    assert pipeline.locks.is_locked_live("p1") is False
    # Status was never durably flipped away from INGESTED — still retryable.
    assert pipeline.knowledge.get("p1").status == EntryStatus.INGESTED


@pytest.mark.asyncio
async def test_distill_processing_flip_and_release_both_fail_reports_stuck(
    tmp_path, monkeypatch
):
    # codex P2 round 5: the compounded-failure case (PROCESSING flip AND the
    # subsequent release both hit a transient DB outage) must report
    # stuck=True, not just errored=True — the lock is genuinely leaked to
    # this still-live process, not "will retry soon" like a plain errored.
    monkeypatch.setattr("researcher.pipeline.asyncio.sleep", _fast_sleep)
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def flaky_set_status(entry_id, status):
        if status == EntryStatus.PROCESSING:
            raise sqlite3.OperationalError("unable to open database file")
        raise AssertionError("should not reach a non-PROCESSING set_status")
    pipeline.knowledge.set_status = flaky_set_status

    def boom_release(entry_id):
        raise sqlite3.OperationalError("database is locked")
    pipeline.locks.release = boom_release

    result = await pipeline.distill("p1")  # must not raise

    assert result.errored is True
    assert result.stuck is True
    assert pipeline.locks.is_locked_live("p1") is True  # genuinely still held


@pytest.mark.asyncio
async def test_safe_release_lock_cancellation_during_backoff_still_releases(tmp_path):
    # codex P2 round 5: a cancellation arriving mid-backoff (e.g. process
    # shutdown) must not skip an already-due release retry — make one
    # best-effort synchronous attempt before letting the cancellation win.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.locks.claim("p1")

    real_release = pipeline.locks.release
    calls = {"n": 0}
    def flaky_release(entry_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return real_release(entry_id)
    pipeline.locks.release = flaky_release

    async def cancel_immediately(_seconds):
        raise asyncio.CancelledError()
    import researcher.pipeline as pipeline_module
    orig_sleep = pipeline_module.asyncio.sleep
    pipeline_module.asyncio.sleep = cancel_immediately
    try:
        with pytest.raises(asyncio.CancelledError):
            await pipeline._safe_release_lock("p1")
    finally:
        pipeline_module.asyncio.sleep = orig_sleep

    assert calls["n"] == 2  # the best-effort retry ran despite the cancellation
    assert pipeline.locks.is_locked_live("p1") is False  # actually released


@pytest.mark.asyncio
async def test_worker_process_item_logs_stuck_distinctly(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("researcher.pipeline.asyncio.sleep", _fast_sleep)
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_claim(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.claim = boom_claim

    async def not_irrelevant(entry_id):
        return False
    pipeline.filter_irrelevant = not_irrelevant
    worker = DistillWorker(pipeline)

    from khonliang_researcher.worker import SKIP
    # claim() itself fails (not the PROCESSING-flip path), so no lock is ever
    # claimed and result.stuck stays False — assert the plain (non-stuck)
    # log path fires, distinctly from the stuck-lock error log.
    with caplog.at_level(logging.INFO):
        ok = await worker.process_item(pipeline.knowledge.get("p1"))

    assert ok is SKIP
    assert any("transient DB error" in rec.message for rec in caplog.records)
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records)


@pytest.mark.asyncio
async def test_worker_process_item_logs_stuck_even_on_success(tmp_path, monkeypatch, caplog):
    # codex P2 round 9: `stuck` is orthogonal to `success` — a leaked lock on
    # the FINAL release can happen even after an otherwise-successful
    # distill, and the worker's plain "OK" log must not silently hide it.
    monkeypatch.setattr("researcher.pipeline.asyncio.sleep", _fast_sleep)
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)

    def boom_release(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.release = boom_release

    async def not_irrelevant(entry_id):
        return False
    pipeline.filter_irrelevant = not_irrelevant
    worker = DistillWorker(pipeline)

    with caplog.at_level(logging.INFO):
        ok = await worker.process_item(pipeline.knowledge.get("p1"))

    assert ok is True  # distillation itself succeeded
    assert any(
        rec.levelno >= logging.ERROR and "STUCK" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_distill_lock_release_failure_does_not_mask_success_result(
    tmp_path, monkeypatch, caplog
):
    # _safe_release_lock retries then swallows a persistently-failing
    # release() so it can't shadow the (already-computed) success result by
    # raising in the finally — but it must log a loud, distinguishable
    # STUCK LOCK marker (codex P1, 706df96b) since the lock is now leaked.
    monkeypatch.setattr("researcher.pipeline.asyncio.sleep", _fast_sleep)
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)

    def boom_release(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.release = boom_release

    with caplog.at_level(logging.ERROR):
        result = await pipeline.distill("p1")  # must not raise

    assert result.success is True
    assert result.stuck is True  # success can co-occur with a leaked lock
    assert pipeline.knowledge.get("p1").status == EntryStatus.DISTILLED
    assert any("STUCK LOCK" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_safe_release_lock_recovers_after_transient_failure(tmp_path, monkeypatch, caplog):
    # A release() that fails once then succeeds must NOT be reported as a
    # stuck lock — retry actually clears it.
    monkeypatch.setattr("researcher.pipeline.asyncio.sleep", _fast_sleep)
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.locks.claim("p1")

    real_release = pipeline.locks.release
    calls = {"n": 0}
    def flaky_release(entry_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return real_release(entry_id)
    pipeline.locks.release = flaky_release

    with caplog.at_level(logging.ERROR):
        await pipeline._safe_release_lock("p1")

    assert calls["n"] == 2
    assert pipeline.locks.is_locked_live("p1") is False  # actually released
    assert not any("STUCK LOCK" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_distill_entry_lookup_non_sqlite_error_propagates(tmp_path):
    # codex P2: only sqlite3.OperationalError (a transient DB-open failure)
    # is swallowed into errored=True. A real programming bug must still
    # surface loudly rather than retrying forever unnoticed.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_get(entry_id):
        raise RuntimeError("not a transient DB error")
    pipeline.knowledge.get = boom_get

    with pytest.raises(RuntimeError):
        await pipeline.distill("p1")


@pytest.mark.asyncio
async def test_distill_lock_claim_non_sqlite_error_propagates(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_claim(entry_id):
        raise RuntimeError("not a transient DB error")
    pipeline.locks.claim = boom_claim

    with pytest.raises(RuntimeError):
        await pipeline.distill("p1")


@pytest.mark.asyncio
async def test_distill_processing_flip_non_sqlite_error_propagates(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    real_set_status = pipeline.knowledge.set_status
    def flaky_set_status(entry_id, status):
        if status == EntryStatus.PROCESSING:
            raise RuntimeError("not a transient DB error")
        return real_set_status(entry_id, status)
    pipeline.knowledge.set_status = flaky_set_status

    with pytest.raises(RuntimeError):
        await pipeline.distill("p1")
    # A real bug must not silently strand the lock either — it's a crash,
    # not a "leave it retryable" outcome, so no cleanup guarantee is claimed
    # here beyond "don't misreport it as errored/retryable".


# ---------------------------------- non-transient OperationalError (codex P1 round 2) ----
#
# sqlite3.OperationalError is raised for BOTH "can't open/lock the DB right
# now" (transient) AND "no such table" / malformed SQL (a persistent bug) —
# matching on exception type alone still swallows a real bug into the
# retryable errored path forever. These confirm a non-transient
# OperationalError (e.g. a schema problem) propagates uncaught, same as any
# other genuine programming error, rather than being treated as retryable.

def test_is_transient_sqlite_error_classifies_known_messages():
    from researcher.pipeline import _is_transient_sqlite_error
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("unable to open database file")
    ) is True
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("database is locked")
    ) is True
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("database is busy")
    ) is True
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("database table is locked")
    ) is True
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("database schema is locked: knowledge")
    ) is True
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("no such table: knowledge")
    ) is False
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("near \"SELCT\": syntax error")
    ) is False


def test_is_transient_sqlite_error_excludes_ambiguous_environment_messages():
    # codex P1 round 3: "disk I/O error" and "readonly database" can just as
    # easily be a permanent regression (failing disk, misconfigured
    # permissions/mount) as a transient blip — deliberately NOT whitelisted,
    # so they raise loudly rather than risk a silent forever-retry stall.
    from researcher.pipeline import _is_transient_sqlite_error
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("disk I/O error")
    ) is False
    assert _is_transient_sqlite_error(
        sqlite3.OperationalError("attempt to write a readonly database")
    ) is False


# --------------------------------- create_pipeline() boot-time db_path guard (round 7) ----
#
# codex round 6 P1: "unable to open database file" is whitelisted as
# transient in _is_transient_sqlite_error, but that same message ALSO covers
# a genuine misconfiguration (bad db_path, missing directory, broken
# permissions) — a message-text heuristic alone can't tell those apart.
# Resolved via a boot-time guard instead of further message-classification:
# create_pipeline() now validates db_path is actually openable right after
# resolving it to absolute, and raises loudly if not, so a bad config fails
# at process startup rather than ever reaching distill()'s pre-LLM guards
# disguised as a transient outage.

def test_create_pipeline_raises_loudly_on_unopenable_db_path(tmp_path):
    # A directory with no write permission: mkdir(parents=True,
    # exist_ok=True) is a no-op (it already exists as a directory, so
    # exist_ok short-circuits before any permission check), but sqlite3
    # then can't create the actual db file inside it — this is exactly the
    # class of misconfiguration (broken filesystem permissions) the boot
    # guard exists to catch instead of deferring to a later, ambiguous
    # mid-distill failure.
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o555)  # read + execute, no write
    bad_db_path = str(readonly_dir / "researcher.db")

    config = {"db_path": bad_db_path, "models": {}, "projects": {}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    try:
        with pytest.raises(RuntimeError, match="not usable at startup"):
            create_pipeline(str(config_path))
    finally:
        readonly_dir.chmod(0o755)  # restore so tmp_path cleanup can remove it


def test_create_pipeline_succeeds_with_a_valid_db_path(tmp_path):
    # Sanity check the guard doesn't false-positive on an ordinary, valid
    # (fresh, not-yet-existing) db_path — the common case must keep working.
    config = {
        "db_path": str(tmp_path / "fresh" / "researcher.db"),
        "models": {}, "projects": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    pipeline = create_pipeline(str(config_path))  # must not raise

    assert Path(pipeline.config["db_path"]).exists()


def test_create_pipeline_boot_probe_catches_readonly_existing_db_file(tmp_path):
    # codex P2 round 8: a read-only-mounted EXISTING db file passes a
    # read-only "SELECT 1" probe fine (the directory is writable, the file
    # opens and reads) — only a probe that exercises an actual WRITE catches
    # this before create_pipeline() reports success and the failure surfaces
    # later, mid-distill, as "attempt to write a readonly database".
    db_file = tmp_path / "researcher.db"
    sqlite3.connect(str(db_file)).close()  # create it first, writable
    db_file.chmod(0o444)  # then make the FILE itself read-only

    config = {"db_path": str(db_file), "models": {}, "projects": {}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    try:
        with pytest.raises(RuntimeError, match="not usable at startup"):
            create_pipeline(str(config_path))
    finally:
        db_file.chmod(0o644)  # restore so tmp_path cleanup can remove it


def test_create_pipeline_boot_probe_leaves_no_schema_residue(tmp_path):
    # The write probe round-trips PRAGMA user_version rather than creating a
    # throwaway table — confirm it leaves user_version untouched (0, the
    # sqlite default for a fresh file) and no stray tables behind.
    db_path = tmp_path / "researcher.db"
    config = {"db_path": str(db_path), "models": {}, "projects": {}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    create_pipeline(str(config_path))

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert not any(t.startswith("_boot") for t in tables)
    assert user_version == 0


def test_create_pipeline_boot_probe_tolerates_brief_sibling_write_contention(tmp_path):
    # codex P2 round 9: this architecture starts multiple independent
    # processes against the SAME db_path (bus agent, standalone worker,
    # MCP-stdio server) — a sibling briefly holding a write transaction at
    # the exact moment another process boots is ordinary contention, not a
    # misconfiguration, and must NOT make create_pipeline() raise. A short
    # busy_timeout would false-positive here; the widened one must not.
    db_path = tmp_path / "researcher.db"
    sqlite3.connect(str(db_path)).close()  # create the file first

    blocker = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")  # hold the write lock

    def _release_after_a_moment():
        time.sleep(0.3)
        blocker.commit()
        blocker.close()
    releaser = threading.Thread(target=_release_after_a_moment)
    releaser.start()
    try:
        config = {"db_path": str(db_path), "models": {}, "projects": {}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))

        create_pipeline(str(config_path))  # must not raise — waits it out
    finally:
        releaser.join()


@pytest.mark.asyncio
async def test_distill_entry_lookup_non_transient_operational_error_propagates(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_get(entry_id):
        raise sqlite3.OperationalError("no such table: knowledge")
    pipeline.knowledge.get = boom_get

    with pytest.raises(sqlite3.OperationalError):
        await pipeline.distill("p1")


@pytest.mark.asyncio
async def test_distill_lock_claim_non_transient_operational_error_propagates(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_claim(entry_id):
        raise sqlite3.OperationalError("no such table: distill_locks")
    pipeline.locks.claim = boom_claim

    with pytest.raises(sqlite3.OperationalError):
        await pipeline.distill("p1")


@pytest.mark.asyncio
async def test_distill_processing_flip_non_transient_operational_error_propagates_and_releases(
    tmp_path,
):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    real_set_status = pipeline.knowledge.set_status
    def flaky_set_status(entry_id, status):
        if status == EntryStatus.PROCESSING:
            raise sqlite3.OperationalError("no such column: bogus")
        return real_set_status(entry_id, status)
    pipeline.knowledge.set_status = flaky_set_status

    with pytest.raises(sqlite3.OperationalError):
        await pipeline.distill("p1")
    # Even on the non-transient (crash) path, the successfully-claimed lock
    # is released rather than leaked.
    assert pipeline.locks.is_locked_live("p1") is False


@pytest.mark.asyncio
async def test_safe_release_lock_non_transient_error_raises_immediately_no_retry(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.locks.claim("p1")

    calls = {"n": 0}
    def boom_release(entry_id):
        calls["n"] += 1
        raise sqlite3.OperationalError("no such table: distill_locks")
    pipeline.locks.release = boom_release

    with pytest.raises(sqlite3.OperationalError):
        await pipeline._safe_release_lock("p1")
    assert calls["n"] == 1  # no retry for a non-transient error


@pytest.mark.asyncio
async def test_worker_process_item_treats_errored_as_skip_not_failure(tmp_path):
    # An errored (transient DB) outcome must not burn the worker's retry
    # budget the way a genuine content failure would — treat it like a skip.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_claim(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.claim = boom_claim

    async def not_irrelevant(entry_id):
        return False
    pipeline.filter_irrelevant = not_irrelevant
    worker = DistillWorker(pipeline)

    from khonliang_researcher.worker import SKIP
    ok = await worker.process_item(pipeline.knowledge.get("p1"))

    assert ok is SKIP


@pytest.mark.asyncio
async def test_distill_pending_mcp_tool_reports_errored_distinctly_from_failed(tmp_path):
    # codex P2 round 6: distill_pending() (the MCP batch tool) only
    # special-cased `skipped`, so a transient DB outage was misreported as
    # a terminal "FAILED" in batch mode even though distill_paper's
    # single-entry tool already distinguished it via `errored`.
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)

    def boom_claim(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.claim = boom_claim

    mcp = create_research_server(pipeline)
    result = await mcp.call_tool("distill_pending", {})
    text = result[-1]["result"]

    assert "[error]" in text
    assert "FAILED" not in text


@pytest.mark.asyncio
async def test_distill_paper_mcp_tool_surfaces_stuck_even_on_success(tmp_path, monkeypatch):
    # codex P2 round 8: `stuck` is orthogonal to `success` — a leaked lock on
    # the FINAL release can happen even after an otherwise-successful
    # distill, and distill_paper() must not silently report a clean success
    # in that case.
    monkeypatch.setattr("researcher.pipeline.asyncio.sleep", _fast_sleep)
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)

    def boom_release(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.release = boom_release

    mcp = create_research_server(pipeline)
    result = await mcp.call_tool("distill_paper", {"entry_id": "p1"})
    text = result[-1]["result"]

    assert "# p1" in text  # the success (markdown title) rendering still happened
    assert "stuck" in text.lower()  # but the leaked lock is surfaced too


@pytest.mark.asyncio
async def test_distill_pending_mcp_tool_marks_stuck_on_otherwise_ok_entry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("researcher.pipeline.asyncio.sleep", _fast_sleep)
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "p1", EntryStatus.INGESTED)
    pipeline.summarizer = SimpleNamespace(handle=_ok_summary)
    pipeline.extractor = SimpleNamespace(handle=_ok_extract)

    def boom_release(entry_id):
        raise sqlite3.OperationalError("unable to open database file")
    pipeline.locks.release = boom_release

    mcp = create_research_server(pipeline)
    result = await mcp.call_tool("distill_pending", {})
    text = result[-1]["result"]

    assert "[ok+stuck]" in text
