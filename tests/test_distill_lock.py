"""PID-ownership distill locks (bug abfe679b).

Two workers can't distill the same paper (atomic claim), and a paper whose
distiller died is reclaimed by the next live worker — detected by the owner PID
no longer running, no timers/heartbeat. The owner token is ``<start>_<pid>`` so a
recycled PID (new start time) isn't mistaken for the original.
"""

from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import pytest
import yaml

from khonliang.knowledge.store import EntryStatus, KnowledgeEntry, Tier

from researcher.distill_lock import DistillLockStore, owner_token_for
from researcher.pipeline import create_pipeline
from researcher.worker import DistillWorker

# A token whose PID is not running (well above any live pid) -> owner is "dead".
DEAD_OWNER = "20200101-000000_2147483646"


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


def test_recover_leaves_live_locked_processing(tmp_path):
    pipeline = create_pipeline(_make_config(tmp_path))
    _add(pipeline, "live", EntryStatus.PROCESSING)
    pipeline.locks.claim("live")  # a live owner holds it

    assert pipeline.recover_stalled_processing() == 0
    assert pipeline.knowledge.get("live").status == EntryStatus.PROCESSING


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

    assert result.success is False
    assert called == []  # never distilled — skipped
    assert pipeline.knowledge.get("p1").status == EntryStatus.INGESTED  # untouched


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
