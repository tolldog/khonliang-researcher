# khonliang-researcher — concern-level invariants

Repo-specific invariants distilled from real cross-vendor review findings
(see PR #29, librarian agent). A local hot-tier model tends to catch
type/docstring/dead-code issues but miss these cross-cutting correctness
invariants. Flag a diff at **concern** severity when it violates one of the
patterns below. Each entry is a *canonical* bad/good pair — match by shape,
not by exact identifier.

## paper_vs_idea_filter

When iterating `Tier.IMPORTED` entries for paper-domain processing, filter to
entries that have **both** the `paper` tag **and** a non-empty `url` in
metadata. Ideas (`tag="idea"`) otherwise contaminate paper-scoped queries.

- Bad: `for e in tier_imported: process_paper(e)`
- Good: `for e in tier_imported:` then `if "paper" in e.tags and e.metadata.get("url"):`

## enum_to_value_coercion

Converting a `str`-based `Enum` (e.g. `EntryStatus`) to a lookup key: use the
`.value` attribute, **not** `str(member)` — `str()` can return
`"ClassName.MEMBER"` on some Python shapes, silently breaking the lookup.

- Bad: `key = str(status)`  *(may be "EntryStatus.IMPORTED")*
- Good: `key = status.value`

## shutdown_preserves_persistence

A process-lifecycle `shutdown()` must cancel live tasks but **preserve**
persisted state. Only the per-user `stop()` / `unregister()` path should
delete persisted rows; graceful shutdown must not destroy data needed for
rehydrate.

- Bad: `def shutdown(self): self.cancel_tasks(); self.store.delete_all()`
- Good: `shutdown()` cancels tasks only; `stop(user)` is the sole deleter.

## nanosecond_resolution_ids

IDs derived from `int(time.time())` collide on same-second events. Use
`time.time_ns()` for any dedupe key, snapshot id, or event key that can fire
more than once per second.

- Bad: `event_id = f"ev-{int(time.time())}"`
- Good: `event_id = f"ev-{time.time_ns()}"`

## per_item_error_isolation

Batch operations (ingest N papers, distill N pending, …) must isolate
per-item errors: `try/except` per item, append to a `failed` list, continue.
Aborting the whole batch on one item's failure loses all in-flight progress.

- Bad: `for p in papers: ingest(p)`  *(one raise aborts the batch)*
- Good: `for p in papers: try: ingest(p) except Exception: failed.append(p)`
