# khonliang-researcher — concern-level invariants

Repo-specific invariants distilled from real cross-vendor review history
(librarian agent PR #29; store-helper PR #16). A local hot-tier model tends to
catch type/docstring/dead-code issues but misses these cross-cutting
correctness invariants. Flag a diff at **concern** severity when it violates
one of the patterns below. Each entry is a *canonical* bad/good pair — match by
shape, not by exact identifier.

## paper_vs_idea_filter

Iterating `Tier.IMPORTED` entries for paper-domain processing must filter on
**both** the `paper` tag AND a non-empty `url` in metadata — ideas
(`tag="idea"`) have no URL and silently contaminate paper-scoped queries. Use
the shared `researcher.pipeline.is_paper_entry` helper (it guards
`entry.metadata or {}` and strips the URL before the truthiness check); don't
reimplement the predicate per call site.

- Bad: `for e in knowledge.get_by_tier(Tier.IMPORTED): process_as_paper(e)`  *(ingests ideas)*
- Good: `if not is_paper_entry(e): continue` then `process_as_paper(e)`. (PR #29)

## enum_to_value_coercion

The string form of an enum member is shape- and version-dependent
(`str(member)` may return `"ClassName.MEMBER"`). Coerce to a lookup/storage key
via `.value` — or a helper that falls back to `str()` for plain-string inputs —
never `str(member)`.

- Bad: `key = str(EntryStatus.IMPORTED)`  *(may be "EntryStatus.IMPORTED")*
- Good: `key = _status_value(EntryStatus.IMPORTED)` → `getattr(status, "value", None)` else `str(status)` (verbatim `researcher.ingest_watcher._status_value`; survives a future promotion from `class EntryStatus(str)` to a real Enum). (PR #29)

## shutdown_preserves_persistence

Process-lifecycle `shutdown()` must cancel live tasks but **preserve** persisted
state. Only the per-user `stop()` / `unregister()` path may delete persisted
rows; graceful shutdown must not destroy data needed to rehydrate on next start.

- Bad: `async def shutdown(self): for uid in self._active: await self.stop(uid)`  *(stop() also deletes the row)*
- Good: `shutdown()` does `task.cancel()` + `asyncio.gather(..., return_exceptions=True)` only; `stop(uid)` cancels AND `await self._store.delete(uid)`. (PR #29)

## nanosecond_resolution_ids

IDs / snapshot ids / dedupe keys derived from `int(time.time())` collide on
same-second events (batch processing and tests routinely issue many per
second). Use `time.time_ns()`.

- Bad: `f"{prefix}-{int(time.time())}"`
- Good: `f"{prefix}-{time.time_ns()}"`. (PR #29 R3/R8)

## non_dict_bus_payload_guard

Bus event handlers must guard `isinstance(payload, dict)` before `.get()`. A
malformed event with a non-dict payload (str/list/None) crashes `.get()`, and
without a guard the handler enters an error loop that spams the bus — the
transport enforces no payload shape, so every subscriber is the last defense.

- Bad: `payload = event.get("payload", {}); user_id = payload.get("user_id")`  *(AttributeError if payload is a string)*
- Good: `raw = event.get("payload"); payload = raw if isinstance(raw, dict) else {}` (warn on non-dict; mirrors `researcher/librarian_agent.py::_handle_bus_event`; use `%`-style logging — keyword `log.warning("m", k=v)` raises here). (PR #29)

## snapshot_id_race_prevention

When a handler reads a snapshot for computation AND persists a reference to it,
capture the `snapshot_id` at the **first** read and thread it through. Calling
`latest_snapshot()` again at persist time lets a rebuild between reads make the
stored reference disagree with the taxonomy actually used. (The attribute is
`.snapshot_id`, not `.id`.)

- Bad: `self._save(result, snapshot_id=self.taxonomy.latest_snapshot().snapshot_id)`  *(second lookup)*
- Good: `content, snapshot_id = self._ensure_snapshot()` then `compute(content, …)` then `self._save(result, snapshot_id=snapshot_id)` (canonical `researcher/librarian_agent.py::_ensure_snapshot`). (PR #29)

## batch_per_item_error_isolation

Batch operations (ingest N papers, distill N pending, …) must `try/except` each
item, append failures to a `failed` list, and continue — a single transient
error must not abort the batch and lose in-flight progress. In async contexts
re-raise `asyncio.CancelledError` first as a defensive default.

- Bad: `for url in urls: paper = await fetch(url); results.append(self._store(paper))`  *(one flaky fetch aborts all)*
- Good: per-item `try: ... except asyncio.CancelledError: raise except Exception as exc: failed.append({"item": url, "error": str(exc)})`, returning `{ingested, failed, status}`. (PR #29 R6 / #39 R4 / #42 R2)
