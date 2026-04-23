---
kind: code_review
severity: concern
---

# `shutdown()` cancels live tasks but must NOT delete persisted rows

**Invariant**: process-lifecycle `shutdown()` must cancel live tasks but PRESERVE persisted state. The per-user `stop()` / `unregister()` method is the only code path authorized to delete persisted rows. Graceful shutdown must not destroy data needed for rehydrate on next start.

**Bad pattern**:
```python
async def shutdown(self):
    for user_id in list(self._active):
        await self.stop(user_id)  # stop() also deletes persisted row
    # next process start: rehydrate finds nothing
```

**Good pattern**:
```python
async def shutdown(self):
    # cancel tasks only; persistence must survive for rehydrate
    for user_id, task in list(self._tasks.items()):
        task.cancel()
    await asyncio.gather(*self._tasks.values(), return_exceptions=True)

async def stop(self, user_id):
    # explicit user-initiated: cancel AND delete persistence
    task = self._tasks.pop(user_id, None)
    if task: task.cancel()
    await self._store.delete(user_id)
```

**Rationale**: conflating process-shutdown with user-stop destroys data on every restart. The two verbs have different semantics. Sourced from PR #29.
