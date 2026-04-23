---
kind: code_review
severity: concern
---

# Capture snapshot_id at first read; reuse — don't re-lookup at persist time

**Invariant**: when a handler reads a snapshot for computation AND persists a reference to that same snapshot, capture the `snapshot_id` from the first read and thread it through. Do not call `latest_snapshot()` again at persist time — a rebuild landing between reads causes the persisted reference to not match the taxonomy actually used.

**Bad pattern**:
```python
def handle(self, event):
    snap = self.taxonomy.latest_snapshot()       # read 1
    result = compute(snap.content, event)
    # ... rebuild may land here ...
    self._save(result, snapshot_id=self.taxonomy.latest_snapshot().id)  # read 2
```

**Good pattern**:
```python
def _ensure_snapshot(self):
    snap = self.taxonomy.latest_snapshot()
    return snap.content, snap.id

def handle(self, event):
    content, snapshot_id = self._ensure_snapshot()
    result = compute(content, event)
    self._save(result, snapshot_id=snapshot_id)
```

**Rationale**: "the snapshot I computed against" and "the snapshot id I stored" must be the same value by construction, not by luck. Sourced from PR #29.
