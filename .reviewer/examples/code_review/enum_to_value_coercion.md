---
kind: code_review
severity: concern
---

# Use `.value`, never `str(member)`, to coerce an Enum to a lookup key

**Invariant**: converting an Enum member to a lookup/storage key: use `.value`, never `str(member)`. The shapes differ:

- `enum.StrEnum` (Python 3.11+) or `class X(str, Enum)`: `str(member)` returns the raw value on 3.11+ — but historically returned `"ClassName.MEMBER"`, and mixing Python versions / enum shapes produces silent mismatches.
- `enum.Enum` (no `str` mixin): `str(member)` returns `"ClassName.MEMBER"` — never the value.

Always prefer explicit `.value`. When input may already be a plain `str`, use a helper that falls back cleanly.

**Bad pattern**:
```python
class EntryStatus(StrEnum):
    IMPORTED = "imported"

key = str(EntryStatus.IMPORTED)  # version / shape dependent
registry[key] = entry            # downstream lookup misses with "imported"
```

**Good pattern** (mirrors `researcher.ingest_watcher._status_value`):
```python
def _status_value(status: Any) -> str:
    value = getattr(status, "value", None)
    if value is not None:
        return str(value)
    return str(status)   # plain str / str-subclass path

key = _status_value(EntryStatus.IMPORTED)  # always "imported"
registry[key] = entry
```

**Rationale**: `.value` is the only shape-stable and version-stable coercion. The `getattr(..., "value", None)` helper also tolerates plain-string inputs, which is what this repo actually sees because `EntryStatus` is `class EntryStatus(str)` today — a future promotion to `StrEnum` must not silently break the mapping lookup. Sourced from PR #29 findings.
