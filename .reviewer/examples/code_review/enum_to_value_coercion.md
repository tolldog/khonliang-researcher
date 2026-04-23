---
kind: code_review
severity: concern
---

# Use `.value`, never `str(member)`, to coerce an Enum to a lookup key

**Invariant**: the string form of an enum member is implementation-dependent across enum shapes (`Enum`, `IntEnum`, `StrEnum`, `class X(str)` subclasses) and Python versions. Always use `.value` explicitly — or a helper that falls back to `str()` for plain-string inputs — when producing a lookup or storage key. Never `str(member)`.

**Bad pattern**:
```python
class EntryStatus(StrEnum):
    IMPORTED = "imported"

key = str(EntryStatus.IMPORTED)  # shape- and version-dependent
registry[key] = entry            # downstream lookup misses with "imported"
```

**Good pattern** (copied verbatim from `researcher.ingest_watcher._status_value`):
```python
def _status_value(status: Any) -> str:
    """Extract the raw string value from a status input.

    Handles three shapes:
    - plain ``str`` / ``str``-subclass (e.g. current ``EntryStatus`` which is
      ``class EntryStatus(str)`` — the class attribute IS the string).
    - ``Enum`` / ``IntEnum`` / ``StrEnum`` where ``str(member)`` returns
      the qualified ``"ClassName.MEMBER"`` form; ``.value`` is the raw value.
    - anything else — fall back to ``str()``.

    Defensive against a future refactor that promotes ``EntryStatus`` to a
    proper ``Enum`` — without this the mapping lookup would silently fail
    and no ingest events would be emitted.
    """
    value = getattr(status, "value", None)
    if value is not None:
        return str(value)
    return str(status)

key = _status_value(EntryStatus.IMPORTED)  # always "imported"
registry[key] = entry
```

**Rationale**: `.value` (or `getattr(x, 'value', x)`) is the only coercion stable across enum shapes AND plain strings. The helper tolerates a future promotion from `class EntryStatus(str)` to a proper `Enum` without silently breaking mapping lookups. Sourced from PR #29 findings.
