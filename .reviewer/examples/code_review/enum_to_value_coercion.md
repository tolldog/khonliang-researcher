---
kind: code_review
severity: concern
---

# Use `.value`, never `str(member)`, to coerce a str-based Enum

**Invariant**: converting a `str`-based Enum (or `str.Enum`) member to a lookup/storage key: use `.value`, never `str(member)`. `str(enum_member)` returns `"ClassName.MEMBER"` on `str.Enum` but the raw string on `Enum(str, ...)` — behavior differs between enum shapes, producing silent mismatches.

**Bad pattern**:
```python
class EntryStatus(str, Enum):
    IMPORTED = "imported"

key = str(EntryStatus.IMPORTED)  # "EntryStatus.IMPORTED" on some shapes
registry[key] = entry            # downstream lookup misses with "imported"
```

**Good pattern**:
```python
def enum_value(x):
    return getattr(x, "value", str(x))

key = enum_value(EntryStatus.IMPORTED)  # always "imported"
registry[key] = entry
```

**Rationale**: the two enum shapes are indistinguishable at call site but behave differently under `str()`; `.value` is the only stable coercion. Sourced from PR #29 findings.
