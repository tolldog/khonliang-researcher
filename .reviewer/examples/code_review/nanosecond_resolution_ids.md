---
kind: code_review
severity: concern
---

# IDs and dedupe keys need `time.time_ns()`, not `int(time.time())`

**Invariant**: IDs, snapshot ids, and dedupe keys derived from `int(time.time())` collide on same-second events. Use `time.time_ns()` for any key that can fire multiple times per second.

**Bad pattern**:
```python
def make_id(prefix):
    return f"{prefix}-{int(time.time())}"

# two events in the same second -> same id -> second event overwrites first
```

**Good pattern**:
```python
def make_id(prefix):
    return f"{prefix}-{time.time_ns()}"
```

**Rationale**: "seems unlikely" is a production incident waiting to happen; batch processing and test runs routinely issue multiple IDs per second. Sourced from PR #29 R3 and R8.
