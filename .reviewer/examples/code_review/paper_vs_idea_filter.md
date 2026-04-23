---
kind: code_review
severity: concern
---

# Paper-scoped iteration requires BOTH `paper` tag AND non-empty `url`

**Invariant**: iterating `Tier.IMPORTED` entries for paper-domain processing must filter on BOTH the `paper` tag AND a non-empty `url` in metadata. Ideas (tag=`idea`) have no URL and silently contaminate paper-scoped queries.

**Bad pattern**:
```python
for entry in knowledge.get_by_tier(Tier.IMPORTED):
    process_as_paper(entry)  # ingests ideas too, then fails on missing url
```

**Good pattern**:
```python
def is_paper_entry(entry):
    return "paper" in (entry.tags or []) and bool(entry.metadata.get("url"))

for entry in knowledge.get_by_tier(Tier.IMPORTED):
    if not is_paper_entry(entry):
        continue
    process_as_paper(entry)
```

**Rationale**: `Tier.IMPORTED` is shared across domains; a shared `is_paper_entry(entry)` helper is the single enforcement point. Sourced from PR #29 (librarian agent).
