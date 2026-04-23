---
kind: code_review
severity: concern
---

# Paper-scoped iteration requires BOTH `paper` tag AND non-empty `url`

**Invariant**: iterating `Tier.IMPORTED` entries for paper-domain processing must filter on BOTH the `paper` tag AND a non-empty `url` in metadata. Ideas (tag=`idea`) have no URL and silently contaminate paper-scoped queries. Use the shared `researcher.pipeline.is_paper_entry` helper — don't reimplement the predicate per call site.

**Bad pattern**:
```python
for entry in knowledge.get_by_tier(Tier.IMPORTED):
    process_as_paper(entry)  # ingests ideas too, then fails on missing url
```

**Good pattern** (verbatim from `researcher/pipeline.py`):
```python
def is_paper_entry(entry: KnowledgeEntry) -> bool:
    if "paper" not in (entry.tags or []):
        return False
    meta = entry.metadata or {}
    url = str(meta.get("url", "") or "").strip()
    return bool(url)

for entry in knowledge.get_by_tier(Tier.IMPORTED):
    if not is_paper_entry(entry):
        continue
    process_as_paper(entry)
```

**Rationale**: `Tier.IMPORTED` is shared across domains; a single enforcement point prevents drift. The real helper guards `entry.metadata` with `or {}` (tolerates `None` metadata) and strips the URL before the truthiness check (rejects whitespace-only URLs) — subtle details a reimplementation would miss. Sourced from PR #29 (librarian agent).
