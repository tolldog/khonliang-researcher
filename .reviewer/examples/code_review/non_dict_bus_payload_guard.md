---
kind: code_review
severity: concern
---

# Bus payload access must guard `isinstance(payload, dict)`

**Invariant**: bus event handlers receiving `event["payload"]` must guard with `isinstance(payload, dict)` before calling `.get()`. A malformed event with a non-dict payload (string, list, None) crashes `.get()`; without a guard the handler enters an error loop that spams the bus.

**Bad pattern**:
```python
def on_event(self, event):
    payload = event.get("payload", {})
    user_id = payload.get("user_id")  # AttributeError if payload is a string
```

**Good pattern**:
```python
def on_event(self, event):
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        log.warning("non-dict payload", event=event)
        payload = {}
    user_id = payload.get("user_id")
```

**Rationale**: the bus has no payload-shape enforcement at the transport layer; every subscriber is the last line of defense. One bad publisher must not DOS a subscriber. Sourced from PR #29.
