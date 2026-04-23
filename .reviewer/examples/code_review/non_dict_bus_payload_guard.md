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

**Good pattern** (mirrors `researcher.librarian_agent._handle_bus_event`):
```python
async def _handle_bus_event(self, event: dict[str, Any]) -> None:
    topic = str(event.get("topic", "")).strip()
    raw_payload = event.get("payload")
    if raw_payload is not None and not isinstance(raw_payload, dict):
        logger.warning(
            "non-dict payload for topic %s (type=%s); treating as empty",
            topic,
            type(raw_payload).__name__,
        )
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    user_id = payload.get("user_id")
```

**Rationale**: the bus has no payload-shape enforcement at the transport layer; every subscriber is the last line of defense. One bad publisher must not DOS a subscriber. Use stdlib `logging` with positional `%`-formatting (this repo's style) — structured keyword args like `log.warning("msg", event=event)` would raise `TypeError` here. Sourced from PR #29.
