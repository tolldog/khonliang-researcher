"""Tests for researcher/server.py's _parse_feed_metadata helper.

Covers a Copilot finding on PR #71: a plain {"error": ...} sentinel dict
misfires when a caller's legitimate metadata payload happens to contain
an "error" key. The fix carries the parse-failure signal out-of-band via
a (parsed, error_message) tuple instead.
"""

from __future__ import annotations

from researcher.server import _parse_feed_metadata


def test_parse_feed_metadata_valid_object():
    parsed, err = _parse_feed_metadata('{"category": "infra"}')
    assert err is None
    assert parsed == {"category": "infra"}


def test_parse_feed_metadata_legitimate_error_key_not_misread():
    parsed, err = _parse_feed_metadata('{"error": "rate limited last week"}')
    assert err is None
    assert parsed == {"error": "rate limited last week"}


def test_parse_feed_metadata_invalid_json():
    parsed, err = _parse_feed_metadata("not json")
    assert parsed is None
    assert err is not None and "valid JSON" in err


def test_parse_feed_metadata_non_object_json():
    parsed, err = _parse_feed_metadata("[1, 2, 3]")
    assert parsed is None
    assert err is not None and "JSON object" in err
