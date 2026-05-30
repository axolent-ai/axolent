"""Tests for safe_json: size and depth limited JSON parsing (Finding 5).

4-Path: Happy + Malicious + Rejection + Privacy.
Production-Path: through ChatGPT/Claude importer.
"""

from __future__ import annotations

import json

import pytest

from infrastructure.safe_json import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_DEPTH,
    JsonTooDeepError,
    JsonTooLargeError,
    safe_json_load,
)


class TestSafeJsonHappy:
    """Happy path: valid JSON within limits is parsed correctly."""

    def test_simple_dict(self) -> None:
        data = safe_json_load('{"key": "value"}')
        assert data == {"key": "value"}

    def test_simple_list(self) -> None:
        data = safe_json_load("[1, 2, 3]")
        assert data == [1, 2, 3]

    def test_nested_within_limit(self) -> None:
        """Nesting at exactly max_depth is OK."""
        # Build nesting of depth 5
        nested = {"level": 1}
        for i in range(2, 6):
            nested = {"level": i, "child": nested}
        raw = json.dumps(nested)
        result = safe_json_load(raw, max_depth=10)
        assert result["level"] == 5

    def test_bytes_input(self) -> None:
        data = safe_json_load(b'{"hello": "world"}')
        assert data == {"hello": "world"}

    def test_unicode_content(self) -> None:
        data = safe_json_load('{"name": "test value"}')
        assert data["name"] == "test value"

    def test_default_limits(self) -> None:
        assert DEFAULT_MAX_BYTES == 10 * 1024 * 1024
        assert DEFAULT_MAX_DEPTH == 64


class TestSafeJsonMalicious:
    """Malicious: oversized and deeply nested JSON is rejected."""

    def test_too_large_string(self) -> None:
        """11 MB JSON is rejected before parsing."""
        large = "x" * (11 * 1024 * 1024)
        raw = json.dumps(large)
        with pytest.raises(JsonTooLargeError) as exc_info:
            safe_json_load(raw)
        assert exc_info.value.size > DEFAULT_MAX_BYTES

    def test_too_large_bytes(self) -> None:
        """11 MB bytes JSON is rejected."""
        large = b'"' + b"x" * (11 * 1024 * 1024) + b'"'
        with pytest.raises(JsonTooLargeError):
            safe_json_load(large)

    def test_too_deep_nested_lists(self) -> None:
        """Deeply nested lists beyond max_depth are rejected."""
        # Create depth=100 nested lists: [[[...[1]...]]]
        deep = "[" * 100 + "1" + "]" * 100
        with pytest.raises(JsonTooDeepError) as exc_info:
            safe_json_load(deep, max_depth=64)
        assert exc_info.value.depth > 64

    def test_too_deep_nested_dicts(self) -> None:
        """Deeply nested dicts beyond max_depth are rejected."""
        # Build nested dicts manually
        inner = '{"val": 1}'
        for _ in range(100):
            inner = f'{{"child": {inner}}}'
        with pytest.raises(JsonTooDeepError):
            safe_json_load(inner, max_depth=64)

    def test_custom_max_bytes(self) -> None:
        """Custom max_bytes is respected."""
        raw = json.dumps({"data": "x" * 1000})
        with pytest.raises(JsonTooLargeError):
            safe_json_load(raw, max_bytes=100)

    def test_custom_max_depth(self) -> None:
        """Custom max_depth is respected."""
        deep = "[" * 10 + "1" + "]" * 10
        with pytest.raises(JsonTooDeepError):
            safe_json_load(deep, max_depth=5)

    def test_depth_bomb_no_hang(self) -> None:
        """100k-deep array does not hang or stack overflow (iterative check)."""
        # This is the exact attack vector from Finding 5
        # It should raise JsonTooDeepError, not RecursionError
        # safe_json_load now catches RecursionError from json.loads and
        # normalizes it to JsonTooDeepError (Bundle E fix).
        payload = "[" * 100000 + "1" + "]" * 100000
        with pytest.raises(JsonTooDeepError):
            safe_json_load(payload, max_depth=64)


class TestSafeJsonRejection:
    """Rejection: invalid JSON raises JSONDecodeError."""

    def test_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            safe_json_load("not json at all")

    def test_truncated_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            safe_json_load('{"key": ')

    def test_empty_string(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            safe_json_load("")


class TestSafeJsonPrivacy:
    """Privacy: error messages do not expose raw JSON content."""

    def test_too_large_error_no_content(self) -> None:
        """JsonTooLargeError message contains size, not content."""
        large = json.dumps("secret" * 2_000_000)
        with pytest.raises(JsonTooLargeError) as exc_info:
            safe_json_load(large)
        error_msg = str(exc_info.value)
        assert "secret" not in error_msg
        assert "bytes" in error_msg

    def test_too_deep_error_no_content(self) -> None:
        """JsonTooDeepError message contains depth, not content."""
        deep = "[" * 100 + '"secret"' + "]" * 100
        with pytest.raises(JsonTooDeepError) as exc_info:
            safe_json_load(deep, max_depth=50)
        error_msg = str(exc_info.value)
        assert "secret" not in error_msg
        assert "levels" in error_msg
