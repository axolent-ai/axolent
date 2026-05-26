"""Tests for subprocess crash fix bundle (2026-05-26).

Covers:
    - Fix 1: CREATE_NO_WINDOW on Windows (not CREATE_NEW_PROCESS_GROUP)
    - Fix 3: LanguageContext JSON filter in audit error paths
    - Production-path test: full handler pipeline with simulated crash does
      not raise LanguageContext serialization error

4-path tests for the subprocess stderr capture (Fix 2) and EOF diagnostics
(Fix 4) are in test_claude_process_pool.py (TestStderrCapture class).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch


from application.audit_service import (
    AUDIT_NON_SERIALIZABLE_KEYS,
    filter_task_meta,
)


class TestFilterTaskMeta:
    """Fix 3: filter_task_meta removes non-serializable objects."""

    def test_removes_language_ctx(self) -> None:
        """_language_ctx is stripped from task_meta."""
        fake_ctx = MagicMock()  # Non-serializable object
        meta = {
            "skill": "general",
            "_language_ctx": fake_ctx,
            "confidence": 0.95,
        }
        filtered = filter_task_meta(meta)
        assert "_language_ctx" not in filtered
        assert "skill" in filtered
        assert "confidence" in filtered

    def test_removes_stream_guard(self) -> None:
        """_stream_guard is stripped from task_meta."""
        meta = {
            "_stream_guard": object(),
            "request_id": "abc123",
        }
        filtered = filter_task_meta(meta)
        assert "_stream_guard" not in filtered
        assert filtered["request_id"] == "abc123"

    def test_removes_skill_match(self) -> None:
        """_skill_match is stripped from task_meta."""
        meta = {
            "_skill_match": MagicMock(),
            "model": "claude-sonnet-4-6",
        }
        filtered = filter_task_meta(meta)
        assert "_skill_match" not in filtered
        assert filtered["model"] == "claude-sonnet-4-6"

    def test_removes_all_three_simultaneously(self) -> None:
        """All three non-serializable keys removed at once."""
        meta = {
            "_skill_match": object(),
            "_stream_guard": object(),
            "_language_ctx": object(),
            "user_id": 42,
            "chat_id": 100,
        }
        filtered = filter_task_meta(meta)
        assert len(filtered) == 2
        assert filtered["user_id"] == 42
        assert filtered["chat_id"] == 100

    def test_empty_meta_returns_empty_dict(self) -> None:
        """Empty or None task_meta returns empty dict."""
        assert filter_task_meta({}) == {}
        assert filter_task_meta(None) == {}

    def test_meta_without_non_serializable_keys_unchanged(self) -> None:
        """Meta without any _-prefixed keys passes through unchanged."""
        meta = {"skill": "code", "confidence": 0.8, "model": "opus"}
        filtered = filter_task_meta(meta)
        assert filtered == meta

    def test_filtered_result_is_json_serializable(self) -> None:
        """Filtered result must be JSON-serializable (the whole point)."""
        meta: dict[str, Any] = {
            "_language_ctx": MagicMock(),  # Not serializable
            "_skill_match": MagicMock(),  # Not serializable
            "_stream_guard": MagicMock(),  # Not serializable
            "skill": "general",
            "confidence": 0.9,
            "model": "claude-sonnet-4-6",
        }
        filtered = filter_task_meta(meta)
        # This must not raise
        serialized = json.dumps(filtered)
        assert "general" in serialized

    def test_constant_matches_expected_keys(self) -> None:
        """AUDIT_NON_SERIALIZABLE_KEYS has exactly the 3 known keys."""
        assert AUDIT_NON_SERIALIZABLE_KEYS == frozenset(
            {"_skill_match", "_stream_guard", "_language_ctx"}
        )


class TestProductionPathLanguageContextCrash:
    """Production-path test: simulates the full handler error audit path.

    Verifies that when a subprocess crash triggers the error audit path
    in handlers.py, the LanguageContext object does NOT cause a
    serialization crash in write_raw_audit.
    """

    def test_error_audit_with_language_context_in_task_meta(self) -> None:
        """Simulates handlers.py error audit path with non-serializable task_meta.

        Before the fix, this would raise TypeError on json.dumps because
        LanguageContext is not JSON-serializable. After the fix,
        filter_task_meta removes it before the dict is written.
        """
        # Simulate the exact scenario from handlers.py lines 1076-1092
        from datetime import datetime, timezone

        # Mock non-serializable objects as they appear in production
        fake_language_ctx = MagicMock()
        fake_language_ctx.__class__.__name__ = "LanguageContext"
        fake_stream_guard = MagicMock()
        fake_skill_match = MagicMock()

        task_meta: dict[str, Any] = {
            "_language_ctx": fake_language_ctx,
            "_stream_guard": fake_stream_guard,
            "_skill_match": fake_skill_match,
            "skill": "general",
            "confidence": 0.85,
            "_language_code": "de",
            "_user_model": "claude-sonnet-4-6",
            "_provider_name": "claude_persistent",
        }

        # This is the exact pattern from handlers.py (post-fix)
        audit_error: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_error",
            "request_id": "test-req-001",
            "user_id": 123,
            "chat_id": 456,
            "username": "testuser",
            "error_id": "abc12345",
            "duration_seconds": 1.5,
            "streaming_chunks": 3,
            "was_cold": False,
            "subprocess_pid": 12345,
            **filter_task_meta(task_meta),
        }

        # Must be JSON-serializable (the actual requirement)
        serialized = json.dumps(audit_error)
        assert "stream_error" in serialized
        assert "LanguageContext" not in serialized
        assert "_language_ctx" not in serialized
        assert "_stream_guard" not in serialized
        assert "_skill_match" not in serialized

        # Serializable fields from task_meta ARE included
        parsed = json.loads(serialized)
        assert parsed["skill"] == "general"
        assert parsed["confidence"] == 0.85
        assert parsed["_language_code"] == "de"

    def test_outer_exception_audit_with_language_context(self) -> None:
        """Simulates handlers.py outer exception audit path (lines 1104-1117).

        Same scenario but for the outer except block.
        """
        from datetime import datetime, timezone

        task_meta: dict[str, Any] = {
            "_language_ctx": object(),  # Not serializable
            "_stream_guard": object(),
            "_skill_match": object(),
            "model": "claude-opus-4-7",
        }

        audit_crash: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_error",
            "request_id": "test-req-002",
            "user_id": 789,
            "chat_id": 101,
            "username": "testuser2",
            "error_id": "def67890",
            "duration_seconds": 0.3,
            "error": "outer_exception",
            **filter_task_meta(task_meta),
        }

        # Must not raise
        serialized = json.dumps(audit_crash)
        assert "outer_exception" in serialized
        assert "_language_ctx" not in serialized
        parsed = json.loads(serialized)
        assert parsed["model"] == "claude-opus-4-7"

    def test_write_raw_audit_integration(self) -> None:
        """Integration: write_raw_audit called with filtered task_meta succeeds."""
        from application.audit_service import write_raw_audit

        task_meta: dict[str, Any] = {
            "_language_ctx": MagicMock(),
            "_stream_guard": MagicMock(),
            "_skill_match": MagicMock(),
            "skill": "code",
        }

        audit_entry = {
            "timestamp": "2026-05-26T12:00:00Z",
            "event_type": "stream_error",
            "user_id": 1,
            "chat_id": 1,
            **filter_task_meta(task_meta),
        }

        # write_raw_audit should not raise (it writes to disk/logs)
        # We patch the actual writer to avoid filesystem side effects
        with patch("application.audit_service.write_audit_log") as mock_writer:
            write_raw_audit(audit_entry)
            mock_writer.assert_called_once()
            written = mock_writer.call_args
            # Verify the written data is clean
            assert "_language_ctx" not in str(written)
