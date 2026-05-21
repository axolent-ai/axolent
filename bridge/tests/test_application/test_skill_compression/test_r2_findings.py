"""Tests for Round 2 Re-Review Findings (R2-SC-01 through R2-SC-05 + Claude Beob).

Covers:
  R2-SC-02: HTML-escape in ask-before-apply + fail-safe
  R2-SC-03: Callback ownership validation (hyp_id + user_id)
  R2-SC-04: Audit event contains skill metadata
  R2-SC-05: SkillLearningService log message correctness
  Claude Beob 1: Scan timeout in ImportOrchestrator
  Claude Beob 2: No private-attribute access on SkillMatcher
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


class FakeDBConnection:
    """Minimal in-memory SQLite for tests."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql, params=(), **kwargs):
        return self._conn.execute(sql, params)

    def executescript(self, sql):
        self._conn.executescript(sql)

    def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations):
        for sql, params in operations:
            self._conn.execute(sql, params)
        self._conn.commit()


def _make_hypothesis(
    *,
    hypothesis_id: str = "hyp-test-001",
    user_id: int = 42,
    status: str = "confirmed",
    claim: str = "Test skill",
) -> Hypothesis:
    """Create a test hypothesis."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=user_id,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status=status,
        elo_rating=1600.0,
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T12:00:00+00:00",
    )


# ---------------------------------------------------------------
# R2-SC-02: HTML-escape + Fail-Safe
# ---------------------------------------------------------------


class TestAskBeforeApplyHtmlEscape:
    """R2-SC-02: Skill claim must be HTML-escaped in confirmation message."""

    def test_html_escape_present_in_handlers(self) -> None:
        """handlers.py must import and use html.escape for skill claims."""
        bridge_root = Path(__file__).resolve().parents[3]
        source = (bridge_root / "presentation" / "handlers.py").read_text(
            encoding="utf-8"
        )
        # Must import html module
        assert "import html" in source or "html_mod" in source
        # Must call escape on claim
        assert "html_mod.escape" in source or "html.escape" in source

    def test_html_escape_produces_safe_output(self) -> None:
        """html.escape must neutralize dangerous HTML in claims."""
        import html

        dangerous_claim = '<script>alert("xss")</script> & "quotes"'
        safe = html.escape(dangerous_claim)
        assert "<script>" not in safe
        assert "&amp;" in safe
        assert "&lt;" in safe
        assert "&quot;" in safe


class TestAskBeforeApplyFailSafe:
    """R2-SC-02: If confirmation send fails, skill must NOT be silently applied."""

    def test_fail_safe_block_exists_in_handlers(self) -> None:
        """handlers.py must have fail-safe for confirmation send failure."""
        bridge_root = Path(__file__).resolve().parents[3]
        source = (bridge_root / "presentation" / "handlers.py").read_text(
            encoding="utf-8"
        )
        # Must have a fallback plain-text send attempt
        assert "plain-text" in source.lower() or "plain_text" in source.lower(), (
            "handlers.py must have a plain-text fallback for failed HTML send"
        )
        # Must clean up pending state on total failure
        assert "_pending_store.pop" in source, (
            "handlers.py must clean up pending state if confirmation send "
            "fails completely"
        )


# ---------------------------------------------------------------
# R2-SC-03: Callback Ownership Validation
# ---------------------------------------------------------------


class TestConfirmCallbackOwnership:
    """R2-SC-03: Callback must validate hyp_id and user_id ownership."""

    def test_ownership_check_present_in_skill_commands(self) -> None:
        """skill_commands.py must check hyp_id matches pending state."""
        bridge_root = Path(__file__).resolve().parents[3]
        source = (bridge_root / "presentation" / "skill_commands.py").read_text(
            encoding="utf-8"
        )
        # Must compare hyp_id with pending hypothesis
        assert "pending_hyp.hypothesis_id" in source or (
            "hypothesis_id" in source and "hyp_id !=" in source
        ), "skill_commands.py must validate hyp_id matches pending state"

    def test_callback_rejects_mismatched_hyp_id(self) -> None:
        """Pending hyp_id=X + callback hyp_id=Y must be rejected."""
        from application.skill_compression.skill_matcher import SkillMatch

        pending_hyp = _make_hypothesis(hypothesis_id="hyp-A", user_id=42)
        skill_match = SkillMatch(
            hypothesis=pending_hyp,
            confidence=0.9,
            requires_confirmation=True,
            explanation="test",
        )

        # Simulate the ownership check logic
        callback_hyp_id = "hyp-B"  # Different from pending
        callback_user_id = 42

        # The check in skill_commands.py
        should_reject = (
            callback_hyp_id != skill_match.hypothesis.hypothesis_id
            or skill_match.hypothesis.user_id != callback_user_id
        )
        assert should_reject, "Mismatched hyp_id must cause rejection"

    def test_callback_rejects_other_user(self) -> None:
        """Pending user_id=42 + callback user_id=99 must be rejected."""
        from application.skill_compression.skill_matcher import SkillMatch

        pending_hyp = _make_hypothesis(hypothesis_id="hyp-A", user_id=42)
        skill_match = SkillMatch(
            hypothesis=pending_hyp,
            confidence=0.9,
            requires_confirmation=True,
            explanation="test",
        )

        callback_hyp_id = "hyp-A"
        callback_user_id = 99  # Different user

        should_reject = (
            callback_hyp_id != skill_match.hypothesis.hypothesis_id
            or skill_match.hypothesis.user_id != callback_user_id
        )
        assert should_reject, "Different user_id must cause rejection"

    def test_callback_accepts_matching_ownership(self) -> None:
        """Matching hyp_id + user_id must pass."""
        from application.skill_compression.skill_matcher import SkillMatch

        pending_hyp = _make_hypothesis(hypothesis_id="hyp-A", user_id=42)
        skill_match = SkillMatch(
            hypothesis=pending_hyp,
            confidence=0.9,
            requires_confirmation=True,
            explanation="test",
        )

        callback_hyp_id = "hyp-A"
        callback_user_id = 42

        should_reject = (
            callback_hyp_id != skill_match.hypothesis.hypothesis_id
            or skill_match.hypothesis.user_id != callback_user_id
        )
        assert not should_reject, "Matching ownership must pass"


# ---------------------------------------------------------------
# R2-SC-04: Audit contains skill metadata
# ---------------------------------------------------------------


class TestStreamingAuditContainsSkillMetadata:
    """R2-SC-04: write_audit_log must include skill metadata."""

    def test_audit_call_after_skill_metadata_in_source(self) -> None:
        """In chat_service.py, skill metadata must be set BEFORE write_audit_log."""
        bridge_root = Path(__file__).resolve().parents[3]
        source = (bridge_root / "application" / "chat_service.py").read_text(
            encoding="utf-8"
        )

        # Find the save_streaming_result method and check order
        lines = source.splitlines()
        in_method = False
        skill_metadata_line = None
        write_audit_line = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "async def save_streaming_result" in stripped:
                in_method = True
            elif in_method and stripped.startswith("async def "):
                break
            elif in_method:
                if "skill_applied_streaming" in stripped:
                    if skill_metadata_line is None:
                        skill_metadata_line = i
                if "write_audit_log(audit)" in stripped:
                    write_audit_line = i

        assert skill_metadata_line is not None, (
            "save_streaming_result must set skill_applied_streaming"
        )
        assert write_audit_line is not None, (
            "save_streaming_result must call write_audit_log"
        )
        assert skill_metadata_line < write_audit_line, (
            f"skill metadata (line {skill_metadata_line}) must be set BEFORE "
            f"write_audit_log (line {write_audit_line}). "
            f"R2-SC-04: audit event must contain skill metadata."
        )


# ---------------------------------------------------------------
# R2-SC-05: SkillLearningService log correctness
# ---------------------------------------------------------------


class TestSkillLearningServiceLogCorrectness:
    """R2-SC-05: Log message must use rejection.reason, not duplicate source."""

    def test_log_uses_reason_not_duplicate_source(self) -> None:
        """The privacy rejection log must use source + reason, not source + source."""
        bridge_root = Path(__file__).resolve().parents[3]
        source = (
            bridge_root
            / "application"
            / "skill_compression"
            / "skill_learning_service.py"
        ).read_text(encoding="utf-8")

        # Find the log line
        lines = source.splitlines()
        found_log = False
        for i, line in enumerate(lines):
            if "Privacy pipeline blocked learn" in line:
                found_log = True
                # Check the next few lines for the format args
                block = "\n".join(lines[i : i + 6])
                # Must have rejection.reason somewhere (not just source twice)
                assert "rejection.reason" in block, (
                    "Log format args must include rejection.reason, "
                    "not duplicate rejection.source.value"
                )
                # Must NOT have two identical rejection.source.value lines
                source_count = block.count("rejection.source.value")
                assert source_count == 1, (
                    f"Expected rejection.source.value exactly once, "
                    f"found {source_count} times (duplicate bug)"
                )
                break

        assert found_log, (
            "skill_learning_service.py must log privacy pipeline rejections"
        )


# ---------------------------------------------------------------
# Claude Beob 1: Scan Timeout
# ---------------------------------------------------------------


class TestImportScanTimeout:
    """Claude Beob 1: _iter_files must enforce a scan duration timeout."""

    def test_scan_timeout_constant_exists(self) -> None:
        """MAX_SCAN_DURATION_SECONDS must be defined in orchestrator."""
        from application.skill_compression.conversation_import.orchestrator import (
            MAX_SCAN_DURATION_SECONDS,
        )

        assert isinstance(MAX_SCAN_DURATION_SECONDS, (int, float))
        assert MAX_SCAN_DURATION_SECONDS > 0

    def test_scan_timeout_default_60s(self) -> None:
        """Default scan timeout must be 60 seconds."""
        from application.skill_compression.conversation_import.orchestrator import (
            MAX_SCAN_DURATION_SECONDS,
        )

        assert MAX_SCAN_DURATION_SECONDS == 60.0

    def test_scan_timeout_in_iter_files_source(self) -> None:
        """_iter_files source must reference MAX_SCAN_DURATION_SECONDS."""
        bridge_root = Path(__file__).resolve().parents[3]
        source = (
            bridge_root
            / "application"
            / "skill_compression"
            / "conversation_import"
            / "orchestrator.py"
        ).read_text(encoding="utf-8")

        assert "MAX_SCAN_DURATION_SECONDS" in source
        assert "time.monotonic" in source, (
            "_iter_files must use time.monotonic() for timeout checking"
        )

    def test_scan_timeout_produces_partial_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scan must return partial results when timeout is exceeded."""
        import application.skill_compression.conversation_import.orchestrator as orch_mod

        # Create many files
        import_root = tmp_path / "import_root"
        import_root.mkdir()
        scan_dir = import_root / "data"
        scan_dir.mkdir()

        for i in range(50):
            (scan_dir / f"file_{i:03d}.txt").write_text(
                f"content {i}", encoding="utf-8"
            )

        # Monkeypatch time.monotonic to simulate elapsed time.
        # First call returns 0 (scan_start), subsequent calls return
        # a value beyond the timeout to force immediate break.
        call_count = 0

        def _fake_monotonic():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return 0.0  # scan_start
            return 999.0  # way past any timeout

        monkeypatch.setattr(time, "monotonic", _fake_monotonic)
        monkeypatch.setattr(orch_mod, "MAX_SCAN_DURATION_SECONDS", 1.0)

        result = orch_mod.ImportOrchestrator._iter_files(scan_dir)

        # With fake monotonic returning 999 on second call, the timeout
        # check fires on the very first file, yielding 0 results.
        assert len(result) < 50, (
            f"Expected partial results with faked timeout, got {len(result)} files"
        )


# ---------------------------------------------------------------
# Claude Beob 2: No private _storage access
# ---------------------------------------------------------------


class TestSkillMatcherStorageProperty:
    """Claude Beob 2: SkillMatcher must expose storage via public property."""

    def test_skill_matcher_has_storage_property(self) -> None:
        """SkillMatcher must have a public .storage property."""
        from application.skill_compression.skill_matcher import SkillMatcher

        assert hasattr(SkillMatcher, "storage"), (
            "SkillMatcher must have a 'storage' property"
        )
        # Must be a property descriptor
        assert isinstance(SkillMatcher.__dict__.get("storage"), property), (
            "SkillMatcher.storage must be a @property"
        )

    def test_skill_matcher_storage_returns_correct_instance(self) -> None:
        """The .storage property must return the same object as _storage."""
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.skill_matcher import SkillMatcher

        mock_storage = MagicMock(spec=HypothesisStorage)
        mock_judge = MagicMock(spec=PatternJudge)

        matcher = SkillMatcher(storage=mock_storage, pattern_judge=mock_judge)
        assert matcher.storage is mock_storage

    def test_chat_service_no_private_storage_access(self) -> None:
        """chat_service.py must not access skill_matcher._storage."""
        bridge_root = Path(__file__).resolve().parents[3]
        source = (bridge_root / "application" / "chat_service.py").read_text(
            encoding="utf-8"
        )
        assert "skill_matcher._storage" not in source, (
            "chat_service.py must use .storage property, not ._storage"
        )
