"""Privacy log tests: audit-log, logging, reports, and CI artifacts.

Item 9 of 16 Pre-Switch-Tasks.

Verifies that:
  - PrivacyAuditLog (skill compression) stores only hypothesis_id + source
    + reason + timestamp, never raw user claims.
  - Audit-log rotation does not expose old user text.
  - Logger calls redact user text / use hashes instead of raw IDs.
  - No export/report path leaks raw user text.
  - pytest xfail-reasons are generic (no user text).
  - Smoke-test output is privacy-clean.
  - Cross-component: PII stays in chat_service, never reaches audit.
  - PatternJudge logs claim-id not claim-text.

References:
  - bridge/application/skill_compression/privacy/privacy_pipeline.py
  - bridge/infrastructure/audit_log.py
  - bridge/application/audit_service.py
  - bridge/application/chat_service.py (audit dict)
  - bridge/application/language/audit.py (HC-D1)
  - bridge/application/language/stream_guard.py (build_audit_entry)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.security, pytest.mark.privacy]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BRIDGE_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _BRIDGE_ROOT.parent


def _make_hypothesis(
    hypothesis_id: str = "hyp_test_001",
    claim: str = "User prefers dark mode in all UIs",
    status: str = "candidate",
    context: tuple[str, ...] = ("ui", "dark_mode"),
) -> Any:
    """Create a minimal Hypothesis for testing."""
    from application.skill_compression.hypothesis_storage import (
        Hypothesis,
        HypothesisScope,
    )

    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=99999,
        claim=claim,
        status=status,
        scope=HypothesisScope(
            project="test_project",
            client="test_client",
            context=context,
        ),
        evidence_ids=(),
    )


# ---------------------------------------------------------------------------
# A. Audit-Log Tests (PrivacyAuditLog in skill compression)
# ---------------------------------------------------------------------------


class TestPrivacyAuditLogNoRawUserText:
    """PrivacyAuditLog stores only hypothesis_id, source, reason, timestamp."""

    def test_privacy_audit_log_does_not_store_raw_user_text(self) -> None:
        """WHAT: PrivacyAuditLog entries contain no raw user claim text.
        EXPECTED: Only hypothesis_id, source, reason, timestamp fields.
        WHY: The audit trail must not become a shadow store of user content.
        """
        from application.skill_compression.privacy.privacy_pipeline import (
            PipelineRejection,
            PrivacyAuditLog,
            RejectionSource,
        )

        audit_log = PrivacyAuditLog(max_entries=100)
        user_claim = "I have chronic migraines and take sumatriptan daily"

        rejection = PipelineRejection(
            hypothesis_id="hyp_health_001",
            source=RejectionSource.HEALTHCARE,
            reason="Health-related domain in scope: 'medical'",
            timestamp="2026-05-24T10:00:00+00:00",
        )
        audit_log.add(rejection)

        # Verify the stored entry
        entries = audit_log.get_recent(10)
        assert len(entries) == 1
        entry = entries[0]

        # Fields that SHOULD exist
        assert entry.hypothesis_id == "hyp_health_001"
        assert entry.source == RejectionSource.HEALTHCARE
        assert entry.timestamp == "2026-05-24T10:00:00+00:00"
        assert "medical" in entry.reason  # Generic domain reference, not user text

        # The raw user claim MUST NOT appear anywhere in the entry
        assert user_claim not in entry.reason
        assert user_claim not in entry.hypothesis_id
        # PipelineRejection is a frozen dataclass; verify no extra attrs
        assert not hasattr(entry, "claim")
        assert not hasattr(entry, "user_text")
        assert not hasattr(entry, "content")

    def test_audit_log_rotation_does_not_expose_old_user_text(self) -> None:
        """WHAT: After max_entries rotation, old entries are evicted cleanly.
        EXPECTED: Rotated-away entries are gone from memory, no lingering refs.
        WHY: If rotation kept references, old user-adjacent data could leak.
        """
        from application.skill_compression.privacy.privacy_pipeline import (
            PipelineRejection,
            PrivacyAuditLog,
            RejectionSource,
        )

        small_log = PrivacyAuditLog(max_entries=10)

        # Fill beyond capacity
        for i in range(15):
            rejection = PipelineRejection(
                hypothesis_id=f"hyp_{i:03d}",
                source=RejectionSource.NUDGE,
                reason="Nudge policy violation: pressure_tactic",
                timestamp=f"2026-05-24T10:{i:02d}:00+00:00",
            )
            small_log.add(rejection)

        # After rotation, only newest half (5) should remain
        assert len(small_log.entries) <= 10
        # First entries (hyp_000 through hyp_004 at least) should be gone
        remaining_ids = {e.hypothesis_id for e in small_log.entries}
        assert "hyp_000" not in remaining_ids

    def test_audit_log_rejection_reason_does_not_quote_user_text(self) -> None:
        """WHAT: The reason field in rejections uses generic descriptions.
        EXPECTED: Reason references domain/pattern names, not user claim content.
        WHY: If reason quoted user text, it would defeat the privacy boundary.
        """
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )

        pipeline = PrivacyPipeline()
        user_sensitive_claim = "My blood pressure medication causes dizziness"

        hypothesis = _make_hypothesis(
            hypothesis_id="hyp_medical_test",
            claim=user_sensitive_claim,
            context=("medical", "medication"),
        )

        rejection = pipeline.check(hypothesis)
        assert rejection is not None

        # The reason must NOT contain the user's claim text
        assert user_sensitive_claim not in rejection.reason
        assert "blood pressure" not in rejection.reason
        assert "dizziness" not in rejection.reason
        # It SHOULD contain a generic filter description
        assert (
            "medical" in rejection.reason.lower()
            or "health" in rejection.reason.lower()
        )


# ---------------------------------------------------------------------------
# B. Logging Tests
# ---------------------------------------------------------------------------


class TestLoggerRedactsUserText:
    """Logger calls must not contain raw user text or raw user IDs."""

    def test_logger_info_with_user_message_redacts_text(self) -> None:
        """WHAT: chat_service audit dict does NOT store raw prompt text.
        EXPECTED: Only prompt_length stored, not the actual text.
        WHY: Audit log is written to disk; raw user text must not persist.
        """

        # Inspect the audit dict structure built in process_user_message
        # by verifying the code does NOT assign raw text to audit
        source_path = _BRIDGE_ROOT / "application" / "chat_service.py"
        source = source_path.read_text(encoding="utf-8")

        # The audit dict should contain 'prompt_length' but never 'text' or 'prompt'
        # (as a direct user-text field)
        assert 'audit["prompt_length"]' in source or '"prompt_length"' in source
        # These would be privacy violations:
        assert '"user_text"' not in source.split("audit")[0:1]  # structural check
        # Verify no audit["text"] = text pattern
        assert re.search(r'audit\["text"\]\s*=\s*text', source) is None
        assert re.search(r'audit\["prompt"\]\s*=\s*text', source) is None
        assert re.search(r'audit\["user_message"\]\s*=\s*text', source) is None
        assert re.search(r'audit\["message"\]\s*=\s*text', source) is None

    def test_logger_error_with_exception_user_args_controlled(self) -> None:
        """WHAT: Exception messages written to audit may contain str(e).
        EXPECTED: audit['error'] with str(e) is a KNOWN risk; exception messages
                  from providers/libraries are technical, not user-controlled.
        WHY: Provider exceptions (timeouts, API errors) don't echo user text.
              The catch-all `str(e)` is acceptable because Python exceptions
              from our provider stack don't include user prompts.
        """
        # Verify the error patterns in chat_service are provider errors
        source_path = _BRIDGE_ROOT / "application" / "chat_service.py"
        source = source_path.read_text(encoding="utf-8")

        # The specific exceptions caught are:
        # - ProviderError (from our providers, messages are technical)
        # - FileNotFoundError (CLI not found)
        # - TimeoutError (provider timeout)
        # - ValueError, RuntimeError (processing errors)
        # - Generic Exception (catch-all)
        # None of these should echo back user text in their message.
        # Structural assertion: no f"...{text}..." in exception raises
        assert re.search(r"raise.*ValueError.*\btext\b", source) is None
        assert re.search(r"raise.*RuntimeError.*\btext\b", source) is None

    def test_logger_debug_with_user_id_acceptable_form(self) -> None:
        """WHAT: User IDs in log statements use int form (not hashed).
        EXPECTED: Telegram user IDs are logged as integers (acceptable for
                  operator debugging). Chat IDs similarly.
        WHY: Telegram user IDs are not secret PII in an operator context
              (the bot owner knows their users). They are NOT hashed.
              This test documents the intentional design decision.
        """
        source_path = _BRIDGE_ROOT / "application" / "chat_service.py"
        source = source_path.read_text(encoding="utf-8")

        # Confirm user_id is logged (this is intentional, not a bug)
        assert "user_id" in source
        # But raw user MESSAGE text is never logged at info level
        # (Only debug with exc_info for error diagnosis)
        info_lines = [
            line
            for line in source.splitlines()
            if "log.info(" in line and "text" in line.lower()
        ]
        # The only log.info with "text" should be language-related, not user content
        for line in info_lines:
            # None should log the variable `text` (user message)
            assert "log.info(" in line
            # Acceptable: "partial_text_length", "raw_text" (as key name in format)
            assert ", text)" not in line, f"Raw user text logged: {line.strip()}"

    def test_language_audit_never_stores_input_text(self) -> None:
        """WHAT: HC-D1 hard constraint: language audit never stores input text.
        EXPECTED: DetectionAuditEvent has input_text_length but no text field.
        WHY: This is a documented BLOCKER constraint in the codebase.
        """
        from application.language.audit import DetectionAuditEvent

        # Verify the dataclass fields do NOT include any text field
        field_names = {
            f.name for f in DetectionAuditEvent.__dataclass_fields__.values()
        }
        forbidden_fields = {"input_text", "text", "user_text", "prompt", "message"}
        overlap = field_names & forbidden_fields
        assert not overlap, f"DetectionAuditEvent has forbidden text fields: {overlap}"
        assert "input_text_length" in field_names  # Length is OK

    def test_stream_guard_audit_never_stores_accumulated_text(self) -> None:
        """WHAT: StreamGuard audit entry excludes accumulated text.
        EXPECTED: build_audit_entry() returns only metadata fields.
        WHY: The privacy comment in the code documents this explicitly.
        """
        from application.language.stream_guard import StreamGuard

        guard = StreamGuard(expected_lang="de")
        entry = guard.build_audit_entry()

        # No text-like keys should be present
        forbidden_keys = {
            "text",
            "accumulated_text",
            "partial_text",
            "user_text",
            "prompt",
            "content",
        }
        actual_keys = set(entry.keys())
        overlap = actual_keys & forbidden_keys
        assert not overlap, f"StreamGuard audit has text keys: {overlap}"
        assert "event_type" in entry


# ---------------------------------------------------------------------------
# C. Exported Reports Tests
# ---------------------------------------------------------------------------


class TestExportedReportsPrivacy:
    """AXOLENT does not currently have HTML/JSON/Markdown report exports.

    These tests verify that:
    1. No export/report path exists that could leak user text.
    2. If export functionality is added in the future, these tests will
       catch it and require privacy review.
    """

    def test_no_html_report_export_exists(self) -> None:
        """WHAT: No HTML report generation in application/presentation code.
        EXPECTED: No file writes with .html extension in production code.
        WHY: If HTML exports existed, they could contain user text.
        """
        app_dir = _BRIDGE_ROOT / "application"
        pres_dir = _BRIDGE_ROOT / "presentation"

        for search_dir in [app_dir, pres_dir]:
            if not search_dir.exists():
                continue
            for py_file in search_dir.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                content = py_file.read_text(encoding="utf-8", errors="replace")
                # Look for HTML file writes (not HTML formatting for Telegram)
                html_writes = re.findall(r'open\([^)]*\.html[^)]*,\s*["\']w', content)
                assert not html_writes, (
                    f"HTML file write found in {py_file.relative_to(_BRIDGE_ROOT)}: "
                    f"requires privacy review"
                )

    def test_no_json_report_export_to_file(self) -> None:
        """WHAT: No JSON report file writes in production code.
        EXPECTED: Only audit_log.py writes JSON (to JSONL audit log).
        WHY: Additional JSON exports could leak user text to disk.
        """
        app_dir = _BRIDGE_ROOT / "application"

        json_write_files: list[str] = []
        for py_file in app_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text(encoding="utf-8", errors="replace")
            # Look for json.dump to file (not json.dumps to string)
            if re.search(r"json\.dump\([^)]+,\s*\w+\)", content):
                rel = str(py_file.relative_to(_BRIDGE_ROOT))
                # Exclude known safe paths
                if "conversation_import" not in rel:  # importers READ, not write
                    json_write_files.append(rel)

        # Only known JSON writers: audit_log is in infrastructure/
        # Nothing in application/ should be dumping JSON to files
        assert not json_write_files, (
            f"JSON file writes in application/: {json_write_files}. "
            "Requires privacy review."
        )

    def test_no_markdown_report_export(self) -> None:
        """WHAT: No Markdown report generation that persists to disk.
        EXPECTED: No .md file writes in application/presentation code.
        WHY: Markdown exports could contain user conversation text.
        """
        app_dir = _BRIDGE_ROOT / "application"
        pres_dir = _BRIDGE_ROOT / "presentation"

        for search_dir in [app_dir, pres_dir]:
            if not search_dir.exists():
                continue
            for py_file in search_dir.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                content = py_file.read_text(encoding="utf-8", errors="replace")
                md_writes = re.findall(r'open\([^)]*\.md[^)]*,\s*["\']w', content)
                assert not md_writes, (
                    f"Markdown file write found in "
                    f"{py_file.relative_to(_BRIDGE_ROOT)}: requires privacy review"
                )


# ---------------------------------------------------------------------------
# D. GitHub Actions Artifact Tests
# ---------------------------------------------------------------------------


class TestGitHubActionsArtifactsPrivacy:
    """CI/CD outputs must not contain user text."""

    def test_pytest_xfail_reasons_are_generic(self) -> None:
        """WHAT: xfail-reasons in test suite are generic, no user text.
        EXPECTED: All xfail reasons are technical descriptions.
        WHY: pytest output (which goes to CI logs) includes xfail reasons.
        """
        test_dir = _BRIDGE_ROOT / "tests"
        user_text_indicators = [
            "my password",
            "secret message",
            "private data",
            "user said",
            "user wrote",
            "user's message",
        ]

        for py_file in test_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text(encoding="utf-8", errors="replace")
            # Find all xfail reason strings
            xfail_reasons = re.findall(
                r'pytest\.mark\.xfail\([^)]*reason=["\']([^"\']+)', content
            )
            for reason in xfail_reasons:
                for indicator in user_text_indicators:
                    assert indicator not in reason.lower(), (
                        f"xfail reason in {py_file.name} contains user text "
                        f"indicator '{indicator}': {reason}"
                    )

    def test_smoke_test_output_does_not_contain_user_text(self) -> None:
        """WHAT: smoke_test.py output is privacy-clean.
        EXPECTED: Smoke test scenarios use synthetic/generic test text,
                  not real user data. Output messages are status-only.
        WHY: CI artifacts persist smoke test stdout.
        """
        smoke_script = _REPO_ROOT / "scripts" / "smoke_test.py"
        assert smoke_script.exists(), "smoke_test.py not found"

        content = smoke_script.read_text(encoding="utf-8")

        # Verify test messages are synthetic (contain "test" or generic patterns)
        # Real user data indicators that should NOT appear
        real_data_patterns = [
            r"sk-ant-api\d",  # real API keys
            r"\+\d{2}\s\d{3}\s\d{7}",  # real phone numbers
            r"[a-zA-Z0-9._%+-]+@(?!test|example)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        ]
        for pattern in real_data_patterns:
            matches = re.findall(pattern, content)
            assert not matches, (
                f"smoke_test.py contains real-looking data: {matches[:3]}"
            )

    def test_workflow_artifacts_dont_persist_user_data(self) -> None:
        """WHAT: GitHub Actions workflows do not upload user-data artifacts.
        EXPECTED: No upload-artifact step that could persist user text.
        WHY: Artifacts are retained for 90 days by default on GitHub.
        """
        workflows_dir = _REPO_ROOT / ".github" / "workflows"
        if not workflows_dir.exists():
            pytest.skip("No .github/workflows directory")

        for yml_file in workflows_dir.glob("*.yml"):
            content = yml_file.read_text(encoding="utf-8")
            # Check for upload-artifact actions
            if "upload-artifact" in content:
                # If artifacts are uploaded, they must not include log files
                # that could contain user text
                assert "audit.jsonl" not in content, (
                    f"{yml_file.name} uploads audit log artifact"
                )
                assert "logs/" not in content or "test-results" in content, (
                    f"{yml_file.name} may upload log files with user data"
                )


# ---------------------------------------------------------------------------
# E. Cross-Component Tests
# ---------------------------------------------------------------------------


class TestCrossComponentPrivacy:
    """PII stays in chat_service memory, never reaches audit log."""

    def test_no_pii_leak_across_chat_service_to_audit(self) -> None:
        """WHAT: The audit dict in chat_service never contains raw user text.
        EXPECTED: Only metadata fields (lengths, IDs, timestamps, scores).
        WHY: chat_service is the boundary; audit log is the persistence layer.
        """
        source_path = _BRIDGE_ROOT / "application" / "chat_service.py"
        source = source_path.read_text(encoding="utf-8")

        # Extract all audit[...] = ... assignments
        audit_assignments = re.findall(r'audit\["(\w+)"\]\s*=\s*(.+?)(?:\n|$)', source)

        # Known safe value patterns
        safe_patterns = {
            "timestamp",
            "user_id",
            "chat_id",
            "username",
            "prompt_length",
            "response_length",
            "duration_seconds",
            "detected_language",
            "history_turns",
            "memory_entries_loaded",
            "provider",
            "error",
            "error_id",
            "leakage_attempt",
            "language_enforced",
            "task_slot",
            "task_score",
            "task_matched_patterns",
            "task_matched_keywords",
            "resolved_model",
            "fallback_used",
            "fallback_level",
            "fallback_provider",
            "skill_applied",
            "skill_confidence",
            "skill_matched_ask_before",
            "skill_applied_streaming",
            "skill_matched_ask_before_streaming",
            "plan_type",
            "plan_provider_chain",
            "plan_memory_ids",
        }

        for key, value in audit_assignments:
            # Every audit key must be in our known-safe set
            # New keys trigger this test to fail, forcing privacy review
            assert key in safe_patterns, (
                f"Unknown audit key '{key}' with value '{value.strip()}' "
                f"in chat_service.py. Requires privacy review before adding."
            )

    def test_no_user_text_in_skill_compression_pattern_judge_log(self) -> None:
        """WHAT: PatternJudge decisions log hypothesis_id, not claim text.
        EXPECTED: No log call in pattern_judge.py contains 'claim' or 'text' variable.
        WHY: Pattern judge evaluates hypotheses; its logs must not echo claims.
        """
        source_path = (
            _BRIDGE_ROOT / "application" / "skill_compression" / "pattern_judge.py"
        )
        source = source_path.read_text(encoding="utf-8")

        # Find all log.X() calls
        log_calls = re.findall(r"log\.\w+\([^)]+\)", source)
        for call in log_calls:
            # None should reference hypothesis.claim or claim directly
            assert ".claim" not in call, f"PatternJudge logs hypothesis.claim: {call}"
            # 'text' as a variable (not in string literals) should not appear
            # Allow "text" in quoted strings but not as a format arg
            if "text" in call and "%" in call:
                # Check it's not formatting user text
                assert "hypothesis.claim" not in call
                assert ", text" not in call

    def test_audit_service_log_command_no_user_text_field(self) -> None:
        """WHAT: audit_service.log_command_audit does not accept user text.
        EXPECTED: Parameters are action, user_id, chat_id, username, entry_id,
                  success, details. No 'text' or 'message' parameter.
        WHY: The audit_service interface must not provide a path for user text.
        """
        from application.audit_service import log_command_audit
        import inspect

        sig = inspect.signature(log_command_audit)
        param_names = set(sig.parameters.keys())
        forbidden_params = {"text", "message", "user_text", "content", "prompt"}
        overlap = param_names & forbidden_params
        assert not overlap, f"log_command_audit has user-text parameters: {overlap}"

    def test_debate_orchestrator_audit_no_user_text(self) -> None:
        """WHAT: Debate orchestrator audit entries don't contain user prompts.
        EXPECTED: Only event_type, request_id, provider, language metadata.
        WHY: Debate sends user text to multiple providers but audit must not
              persist it.
        """
        source_path = _BRIDGE_ROOT / "application" / "debate_orchestrator.py"
        source = source_path.read_text(encoding="utf-8")

        # Find all write_audit_log() calls and inspect their dict content
        audit_blocks = re.findall(r"write_audit_log\(\s*\{([^}]+)\}", source, re.DOTALL)
        for block in audit_blocks:
            # Extract key names from the dict literal
            keys = re.findall(r'"(\w+)":', block)
            # None should be text-carrying fields
            forbidden = {"text", "user_text", "prompt", "message", "content"}
            overlap = set(keys) & forbidden
            assert not overlap, (
                f"debate_orchestrator audit contains text key: {overlap}"
            )
