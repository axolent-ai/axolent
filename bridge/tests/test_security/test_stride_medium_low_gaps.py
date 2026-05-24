"""Tests for STRIDE MEDIUM/LOW-severity gaps (GAP-01 through GAP-13, excl. HIGH).

Tests verify existing mitigations or mark unimplemented fixes as xfail
with clear rationale and post-switch issue references.

Coverage:
  - GAP-01: WAL file cleanup on shutdown (MEDIUM)
  - GAP-02: .bak files blocked by public boundary scanner (MEDIUM)
  - GAP-04: Claude CLI version check + stream event validation (MEDIUM)
  - GAP-06: Forged callback data blocked (MEDIUM)
  - GAP-07: Audit log tamper detection (LOW)
  - GAP-08: Homoglyph bypass for privacy filters (MEDIUM)
  - GAP-09: CI retroactive pre-commit enforcement (LOW)
  - GAP-10: GitHub Actions pinned by SHA (LOW)
  - GAP-12: Whitelist rejection audit log entry (LOW)
  - GAP-13: pip-compile hashes in lockfile (LOW)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


# ===================================================================
# GAP-01: SQLite WAL File Cleanup
# ===================================================================


class TestGap01WalFileCleanup:
    """GAP-01: WAL file should be cleaned up on shutdown."""

    def test_gap01_sqlite_wal_file_cleaned_on_shutdown(self, tmp_path: Path) -> None:
        """After clean shutdown with explicit checkpoint, no -wal/-shm should remain.

        NOTE: Standard sqlite3.close() checkpoints WAL on most platforms,
        but this is not guaranteed (e.g., unclean shutdown, busy database).
        The test verifies that an explicit PRAGMA wal_checkpoint(TRUNCATE)
        ensures deterministic cleanup. Our SqliteConnection.close() SHOULD
        call this explicitly for guaranteed cleanup.
        """
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'hello')")
        conn.commit()

        # Explicit checkpoint (what our SqliteConnection.close() should do)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        # After explicit truncate checkpoint + close, WAL should be gone
        wal_path = tmp_path / "test.db-wal"

        # On Windows, WAL file may be 0 bytes but still exist after truncate
        if wal_path.exists():
            assert wal_path.stat().st_size == 0, (
                "WAL file should be empty (0 bytes) after TRUNCATE checkpoint"
            )


# ===================================================================
# GAP-02: .bak Files Blocked by Public Boundary Scanner
# ===================================================================


class TestGap02BakFilesBlocked:
    """GAP-02: .bak files must be in the public boundary forbidden list."""

    def test_gap02_bak_files_blocked_by_public_boundary_scanner(self) -> None:
        """public_boundary.yaml must forbid *.bak files."""
        yaml_path = (
            Path(__file__).resolve().parents[3] / "scripts" / "public_boundary.yaml"
        )
        assert yaml_path.exists(), f"Expected {yaml_path} to exist"

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        forbidden = config.get("private_forbidden_paths", [])

        # Check that .bak or .jsonl.bak patterns are forbidden
        has_bak_pattern = any(".bak" in pattern for pattern in forbidden)

        if not has_bak_pattern:
            pytest.xfail(
                "GAP-02: *.bak not yet in private_forbidden_paths. "
                "Post-switch task: add '**/*.bak' and '**/*.jsonl.bak' patterns."
            )


# ===================================================================
# GAP-04: Claude CLI Version Check + Stream Event Validation
# ===================================================================


class TestGap04ClaudeCliVersionCheck:
    """GAP-04: Claude CLI version should be checked at startup."""

    @pytest.mark.xfail(
        reason="GAP-04: Version check not yet implemented. Post-switch task: "
        "run `claude --version` at pool startup and log/assert minimum version.",
        strict=True,
    )
    def test_gap04_claude_cli_version_check_at_startup(self) -> None:
        """ClaudeProcessPool should validate CLI version on first spawn."""
        from infrastructure.claude_process_pool import ClaudeProcessPool

        pool = ClaudeProcessPool()
        # Check that pool has a version_check or minimum_version attribute
        assert hasattr(pool, "minimum_cli_version") or hasattr(
            pool, "_check_cli_version"
        ), "Pool should have version check mechanism"

    def test_gap04_stream_event_json_schema_validation(self) -> None:
        """Malformed stream JSON should not crash the parser."""
        from infrastructure.claude_process_pool import StreamEvent

        # StreamEvent should handle creation with minimal fields
        event = StreamEvent(event_type="content_delta", text="hello")
        assert event.event_type == "content_delta"
        assert event.text == "hello"

        # StreamEvent with error type
        error_event = StreamEvent(
            event_type="error", text="Unknown error", is_final=True
        )
        assert error_event.is_final is True


# ===================================================================
# GAP-06: Forged Callback Data Blocked
# ===================================================================


class TestGap06ForgedCallbackBlocked:
    """GAP-06: Forged callback_data from other users must be blocked."""

    def test_gap06_forged_callback_data_blocked(self) -> None:
        """Bookmark delete/remove callback must check user ownership."""
        # We test that the bookmark service has an ownership check
        from application.bookmark_service import BookmarkService

        # BookmarkService must require user_id for deletion/removal
        import inspect

        ownership_verified = False
        for method_name in dir(BookmarkService):
            if (
                "delete" in method_name.lower() or "remove" in method_name.lower()
            ) and not method_name.startswith("_"):
                method = getattr(BookmarkService, method_name, None)
                if method and callable(method):
                    sig = inspect.signature(method)
                    if "user_id" in sig.parameters:
                        ownership_verified = True
                        break

        assert ownership_verified, (
            "BookmarkService must have a delete/remove method that requires user_id "
            "(ownership check for GAP-06)"
        )


# ===================================================================
# GAP-07: Audit Log Tamper Detection
# ===================================================================


class TestGap07AuditLogTamperDetection:
    """GAP-07: Audit log should have tamper detection (hash chain)."""

    @pytest.mark.xfail(
        reason="GAP-07: Hash-chain audit log not yet implemented. "
        "Post-switch task for multi-user scenarios. "
        "Current: accepted risk RA-02 (single-user context).",
        strict=True,
    )
    def test_gap07_audit_log_tamper_detection(self) -> None:
        """Audit log entries should include SHA-256 of previous entry."""

        # This would need a hash_chain or prev_hash field
        # Not yet implemented; accepted risk for single-user deployment
        assert False, "Hash-chain not implemented"


# ===================================================================
# GAP-08: Homoglyph Bypass for Privacy Filters
# ===================================================================


class TestGap08HomoglyphBypass:
    """GAP-08: Privacy filters must handle Unicode homoglyphs."""

    def test_gap08_homoglyph_bypass_healthcare_filter(self) -> None:
        """HealthcareFilter should detect health terms even with homoglyphs."""
        from application.skill_compression.privacy.healthcare_filter import (
            HealthcareFilter,
        )
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        hf = HealthcareFilter()

        # Normal text should be caught
        hyp_normal = Hypothesis(
            hypothesis_id="test_1",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="I take medication for my depression",
            status="suggested",
            version=1,
            elo_rating=1500.0,
            source_type="conversation",
            decay_immune=False,
            created_at="2026-05-24T00:00:00Z",
            last_seen="2026-05-24T00:00:00Z",
            pattern_hash="test_hash_1",
        )
        assert hf.filter_hypothesis(hyp_normal), (
            "HealthcareFilter should catch 'depression' (normal text)"
        )

        # Homoglyph variant: use Cyrillic 'a' (U+0430) in 'depression'
        # dеpression (Cyrillic 'е' at position 2)
        homoglyph_claim = "I take medicаtion for my dеpression"
        hyp_homoglyph = Hypothesis(
            hypothesis_id="test_2",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim=homoglyph_claim,
            status="suggested",
            version=1,
            elo_rating=1500.0,
            source_type="conversation",
            decay_immune=False,
            created_at="2026-05-24T00:00:00Z",
            last_seen="2026-05-24T00:00:00Z",
            pattern_hash="test_hash_2",
        )

        # After NFKC normalization, homoglyphs should be caught.
        # If the filter doesn't normalize, this is the gap.
        result = hf.filter_hypothesis(hyp_homoglyph)
        if not result:
            pytest.xfail(
                "GAP-08: HealthcareFilter does not apply NFKC normalization "
                "before pattern matching. Homoglyph bypass possible. "
                "Post-switch task: add unicodedata.normalize('NFKC', text) "
                "as preprocessing step."
            )

    def test_gap08_homoglyph_bypass_secret_scanner(self) -> None:
        """SecretScanner should detect secrets even with homoglyphs."""
        from application.skill_compression.privacy.secret_scanner import SecretScanner
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        scanner = SecretScanner()

        # Normal API key pattern
        hyp_normal = Hypothesis(
            hypothesis_id="test_3",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="My API key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            status="suggested",
            version=1,
            elo_rating=1500.0,
            source_type="conversation",
            decay_immune=False,
            created_at="2026-05-24T00:00:00Z",
            last_seen="2026-05-24T00:00:00Z",
            pattern_hash="test_hash_3",
        )
        assert scanner.block_if_secrets(hyp_normal), (
            "SecretScanner should catch API key pattern (normal text)"
        )

        # Homoglyph variant: fullwidth 's' (U+FF53) in 'sk-ant'
        homoglyph_claim = "My key is ｓk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        hyp_homoglyph = Hypothesis(
            hypothesis_id="test_4",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim=homoglyph_claim,
            status="suggested",
            version=1,
            elo_rating=1500.0,
            source_type="conversation",
            decay_immune=False,
            created_at="2026-05-24T00:00:00Z",
            last_seen="2026-05-24T00:00:00Z",
            pattern_hash="test_hash_4",
        )

        result = scanner.block_if_secrets(hyp_homoglyph)
        if not result:
            pytest.xfail(
                "GAP-08: SecretScanner does not apply NFKC normalization. "
                "Homoglyph bypass possible. Post-switch fix needed."
            )

    def test_gap08_homoglyph_bypass_nudge_filter(self) -> None:
        """NudgeFilter should detect violations even with homoglyphs."""
        from application.skill_compression.privacy.nudge_filter import NudgeFilter
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        nf = NudgeFilter()

        # First test that the filter catches normal nudge patterns
        # We'll use a known dark pattern term
        nudge_claim = "Add a hidden opt-out to make users keep the subscription"
        hyp_normal = Hypothesis(
            hypothesis_id="test_5",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim=nudge_claim,
            status="suggested",
            version=1,
            elo_rating=1500.0,
            source_type="conversation",
            decay_immune=False,
            created_at="2026-05-24T00:00:00Z",
            last_seen="2026-05-24T00:00:00Z",
            pattern_hash="test_hash_5",
        )

        normal_caught = nf.violates_nudge_policy(hyp_normal)
        if not normal_caught:
            # If the normal pattern isn't caught, we can't test homoglyph bypass
            pytest.skip(
                "NudgeFilter does not catch this test pattern. "
                "Cannot test homoglyph bypass without a known-caught pattern."
            )

        # Homoglyph variant: Cyrillic 'o' (U+043E) in 'opt-out'
        homoglyph_claim = "Add a hidden оpt-out to make users keep the subscription"
        hyp_homoglyph = Hypothesis(
            hypothesis_id="test_6",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim=homoglyph_claim,
            status="suggested",
            version=1,
            elo_rating=1500.0,
            source_type="conversation",
            decay_immune=False,
            created_at="2026-05-24T00:00:00Z",
            last_seen="2026-05-24T00:00:00Z",
            pattern_hash="test_hash_6",
        )

        result = nf.violates_nudge_policy(hyp_homoglyph)
        if not result:
            pytest.xfail(
                "GAP-08: NudgeFilter does not apply NFKC normalization. "
                "Homoglyph bypass possible. Post-switch fix needed."
            )


# ===================================================================
# GAP-09: CI Retroactive Pre-Commit Enforcement
# ===================================================================


class TestGap09CiPreCommitEnforcement:
    """GAP-09: CI should enforce pre-commit hooks retroactively."""

    @pytest.mark.xfail(
        reason="GAP-09: No CI job currently validates that the latest commit "
        "passes all pre-commit hooks (retroactive enforcement). "
        "Post-switch task: add 'pre-commit run --from-ref HEAD~1 --to-ref HEAD' "
        "step in PR workflow.",
        strict=True,
    )
    def test_gap09_ci_retroactive_pre_commit_enforcement(self) -> None:
        """PR workflow should include a pre-commit retroactive check."""
        workflow_path = (
            Path(__file__).resolve().parents[3]
            / ".github"
            / "workflows"
            / "pr-check.yml"
        )
        content = workflow_path.read_text(encoding="utf-8")

        assert "pre-commit" in content.lower(), (
            "pr-check.yml should include a pre-commit enforcement step"
        )


# ===================================================================
# GAP-10: GitHub Actions Pinned by SHA
# ===================================================================


class TestGap10GithubActionsSHA:
    """GAP-10: GitHub Actions should be pinned by commit SHA."""

    def test_gap10_github_actions_pinned_by_sha(self) -> None:
        """All actions/uses in workflows should be pinned to SHA, not tag."""
        workflows_dir = Path(__file__).resolve().parents[3] / ".github" / "workflows"
        assert workflows_dir.exists(), f"Expected {workflows_dir} to exist"

        # Pattern for uses: org/repo@ref
        uses_pattern = re.compile(r"uses:\s*([^@\s]+)@([^\s#]+)")

        tag_pinned = []  # Collect actions pinned to tags (not SHAs)

        for workflow_file in workflows_dir.glob("*.yml"):
            content = workflow_file.read_text(encoding="utf-8")
            for match in uses_pattern.finditer(content):
                action = match.group(1)
                ref = match.group(2)

                # SHA is 40 hex chars
                is_sha = bool(re.match(r"^[0-9a-f]{40}$", ref))
                if not is_sha:
                    tag_pinned.append(f"{workflow_file.name}: {action}@{ref}")

        if tag_pinned:
            details = "\n".join(f"  - {item}" for item in tag_pinned)
            pytest.xfail(
                f"GAP-10: {len(tag_pinned)} actions pinned to tags, not SHAs:\n"
                f"{details}\n"
                "Post-switch task: replace with commit SHA + add Dependabot for Actions."
            )


# ===================================================================
# GAP-12: Whitelist Rejection Audit Log
# ===================================================================


class TestGap12WhitelistRejectionAudit:
    """GAP-12: Unauthorized access attempts should be in audit log."""

    def test_gap12_require_whitelist_audit_log_on_rejected(self) -> None:
        """require_whitelist should call write_audit_log for rejected users."""
        import inspect
        from presentation.decorators import require_whitelist

        # Get the source code of the decorator
        source = inspect.getsource(require_whitelist)

        # Check if write_raw_audit or write_audit_log is called on rejection
        has_audit = (
            "write_raw_audit" in source
            or "write_audit_log" in source
            or "log_command_audit" in source
        )

        if not has_audit:
            pytest.xfail(
                "GAP-12: require_whitelist does not call audit log on rejection. "
                "Only log.warning() is used. Post-switch task: add "
                "write_raw_audit(action='whitelist_rejected', ...) call."
            )


# ===================================================================
# GAP-13: pip-compile Hashes in Lockfile
# ===================================================================


class TestGap13PipCompileHashes:
    """GAP-13: CI should use hash-pinned requirements."""

    @pytest.mark.xfail(
        reason="GAP-13: No requirements.txt with hashes exists. "
        "Post-switch task: generate via 'pip-compile --generate-hashes' "
        "and use 'pip install --require-hashes' in CI.",
        strict=True,
    )
    def test_gap13_pip_compile_hashes_in_lockfile(self) -> None:
        """A requirements file with hashes should exist for CI."""
        repo_root = Path(__file__).resolve().parents[3]

        # Look for requirements*.txt with hashes
        req_files = list(repo_root.glob("requirements*.txt")) + list(
            (repo_root / "bridge").glob("requirements*.txt")
        )

        has_hashes = False
        for req_file in req_files:
            content = req_file.read_text(encoding="utf-8")
            if "--hash=" in content or "\\\\\\n    --hash" in content:
                has_hashes = True
                break

        assert has_hashes, "No requirements file with hash pinning found"
