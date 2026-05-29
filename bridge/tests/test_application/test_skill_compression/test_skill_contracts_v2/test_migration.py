"""Tests for Legacy Hypothesis -> Contract Migration (Etappe 4, T11).

Tests:
  1. Idempotent: second run creates no duplicates
  2. Dry-run: no writes, report only
  3. Report counts: migrated / needs_review / skipped / failed_validation
  4. Unsure extraction -> needs_review (not confirmed)
  5. Legacy skill triggers after migration (via contract matcher)
  6. Failure-Injection: contract persist fails -> legacy unchanged
  7. origin=migrated is set on migrated contracts
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from application.skill_compression.contract_store import (
    ContractStore,
    ContractStoreError,
)
from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.hypothesis_storage import (
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import PatternJudge
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_contract import now_iso
from application.skill_compression.skill_matcher import SkillMatcher
from scripts.migrate_hypotheses_to_contracts import (
    _extract_trigger_instruction,
    run,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def db_conn(tmp_path: Path):
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_migration.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    return conn


@pytest.fixture
def hypothesis_storage(db_conn) -> HypothesisStorage:
    storage = HypothesisStorage(db_conn)
    storage.init_schema()
    return storage


@pytest.fixture
def contract_store(db_conn) -> ContractStore:
    store = ContractStore(db_conn)
    store.init_schema()
    return store


@pytest.fixture
def privacy_pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


def _make_event(user_id: int, text: str) -> NormalizedEvent:
    return NormalizedEvent(
        event_id="evt_test",
        user_id=user_id,
        timestamp=now_iso(),
        raw_text=text,
        intent="",
        domain="",
        format_type="",
        language="en",
        fingerprint_hash="",
    )


def _create_hypothesis(
    storage: HypothesisStorage,
    user_id: int,
    claim: str,
    status: str = "confirmed",
) -> str:
    """Create a hypothesis and return its ID."""
    from uuid import uuid4

    from application.skill_compression.hypothesis_storage import Hypothesis

    hyp_id = f"hyp_{uuid4().hex[:16]}"
    ts = now_iso()
    hyp = Hypothesis(
        hypothesis_id=hyp_id,
        user_id=user_id,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status=status,
        version=1,
        elo_rating=1500.0,
        source_type="learn_command",
        decay_immune=True,
        created_at=ts,
        last_seen=ts,
    )
    storage.insert_hypothesis(hyp)
    # Add alias for matching
    trigger_parts = claim.split(",")[0] if "," in claim else claim
    alias_text = trigger_parts.replace("when I say ", "").strip()
    if alias_text and alias_text != claim:
        storage.insert_alias(
            alias_id=f"alias_{uuid4().hex[:12]}",
            hypothesis_id=hyp_id,
            alias_text=alias_text,
            first_seen=ts,
            last_seen=ts,
            confidence=0.9,
        )
    return hyp_id


# ---------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------


class TestTriggerExtraction:
    """Test trigger/instruction extraction from claim text."""

    def test_when_i_say_pattern(self):
        trigger, instruction, confident = _extract_trigger_instruction(
            "when I say hello, respond with hi"
        )
        assert trigger == "hello"
        assert instruction == "respond with hi"
        assert confident is True

    def test_wenn_ich_sage_pattern(self):
        trigger, instruction, confident = _extract_trigger_instruction(
            "wenn ich gruss sage, antworte mit hallo"
        )
        assert trigger == "gruss"
        assert instruction == "antworte mit hallo"
        assert confident is True

    def test_unclear_format_not_confident(self):
        trigger, instruction, confident = _extract_trigger_instruction(
            "always respond in German"
        )
        assert trigger == ""
        assert instruction == "always respond in German"
        assert confident is False


# ---------------------------------------------------------------
# Migration core tests
# ---------------------------------------------------------------


class TestMigrationIdempotent:
    """Migration is idempotent: second run creates no duplicates."""

    def test_idempotent_second_run_no_duplicates(
        self, hypothesis_storage, contract_store
    ):
        """Running migration twice produces same result."""
        _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )

        # First run
        report1 = run(hypothesis_storage, contract_store, dry_run=False)
        assert report1.migrated == 1
        assert report1.skipped == 0

        # Second run (idempotent)
        report2 = run(hypothesis_storage, contract_store, dry_run=False)
        assert report2.migrated == 0
        assert report2.skipped == 1

        # Still exactly 1 contract
        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1


class TestMigrationDryRun:
    """Dry-run mode: report only, no writes."""

    def test_dry_run_no_writes(self, hypothesis_storage, contract_store):
        """Dry-run does not create any contracts."""
        _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )

        report = run(hypothesis_storage, contract_store, dry_run=True)
        assert report.migrated == 1

        # No contracts actually created
        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 0


class TestMigrationReport:
    """Report contains correct counts."""

    def test_report_counts(self, hypothesis_storage, contract_store):
        """Report has correct migrated/needs_review/skipped counts."""
        # Clear hypothesis: will be migrated
        _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )
        # Unclear hypothesis: will be needs_review
        _create_hypothesis(
            hypothesis_storage,
            42,
            "always respond in German",
        )

        report = run(hypothesis_storage, contract_store, dry_run=False)
        assert report.migrated == 1
        assert report.needs_review == 1
        assert report.skipped == 0
        assert report.failed_validation == 0

    def test_needs_review_hypothesis(self, hypothesis_storage, contract_store):
        """Unclear claim -> needs_review contract (not confirmed)."""
        _create_hypothesis(
            hypothesis_storage,
            42,
            "remember to use formal language",
        )

        report = run(hypothesis_storage, contract_store, dry_run=False)
        assert report.needs_review == 1

        # Check the created contract has review status
        contracts = contract_store.get_by_user(42)
        review_contracts = [
            c for c in contracts if c.lifecycle.status == "needs_review"
        ]
        assert len(review_contracts) == 1


class TestMigrationOrigin:
    """Migrated contracts have origin=migrated."""

    def test_origin_migrated(self, hypothesis_storage, contract_store):
        _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )

        run(hypothesis_storage, contract_store, dry_run=False)

        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1
        assert contracts[0].origin == "migrated"


class TestMigrationLegacyTriggers:
    """Legacy skill triggers after migration (via contract matcher)."""

    def test_legacy_triggers_via_contract_after_migration(
        self, hypothesis_storage, contract_store, privacy_pipeline
    ):
        """After migration, the skill triggers via contract matcher."""
        hyp_id = _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )

        # Migrate
        run(hypothesis_storage, contract_store, dry_run=False)

        # Create matcher with contract store
        judge = PatternJudge(privacy_pipeline=privacy_pipeline)
        matcher = SkillMatcher(
            storage=hypothesis_storage,
            pattern_judge=judge,
            contract_store=contract_store,
        )

        # Match via contract (not legacy)
        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.contract is not None
        assert result.match_source == "contract"
        assert result.contract.hypothesis_id == hyp_id


class TestMigrationFailureInjection:
    """Contract persist failure -> legacy unchanged, no partial write."""

    def test_contract_persist_fail_legacy_unchanged(self, hypothesis_storage):
        """If contract persist fails, hypothesis is not marked as migrated."""
        hyp_id = _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )

        # Mock contract store that always fails
        mock_cs = MagicMock(spec=ContractStore)
        mock_cs.get_by_hypothesis_id = MagicMock(return_value=None)
        mock_cs.persist = MagicMock(side_effect=ContractStoreError("Simulated failure"))

        report = run(hypothesis_storage, mock_cs, dry_run=False)
        assert report.failed_validation == 1
        assert report.migrated == 0

        # Legacy hypothesis is still confirmed (not modified)
        hyp = hypothesis_storage.get_hypothesis(hyp_id)
        assert hyp is not None
        assert hyp.status == "confirmed"

    def test_both_directions_atomic(self, hypothesis_storage, contract_store):
        """Migration is atomic: contract created IFF migration succeeds."""
        hyp_id = _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )

        # Successful migration
        report = run(hypothesis_storage, contract_store, dry_run=False)
        assert report.migrated == 1

        # Both exist: hypothesis (unchanged) and contract (new)
        hyp = hypothesis_storage.get_hypothesis(hyp_id)
        assert hyp is not None
        assert hyp.status == "confirmed"

        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1
        assert contracts[0].hypothesis_id == hyp_id


class TestMigrationDoubleRunRace:
    """Race condition: two migrations skip the idempotent check simultaneously."""

    def test_double_run_race_reports_failed_not_crash(
        self, hypothesis_storage, contract_store
    ):
        """If idempotent check passes but persist hits hypothesis_id duplicate,
        migration reports failed_validation instead of crashing with raw
        sqlite3.IntegrityError."""
        import sqlite3

        _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say alpha, respond with beta",
        )

        # First migration succeeds normally
        report1 = run(hypothesis_storage, contract_store, dry_run=False)
        assert report1.migrated == 1

        # Simulate race: patch get_by_hypothesis_id to return None
        # (as if a concurrent process hasn't committed yet)
        # so the idempotent check is bypassed, forcing a DB-level clash
        original_get = contract_store.get_by_hypothesis_id
        contract_store.get_by_hypothesis_id = lambda *a, **kw: None

        try:
            report2 = run(hypothesis_storage, contract_store, dry_run=False)
            # Must NOT crash with raw sqlite3.IntegrityError
            # Should report failed_validation (caught ContractStoreError)
            assert report2.failed_validation >= 1, (
                "Race-induced duplicate should be reported as failed_validation"
            )
            assert report2.migrated == 0
        except sqlite3.IntegrityError:
            pytest.fail(
                "Raw sqlite3.IntegrityError leaked during migration double-run. "
                "Should be caught and reported as failed_validation."
            )
        finally:
            contract_store.get_by_hypothesis_id = original_get


class TestMigrationMultiUser:
    """Migration handles multiple users correctly."""

    def test_multi_user_migration(self, hypothesis_storage, contract_store):
        """Each user's hypotheses are migrated independently."""
        _create_hypothesis(
            hypothesis_storage,
            42,
            "when I say hello, respond with hi",
        )
        _create_hypothesis(
            hypothesis_storage,
            99,
            "when I say bonjour, respond with salut",
        )

        report = run(hypothesis_storage, contract_store, dry_run=False)
        assert report.migrated == 2

        assert len(contract_store.get_by_user(42)) == 1
        assert len(contract_store.get_by_user(99)) == 1
