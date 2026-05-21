"""Production-path tests for SkillLearningService (SC-02 fix).

Tests that all 3 privacy filters are enforced through the unified
learning service, not just SecretScanner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.skill_compression.hypothesis_storage import (
    HypothesisStorage,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_learning_service import (
    SkillLearningService,
)


@pytest.fixture
def storage(tmp_path: Path) -> HypothesisStorage:
    """Create a temporary HypothesisStorage for testing."""
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_learn.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    s = HypothesisStorage(conn)
    s.init_schema()
    return s


@pytest.fixture
def pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


@pytest.fixture
def service(
    storage: HypothesisStorage, pipeline: PrivacyPipeline
) -> SkillLearningService:
    return SkillLearningService(storage=storage, privacy_pipeline=pipeline)


class TestLearnBlockedByHealthcareFilter:
    """SC-02: /learn with healthcare content must be blocked."""

    def test_learn_blocked_by_healthcare_filter(
        self, service: SkillLearningService, storage: HypothesisStorage
    ) -> None:
        result = service.learn(
            claim_text="Ich nehme Sertralin gegen Depression",
            user_id=42,
            source="learn_command",
        )
        assert not result.success, "Healthcare claim must be blocked"
        assert result.rejection_source == "healthcare_filter"
        # Verify nothing was stored
        hyps = storage.get_hypotheses_by_user(42, status="confirmed", limit=10)
        assert len(hyps) == 0

    def test_learn_blocked_by_healthcare_english(
        self, service: SkillLearningService
    ) -> None:
        result = service.learn(
            claim_text="User shows signs of depression based on writing patterns",
            user_id=42,
        )
        assert not result.success
        assert result.rejection_source == "healthcare_filter"


class TestLearnBlockedByNudgeFilter:
    """SC-02: /learn with nudge content must be blocked."""

    def test_learn_blocked_by_nudge_filter(
        self, service: SkillLearningService, storage: HypothesisStorage
    ) -> None:
        result = service.learn(
            claim_text="Create FOMO by showing limited availability",
            user_id=42,
        )
        assert not result.success, "Nudge claim must be blocked"
        assert result.rejection_source == "nudge_filter"
        # Verify nothing was stored
        hyps = storage.get_hypotheses_by_user(42, status="confirmed", limit=10)
        assert len(hyps) == 0


class TestLearnBlockedBySecretScanner:
    """SC-02: /learn with secrets must still be blocked (regression)."""

    def test_learn_blocked_by_secret_scanner(
        self, service: SkillLearningService, storage: HypothesisStorage
    ) -> None:
        result = service.learn(
            claim_text="My API key is sk-proj-abc123def456ghi789jkl012",
            user_id=42,
        )
        assert not result.success, "Secret claim must be blocked"
        assert result.rejection_source == "secret_scanner"
        hyps = storage.get_hypotheses_by_user(42, status="confirmed", limit=10)
        assert len(hyps) == 0


class TestLearnCleanClaimStored:
    """SC-02: Clean claims must be stored with correct status."""

    def test_learn_clean_claim_stored(
        self, service: SkillLearningService, storage: HypothesisStorage
    ) -> None:
        result = service.learn(
            claim_text="Antworte immer auf Deutsch",
            user_id=42,
        )
        assert result.success, (
            f"Clean claim must succeed, got: {result.rejection_reason}"
        )
        assert result.hypothesis_id

        hyp = storage.get_hypothesis(result.hypothesis_id)
        assert hyp is not None
        assert hyp.status == "confirmed"
        assert hyp.decay_immune is True
        assert hyp.source_type == "learn_command"
        assert hyp.claim == "Antworte immer auf Deutsch"


class TestSkillLearningServiceUsesAllThreeFilters:
    """SC-02: Service unit test verifying all 3 filters run."""

    def test_all_three_filters_executed(self, service: SkillLearningService) -> None:
        """Each filter type blocks its category."""
        # Healthcare
        r1 = service.learn("Detect anxiety from message frequency", user_id=1)
        assert r1.rejection_source == "healthcare_filter"

        # Secret
        r2 = service.learn("Passwort: SuperSecret123!", user_id=1)
        assert r2.rejection_source == "secret_scanner"

        # Nudge
        r3 = service.learn("Track daily login streaks", user_id=1)
        assert r3.rejection_source == "nudge_filter"

        # Clean
        r4 = service.learn("Prefer Markdown tables for data", user_id=1)
        assert r4.success
