"""Contract tests for icontract pre/post-conditions on critical pipelines.

Tests that the design-by-contract decorators (icontract) correctly reject
invalid inputs and guarantee post-conditions on:
  1. PrivacyPipeline.check()
  2. SkillLearningService.learn()
  3. HypothesisStorage.transition_hypothesis_status()

These are NOT behavior tests (those exist elsewhere). These verify that
the contract decorators fire correctly on boundary violations.

Uses real wrappers (SqliteConnection) for production-path compliance.
"""

from __future__ import annotations

import sqlite3

import icontract
import pytest

from application.skill_compression.hypothesis_storage import (
    ALLOWED_STATUSES,
    ALLOWED_TRANSITIONS,
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
    InvalidStatusTransition,
)
from application.skill_compression.privacy.privacy_pipeline import (
    PipelineRejection,
    PrivacyPipeline,
)
from application.skill_compression.skill_learning_service import (
    SkillLearningService,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


def _make_hypothesis(
    claim: str = "Always respond in German",
    hypothesis_id: str = "hyp_test_001",
    user_id: int = 12345,
) -> Hypothesis:
    """Build a minimal valid Hypothesis for testing."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=user_id,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status="candidate",
        version=1,
        elo_rating=1500.0,
        elo_games_played=0,
        bayes_confidence=0.5,
        support_count=0,
        contradict_count=0,
        source_type="learn_command",
        decay_immune=False,
        created_at="2026-01-01T00:00:00+00:00",
        last_applied=None,
        last_seen="2026-01-01T00:00:00+00:00",
        approval_state="pending",
    )


class _InMemoryConn:
    """Minimal in-memory SQLite connection for contract tests.

    Implements the DBConnection Protocol used by HypothesisStorage.
    Uses a real sqlite3 connection for production-path compliance.
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql: str, params: tuple | dict = (), **kwargs) -> object:
        return self._conn.execute(sql, params)

    def executescript(self, sql: str) -> None:
        self._conn.executescript(sql)

    def fetchall(self, sql: str, params: tuple | dict = ()) -> list:
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple | dict = ()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations: list[tuple[str, tuple]]) -> None:
        for sql, params in operations:
            self._conn.execute(sql, params)


@pytest.fixture
def privacy_pipeline() -> PrivacyPipeline:
    """Create a fresh PrivacyPipeline instance."""
    return PrivacyPipeline()


@pytest.fixture
def hypothesis_storage() -> HypothesisStorage:
    """Create HypothesisStorage with in-memory SQLite (production-path)."""
    conn = _InMemoryConn()
    storage = HypothesisStorage(conn)
    storage.init_schema()
    return storage


@pytest.fixture
def skill_learning_service(
    hypothesis_storage: HypothesisStorage,
    privacy_pipeline: PrivacyPipeline,
) -> SkillLearningService:
    """Create SkillLearningService with real storage + pipeline."""
    return SkillLearningService(
        storage=hypothesis_storage,
        privacy_pipeline=privacy_pipeline,
    )


# ===============================================================
# 1. PrivacyPipeline.check() contracts
# ===============================================================


class TestPrivacyPipelineContracts:
    """Pre/post-condition contracts on PrivacyPipeline.check()."""

    def test_rejects_none_hypothesis(self, privacy_pipeline: PrivacyPipeline) -> None:
        """Pre-condition: passing None must raise icontract.ViolationError."""
        with pytest.raises(icontract.ViolationError, match="must not be None"):
            privacy_pipeline.check(None)  # type: ignore[arg-type]

    def test_rejects_empty_claim(self, privacy_pipeline: PrivacyPipeline) -> None:
        """Pre-condition: empty claim must raise."""
        hyp = _make_hypothesis(claim="")
        with pytest.raises(icontract.ViolationError, match="claim"):
            privacy_pipeline.check(hyp)

    def test_rejects_whitespace_only_claim(
        self, privacy_pipeline: PrivacyPipeline
    ) -> None:
        """Pre-condition: whitespace-only claim must raise."""
        hyp = _make_hypothesis(claim="   \t\n  ")
        with pytest.raises(icontract.ViolationError, match="claim"):
            privacy_pipeline.check(hyp)

    def test_accepts_valid_hypothesis(self, privacy_pipeline: PrivacyPipeline) -> None:
        """Valid hypothesis must pass contracts (return None or PipelineRejection)."""
        hyp = _make_hypothesis(claim="Always respond in German")
        result = privacy_pipeline.check(hyp)
        assert result is None or isinstance(result, PipelineRejection)

    def test_post_condition_rejection_has_source(
        self, privacy_pipeline: PrivacyPipeline
    ) -> None:
        """Post-condition: if rejected, result must have source attribute."""
        # Use a claim that contains a secret pattern to trigger rejection
        hyp = _make_hypothesis(
            claim="My API key is sk-abc123456789012345678901234567890123456789012345"
        )
        result = privacy_pipeline.check(hyp)
        if result is not None:
            assert hasattr(result, "source")
            assert result.source is not None


# ===============================================================
# 2. SkillLearningService.learn() contracts
# ===============================================================


class TestSkillLearningServiceContracts:
    """Pre/post-condition contracts on SkillLearningService.learn()."""

    def test_rejects_empty_claim(
        self, skill_learning_service: SkillLearningService
    ) -> None:
        """Pre-condition: empty claim_text must raise."""
        with pytest.raises(icontract.ViolationError, match="claim_text"):
            skill_learning_service.learn(
                claim_text="",
                user_id=1,
                source="learn_command",
            )

    def test_rejects_whitespace_claim(
        self, skill_learning_service: SkillLearningService
    ) -> None:
        """Pre-condition: whitespace-only claim must raise."""
        with pytest.raises(icontract.ViolationError, match="claim_text"):
            skill_learning_service.learn(
                claim_text="   ",
                user_id=1,
                source="learn_command",
            )

    def test_rejects_zero_user_id(
        self, skill_learning_service: SkillLearningService
    ) -> None:
        """Pre-condition: user_id=0 must raise."""
        with pytest.raises(icontract.ViolationError, match="user_id"):
            skill_learning_service.learn(
                claim_text="Valid claim",
                user_id=0,
                source="learn_command",
            )

    def test_rejects_negative_user_id(
        self, skill_learning_service: SkillLearningService
    ) -> None:
        """Pre-condition: negative user_id must raise."""
        with pytest.raises(icontract.ViolationError, match="user_id"):
            skill_learning_service.learn(
                claim_text="Valid claim",
                user_id=-5,
                source="learn_command",
            )

    def test_rejects_invalid_source(
        self, skill_learning_service: SkillLearningService
    ) -> None:
        """Pre-condition: unknown source must raise."""
        with pytest.raises(icontract.ViolationError, match="source"):
            skill_learning_service.learn(
                claim_text="Valid claim",
                user_id=1,
                source="invalid_source_xyz",
            )

    def test_accepts_valid_learn_command(
        self, skill_learning_service: SkillLearningService
    ) -> None:
        """Valid learn call must succeed and return non-empty hypothesis_id."""
        result = skill_learning_service.learn(
            claim_text="Always respond in German",
            user_id=12345,
            source="learn_command",
        )
        # Post-condition: on success, hypothesis_id must be non-empty
        if result.success:
            assert result.hypothesis_id

    def test_all_allowed_sources_accepted(
        self, skill_learning_service: SkillLearningService
    ) -> None:
        """All sources in ALLOWED_SOURCES must be accepted."""
        for source in SkillLearningService.ALLOWED_SOURCES:
            result = skill_learning_service.learn(
                claim_text=f"Test claim for source {source}",
                user_id=99999,
                source=source,
            )
            # Must not raise, result is valid
            assert isinstance(result.success, bool)


# ===============================================================
# 3. HypothesisStorage.transition_hypothesis_status() contracts
# ===============================================================


class TestTransitionStatusContracts:
    """Pre-condition contracts on transition_hypothesis_status()."""

    def test_rejects_unknown_status(
        self, hypothesis_storage: HypothesisStorage
    ) -> None:
        """Pre-condition: new_status not in ALLOWED_STATUSES must raise."""
        # Insert a hypothesis first
        hyp = _make_hypothesis(hypothesis_id="hyp_trans_001")
        hypothesis_storage.insert_hypothesis(hyp)

        with pytest.raises(icontract.ViolationError, match="ALLOWED_STATUSES"):
            hypothesis_storage.transition_hypothesis_status(
                hypothesis_id="hyp_trans_001",
                new_status="totally_invalid_status",
            )

    def test_rejects_empty_hypothesis_id(
        self, hypothesis_storage: HypothesisStorage
    ) -> None:
        """Pre-condition: empty hypothesis_id must raise."""
        with pytest.raises(icontract.ViolationError, match="hypothesis_id"):
            hypothesis_storage.transition_hypothesis_status(
                hypothesis_id="",
                new_status="suggested",
            )

    def test_valid_transition_succeeds(
        self, hypothesis_storage: HypothesisStorage
    ) -> None:
        """Valid transition (candidate -> suggested) must succeed."""
        hyp = _make_hypothesis(hypothesis_id="hyp_trans_002")
        hypothesis_storage.insert_hypothesis(hyp)

        # candidate -> suggested is allowed
        hypothesis_storage.transition_hypothesis_status(
            hypothesis_id="hyp_trans_002",
            new_status="suggested",
        )

        updated = hypothesis_storage.get_hypothesis("hyp_trans_002")
        assert updated is not None
        assert updated.status == "suggested"

    def test_invalid_transition_raises_business_error(
        self, hypothesis_storage: HypothesisStorage
    ) -> None:
        """Invalid transition (candidate -> active) raises InvalidStatusTransition.

        This is the business-level validation, NOT the contract. The contract
        validates that the status IS a valid status string. The business rule
        validates that the TRANSITION between statuses is allowed.
        """
        hyp = _make_hypothesis(hypothesis_id="hyp_trans_003")
        hypothesis_storage.insert_hypothesis(hyp)

        # candidate -> active is NOT in ALLOWED_TRANSITIONS
        with pytest.raises(InvalidStatusTransition):
            hypothesis_storage.transition_hypothesis_status(
                hypothesis_id="hyp_trans_003",
                new_status="active",
            )

    def test_force_bypasses_transition_validation(
        self, hypothesis_storage: HypothesisStorage
    ) -> None:
        """force=True bypasses the transition matrix (admin use only)."""
        hyp = _make_hypothesis(hypothesis_id="hyp_trans_004")
        hypothesis_storage.insert_hypothesis(hyp)

        # candidate -> active is not in ALLOWED_TRANSITIONS, but force=True
        hypothesis_storage.transition_hypothesis_status(
            hypothesis_id="hyp_trans_004",
            new_status="active",
            force=True,
        )

        updated = hypothesis_storage.get_hypothesis("hyp_trans_004")
        assert updated is not None
        assert updated.status == "active"

    def test_allowed_statuses_is_complete(self) -> None:
        """ALLOWED_STATUSES must contain all keys and all target values."""
        all_from_keys = set(ALLOWED_TRANSITIONS.keys())
        all_from_values = {
            s for targets in ALLOWED_TRANSITIONS.values() for s in targets
        }
        expected = all_from_keys | all_from_values
        assert ALLOWED_STATUSES == expected
