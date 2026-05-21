"""Tests for PrivacyPipeline integration (Step 8).

Covers:
  - Pipeline orchestration: all three filters run in order
  - Audit log: rejections are recorded
  - PatternJudge integration: blocked patterns are NOT promoted
  - Architecture guards: privacy filters cannot be bypassed

AG-SC-2 [GUARD]: No secret patterns in hypotheses.
AG-SC-6 [GUARD]: No phenotyping inferences.
"""

from __future__ import annotations

import pytest

from application.skill_compression.bkt import BKTState
from application.skill_compression.evidence_ledger import EvidenceSummary
from application.skill_compression.fsrs_decay import FSRSState
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.pattern_judge import (
    STATUS_CANDIDATE,
    STATUS_CONFIRMED,
    STATUS_PRIVACY_REJECTED,
    STATUS_SUGGESTED,
    PatternJudge,
)
from application.skill_compression.privacy.privacy_pipeline import (
    PrivacyPipeline,
    RejectionSource,
)

# Default BKT for evidence summaries
_DEFAULT_BKT = BKTState(p_knowledge=0.8)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


@pytest.fixture
def judge(pipeline: PrivacyPipeline) -> PatternJudge:
    return PatternJudge(privacy_pipeline=pipeline)


def _hyp(
    claim: str,
    *,
    status: str = STATUS_CANDIDATE,
    support_count: int = 5,
    elo_rating: float = 1700.0,
) -> Hypothesis:
    """Create a test hypothesis."""
    return Hypothesis(
        hypothesis_id="hyp-pipe-test",
        user_id=42,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status=status,
        elo_rating=elo_rating,
        support_count=support_count,
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
    )


def _evidence(
    *,
    total: int = 5,
    positive: int = 5,
    sessions: int = 3,
) -> EvidenceSummary:
    """Create a test evidence summary."""
    return EvidenceSummary(
        total_count=total,
        positive_count=positive,
        negative_count=0,
        weighted_score=0.8,
        bkt_state=_DEFAULT_BKT,
        distinct_sessions=sessions,
        last_positive_at="2026-05-20T10:00:00+00:00",
        last_negative_at=None,
    )


def _bkt(p_knowledge: float = 0.8) -> BKTState:
    """Create a test BKT state."""
    return BKTState(p_knowledge=p_knowledge)


def _fsrs() -> FSRSState:
    """Create a test FSRS state."""
    return FSRSState()


# ---------------------------------------------------------------
# Pipeline orchestration tests
# ---------------------------------------------------------------


class TestPipelineOrchestration:
    """Tests for pipeline filter orchestration."""

    def test_healthcare_blocked(self, pipeline: PrivacyPipeline) -> None:
        """Healthcare filter should block first."""
        h = _hyp("User shows signs of depression")
        rejection = pipeline.check(h)
        assert rejection is not None
        assert rejection.source == RejectionSource.HEALTHCARE

    def test_secret_blocked(self, pipeline: PrivacyPipeline) -> None:
        """Secret scanner should block."""
        h = _hyp("Use sk-1234567890abcdef1234567890abcdef as key")
        rejection = pipeline.check(h)
        assert rejection is not None
        assert rejection.source == RejectionSource.SECRET

    def test_nudge_blocked(self, pipeline: PrivacyPipeline) -> None:
        """Nudge filter should block."""
        h = _hyp("Create engagement loops for user retention")
        rejection = pipeline.check(h)
        assert rejection is not None
        assert rejection.source == RejectionSource.NUDGE

    def test_clean_passes(self, pipeline: PrivacyPipeline) -> None:
        """Clean hypothesis should pass all filters."""
        h = _hyp("User prefers bullet points in summaries")
        rejection = pipeline.check(h)
        assert rejection is None

    def test_is_blocked_convenience(self, pipeline: PrivacyPipeline) -> None:
        """is_blocked convenience method should work."""
        h_bad = _hyp("User has ADHD based on patterns")
        h_good = _hyp("User prefers Markdown tables")
        assert pipeline.is_blocked(h_bad) is True
        assert pipeline.is_blocked(h_good) is False


# ---------------------------------------------------------------
# Audit log tests
# ---------------------------------------------------------------


class TestPipelineAuditLog:
    """Tests for privacy audit logging."""

    def test_rejection_logged(self, pipeline: PrivacyPipeline) -> None:
        """Rejections should be added to audit log."""
        h = _hyp("User shows signs of depression")
        pipeline.check(h)
        assert pipeline.audit_log.total_rejections >= 1

    def test_multiple_rejections_logged(self, pipeline: PrivacyPipeline) -> None:
        """Multiple rejections should accumulate."""
        h1 = _hyp("User shows depression patterns")
        h2 = _hyp("Use sk-abc123def456ghi789jkl012 as default")
        h3 = _hyp("Create engagement loops for sessions")
        pipeline.check(h1)
        pipeline.check(h2)
        pipeline.check(h3)
        assert pipeline.audit_log.total_rejections >= 3

    def test_clean_not_logged(self, pipeline: PrivacyPipeline) -> None:
        """Clean hypotheses should not be logged."""
        h = _hyp("User prefers bullet points")
        pipeline.check(h)
        assert pipeline.audit_log.total_rejections == 0

    def test_get_recent(self, pipeline: PrivacyPipeline) -> None:
        """get_recent should return recent rejections."""
        h = _hyp("User seems sad and depressed")
        pipeline.check(h)
        recent = pipeline.audit_log.get_recent(10)
        assert len(recent) >= 1
        assert recent[0].source == RejectionSource.HEALTHCARE


# ---------------------------------------------------------------
# PatternJudge integration tests
# ---------------------------------------------------------------


class TestPatternJudgePrivacyIntegration:
    """Tests that PatternJudge respects privacy pipeline."""

    def test_candidate_promotion_blocked_by_healthcare(
        self, judge: PatternJudge
    ) -> None:
        """Candidate with health claim should NOT be promoted."""
        h = _hyp("User shows signs of anxiety", status=STATUS_CANDIDATE)
        evidence = _evidence(total=5, positive=5, sessions=3)
        decision = judge.evaluate(h, evidence, _bkt(), 1700.0, _fsrs())
        # Should be rejected, not promoted to suggested
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_PRIVACY_REJECTED

    def test_candidate_promotion_blocked_by_secret(self, judge: PatternJudge) -> None:
        """Candidate with secret should NOT be promoted."""
        h = _hyp(
            "Use token sk-abc123def456ghi789jkl0123456789",
            status=STATUS_CANDIDATE,
        )
        evidence = _evidence(total=5, positive=5, sessions=3)
        decision = judge.evaluate(h, evidence, _bkt(), 1700.0, _fsrs())
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_PRIVACY_REJECTED

    def test_candidate_promotion_blocked_by_nudge(self, judge: PatternJudge) -> None:
        """Candidate with nudge violation should NOT be promoted."""
        h = _hyp(
            "Track daily login streak to maximize engagement",
            status=STATUS_CANDIDATE,
        )
        evidence = _evidence(total=5, positive=5, sessions=3)
        decision = judge.evaluate(h, evidence, _bkt(), 1700.0, _fsrs())
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_PRIVACY_REJECTED

    def test_clean_candidate_promoted(self, judge: PatternJudge) -> None:
        """Clean candidate should be promoted normally."""
        h = _hyp("User prefers bullet points", status=STATUS_CANDIDATE)
        evidence = _evidence(total=5, positive=5, sessions=3)
        decision = judge.evaluate(h, evidence, _bkt(), 1700.0, _fsrs())
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_SUGGESTED

    def test_confirmed_promotion_blocked(self, judge: PatternJudge) -> None:
        """Confirmed hypothesis with health claim should not auto-apply."""
        h = _hyp(
            "User shows signs of burnout",
            status=STATUS_CONFIRMED,
            elo_rating=1800.0,
        )
        evidence = _evidence(total=10, positive=10, sessions=4)
        decision = judge.evaluate(h, evidence, _bkt(0.9), 1800.0, _fsrs())
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_PRIVACY_REJECTED

    def test_judge_without_pipeline_skips_privacy(self) -> None:
        """Judge without pipeline should skip privacy checks (backward compat)."""
        judge = PatternJudge()  # No pipeline
        h = _hyp("User shows signs of depression", status=STATUS_CANDIDATE)
        evidence = _evidence(total=5, positive=5, sessions=3)
        decision = judge.evaluate(h, evidence, _bkt(), 1700.0, _fsrs())
        # Without pipeline, it should promote normally
        assert decision.recommended_status == STATUS_SUGGESTED

    def test_privacy_rejection_reason_in_decision(self, judge: PatternJudge) -> None:
        """Privacy rejection reason should be in the decision."""
        h = _hyp("User has ADHD based on typing patterns")
        evidence = _evidence(total=5, positive=5, sessions=3)
        decision = judge.evaluate(h, evidence, _bkt(), 1700.0, _fsrs())
        assert "Privacy pipeline rejection" in decision.reason


# ---------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------


class TestPrivacyArchitectureGuards:
    """Verify privacy filter architecture constraints."""

    def test_privacy_pipeline_has_all_three_filters(
        self, pipeline: PrivacyPipeline
    ) -> None:
        """Pipeline must have all three filter instances."""
        assert pipeline.healthcare_filter is not None
        assert pipeline.secret_scanner is not None
        assert pipeline.nudge_filter is not None

    def test_privacy_filters_in_application_layer(self) -> None:
        """Privacy modules should be in application layer."""
        import application.skill_compression.privacy.healthcare_filter as hc
        import application.skill_compression.privacy.nudge_filter as nf
        import application.skill_compression.privacy.secret_scanner as ss

        # Verify they don't import from infrastructure
        for mod in [hc, nf, ss]:
            source = mod.__file__
            with open(source, encoding="utf-8") as f:
                content = f.read()
            assert "from infrastructure" not in content
            assert "import infrastructure" not in content

    def test_pattern_judge_privacy_status_exists(self) -> None:
        """STATUS_PRIVACY_REJECTED should be importable."""
        from application.skill_compression.pattern_judge import (
            STATUS_PRIVACY_REJECTED,
        )

        assert STATUS_PRIVACY_REJECTED == "privacy_rejected"

    def test_no_bypass_without_pipeline(self) -> None:
        """Without pipeline, judge works but cannot enforce privacy.
        This test documents that production MUST use a pipeline."""
        judge_no_pipe = PatternJudge()
        assert judge_no_pipe._privacy is None  # noqa: SLF001

    def test_pipeline_fail_fast(self, pipeline: PrivacyPipeline) -> None:
        """Pipeline should return on first rejection (fail-fast)."""
        # This hypothesis triggers BOTH healthcare AND nudge.
        # "depression" is a healthcare keyword (Layer 2),
        # and "engagement loops" is a nudge violation.
        # Pipeline should return healthcare (checked first).
        h = _hyp("Detect depression patterns to create engagement loops")
        rejection = pipeline.check(h)
        assert rejection is not None
        # Healthcare is checked first, so it should be the rejection source
        assert rejection.source == RejectionSource.HEALTHCARE
