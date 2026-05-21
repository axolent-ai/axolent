"""Tests for Pattern Judge (Layer 4) lifecycle management.

Covers:
  - evaluate: full lifecycle evaluation
  - should_promote: candidate->suggested, confirmed->active
  - should_archive: FSRS decay, decay_immune protection
  - should_challenge: contradiction burst detection
  - detect_collision: scope specificity resolution
  - Risk-differentiated thresholds (negative/preference/procedural)
"""

from __future__ import annotations

from application.skill_compression.bkt import BKTState, create_initial_state
from application.skill_compression.evidence_ledger import EvidenceSummary
from application.skill_compression.fsrs_decay import FSRSState
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.pattern_judge import (
    CONTRADICTION_THRESHOLD,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_CANDIDATE,
    STATUS_CONFIRMED,
    STATUS_NEEDS_REVIEW,
    STATUS_SUGGESTED,
    PatternJudge,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_hypothesis(
    *,
    status: str = STATUS_CANDIDATE,
    h_type: str = "preference",
    decay_immune: bool = False,
    claim: str = "User prefers bullet points",
    client: str = "",
    project: str = "",
    context: tuple[str, ...] = (),
) -> Hypothesis:
    """Create a test hypothesis with sensible defaults."""
    return Hypothesis(
        hypothesis_id="test-hyp-001",
        user_id=12345,
        type=h_type,
        scope=HypothesisScope(project=project, client=client, context=context),
        claim=claim,
        status=status,
        decay_immune=decay_immune,
        elo_rating=1500.0,
    )


def _make_evidence(
    *,
    positive_count: int = 5,
    negative_count: int = 0,
    distinct_sessions: int = 3,
) -> EvidenceSummary:
    """Create a test evidence summary."""
    return EvidenceSummary(
        positive_count=positive_count,
        negative_count=negative_count,
        total_count=positive_count + negative_count,
        weighted_score=0.8,
        bkt_state=create_initial_state(),
        distinct_sessions=distinct_sessions,
        last_positive_at="2026-05-20T10:00:00+00:00",
        last_negative_at=None,
    )


def _high_confidence_bkt() -> BKTState:
    """BKT state with high confidence (> 0.8)."""
    return BKTState(
        p_knowledge=0.85,
        p_init=0.5,
        p_transition=0.1,
        p_slip=0.1,
        p_guess=0.2,
        observations=10,
    )


def _low_confidence_bkt() -> BKTState:
    """BKT state with low confidence (< 0.4)."""
    return BKTState(
        p_knowledge=0.3,
        p_init=0.5,
        p_transition=0.1,
        p_slip=0.1,
        p_guess=0.2,
        observations=5,
    )


# ---------------------------------------------------------------
# Tests: evaluate (full lifecycle)
# ---------------------------------------------------------------


class TestPatternJudgeEvaluate:
    """Tests for the main evaluate method."""

    def test_no_transition_when_criteria_not_met(self) -> None:
        """Evaluate should return no transition for a fresh candidate with weak evidence."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CANDIDATE)
        evidence = _make_evidence(positive_count=1, distinct_sessions=1)
        bkt = create_initial_state()
        fsrs = FSRSState(stability=10.0, last_reviewed="2026-05-20T10:00:00+00:00")

        decision = judge.evaluate(
            hyp, evidence, bkt, 1500.0, fsrs, current_time="2026-05-20T12:00:00+00:00"
        )
        assert decision.should_transition is False
        assert decision.recommended_status == STATUS_CANDIDATE

    def test_archive_takes_priority_over_promotion(self) -> None:
        """Archive check fires before promotion check."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED)
        evidence = _make_evidence(positive_count=10, distinct_sessions=5)
        bkt = _high_confidence_bkt()
        # Very low stability, 200 days elapsed -> should archive
        fsrs = FSRSState(stability=0.5, last_reviewed="2025-11-01T00:00:00+00:00")

        decision = judge.evaluate(
            hyp, evidence, bkt, 1800.0, fsrs, current_time="2026-05-20T00:00:00+00:00"
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_ARCHIVED


# ---------------------------------------------------------------
# Tests: should_promote
# ---------------------------------------------------------------


class TestShouldPromote:
    """Tests for promotion logic."""

    def test_candidate_to_suggested_with_enough_evidence(self) -> None:
        """Candidate with 3+ evidence over 2+ sessions should promote to suggested."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CANDIDATE)
        evidence = _make_evidence(positive_count=3, distinct_sessions=2)
        bkt = create_initial_state()

        assert judge.should_promote(hyp, evidence, bkt) is True

    def test_candidate_not_promoted_with_insufficient_evidence(self) -> None:
        """Candidate with < 3 evidence should NOT promote."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CANDIDATE)
        evidence = _make_evidence(positive_count=2, distinct_sessions=1)
        bkt = create_initial_state()

        assert judge.should_promote(hyp, evidence, bkt) is False

    def test_candidate_not_promoted_single_session(self) -> None:
        """Candidate with enough evidence but only 1 session should NOT promote."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CANDIDATE)
        evidence = _make_evidence(positive_count=5, distinct_sessions=1)
        bkt = create_initial_state()

        assert judge.should_promote(hyp, evidence, bkt) is False

    def test_confirmed_to_active_preference_threshold(self) -> None:
        """Confirmed preference hypothesis meeting all thresholds should promote."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED, h_type="preference")
        # preference threshold: 5 confirmations, Elo >= 1700, 2+ sessions, BKT >= 0.70
        evidence = _make_evidence(positive_count=5, distinct_sessions=3)
        bkt = BKTState(p_knowledge=0.75, observations=8)

        # Elo is passed through evaluate, not stored on evidence
        decision = judge._check_promotion(hyp, evidence, bkt, 1750.0)
        assert decision is not None
        assert decision.recommended_status == STATUS_ACTIVE

    def test_confirmed_not_promoted_low_elo(self) -> None:
        """Confirmed with sufficient evidence but low Elo should NOT promote."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED, h_type="preference")
        evidence = _make_evidence(positive_count=5, distinct_sessions=3)
        bkt = _high_confidence_bkt()

        decision = judge._check_promotion(hyp, evidence, bkt, 1500.0)
        assert decision is None


# ---------------------------------------------------------------
# Tests: should_archive
# ---------------------------------------------------------------


class TestShouldArchive:
    """Tests for archive logic."""

    def test_archive_after_180_days_low_stability(self) -> None:
        """Hypothesis with low stability and 200+ days elapsed should archive."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED)
        fsrs = FSRSState(stability=0.5, last_reviewed="2025-11-01T00:00:00+00:00")

        # ~200 days later
        assert judge.should_archive(hyp, fsrs, "2026-05-20T00:00:00+00:00") is True

    def test_no_archive_within_180_days(self) -> None:
        """Hypothesis used within 180 days should NOT archive."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED)
        fsrs = FSRSState(stability=5.0, last_reviewed="2026-04-01T00:00:00+00:00")

        # Only ~49 days
        assert judge.should_archive(hyp, fsrs, "2026-05-20T00:00:00+00:00") is False

    def test_decay_immune_never_archives(self) -> None:
        """decay_immune=True should prevent archiving regardless of time."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED, decay_immune=True)
        fsrs = FSRSState(stability=0.1, last_reviewed="2025-01-01T00:00:00+00:00")

        # 500+ days, extremely low recall, but immune
        assert judge.should_archive(hyp, fsrs, "2026-05-20T00:00:00+00:00") is False

    def test_high_stability_resists_archive(self) -> None:
        """Very high stability keeps recall above archive threshold."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED)
        fsrs = FSRSState(stability=500.0, last_reviewed="2025-11-01T00:00:00+00:00")

        # 200 days but S=500 -> R still high
        assert judge.should_archive(hyp, fsrs, "2026-05-20T00:00:00+00:00") is False


# ---------------------------------------------------------------
# Tests: should_challenge
# ---------------------------------------------------------------


class TestShouldChallenge:
    """Tests for contradiction-based challenge logic."""

    def test_challenge_when_contradictions_exceed_threshold(self) -> None:
        """Should challenge when recent contradictions >= CONTRADICTION_THRESHOLD."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_ACTIVE)

        assert judge.should_challenge(hyp, CONTRADICTION_THRESHOLD) is True
        assert judge.should_challenge(hyp, CONTRADICTION_THRESHOLD + 1) is True

    def test_no_challenge_below_threshold(self) -> None:
        """Should NOT challenge when contradictions < threshold."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_ACTIVE)

        assert judge.should_challenge(hyp, CONTRADICTION_THRESHOLD - 1) is False

    def test_no_challenge_for_candidates(self) -> None:
        """Candidates and suggested hypotheses are not challenged."""
        judge = PatternJudge()
        for status in (STATUS_CANDIDATE, STATUS_SUGGESTED):
            hyp = _make_hypothesis(status=status)
            assert judge.should_challenge(hyp, CONTRADICTION_THRESHOLD + 5) is False

    def test_challenge_triggers_needs_review_in_evaluate(self) -> None:
        """Full evaluate with high contradictions should recommend needs_review."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_ACTIVE)
        evidence = _make_evidence(positive_count=5, distinct_sessions=3)
        bkt = _high_confidence_bkt()
        fsrs = FSRSState(stability=50.0, last_reviewed="2026-05-15T00:00:00+00:00")

        decision = judge.evaluate(
            hyp,
            evidence,
            bkt,
            1700.0,
            fsrs,
            current_time="2026-05-20T00:00:00+00:00",
            recent_contradictions=CONTRADICTION_THRESHOLD,
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_NEEDS_REVIEW


# ---------------------------------------------------------------
# Tests: detect_collision (scope specificity)
# ---------------------------------------------------------------


class TestDetectCollision:
    """Tests for skill collision detection (HC-SC-11)."""

    def test_specific_scope_wins_over_global(self) -> None:
        """More specific scope should auto-resolve as winner."""
        judge = PatternJudge()
        # Specific: client + project = specificity 3
        specific = _make_hypothesis(
            claim="Client-specific rule", client="acme", project="ads"
        )
        # Global: no client, no project = specificity 0
        global_hyp = _make_hypothesis(claim="Global rule")

        result = judge.detect_collision(specific, global_hyp)
        assert result.has_collision is True
        assert result.winner_id == specific.hypothesis_id
        assert result.needs_user_decision is False

    def test_equal_specificity_needs_user_decision(self) -> None:
        """Equal specificity should require user decision."""
        judge = PatternJudge()
        hyp_a = _make_hypothesis(claim="Rule A", client="acme")
        hyp_b = _make_hypothesis(claim="Rule B", client="beta")

        result = judge.detect_collision(hyp_a, hyp_b)
        assert result.has_collision is True
        assert result.needs_user_decision is True
        assert result.winner_id is None


# ---------------------------------------------------------------
# Tests: Risk-differentiated thresholds
# ---------------------------------------------------------------


class TestRiskDifferentiatedThresholds:
    """Tests verifying different threshold levels per pattern type."""

    def test_negative_specific_lowest_bar(self) -> None:
        """Negative-specific patterns need only 2 confirmations + Elo 1650."""
        judge = PatternJudge()
        # negative + client-scoped = negative_specific
        hyp = _make_hypothesis(
            status=STATUS_CONFIRMED,
            h_type="negative",
            client="acme",
            project="web",
        )
        evidence = _make_evidence(positive_count=2, distinct_sessions=2)
        bkt = BKTState(p_knowledge=0.70, observations=5)

        decision = judge._check_promotion(hyp, evidence, bkt, 1660.0)
        assert decision is not None
        assert decision.threshold_key == "negative_specific"
        assert decision.recommended_status == STATUS_ACTIVE

    def test_procedural_highest_bar(self) -> None:
        """Procedural patterns need 8 confirmations + Elo 1800 + BKT 0.80."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED, h_type="procedural")

        # Meets all thresholds
        evidence = _make_evidence(positive_count=8, distinct_sessions=3)
        bkt = BKTState(p_knowledge=0.85, observations=15)
        decision = judge._check_promotion(hyp, evidence, bkt, 1850.0)
        assert decision is not None
        assert decision.threshold_key == "procedural"

    def test_procedural_fails_with_insufficient_evidence(self) -> None:
        """Procedural with only 5 confirmations should NOT promote."""
        judge = PatternJudge()
        hyp = _make_hypothesis(status=STATUS_CONFIRMED, h_type="procedural")

        evidence = _make_evidence(positive_count=5, distinct_sessions=3)
        bkt = BKTState(p_knowledge=0.85, observations=15)
        decision = judge._check_promotion(hyp, evidence, bkt, 1850.0)
        assert decision is None
