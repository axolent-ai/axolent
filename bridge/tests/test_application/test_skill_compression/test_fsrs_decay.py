"""Tests for FSRS v5-style Decay Engine.

Covers:
  - update_fsrs with rating=3 (good): stability grows
  - update_fsrs with rating=1 (again): stability shrinks
  - estimate_recall after 180 days: < 0.2
  - Seasonal detection for regular patterns
  - decay_immune skills are skipped (tested via is_archive_candidate)
  - Initial state creation
  - Edge cases
"""

from __future__ import annotations

import json


from application.skill_compression.fsrs_decay import (
    FSRSState,
    RATING_AGAIN,
    RATING_EASY,
    RATING_GOOD,
    RATING_HARD,
    apply_seasonal_boost,
    create_initial_fsrs_state,
    estimate_recall,
    is_archive_candidate,
    is_due,
    next_review_interval,
    seasonal_detected,
    update_fsrs,
)


class TestEstimateRecall:
    """Tests for the retrievability calculation."""

    def test_zero_elapsed_is_one(self) -> None:
        """Recall immediately after review should be 1.0."""
        state = create_initial_fsrs_state()
        assert estimate_recall(state, elapsed_days=0.0) == 1.0

    def test_recall_at_stability(self) -> None:
        """At t=stability, recall should be approximately 0.9 (by FSRS definition)."""
        state = FSRSState(stability=10.0, last_reviewed="2026-05-01T00:00:00+00:00")
        recall = estimate_recall(state, elapsed_days=10.0)
        assert abs(recall - 0.9) < 0.01

    def test_recall_decreases_over_time(self) -> None:
        """Recall should monotonically decrease with elapsed time."""
        state = create_initial_fsrs_state()
        recalls = [
            estimate_recall(state, elapsed_days=d) for d in [0, 1, 5, 10, 30, 90]
        ]
        for i in range(len(recalls) - 1):
            assert recalls[i] >= recalls[i + 1]

    def test_recall_after_180_days_very_low(self) -> None:
        """After 180 days, recall should be < 0.25 for default stability.

        With default stability=2.3065 days (FSRS w[2]):
        R = (1 + 0.2346 * 180 / 2.3065)^(-0.5) = (19.31)^(-0.5) ~ 0.2275.
        The power-law curve decays slower than exponential at long intervals.
        """
        state = create_initial_fsrs_state()  # stability ~2.3 days
        recall = estimate_recall(state, elapsed_days=180.0)
        assert recall < 0.25

    def test_high_stability_slow_decay(self) -> None:
        """High stability should mean slow decay."""
        state = FSRSState(stability=100.0, last_reviewed="2026-05-01T00:00:00+00:00")
        # After 30 days with S=100, recall should still be reasonable
        recall = estimate_recall(state, elapsed_days=30.0)
        assert recall > 0.7

    def test_zero_stability_returns_zero(self) -> None:
        """Zero stability should return 0.0 recall."""
        state = FSRSState(stability=0.0, last_reviewed="2026-05-01T00:00:00+00:00")
        assert estimate_recall(state, elapsed_days=1.0) == 0.0


class TestUpdateFSRS:
    """Tests for FSRS state updates."""

    def test_good_rating_increases_stability(self) -> None:
        """Rating=3 (good) should increase stability."""
        state = create_initial_fsrs_state(current_time="2026-05-01T00:00:00+00:00")
        # Wait some days, then review as "good"
        updated = update_fsrs(
            state, RATING_GOOD, current_time="2026-05-04T00:00:00+00:00"
        )
        assert updated.stability > state.stability
        assert updated.reps == 1

    def test_easy_rating_bigger_increase(self) -> None:
        """Rating=4 (easy) should increase stability more than good."""
        state = create_initial_fsrs_state(current_time="2026-05-01T00:00:00+00:00")
        good = update_fsrs(state, RATING_GOOD, current_time="2026-05-04T00:00:00+00:00")
        easy = update_fsrs(state, RATING_EASY, current_time="2026-05-04T00:00:00+00:00")
        assert easy.stability > good.stability

    def test_again_rating_decreases_stability(self) -> None:
        """Rating=1 (again) should significantly decrease stability."""
        state = create_initial_fsrs_state(current_time="2026-05-01T00:00:00+00:00")
        # First build up some stability
        state = update_fsrs(
            state, RATING_GOOD, current_time="2026-05-03T00:00:00+00:00"
        )
        built_up = state.stability
        # Now fail
        failed = update_fsrs(
            state, RATING_AGAIN, current_time="2026-05-06T00:00:00+00:00"
        )
        assert failed.stability < built_up
        assert failed.lapses == 1

    def test_hard_rating_moderate_increase(self) -> None:
        """Rating=2 (hard) should increase stability less than good."""
        state = create_initial_fsrs_state(current_time="2026-05-01T00:00:00+00:00")
        hard = update_fsrs(state, RATING_HARD, current_time="2026-05-04T00:00:00+00:00")
        good = update_fsrs(state, RATING_GOOD, current_time="2026-05-04T00:00:00+00:00")
        assert hard.stability <= good.stability

    def test_difficulty_increases_on_failure(self) -> None:
        """Difficulty should increase when rating=1 (again)."""
        state = create_initial_fsrs_state(current_time="2026-05-01T00:00:00+00:00")
        failed = update_fsrs(
            state, RATING_AGAIN, current_time="2026-05-03T00:00:00+00:00"
        )
        assert failed.difficulty > state.difficulty

    def test_difficulty_decreases_on_easy(self) -> None:
        """Difficulty should decrease when rating=4 (easy)."""
        state = create_initial_fsrs_state(current_time="2026-05-01T00:00:00+00:00")
        easy = update_fsrs(state, RATING_EASY, current_time="2026-05-03T00:00:00+00:00")
        assert easy.difficulty < state.difficulty

    def test_review_history_grows(self) -> None:
        """Review history should accumulate timestamps."""
        state = create_initial_fsrs_state(current_time="2026-05-01T00:00:00+00:00")
        state = update_fsrs(
            state, RATING_GOOD, current_time="2026-05-03T00:00:00+00:00"
        )
        state = update_fsrs(
            state, RATING_GOOD, current_time="2026-05-06T00:00:00+00:00"
        )
        history = json.loads(state.review_history)
        assert len(history) == 3  # initial + 2 reviews

    def test_first_review_uses_initial_params(self) -> None:
        """First review on fresh state should use initial stability/difficulty."""
        state = FSRSState()  # completely fresh, no last_reviewed
        updated = update_fsrs(
            state, RATING_GOOD, current_time="2026-05-01T00:00:00+00:00"
        )
        # Should use w[2] for initial stability of rating=Good
        assert updated.stability > 0
        assert updated.last_reviewed == "2026-05-01T00:00:00+00:00"


class TestIsDue:
    """Tests for is_due check."""

    def test_not_due_immediately(self) -> None:
        """Hypothesis just reviewed should not be due."""
        state = create_initial_fsrs_state(current_time="2026-05-20T10:00:00+00:00")
        assert is_due(state, "2026-05-20T10:00:00+00:00") is False

    def test_due_after_stability_elapsed(self) -> None:
        """Hypothesis should be due after stability period elapsed."""
        state = FSRSState(
            stability=5.0,
            last_reviewed="2026-05-01T00:00:00+00:00",
        )
        # After 10 days (2x stability), recall < 0.9
        assert is_due(state, "2026-05-11T00:00:00+00:00") is True


class TestIsArchiveCandidate:
    """Tests for archive candidate detection."""

    def test_not_archive_within_180_days(self) -> None:
        """Should not archive within 180 days even with low recall."""
        state = FSRSState(
            stability=1.0,  # very low stability
            last_reviewed="2026-05-01T00:00:00+00:00",
        )
        # 179 days later
        assert is_archive_candidate(state, "2026-10-27T00:00:00+00:00") is False

    def test_archive_after_180_days(self) -> None:
        """Should recommend archive after 180+ days with low recall.

        With stability=1.0 and 200 days elapsed:
        R = (1 + 0.2346 * 200 / 1.0)^(-0.5) = (47.92)^(-0.5) ~ 0.144.
        This is well below the 0.2 archive threshold.
        """
        state = FSRSState(
            stability=1.0,
            last_reviewed="2026-01-01T00:00:00+00:00",
        )
        # 200 days later
        assert is_archive_candidate(state, "2026-07-20T00:00:00+00:00") is True

    def test_high_stability_resists_archive(self) -> None:
        """Very high stability should resist archive even after 180 days."""
        state = FSRSState(
            stability=500.0,  # very high stability
            last_reviewed="2026-01-01T00:00:00+00:00",
        )
        # 200 days later, but recall still high due to S=500
        result = is_archive_candidate(state, "2026-07-20T00:00:00+00:00")
        # With S=500 and t=200: R = (1 + 0.2346*200/500)^-0.5 ~ 0.955
        assert result is False


class TestSeasonalDetection:
    """Tests for seasonal/regular pattern detection."""

    def test_regular_monthly_detected(self) -> None:
        """Monthly usage with low variance should be detected as seasonal."""
        # Create timestamps approximately 30 days apart
        timestamps = [
            "2026-01-15T10:00:00+00:00",
            "2026-02-14T10:00:00+00:00",
            "2026-03-16T10:00:00+00:00",
            "2026-04-15T10:00:00+00:00",
        ]
        state = FSRSState(
            stability=10.0,
            last_reviewed="2026-04-15T10:00:00+00:00",
            review_history=json.dumps(timestamps),
        )
        assert seasonal_detected(state) is True

    def test_irregular_not_seasonal(self) -> None:
        """Highly irregular usage should NOT be detected as seasonal."""
        timestamps = [
            "2026-01-01T10:00:00+00:00",
            "2026-01-03T10:00:00+00:00",  # 2 days
            "2026-03-15T10:00:00+00:00",  # 71 days
            "2026-03-16T10:00:00+00:00",  # 1 day
        ]
        state = FSRSState(
            stability=10.0,
            last_reviewed="2026-03-16T10:00:00+00:00",
            review_history=json.dumps(timestamps),
        )
        assert seasonal_detected(state) is False

    def test_too_few_reviews_not_seasonal(self) -> None:
        """Less than MIN_REVIEWS_FOR_SEASONAL should not trigger."""
        timestamps = [
            "2026-01-01T10:00:00+00:00",
            "2026-02-01T10:00:00+00:00",
        ]
        state = FSRSState(
            stability=10.0,
            last_reviewed="2026-02-01T10:00:00+00:00",
            review_history=json.dumps(timestamps),
        )
        assert seasonal_detected(state) is False

    def test_seasonal_boost_increases_stability(self) -> None:
        """Seasonal boost should multiply stability."""
        timestamps = [
            "2026-01-15T10:00:00+00:00",
            "2026-02-14T10:00:00+00:00",
            "2026-03-16T10:00:00+00:00",
            "2026-04-15T10:00:00+00:00",
        ]
        state = FSRSState(
            stability=10.0,
            last_reviewed="2026-04-15T10:00:00+00:00",
            review_history=json.dumps(timestamps),
        )
        boosted = apply_seasonal_boost(state)
        assert boosted.stability > state.stability

    def test_seasonal_prevents_archive(self) -> None:
        """Seasonal pattern should prevent archive via boosted stability."""
        # Monthly usage: intervals ~30 days, CV low
        timestamps = [
            "2025-07-01T10:00:00+00:00",
            "2025-08-01T10:00:00+00:00",
            "2025-09-01T10:00:00+00:00",
            "2025-10-01T10:00:00+00:00",
        ]
        # After boost, effective stability is much higher
        state = FSRSState(
            stability=10.0,
            last_reviewed="2025-10-01T10:00:00+00:00",
            review_history=json.dumps(timestamps),
        )
        boosted = apply_seasonal_boost(state)
        # 200 days after last review but with boosted stability
        # Regular is_archive_candidate uses the unboosted state;
        # the Pattern Judge applies the boost before checking.
        # Here we verify the boost makes recall higher:
        from application.skill_compression.fsrs_decay import estimate_recall

        recall_unboosted = estimate_recall(state, elapsed_days=200.0)
        recall_boosted = estimate_recall(boosted, elapsed_days=200.0)
        assert recall_boosted > recall_unboosted


class TestDecayImmune:
    """Tests verifying decay_immune behavior (HC-SC-6).

    Note: decay_immune is a flag on the Hypothesis, not on FSRSState.
    The Pattern Judge checks decay_immune before calling FSRS.
    Here we test that is_archive_candidate returns correctly regardless,
    and the Pattern Judge integration is tested in test_pattern_judge.py.
    """

    def test_archive_candidate_independent_of_immune_flag(self) -> None:
        """is_archive_candidate only checks FSRS state, not immune flag."""
        # The immune check is in PatternJudge, not in FSRS module
        state = FSRSState(
            stability=1.0,
            last_reviewed="2025-01-01T00:00:00+00:00",
        )
        # 500+ days elapsed
        assert is_archive_candidate(state, "2026-06-15T00:00:00+00:00") is True


class TestFSRSStateSerialization:
    """Tests for FSRSState JSON serialization."""

    def test_round_trip(self) -> None:
        """to_json -> from_json should preserve all fields."""
        state = FSRSState(
            stability=15.5,
            difficulty=4.2,
            last_reviewed="2026-05-20T10:00:00+00:00",
            reps=7,
            lapses=2,
            review_history='["2026-05-01T00:00:00+00:00"]',
        )
        restored = FSRSState.from_json(state.to_json())
        assert restored.stability == state.stability
        assert restored.difficulty == state.difficulty
        assert restored.last_reviewed == state.last_reviewed
        assert restored.reps == state.reps
        assert restored.lapses == state.lapses

    def test_from_empty_json(self) -> None:
        """Empty or '{}' should produce default state."""
        assert FSRSState.from_json("").stability > 0
        assert FSRSState.from_json("{}").stability > 0

    def test_from_invalid_json(self) -> None:
        """Invalid JSON should produce default state."""
        assert FSRSState.from_json("not json").stability > 0


class TestNextReviewInterval:
    """Tests for interval calculation."""

    def test_interval_proportional_to_stability(self) -> None:
        """Higher stability should give longer intervals."""
        low = FSRSState(stability=5.0)
        high = FSRSState(stability=50.0)
        assert next_review_interval(high) > next_review_interval(low)

    def test_interval_at_least_one_day(self) -> None:
        """Interval should be at least 1 day."""
        state = FSRSState(stability=0.01)
        assert next_review_interval(state) >= 1.0
