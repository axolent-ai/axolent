"""Tests for Bayesian Knowledge Tracing (BKT) module.

Covers:
  - Initial state correctness
  - Positive observation convergence
  - Negative observation convergence
  - Mixed observations
  - Weighted updates
  - Batch updates
  - Edge cases (slip/guess probabilities)
"""

from __future__ import annotations

import pytest

from application.skill_compression.bkt import (
    batch_update_bkt,
    create_initial_state,
    update_bkt,
    update_bkt_weighted,
)


class TestBKTInitialState:
    """Tests for BKT initial state creation."""

    def test_default_initial_state(self) -> None:
        """Initial state should have p_knowledge = 0.5 (uninformative prior)."""
        state = create_initial_state()
        assert state.p_knowledge == 0.5
        assert state.p_init == 0.5
        assert state.p_transition == 0.1
        assert state.p_slip == 0.1
        assert state.p_guess == 0.2
        assert state.observations == 0

    def test_custom_initial_state(self) -> None:
        """Custom parameters should be preserved."""
        state = create_initial_state(
            p_init=0.3,
            p_transition=0.2,
            p_slip=0.05,
            p_guess=0.1,
        )
        assert state.p_knowledge == 0.3
        assert state.p_transition == 0.2
        assert state.p_slip == 0.05
        assert state.p_guess == 0.1

    def test_state_is_frozen(self) -> None:
        """BKTState should be immutable (frozen=True)."""
        state = create_initial_state()
        with pytest.raises(AttributeError):
            state.p_knowledge = 0.9  # type: ignore[misc]


class TestBKTPositiveObservations:
    """Tests for positive observation updates."""

    def test_single_positive_increases_knowledge(self) -> None:
        """One positive observation should increase p_knowledge from 0.5."""
        state = create_initial_state()
        updated = update_bkt(state, positive_observation=True)
        assert updated.p_knowledge > state.p_knowledge
        assert updated.observations == 1

    def test_five_positives_high_confidence(self) -> None:
        """5 positive observations should push p_knowledge > 0.85."""
        state = create_initial_state()
        for _ in range(5):
            state = update_bkt(state, positive_observation=True)
        assert state.p_knowledge > 0.85
        assert state.observations == 5

    def test_ten_positives_very_high_confidence(self) -> None:
        """10 positives should approach certainty (> 0.95)."""
        state = create_initial_state()
        for _ in range(10):
            state = update_bkt(state, positive_observation=True)
        assert state.p_knowledge > 0.95

    def test_positive_never_exceeds_one(self) -> None:
        """p_knowledge should never exceed 0.999 (clamped)."""
        state = create_initial_state()
        for _ in range(100):
            state = update_bkt(state, positive_observation=True)
        assert state.p_knowledge <= 0.999


class TestBKTNegativeObservations:
    """Tests for negative observation updates."""

    def test_single_negative_decreases_knowledge(self) -> None:
        """One negative observation should decrease p_knowledge from 0.5."""
        state = create_initial_state()
        updated = update_bkt(state, positive_observation=False)
        assert updated.p_knowledge < state.p_knowledge
        assert updated.observations == 1

    def test_five_negatives_low_confidence(self) -> None:
        """5 negative observations should push p_knowledge < 0.20."""
        state = create_initial_state()
        for _ in range(5):
            state = update_bkt(state, positive_observation=False)
        assert state.p_knowledge < 0.20
        assert state.observations == 5

    def test_negative_never_below_zero(self) -> None:
        """p_knowledge should never drop below 0.001 (clamped)."""
        state = create_initial_state()
        for _ in range(100):
            state = update_bkt(state, positive_observation=False)
        assert state.p_knowledge >= 0.001


class TestBKTMixedObservations:
    """Tests for mixed positive/negative observation sequences."""

    def test_three_positive_two_negative(self) -> None:
        """Mixed (3+, 2-): p_knowledge should be between 0.4 and 0.8.

        With standard BKT parameters (P_init=0.5, P_T=0.1, P_S=0.1, P_G=0.2),
        the learning transition step P' = P + (1-P)*P_T pushes knowledge up
        after each observation. 3 positives drive p_knowledge ~0.88, then
        2 negatives bring it down to ~0.72. This is mathematically correct
        for BKT with learning transition applied after each Bayes update.
        """
        state = create_initial_state()
        # Three positives
        for _ in range(3):
            state = update_bkt(state, positive_observation=True)
        # Two negatives
        for _ in range(2):
            state = update_bkt(state, positive_observation=False)
        assert 0.4 <= state.p_knowledge <= 0.8
        assert state.observations == 5

    def test_alternating_converges_near_prior(self) -> None:
        """Alternating +/- should keep knowledge near the prior range."""
        state = create_initial_state()
        for _ in range(10):
            state = update_bkt(state, positive_observation=True)
            state = update_bkt(state, positive_observation=False)
        # Should stay in a reasonable middle range
        assert 0.3 < state.p_knowledge < 0.8

    def test_recovery_after_negatives(self) -> None:
        """After negatives, positives should recover confidence."""
        state = create_initial_state()
        # Drive down
        for _ in range(5):
            state = update_bkt(state, positive_observation=False)
        low_point = state.p_knowledge
        # Recover
        for _ in range(8):
            state = update_bkt(state, positive_observation=True)
        assert state.p_knowledge > low_point
        assert state.p_knowledge > 0.5


class TestBKTSlipAndGuess:
    """Tests verifying slip and guess parameters are respected."""

    def test_high_slip_reduces_positive_impact(self) -> None:
        """High slip probability should reduce the impact of positive observations."""
        normal = create_initial_state(p_slip=0.1)
        high_slip = create_initial_state(p_slip=0.4)

        normal_updated = update_bkt(normal, positive_observation=True)
        high_slip_updated = update_bkt(high_slip, positive_observation=True)

        # High slip means even with positive obs, we are less certain
        # (because positives could be due to correct hypothesis OR slip)
        # Actually, high slip reduces P(obs+|know) = 1-P(S), making
        # the positive evidence weaker
        assert high_slip_updated.p_knowledge < normal_updated.p_knowledge

    def test_high_guess_reduces_positive_impact(self) -> None:
        """High guess probability makes positive observations less informative."""
        normal = create_initial_state(p_guess=0.2)
        high_guess = create_initial_state(p_guess=0.5)

        normal_updated = update_bkt(normal, positive_observation=True)
        high_guess_updated = update_bkt(high_guess, positive_observation=True)

        # High guess means positive obs could be a lucky guess even
        # when hypothesis is wrong, making positive evidence weaker
        assert high_guess_updated.p_knowledge < normal_updated.p_knowledge


class TestBKTWeightedUpdate:
    """Tests for weighted BKT updates (signal_strength modulation)."""

    def test_full_weight_equals_normal_update(self) -> None:
        """Weight=1.0 should produce the same result as normal update."""
        state = create_initial_state()
        normal = update_bkt(state, positive_observation=True)
        weighted = update_bkt_weighted(state, positive_observation=True, weight=1.0)
        assert abs(normal.p_knowledge - weighted.p_knowledge) < 1e-10

    def test_zero_weight_no_change(self) -> None:
        """Weight=0.0 should not change p_knowledge."""
        state = create_initial_state()
        weighted = update_bkt_weighted(state, positive_observation=True, weight=0.0)
        assert weighted.p_knowledge == state.p_knowledge
        assert weighted.observations == state.observations + 1

    def test_half_weight_intermediate(self) -> None:
        """Weight=0.5 should produce an intermediate result."""
        state = create_initial_state()
        full = update_bkt(state, positive_observation=True)
        half = update_bkt_weighted(state, positive_observation=True, weight=0.5)

        assert state.p_knowledge < half.p_knowledge < full.p_knowledge

    def test_weight_clamped(self) -> None:
        """Weight > 1.0 should be clamped to 1.0."""
        state = create_initial_state()
        normal = update_bkt(state, positive_observation=True)
        over = update_bkt_weighted(state, positive_observation=True, weight=2.0)
        assert abs(normal.p_knowledge - over.p_knowledge) < 1e-10


class TestBKTBatchUpdate:
    """Tests for batch_update_bkt convenience function."""

    def test_batch_equals_sequential(self) -> None:
        """Batch update should produce same result as sequential calls."""
        state = create_initial_state()
        observations = [True, True, False, True, False]

        # Sequential
        sequential = state
        for obs in observations:
            sequential = update_bkt(sequential, obs)

        # Batch
        batched = batch_update_bkt(state, observations)

        assert abs(sequential.p_knowledge - batched.p_knowledge) < 1e-10
        assert sequential.observations == batched.observations

    def test_empty_batch(self) -> None:
        """Empty observation list should return unchanged state."""
        state = create_initial_state()
        result = batch_update_bkt(state, [])
        assert result.p_knowledge == state.p_knowledge
        assert result.observations == 0
