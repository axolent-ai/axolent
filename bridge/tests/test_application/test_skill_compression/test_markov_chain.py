"""Tests for the Per-User Markov Chain (Step 2.2/10).

Covers:
  - Incremental update from events
  - Probability computation after observations
  - 100 observations A -> B gives probability > 0.8
  - Prediction ranking
  - Batch update
  - Serialization roundtrip (to_dict / from_dict)
  - Reset behavior
  - MarkovTransition immutability
  - HC-LAYER2-1: Markov is candidate signal, not truth
"""

from __future__ import annotations

import pytest

from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.markov_chain import (
    MarkovChain,
    MarkovTransition,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


def _make_event(intent: str, domain: str, ts_offset: int = 0) -> NormalizedEvent:
    """Create a minimal NormalizedEvent for testing."""
    ts = f"2026-05-20T{ts_offset:02d}:00:00+00:00"
    return NormalizedEvent(
        event_id=f"evt_{intent}_{domain}_{ts_offset}",
        user_id=42,
        timestamp=ts,
        intent=intent,
        domain=domain,
    )


# ---------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------


class TestMarkovUpdate:
    """Tests for incremental Markov chain updates."""

    def test_first_event_no_transition(self):
        """First event should not create a transition (no previous state)."""
        chain = MarkovChain()
        chain.update(_make_event("create_ad_copy", "marketing"))
        assert chain.total_transitions == 0

    def test_second_event_creates_transition(self):
        """Second event should create one transition."""
        chain = MarkovChain()
        chain.update(_make_event("create_ad_copy", "marketing"))
        chain.update(_make_event("analyze", "data"))
        assert chain.total_transitions == 1

    def test_transition_count_increments(self):
        """Repeated transitions should increment count."""
        chain = MarkovChain()
        for _ in range(5):
            chain.update(_make_event("create_ad_copy", "marketing"))
            chain.update(_make_event("analyze", "data"))
        assert chain.total_transitions == 9  # 5 A->B + 4 B->A


class TestMarkovProbability:
    """Tests for probability computation."""

    def test_100_observations_high_probability(self):
        """After 100 A -> B observations, P(B|A) should be > 0.8.

        This is the key spec test: a strongly observed transition
        should have high probability.
        """
        chain = MarkovChain()
        for i in range(100):
            chain.update(_make_event("create_ad_copy", "marketing", ts_offset=i * 2))
            chain.update(_make_event("analyze", "data", ts_offset=i * 2 + 1))

        prob = chain.get_transition_probability(
            "marketing.create_ad_copy",
            "data.analyze",
        )
        assert prob > 0.8, f"Expected probability > 0.8, got {prob}"

    def test_uniform_distribution(self):
        """Equal transitions should have equal probabilities."""
        chain = MarkovChain()
        # A -> B 10 times, A -> C 10 times
        for _ in range(10):
            chain.update(_make_event("start", "general"))
            chain.update(_make_event("create_code", "development"))
        for _ in range(10):
            chain.update(_make_event("start", "general"))
            chain.update(_make_event("create_ad_copy", "marketing"))

        prob_b = chain.get_transition_probability(
            "general.start", "development.create_code"
        )
        prob_c = chain.get_transition_probability(
            "general.start", "marketing.create_ad_copy"
        )
        assert abs(prob_b - prob_c) < 0.05  # Near equal

    def test_unknown_state_returns_zero(self):
        """Unknown state should return 0.0 probability."""
        chain = MarkovChain()
        chain.update(_make_event("create_ad_copy", "marketing"))
        chain.update(_make_event("analyze", "data"))

        prob = chain.get_transition_probability("unknown.state", "data.analyze")
        assert prob == 0.0

    def test_unknown_transition_returns_zero(self):
        """Known state but unknown transition should return 0.0."""
        chain = MarkovChain()
        chain.update(_make_event("create_ad_copy", "marketing"))
        chain.update(_make_event("analyze", "data"))

        prob = chain.get_transition_probability(
            "marketing.create_ad_copy", "unknown.state"
        )
        assert prob == 0.0


class TestMarkovPrediction:
    """Tests for next-state prediction."""

    def test_predict_returns_sorted(self):
        """Predictions should be sorted by probability descending."""
        chain = MarkovChain()
        # A -> B 8 times, A -> C 2 times
        for _ in range(8):
            chain.update(_make_event("start", "general"))
            chain.update(_make_event("create_code", "development"))
        for _ in range(2):
            chain.update(_make_event("start", "general"))
            chain.update(_make_event("create_ad_copy", "marketing"))

        predictions = chain.predict_next("general.start")
        assert len(predictions) == 2
        assert predictions[0].probability >= predictions[1].probability
        assert predictions[0].to_state == "development.create_code"

    def test_predict_top_k(self):
        """top_k should limit results."""
        chain = MarkovChain()
        for i in range(5):
            chain.update(_make_event("start", "general"))
            chain.update(_make_event(f"action_{i}", "domain"))

        predictions = chain.predict_next("general.start", top_k=2)
        assert len(predictions) <= 2

    def test_predict_unknown_state_empty(self):
        """Predicting from unknown state should return empty."""
        chain = MarkovChain()
        predictions = chain.predict_next("unknown.state")
        assert predictions == []

    def test_prediction_probabilities_sum_to_one(self):
        """All predictions from a state should sum to ~1.0."""
        chain = MarkovChain()
        for _ in range(10):
            chain.update(_make_event("start", "general"))
            chain.update(_make_event("create_code", "development"))
        for _ in range(5):
            chain.update(_make_event("start", "general"))
            chain.update(_make_event("analyze", "data"))

        predictions = chain.predict_next("general.start", top_k=100)
        total = sum(p.probability for p in predictions)
        assert abs(total - 1.0) < 0.01


class TestMarkovBatchUpdate:
    """Tests for batch update."""

    def test_batch_matches_sequential(self):
        """Batch update should produce same result as sequential updates."""
        events = [
            _make_event("create_ad_copy", "marketing", ts_offset=0),
            _make_event("analyze", "data", ts_offset=1),
            _make_event("plan", "business", ts_offset=2),
            _make_event("create_ad_copy", "marketing", ts_offset=3),
        ]

        chain_seq = MarkovChain()
        for e in events:
            chain_seq.update(e)

        chain_batch = MarkovChain()
        chain_batch.update_batch(events)

        assert chain_seq.total_transitions == chain_batch.total_transitions
        assert chain_seq.to_dict()["counts"] == chain_batch.to_dict()["counts"]


class TestMarkovSerialization:
    """Tests for serialization roundtrip."""

    def test_to_dict_and_from_dict(self):
        """Chain should survive serialization roundtrip."""
        chain = MarkovChain()
        chain.update(_make_event("create_ad_copy", "marketing"))
        chain.update(_make_event("analyze", "data"))
        chain.update(_make_event("plan", "business"))

        data = chain.to_dict()
        restored = MarkovChain.from_dict(data)

        assert restored.total_transitions == chain.total_transitions
        assert restored.to_dict()["counts"] == data["counts"]

        # Predictions should match
        pred_orig = chain.predict_next("marketing.create_ad_copy")
        pred_rest = restored.predict_next("marketing.create_ad_copy")
        assert len(pred_orig) == len(pred_rest)
        if pred_orig:
            assert pred_orig[0].to_state == pred_rest[0].to_state

    def test_from_empty_dict(self):
        """Empty dict should produce empty chain."""
        chain = MarkovChain.from_dict({})
        assert chain.total_transitions == 0


class TestMarkovReset:
    """Tests for chain reset."""

    def test_reset_clears_all(self):
        """Reset should clear all state."""
        chain = MarkovChain()
        chain.update(_make_event("create_ad_copy", "marketing"))
        chain.update(_make_event("analyze", "data"))
        assert chain.total_transitions > 0

        chain.reset()
        assert chain.total_transitions == 0
        assert chain.get_all_states() == set()


class TestMarkovGetAllStates:
    """Tests for state enumeration."""

    def test_returns_all_observed_states(self):
        """get_all_states should include both from and to states."""
        chain = MarkovChain()
        chain.update(_make_event("create_ad_copy", "marketing"))
        chain.update(_make_event("analyze", "data"))
        chain.update(_make_event("plan", "business"))

        states = chain.get_all_states()
        assert "marketing.create_ad_copy" in states
        assert "data.analyze" in states
        assert "business.plan" in states


class TestMarkovTransitionFrozen:
    """Guard: MarkovTransition must be immutable."""

    def test_transition_is_frozen(self):
        """MarkovTransition should be frozen (immutable)."""
        t = MarkovTransition(
            from_state="a",
            to_state="b",
            probability=0.5,
            observations=10,
        )
        with pytest.raises(AttributeError):
            t.probability = 0.9  # type: ignore[misc]

    def test_transition_has_slots(self):
        """MarkovTransition should use __slots__."""
        assert hasattr(MarkovTransition, "__slots__")
