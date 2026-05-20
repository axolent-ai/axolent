"""Tests for the Fingerprint Similarity Matcher (Layer 2 Foundation, Step 1.4/10).

Covers:
  - Identical events produce similarity 1.0
  - Completely different events produce < 0.5
  - Similar events produce > 0.7 (candidacy threshold)
  - Language mismatch forces 0.0
  - Field weights are respected
  - find_matches returns sorted results
  - Custom thresholds work
"""

from __future__ import annotations

import pytest

from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.fingerprint_matcher import (
    FingerprintMatch,
    compute_similarity,
    find_matches,
)


def _make_event(
    event_id: str = "evt_test",
    intent: str = "create_code",
    domain: str = "development",
    format_type: str = "code",
    constraints: dict | None = None,
    scope: dict | None = None,
    language: str = "en",
) -> NormalizedEvent:
    """Helper to create NormalizedEvent with defaults."""
    return NormalizedEvent(
        event_id=event_id,
        user_id=1,
        timestamp="2026-05-20T12:00:00+00:00",
        raw_text="test",
        intent=intent,
        domain=domain,
        format_type=format_type,
        constraints=constraints or {},
        scope=scope or {},
        language=language,
    )


class TestIdenticalEvents:
    """Tests for identical event comparison."""

    def test_identical_events_score_1(self):
        """Two identical events must have similarity 1.0."""
        a = _make_event(event_id="a")
        b = _make_event(event_id="b")
        match = compute_similarity(a, b)
        assert match.similarity_score == pytest.approx(1.0)
        assert match.is_candidate is True

    def test_identical_with_constraints(self):
        """Identical events with constraints score 1.0."""
        a = _make_event(
            event_id="a",
            constraints={"duration": "30s", "funnel": "retargeting"},
        )
        b = _make_event(
            event_id="b",
            constraints={"duration": "30s", "funnel": "retargeting"},
        )
        match = compute_similarity(a, b)
        assert match.similarity_score == pytest.approx(1.0)

    def test_identical_with_scope(self):
        """Identical events with scope score 1.0."""
        a = _make_event(
            event_id="a",
            scope={"project": "axolent", "client": "honey"},
        )
        b = _make_event(
            event_id="b",
            scope={"project": "axolent", "client": "honey"},
        )
        match = compute_similarity(a, b)
        assert match.similarity_score == pytest.approx(1.0)


class TestDifferentEvents:
    """Tests for completely different events."""

    def test_completely_different_score_below_05(self):
        """Completely different events must score below 0.5."""
        a = _make_event(
            event_id="a",
            intent="create_code",
            domain="development",
            format_type="code",
        )
        b = _make_event(
            event_id="b",
            intent="analyze",
            domain="marketing",
            format_type="report",
        )
        match = compute_similarity(a, b)
        assert match.similarity_score < 0.5
        assert match.is_candidate is False

    def test_different_constraints(self):
        """Different constraints reduce similarity."""
        a = _make_event(
            event_id="a",
            constraints={"duration": "30s", "funnel": "retargeting"},
        )
        b = _make_event(
            event_id="b",
            constraints={"length": "500 words", "audience": "b2b"},
        )
        match = compute_similarity(a, b)
        # Intent, domain, format same but constraints differ
        assert match.field_similarities["constraints"] < 0.5


class TestLanguageFilter:
    """Tests for language as hard filter."""

    def test_language_mismatch_zero(self):
        """Different languages must produce similarity 0.0."""
        a = _make_event(event_id="a", language="de")
        b = _make_event(event_id="b", language="en")
        match = compute_similarity(a, b)
        assert match.similarity_score == 0.0
        assert match.is_candidate is False

    def test_same_language_passes(self):
        """Same language should not reduce similarity."""
        a = _make_event(event_id="a", language="de")
        b = _make_event(event_id="b", language="de")
        match = compute_similarity(a, b)
        assert match.similarity_score == pytest.approx(1.0)


class TestSimilarEvents:
    """Tests for partially similar events (candidacy threshold)."""

    def test_same_intent_different_format(self):
        """Same intent+domain but different format should be moderate similarity."""
        a = _make_event(event_id="a", intent="create_code", format_type="code")
        b = _make_event(event_id="b", intent="create_code", format_type="markdown")
        match = compute_similarity(a, b)
        # Intent (1.0 * 0.3) + domain (1.0 * 0.2) + constraints (1.0 * 0.2)
        # + format (0.0 * 0.15) + scope (1.0 * 0.15) = 0.85
        assert match.similarity_score == pytest.approx(0.85)
        assert match.is_candidate is True

    def test_related_intents_via_prefix(self):
        """Related intents (shared prefix) should have partial similarity."""
        a = _make_event(event_id="a", intent="create_code")
        b = _make_event(event_id="b", intent="create_text")
        match = compute_similarity(a, b)
        # Intent prefix match: create matches = 0.5
        assert match.field_similarities["intent"] == pytest.approx(0.5)

    def test_threshold_boundary(self):
        """Events at exactly the threshold boundary."""
        # With default weights, all same except format = 0.85 (above threshold)
        a = _make_event(event_id="a", format_type="code")
        b = _make_event(event_id="b", format_type="table")
        match = compute_similarity(a, b, threshold=0.85)
        # Exactly at threshold: is_candidate uses > (not >=)
        assert match.similarity_score == pytest.approx(0.85)
        assert match.is_candidate is False  # > not >=


class TestFieldWeights:
    """Tests that field weights are respected."""

    def test_custom_weights(self):
        """Custom weights should change the similarity score."""
        a = _make_event(event_id="a", intent="create_code")
        b = _make_event(event_id="b", intent="analyze")

        # Default weights: intent at 30%
        default_match = compute_similarity(a, b)

        # Custom: intent at 90%
        heavy_intent = compute_similarity(
            a,
            b,
            weights={
                "intent": 0.90,
                "domain": 0.025,
                "constraints": 0.025,
                "format_type": 0.025,
                "scope": 0.025,
            },
        )
        # With higher intent weight, the mismatch hurts more
        assert heavy_intent.similarity_score < default_match.similarity_score


class TestFindMatches:
    """Tests for the batch matching function."""

    def test_find_matches_sorted(self):
        """Results should be sorted by similarity descending."""
        target = _make_event(event_id="target", intent="create_code")
        candidates = [
            _make_event(event_id="c1", intent="create_code"),  # Exact
            _make_event(event_id="c2", intent="create_text"),  # Partial
            _make_event(event_id="c3", intent="analyze"),  # Different
        ]
        matches = find_matches(target, candidates, threshold=0.0)
        assert len(matches) > 0
        # First match should be the most similar
        assert matches[0].event_b_id == "c1"

    def test_find_matches_excludes_self(self):
        """Self-matches should be excluded."""
        target = _make_event(event_id="target")
        candidates = [
            _make_event(event_id="target"),  # Same ID
            _make_event(event_id="other"),
        ]
        matches = find_matches(target, candidates)
        assert all(m.event_b_id != "target" for m in matches)

    def test_find_matches_respects_threshold(self):
        """Only matches above threshold should be included."""
        target = _make_event(event_id="target", intent="create_code")
        candidates = [
            _make_event(
                event_id="c1",
                intent="analyze",
                domain="marketing",
                format_type="report",
            ),
        ]
        matches = find_matches(target, candidates, threshold=0.9)
        assert len(matches) == 0

    def test_find_matches_max_results(self):
        """max_results should limit the number of results."""
        target = _make_event(event_id="target")
        candidates = [_make_event(event_id=f"c{i}") for i in range(20)]
        matches = find_matches(target, candidates, threshold=0.0, max_results=5)
        assert len(matches) <= 5


class TestFingerprintMatchStructure:
    """Tests for the FingerprintMatch data structure."""

    def test_match_is_frozen(self):
        """FingerprintMatch should be immutable."""
        match = FingerprintMatch()
        with pytest.raises(AttributeError):
            match.similarity_score = 0.5  # type: ignore[misc]

    def test_match_has_field_similarities(self):
        """Match result should include per-field similarity breakdown."""
        a = _make_event(event_id="a")
        b = _make_event(event_id="b")
        match = compute_similarity(a, b)
        assert "intent" in match.field_similarities
        assert "domain" in match.field_similarities
        assert "format_type" in match.field_similarities
        assert "constraints" in match.field_similarities
        assert "scope" in match.field_similarities
