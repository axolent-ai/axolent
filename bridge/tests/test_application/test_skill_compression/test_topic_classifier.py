"""Tests for TopicClassifier (Step 6).

Covers:
  - HC-BERT-1: Library not installed -> graceful skip
  - HC-LAYER2-1: Classifier is suggestion provider, not truth source
  - classify_batch with events
  - suggest_pattern_candidates
  - get_topic_info
  - reset state
  - Too few events for classification
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.topic_classifier import (
    INSTALL_HINT,
    CandidatePattern,
    TopicAssignment,
    TopicClassifier,
    TopicModelState,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_event(
    event_id: str = "evt_001",
    raw_text: str = "Create a marketing campaign for the new product",
    intent: str = "create_text",
    domain: str = "marketing",
) -> NormalizedEvent:
    """Create a test NormalizedEvent."""
    return NormalizedEvent(
        event_id=event_id,
        user_id=1,
        timestamp="2026-05-20T12:00:00Z",
        raw_text=raw_text,
        intent=intent,
        domain=domain,
        format_type="plain_text",
        language="en",
    )


def _make_events(n: int) -> list[NormalizedEvent]:
    """Create n test events with varied text."""
    texts = [
        "Create a marketing campaign for the new product",
        "Write an ad copy for the summer sale",
        "Design a landing page for the new launch",
        "Analyze the conversion data from last month",
        "Build a Python script for data processing",
        "Review the code changes in the pull request",
        "Set up a CI/CD pipeline for deployment",
        "Write a blog post about AI trends",
        "Create a financial report for Q3",
        "Translate the documentation to German",
    ]
    events = []
    for i in range(n):
        text = texts[i % len(texts)]
        events.append(
            _make_event(
                event_id=f"evt_{i:03d}",
                raw_text=f"{text} (variant {i})",
            )
        )
    return events


# ---------------------------------------------------------------
# Tests: Graceful skip when bertopic not installed (HC-BERT-1)
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestTopicClassifierGracefulSkip:
    """HC-BERT-1: All operations skip gracefully when bertopic missing."""

    def test_is_available_false_when_not_installed(self) -> None:
        """is_available() returns False when bertopic is not importable."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = None  # force re-check
            with patch.dict("sys.modules", {"bertopic": None}):
                tc_mod._BERTOPIC_AVAILABLE = None
                classifier = TopicClassifier()
                # Manually set to False to simulate import failure
                tc_mod._BERTOPIC_AVAILABLE = False
                assert classifier.is_available() is False
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original

    def test_classify_batch_empty_when_unavailable(self) -> None:
        """classify_batch returns empty list when bertopic unavailable."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = False
            classifier = TopicClassifier()
            events = _make_events(5)

            result = classifier.classify_batch(events)
            assert result == []
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original

    def test_suggest_candidates_empty_when_unavailable(self) -> None:
        """suggest_pattern_candidates returns empty when unavailable."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = False
            classifier = TopicClassifier()

            result = classifier.suggest_pattern_candidates(0, _make_events(3))
            assert result == []
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original

    def test_get_topic_info_empty_when_unavailable(self) -> None:
        """get_topic_info returns empty when unavailable."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = False
            classifier = TopicClassifier()

            result = classifier.get_topic_info()
            assert result == []
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original


# ---------------------------------------------------------------
# Tests: Empty/minimal input handling
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestTopicClassifierMinimalInput:
    """Edge cases for minimal input."""

    def test_classify_empty_events(self) -> None:
        """Empty events list returns empty."""
        classifier = TopicClassifier()
        result = classifier.classify_batch([])
        assert result == []

    def test_classify_too_few_events(self) -> None:
        """Too few events for clustering returns empty."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = True
            classifier = TopicClassifier(min_topic_size=5)
            events = _make_events(3)  # fewer than min_topic_size

            result = classifier.classify_batch(events)
            assert result == []
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original

    def test_suggest_no_fit(self) -> None:
        """Suggest candidates with unfitted model returns empty."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = True
            classifier = TopicClassifier()
            assert classifier._state.fitted is False

            result = classifier.suggest_pattern_candidates(0, _make_events(3))
            assert result == []
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original

    def test_suggest_outlier_topic_empty(self) -> None:
        """Suggest candidates for outlier topic (-1) returns empty."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = True
            classifier = TopicClassifier()
            classifier._state.fitted = True
            classifier._state.model = object()

            result = classifier.suggest_pattern_candidates(-1, _make_events(3))
            assert result == []
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original

    def test_suggest_empty_events(self) -> None:
        """Suggest candidates with empty events returns empty."""
        import application.skill_compression.topic_classifier as tc_mod

        original = tc_mod._BERTOPIC_AVAILABLE
        try:
            tc_mod._BERTOPIC_AVAILABLE = True
            classifier = TopicClassifier()
            classifier._state.fitted = True
            classifier._state.model = object()

            result = classifier.suggest_pattern_candidates(0, [])
            assert result == []
        finally:
            tc_mod._BERTOPIC_AVAILABLE = original


# ---------------------------------------------------------------
# Tests: Data structure integrity
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestTopicClassifierDataStructures:
    """Data structures are correctly defined."""

    def test_topic_assignment_frozen(self) -> None:
        """TopicAssignment is frozen."""
        ta = TopicAssignment(
            event_id="evt_001",
            topic_id=1,
            topic_label="Test Topic",
            confidence=0.85,
            keywords=("keyword1", "keyword2"),
        )
        with pytest.raises(AttributeError):
            ta.topic_id = 2  # type: ignore[misc]

    def test_candidate_pattern_frozen(self) -> None:
        """CandidatePattern is frozen."""
        cp = CandidatePattern(
            topic_id=1,
            suggested_claim="Users request marketing campaigns",
            supporting_event_ids=("evt_001", "evt_002"),
            topic_keywords=("marketing", "campaign"),
            event_count=2,
            suggestion_confidence=0.7,
        )
        with pytest.raises(AttributeError):
            cp.topic_id = 2  # type: ignore[misc]

    def test_install_hint_contains_pip(self) -> None:
        """Install hint mentions pip install."""
        assert "pip install" in INSTALL_HINT

    def test_reset_clears_state(self) -> None:
        """reset() clears the model state."""
        classifier = TopicClassifier()
        classifier._state.fitted = True
        classifier._state.topic_count = 5

        classifier.reset()

        assert classifier._state.fitted is False
        assert classifier._state.topic_count == 0
        assert classifier._state.model is None

    def test_model_state_defaults(self) -> None:
        """TopicModelState has correct defaults."""
        state = TopicModelState()
        assert state.model is None
        assert state.last_fit_event_count == 0
        assert state.topic_count == 0
        assert state.fitted is False
