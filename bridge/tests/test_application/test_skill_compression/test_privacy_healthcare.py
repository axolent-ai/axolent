"""Tests for HealthcareFilter (HC-SC-14, Step 8).

AG-SC-6 [GUARD]: test_no_phenotyping_inferences verifies healthcare
  filter blocks all health-related pattern materialization.

Covers:
  - Layer 1: Domain-based blocking
  - Layer 2: Healthcare keyword detection (DE + EN, 50+ keywords)
  - Layer 3: Behavioral-change pattern detection
  - Layer 4: Mood-inference pattern detection
  - Legitimate use cases NOT blocked (false-positive prevention)
"""

from __future__ import annotations

import pytest

from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.healthcare_filter import (
    ALL_HEALTHCARE_KEYWORDS,
    BLOCKED_HEALTH_DOMAINS,
    HealthcareFilter,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def hf() -> HealthcareFilter:
    return HealthcareFilter()


def _hyp(claim: str, *, context: tuple[str, ...] = ()) -> Hypothesis:
    """Create a test hypothesis with given claim and scope context."""
    return Hypothesis(
        hypothesis_id="hyp-hc-test",
        user_id=42,
        type="preference",
        scope=HypothesisScope(context=context),
        claim=claim,
        status="candidate",
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
    )


# ---------------------------------------------------------------
# Layer 1: Domain-based blocking
# ---------------------------------------------------------------


class TestHealthcareDomainLayer:
    """Tests for Layer 1: domain-based health blocking."""

    def test_health_domain_blocked(self, hf: HealthcareFilter) -> None:
        """Health domain in scope context should block."""
        h = _hyp("User prefers bullet points", context=("health",))
        assert hf.filter_hypothesis(h) is True

    def test_medical_domain_blocked(self, hf: HealthcareFilter) -> None:
        """Medical domain in scope context should block."""
        h = _hyp("User likes tables", context=("medical",))
        assert hf.filter_hypothesis(h) is True

    def test_psychology_domain_blocked(self, hf: HealthcareFilter) -> None:
        """Psychology domain should block."""
        h = _hyp("User prefers short answers", context=("psychology",))
        assert hf.filter_hypothesis(h) is True

    def test_mental_health_domain_blocked(self, hf: HealthcareFilter) -> None:
        """Mental health domain should block."""
        h = _hyp("User asks about self-care", context=("mental_health",))
        assert hf.filter_hypothesis(h) is True

    def test_event_health_domain(self, hf: HealthcareFilter) -> None:
        """Event with health domain should be detected."""
        event = NormalizedEvent(domain="health")
        assert hf.is_health_related_event(event) is True

    def test_event_normal_domain(self, hf: HealthcareFilter) -> None:
        """Event with normal domain should not be flagged."""
        event = NormalizedEvent(domain="marketing")
        assert hf.is_health_related_event(event) is False

    def test_non_health_domain_passes(self, hf: HealthcareFilter) -> None:
        """Non-health domain in context should pass."""
        h = _hyp("User prefers markdown", context=("marketing",))
        assert hf.filter_hypothesis(h) is False


# ---------------------------------------------------------------
# Layer 2: Healthcare keyword detection
# ---------------------------------------------------------------


class TestHealthcareKeywordLayer:
    """Tests for Layer 2: healthcare keyword scan."""

    def test_depression_keyword_en(self, hf: HealthcareFilter) -> None:
        """English depression keyword should block."""
        h = _hyp("User shows signs of depression in writing")
        assert hf.filter_hypothesis(h) is True

    def test_depression_keyword_de(self, hf: HealthcareFilter) -> None:
        """German depression keyword should block."""
        h = _hyp("User ist depressiv basierend auf Mustern")
        assert hf.filter_hypothesis(h) is True

    def test_anxiety_keyword(self, hf: HealthcareFilter) -> None:
        """Anxiety keyword should block."""
        h = _hyp("Pattern suggests anxiety in user behavior")
        assert hf.filter_hypothesis(h) is True

    def test_adhd_keyword(self, hf: HealthcareFilter) -> None:
        """ADHD keyword should block."""
        h = _hyp("User might have ADHD based on context switching")
        assert hf.filter_hypothesis(h) is True

    def test_bipolar_keyword(self, hf: HealthcareFilter) -> None:
        """Bipolar keyword should block."""
        h = _hyp("Patterns consistent with bipolar mood swings")
        assert hf.filter_hypothesis(h) is True

    def test_burnout_keyword(self, hf: HealthcareFilter) -> None:
        """Burnout keyword should block."""
        h = _hyp("User shows burnout indicators")
        assert hf.filter_hypothesis(h) is True

    def test_dementia_keyword(self, hf: HealthcareFilter) -> None:
        """Dementia keyword should block."""
        h = _hyp("Pattern suggests early dementia")
        assert hf.filter_hypothesis(h) is True

    def test_german_angststoerung(self, hf: HealthcareFilter) -> None:
        """German clinical term should block."""
        h = _hyp("Muster deutet auf Angststörung hin")
        assert hf.filter_hypothesis(h) is True

    def test_german_schlafstörung(self, hf: HealthcareFilter) -> None:
        """German sleep disorder term should block."""
        h = _hyp("User hat Schlafstörung basierend auf Aktivität")
        assert hf.filter_hypothesis(h) is True

    def test_medication_keyword(self, hf: HealthcareFilter) -> None:
        """Medication keyword should block."""
        h = _hyp("User mentions medication schedule patterns")
        assert hf.filter_hypothesis(h) is True

    def test_keyword_count_minimum(self) -> None:
        """At least 50 healthcare keywords should be defined."""
        assert len(ALL_HEALTHCARE_KEYWORDS) >= 50

    def test_all_blocked_domains_present(self) -> None:
        """At least 10 health domains should be defined."""
        assert len(BLOCKED_HEALTH_DOMAINS) >= 10


# ---------------------------------------------------------------
# Layer 3: Behavioral-change pattern detection
# ---------------------------------------------------------------


class TestHealthcareBehavioralLayer:
    """Tests for Layer 3: behavioral-clinical phenotyping patterns."""

    def test_writing_style_change(self, hf: HealthcareFilter) -> None:
        """Pattern about writing style changes over time should block."""
        h = _hyp("User's writing style has changed over the past weeks")
        assert hf.filter_hypothesis(h) is True

    def test_typing_pattern_change(self, hf: HealthcareFilter) -> None:
        """Pattern about typing pattern changes should block."""
        h = _hyp("User's typing speed has declined over time")
        assert hf.filter_hypothesis(h) is True

    def test_language_change_temporal(self, hf: HealthcareFilter) -> None:
        """Pattern about language changes over time should block."""
        h = _hyp("Language complexity has shifted over the past months")
        assert hf.filter_hypothesis(h) is True

    def test_shows_signs_cognitive(self, hf: HealthcareFilter) -> None:
        """'Shows signs of cognitive' patterns should block."""
        h = _hyp("User shows signs of cognitive decline")
        assert hf.filter_hypothesis(h) is True

    def test_correlates_with_health(self, hf: HealthcareFilter) -> None:
        """Correlation with health claims should block."""
        h = _hyp("Pattern correlates with mental health indicators")
        assert hf.filter_hypothesis(h) is True

    def test_german_zunehmend_pattern(self, hf: HealthcareFilter) -> None:
        """German temporal pattern should block."""
        h = _hyp("Schreibstil hat sich in letzter Zeit verschlechtert")
        assert hf.filter_hypothesis(h) is True


# ---------------------------------------------------------------
# Layer 4: Mood-inference pattern detection
# ---------------------------------------------------------------


class TestHealthcareMoodLayer:
    """Tests for Layer 4: mood-inference pattern detection."""

    def test_user_seems_sad(self, hf: HealthcareFilter) -> None:
        """'User seems sad' patterns should block."""
        h = _hyp("User seems sad based on recent messages")
        assert hf.filter_hypothesis(h) is True

    def test_user_appears_stressed(self, hf: HealthcareFilter) -> None:
        """'User appears stressed' should block."""
        h = _hyp("User appears stressed in conversations")
        assert hf.filter_hypothesis(h) is True

    def test_mood_tracking(self, hf: HealthcareFilter) -> None:
        """Mood tracking patterns should block."""
        h = _hyp("Mood detection and tracking over sessions")
        assert hf.filter_hypothesis(h) is True

    def test_user_sentiment(self, hf: HealthcareFilter) -> None:
        """User sentiment inference should block."""
        h = _hyp("User's sentiment has shifted negative")
        assert hf.filter_hypothesis(h) is True

    def test_daily_mood_pattern(self, hf: HealthcareFilter) -> None:
        """Daily mood pattern should block."""
        h = _hyp("Daily mood pattern shows evening dips")
        assert hf.filter_hypothesis(h) is True

    def test_german_user_wirkt_müde(self, hf: HealthcareFilter) -> None:
        """German mood inference should block."""
        h = _hyp("User wirkt müde in der Kommunikation")
        assert hf.filter_hypothesis(h) is True

    def test_german_stimmung_erkennen(self, hf: HealthcareFilter) -> None:
        """German mood detection pattern should block."""
        h = _hyp("Stimmung erkennen und verfolgen")
        assert hf.filter_hypothesis(h) is True


# ---------------------------------------------------------------
# False-positive prevention (legitimate use cases)
# ---------------------------------------------------------------


class TestHealthcareLegitUseCases:
    """Tests that legitimate skills are NOT blocked."""

    def test_recipe_request_passes(self, hf: HealthcareFilter) -> None:
        """'Write me a recipe' should NOT be blocked."""
        h = _hyp("User prefers short recipe instructions")
        assert hf.filter_hypothesis(h) is False

    def test_code_review_passes(self, hf: HealthcareFilter) -> None:
        """Code review preferences should pass."""
        h = _hyp("User prefers root-cause-first code reviews")
        assert hf.filter_hypothesis(h) is False

    def test_markdown_preference_passes(self, hf: HealthcareFilter) -> None:
        """Format preference should pass."""
        h = _hyp("User prefers Markdown tables over plain text")
        assert hf.filter_hypothesis(h) is False

    def test_writing_style_preference_passes(self, hf: HealthcareFilter) -> None:
        """Writing style preference (non-clinical) should pass."""
        h = _hyp("User prefers formal tone in business emails")
        assert hf.filter_hypothesis(h) is False

    def test_drehkonzept_passes(self, hf: HealthcareFilter) -> None:
        """Video concept preference should pass."""
        h = _hyp("User prefers 30s retargeting video concepts")
        assert hf.filter_hypothesis(h) is False

    def test_scheduling_passes(self, hf: HealthcareFilter) -> None:
        """Scheduling preference should pass."""
        h = _hyp("User plans weekly on Monday mornings in Markdown")
        assert hf.filter_hypothesis(h) is False

    def test_technical_preference_passes(self, hf: HealthcareFilter) -> None:
        """Technical preference should pass."""
        h = _hyp("User prefers Python type hints everywhere")
        assert hf.filter_hypothesis(h) is False

    def test_block_reason_returns_string(self, hf: HealthcareFilter) -> None:
        """get_block_reason should return reason for blocked items."""
        h = _hyp("User shows signs of depression")
        reason = hf.get_block_reason(h)
        assert reason is not None
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_block_reason_returns_none_for_clean(self, hf: HealthcareFilter) -> None:
        """get_block_reason should return None for clean items."""
        h = _hyp("User prefers bullet points")
        reason = hf.get_block_reason(h)
        assert reason is None
