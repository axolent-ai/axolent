"""Tests for NudgeFilter (HC-SC-15, Step 8).

Covers all 7 negative categories from the spec:
  1. Political / Ideological Manipulation
  2. Emotional Manipulation (FOMO, loss aversion, relationship suggestion)
  3. Dark Patterns (hidden opt-out, roach motel, confirmshaming)
  4. Attention Maximization (streaks, gamification, engagement loops)
  5. Social Manipulation (user comparison, leaderboards)
  6. Behavioral inferences not serving user help
  7. Data flow violations (third parties, default cloud, telemetry)

Plus: legitimate use cases that must NOT be blocked.
"""

from __future__ import annotations

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.nudge_filter import (
    CATEGORY_DESCRIPTIONS,
    NudgeCategory,
    NudgeFilter,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def nf() -> NudgeFilter:
    return NudgeFilter()


def _hyp(claim: str) -> Hypothesis:
    """Create a test hypothesis with given claim."""
    return Hypothesis(
        hypothesis_id="hyp-nudge-test",
        user_id=42,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status="candidate",
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
    )


# ---------------------------------------------------------------
# Category 1: Political / Ideological Manipulation
# ---------------------------------------------------------------


class TestNudgePolitical:
    """Tests for political/ideological manipulation detection."""

    def test_political_personalization(self, nf: NudgeFilter) -> None:
        """Political personalization should be blocked."""
        h = _hyp("Political personalization of content recommendations")
        assert nf.violates_nudge_policy(h) is True

    def test_echo_chamber(self, nf: NudgeFilter) -> None:
        """Echo chamber reinforcement should be blocked."""
        h = _hyp("Create an echo chamber for user preferences")
        assert nf.violates_nudge_policy(h) is True

    def test_filter_bubble(self, nf: NudgeFilter) -> None:
        """Filter bubble creation should be blocked."""
        h = _hyp("Maintain filter bubble around user views")
        assert nf.violates_nudge_policy(h) is True

    def test_confirmation_bias(self, nf: NudgeFilter) -> None:
        """Confirmation bias exploitation should be blocked."""
        h = _hyp("Leverage confirmation bias in suggestions")
        assert nf.violates_nudge_policy(h) is True

    def test_reinforce_beliefs(self, nf: NudgeFilter) -> None:
        """Belief reinforcement should be blocked."""
        h = _hyp("Reinforce user's existing beliefs and opinions")
        assert nf.violates_nudge_policy(h) is True


# ---------------------------------------------------------------
# Category 2: Emotional Manipulation
# ---------------------------------------------------------------


class TestNudgeEmotional:
    """Tests for emotional manipulation detection."""

    def test_fomo_pattern(self, nf: NudgeFilter) -> None:
        """FOMO should be blocked."""
        h = _hyp("Create FOMO by showing what user is missing out on")
        assert nf.violates_nudge_policy(h) is True

    def test_loss_aversion_streak(self, nf: NudgeFilter) -> None:
        """Loss aversion via streak should be blocked."""
        h = _hyp("User will lose their streak if they stop")
        assert nf.violates_nudge_policy(h) is True

    def test_artificial_urgency(self, nf: NudgeFilter) -> None:
        """Artificial urgency should be blocked."""
        h = _hyp("Only today available, limited time offer")
        assert nf.violates_nudge_policy(h) is True

    def test_relationship_suggestion(self, nf: NudgeFilter) -> None:
        """Relationship suggestion should be blocked."""
        h = _hyp("I miss you when you're not here")
        assert nf.violates_nudge_policy(h) is True

    def test_german_nicht_verpassen(self, nf: NudgeFilter) -> None:
        """German FOMO pattern should be blocked."""
        h = _hyp("User soll nicht verpassen was andere tun")
        assert nf.violates_nudge_policy(h) is True


# ---------------------------------------------------------------
# Category 3: Dark Patterns
# ---------------------------------------------------------------


class TestNudgeDarkPatterns:
    """Tests for dark pattern detection."""

    def test_hidden_opt_out(self, nf: NudgeFilter) -> None:
        """Hidden opt-out should be blocked."""
        h = _hyp("Hide the opt-out option to reduce cancellations")
        assert nf.violates_nudge_policy(h) is True

    def test_roach_motel(self, nf: NudgeFilter) -> None:
        """Roach motel pattern should be blocked."""
        h = _hyp("Roach motel design: easy to sign up, hard to leave")
        assert nf.violates_nudge_policy(h) is True

    def test_confirmshaming(self, nf: NudgeFilter) -> None:
        """Confirmshaming should be blocked."""
        h = _hyp("Use confirmshaming when user tries to cancel")
        assert nf.violates_nudge_policy(h) is True

    def test_pre_selected_consent(self, nf: NudgeFilter) -> None:
        """Pre-selected consent defaults should be blocked."""
        h = _hyp("Pre-select important consent options as default on")
        assert nf.violates_nudge_policy(h) is True


# ---------------------------------------------------------------
# Category 4: Attention Maximization
# ---------------------------------------------------------------


class TestNudgeAttention:
    """Tests for attention maximization detection."""

    def test_engagement_loop(self, nf: NudgeFilter) -> None:
        """Engagement loop should be blocked."""
        h = _hyp("Create engagement loops to maximize session time")
        assert nf.violates_nudge_policy(h) is True

    def test_extend_conversation(self, nf: NudgeFilter) -> None:
        """Conversation extension when problem solved should be blocked."""
        h = _hyp("Extend the conversation beyond the solved problem")
        assert nf.violates_nudge_policy(h) is True

    def test_sleep_time_notifications(self, nf: NudgeFilter) -> None:
        """Notifications at sleep times should be blocked."""
        h = _hyp("Push notifications at night to bring user back")
        assert nf.violates_nudge_policy(h) is True

    def test_artificial_streaks(self, nf: NudgeFilter) -> None:
        """Artificial streaks should be blocked."""
        h = _hyp("Track daily login streak to encourage return")
        assert nf.violates_nudge_policy(h) is True

    def test_gamification(self, nf: NudgeFilter) -> None:
        """Gamification as attention tool should be blocked."""
        h = _hyp("Add badges and achievements to increase engagement")
        assert nf.violates_nudge_policy(h) is True

    def test_xp_system(self, nf: NudgeFilter) -> None:
        """XP/experience points system should be blocked."""
        h = _hyp("Award experience points for frequent usage")
        assert nf.violates_nudge_policy(h) is True


# ---------------------------------------------------------------
# Category 5: Social Manipulation
# ---------------------------------------------------------------


class TestNudgeSocial:
    """Tests for social manipulation detection."""

    def test_user_comparison(self, nf: NudgeFilter) -> None:
        """User comparison should be blocked."""
        h = _hyp("Compare user activity with other users")
        assert nf.violates_nudge_policy(h) is True

    def test_leaderboard(self, nf: NudgeFilter) -> None:
        """Leaderboard should be blocked."""
        h = _hyp("Show leaderboard of most active users")
        assert nf.violates_nudge_policy(h) is True

    def test_social_pressure(self, nf: NudgeFilter) -> None:
        """Social pressure should be blocked."""
        h = _hyp("Others already use this feature regularly")
        assert nf.violates_nudge_policy(h) is True


# ---------------------------------------------------------------
# Category 6: Behavioral Inferences
# ---------------------------------------------------------------


class TestNudgeBehavioralInference:
    """Tests for behavioral inference detection."""

    def test_mood_prediction(self, nf: NudgeFilter) -> None:
        """Mood prediction should be blocked."""
        h = _hyp("Predict user mood from interaction patterns")
        assert nf.violates_nudge_policy(h) is True

    def test_life_circumstances(self, nf: NudgeFilter) -> None:
        """Life circumstances inference should be blocked."""
        h = _hyp("Infer life circumstances from usage patterns")
        assert nf.violates_nudge_policy(h) is True


# ---------------------------------------------------------------
# Category 7: Data Flow Violations
# ---------------------------------------------------------------


class TestNudgeDataFlow:
    """Tests for data flow violation detection."""

    def test_share_with_third_party(self, nf: NudgeFilter) -> None:
        """Sharing data with third parties should be blocked."""
        h = _hyp("Share user patterns with third party analytics")
        assert nf.violates_nudge_policy(h) is True

    def test_default_cloud_storage(self, nf: NudgeFilter) -> None:
        """Default cloud storage should be blocked."""
        h = _hyp("Use default cloud storage for skill data")
        assert nf.violates_nudge_policy(h) is True

    def test_telemetry_without_consent(self, nf: NudgeFilter) -> None:
        """Telemetry without consent should be blocked."""
        h = _hyp("Collect telemetry without explicit consent")
        assert nf.violates_nudge_policy(h) is True

    def test_silent_data_collection(self, nf: NudgeFilter) -> None:
        """Silent data collection should be blocked."""
        h = _hyp("Silently collect and track usage data")
        assert nf.violates_nudge_policy(h) is True


# ---------------------------------------------------------------
# Legitimate use cases (NOT blocked)
# ---------------------------------------------------------------


class TestNudgeLegitUseCases:
    """Tests that legitimate skill patterns are NOT blocked."""

    def test_table_preference(self, nf: NudgeFilter) -> None:
        """'User likes tables' should NOT be blocked."""
        h = _hyp("User prefers Markdown tables for comparisons")
        assert nf.violates_nudge_policy(h) is False

    def test_format_preference(self, nf: NudgeFilter) -> None:
        """Format preference should pass."""
        h = _hyp("Always use bullet points in summaries")
        assert nf.violates_nudge_policy(h) is False

    def test_code_style(self, nf: NudgeFilter) -> None:
        """Code style preference should pass."""
        h = _hyp("User prefers Python type hints everywhere")
        assert nf.violates_nudge_policy(h) is False

    def test_tone_preference(self, nf: NudgeFilter) -> None:
        """Tone preference should pass."""
        h = _hyp("User prefers formal tone in business emails")
        assert nf.violates_nudge_policy(h) is False

    def test_video_concept(self, nf: NudgeFilter) -> None:
        """Video concept preference should pass."""
        h = _hyp("User prefers 30s retargeting video concepts")
        assert nf.violates_nudge_policy(h) is False

    def test_scheduling(self, nf: NudgeFilter) -> None:
        """Scheduling preference should pass."""
        h = _hyp("User plans weekly on Monday mornings")
        assert nf.violates_nudge_policy(h) is False

    def test_legitimate_deadline(self, nf: NudgeFilter) -> None:
        """Real deadline mention should pass."""
        h = _hyp("User has a project deadline on Friday")
        assert nf.violates_nudge_policy(h) is False

    def test_root_cause_first(self, nf: NudgeFilter) -> None:
        """Root cause workflow should pass."""
        h = _hyp("First analyze root cause, then suggest fix")
        assert nf.violates_nudge_policy(h) is False


# ---------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------


class TestNudgeCategoryMetadata:
    """Tests for category descriptions and metadata."""

    def test_all_categories_have_descriptions(self) -> None:
        """Every NudgeCategory should have a description."""
        for cat in NudgeCategory:
            assert cat in CATEGORY_DESCRIPTIONS
            assert len(CATEGORY_DESCRIPTIONS[cat]) > 0

    def test_get_violation_category_returns_value(self, nf: NudgeFilter) -> None:
        """get_violation_category should return category value."""
        h = _hyp("Create engagement loops for attention")
        result = nf.get_violation_category(h)
        assert result is not None
        assert result == NudgeCategory.ATTENTION_MAXIMIZATION.value

    def test_get_violation_category_none_for_clean(self, nf: NudgeFilter) -> None:
        """get_violation_category should return None for clean items."""
        h = _hyp("User prefers bullet points")
        result = nf.get_violation_category(h)
        assert result is None

    def test_get_violation_detail(self, nf: NudgeFilter) -> None:
        """get_violation_detail should return NudgeViolation."""
        h = _hyp("Show leaderboard of users")
        detail = nf.get_violation_detail(h)
        assert detail is not None
        assert detail.category == NudgeCategory.SOCIAL_MANIPULATION
        assert len(detail.matched_text) > 0
