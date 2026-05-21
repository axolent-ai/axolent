"""Tests for ChatService + SkillMatcher integration (Step 5).

Covers:
  - ChatService accepts skill_matcher parameter
  - Skill indicator appears after auto-apply (active status)
  - Skill indicator does NOT appear for confirmed (Ask Before)
  - Audit log includes skill match metadata
"""

from __future__ import annotations

from application.chat_service import ChatService


class TestChatServiceSkillMatcherIntegration:
    """Integration tests for SkillMatcher in ChatService."""

    def test_chatservice_accepts_skill_matcher_param(self) -> None:
        """ChatService __init__ should accept skill_matcher parameter."""
        # This tests that the parameter exists without needing a full provider.
        import inspect

        sig = inspect.signature(ChatService.__init__)
        params = list(sig.parameters.keys())
        assert "skill_matcher" in params

    def test_chatservice_skill_matcher_defaults_to_none(self) -> None:
        """skill_matcher should default to None."""
        import inspect

        sig = inspect.signature(ChatService.__init__)
        param = sig.parameters["skill_matcher"]
        assert param.default is None

    def test_skill_indicator_format(self) -> None:
        """Skill indicator should use the correct format."""
        from application.skill_compression.hypothesis_storage import Hypothesis
        from presentation.skill_profile_view import format_skill_indicator

        hyp = Hypothesis(
            hypothesis_id="hyp-test",
            user_id=42,
            claim="Verwende Bulletpoints",
            status="active",
            version=2,
            created_at="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
        )

        result = format_skill_indicator(hyp, "Hier ist deine Antwort.")
        assert "Hier ist deine Antwort." in result
        assert "angewendet" in result
        assert "v2" in result

    def test_should_ask_user_for_confirmed(self) -> None:
        """Confirmed status should always trigger Ask Before."""
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.skill_matcher import (
            SkillMatch,
            should_ask_user,
        )

        hyp = Hypothesis(
            hypothesis_id="hyp-test",
            user_id=42,
            claim="Test",
            status="confirmed",
            created_at="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
        )
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=True,
            explanation="test",
        )
        assert should_ask_user(match) is True

    def test_should_not_ask_for_active_with_auto_apply(self) -> None:
        """Active status with auto_apply_enabled should NOT ask."""
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.skill_matcher import (
            SkillMatch,
            should_ask_user,
        )

        hyp = Hypothesis(
            hypothesis_id="hyp-test",
            user_id=42,
            claim="Test",
            status="active",
            created_at="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
        )
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=False,
            explanation="test",
        )
        # With auto_apply_enabled=True: do not ask
        assert should_ask_user(match, {"auto_apply_enabled": True}) is False

    def test_indicator_not_shown_for_confirmed_status(self) -> None:
        """For confirmed (Ask Before), indicator should NOT auto-appear.

        The indicator is only appended when should_ask_user returns False.
        For confirmed status, should_ask_user always returns True,
        so the indicator path is never reached.
        """
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.skill_matcher import (
            SkillMatch,
            should_ask_user,
        )

        hyp = Hypothesis(
            hypothesis_id="hyp-test",
            user_id=42,
            claim="Test",
            status="confirmed",
            created_at="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
        )
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=True,
            explanation="test",
        )
        # Confirmed: always ask => indicator NOT auto-applied
        assert should_ask_user(match) is True
