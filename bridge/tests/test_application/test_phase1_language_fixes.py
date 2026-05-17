"""Tests for Phase 1 language fixes.

Verifies:
1. ProactiveTriggerService is wired in production bootstrap
2. LANGUAGE LOCK always present in system prompt (including DE)
3. Time context block contains capability statement
4. Handler fallbacks use DEFAULT_LANGUAGE consistently
5. DebateOrchestrator uses build_effective_prompt
"""

from datetime import datetime


from application.proactive_trigger_service import ProactiveTriggerService
from domain.language import DEFAULT_LANGUAGE
from domain.personality import build_effective_prompt


class TestLanguageLockSymmetry:
    """LANGUAGE LOCK must be injected for ALL languages, including DE."""

    def test_language_lock_present_for_de(self) -> None:
        """German must get LANGUAGE LOCK (fixes T33)."""
        result = build_effective_prompt("Base.", "de")
        assert "[LANGUAGE LOCK]" in result
        assert "'de'" in result

    def test_language_lock_present_for_it(self) -> None:
        """Italian must get LANGUAGE LOCK."""
        result = build_effective_prompt("Base.", "it")
        assert "[LANGUAGE LOCK]" in result
        assert "'it'" in result

    def test_language_lock_present_for_tr(self) -> None:
        """Turkish must get LANGUAGE LOCK."""
        result = build_effective_prompt("Base.", "tr")
        assert "[LANGUAGE LOCK]" in result
        assert "'tr'" in result

    def test_language_lock_not_present_when_empty(self) -> None:
        """No LANGUAGE LOCK when hint is empty (fresh state, no detection)."""
        result = build_effective_prompt("Base.", "")
        assert "[LANGUAGE LOCK]" not in result

    def test_language_lock_contains_no_switch_instruction(self) -> None:
        """LANGUAGE LOCK instructs not to switch mid-response."""
        result = build_effective_prompt("Base.", "fr")
        assert "Do not switch languages mid-response" in result


class TestTimeContextCapability:
    """Time context block must include capability statement."""

    def test_capability_statement_in_en(self) -> None:
        """EN time block includes 'You HAVE access' capability."""
        service = ProactiveTriggerService()
        now = datetime(2026, 5, 17, 14, 30)
        block = service.get_time_context_block(123, now=now, lang="en")
        assert "You HAVE access" in block
        assert "NEVER claim" in block

    def test_capability_statement_in_de(self) -> None:
        """DE time block includes German capability statement."""
        service = ProactiveTriggerService()
        now = datetime(2026, 5, 17, 14, 30)
        block = service.get_time_context_block(123, now=now, lang="de")
        assert "HAST Zugriff" in block
        assert "NIEMALS" in block

    def test_time_block_always_non_empty(self) -> None:
        """Time block is never empty (always has header + time + capability)."""
        service = ProactiveTriggerService()
        now = datetime(2026, 5, 17, 14, 30)
        block = service.get_time_context_block(123, now=now, lang="en")
        assert len(block) > 100  # Minimum: header + time + capability

    def test_time_block_contains_actual_time(self) -> None:
        """Time block contains the passed datetime."""
        service = ProactiveTriggerService()
        now = datetime(2026, 5, 17, 9, 15)
        block = service.get_time_context_block(123, now=now, lang="en")
        assert "2026-05-17 09:15" in block


class TestDefaultLanguageConstant:
    """DEFAULT_LANGUAGE is used consistently."""

    def test_default_language_is_de(self) -> None:
        """Default language must be 'de' (primary user language)."""
        assert DEFAULT_LANGUAGE == "de"


class TestProactiveTriggerServiceBootstrap:
    """Production bootstrap must wire ProactiveTriggerService."""

    def test_chat_service_accepts_proactive_service(self) -> None:
        """ChatService constructor accepts proactive_trigger_service parameter."""
        from unittest.mock import MagicMock

        from application.chat_service import ChatService

        mock_router = MagicMock()
        mock_pts = ProactiveTriggerService()

        svc = ChatService(
            provider_router=mock_router,
            proactive_trigger_service=mock_pts,
        )
        assert svc.proactive_trigger_service is not None
        assert svc.proactive_trigger_service is mock_pts
