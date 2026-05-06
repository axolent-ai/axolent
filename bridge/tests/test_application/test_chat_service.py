"""Tests fuer application.chat_service: LLM-Aufruf-Orchestration via ProviderRouter.

Mockt den ProviderRouter komplett. Kein echter LLM-Aufruf.
Testet History-Integration, Sprach-Detection und Error-Handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.conversation import ConversationTurn
from infrastructure.conversation_storage import _histories, _languages
from infrastructure.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _clear_conversation_storage() -> None:
    """Raeumt Conversation-Storage vor jedem Test auf."""
    _histories.clear()
    _languages.clear()


@pytest.fixture(autouse=True)
def _inject_mock_router():
    """Injiziert einen gemockten ProviderRouter in den ChatService."""
    from application import chat_service

    mock_router = MagicMock()
    mock_router.route = AsyncMock(
        return_value=ProviderResponse(
            text="Antwort von Claude",
            duration_seconds=1.0,
            provider_name="claude",
        )
    )
    old_router = chat_service._provider_router
    chat_service._provider_router = mock_router
    yield mock_router
    chat_service._provider_router = old_router


class TestChatService:
    """Chat-Service Use-Case-Tests mit gemocktem ProviderRouter."""

    async def test_process_user_message_calls_provider_router(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """process_user_message ruft den ProviderRouter auf."""
        from application.chat_service import process_user_message

        result = await process_user_message(
            text="Hallo",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        _inject_mock_router.route.assert_called_once()
        assert result.success is True
        assert result.response == "Antwort von Claude"
        assert result.provider_name == "claude"

    async def test_process_user_message_uses_history(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Bei vorhandener History wird der Context-Block mit History gebaut."""
        from infrastructure.conversation_storage import save_turn

        # Vorherige Konversation simulieren
        await save_turn(1, 10, ConversationTurn(role="user", content="Fruehere Frage"))
        await save_turn(
            1, 10, ConversationTurn(role="assistant", content="Fruehere Antwort")
        )

        from application.chat_service import process_user_message

        await process_user_message(
            text="Neue Frage",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        # Der Prompt an den Router muss die History enthalten
        call_args = _inject_mock_router.route.call_args
        prompt_sent = call_args.kwargs.get("prompt", "")
        assert "Fruehere Frage" in prompt_sent or "VERLAUF" in prompt_sent

    async def test_process_user_message_appends_to_history(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Nach erfolgreichem Aufruf werden User-Turn und Assistant-Turn gespeichert."""
        _inject_mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="Bot sagt hallo",
                duration_seconds=0.5,
                provider_name="claude",
            )
        )

        from application.chat_service import process_user_message
        from infrastructure.conversation_storage import get_history

        await process_user_message(
            text="User sagt hi",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        history = await get_history(1, 10)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "User sagt hi"
        assert history[1].role == "assistant"
        assert history[1].content == "Bot sagt hallo"

    async def test_process_user_message_error_from_provider(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Bei Provider-Fehler wird ein Fehler-Result zurueckgegeben."""
        _inject_mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="",
                duration_seconds=0.3,
                provider_name="claude",
                error="exit_code_1: Error message",
            )
        )

        from application.chat_service import process_user_message

        result = await process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        assert result.success is False
        assert "Fehler-ID" in result.error_message

    async def test_process_user_message_empty_response(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Leere Provider-Antwort erzeugt einen Fehler."""
        _inject_mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="",
                duration_seconds=0.2,
                provider_name="claude",
            )
        )

        from application.chat_service import process_user_message

        result = await process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        assert result.success is False
        assert "leerer Output" in result.error_message

    async def test_process_user_message_detects_language(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Die erkannte Sprache wird im Result zurueckgegeben."""
        _inject_mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="Response",
                duration_seconds=0.4,
                provider_name="claude",
            )
        )

        from application.chat_service import process_user_message

        result = await process_user_message(
            text="I would like to know something",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        assert result.success is True
        assert result.detected_language == "en"

    async def test_process_user_message_language_override(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Expliziter language_override hat Vorrang vor Detection."""
        _inject_mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="Respuesta",
                duration_seconds=0.3,
                provider_name="claude",
            )
        )

        from application.chat_service import process_user_message

        result = await process_user_message(
            text="Hello",  # Wuerde 'en' detektieren
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
            language_override="es",
        )

        assert result.detected_language == "es"

    async def test_process_user_message_provider_not_found(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """ValueError vom Router wird sauber abgefangen."""
        _inject_mock_router.route = AsyncMock(
            side_effect=ValueError("Provider 'xyz' nicht registriert")
        )

        from application.chat_service import process_user_message

        result = await process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
            provider_name="xyz",
        )

        assert result.success is False
        assert "Provider" in result.error_message

    async def test_process_user_message_provider_unavailable(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """RuntimeError vom Router (Provider nicht verfuegbar) wird abgefangen."""
        _inject_mock_router.route = AsyncMock(
            side_effect=RuntimeError("Provider 'gemini' ist nicht verfuegbar")
        )

        from application.chat_service import process_user_message

        result = await process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
            provider_name="gemini",
        )

        assert result.success is False
        assert "System-Fehler" in result.error_message
