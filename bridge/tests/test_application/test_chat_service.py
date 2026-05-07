"""Tests fuer application.chat_service: LLM-Aufruf-Orchestration via ProviderRouter.

Mockt den ProviderRouter komplett. Kein echter LLM-Aufruf.
Testet History-Integration, Sprach-Detection, Error-Handling und Auto-Memory-Loading.
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


@pytest.fixture(autouse=True)
def _clear_memory_service():
    """Raeumt MemoryService-Referenz im ChatService vor/nach jedem Test auf."""
    from application import chat_service

    old_memory = chat_service._memory_service
    chat_service._memory_service = None
    yield
    chat_service._memory_service = old_memory


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


class TestAutoMemoryLoading:
    """Tests fuer Auto-Memory-Loading im Chat-Service."""

    async def test_chat_service_loads_memory_when_relevant(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Wenn MemoryService Treffer hat, wird Memory-Context in System-Prompt eingefuegt."""
        from application import chat_service
        from application.chat_service import process_user_message

        # Mock MemoryService
        mock_memory = MagicMock()
        mock_memory.recall = MagicMock(
            side_effect=lambda uid, q, layer, limit: (
                [{"id": "ep_abc123", "content": "Lieblingsessen ist Pizza"}]
                if layer == "episodic"
                else []
            )
        )
        chat_service._memory_service = mock_memory

        await process_user_message(
            text="Was ist mein Lieblingsessen?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        # Pruefen: system_prompt an Router muss Memory-Context enthalten
        call_args = _inject_mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "GESPEICHERTE NOTIZEN" in system_sent
        assert "Lieblingsessen ist Pizza" in system_sent

    async def test_chat_service_skips_memory_when_no_keywords(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Bei kurzen Worten (alle <= 3 Zeichen) wird kein Memory geladen."""
        from application import chat_service
        from application.chat_service import process_user_message

        mock_memory = MagicMock()
        mock_memory.recall = MagicMock(return_value=[])
        chat_service._memory_service = mock_memory

        await process_user_message(
            text="hi da",  # Alle Worte <= 3 Zeichen
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        # recall sollte NIE aufgerufen werden (kein Keyword > 3 Zeichen)
        mock_memory.recall.assert_not_called()

    async def test_chat_service_includes_all_three_layers(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Alle drei Layer (episodic, semantic, procedural) erscheinen im Context."""
        from application import chat_service
        from application.chat_service import process_user_message

        def mock_recall(uid, q, layer, limit):
            if layer == "episodic":
                return [{"id": "ep_001", "content": "Event gestern"}]
            elif layer == "semantic":
                return [
                    {
                        "id": "sem_001",
                        "content": "Python ist Lieblingssprache",
                        "category": "praeferenz",
                    }
                ]
            elif layer == "procedural":
                return [
                    {
                        "id": "pro_001",
                        "content": "Immer Tests schreiben",
                        "skill_name": "testing",
                    }
                ]
            return []

        mock_memory = MagicMock()
        mock_memory.recall = MagicMock(side_effect=mock_recall)
        chat_service._memory_service = mock_memory

        await process_user_message(
            text="Erzaehl mir etwas ueber Python und Testing",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        call_args = _inject_mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "Episodic" in system_sent
        assert "Semantic" in system_sent
        assert "Procedural" in system_sent
        assert "ep_001" in system_sent
        assert "sem_001" in system_sent
        assert "pro_001" in system_sent
        assert "praeferenz" in system_sent
        assert "testing" in system_sent

    async def test_chat_service_no_memory_when_service_is_none(
        self, _inject_mock_router: MagicMock
    ) -> None:
        """Ohne MemoryService (None) laeuft alles normal ohne Memory-Block."""
        from application import chat_service
        from application.chat_service import process_user_message

        # Sicherstellen: kein MemoryService
        chat_service._memory_service = None

        result = await process_user_message(
            text="Was ist mein Lieblingsessen?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        assert result.success is True
        # System-Prompt darf keinen Memory-Block haben
        call_args = _inject_mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "GESPEICHERTE NOTIZEN" not in system_sent
