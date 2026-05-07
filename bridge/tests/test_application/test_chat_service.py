"""Tests fuer application.chat_service: LLM-Aufruf-Orchestration via ChatService-Klasse.

Erstellt eine ChatService-Instanz mit gemockten Dependencies.
Kein echter LLM-Aufruf.
Testet History-Integration, Sprach-Detection, Error-Handling und Auto-Memory-Loading.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.chat_service import ChatService
from domain.conversation import ConversationTurn
from infrastructure.conversation_storage import _reset_all_for_tests
from infrastructure.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _clear_conversation_storage() -> None:
    """Räumt Conversation-Storage vor jedem Test auf."""
    _reset_all_for_tests()


def _make_chat_service(
    route_return: ProviderResponse | None = None,
    route_side_effect: Exception | None = None,
    memory_service: MagicMock | None = None,
) -> tuple[ChatService, MagicMock]:
    """Erstellt einen ChatService mit gemocktem ProviderRouter.

    Returns:
        Tuple von (ChatService, mock_router).
    """
    mock_router = MagicMock()
    if route_side_effect:
        mock_router.route = AsyncMock(side_effect=route_side_effect)
    else:
        mock_router.route = AsyncMock(
            return_value=route_return
            or ProviderResponse(
                text="Antwort von Claude",
                duration_seconds=1.0,
                provider_name="claude",
            )
        )
    svc = ChatService(
        provider_router=mock_router,
        memory_service=memory_service,
    )
    return svc, mock_router


class TestChatService:
    """Chat-Service Use-Case-Tests mit gemocktem ProviderRouter."""

    async def test_process_user_message_calls_provider_router(self) -> None:
        """process_user_message ruft den ProviderRouter auf."""
        svc, mock_router = _make_chat_service()

        result = await svc.process_user_message(
            text="Hallo",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        mock_router.route.assert_called_once()
        assert result.success is True
        assert result.response == "Antwort von Claude"
        assert result.provider_name == "claude"

    async def test_process_user_message_uses_history(self) -> None:
        """Bei vorhandener History wird der Context-Block mit History gebaut."""
        from infrastructure.conversation_storage import save_turn

        # Vorherige Konversation simulieren
        await save_turn(1, 10, ConversationTurn(role="user", content="Fruehere Frage"))
        await save_turn(
            1, 10, ConversationTurn(role="assistant", content="Fruehere Antwort")
        )

        svc, mock_router = _make_chat_service()

        await svc.process_user_message(
            text="Neue Frage",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        # Der Prompt an den Router muss die History enthalten
        call_args = mock_router.route.call_args
        prompt_sent = call_args.kwargs.get("prompt", "")
        assert "Fruehere Frage" in prompt_sent or "VERLAUF" in prompt_sent

    async def test_process_user_message_appends_to_history(self) -> None:
        """Nach erfolgreichem Aufruf werden User-Turn und Assistant-Turn gespeichert."""
        svc, _ = _make_chat_service(
            route_return=ProviderResponse(
                text="Bot sagt hallo",
                duration_seconds=0.5,
                provider_name="claude",
            )
        )

        from infrastructure.conversation_storage import get_history

        await svc.process_user_message(
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

    async def test_process_user_message_error_from_provider(self) -> None:
        """Bei Provider-Fehler wird ein Fehler-Result zurueckgegeben."""
        svc, _ = _make_chat_service(
            route_return=ProviderResponse(
                text="",
                duration_seconds=0.3,
                provider_name="claude",
                error="exit_code_1: Error message",
            )
        )

        result = await svc.process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        assert result.success is False
        assert "Fehler-ID" in result.error_message

    async def test_process_user_message_empty_response(self) -> None:
        """Leere Provider-Antwort erzeugt einen Fehler."""
        svc, _ = _make_chat_service(
            route_return=ProviderResponse(
                text="",
                duration_seconds=0.2,
                provider_name="claude",
            )
        )

        result = await svc.process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        assert result.success is False
        assert "leerer Output" in result.error_message

    async def test_process_user_message_detects_language(self) -> None:
        """Die erkannte Sprache wird im Result zurueckgegeben."""
        svc, _ = _make_chat_service(
            route_return=ProviderResponse(
                text="Response",
                duration_seconds=0.4,
                provider_name="claude",
            )
        )

        result = await svc.process_user_message(
            text="I would like to know something",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        assert result.success is True
        assert result.detected_language == "en"

    async def test_process_user_message_language_override(self) -> None:
        """Expliziter language_override hat Vorrang vor Detection."""
        svc, _ = _make_chat_service(
            route_return=ProviderResponse(
                text="Respuesta",
                duration_seconds=0.3,
                provider_name="claude",
            )
        )

        result = await svc.process_user_message(
            text="Hello",  # Wuerde 'en' detektieren
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
            language_override="es",
        )

        assert result.detected_language == "es"

    async def test_process_user_message_provider_not_found(self) -> None:
        """ValueError vom Router wird sauber abgefangen."""
        svc, _ = _make_chat_service(
            route_side_effect=ValueError("Provider 'xyz' nicht registriert")
        )

        result = await svc.process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
            provider_name="xyz",
        )

        assert result.success is False
        assert "Provider" in result.error_message

    async def test_process_user_message_provider_unavailable(self) -> None:
        """RuntimeError vom Router (Provider nicht verfuegbar) wird abgefangen."""
        svc, _ = _make_chat_service(
            route_side_effect=RuntimeError("Provider 'gemini' ist nicht verfuegbar")
        )

        result = await svc.process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
            provider_name="gemini",
        )

        assert result.success is False
        assert "System-Fehler" in result.error_message

    async def test_reset_clears_conversation(self) -> None:
        """reset() setzt Conversation-History und Sticky-Language zurueck."""
        from infrastructure.conversation_storage import (
            get_history,
            get_language,
            save_turn,
            set_language,
        )

        await save_turn(1, 10, ConversationTurn(role="user", content="Etwas"))
        await set_language(1, 10, "en")

        svc, _ = _make_chat_service()
        await svc.reset(1, 10)

        history = await get_history(1, 10)
        lang = await get_language(1, 10)
        assert history == []
        assert lang is None

    async def test_set_chat_language(self) -> None:
        """set_chat_language() setzt die Sticky-Language."""
        from infrastructure.conversation_storage import get_language

        svc, _ = _make_chat_service()
        await svc.set_chat_language(1, 10, "fr")

        lang = await get_language(1, 10)
        assert lang == "fr"


class TestAutoMemoryLoading:
    """Tests fuer Auto-Memory-Loading im ChatService."""

    async def test_chat_service_loads_memory_when_relevant(self) -> None:
        """Wenn MemoryService Treffer hat, wird Memory-Context in System-Prompt eingefuegt."""
        mock_memory = MagicMock()
        mock_memory.recall = MagicMock(
            side_effect=lambda uid, q, layer, limit: (
                [{"id": "ep_abc123", "content": "Lieblingsessen ist Pizza"}]
                if layer == "episodic"
                else []
            )
        )

        svc, mock_router = _make_chat_service(memory_service=mock_memory)

        await svc.process_user_message(
            text="Was ist mein Lieblingsessen?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        # Pruefen: system_prompt an Router muss Memory-Context enthalten
        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "GESPEICHERTE NOTIZEN" in system_sent
        assert "Lieblingsessen ist Pizza" in system_sent

    async def test_chat_service_skips_memory_when_no_keywords(self) -> None:
        """Bei kurzen Worten (alle <= 3 Zeichen) wird kein Memory geladen."""
        mock_memory = MagicMock()
        mock_memory.recall = MagicMock(return_value=[])

        svc, _ = _make_chat_service(memory_service=mock_memory)

        await svc.process_user_message(
            text="hi da",  # Alle Worte <= 3 Zeichen
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        # recall sollte NIE aufgerufen werden (kein Keyword > 3 Zeichen)
        mock_memory.recall.assert_not_called()

    async def test_chat_service_includes_all_three_layers(self) -> None:
        """Alle drei Layer (episodic, semantic, procedural) erscheinen im Context."""

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

        svc, mock_router = _make_chat_service(memory_service=mock_memory)

        await svc.process_user_message(
            text="Erzaehl mir etwas ueber Python und Testing",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "Episodic" in system_sent
        assert "Semantic" in system_sent
        assert "Procedural" in system_sent
        assert "ep_001" in system_sent
        assert "sem_001" in system_sent
        assert "pro_001" in system_sent
        assert "praeferenz" in system_sent
        assert "testing" in system_sent

    async def test_chat_service_no_memory_when_service_is_none(self) -> None:
        """Ohne MemoryService (None) laeuft alles normal ohne Memory-Block."""
        svc, mock_router = _make_chat_service(memory_service=None)

        result = await svc.process_user_message(
            text="Was ist mein Lieblingsessen?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        assert result.success is True
        # System-Prompt darf keinen Memory-Block haben
        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "GESPEICHERTE NOTIZEN" not in system_sent


class TestExtractKeywords:
    """Tests fuer _extract_keywords (Interpunktions-Bug Regression)."""

    def test_strips_question_mark(self) -> None:
        """Fragezeichen wird vom Keyword entfernt (Regression: mem=0 Bug)."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords("Was ist meine Lieblingssprache?")
        assert "lieblingssprache" in keywords
        assert "lieblingssprache?" not in keywords

    def test_strips_comma(self) -> None:
        """Komma wird vom Keyword entfernt."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords("Hallo, wie gehts?")
        assert "hallo" in keywords
        assert "hallo," not in keywords

    def test_strips_exclamation(self) -> None:
        """Ausrufezeichen wird vom Keyword entfernt."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords("Super! Das funktioniert!")
        assert "super" in keywords
        assert "funktioniert" in keywords
        assert "super!" not in keywords
        assert "funktioniert!" not in keywords

    def test_no_empty_keywords_after_strip(self) -> None:
        """Reine Interpunktion ergibt keine leeren Keywords."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords("... ??? !!! hi")
        # Alle sollten nach Strip zu kurz sein oder leer
        for kw in keywords:
            assert len(kw) > 3
            assert kw.strip() != ""

    def test_sorts_longest_first(self) -> None:
        """Keywords werden nach Laenge absteigend sortiert."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords("Python Lieblingssprache Programmierung")
        assert keywords[0] == "programmierung" or keywords[0] == "lieblingssprache"
        assert len(keywords[0]) >= len(keywords[-1])
