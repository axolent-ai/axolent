"""Tests für application.chat_service: LLM-Aufruf-Orchestration via ChatService-Klasse.

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
from infrastructure.providers.base import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)


@pytest.fixture(autouse=True)
def _clear_conversation_storage() -> None:
    """Räumt Conversation-Storage vor jedem Test auf."""
    _reset_all_for_tests()


def _make_chat_service(
    route_return: ProviderResponse | None = None,
    route_side_effect: Exception | None = None,
    memory_service: MagicMock | None = None,
    self_awareness_service: object | None = None,
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
        self_awareness_service=self_awareness_service,
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
        """Bei Provider-Fehler wird ein Fehler-Result zurückgegeben."""
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
        """Die erkannte Sprache wird im Result zurückgegeben."""
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
        """ValueError vom Router wird sauber abgefangen mit generischer Meldung."""
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
        # Generische Meldung, keine Implementierungs-Details
        assert "Anfrage konnte nicht verarbeitet werden" in result.error_message
        assert "ref:" in result.error_message
        assert result.error_id != ""
        # Original-Exception darf NICHT im User-Text stehen
        assert "xyz" not in result.error_message
        assert "nicht registriert" not in result.error_message

    async def test_process_user_message_provider_unavailable(self) -> None:
        """RuntimeError vom Router wird mit generischer Meldung abgefangen."""
        svc, _ = _make_chat_service(
            route_side_effect=RuntimeError("Provider 'gemini' ist nicht verfügbar")
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
        # Generische Meldung, keine Implementierungs-Details
        assert "Interner Fehler" in result.error_message
        assert "ref:" in result.error_message
        assert result.error_id != ""
        # Original-Exception darf NICHT im User-Text stehen
        assert "gemini" not in result.error_message
        assert "nicht verfügbar" not in result.error_message

    async def test_process_user_message_provider_error_generic(self) -> None:
        """ProviderError liefert generische Meldung ohne Implementierungs-Details."""
        svc, _ = _make_chat_service(
            route_side_effect=ProviderTimeout("claude", timeout_seconds=120)
        )

        result = await svc.process_user_message(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="",
        )

        assert result.success is False
        assert "Sprachmodell-Anbieter" in result.error_message
        assert "ref:" in result.error_message
        assert result.error_id != ""
        # Retryable-Hint muss enthalten sein
        assert "Versuch es gleich noch mal" in result.error_message
        # Original-Exception darf NICHT im User-Text stehen
        assert "Timeout" not in result.error_message
        assert "120" not in result.error_message

    async def test_process_user_message_provider_error_non_retryable(self) -> None:
        """Nicht-retryable ProviderError hat keinen Retry-Hint."""
        svc, _ = _make_chat_service(
            route_side_effect=ProviderError(
                "gemini", retryable=False, message="API key invalid"
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
        assert "Sprachmodell-Anbieter" in result.error_message
        assert "Versuch es gleich noch mal" not in result.error_message
        assert "API key" not in result.error_message

    async def test_reset_clears_conversation(self) -> None:
        """reset() setzt Conversation-History und Sticky-Language zurück."""
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


class TestSaveStreamingResult:
    """Tests für save_streaming_result: History + Audit-Log."""

    async def test_save_streaming_result_writes_audit_with_event_type(self) -> None:
        """save_streaming_result schreibt Audit mit event_type 'stream_completed'."""
        from unittest.mock import patch

        svc, _ = _make_chat_service()

        with patch("application.chat_service.write_audit_log") as mock_audit:
            await svc.save_streaming_result(
                user_id=1,
                chat_id=10,
                user_text="Hallo",
                response_text="Antwort",
                duration_seconds=2.5,
                username="testuser",
                was_cold=True,
                streaming_chunks=5,
                subprocess_pid=1234,
                memory_entries_loaded=2,
            )

            mock_audit.assert_called_once()
            entry = mock_audit.call_args[0][0]
            assert entry["event_type"] == "stream_completed"
            assert entry["user_id"] == 1
            assert entry["chat_id"] == 10
            assert entry["was_cold"] is True
            assert entry["was_warm"] is False
            assert entry["streaming_chunks"] == 5
            assert entry["subprocess_pid"] == 1234
            assert entry["memory_entries_loaded"] == 2
            assert entry["provider"] == "claude_persistent"

    async def test_save_streaming_result_saves_history(self) -> None:
        """save_streaming_result speichert User- und Assistant-Turn in History."""
        from infrastructure.conversation_storage import get_history

        svc, _ = _make_chat_service()

        await svc.save_streaming_result(
            user_id=1,
            chat_id=10,
            user_text="Frage",
            response_text="Antwort",
            duration_seconds=1.0,
        )

        history = await get_history(1, 10)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Frage"
        assert history[1].role == "assistant"
        assert history[1].content == "Antwort"


class TestAutoMemoryLoading:
    """Tests für Auto-Memory-Loading im ChatService."""

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

        # Prüfen: system_prompt an Router muss Memory-Context enthalten
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
            text="Erzähl mir etwas über Python und Testing",
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


class TestChatServiceModelOverride:
    """Tests für User-Modell-Override-Durchreichung an ProviderRouter."""

    async def test_chat_service_passes_user_model_to_router(self) -> None:
        """ChatService reicht User-Override als model= an ProviderRouter.route() weiter."""
        from application.model_service import ModelService
        from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_model_override.db"
            conn = SqliteConnection(db_path)
            try:
                storage = SqliteModelStorage(conn)
                model_service = ModelService(storage=storage)
                model_service.set_user_model(user_id=1, alias_or_id="opus")

                mock_router = MagicMock()
                mock_router.route = AsyncMock(
                    return_value=ProviderResponse(
                        text="Antwort",
                        duration_seconds=1.0,
                        provider_name="claude",
                    )
                )

                svc = ChatService(
                    provider_router=mock_router,
                    model_service=model_service,
                )

                await svc.process_user_message(
                    text="Hallo",
                    user_id=1,
                    chat_id=10,
                    username="test",
                    system_prompt="System.",
                )

                call_kwargs = mock_router.route.call_args[1]
                assert call_kwargs.get("model") == "claude-opus-4-7"
            finally:
                conn.close()


class TestExtractKeywords:
    """Tests für _extract_keywords (Interpunktions-Bug Regression)."""

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
        """Keywords werden nach Länge absteigend sortiert."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords("Python Lieblingssprache Programmierung")
        assert keywords[0] == "programmierung" or keywords[0] == "lieblingssprache"
        assert len(keywords[0]) >= len(keywords[-1])

    def test_filters_german_stop_words(self) -> None:
        """Deutsche Stop-Words werden entfernt."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords(
            "Diese meine Lieblingssprache sollte auch Python sein"
        )
        assert "diese" not in keywords
        assert "meine" not in keywords
        assert "sollte" not in keywords
        assert "auch" not in keywords
        assert "lieblingssprache" in keywords
        assert "python" in keywords

    def test_filters_english_stop_words(self) -> None:
        """Englische Stop-Words werden entfernt."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords(
            "What would their favorite programming language have been"
        )
        assert "what" not in keywords
        assert "would" not in keywords
        assert "their" not in keywords
        assert "have" not in keywords
        assert "been" not in keywords
        assert "favorite" in keywords
        assert "programming" in keywords
        assert "language" in keywords

    def test_stop_words_combined_with_punctuation(self) -> None:
        """Stop-Words werden auch nach Interpunktions-Strip gefiltert."""
        from application.chat_service import _extract_keywords

        keywords = _extract_keywords("Was ist meine Lieblingssprache?")
        assert "meine" not in keywords
        assert "lieblingssprache" in keywords


class TestGetMemoryBudget:
    """Tests für _get_memory_budget: liest max_memory_chars aus Provider-Capabilities."""

    def test_budget_from_provider_capabilities(self) -> None:
        """Wenn Provider max_memory_chars definiert, wird dieser Wert genutzt."""
        from infrastructure.providers.base import ProviderCapabilities

        mock_provider = MagicMock()
        mock_provider.get_capabilities = MagicMock(
            return_value=ProviderCapabilities(max_memory_chars=8000)
        )

        mock_router = MagicMock()
        mock_router.providers = {"claude": mock_provider}
        mock_router.default = "claude"

        svc = ChatService(provider_router=mock_router)
        budget = svc._get_memory_budget("claude")
        assert budget == 8000

    def test_budget_fallback_when_no_provider(self) -> None:
        """Wenn Provider nicht gefunden, Fallback auf MAX_MEMORY_TOTAL_CHARS."""
        from application.chat_service import MAX_MEMORY_TOTAL_CHARS

        mock_router = MagicMock()
        mock_router.providers = {}
        mock_router.default = "claude"

        svc = ChatService(provider_router=mock_router)
        budget = svc._get_memory_budget("nonexistent")
        assert budget == MAX_MEMORY_TOTAL_CHARS

    def test_budget_uses_default_provider_when_none(self) -> None:
        """Wenn provider_name=None, wird der Default-Provider genutzt."""
        from infrastructure.providers.base import ProviderCapabilities

        mock_provider = MagicMock()
        mock_provider.get_capabilities = MagicMock(
            return_value=ProviderCapabilities(max_memory_chars=6000)
        )

        mock_router = MagicMock()
        mock_router.providers = {"claude": mock_provider}
        mock_router.default = "claude"

        svc = ChatService(provider_router=mock_router)
        budget = svc._get_memory_budget(None)
        assert budget == 6000

    def test_budget_fallback_when_router_is_none(self) -> None:
        """Wenn provider_router=None, Fallback auf MAX_MEMORY_TOTAL_CHARS."""
        from application.chat_service import MAX_MEMORY_TOTAL_CHARS

        svc = ChatService(provider_router=None)
        budget = svc._get_memory_budget()
        assert budget == MAX_MEMORY_TOTAL_CHARS


class TestSmartLanguageDetection:
    """Bug-Fix R02-B: Smart-Language-Switch ohne expliziten /lang-Befehl."""

    async def test_implicit_switch_back_to_german(self) -> None:
        """User wechselt nach /lang en implizit zurück auf Deutsch.

        Szenario: /lang en -> Bot auf Englisch -> User schreibt Deutsch -> Sticky wird de.
        """
        from infrastructure.conversation_storage import get_language, set_language

        svc, _ = _make_chat_service()

        # Simuliere: /lang en hat Sticky auf "en" gesetzt
        await set_language(1, 10, "en")

        # User schreibt eine klar deutsche Nachricht
        result = await svc.process_user_message(
            text="Was ist die Hauptstadt von Frankreich?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        # Sprache muss auf Deutsch gewechselt sein
        assert result.detected_language == "de"
        # Sticky muss jetzt "de" sein
        sticky = await get_language(1, 10)
        assert sticky == "de"

    async def test_sticky_stays_when_detection_unclear(self) -> None:
        """Bei kurzer/ambiger Nachricht bleibt Sticky erhalten.

        "ok" hat keine klaren Sprach-Marker -> Confidence niedrig -> Sticky bleibt.
        """
        from infrastructure.conversation_storage import get_language, set_language

        svc, _ = _make_chat_service()

        await set_language(1, 10, "en")

        result = await svc.process_user_message(
            text="ok",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        # Sticky bleibt "en" (Confidence zu niedrig für Wechsel)
        assert result.detected_language == "en"
        sticky = await get_language(1, 10)
        assert sticky == "en"

    async def test_explicit_override_still_works(self) -> None:
        """language_override (von /lang-Command) hat weiterhin Vorrang."""
        from infrastructure.conversation_storage import set_language

        svc, _ = _make_chat_service()

        await set_language(1, 10, "de")

        # Expliziter Override auf Englisch (wie /lang en)
        result = await svc.process_user_message(
            text="Was ist das?",  # Klar Deutsch
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            language_override="en",
        )

        # Override gewinnt
        assert result.detected_language == "en"

    async def test_first_turn_detection_sets_sticky(self) -> None:
        """Beim ersten Turn (kein Sticky vorhanden) setzt Detection die Sprache."""
        from infrastructure.conversation_storage import get_language

        svc, _ = _make_chat_service()

        # Kein vorheriges set_language -> sticky ist None
        result = await svc.process_user_message(
            text="Hello, how are you doing today?",
            user_id=2,
            chat_id=20,
            username="test",
            system_prompt="System.",
        )

        assert result.detected_language == "en"
        sticky = await get_language(2, 20)
        assert sticky == "en"

    async def test_switch_from_german_to_english(self) -> None:
        """User wechselt von Deutsch auf Englisch durch klare englische Nachricht."""
        from infrastructure.conversation_storage import get_language, set_language

        svc, _ = _make_chat_service()

        await set_language(1, 10, "de")

        result = await svc.process_user_message(
            text="Can you please help me with this problem?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
        )

        assert result.detected_language == "en"
        sticky = await get_language(1, 10)
        assert sticky == "en"


class TestSelfAwareness:
    """Tests fuer Self-Awareness-Block: Modell-Info im System-Prompt."""

    async def test_system_prompt_contains_self_awareness_block(self) -> None:
        """System-Prompt an Provider enthaelt Self-Awareness mit Modell-Info."""
        from application.model_registry import ModelRegistry
        from application.self_awareness_service import SelfAwarenessService

        sa_svc = SelfAwarenessService(
            model_service=None,
            task_router=None,
            model_registry=ModelRegistry(),
        )
        svc, mock_router = _make_chat_service(self_awareness_service=sa_svc)

        await svc.process_user_message(
            text="Welches Modell bist du?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "[SELF-AWARENESS]" in system_sent
        assert "Modell:" in system_sent
        assert "Slot:" in system_sent
        assert "Provider:" in system_sent

    async def test_self_awareness_reflects_model_override(self) -> None:
        """Nach /setmodel opus enthaelt Self-Awareness Opus-Daten."""
        from application.model_registry import ModelRegistry
        from application.model_service import ModelService
        from application.self_awareness_service import SelfAwarenessService
        from application.task_router import TaskRouter, load_slot_configs
        from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_self_awareness.db"
            conn = SqliteConnection(db_path)
            try:
                storage = SqliteModelStorage(conn)
                slot_configs = load_slot_configs()
                slot_defaults = {
                    cfg.slot.value: cfg.default_model for cfg in slot_configs
                }
                model_service = ModelService(
                    storage=storage, slot_defaults=slot_defaults
                )
                model_service.set_user_model(user_id=1, alias_or_id="opus")

                task_router = TaskRouter(
                    slot_configs=slot_configs,
                    model_service=model_service,
                )

                registry = ModelRegistry()
                sa_svc = SelfAwarenessService(
                    model_service=model_service,
                    task_router=task_router,
                    model_registry=registry,
                )

                mock_router = MagicMock()
                mock_router.route = AsyncMock(
                    return_value=ProviderResponse(
                        text="Antwort",
                        duration_seconds=1.0,
                        provider_name="claude",
                    )
                )

                svc = ChatService(
                    provider_router=mock_router,
                    model_service=model_service,
                    task_router=task_router,
                    self_awareness_service=sa_svc,
                )

                await svc.process_user_message(
                    text="Welches Modell nutzt du?",
                    user_id=1,
                    chat_id=10,
                    username="test",
                    system_prompt="System.",
                )

                call_args = mock_router.route.call_args
                system_sent = call_args.kwargs.get("system_prompt", "")
                assert "Opus 4.7" in system_sent
                assert "claude-opus-4-7" in system_sent
                assert "anthropic" in system_sent
            finally:
                conn.close()

    async def test_self_awareness_contains_task_slot(self) -> None:
        """Self-Awareness enthaelt den erkannten Task-Slot."""
        from application.model_registry import ModelRegistry
        from application.model_service import ModelService
        from application.self_awareness_service import SelfAwarenessService
        from application.task_router import TaskRouter, load_slot_configs
        from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_self_awareness_slot.db"
            conn = SqliteConnection(db_path)
            try:
                storage = SqliteModelStorage(conn)
                slot_configs = load_slot_configs()
                model_service = ModelService(storage=storage)
                task_router = TaskRouter(
                    slot_configs=slot_configs,
                    model_service=model_service,
                )

                registry = ModelRegistry()
                sa_svc = SelfAwarenessService(
                    model_service=model_service,
                    task_router=task_router,
                    model_registry=registry,
                )

                mock_router = MagicMock()
                mock_router.route = AsyncMock(
                    return_value=ProviderResponse(
                        text="Antwort",
                        duration_seconds=1.0,
                        provider_name="claude",
                    )
                )

                svc = ChatService(
                    provider_router=mock_router,
                    model_service=model_service,
                    task_router=task_router,
                    self_awareness_service=sa_svc,
                )

                # Code-Nachricht um den CODE-Slot zu triggern
                await svc.process_user_message(
                    text="/code Schreibe eine Python-Funktion",
                    user_id=1,
                    chat_id=10,
                    username="test",
                    system_prompt="System.",
                )

                call_args = mock_router.route.call_args
                system_sent = call_args.kwargs.get("system_prompt", "")
                assert "Slot: code" in system_sent
            finally:
                conn.close()

    async def test_no_self_awareness_when_service_is_none(self) -> None:
        """Ohne SelfAwarenessService (None) laeuft alles normal ohne Block."""
        svc, mock_router = _make_chat_service(self_awareness_service=None)

        result = await svc.process_user_message(
            text="Welches Modell bist du?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        assert result.success is True
        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")
        assert "[SELF-AWARENESS]" not in system_sent
