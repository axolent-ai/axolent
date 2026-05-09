"""Tests fuer den Status-Manager (R02-B).

Verifiziert:
    - Status-Update Rate-Limiting (max alle 0.5s)
    - Status-Updates stoppen wenn Stream beginnt
    - Sprach-Auswahl (Sticky-Language)
    - Status zeigt Memory-Anzahl korrekt
    - SHOW_STATUS_UPDATES toggle
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock


from application.status_manager import (
    STATUS_RATE_LIMIT_SECONDS,
    StatusSession,
    get_status_text,
)


class TestGetStatusText:
    """Tests fuer get_status_text()."""

    def test_german_memory_loading(self) -> None:
        """Deutsche Status-Texte fuer Memory-Loading."""
        text = get_status_text("memory_loading", "de")
        assert "Lade Notizen" in text

    def test_english_memory_loading(self) -> None:
        """Englische Status-Texte fuer Memory-Loading."""
        text = get_status_text("memory_loading", "en")
        assert "Loading memory" in text

    def test_memory_loaded_with_count(self) -> None:
        """Memory-Loaded zeigt Anzahl korrekt."""
        text = get_status_text("memory_loaded", "de", n=3)
        assert "3 gefunden" in text

    def test_memory_loaded_english_with_count(self) -> None:
        """Memory-Loaded English zeigt Anzahl korrekt."""
        text = get_status_text("memory_loaded", "en", n=5)
        assert "5 entries" in text

    def test_thinking_german(self) -> None:
        """Thinking-Status auf Deutsch."""
        text = get_status_text("thinking", "de")
        assert "Denke nach" in text

    def test_thinking_english(self) -> None:
        """Thinking-Status auf Englisch."""
        text = get_status_text("thinking", "en")
        assert "Thinking" in text

    def test_formatting_german(self) -> None:
        """Formatting-Status auf Deutsch."""
        text = get_status_text("formatting", "de")
        assert "Formatiere" in text

    def test_formatting_english(self) -> None:
        """Formatting-Status auf Englisch."""
        text = get_status_text("formatting", "en")
        assert "Formatting" in text

    def test_unknown_language_falls_back_to_german(self) -> None:
        """Unbekannte Sprache faellt auf Deutsch zurueck."""
        text = get_status_text("thinking", "xx")
        assert "Denke nach" in text

    def test_unknown_key_returns_key(self) -> None:
        """Unbekannter Key gibt den Key selbst zurueck."""
        text = get_status_text("nonexistent_key", "de")
        assert text == "nonexistent_key"


class TestStatusSession:
    """Tests fuer StatusSession."""

    async def test_update_calls_callback(self) -> None:
        """update() ruft den Callback mit formatiertem Text auf."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("thinking")

        mock_callback.assert_called_once()
        call_text = mock_callback.call_args[0][0]
        assert "Denke nach" in call_text

    async def test_rate_limiting(self) -> None:
        """Schnelle aufeinanderfolgende Updates werden rate-limited."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        # Erstes Update geht durch
        await session.update("thinking")
        assert mock_callback.call_count == 1

        # Zweites Update sofort danach wird geblockt (rate-limit)
        await session.update("memory_loading")
        assert mock_callback.call_count == 1

    async def test_rate_limiting_allows_after_interval(self) -> None:
        """Nach dem Rate-Limit-Intervall geht das naechste Update durch."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("thinking")
        assert mock_callback.call_count == 1

        # Simuliere Zeitvergehen
        session.last_update_time = time.monotonic() - STATUS_RATE_LIMIT_SECONDS - 0.1

        await session.update("memory_loading")
        assert mock_callback.call_count == 2

    async def test_stream_started_stops_updates(self) -> None:
        """Nach mark_stream_started() werden keine Updates mehr gesendet."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("thinking")
        assert mock_callback.call_count == 1

        session.mark_stream_started()
        session.last_update_time = 0  # Rate-Limit zuruecksetzen

        await session.update("formatting")
        # Immer noch nur 1 Call (nach mark_stream_started blockiert)
        assert mock_callback.call_count == 1

    async def test_disabled_session_sends_nothing(self) -> None:
        """Deaktivierte Session sendet keine Updates."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de", enabled=False)

        await session.update("thinking")
        mock_callback.assert_not_called()

    async def test_language_selection(self) -> None:
        """Session nutzt die konfigurierte Sprache."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="en")

        await session.update("thinking")

        call_text = mock_callback.call_args[0][0]
        assert "Thinking" in call_text

    async def test_callback_exception_handled_gracefully(self) -> None:
        """Exception im Callback crasht nicht die Session."""
        mock_callback = AsyncMock(side_effect=Exception("Telegram down"))
        session = StatusSession(callback=mock_callback, language="de")

        # Darf nicht raisen
        await session.update("thinking")
        mock_callback.assert_called_once()

    async def test_memory_count_in_status(self) -> None:
        """Memory-Loaded-Status zeigt die korrekte Anzahl."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("memory_loaded", n=7)

        call_text = mock_callback.call_args[0][0]
        assert "7 gefunden" in call_text


class TestStatusInStreaming:
    """Integration-Tests: Status-Updates im Streaming-Flow."""

    async def test_status_session_passed_to_chat_service(self) -> None:
        """StatusSession wird an process_user_message_streaming durchgereicht."""
        from unittest.mock import MagicMock

        from application.chat_service import ChatService
        from infrastructure.claude_process_pool import StreamEvent
        from infrastructure.conversation_storage import _reset_all_for_tests

        _reset_all_for_tests()

        mock_router = MagicMock()
        mock_router.providers = {}
        mock_router.default = "claude"
        svc = ChatService(provider_router=mock_router, memory_service=None)

        # Mock persistent_provider
        mock_provider = MagicMock()

        async def mock_stream(**kwargs):
            yield StreamEvent(event_type="content_delta", text="Hello")
            yield StreamEvent(
                event_type="result", full_text="Hello World", is_final=True
            )

        mock_provider.query_streaming = mock_stream

        # Status-Session
        mock_callback = AsyncMock()
        status = StatusSession(callback=mock_callback, language="de")

        stream_iter, mem_count = await svc.process_user_message_streaming(
            text="Test Frage",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
            status_session=status,
        )

        # Status-Updates muessen VOR dem Stream gesendet worden sein
        # (memory_loading + thinking)
        assert mock_callback.call_count >= 1
        first_call = mock_callback.call_args_list[0][0][0]
        assert "Lade Notizen" in first_call or "Denke nach" in first_call

        # Stream konsumieren
        events = []
        async for event in stream_iter:
            events.append(event)

        # Nach erstem Token muss stream_started gesetzt sein
        assert status.stream_started is True

    async def test_no_status_when_session_is_none(self) -> None:
        """Ohne StatusSession laeuft alles normal (Backward-Compat)."""
        from unittest.mock import MagicMock

        from application.chat_service import ChatService
        from infrastructure.claude_process_pool import StreamEvent
        from infrastructure.conversation_storage import _reset_all_for_tests

        _reset_all_for_tests()

        mock_router = MagicMock()
        mock_router.providers = {}
        mock_router.default = "claude"
        svc = ChatService(provider_router=mock_router, memory_service=None)

        mock_provider = MagicMock()

        async def mock_stream(**kwargs):
            yield StreamEvent(event_type="result", full_text="OK", is_final=True)

        mock_provider.query_streaming = mock_stream

        # Kein status_session
        stream_iter, _ = await svc.process_user_message_streaming(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
            status_session=None,
        )

        events = []
        async for event in stream_iter:
            events.append(event)
        assert len(events) == 1
        assert events[0].event_type == "result"
