"""Tests for the status manager (R02-B).

Verifies:
    - Status update rate limiting (max every 0.5s)
    - Minimum display duration (MIN_STATUS_DISPLAY_MS)
    - Status updates stop when stream begins
    - Language selection (sticky language)
    - Status shows memory count correctly
    - SHOW_STATUS_UPDATES toggle
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock


from application.status_manager import (
    MIN_STATUS_DISPLAY_MS,
    STATUS_RATE_LIMIT_SECONDS,
    StatusSession,
    get_status_text,
)


class TestGetStatusText:
    """Tests für get_status_text()."""

    def test_german_memory_loading(self) -> None:
        """Deutsche Status-Texte für Memory-Loading."""
        text = get_status_text("memory_loading", "de")
        assert "Lade Notizen" in text

    def test_english_memory_loading(self) -> None:
        """Englische Status-Texte für Memory-Loading."""
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

    def test_unknown_language_falls_back_to_english(self) -> None:
        """Unbekannte Sprache fällt auf Englisch zurück."""
        text = get_status_text("thinking", "xx")
        assert "Thinking" in text

    def test_unknown_key_returns_key(self) -> None:
        """Unbekannter Key gibt den Key selbst zurück."""
        text = get_status_text("nonexistent_key", "de")
        assert text == "nonexistent_key"


class TestStatusSession:
    """Tests für StatusSession."""

    async def test_update_calls_callback(self) -> None:
        """update() ruft den Callback mit formatiertem Text auf."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("thinking")

        mock_callback.assert_called_once()
        call_text = mock_callback.call_args[0][0]
        assert "Denke nach" in call_text

    async def test_rate_limiting(self) -> None:
        """Schnelle aufeinanderfolgende Updates mit gleichem Key werden rate-limited."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        # Erstes Update geht durch
        await session.update("thinking")
        assert mock_callback.call_count == 1

        # Zweites Update mit GLEICHEM Key sofort danach wird geblockt (rate-limit)
        await session.update("thinking")
        assert mock_callback.call_count == 1

    async def test_rate_limiting_allows_after_interval(self) -> None:
        """Nach dem Rate-Limit-Intervall geht das nächste Update durch."""
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
        session.last_update_time = 0  # Rate-Limit zurücksetzen

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


class TestPhaseChangeBypassesRateLimit:
    """Bug-Fix-Tests: Phase-Change umgeht Rate-Limit."""

    async def test_different_key_bypasses_rate_limit(self) -> None:
        """Wechsel des Status-Keys (neue Phase) umgeht das Rate-Limit."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        # Erstes Update: memory_loading
        await session.update("memory_loading")
        assert mock_callback.call_count == 1

        # Zweites Update SOFORT danach mit anderem Key: muss durchgehen
        await session.update("thinking")
        assert mock_callback.call_count == 2

        # Dritter Aufruf mit gleichem Key sofort: wird rate-limited
        await session.update("thinking")
        assert mock_callback.call_count == 2

    async def test_same_key_still_rate_limited(self) -> None:
        """Gleicher Status-Key wird weiterhin rate-limited."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("memory_loaded", n=3)
        assert mock_callback.call_count == 1

        # Gleicher Key sofort nochmal: wird geblockt
        await session.update("memory_loaded", n=5)
        assert mock_callback.call_count == 1

    async def test_phase_change_sequence_memory_thinking(self) -> None:
        """Realistischer Flow: memory_loading -> thinking ohne Verzoegerung."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("memory_loading")
        await session.update("thinking")

        assert mock_callback.call_count == 2
        calls = [c[0][0] for c in mock_callback.call_args_list]
        assert "Lade Notizen" in calls[0]
        assert "Denke nach" in calls[1]


class TestStatusLanguageUpdate:
    """Bug-Fix-Tests: Status-Sprache respektiert Sticky-Language."""

    async def test_set_language_changes_output(self) -> None:
        """set_language() ändert die Sprache für folgende Updates."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("memory_loading")
        first_text = mock_callback.call_args_list[0][0][0]
        assert "Lade Notizen" in first_text

        # Sprache wechseln
        session.set_language("en")

        # Nächstes Update muss Englisch sein (Phase-Change -> kein Rate-Limit)
        await session.update("thinking")
        second_text = mock_callback.call_args_list[1][0][0]
        assert "Thinking" in second_text

    async def test_english_user_gets_english_status(self) -> None:
        """User auf Englisch bekommt englische Status-Texte."""
        mock_callback = AsyncMock()
        # Session startet mit Default "de" (wie in handlers.py wenn kein Sticky)
        session = StatusSession(callback=mock_callback, language="de")

        # Simuliere: chat_service bestimmt Sprache als "en" und ruft set_language auf
        session.set_language("en")

        await session.update("memory_loading")
        await session.update("thinking")

        calls = [c[0][0] for c in mock_callback.call_args_list]
        assert "Loading memory" in calls[0]
        assert "Thinking" in calls[1]


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

        stream_iter, mem_count, _task_meta = await svc.process_user_message_streaming(
            text="Test Frage",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
            status_session=status,
        )

        # Status-Updates müssen VOR dem Stream gesendet worden sein
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
        stream_iter, _, _task_meta = await svc.process_user_message_streaming(
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

    async def test_english_user_gets_english_thinking_status(self) -> None:
        """Bug-Fix: Englischer User bekommt englischen Thinking-Status."""
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
            yield StreamEvent(event_type="content_delta", text="Sure")
            yield StreamEvent(
                event_type="result", full_text="Sure thing!", is_final=True
            )

        mock_provider.query_streaming = mock_stream

        # StatusSession mit Default "de" (wie handlers.py bei erstem Turn)
        mock_callback = AsyncMock()
        status = StatusSession(callback=mock_callback, language="de")

        # Englische Nachricht: Sprach-Detection erkennt "en"
        stream_iter, _, _task_meta = await svc.process_user_message_streaming(
            text="Hey, what is the weather like today?",
            user_id=99,
            chat_id=99,
            username="english_user",
            system_prompt="You are a helpful assistant.",
            persistent_provider=mock_provider,
            status_session=status,
        )

        # "thinking" Status muss auf Englisch sein
        # (chat_service ruft set_language auf der Session auf)
        thinking_calls = [
            c[0][0]
            for c in mock_callback.call_args_list
            if "Thinking" in c[0][0] or "Denke" in c[0][0]
        ]
        assert len(thinking_calls) == 1
        assert "Thinking" in thinking_calls[0]  # Englisch, nicht Deutsch

        # Session-Sprache muss "en" sein
        assert status.language == "en"

        # Stream konsumieren
        async for _ in stream_iter:
            pass

    async def test_both_status_updates_shown_in_streaming(self) -> None:
        """Bug-Fix: Sowohl memory_loading ALS AUCH thinking werden angezeigt."""
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
            yield StreamEvent(event_type="content_delta", text="Hallo")
            yield StreamEvent(event_type="result", full_text="Hallo!", is_final=True)

        mock_provider.query_streaming = mock_stream

        mock_callback = AsyncMock()
        status = StatusSession(callback=mock_callback, language="de")

        stream_iter, _, _task_meta = await svc.process_user_message_streaming(
            text="Hallo Welt",
            user_id=1,
            chat_id=1,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
            status_session=status,
        )

        # Beide Status-Updates müssen gesendet worden sein
        # (memory_loading + thinking, weil Phase-Change Rate-Limit umgeht)
        assert mock_callback.call_count == 2
        calls = [c[0][0] for c in mock_callback.call_args_list]
        assert "Lade Notizen" in calls[0]
        assert "Denke nach" in calls[1]

        # Stream konsumieren
        async for _ in stream_iter:
            pass


class TestMinStatusDisplayTime:
    """Bug-Fix R02-B: Mindest-Anzeigedauer für Status-Updates."""

    async def test_min_display_time_enforced_between_updates(self) -> None:
        """Zwischen zwei Status-Updates liegen mindestens MIN_STATUS_DISPLAY_MS."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        # Erstes Update
        await session.update("memory_loading")
        first_time = session.last_update_time

        # Zweites Update sofort (Phase-Change)
        await session.update("thinking")
        second_time = session.last_update_time

        # Mindestens MIN_STATUS_DISPLAY_MS ms zwischen den beiden Callbacks
        elapsed_ms = (second_time - first_time) * 1000
        assert elapsed_ms >= MIN_STATUS_DISPLAY_MS - 50  # 50ms Toleranz für Timer

    async def test_min_display_time_not_applied_on_first_update(self) -> None:
        """Erstes Status-Update hat keine Verzoegerung."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        before = time.monotonic()
        await session.update("memory_loading")
        after = time.monotonic()

        # Kein Sleep beim allerersten Update (last_update_time war 0)
        elapsed_ms = (after - before) * 1000
        assert elapsed_ms < 100  # Sollte quasi instant sein

    async def test_min_display_time_skipped_when_stream_started(self) -> None:
        """Kein Sleep wenn Stream bereits gestartet ist."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("memory_loading")
        session.mark_stream_started()

        # Dieses Update darf nicht blockieren (und wird ohnehin ignoriert)
        before = time.monotonic()
        await session.update("thinking")
        after = time.monotonic()

        elapsed_ms = (after - before) * 1000
        assert elapsed_ms < 50
        # Nur 1 Call (zweites wird durch stream_started blockiert)
        assert mock_callback.call_count == 1

    async def test_no_sleep_when_enough_time_passed(self) -> None:
        """Kein Sleep wenn bereits genug Zeit vergangen ist."""
        mock_callback = AsyncMock()
        session = StatusSession(callback=mock_callback, language="de")

        await session.update("memory_loading")
        # Simuliere: 2 Sekunden vergangen
        session.last_update_time = time.monotonic() - 2.0

        before = time.monotonic()
        await session.update("thinking")
        after = time.monotonic()

        # Kein Sleep noetig (2s > 1100ms)
        elapsed_ms = (after - before) * 1000
        assert elapsed_ms < 50
