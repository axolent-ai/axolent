"""Tests for the typing keepalive mechanism (R02-A).

Tests:
    1. Keepalive sends typing action periodically
    2. Keepalive aborts cleanly on cancel
    3. Keepalive does not crash on API error
    4. Integration: handle_message starts and cancels keepalive correctly
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from presentation.handlers import (
    TYPING_KEEPALIVE_INTERVAL_SECONDS,
    _typing_keepalive,
)


def _make_mock_context_kernel():
    """Create a mock ContextKernel for typing keepalive tests."""
    from application.execution.context import ExecutionContext
    from application.execution.kernel import ContextKernel
    from application.language_resolver import LanguageContext

    kernel = AsyncMock(spec=ContextKernel)

    async def _build(envelope, language_override=None):
        return ExecutionContext(
            request_id=envelope.request_id,
            user_id=envelope.user_id,
            chat_id=envelope.chat_id,
            channel="telegram",
            language=LanguageContext(
                code="de",
                source="default",
                confidence=1.0,
                switched_from=None,
                request_id=envelope.request_id,
            ),
        )

    kernel.build = AsyncMock(side_effect=_build)
    return kernel


class TestTypingKeepaliveFunction:
    """Unit-Tests für die _typing_keepalive Coroutine."""

    async def test_keepalive_sends_typing_periodically(self) -> None:
        """Keepalive sendet Typing-Action mehrmals bevor er gecancelt wird."""
        mock_chat = MagicMock()
        mock_chat.send_chat_action = AsyncMock()

        # Task mit kurzem Intervall starten
        task = asyncio.create_task(_typing_keepalive(mock_chat, interval=0.05))

        # Warten bis mindestens 2 Calls passiert sein sollten
        await asyncio.sleep(0.25)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Mindestens 2 Aufrufe erwartet (0.25s / 0.05s = 5, grosszuegige Toleranz
        # für Windows-Timing-Jitter unter Last)
        assert mock_chat.send_chat_action.call_count >= 2

    async def test_keepalive_cancels_cleanly(self) -> None:
        """Keepalive beendet sich sauber bei CancelledError ohne Exception."""
        mock_chat = MagicMock()
        mock_chat.send_chat_action = AsyncMock()

        task = asyncio.create_task(_typing_keepalive(mock_chat, interval=0.05))

        # Kurz laufen lassen, dann canceln
        await asyncio.sleep(0.08)
        task.cancel()

        # Darf keine Exception raisen
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Task ist beendet
        assert task.done()

    async def test_keepalive_survives_api_error(self) -> None:
        """Keepalive läuft weiter auch wenn send_chat_action fehlschlägt."""
        call_count = 0

        async def flaky_send_action(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Netzwerkfehler simuliert")
            # Ab dem 3. Aufruf kein Fehler mehr

        mock_chat = MagicMock()
        mock_chat.send_chat_action = flaky_send_action

        task = asyncio.create_task(_typing_keepalive(mock_chat, interval=0.03))

        # Lang genug laufen lassen, dass sowohl Fehler als auch Erfolg passieren
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Task hat trotz Fehler weitergearbeitet (mehr als 2 Aufrufe)
        assert call_count >= 4

    async def test_keepalive_uses_correct_chat_action(self) -> None:
        """Keepalive sendet ChatAction.TYPING als Argument."""
        from telegram.constants import ChatAction

        mock_chat = MagicMock()
        mock_chat.send_chat_action = AsyncMock()

        task = asyncio.create_task(_typing_keepalive(mock_chat, interval=0.05))
        await asyncio.sleep(0.08)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Prüfen dass ChatAction.TYPING übergeben wurde
        mock_chat.send_chat_action.assert_called_with(ChatAction.TYPING)


class TestTypingKeepaliveIntegration:
    """Integration: handle_message startet und cancelt Keepalive korrekt."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_handle_message_starts_and_cancels_keepalive(self) -> None:
        """handle_message startet Keepalive vor LLM-Call und cancelt danach."""
        from infrastructure.conversation_storage import _reset_all_for_tests
        from infrastructure.providers.base import ProviderResponse
        from presentation.handlers import handle_message

        _reset_all_for_tests()

        # Chat-Mock mit Tracking für send_chat_action
        typing_calls: list[str] = []

        async def track_chat_action(action):
            typing_calls.append(str(action))

        # ChatService-Mock der etwas dauert (simuliert LLM-Latenz)
        mock_router = MagicMock()

        async def slow_route(**kwargs):
            # Simuliere kurze LLM-Latenz
            await asyncio.sleep(0.15)
            return ProviderResponse(
                text="Antwort",
                duration_seconds=0.15,
                provider_name="claude",
            )

        mock_router.route = slow_route

        from application.chat_service import ChatService

        svc = ChatService(provider_router=mock_router, memory_service=None)

        # Update erstellen
        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 1
        update.effective_user.username = "testuser"
        update.effective_chat = MagicMock()
        update.effective_chat.id = 10
        update.effective_chat.type = "private"
        update.effective_chat.send_chat_action = AsyncMock(
            side_effect=track_chat_action
        )
        update.message = MagicMock()
        update.message.text = "Hallo"
        update.message.reply_to_message = None
        update.message.reply_text = AsyncMock()

        context = MagicMock()
        context.args = []
        context.bot = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.application = MagicMock()
        context.application.bot_data = {
            "chat_service": svc,
            "system_prompt": "Test prompt.",
            "memory_service": None,
            "context_kernel": _make_mock_context_kernel(),
        }

        # Keepalive mit kurzem Intervall patchen damit Test schnell läuft
        with patch("presentation.handlers.TYPING_KEEPALIVE_INTERVAL_SECONDS", 0.04):
            await handle_message(update, context)

        # Mindestens ein Re-Trigger muss passiert sein (0.15s / 0.04s = 3+)
        # Der erste send_chat_action geht über context.bot, die Re-Triggers
        # gehen über update.effective_chat.send_chat_action
        assert len(typing_calls) >= 2

    async def test_keepalive_cancelled_on_llm_error(self) -> None:
        """Keepalive wird auch bei LLM-Fehler sauber gecancelt (kein Leak)."""
        from infrastructure.conversation_storage import _reset_all_for_tests
        from presentation.handlers import handle_message

        _reset_all_for_tests()

        # ChatService-Mock der einen Fehler wirft
        mock_router = MagicMock()

        async def failing_route(**kwargs):
            await asyncio.sleep(0.05)
            raise RuntimeError("LLM nicht erreichbar")

        mock_router.route = failing_route

        from application.chat_service import ChatService

        svc = ChatService(provider_router=mock_router, memory_service=None)

        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 1
        update.effective_user.username = "testuser"
        update.effective_chat = MagicMock()
        update.effective_chat.id = 10
        update.effective_chat.type = "private"
        update.effective_chat.send_chat_action = AsyncMock()
        update.message = MagicMock()
        update.message.text = "Test"
        update.message.reply_to_message = None
        update.message.reply_text = AsyncMock()

        context = MagicMock()
        context.args = []
        context.bot = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.application = MagicMock()
        context.application.bot_data = {
            "chat_service": svc,
            "system_prompt": "Test prompt.",
            "memory_service": None,
            "context_kernel": _make_mock_context_kernel(),
        }

        with patch("presentation.handlers.TYPING_KEEPALIVE_INTERVAL_SECONDS", 0.02):
            # Der Handler fängt Errors intern, also checken wir dass er
            # nicht hängt und kein Task leaked
            try:
                await handle_message(update, context)
            except RuntimeError:
                pass  # Manche ChatService-Impls leaken den Error durch

        # Kurz warten, dann prüfen dass kein Background-Task mehr läuft
        await asyncio.sleep(0.1)
        # Kein hängender Task: send_chat_action darf nicht mehr aufgerufen werden
        # nach dem wir gewartet haben
        call_count_after = update.effective_chat.send_chat_action.call_count
        await asyncio.sleep(0.1)
        assert update.effective_chat.send_chat_action.call_count == call_count_after


class TestTypingKeepaliveConstant:
    """Test: Konstante ist korrekt definiert."""

    def test_interval_is_four_seconds(self) -> None:
        """Default-Intervall ist 4.0 Sekunden."""
        assert TYPING_KEEPALIVE_INTERVAL_SECONDS == 4.0
