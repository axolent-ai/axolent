"""Tests fuer presentation.handlers: Telegram Command-Handler.

Testet /save, /lang, /reset Commands mit gemockten Telegram-Objekten.
Kein echter Bot, keine echten API-Aufrufe.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from filelock import FileLock

from infrastructure.conversation_storage import _histories, _languages


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Raeumt Conversation-Storage vor jedem Test auf."""
    _histories.clear()
    _languages.clear()


def _make_update(user_id: int = 1, chat_id: int = 10, text: str = "") -> MagicMock:
    """Erstellt ein gemocktes Telegram-Update."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.reply_to_message = None
    return update


def _make_context(args: list[str] | None = None) -> MagicMock:
    """Erstellt einen gemockten Telegram-Context."""
    context = MagicMock()
    context.args = args or []
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()
    return context


class TestHandleSaveCommand:
    """Tests fuer /save Command."""

    @pytest.fixture(autouse=True)
    def _isolate_bookmark_storage(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage."""
        bm_path = tmp_path / "bookmarks.jsonl"
        lock_path = str(bm_path) + ".lock"
        new_lock = FileLock(lock_path)

        self._patches = [
            patch("infrastructure.bookmark_storage.BOOKMARKS_PATH", bm_path),
            patch("infrastructure.bookmark_storage._BM_LOCK_PATH", lock_path),
            patch("infrastructure.bookmark_storage._BM_LOCK", new_lock),
            # Whitelist-Bypass fuer Handler-Tests
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
        ]
        for p in self._patches:
            p.start()

        yield  # type: ignore[misc]

        for p in self._patches:
            p.stop()

    async def test_save_command_creates_bookmark(self) -> None:
        """/save als Reply auf Bot-Nachricht erstellt einen Bookmark."""
        from presentation.handlers import handle_save_command

        update = _make_update(user_id=1, chat_id=10)
        # Simuliere Reply auf eine Bot-Nachricht
        reply_msg = MagicMock()
        reply_msg.message_id = 50
        reply_msg.text = "Bot-Antwort zum Speichern"
        update.message.reply_to_message = reply_msg

        context = _make_context()
        await handle_save_command(update, context)

        # Bestaetigung muss gesendet worden sein
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "gespeichert" in reply_text.lower() or "Bookmark" in reply_text

    async def test_save_command_without_reply(self) -> None:
        """/save ohne Reply sendet Hilfe-Text."""
        from presentation.handlers import handle_save_command

        update = _make_update(user_id=1, chat_id=10)
        update.message.reply_to_message = None

        context = _make_context()
        await handle_save_command(update, context)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "antworte" in reply_text.lower() or "reply" in reply_text.lower()


class TestHandleLangCommand:
    """Tests fuer /lang Command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_lang_command_sets_sticky_language(self) -> None:
        """/lang en setzt die Sprache korrekt."""
        from infrastructure.conversation_storage import get_language
        from presentation.handlers import handle_lang_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(args=["en"])

        await handle_lang_command(update, context)

        lang = await get_language(1, 10)
        assert lang == "en"

        # Bestaetigung gesendet
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "English" in reply_text or "en" in reply_text

    async def test_lang_command_no_args_shows_help(self) -> None:
        """/lang ohne Argument zeigt Hilfe."""
        from presentation.handlers import handle_lang_command

        update = _make_update()
        context = _make_context(args=[])

        await handle_lang_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Benutzung" in reply_text or "/lang" in reply_text

    async def test_lang_command_invalid_code(self) -> None:
        """/lang xyz mit ungueltigem Code gibt Fehler."""
        from presentation.handlers import handle_lang_command

        update = _make_update()
        context = _make_context(args=["xyz"])

        await handle_lang_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Unbekannte" in reply_text or "unknown" in reply_text.lower()


class TestHandleResetCommand:
    """Tests fuer /reset Command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_command_clears_history(self) -> None:
        """/reset loescht History und Language."""
        from domain.conversation import ConversationTurn
        from infrastructure.conversation_storage import (
            get_history,
            get_language,
            save_turn,
            set_language,
        )
        from presentation.handlers import handle_reset_command

        # Setup: History und Language setzen
        await save_turn(1, 10, ConversationTurn(role="user", content="Old msg"))
        await set_language(1, 10, "fr")

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context()

        await handle_reset_command(update, context)

        # Alles geloescht
        history = await get_history(1, 10)
        lang = await get_language(1, 10)
        assert history == []
        assert lang is None

        # Bestaetigung gesendet
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "zurückgesetzt" in reply_text.lower() or "reset" in reply_text.lower()
