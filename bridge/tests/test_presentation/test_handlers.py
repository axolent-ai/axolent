"""Tests für presentation.handlers: Telegram Command-Handler.

Testet /save, /lang, /reset Commands mit gemockten Telegram-Objekten.
Kein echter Bot, keine echten API-Aufrufe.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from filelock import FileLock

from application.chat_service import ChatService
from infrastructure.conversation_storage import _reset_all_for_tests
from infrastructure.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Räumt Conversation-Storage vor jedem Test auf."""
    _reset_all_for_tests()


def _make_mock_chat_service(
    route_return: ProviderResponse | None = None,
) -> ChatService:
    """Erstellt einen ChatService mit gemocktem ProviderRouter."""
    mock_router = MagicMock()
    mock_router.route = AsyncMock(
        return_value=route_return
        or ProviderResponse(
            text="Antwort von Claude",
            duration_seconds=1.0,
            provider_name="claude",
        )
    )
    return ChatService(provider_router=mock_router, memory_service=None)


def _make_update(user_id: int = 1, chat_id: int = 10, text: str = "") -> MagicMock:
    """Erstellt ein gemocktes Telegram-Update."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.reply_to_message = None
    return update


def _make_context(
    args: list[str] | None = None,
    chat_service: ChatService | None = None,
    system_prompt: str = "Test system prompt.",
    memory_service: object | None = None,
) -> MagicMock:
    """Erstellt einen gemockten Telegram-Context mit bot_data."""
    context = MagicMock()
    context.args = args or []
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()
    # bot_data für ChatService-Injection + system_prompt + memory_service
    svc = chat_service or _make_mock_chat_service()
    context.application = MagicMock()
    context.application.bot_data = {
        "chat_service": svc,
        "system_prompt": system_prompt,
        "memory_service": memory_service,
    }
    return context


class TestHandleSaveCommand:
    """Tests für /save Command."""

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
            # Whitelist-Bypass für Handler-Tests
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
    """Tests für /lang Command."""

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
    """Tests für /reset Command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_command_clears_history(self) -> None:
        """/reset loescht History und Language, speichert dann die Reset-Bestaetigung."""
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

        # Alte History geloescht, aber Reset-Bestaetigung gespeichert
        history = await get_history(1, 10)
        lang = await get_language(1, 10)
        assert len(history) == 1
        assert history[0].role == "assistant"
        assert (
            "zurueckgesetzt" in history[0].content.lower()
            or "frisch" in history[0].content.lower()
        )
        assert lang is None

        # Bestaetigung gesendet
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "zurueckgesetzt" in reply_text.lower() or "frisch" in reply_text.lower()


class TestStartCommandHistory:
    """Tests: /start speichert Bot-Antwort in Conversation-History (Fix A)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_start_command_saves_to_history(self) -> None:
        """/start speichert START_TEXT als assistant-Turn in History."""
        from infrastructure.conversation_storage import get_history
        from presentation.handlers import START_TEXT, handle_start_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context()

        await handle_start_command(update, context)

        history = await get_history(1, 10)
        assert len(history) == 1
        assert history[0].role == "assistant"
        assert history[0].content == START_TEXT

    async def test_help_command_saves_to_history(self) -> None:
        """/help speichert HELP_TEXT als assistant-Turn in History."""
        from infrastructure.conversation_storage import get_history
        from presentation.handlers import HELP_TEXT, handle_help_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context()

        await handle_help_command(update, context)

        history = await get_history(1, 10)
        assert len(history) == 1
        assert history[0].role == "assistant"
        assert history[0].content == HELP_TEXT


class TestReplyToContext:
    """Tests: Telegram-Reply-To wird als Kontext extrahiert (Fix B)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    def _make_reply_chat_service(self) -> tuple[ChatService, MagicMock]:
        """Erstellt ChatService mit mockbarem Router für Reply-Tests."""
        mock_router = MagicMock()
        mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="Antwort mit Kontext",
                duration_seconds=0.5,
                provider_name="claude",
            )
        )
        svc = ChatService(provider_router=mock_router, memory_service=None)
        return svc, mock_router

    async def test_reply_to_context_passed_to_provider(self) -> None:
        """Wenn User auf Bot-Nachricht antwortet, wird Reply-Text im Prompt eingefügt."""
        from presentation.handlers import handle_message

        svc, mock_router = self._make_reply_chat_service()

        update = _make_update(user_id=1, chat_id=10, text="Was bedeutet das?")
        # Simuliere Reply auf eine Bot-Nachricht
        reply_msg = MagicMock()
        reply_msg.text = "Tipp: Du kannst Bot-Nachrichten als Bookmark speichern."
        update.message.reply_to_message = reply_msg

        context = _make_context(chat_service=svc, system_prompt="Test prompt.")

        await handle_message(update, context)

        # Der Prompt an den Router muss den Reply-Kontext enthalten
        call_args = mock_router.route.call_args
        prompt_sent = call_args.kwargs.get("prompt", "")
        assert "REPLIED TO PREVIOUS BOT MESSAGE" in prompt_sent
        assert "Bookmark speichern" in prompt_sent
        assert "Was bedeutet das?" in prompt_sent

    async def test_no_reply_to_sends_plain_text(self) -> None:
        """Ohne Reply wird nur der normale Text gesendet."""
        from presentation.handlers import handle_message

        svc, mock_router = self._make_reply_chat_service()

        update = _make_update(user_id=1, chat_id=10, text="Einfache Frage")
        update.message.reply_to_message = None

        context = _make_context(chat_service=svc, system_prompt="Test prompt.")

        await handle_message(update, context)

        call_args = mock_router.route.call_args
        prompt_sent = call_args.kwargs.get("prompt", "")
        assert "REPLIED TO PREVIOUS BOT MESSAGE" not in prompt_sent
        assert "Einfache Frage" in prompt_sent


class TestAuditLoggingReset:
    """Tests: /reset schreibt Audit-Log-Eintrag."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_reset_writes_audit(self, mock_audit: MagicMock) -> None:
        """/reset schreibt einen Audit-Eintrag mit action='reset'."""
        from presentation.handlers import handle_reset_command

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context()

        await handle_reset_command(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["event_type"] == "command"
        assert entry["action"] == "reset"
        assert entry["user_id"] == 42
        assert entry["chat_id"] == 99
        assert entry["success"] is True


class TestAuditLoggingLang:
    """Tests: /lang schreibt Audit-Log-Eintrag."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_lang_change_writes_audit(self, mock_audit: MagicMock) -> None:
        """/lang en schreibt Audit mit alter und neuer Sprache."""
        from presentation.handlers import handle_lang_command

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["en"])

        await handle_lang_command(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["event_type"] == "command"
        assert entry["action"] == "lang_change"
        assert entry["user_id"] == 42
        assert "en" in entry["details"]

    @patch("application.audit_service.write_audit_log")
    async def test_lang_invalid_no_audit(self, mock_audit: MagicMock) -> None:
        """/lang xyz (ungueltig) schreibt KEINEN Audit-Eintrag."""
        from presentation.handlers import handle_lang_command

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["xyz"])

        await handle_lang_command(update, context)

        mock_audit.assert_not_called()


class TestAuditLoggingSave:
    """Tests: /save schreibt Audit-Log-Eintrag."""

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
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
        ]
        for p in self._patches:
            p.start()

        yield  # type: ignore[misc]

        for p in self._patches:
            p.stop()

    @patch("application.audit_service.write_audit_log")
    async def test_save_bookmark_writes_audit(self, mock_audit: MagicMock) -> None:
        """/save als Reply erstellt Audit-Eintrag mit action='save_bookmark'."""
        from presentation.handlers import handle_save_command

        update = _make_update(user_id=1, chat_id=10)
        reply_msg = MagicMock()
        reply_msg.message_id = 50
        reply_msg.text = "Bot-Antwort"
        update.message.reply_to_message = reply_msg

        context = _make_context()

        await handle_save_command(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "save_bookmark"
        assert entry["user_id"] == 1
        assert entry["entry_id"] == "msg_50"


class TestAuditLoggingBookmarks:
    """Tests: /bookmarks schreibt Audit-Log-Eintrag."""

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
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
        ]
        for p in self._patches:
            p.start()

        yield  # type: ignore[misc]

        for p in self._patches:
            p.stop()

    @patch("application.audit_service.write_audit_log")
    async def test_bookmarks_empty_writes_audit(self, mock_audit: MagicMock) -> None:
        """/bookmarks ohne Bookmarks schreibt Audit mit '0 bookmarks'."""
        from presentation.handlers import handle_bookmarks_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(args=[])

        await handle_bookmarks_command(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "list_bookmarks"
        assert "0" in entry["details"]


class TestAuditLoggingRemember:
    """Tests: /remember schreibt Audit-Log-Eintrag."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_remember_writes_audit(self, mock_audit: MagicMock) -> None:
        """/remember text schreibt Audit mit action='remember' und entry_id."""
        from presentation.handlers import handle_remember_command

        mock_memory = MagicMock()
        mock_memory.remember_episodic = MagicMock(return_value="ep_test123")

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["mein", "Test"], memory_service=mock_memory)

        await handle_remember_command(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "remember"
        assert entry["user_id"] == 42
        assert entry["entry_id"] == "ep_test123"


class TestAuditLoggingForget:
    """Tests: /forget schreibt Audit-Log-Eintrag."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_forget_success_writes_audit(self, mock_audit: MagicMock) -> None:
        """/forget ep_123 (gefunden) schreibt Audit mit success=True."""
        from presentation.handlers import handle_forget_command

        mock_memory = MagicMock()
        mock_memory.forget = MagicMock(return_value=True)

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["ep_123"], memory_service=mock_memory)

        await handle_forget_command(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "forget"
        assert entry["entry_id"] == "ep_123"
        assert entry["success"] is True

    @patch("application.audit_service.write_audit_log")
    async def test_forget_not_found_writes_audit(self, mock_audit: MagicMock) -> None:
        """/forget ep_999 (nicht gefunden) schreibt Audit mit success=False."""
        from presentation.handlers import handle_forget_command

        mock_memory = MagicMock()
        mock_memory.forget = MagicMock(return_value=False)

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["ep_999"], memory_service=mock_memory)

        await handle_forget_command(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "forget"
        assert entry["entry_id"] == "ep_999"
        assert entry["success"] is False
