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
    persistent_provider: object | None = None,
    process_pool: object | None = None,
    rate_limiter: object | None = None,
    bookmark_service: object | None = None,
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
        "persistent_provider": persistent_provider,
        "process_pool": process_pool,
        "rate_limiter": rate_limiter,
        "bookmark_service": bookmark_service,
    }
    return context


class TestHandleSaveCommand:
    """Tests für /save Command."""

    @pytest.fixture(autouse=True)
    def _isolate_bookmark_storage(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage und erstellt BookmarkService."""
        bm_path = tmp_path / "bookmarks.jsonl"
        lock_path = str(bm_path) + ".lock"
        new_lock = FileLock(lock_path)

        from application.bookmark_service import BookmarkService
        from infrastructure.bookmark_storage import JsonlBookmarkStorageAdapter

        self._bookmark_svc = BookmarkService(storage=JsonlBookmarkStorageAdapter())

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

        context = _make_context(bookmark_service=self._bookmark_svc)
        await handle_save_command(update, context)

        # Bestätigung muss gesendet worden sein
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "gespeichert" in reply_text.lower() or "Bookmark" in reply_text

    async def test_save_command_without_reply(self) -> None:
        """/save ohne Reply sendet Hilfe-Text."""
        from presentation.handlers import handle_save_command

        update = _make_update(user_id=1, chat_id=10)
        update.message.reply_to_message = None

        context = _make_context(bookmark_service=self._bookmark_svc)
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
        """/reset löscht History und Language, speichert dann die Reset-Bestätigung."""
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

        # Alte History gelöscht, aber Reset-Bestätigung gespeichert
        history = await get_history(1, 10)
        lang = await get_language(1, 10)
        assert len(history) == 1
        assert history[0].role == "assistant"
        assert (
            "zurückgesetzt" in history[0].content.lower()
            or "frisch" in history[0].content.lower()
        )
        assert lang is None

        # Bestaetigung gesendet
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "zurückgesetzt" in reply_text.lower() or "frisch" in reply_text.lower()


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

    async def test_help_text_contains_all_commands(self) -> None:
        """/help Text enthält alle tatsächlich existierenden Commands."""
        from presentation.handlers import HELP_TEXT

        expected_commands = [
            "/save",
            "/bookmarks",
            "/remember",
            "/forget",
            "/memory",
            "/usage",
            "/setlimit",
            "/reset",
            "/lang",
            "/start",
            "/help",
            "/debate",
        ]
        for cmd in expected_commands:
            assert cmd in HELP_TEXT, f"Command {cmd} fehlt im HELP_TEXT"

    async def test_help_text_does_not_contain_nonexistent_commands(self) -> None:
        """/help Text enthält KEINE nicht-existenten Commands."""
        from presentation.handlers import HELP_TEXT

        nonexistent = ["/unsave", "/delete", "/clear", "/settings", "/config"]
        for cmd in nonexistent:
            assert cmd not in HELP_TEXT, f"Nicht-existenter Command {cmd} im HELP_TEXT"

    async def test_help_text_is_structured(self) -> None:
        """/help Text hat die gewuenschte Struktur mit Kategorien."""
        from presentation.handlers import HELP_TEXT

        assert "Multi-AI" in HELP_TEXT
        assert "Bookmarks" in HELP_TEXT
        assert "Memory" in HELP_TEXT
        assert "Limits" in HELP_TEXT or "Profile" in HELP_TEXT
        assert "Konversation" in HELP_TEXT
        assert "Ohne Slash" in HELP_TEXT


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
        """Patcht Bookmark-Storage und erstellt BookmarkService."""
        bm_path = tmp_path / "bookmarks.jsonl"
        lock_path = str(bm_path) + ".lock"
        new_lock = FileLock(lock_path)

        from application.bookmark_service import BookmarkService
        from infrastructure.bookmark_storage import JsonlBookmarkStorageAdapter

        self._bookmark_svc = BookmarkService(storage=JsonlBookmarkStorageAdapter())

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

        context = _make_context(bookmark_service=self._bookmark_svc)

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
        """Patcht Bookmark-Storage und erstellt BookmarkService."""
        bm_path = tmp_path / "bookmarks.jsonl"
        lock_path = str(bm_path) + ".lock"
        new_lock = FileLock(lock_path)

        from application.bookmark_service import BookmarkService
        from infrastructure.bookmark_storage import JsonlBookmarkStorageAdapter

        self._bookmark_svc = BookmarkService(storage=JsonlBookmarkStorageAdapter())

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
        context = _make_context(args=[], bookmark_service=self._bookmark_svc)

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
    async def test_forget_success_shows_reset_hint(self, mock_audit: MagicMock) -> None:
        """/forget ep_123 zeigt Hinweis auf /reset für History-Bereinigung."""
        from presentation.handlers import handle_forget_command

        mock_memory = MagicMock()
        mock_memory.forget = MagicMock(return_value=True)

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["ep_123"], memory_service=mock_memory)

        await handle_forget_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "ep_123" in reply_text
        assert "/reset" in reply_text
        assert "Hinweis" in reply_text

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


class TestPrivacyGuardHandleMessage:
    """Tests: handle_message blockiert Gruppen-Chats (P0-1C)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_group_chat_blocked(self) -> None:
        """handle_message in Gruppe sendet Privacy-Block-Meldung."""
        from presentation.handlers import handle_message

        update = _make_update(user_id=1, chat_id=10, text="Hallo")
        update.effective_chat.type = "group"

        context = _make_context()
        await handle_message(update, context)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "privaten Chat" in reply_text

    async def test_private_chat_allowed(self) -> None:
        """handle_message im privaten Chat funktioniert normal."""
        from presentation.handlers import handle_message

        update = _make_update(user_id=1, chat_id=10, text="Hallo")
        update.effective_chat.type = "private"

        context = _make_context()
        # Kein persistent_provider => Legacy-Fallback
        await handle_message(update, context)

        # Sollte KEINE Privacy-Block-Meldung sein
        if update.message.reply_text.called:
            reply_text = update.message.reply_text.call_args[0][0]
            assert "privaten Chat" not in reply_text


class TestStreamingErrorRedaction:
    """Tests: Streaming-Fehler werden redacted an User gesendet (P0-2)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("presentation.handlers.write_raw_audit")
    async def test_error_event_shows_generic_message(
        self, mock_audit: MagicMock
    ) -> None:
        """Error-Event mit sensiblem Text zeigt nur generische Meldung mit ref."""
        from infrastructure.claude_process_pool import StreamEvent
        from presentation.handlers import _handle_message_streaming

        # Mock persistent_provider der ein Error-Event liefert
        mock_provider = MagicMock()
        mock_provider.is_available = MagicMock(return_value=True)

        # Mock chat_service.process_user_message_streaming
        mock_svc = _make_mock_chat_service()

        async def mock_stream(**kwargs):
            yield StreamEvent(
                event_type="error",
                text="/secret/path/to/file.py: PermissionError traceback",
                raw={"error": {"message": "secret stacktrace"}},
                is_final=True,
            )

        mock_svc.process_user_message_streaming = mock_stream

        # Mock pool
        mock_pool = MagicMock()
        mock_managed = MagicMock()
        mock_managed.pid = 12345

        async def mock_get_or_create(user_id, chat_id):
            return mock_managed, True

        mock_pool.get_or_create = mock_get_or_create

        update = _make_update(user_id=42, chat_id=99, text="Test")
        context = _make_context(
            chat_service=mock_svc,
            persistent_provider=mock_provider,
            process_pool=mock_pool,
        )

        # Mock create_streaming_message
        mock_msg = AsyncMock()
        mock_msg.edit_text = AsyncMock()
        mock_msg.chat = MagicMock()

        with patch(
            "presentation.handlers.create_streaming_message",
            return_value=mock_msg,
        ):
            await _handle_message_streaming(
                update=update,
                context=context,
                chat_service=mock_svc,
                persistent_provider=mock_provider,
                user_id=42,
                chat_id=99,
                username="testuser",
                text="Test",
                reply_to_text=None,
            )

        # Der User-facing Text darf KEINEN Pfad/Stacktrace enthalten
        edit_calls = mock_msg.edit_text.call_args_list
        for call in edit_calls:
            text_sent = call[0][0]
            assert "/secret/path" not in text_sent
            assert "PermissionError" not in text_sent
            assert "traceback" not in text_sent.lower()

        # Muss "ref:" enthalten
        last_edit_text = edit_calls[-1][0][0]
        assert "ref:" in last_edit_text

    @patch("presentation.handlers.write_raw_audit")
    async def test_runtime_error_shows_generic_message(
        self, mock_audit: MagicMock
    ) -> None:
        """RuntimeError mit Path-Info zeigt nur generische Meldung."""
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()

        async def mock_stream(**kwargs):
            raise RuntimeError("C:\\Users\\secret\\pipe_broken.txt")

        mock_svc.process_user_message_streaming = mock_stream

        mock_pool = MagicMock()
        mock_managed = MagicMock()
        mock_managed.pid = 12345

        async def mock_get_or_create(user_id, chat_id):
            return mock_managed, False

        mock_pool.get_or_create = mock_get_or_create

        update = _make_update(user_id=42, chat_id=99, text="Test")
        context = _make_context(
            chat_service=mock_svc,
            persistent_provider=mock_provider,
            process_pool=mock_pool,
        )

        mock_msg = AsyncMock()
        mock_msg.edit_text = AsyncMock()
        mock_msg.chat = MagicMock()

        with patch(
            "presentation.handlers.create_streaming_message",
            return_value=mock_msg,
        ):
            await _handle_message_streaming(
                update=update,
                context=context,
                chat_service=mock_svc,
                persistent_provider=mock_provider,
                user_id=42,
                chat_id=99,
                username="testuser",
                text="Test",
                reply_to_text=None,
            )

        # Letzter Edit-Text darf keinen Pfad enthalten
        last_edit_text = mock_msg.edit_text.call_args_list[-1][0][0]
        assert "secret" not in last_edit_text
        assert "pipe_broken" not in last_edit_text
        assert "ref:" in last_edit_text


class TestStreamingAuditEntries:
    """Tests: Streaming erzeugt 2 Audit-Einträge (started + completed/crashed)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("presentation.handlers.write_raw_audit")
    async def test_successful_stream_writes_two_audit_entries(
        self, mock_audit: MagicMock
    ) -> None:
        """Erfolgreicher Stream: 'stream_started' + save_streaming_result Audit."""
        from infrastructure.claude_process_pool import StreamEvent
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()
        mock_svc.save_streaming_result = AsyncMock()

        async def _stream_events():
            yield StreamEvent(event_type="content_delta", text="Hallo")
            yield StreamEvent(
                event_type="result", full_text="Hallo Welt", is_final=True
            )

        async def mock_stream(**kwargs):
            return _stream_events(), 3  # 3 Memory-Einträge geladen

        mock_svc.process_user_message_streaming = mock_stream

        mock_pool = MagicMock()
        mock_managed = MagicMock()
        mock_managed.pid = 999

        async def mock_get_or_create(user_id, chat_id):
            return mock_managed, True

        mock_pool.get_or_create = mock_get_or_create

        update = _make_update(user_id=42, chat_id=99, text="Frage")
        context = _make_context(
            chat_service=mock_svc,
            persistent_provider=mock_provider,
            process_pool=mock_pool,
        )

        mock_msg = AsyncMock()
        mock_msg.edit_text = AsyncMock()
        mock_msg.chat = MagicMock()

        with patch(
            "presentation.handlers.create_streaming_message",
            return_value=mock_msg,
        ):
            await _handle_message_streaming(
                update=update,
                context=context,
                chat_service=mock_svc,
                persistent_provider=mock_provider,
                user_id=42,
                chat_id=99,
                username="testuser",
                text="Frage",
                reply_to_text=None,
            )

        # Mindestens 1 Audit-Eintrag (stream_started)
        assert mock_audit.call_count >= 1
        first_entry = mock_audit.call_args_list[0][0][0]
        assert first_entry["event_type"] == "stream_started"
        assert first_entry["user_id"] == 42

        # save_streaming_result wurde aufgerufen (= zweiter Audit)
        mock_svc.save_streaming_result.assert_called_once()
        save_kwargs = mock_svc.save_streaming_result.call_args
        assert save_kwargs.kwargs.get("was_cold") is True
        assert save_kwargs.kwargs.get("subprocess_pid") == 999
        # memory_entries_loaded muss korrekt durchgereicht werden
        assert save_kwargs.kwargs.get("memory_entries_loaded") == 3

    @patch("presentation.handlers.write_raw_audit")
    async def test_crashed_stream_writes_error_audit(
        self, mock_audit: MagicMock
    ) -> None:
        """Crash mid-stream: 'stream_started' + 'stream_error' Audit."""
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()

        async def mock_stream(**kwargs):
            async def _crash_gen():
                if True:
                    raise RuntimeError("unexpected crash")
                yield  # pragma: no cover

            return _crash_gen(), 0

        mock_svc.process_user_message_streaming = mock_stream

        mock_pool = MagicMock()
        mock_managed = MagicMock()
        mock_managed.pid = 888

        async def mock_get_or_create(user_id, chat_id):
            return mock_managed, False

        mock_pool.get_or_create = mock_get_or_create

        update = _make_update(user_id=42, chat_id=99, text="Crash")
        context = _make_context(
            chat_service=mock_svc,
            persistent_provider=mock_provider,
            process_pool=mock_pool,
        )

        mock_msg = AsyncMock()
        mock_msg.edit_text = AsyncMock()
        mock_msg.chat = MagicMock()

        with patch(
            "presentation.handlers.create_streaming_message",
            return_value=mock_msg,
        ):
            await _handle_message_streaming(
                update=update,
                context=context,
                chat_service=mock_svc,
                persistent_provider=mock_provider,
                user_id=42,
                chat_id=99,
                username="testuser",
                text="Crash",
                reply_to_text=None,
            )

        # Mindestens 2 Audit-Einträge: stream_started + stream_error
        assert mock_audit.call_count >= 2
        event_types = [c[0][0]["event_type"] for c in mock_audit.call_args_list]
        assert "stream_started" in event_types
        assert "stream_error" in event_types

        # stream_error muss error_id haben
        error_entries = [
            c[0][0]
            for c in mock_audit.call_args_list
            if c[0][0]["event_type"] == "stream_error"
        ]
        assert error_entries[0]["error_id"] != ""


class TestOuterExceptionCoverage:
    """Tests: Outer Exception in _handle_message_streaming (P1-8)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("presentation.handlers.write_raw_audit")
    async def test_create_streaming_message_exception(
        self, mock_audit: MagicMock
    ) -> None:
        """Wenn create_streaming_message wirft, bekommt User Fehlermeldung."""
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()
        mock_pool = MagicMock()

        update = _make_update(user_id=42, chat_id=99, text="Test")
        context = _make_context(
            chat_service=mock_svc,
            persistent_provider=mock_provider,
            process_pool=mock_pool,
        )

        with patch(
            "presentation.handlers.create_streaming_message",
            side_effect=Exception("Telegram API down"),
        ):
            # Sollte NICHT crashen
            await _handle_message_streaming(
                update=update,
                context=context,
                chat_service=mock_svc,
                persistent_provider=mock_provider,
                user_id=42,
                chat_id=99,
                username="testuser",
                text="Test",
                reply_to_text=None,
            )

        # User muss eine Nachricht bekommen
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "ref:" in reply_text

        # Audit: stream_started + stream_error
        event_types = [c[0][0]["event_type"] for c in mock_audit.call_args_list]
        assert "stream_started" in event_types
        assert "stream_error" in event_types


class TestHandleMessageRateLimit:
    """Tests für Rate-Limiting im handle_message Handler (C-2)."""

    @pytest.fixture(autouse=True)
    def _bypass_whitelist(self) -> None:
        """Whitelist-Bypass für Handler-Tests."""
        self._patches = [
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
        ]
        for p in self._patches:
            p.start()
        yield  # type: ignore[misc]
        for p in self._patches:
            p.stop()

    async def test_rate_limit_blocks_message(self) -> None:
        """Rate-Limited User bekommt Meldung, kein LLM-Call."""
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_message

        limiter = RateLimiter()
        # Alle Minute-Tokens verbrauchen (Normal: 25/min)
        normal_min = PROFILES["normal"]["per_minute"]
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        update = _make_update(user_id=1, chat_id=10, text="Hallo")
        context = _make_context(rate_limiter=limiter)

        with patch("presentation.handlers.write_raw_audit") as mock_audit:
            await handle_message(update, context)

        # User bekommt Limit-Meldung
        # Beachte: 70%-Warnung kann vorher reply_text aufrufen
        calls = update.message.reply_text.call_args_list
        # Letzter oder einziger Call muss Limit-Meldung sein
        limit_reply = calls[-1][0][0]
        assert "Limit" in limit_reply

        # Kein Typing-Indicator gesendet (kein LLM-Call)
        context.bot.send_chat_action.assert_not_called()

        # Audit-Log enthält rate_limit_exceeded
        mock_audit.assert_called()
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["event_type"] == "rate_limit_exceeded"
        assert audit_entry["user_id"] == 1
        assert audit_entry["profile"] == "normal"
        assert audit_entry["period"] == "minute"

    async def test_rate_limit_allows_normal_request(self) -> None:
        """Unter dem Limit: normaler LLM-Call Ablauf."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_message

        limiter = RateLimiter()

        update = _make_update(user_id=2, chat_id=20, text="Test")
        context = _make_context(rate_limiter=limiter)

        with patch("presentation.handlers.write_raw_audit"):
            await handle_message(update, context)

        # Typing-Indicator wurde gesendet (= LLM-Call wurde gestartet)
        context.bot.send_chat_action.assert_called()

    async def test_no_rate_limiter_in_context_allows(self) -> None:
        """Wenn kein RateLimiter in bot_data: normal durchlassen."""
        from presentation.handlers import handle_message

        update = _make_update(user_id=3, chat_id=30, text="Test")
        context = _make_context(rate_limiter=None)

        await handle_message(update, context)

        # Typing-Indicator gesendet = LLM-Pfad betreten
        context.bot.send_chat_action.assert_called()

    async def test_rate_limit_exceeded_shows_profile_info(self) -> None:
        """Rate-Limit-Meldung zeigt Profil-Info und Optionen."""
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_message

        limiter = RateLimiter()
        normal_min = PROFILES["normal"]["per_minute"]

        # Alle Minute-Tokens verbrauchen
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        update = _make_update(user_id=1, chat_id=10, text="Hallo")
        context = _make_context(rate_limiter=limiter)

        with patch("presentation.handlers.write_raw_audit") as mock_audit:
            await handle_message(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Normal-Profil" in reply_text
        assert "/usage" in reply_text
        assert "/setlimit" in reply_text

        # Audit enthält Profil- und Period-Info
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["profile"] == "normal"
        assert audit_entry["period"] == "minute"


class TestHandleUsageCommand:
    """Tests für /usage Command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_usage_shows_profile_and_limits(self) -> None:
        """/usage zeigt Profil und Limits."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_usage_command

        limiter = RateLimiter()
        # Ein paar Anfragen machen
        for _ in range(3):
            limiter.check_and_consume(user_id=1)

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(rate_limiter=limiter)

        await handle_usage_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Profil: Normal" in reply_text
        assert "Diese Minute" in reply_text
        assert "Diese Stunde" in reply_text
        assert "Heute" in reply_text
        assert "/setlimit" in reply_text

    async def test_usage_unlimited_profile(self, tmp_path: Path) -> None:
        """/usage zeigt Unlimited-Info wenn Profil unlimited."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_usage_command

        with patch(
            "application.rate_limiter._PROFILES_PATH",
            tmp_path / "user_profiles.jsonl",
        ):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=5, chat_id=5, profile="unlimited")

        update = _make_update(user_id=5, chat_id=5)
        context = _make_context(rate_limiter=limiter)

        await handle_usage_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Unlimited" in reply_text
        assert "Keine Limits" in reply_text

    async def test_usage_no_limiter_shows_error(self) -> None:
        """/usage ohne Rate-Limiter zeigt Fehlermeldung."""
        from presentation.handlers import handle_usage_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(rate_limiter=None)

        await handle_usage_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "nicht initialisiert" in reply_text


class TestHandleDebateCommand:
    """Tests fuer /debate Command (R10: Multi-AI-Debate)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_debate_without_args_shows_help(self) -> None:
        """/debate ohne Argumente zeigt Hilfe-Text."""
        from presentation.handlers import handle_debate_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(args=[])

        await handle_debate_command(update, context)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "/debate" in reply_text
        assert "Frage" in reply_text

    @patch("presentation.handlers.write_raw_audit")
    async def test_debate_with_question_calls_orchestrator(
        self, mock_audit: MagicMock
    ) -> None:
        """/debate mit Frage ruft DebateOrchestrator auf."""
        from application.debate_orchestrator import DebateResult
        from presentation.handlers import handle_debate_command

        mock_result = DebateResult(
            question="Was ist KI?",
            responses={"claude_persistent": "KI ist kuenstliche Intelligenz."},
            errors={},
            consensus_analysis="Nur ein Provider hat geantwortet.",
            duration_seconds=2.5,
            providers_queried=["claude_persistent"],
        )

        update = _make_update(user_id=1, chat_id=10, text="/debate Was ist KI?")
        # Mock: reply_text returns a message with delete method
        status_msg = MagicMock()
        status_msg.delete = AsyncMock()
        update.message.reply_text = AsyncMock(side_effect=[status_msg, None])

        context = _make_context(args=["Was", "ist", "KI?"])

        with patch(
            "application.debate_orchestrator.DebateOrchestrator.debate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await handle_debate_command(update, context)

        # Status-Message sollte geloescht werden
        status_msg.delete.assert_called_once()

        # Mindestens 2 reply_text calls: Status + Ergebnis
        assert update.message.reply_text.call_count >= 2

        # Audit-Log geschrieben
        mock_audit.assert_called()
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["event_type"] == "debate"
        assert audit_entry["user_id"] == 1
        assert audit_entry["providers_queried"] == ["claude_persistent"]

    async def test_debate_privacy_guard_blocks_group(self) -> None:
        """/debate in Gruppe wird blockiert (Privacy-Guard)."""
        from presentation.handlers import handle_debate_command

        update = _make_update(user_id=1, chat_id=10, text="/debate Test")
        update.effective_chat.type = "group"
        context = _make_context(args=["Test"])

        await handle_debate_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "privaten Chat" in reply_text

    @patch("presentation.handlers.write_raw_audit")
    async def test_debate_rate_limit_blocks(self, mock_audit: MagicMock) -> None:
        """/debate respektiert Rate-Limiting."""
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_debate_command

        limiter = RateLimiter()
        # Alle Minute-Tokens verbrauchen
        normal_min = PROFILES["normal"]["per_minute"]
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        update = _make_update(user_id=1, chat_id=10, text="/debate Test?")
        context = _make_context(args=["Test?"], rate_limiter=limiter)

        await handle_debate_command(update, context)

        # User bekommt Limit-Meldung
        reply_text = update.message.reply_text.call_args[0][0]
        assert "Limit" in reply_text

        # Audit: rate_limit_exceeded
        mock_audit.assert_called()
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["event_type"] == "rate_limit_exceeded"
        assert audit_entry["command"] == "debate"


class TestHandleSetlimitCommand:
    """Tests für /setlimit Command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_setlimit_normal(self, tmp_path: Path) -> None:
        """/setlimit normal wechselt Profil."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=1, chat_id=10, profile="light")

            update = _make_update(user_id=1, chat_id=10)
            context = _make_context(args=["normal"], rate_limiter=limiter)

            await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Normal" in reply_text
        assert "350/Stunde" in reply_text or "350" in reply_text

    async def test_setlimit_light(self, tmp_path: Path) -> None:
        """/setlimit light wechselt Profil."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()

            update = _make_update(user_id=2, chat_id=20)
            context = _make_context(args=["light"], rate_limiter=limiter)

            await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Light" in reply_text

    async def test_setlimit_power(self, tmp_path: Path) -> None:
        """/setlimit power wechselt Profil."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()

            update = _make_update(user_id=3, chat_id=30)
            context = _make_context(args=["power"], rate_limiter=limiter)

            await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Power" in reply_text
        assert "900" in reply_text

    async def test_setlimit_unlimited_requires_confirm(self) -> None:
        """/setlimit unlimited ohne confirm zeigt Warnung."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        limiter = RateLimiter()

        update = _make_update(user_id=4, chat_id=40)
        context = _make_context(args=["unlimited"], rate_limiter=limiter)

        await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Bestätigen" in reply_text or "confirm" in reply_text.lower()
        assert "/setlimit unlimited confirm" in reply_text

        # Profil wurde NICHT gewechselt
        assert limiter.get_user_profile(4) == "normal"

    async def test_setlimit_unlimited_confirm_works(self, tmp_path: Path) -> None:
        """/setlimit unlimited confirm wechselt Profil."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()

            update = _make_update(user_id=5, chat_id=50)
            context = _make_context(args=["unlimited", "confirm"], rate_limiter=limiter)

            await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Unlimited" in reply_text
        assert limiter.get_user_profile(5) == "unlimited"

    async def test_setlimit_invalid_profile(self) -> None:
        """/setlimit invalid zeigt Fehlermeldung."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        limiter = RateLimiter()

        update = _make_update(user_id=6, chat_id=60)
        context = _make_context(args=["megapower"], rate_limiter=limiter)

        await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Unbekanntes Profil" in reply_text

    async def test_setlimit_no_args_shows_current(self) -> None:
        """/setlimit ohne Argumente zeigt aktuelles Profil."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        limiter = RateLimiter()

        update = _make_update(user_id=7, chat_id=70)
        context = _make_context(args=[], rate_limiter=limiter)

        await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Aktuelles Profil" in reply_text
        assert "Normal" in reply_text
