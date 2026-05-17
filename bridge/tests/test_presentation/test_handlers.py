"""Tests for presentation.handlers: Telegram command handlers.

Tests /save, /lang, /reset commands with mocked Telegram objects.
No real bot, no real API calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from filelock import FileLock

from application.chat_service import ChatService
from application.execution.envelope import RequestEnvelope
from infrastructure.conversation_storage import _reset_all_for_tests
from infrastructure.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Clear conversation storage before each test."""
    _reset_all_for_tests()


def _make_mock_chat_service(
    route_return: ProviderResponse | None = None,
) -> ChatService:
    """Create a ChatService with a mocked ProviderRouter."""
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
    """Create a mocked Telegram update."""
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
    # Simulate non-callback invocation (command, not inline button tap)
    update.callback_query = None
    return update


def _make_mock_context_kernel():
    """Create a mock ContextKernel that returns a default ExecutionContext."""
    from application.execution import ContextKernel, ExecutionContext
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


def _make_context(
    args: list[str] | None = None,
    chat_service: ChatService | None = None,
    system_prompt: str = "Test system prompt.",
    memory_service: object | None = None,
    persistent_provider: object | None = None,
    process_pool: object | None = None,
    rate_limiter: object | None = None,
    bookmark_service: object | None = None,
    context_kernel: object | None = None,
) -> MagicMock:
    """Create a mocked Telegram context with bot_data."""
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
        "context_kernel": context_kernel or _make_mock_context_kernel(),
    }
    return context


class TestHandleSaveCommand:
    """Tests for /save command."""

    @pytest.fixture(autouse=True)
    def _isolate_bookmark_storage(self, tmp_path: Path) -> None:
        """Patch bookmark storage and create BookmarkService."""
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
        """/save as reply to bot message creates a bookmark."""
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
        """/save without reply sends help text."""
        from presentation.handlers import handle_save_command

        update = _make_update(user_id=1, chat_id=10)
        update.message.reply_to_message = None

        context = _make_context(bookmark_service=self._bookmark_svc)
        await handle_save_command(update, context)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "antworte" in reply_text.lower() or "reply" in reply_text.lower()


class TestHandleLangCommand:
    """Tests for /lang command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_lang_command_sets_sticky_language(self) -> None:
        """/lang en sets the language correctly."""
        from infrastructure.conversation_storage import get_language
        from presentation.handlers import handle_lang_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(args=["en"])

        await handle_lang_command(update, context)

        lang = await get_language(1, 10)
        assert lang == "en"

        # Confirmation sent
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "English" in reply_text or "en" in reply_text

    async def test_lang_command_no_args_shows_inline_keyboard(self) -> None:
        """/lang without argument shows inline keyboard with language buttons."""
        from presentation.handlers import handle_lang_command

        update = _make_update()
        context = _make_context(args=[])

        await handle_lang_command(update, context)

        call_kwargs = update.message.reply_text.call_args
        reply_text = call_kwargs[0][0]
        # Header text (DE default: "Sprache wählen" or EN: "Choose language")
        assert "Sprache" in reply_text or "language" in reply_text.lower()
        # Inline keyboard must be present
        assert "reply_markup" in call_kwargs[1]
        keyboard = call_kwargs[1]["reply_markup"]
        # Verify it has buttons with lang_set: callback data
        all_buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert len(all_buttons) == 20  # 20 supported languages
        assert any("lang_set:de" in btn.callback_data for btn in all_buttons)

    async def test_lang_command_invalid_code(self) -> None:
        """/lang xyz with invalid code returns error."""
        from presentation.handlers import handle_lang_command

        update = _make_update()
        context = _make_context(args=["xyz"])

        await handle_lang_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Unbekannte" in reply_text or "unknown" in reply_text.lower()


class TestHandleResetCommand:
    """Tests for /reset command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_command_clears_history(self) -> None:
        """/reset clears history but preserves sticky language.

        Since i18n: the reset message is in the user's sticky language.
        Language is restored after reset (preference survives /reset).
        """
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

        # Old history cleared, reset confirmation saved as assistant turn
        history = await get_history(1, 10)
        assert len(history) == 1
        assert history[0].role == "assistant"
        # i18n: French reset message (since sticky language was "fr")
        assert (
            "réinitialisée" in history[0].content.lower()
            or "zurückgesetzt" in history[0].content.lower()
            or "reset" in history[0].content.lower()
            or "fresh" in history[0].content.lower()
        )

        # Language is PRESERVED after reset (not cleared)
        lang = await get_language(1, 10)
        assert lang == "fr"

        # Confirmation sent
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert (
            "réinitialisée" in reply_text.lower()
            or "zurückgesetzt" in reply_text.lower()
            or "reset" in reply_text.lower()
            or "fresh" in reply_text.lower()
        )


class TestStartCommandHistory:
    """Tests: /start speichert Bot-Antwort in Conversation-History (Fix A)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_start_command_saves_to_history(self) -> None:
        """/start saves welcome text as assistant turn in history."""
        from domain.onboarding import get_start_welcome_text
        from infrastructure.conversation_storage import get_history
        from presentation.handlers import handle_start_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context()

        await handle_start_command(update, context)

        # Default language is "de" (no sticky language set), so DE welcome text
        expected = get_start_welcome_text("de")
        history = await get_history(1, 10)
        assert len(history) == 1
        assert history[0].role == "assistant"
        assert history[0].content == expected

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
        """/help text contains all actually existing commands."""
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
            "/settings",
        ]
        for cmd in expected_commands:
            assert cmd in HELP_TEXT, f"Command {cmd} fehlt im HELP_TEXT"

    async def test_help_text_does_not_contain_nonexistent_commands(self) -> None:
        """/help text does NOT contain non-existent commands."""
        from presentation.handlers import HELP_TEXT

        nonexistent = ["/unsave", "/delete", "/clear", "/config"]
        for cmd in nonexistent:
            assert cmd not in HELP_TEXT, f"Nicht-existenter Command {cmd} im HELP_TEXT"

    async def test_help_text_is_structured(self) -> None:
        """/help text has the desired structure with categories."""
        from presentation.handlers import HELP_TEXT_DE, HELP_TEXT_EN

        # DE version
        assert "Chat" in HELP_TEXT_DE
        assert "Memory" in HELP_TEXT_DE
        assert "Bookmarks" in HELP_TEXT_DE
        assert "Multi-AI" in HELP_TEXT_DE
        assert "Konfiguration" in HELP_TEXT_DE
        assert "/onboarding" in HELP_TEXT_DE
        assert "/help" in HELP_TEXT_DE

        # EN version
        assert "Chat" in HELP_TEXT_EN
        assert "Memory" in HELP_TEXT_EN
        assert "Bookmarks" in HELP_TEXT_EN
        assert "/onboarding" in HELP_TEXT_EN


class TestReplyToContext:
    """Tests: Telegram reply-to is extracted as context (Fix B)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    def _make_reply_chat_service(self) -> tuple[ChatService, MagicMock]:
        """Create ChatService with mockable router for reply tests."""
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
        """When user replies to a bot message, the reply text is inserted into the prompt."""
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
        """Without reply, only the normal text is sent."""
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
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_reset_writes_audit(self, mock_audit: MagicMock) -> None:
        """/reset writes an audit entry with action='reset'."""
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
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_lang_change_writes_audit(self, mock_audit: MagicMock) -> None:
        """/lang en writes audit with old and new language."""
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
        """/lang xyz (invalid) writes NO audit entry."""
        from presentation.handlers import handle_lang_command

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["xyz"])

        await handle_lang_command(update, context)

        mock_audit.assert_not_called()


class TestAuditLoggingSave:
    """Tests: /save schreibt Audit-Log-Eintrag."""

    @pytest.fixture(autouse=True)
    def _isolate_bookmark_storage(self, tmp_path: Path) -> None:
        """Patch bookmark storage and create BookmarkService."""
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
        """/save as reply creates audit entry with action='save_bookmark'."""
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
        """Patch bookmark storage and create BookmarkService."""
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
        """/bookmarks without bookmarks writes audit with '0 bookmarks'."""
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
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_remember_writes_audit(self, mock_audit: MagicMock) -> None:
        """/remember text writes audit with action='remember' and entry_id."""
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
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("application.audit_service.write_audit_log")
    async def test_forget_success_writes_audit(self, mock_audit: MagicMock) -> None:
        """/forget ep_123 (found) writes audit with success=True."""
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
        """/forget ep_123 shows hint about /reset for history cleanup."""
        from presentation.handlers import handle_forget_command

        mock_memory = MagicMock()
        mock_memory.forget = MagicMock(return_value=True)

        update = _make_update(user_id=42, chat_id=99)
        context = _make_context(args=["ep_123"], memory_service=mock_memory)

        await handle_forget_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "ep_123" in reply_text
        assert "/reset" in reply_text
        # i18n: default language is DE, so "Hinweis" instead of "Note"
        assert "Hinweis" in reply_text or "Note" in reply_text

    @patch("application.audit_service.write_audit_log")
    async def test_forget_not_found_writes_audit(self, mock_audit: MagicMock) -> None:
        """/forget ep_999 (not found) writes audit with success=False."""
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
        """Whitelist bypass."""
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
        assert "private chat" in reply_text.lower()

    async def test_private_chat_allowed(self) -> None:
        """handle_message im privaten Chat funktioniert normal."""
        from presentation.handlers import handle_message

        update = _make_update(user_id=1, chat_id=10, text="Hallo")
        update.effective_chat.type = "private"

        context = _make_context()
        # No persistent_provider => legacy fallback
        await handle_message(update, context)

        # Sollte KEINE Privacy-Block-Meldung sein
        if update.message.reply_text.called:
            reply_text = update.message.reply_text.call_args[0][0]
            assert "private chat" not in reply_text.lower()


class TestStreamingErrorRedaction:
    """Tests: streaming errors are redacted before being sent to user (P0-2)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("presentation.handlers.write_raw_audit")
    async def test_error_event_shows_generic_message(
        self, mock_audit: MagicMock
    ) -> None:
        """Error event with sensitive text shows only generic message with ref."""
        from infrastructure.claude_process_pool import StreamEvent
        from presentation.handlers import _handle_message_streaming

        # Mock persistent_provider der ein Error-Event liefert
        mock_provider = MagicMock()
        mock_provider.is_available = MagicMock(return_value=True)

        # Mock chat_service.process_user_message_streaming
        mock_svc = _make_mock_chat_service()

        async def _error_events():
            yield StreamEvent(
                event_type="error",
                text="/secret/path/to/file.py: PermissionError traceback",
                raw={"error": {"message": "secret stacktrace"}},
                is_final=True,
            )

        async def mock_stream(**kwargs):
            return _error_events(), 0, {}

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
                envelope=RequestEnvelope.from_telegram(
                    user_id=42,
                    chat_id=99,
                    text="Test",
                    username="testuser",
                ),
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
        """RuntimeError with path info shows only generic message."""
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()

        async def mock_stream_raise(**kwargs):
            raise RuntimeError("C:\\Users\\secret\\pipe_broken.txt")

        mock_svc.process_user_message_streaming = mock_stream_raise

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
                envelope=RequestEnvelope.from_telegram(
                    user_id=42,
                    chat_id=99,
                    text="Test",
                    username="testuser",
                ),
            )

        # Letzter Edit-Text darf keinen Pfad enthalten
        last_edit_text = mock_msg.edit_text.call_args_list[-1][0][0]
        assert "secret" not in last_edit_text
        assert "pipe_broken" not in last_edit_text
        assert "ref:" in last_edit_text


class TestStreamingAuditEntries:
    """Tests: streaming produces 2 audit entries (started + completed/crashed)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
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
            # init-Event liefert was_cold/subprocess_pid (Fix für Model-Switch-Bug)
            yield StreamEvent(event_type="init", was_cold=True, subprocess_pid=999)
            yield StreamEvent(event_type="content_delta", text="Hallo")
            yield StreamEvent(
                event_type="result", full_text="Hallo Welt", is_final=True
            )

        async def mock_stream(**kwargs):
            return _stream_events(), 3, {}  # 3 Memory-Einträge geladen

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
                envelope=RequestEnvelope.from_telegram(
                    user_id=42,
                    chat_id=99,
                    text="Frage",
                    username="testuser",
                ),
            )

        # At least 1 audit entry (stream_started)
        assert mock_audit.call_count >= 1
        first_entry = mock_audit.call_args_list[0][0][0]
        assert first_entry["event_type"] == "stream_started"
        assert first_entry["user_id"] == 42

        # save_streaming_result was called (= second audit)
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

        async def mock_stream_crash(**kwargs):
            async def _crash_gen():
                if True:
                    raise RuntimeError("unexpected crash")
                yield  # pragma: no cover

            return _crash_gen(), 0, {}

        mock_svc.process_user_message_streaming = mock_stream_crash

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
                envelope=RequestEnvelope.from_telegram(
                    user_id=42,
                    chat_id=99,
                    text="Crash",
                    username="testuser",
                ),
            )

        # At least 2 audit entries: stream_started + stream_error
        assert mock_audit.call_count >= 2
        event_types = [c[0][0]["event_type"] for c in mock_audit.call_args_list]
        assert "stream_started" in event_types
        assert "stream_error" in event_types

        # stream_error must have error_id
        error_entries = [
            c[0][0]
            for c in mock_audit.call_args_list
            if c[0][0]["event_type"] == "stream_error"
        ]
        assert error_entries[0]["error_id"] != ""

    @patch("presentation.handlers.write_raw_audit")
    async def test_stream_error_includes_task_meta(self, mock_audit: MagicMock) -> None:
        """stream_error audit event includes task_meta (task_slot, resolved_model etc.)."""
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()

        sample_task_meta = {
            "task_slot": "code",
            "task_score": 103,
            "resolved_model": "claude-opus-4-7",
        }

        async def mock_stream_crash(**kwargs):
            async def _crash_gen():
                if True:
                    raise RuntimeError("crash with meta")
                yield  # pragma: no cover

            return _crash_gen(), 0, sample_task_meta

        mock_svc.process_user_message_streaming = mock_stream_crash

        mock_pool = MagicMock()
        mock_managed = MagicMock()
        mock_managed.pid = 999

        async def mock_get_or_create(user_id, chat_id):
            return mock_managed, False

        mock_pool.get_or_create = mock_get_or_create

        update = _make_update(user_id=42, chat_id=99, text="Crash meta test")
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
                text="Crash meta test",
                reply_to_text=None,
                envelope=RequestEnvelope.from_telegram(
                    user_id=42,
                    chat_id=99,
                    text="Crash meta test",
                    username="testuser",
                ),
            )

        # Find stream_error entries
        error_entries = [
            c[0][0]
            for c in mock_audit.call_args_list
            if c[0][0]["event_type"] == "stream_error"
        ]
        assert len(error_entries) >= 1
        err = error_entries[0]
        # task_meta fields must be present
        assert err["task_slot"] == "code"
        assert err["task_score"] == 103
        assert err["resolved_model"] == "claude-opus-4-7"


class TestOuterExceptionCoverage:
    """Tests: Outer Exception in _handle_message_streaming (P1-8)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
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
                envelope=RequestEnvelope.from_telegram(
                    user_id=42,
                    chat_id=99,
                    text="Test",
                    username="testuser",
                ),
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
    """Tests for rate limiting in the handle_message handler (C-2)."""

    @pytest.fixture(autouse=True)
    def _bypass_whitelist(self) -> None:
        """Whitelist bypass for handler tests."""
        self._patches = [
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
            patch("presentation.handlers.DEFAULT_LANGUAGE", "en"),
        ]
        for p in self._patches:
            p.start()
        yield  # type: ignore[misc]
        for p in self._patches:
            p.stop()

    async def test_rate_limit_blocks_message(self) -> None:
        """Rate-limited user gets message, no LLM call."""
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_message

        limiter = RateLimiter()
        # Consume all minute tokens (Normal: 25/min)
        normal_min = PROFILES["normal"]["per_minute"]
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        update = _make_update(user_id=1, chat_id=10, text="Hallo")
        context = _make_context(rate_limiter=limiter)

        with patch("presentation.handlers.write_raw_audit") as mock_audit:
            await handle_message(update, context)

        # User gets limit message
        # Beachte: 70%-Warnung kann vorher reply_text aufrufen
        calls = update.message.reply_text.call_args_list
        # Letzter oder einziger Call muss Limit-Meldung sein
        limit_reply = calls[-1][0][0]
        assert "limit" in limit_reply.lower()

        # No typing indicator sent (no LLM call)
        context.bot.send_chat_action.assert_not_called()

        # Audit log contains rate_limit_exceeded
        mock_audit.assert_called()
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["event_type"] == "rate_limit_exceeded"
        assert audit_entry["user_id"] == 1
        assert audit_entry["profile"] == "normal"
        assert audit_entry["period"] == "minute"

    async def test_rate_limit_allows_normal_request(self) -> None:
        """Below the limit: normal LLM call flow."""
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
        """If no RateLimiter in bot_data: let through normally."""
        from presentation.handlers import handle_message

        update = _make_update(user_id=3, chat_id=30, text="Test")
        context = _make_context(rate_limiter=None)

        await handle_message(update, context)

        # Typing-Indicator gesendet = LLM-Pfad betreten
        context.bot.send_chat_action.assert_called()

    async def test_rate_limit_exceeded_shows_profile_info(self) -> None:
        """Rate limit message shows profile info and options."""
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_message

        limiter = RateLimiter()
        normal_min = PROFILES["normal"]["per_minute"]

        # Consume all minute tokens
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        update = _make_update(user_id=1, chat_id=10, text="Hallo")
        context = _make_context(rate_limiter=limiter)

        with patch("presentation.handlers.write_raw_audit") as mock_audit:
            await handle_message(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Normal profile" in reply_text
        assert "/usage" in reply_text
        assert "/setlimit" in reply_text

        # Audit contains profile and period info
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["profile"] == "normal"
        assert audit_entry["period"] == "minute"


class TestHandleUsageCommand:
    """Tests for /usage command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass + force English for assertion stability."""
        with (
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
            patch("presentation.handlers.DEFAULT_LANGUAGE", "en"),
        ):
            yield  # type: ignore[misc]

    async def test_usage_shows_profile_and_limits(self) -> None:
        """/usage shows profile and limits."""
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
        assert "Profile: Normal" in reply_text
        assert "This minute" in reply_text
        assert "This hour" in reply_text
        assert "Today" in reply_text
        assert "/setlimit" in reply_text

    async def test_usage_unlimited_profile(self, tmp_path: Path) -> None:
        """/usage shows unlimited info when profile is unlimited."""
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
        assert "No limits" in reply_text

    async def test_usage_no_limiter_shows_error(self) -> None:
        """/usage without rate limiter shows error message."""
        from presentation.handlers import handle_usage_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(rate_limiter=None)

        await handle_usage_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "not initialized" in reply_text.lower()


class TestHandleDebateCommand:
    """Tests for /debate command (R10: Multi-AI debate)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_debate_without_args_shows_help(self) -> None:
        """/debate without arguments shows help text."""
        from presentation.handlers import handle_debate_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context(args=[])

        await handle_debate_command(update, context)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "/debate" in reply_text
        # Help text contains usage example (language-independent check)
        assert (
            "Bitcoin" in reply_text or "question" in reply_text or "Frage" in reply_text
        )

    @patch("presentation.handlers.write_raw_audit")
    async def test_debate_with_question_calls_orchestrator(
        self, mock_audit: MagicMock
    ) -> None:
        """/debate with question calls DebateOrchestrator."""
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

        # Audit log written
        mock_audit.assert_called()
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["event_type"] == "debate"
        assert audit_entry["user_id"] == 1
        assert audit_entry["providers_queried"] == ["claude_persistent"]

    async def test_debate_privacy_guard_blocks_group(self) -> None:
        """/debate in group is blocked (privacy guard)."""
        from presentation.handlers import handle_debate_command

        update = _make_update(user_id=1, chat_id=10, text="/debate Test")
        update.effective_chat.type = "group"
        context = _make_context(args=["Test"])

        await handle_debate_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "private chat" in reply_text.lower()

    @patch("presentation.handlers.write_raw_audit")
    async def test_debate_rate_limit_blocks(self, mock_audit: MagicMock) -> None:
        """/debate respects rate limiting."""
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_debate_command

        limiter = RateLimiter()
        # Consume all minute tokens
        normal_min = PROFILES["normal"]["per_minute"]
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        update = _make_update(user_id=1, chat_id=10, text="/debate Test?")
        context = _make_context(args=["Test?"], rate_limiter=limiter)

        await handle_debate_command(update, context)

        # User gets limit message
        reply_text = update.message.reply_text.call_args[0][0]
        assert "limit" in reply_text

        # Audit: rate_limit_exceeded
        mock_audit.assert_called()
        audit_entry = mock_audit.call_args[0][0]
        assert audit_entry["event_type"] == "rate_limit_exceeded"
        assert audit_entry["command"] == "debate"

    @patch("presentation.handlers.write_raw_audit")
    async def test_debate_rate_limit_does_not_build_context(
        self, mock_audit: MagicMock
    ) -> None:
        """EK-02: Rate-limited debate must NOT call ContextKernel.build.

        This ensures rejected requests cannot mutate sticky language
        via LanguageResolver side-effects.
        """
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_debate_command

        limiter = RateLimiter()
        # Exhaust all minute tokens
        normal_min = PROFILES["normal"]["per_minute"]
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        mock_kernel = _make_mock_context_kernel()
        update = _make_update(user_id=1, chat_id=10, text="/debate Hallo?")
        context = _make_context(
            args=["Hallo?"],
            rate_limiter=limiter,
            context_kernel=mock_kernel,
        )

        await handle_debate_command(update, context)

        # ContextKernel.build must NOT have been called
        mock_kernel.build.assert_not_called()

    @patch("presentation.handlers.write_raw_audit")
    async def test_debate_rate_limit_reject_message_uses_sticky(
        self, mock_audit: MagicMock
    ) -> None:
        """EK-02: Reject message uses sticky language (read-only, no mutation)."""
        from application.rate_limiter import PROFILES, RateLimiter
        from presentation.handlers import handle_debate_command

        limiter = RateLimiter()
        # Exhaust all minute tokens
        normal_min = PROFILES["normal"]["per_minute"]
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        # Create a chat service that returns "en" as sticky language
        mock_svc = _make_mock_chat_service()
        mock_svc.get_chat_language = AsyncMock(return_value="en")

        update = _make_update(user_id=1, chat_id=10, text="/debate Test?")
        context = _make_context(
            args=["Test?"],
            rate_limiter=limiter,
            chat_service=mock_svc,
        )

        await handle_debate_command(update, context)

        # Sticky language was read (non-mutating)
        mock_svc.get_chat_language.assert_called_once_with(1, 10)

        # Reply was sent (rate limit message)
        update.message.reply_text.assert_called_once()


class TestFormatDebateSynthesis:
    """Tests for the synthesis display in the debate formatter."""

    def test_format_debate_shows_synthesis_prominently(self) -> None:
        """Synthesis is displayed as a prominent element in the output."""
        from application.debate_orchestrator import (
            DebateResult,
            FinalVerdict,
            ProviderEvaluation,
        )
        from presentation.handlers import _format_debate_result

        verdict = FinalVerdict(
            winner="claude_persistent",
            recommendation="Claude liefert die klarere Antwort.",
            synthesis="Bitcoin ist eine dezentrale digitale Währung die Peer-to-Peer-Zahlungen ermöglicht.",
            evaluations=[
                ProviderEvaluation(
                    provider="claude_persistent", pros=["Klar"], cons=[]
                ),
                ProviderEvaluation(
                    provider="ollama_local", pros=["Ausfuehrlich"], cons=["Vage"]
                ),
            ],
            reasoning="Claude ist praeziser.",
            judge_provider="claude_persistent",
        )

        result = DebateResult(
            question="Was ist Bitcoin?",
            responses={
                "claude_persistent": "Bitcoin ist digitales Geld.",
                "ollama_local": "Bitcoin ist eine Kryptowährung.",
            },
            errors={},
            consensus_analysis="Hohe Übereinstimmung.",
            final_verdict=verdict,
            duration_seconds=3.2,
            providers_queried=["claude_persistent", "ollama_local"],
        )

        formatted = _format_debate_result(result, lang="de")

        # Synthese muss im Output sein (prominent)
        assert "dezentrale digitale Währung" in formatted
        assert "Peer-to-Peer" in formatted
        # "Synthese" als Überschrift
        assert "Synthese" in formatted
        # Winner-Label ist jetzt "Stärkster Beitrag" (nicht "Beste Einzelantwort")
        assert "Stärkster Beitrag" in formatted
        assert "Beste Einzelantwort" not in formatted
        assert "Winner" not in formatted
        # Kernaussage ist da
        assert "Kernaussage:" in formatted

        # BLUF-Reihenfolge: Kernaussage vor Synthese vor Detail-Antworten
        emp_pos = formatted.index("Kernaussage:")
        syn_pos = formatted.index("Synthese")
        detail_pos = formatted.index("Detail-Antworten")
        assert emp_pos < syn_pos < detail_pos

    def test_format_debate_bluf_order_detail_answers_last(self) -> None:
        """Detail responses from the AIs appear at the end (before timer)."""
        from application.debate_orchestrator import (
            DebateResult,
            FinalVerdict,
            ProviderEvaluation,
        )
        from presentation.handlers import _format_debate_result

        verdict = FinalVerdict(
            winner="claude_persistent",
            recommendation="Claude liefert bessere Antwort.",
            synthesis="Zusammenfassung beider Antworten.",
            evaluations=[
                ProviderEvaluation(
                    provider="claude_persistent", pros=["Klar"], cons=[]
                ),
                ProviderEvaluation(
                    provider="ollama_local", pros=["Ausfuehrlich"], cons=["Vage"]
                ),
            ],
            reasoning="Claude ist praeziser.",
            judge_provider="claude_persistent",
        )

        result = DebateResult(
            question="Testfrage?",
            responses={
                "claude_persistent": "Claude-Antwort hier.",
                "ollama_local": "Llama-Antwort hier.",
            },
            errors={},
            consensus_analysis="Hohe Übereinstimmung.",
            final_verdict=verdict,
            duration_seconds=2.0,
            providers_queried=["claude_persistent", "ollama_local"],
        )

        formatted = _format_debate_result(result, lang="de")

        # BLUF: Kernaussage -> Staerkster Beitrag -> Synthese -> Details -> Pro/Contra -> Timer
        emp_pos = formatted.index("Kernaussage:")
        strongest_pos = formatted.index("Stärkster Beitrag:")
        syn_pos = formatted.index("Synthese")
        detail_pos = formatted.index("Detail-Antworten")
        # Pro/Contra kommt nach Detail-Antworten (Analyse der Originale)
        pro_pos = formatted.index("✅")
        timer_pos = formatted.index("⏱")

        assert emp_pos < strongest_pos
        assert strongest_pos < syn_pos
        assert syn_pos < detail_pos
        assert detail_pos < pro_pos
        assert pro_pos < timer_pos

        # Detail-Antworten enthalten die Original-Texte
        assert "Claude-Antwort hier." in formatted
        assert "Llama-Antwort hier." in formatted

    def test_format_debate_english_labels(self) -> None:
        """English labels are used correctly when lang='en'."""
        from application.debate_orchestrator import (
            DebateResult,
            FinalVerdict,
            ProviderEvaluation,
        )
        from presentation.handlers import _format_debate_result

        verdict = FinalVerdict(
            winner="claude_persistent",
            recommendation="Claude wins.",
            synthesis="Combined answer.",
            evaluations=[
                ProviderEvaluation(
                    provider="claude_persistent", pros=["Clear"], cons=[]
                ),
            ],
            reasoning="Better answer.",
            judge_provider="claude_persistent",
        )

        result = DebateResult(
            question="What is AI?",
            responses={
                "claude_persistent": "AI is artificial intelligence.",
                "ollama_local": "AI means smart machines.",
            },
            errors={},
            consensus_analysis=None,
            final_verdict=verdict,
            duration_seconds=1.5,
            providers_queried=["claude_persistent", "ollama_local"],
        )

        formatted = _format_debate_result(result, lang="en")

        # English labels
        assert "Key Takeaway:" in formatted
        assert "Strongest Contribution:" in formatted
        assert "Synthesis" in formatted
        assert "Detail Responses" in formatted
        # No German labels
        assert "Kernaussage:" not in formatted
        assert "Stärkster Beitrag:" not in formatted

    def test_format_debate_without_synthesis_shows_no_empty_block(self) -> None:
        """When synthesis is empty, no empty block is displayed."""
        from application.debate_orchestrator import DebateResult, FinalVerdict
        from presentation.handlers import _format_debate_result

        verdict = FinalVerdict(
            winner="claude_persistent",
            recommendation="Claude gewinnt.",
            synthesis="",  # Leer (Backward-Compat / schwacher Judge)
            evaluations=[],
            reasoning="Claude ist besser.",
            judge_provider="ollama_local",
            judge_quality_warning="Lokaler Judge (Ollama), Bewertungsqualität reduziert",
        )

        result = DebateResult(
            question="Test?",
            responses={
                "claude_persistent": "Antwort A",
                "ollama_local": "Antwort B",
            },
            errors={},
            consensus_analysis=None,
            final_verdict=verdict,
            duration_seconds=1.0,
            providers_queried=["claude_persistent", "ollama_local"],
        )

        formatted = _format_debate_result(result, lang="de")

        # Synthese-Block wird NICHT gerendert wenn synthesis leer ist
        assert "Synthese" not in formatted
        # Kernaussage und Staerkster Beitrag sind trotzdem da
        assert "Claude gewinnt." in formatted
        assert "Stärkster Beitrag" in formatted
        # Quality warning wird angezeigt
        assert "Lokaler Judge" in formatted


class TestHandleSetlimitCommand:
    """Tests for /setlimit command."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
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
        """/setlimit unlimited without confirm shows warning."""
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


class TestResetCancelsActiveStream:
    """T25: /reset cancels an active streaming session before clearing state."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_cancels_registered_session(self) -> None:
        """/reset sets cancel_event on the active StreamingSession."""
        from application.streaming_handler import StreamingSession
        from presentation.handlers import (
            _active_sessions_lock,
            _active_streaming_sessions,
            handle_reset_command,
        )

        # Register a fake active session
        fake_msg = AsyncMock()
        fake_msg.edit_text = AsyncMock()
        fake_msg.chat = MagicMock()
        fake_msg.chat.send_message = AsyncMock()
        session = StreamingSession(message=fake_msg, started_at=0.0)

        user_id, chat_id = 42, 99
        with _active_sessions_lock:
            _active_streaming_sessions[(user_id, chat_id)] = session

        try:
            update = _make_update(user_id=user_id, chat_id=chat_id)
            context = _make_context()

            # Patch asyncio.sleep to avoid real delay
            with patch("presentation.handlers.asyncio.sleep", new_callable=AsyncMock):
                await handle_reset_command(update, context)

            # Session must be cancelled
            assert session.is_cancelled is True
        finally:
            # Cleanup
            with _active_sessions_lock:
                _active_streaming_sessions.pop((user_id, chat_id), None)

    async def test_reset_without_active_session_still_works(self) -> None:
        """/reset works normally when no streaming session is active."""
        from presentation.handlers import handle_reset_command

        update = _make_update(user_id=1, chat_id=10)
        context = _make_context()

        # Should not raise
        await handle_reset_command(update, context)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert (
            "zurückgesetzt" in reply_text.lower()
            or "frisch" in reply_text.lower()
            or "reset" in reply_text.lower()
            or "fresh" in reply_text.lower()
        )

    async def test_setlimit_invalid_profile(self) -> None:
        """/setlimit invalid zeigt Fehlermeldung."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        limiter = RateLimiter()

        update = _make_update(user_id=6, chat_id=60)
        context = _make_context(args=["megapower"], rate_limiter=limiter)

        await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        # The invalid profile name must appear in the error message (language-independent)
        assert "megapower" in reply_text
        assert "Unknown profile" in reply_text or "Unbekanntes Profil" in reply_text

    async def test_setlimit_no_args_shows_current(self) -> None:
        """/setlimit without arguments shows current profile."""
        from application.rate_limiter import RateLimiter
        from presentation.handlers import handle_setlimit_command

        limiter = RateLimiter()

        update = _make_update(user_id=7, chat_id=70)
        context = _make_context(args=[], rate_limiter=limiter)

        await handle_setlimit_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        # Must show current profile name and available profiles (language-independent)
        assert "Current profile" in reply_text or "Aktuelles Profil" in reply_text
        assert "Normal" in reply_text


class TestContextKernelIntegrationInHandler:
    """Phase 0 Commit 2: ContextKernel is used in _handle_message_streaming."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("presentation.handlers.write_raw_audit")
    async def test_streaming_handler_calls_kernel_build(
        self, mock_audit: MagicMock
    ) -> None:
        """_handle_message_streaming calls ContextKernel.build with correct envelope."""
        from infrastructure.claude_process_pool import StreamEvent
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()

        async def _events():
            yield StreamEvent(event_type="init", was_cold=False, subprocess_pid=111)
            yield StreamEvent(event_type="content_delta", text="Hi")
            yield StreamEvent(event_type="result", full_text="Hi", is_final=True)

        async def mock_stream(**kwargs):
            return _events(), 0, {}

        mock_svc.process_user_message_streaming = mock_stream
        mock_svc.save_streaming_result = AsyncMock(return_value="Hi")

        # Custom mock kernel that tracks calls
        mock_kernel = _make_mock_context_kernel()

        update = _make_update(user_id=7, chat_id=77, text="Bonjour")
        context = _make_context(
            chat_service=mock_svc,
            persistent_provider=mock_provider,
            context_kernel=mock_kernel,
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
                user_id=7,
                chat_id=77,
                username="testuser",
                text="Bonjour",
                reply_to_text=None,
                envelope=RequestEnvelope.from_telegram(
                    user_id=7,
                    chat_id=77,
                    text="Bonjour",
                    username="testuser",
                ),
            )

        # Kernel.build was called exactly once
        mock_kernel.build.assert_called_once()
        # Envelope passed to kernel has correct user/chat ids
        call_args = mock_kernel.build.call_args
        envelope = call_args[0][0]
        assert envelope.user_id == 7
        assert envelope.chat_id == 77
        assert envelope.raw_text == "Bonjour"

    @patch("presentation.handlers.write_raw_audit")
    async def test_streaming_audit_includes_request_id(
        self, mock_audit: MagicMock
    ) -> None:
        """Audit 'stream_started' event includes request_id from envelope."""
        from infrastructure.claude_process_pool import StreamEvent
        from presentation.handlers import _handle_message_streaming

        mock_provider = MagicMock()
        mock_svc = _make_mock_chat_service()

        async def _events():
            yield StreamEvent(event_type="init", was_cold=False, subprocess_pid=222)
            yield StreamEvent(event_type="result", full_text="OK", is_final=True)

        async def mock_stream(**kwargs):
            return _events(), 0, {}

        mock_svc.process_user_message_streaming = mock_stream
        mock_svc.save_streaming_result = AsyncMock(return_value="OK")

        update = _make_update(user_id=8, chat_id=88, text="Test")
        context = _make_context(
            chat_service=mock_svc,
            persistent_provider=mock_provider,
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
                user_id=8,
                chat_id=88,
                username="testuser",
                text="Test",
                reply_to_text=None,
                envelope=RequestEnvelope.from_telegram(
                    user_id=8,
                    chat_id=88,
                    text="Test",
                    username="testuser",
                ),
            )

        # First audit call must be stream_started with request_id
        first_entry = mock_audit.call_args_list[0][0][0]
        assert first_entry["event_type"] == "stream_started"
        assert "request_id" in first_entry
        assert len(first_entry["request_id"]) == 12


class TestStreamingSessionCancelPropagation:
    """T25/EK-03: StreamingSession.cancel_event propagation primitives."""

    @pytest.mark.asyncio
    async def test_cancel_method_sets_event_and_is_cancelled_flag(self) -> None:
        """session.cancel() sets the asyncio.Event AND is_cancelled flag."""
        from application.streaming_handler import StreamingSession

        mock_msg = MagicMock()
        session = StreamingSession(message=mock_msg)
        assert not session.is_cancelled

        session.cancel()
        assert session.is_cancelled
        assert session.cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_direct_event_set_reflects_in_is_cancelled(self) -> None:
        """Setting cancel_event directly (as /reset does) flips is_cancelled."""
        from application.streaming_handler import StreamingSession

        mock_msg = MagicMock()
        session = StreamingSession(message=mock_msg)

        assert not session.is_cancelled
        session.cancel_event.set()
        assert session.is_cancelled
