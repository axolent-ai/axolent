"""Tests for presentation.callbacks: Bookmark inline-button callbacks.

Tests handle_bookmark_show_callback and handle_bookmark_delete_callback
with mocked Telegram objects and bookmark storage.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from filelock import FileLock

from infrastructure.bookmark_storage import save_bookmark


def _make_callback_update(
    callback_data: str,
    user_id: int = 1,
    chat_type: str = "private",
) -> MagicMock:
    """Create a mocked Telegram update for CallbackQuery."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.type = chat_type
    # message for require_private_chat error message
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    query = MagicMock()
    query.data = callback_data
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.username = "testuser"
    query.answer = AsyncMock()
    query.message = MagicMock()
    query.message.reply_text = AsyncMock()
    update.callback_query = query
    return update


def _make_context(bookmark_service: object | None = None) -> MagicMock:
    """Create a mocked Telegram context with bot_data."""
    context = MagicMock()
    context.application = MagicMock()
    context.application.bot_data = {
        "bookmark_service": bookmark_service,
    }
    return context


class TestBookmarkShowCallback:
    """Tests for handle_bookmark_show_callback (bm_show)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patch bookmark storage and whitelist for isolation."""
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

    async def test_show_existing_bookmark(self) -> None:
        """bm_show with valid ID displays the bookmark content."""
        from presentation.callbacks import handle_bookmark_show_callback

        save_bookmark(
            user_id=1,
            username="testuser",
            message_id=100,
            chat_id=10,
            content="Das ist der Bookmark-Volltext mit Umlauten: äöüß",
        )

        update = _make_callback_update("bm_show:10:100", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        query = update.callback_query
        # answer() must be called (mandatory for Telegram)
        query.answer.assert_called()
        # Content must be sent as reply_text
        query.message.reply_text.assert_called()
        sent_text = query.message.reply_text.call_args[0][0]
        assert "Bookmark-Volltext" in sent_text or "äöüß" in sent_text

    async def test_show_nonexistent_bookmark(self) -> None:
        """bm_show with non-existent ID shows 'not found'."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_show:10:999", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        answer_kwargs = query.answer.call_args
        assert (
            "not found" in answer_kwargs.kwargs.get("text", "").lower()
            or "not found" in str(answer_kwargs).lower()
        )

    async def test_show_invalid_callback_data(self) -> None:
        """bm_show with invalid data shows 'Invalid ID'."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_show:abc:xyz", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        # Must report "Invalid ID"
        answer_call = query.answer.call_args
        assert "Invalid ID" in str(answer_call) or "invalid" in str(answer_call).lower()

    async def test_show_ignores_wrong_prefix(self) -> None:
        """Callback with wrong prefix is ignored (no crash)."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_del:10:100", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        # Nothing should happen
        update.callback_query.answer.assert_not_called()
        update.callback_query.message.reply_text.assert_not_called()


class TestBookmarkDeleteCallback:
    """Tests for handle_bookmark_delete_callback (bm_del)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patch bookmark storage and whitelist for isolation."""
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

    async def test_delete_existing_bookmark(self) -> None:
        """bm_del with valid ID deletes the bookmark and sends confirmation."""
        from presentation.callbacks import handle_bookmark_delete_callback

        save_bookmark(
            user_id=1,
            username="testuser",
            message_id=200,
            chat_id=10,
            content="Zu löschender Bookmark",
        )

        update = _make_callback_update("bm_del:10:200", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        answer_call = query.answer.call_args
        assert "Removed" in str(answer_call) or "removed" in str(answer_call).lower()

        # Chat confirmation must have been sent
        query.message.reply_text.assert_called_once()
        confirm_text = query.message.reply_text.call_args[0][0]
        assert "Bookmark" in confirm_text
        assert "removed" in confirm_text

        # Bookmark should actually be gone
        assert self._bookmark_svc.get_bookmark(1, 10, 200) is None

    async def test_delete_confirmation_includes_date(self) -> None:
        """bm_del confirmation includes the bookmark date."""
        from presentation.callbacks import handle_bookmark_delete_callback

        save_bookmark(
            user_id=1,
            username="testuser",
            message_id=201,
            chat_id=10,
            content="Bookmark mit Datum",
        )

        update = _make_callback_update("bm_del:10:201", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        query = update.callback_query
        confirm_text = query.message.reply_text.call_args[0][0]
        # Date in DD.MM.YYYY format must be present
        assert "from" in confirm_text
        assert "2026" in confirm_text or "20" in confirm_text

    async def test_delete_nonexistent_bookmark(self) -> None:
        """bm_del with non-existent ID reports 'not found', no chat message."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_del:10:999", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        answer_call = query.answer.call_args
        assert "not found" in str(answer_call).lower()
        # No chat confirmation for non-existent bookmark
        query.message.reply_text.assert_not_called()

    async def test_delete_invalid_callback_data(self) -> None:
        """bm_del with invalid data shows 'Invalid ID'."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_del:abc:xyz", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        answer_call = query.answer.call_args
        assert "Invalid ID" in str(answer_call) or "invalid" in str(answer_call).lower()

    async def test_delete_ignores_wrong_prefix(self) -> None:
        """Callback with wrong prefix is ignored."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_show:10:100", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        update.callback_query.answer.assert_not_called()


class TestCallbackPrivacyGuard:
    """Tests: callbacks are blocked in groups by the decorator."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patch whitelist (allowed) but no private chat."""
        self._patches = [
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
        ]
        for p in self._patches:
            p.start()

        yield  # type: ignore[misc]

        for p in self._patches:
            p.stop()

    async def test_show_blocked_in_group(self) -> None:
        """bm_show in a group is blocked by the require_private_chat decorator."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_show:10:100", user_id=1, chat_type="group")
        context = _make_context()

        await handle_bookmark_show_callback(update, context)

        # Callback body must not have been executed (no query.answer)
        # The decorator sends an error message via update.message
        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        assert "private chat" in msg.lower()

    async def test_delete_blocked_in_group(self) -> None:
        """bm_del in a group is blocked by the require_private_chat decorator."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_del:10:200", user_id=1, chat_type="group")
        context = _make_context()

        await handle_bookmark_delete_callback(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        assert "private chat" in msg.lower()


class TestAuditLoggingBmShow:
    """Tests: bm_show callback writes audit log entry."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patch bookmark storage and whitelist for isolation."""
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
    async def test_bm_show_existing_writes_audit(self, mock_audit: MagicMock) -> None:
        """bm_show with valid bookmark writes audit with success=True."""
        from presentation.callbacks import handle_bookmark_show_callback

        save_bookmark(
            user_id=1,
            username="testuser",
            message_id=100,
            chat_id=10,
            content="Volltext",
        )

        update = _make_callback_update("bm_show:10:100", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "bm_show"
        assert entry["user_id"] == 1
        assert entry["username"] == "testuser"
        assert entry["entry_id"] == "msg_100"
        assert entry["success"] is True

    @patch("application.audit_service.write_audit_log")
    async def test_bm_show_not_found_writes_audit(self, mock_audit: MagicMock) -> None:
        """bm_show with non-existent bookmark writes audit with success=False."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_show:10:999", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "bm_show"
        assert entry["username"] == "testuser"
        assert entry["success"] is False
        assert entry["entry_id"] == "msg_999"


class TestAuditLoggingBmDel:
    """Tests: bm_del callback writes audit log entry."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patch bookmark storage and whitelist for isolation."""
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
    async def test_bm_del_existing_writes_audit(self, mock_audit: MagicMock) -> None:
        """bm_del with valid bookmark writes audit with success=True."""
        from presentation.callbacks import handle_bookmark_delete_callback

        save_bookmark(
            user_id=1,
            username="testuser",
            message_id=200,
            chat_id=10,
            content="Zu löschender Bookmark",
        )

        update = _make_callback_update("bm_del:10:200", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "bm_del"
        assert entry["user_id"] == 1
        assert entry["username"] == "testuser"
        assert entry["entry_id"] == "msg_200"
        assert entry["success"] is True

    @patch("application.audit_service.write_audit_log")
    async def test_bm_del_not_found_writes_audit(self, mock_audit: MagicMock) -> None:
        """bm_del with non-existent bookmark writes audit with success=False."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_del:10:999", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "bm_del"
        assert entry["username"] == "testuser"
        assert entry["success"] is False
        assert entry["entry_id"] == "msg_999"
