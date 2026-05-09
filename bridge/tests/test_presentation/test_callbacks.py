"""Tests für presentation.callbacks: Bookmark Inline-Button Callbacks.

Testet handle_bookmark_show_callback und handle_bookmark_delete_callback
mit gemockten Telegram-Objekten und Bookmark-Storage.
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
    """Erstellt ein gemocktes Telegram-Update für CallbackQuery."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.type = chat_type
    # message für require_private_chat Fehlermeldung
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
    """Erstellt einen gemockten Telegram-Context mit bot_data."""
    context = MagicMock()
    context.application = MagicMock()
    context.application.bot_data = {
        "bookmark_service": bookmark_service,
    }
    return context


class TestBookmarkShowCallback:
    """Tests für handle_bookmark_show_callback (bm_show)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage und Whitelist für Isolation."""
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
        """bm_show mit gültiger ID zeigt den Bookmark-Inhalt an."""
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
        # answer() muss aufgerufen werden (Pflicht bei Telegram)
        query.answer.assert_called()
        # Inhalt muss als reply_text gesendet werden
        query.message.reply_text.assert_called()
        sent_text = query.message.reply_text.call_args[0][0]
        assert "Bookmark-Volltext" in sent_text or "äöüß" in sent_text

    async def test_show_nonexistent_bookmark(self) -> None:
        """bm_show mit nicht-existierender ID zeigt 'nicht gefunden'."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_show:10:999", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        answer_kwargs = query.answer.call_args
        assert (
            "nicht gefunden" in answer_kwargs.kwargs.get("text", "").lower()
            or "nicht gefunden" in str(answer_kwargs).lower()
        )

    async def test_show_invalid_callback_data(self) -> None:
        """bm_show mit ungültigen Daten zeigt 'Ungültige ID'."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_show:abc:xyz", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        # Muss "Ungültige ID" melden
        answer_call = query.answer.call_args
        assert (
            "Ungültige ID" in str(answer_call)
            or "ungültige" in str(answer_call).lower()
        )

    async def test_show_ignores_wrong_prefix(self) -> None:
        """Callback mit falschem Prefix wird ignoriert (kein Crash)."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_del:10:100", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_show_callback(update, context)

        # Nichts sollte passieren
        update.callback_query.answer.assert_not_called()
        update.callback_query.message.reply_text.assert_not_called()


class TestBookmarkDeleteCallback:
    """Tests für handle_bookmark_delete_callback (bm_del)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage und Whitelist für Isolation."""
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
        """bm_del mit gültiger ID löscht den Bookmark und sendet Bestätigung."""
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
        assert "Entfernt" in str(answer_call) or "entfernt" in str(answer_call).lower()

        # Chat-Bestätigung muss gesendet worden sein
        query.message.reply_text.assert_called_once()
        confirm_text = query.message.reply_text.call_args[0][0]
        assert "Bookmark" in confirm_text
        assert "entfernt" in confirm_text

        # Bookmark sollte wirklich weg sein
        assert self._bookmark_svc.get_bookmark(1, 10, 200) is None

    async def test_delete_confirmation_includes_date(self) -> None:
        """bm_del Bestätigung enthält Datum des Bookmarks."""
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
        # Datum im Format DD.MM.YYYY muss enthalten sein
        assert "vom" in confirm_text
        assert "2026" in confirm_text or "20" in confirm_text

    async def test_delete_nonexistent_bookmark(self) -> None:
        """bm_del mit nicht-existierender ID meldet 'nicht gefunden', keine Chat-Nachricht."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_del:10:999", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        answer_call = query.answer.call_args
        assert "nicht gefunden" in str(answer_call).lower()
        # Keine Chat-Bestätigung bei nicht-existentem Bookmark
        query.message.reply_text.assert_not_called()

    async def test_delete_invalid_callback_data(self) -> None:
        """bm_del mit ungültigen Daten zeigt 'Ungültige ID'."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_del:abc:xyz", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        answer_call = query.answer.call_args
        assert (
            "Ungültige ID" in str(answer_call)
            or "ungültige" in str(answer_call).lower()
        )

    async def test_delete_ignores_wrong_prefix(self) -> None:
        """Callback mit falschem Prefix wird ignoriert."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_show:10:100", user_id=1)
        context = _make_context(bookmark_service=self._bookmark_svc)

        await handle_bookmark_delete_callback(update, context)

        update.callback_query.answer.assert_not_called()


class TestCallbackPrivacyGuard:
    """Tests: Callbacks werden in Gruppen vom Decorator blockiert."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patcht Whitelist (erlaubt), aber kein private-chat."""
        self._patches = [
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
        ]
        for p in self._patches:
            p.start()

        yield  # type: ignore[misc]

        for p in self._patches:
            p.stop()

    async def test_show_blocked_in_group(self) -> None:
        """bm_show in Gruppe wird vom require_private_chat Decorator blockiert."""
        from presentation.callbacks import handle_bookmark_show_callback

        update = _make_callback_update("bm_show:10:100", user_id=1, chat_type="group")
        context = _make_context()

        await handle_bookmark_show_callback(update, context)

        # Callback body darf nicht ausgeführt worden sein (kein query.answer)
        # Der Decorator sendet eine Fehlermeldung über update.message
        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        assert "privaten Chat" in msg

    async def test_delete_blocked_in_group(self) -> None:
        """bm_del in Gruppe wird vom require_private_chat Decorator blockiert."""
        from presentation.callbacks import handle_bookmark_delete_callback

        update = _make_callback_update("bm_del:10:200", user_id=1, chat_type="group")
        context = _make_context()

        await handle_bookmark_delete_callback(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        assert "privaten Chat" in msg


class TestAuditLoggingBmShow:
    """Tests: bm_show Callback schreibt Audit-Log-Eintrag."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage und Whitelist für Isolation."""
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
        """bm_show mit gueltigem Bookmark schreibt Audit mit success=True."""
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
        """bm_show mit nicht-existierendem Bookmark schreibt Audit mit success=False."""
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
    """Tests: bm_del Callback schreibt Audit-Log-Eintrag."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage und Whitelist für Isolation."""
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
        """bm_del mit gueltigem Bookmark schreibt Audit mit success=True."""
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
        """bm_del mit nicht-existierendem Bookmark schreibt Audit mit success=False."""
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
