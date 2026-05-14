"""Tests for presentation.decorators: whitelist guard and privacy guard.

Tests authorization via user ID whitelist, ALLOW_ALL_USERS bypass,
require_private_chat decorator, and _parse_whitelist validation.
Mocks Telegram Update/Context objects.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch


class TestRequireWhitelist:
    """require_whitelist Decorator-Tests."""

    def _make_update(self, user_id: int) -> MagicMock:
        """Erstellt ein gemocktes Telegram-Update mit User-ID."""
        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = user_id
        update.effective_user.username = "testuser"
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        return update

    async def test_require_whitelist_blocks_unauthorized(self) -> None:
        """Nicht-whitelisted User werden blockiert und erhalten Fehlermeldung."""
        with (
            patch("presentation.decorators.WHITELIST", {111, 222}),
            patch("presentation.decorators.ALLOW_ALL_USERS", False),
        ):
            from presentation.decorators import require_whitelist

            handler = AsyncMock()
            decorated = require_whitelist(handler)

            update = self._make_update(user_id=999)  # Nicht in Whitelist
            context = MagicMock()

            await decorated(update, context)

            handler.assert_not_called()
            update.message.reply_text.assert_called_once()
            call_text = update.message.reply_text.call_args[0][0]
            assert "not authorized" in call_text.lower()

    async def test_require_whitelist_allows_authorized(self) -> None:
        """Whitelisted User werden durchgelassen."""
        with (
            patch("presentation.decorators.WHITELIST", {111, 222}),
            patch("presentation.decorators.ALLOW_ALL_USERS", False),
        ):
            from presentation.decorators import require_whitelist

            handler = AsyncMock()
            decorated = require_whitelist(handler)

            update = self._make_update(user_id=111)  # In Whitelist
            context = MagicMock()

            await decorated(update, context)

            handler.assert_called_once_with(update, context)

    async def test_allow_all_users_bypass(self) -> None:
        """Bei ALLOW_ALL_USERS=True werden alle User durchgelassen."""
        with (
            patch("presentation.decorators.WHITELIST", set()),
            patch("presentation.decorators.ALLOW_ALL_USERS", True),
        ):
            from presentation.decorators import require_whitelist

            handler = AsyncMock()
            decorated = require_whitelist(handler)

            update = self._make_update(user_id=999)  # Beliebige ID
            context = MagicMock()

            await decorated(update, context)

            handler.assert_called_once_with(update, context)

    async def test_whitelist_no_user_blocks(self) -> None:
        """Wenn kein User im Update ist (user_id=0), wird blockiert."""
        with (
            patch("presentation.decorators.WHITELIST", {111}),
            patch("presentation.decorators.ALLOW_ALL_USERS", False),
        ):
            from presentation.decorators import require_whitelist

            handler = AsyncMock()
            decorated = require_whitelist(handler)

            update = MagicMock()
            update.effective_user = None
            update.message = MagicMock()
            update.message.reply_text = AsyncMock()
            context = MagicMock()

            await decorated(update, context)

            handler.assert_not_called()


class TestRequirePrivateChat:
    """require_private_chat Decorator-Tests."""

    def _make_update(self, chat_type: str = "private") -> MagicMock:
        """Erstellt ein gemocktes Telegram-Update mit Chat-Type."""
        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 111
        update.effective_user.username = "testuser"
        update.effective_chat = MagicMock()
        update.effective_chat.type = chat_type
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        return update

    async def test_allows_private_chat(self) -> None:
        """Im privaten Chat wird der Handler ausgefuehrt."""
        from presentation.decorators import require_private_chat

        handler = AsyncMock()
        decorated = require_private_chat(handler)

        update = self._make_update(chat_type="private")
        context = MagicMock()

        await decorated(update, context)

        handler.assert_called_once_with(update, context)

    async def test_blocks_group_chat(self) -> None:
        """In Gruppen-Chats wird der Handler blockiert."""
        from presentation.decorators import require_private_chat

        handler = AsyncMock()
        decorated = require_private_chat(handler)

        update = self._make_update(chat_type="group")
        context = MagicMock()

        await decorated(update, context)

        handler.assert_not_called()
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "private chat" in call_text.lower()

    async def test_blocks_supergroup_chat(self) -> None:
        """In Supergruppen-Chats wird der Handler blockiert."""
        from presentation.decorators import require_private_chat

        handler = AsyncMock()
        decorated = require_private_chat(handler)

        update = self._make_update(chat_type="supergroup")
        context = MagicMock()

        await decorated(update, context)

        handler.assert_not_called()
        update.message.reply_text.assert_called_once()


class TestParseWhitelist:
    """Tests für _parse_whitelist: Validierung von WHITELIST_USER_IDS."""

    def test_valid_ids_parsed(self) -> None:
        """Gültige komma-separierte IDs werden korrekt geparst."""
        with patch.dict("os.environ", {"WHITELIST_USER_IDS": "123,456,789"}):
            from presentation.decorators import _parse_whitelist

            result = _parse_whitelist()
            assert result == {123, 456, 789}

    def test_empty_string_returns_empty_set(self) -> None:
        """Leerer String ergibt leeres Set."""
        with patch.dict("os.environ", {"WHITELIST_USER_IDS": ""}):
            from presentation.decorators import _parse_whitelist

            result = _parse_whitelist()
            assert result == set()

    def test_malformed_entries_logged_and_ignored(self, caplog: object) -> None:
        """Ungültige Einträge werden als critical geloggt, gültige behalten."""
        with patch.dict("os.environ", {"WHITELIST_USER_IDS": "123,abc,456,def"}):
            from presentation.decorators import _parse_whitelist

            with caplog.at_level(logging.CRITICAL):  # type: ignore[union-attr]
                result = _parse_whitelist()

            assert result == {123, 456}
            assert any(
                "abc" in r.message
                for r in caplog.records  # type: ignore[union-attr]
            )
            assert any(
                "def" in r.message
                for r in caplog.records  # type: ignore[union-attr]
            )

    def test_whitespace_handling(self) -> None:
        """Whitespace um IDs wird korrekt getrimmt."""
        with patch.dict("os.environ", {"WHITELIST_USER_IDS": " 123 , 456 , "}):
            from presentation.decorators import _parse_whitelist

            result = _parse_whitelist()
            assert result == {123, 456}

    def test_single_id(self) -> None:
        """Einzelne ID (ohne Komma) wird korrekt geparst."""
        with patch.dict("os.environ", {"WHITELIST_USER_IDS": "999"}):
            from presentation.decorators import _parse_whitelist

            result = _parse_whitelist()
            assert result == {999}
