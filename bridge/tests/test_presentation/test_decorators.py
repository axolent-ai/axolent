"""Tests fuer presentation.decorators: Whitelist-Guard.

Testet Autorisierung via User-ID-Whitelist und ALLOW_ALL_USERS Bypass.
Mockt Telegram Update/Context Objekte.
"""

from __future__ import annotations

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
            assert "nicht autorisiert" in call_text.lower()

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
