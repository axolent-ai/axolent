"""Tests for UX fixes: slash sanitizer and /stop command."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from presentation.render import sanitize_telegram_slashes


# ---------------------------------------------------------------------------
# TestSlashSanitizer
# ---------------------------------------------------------------------------


class TestSlashSanitizer:
    """Tests for sanitize_telegram_slashes()."""

    def test_replace_slash_before_letter(self) -> None:
        assert sanitize_telegram_slashes("Use /reset to clear") == "Use ⁄reset to clear"

    def test_preserve_path_slashes(self) -> None:
        assert sanitize_telegram_slashes("path/to/file") == "path⁄to⁄file"

    def test_slash_with_space_untouched(self) -> None:
        assert sanitize_telegram_slashes("50 / 100") == "50 / 100"

    def test_multiple_slashes(self) -> None:
        result = sanitize_telegram_slashes("/help and /reset and /stop")
        assert "⁄help" in result
        assert "⁄reset" in result
        assert "⁄stop" in result

    def test_empty_string(self) -> None:
        assert sanitize_telegram_slashes("") == ""


# ---------------------------------------------------------------------------
# TestStopCommand
# ---------------------------------------------------------------------------


def _make_update(user_id: int = 1, chat_id: int = 10) -> MagicMock:
    """Creates a mocked Telegram Update."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


def _make_context() -> MagicMock:
    """Creates a mocked Telegram Context with chat_service."""
    mock_chat_service = MagicMock()
    mock_chat_service.get_chat_language = AsyncMock(return_value=None)

    context = MagicMock()
    context.args = []
    context.bot = MagicMock()
    context.application = MagicMock()
    context.application.bot_data = {
        "chat_service": mock_chat_service,
        "system_prompt": "test",
        "memory_service": None,
        "persistent_provider": None,
        "process_pool": None,
        "rate_limiter": None,
        "bookmark_service": None,
    }
    return context


class TestStopCommand:
    """Tests for handle_stop_command()."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass for tests."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_stop_no_active_stream(self) -> None:
        """When no stream is active, /stop replies with no_active_stream message."""
        from presentation.handlers import handle_stop_command

        update = _make_update()
        context = _make_context()

        with (
            patch("presentation.handlers._active_streaming_sessions", {}),
            patch("presentation.handlers.log_command_audit"),
        ):
            await handle_stop_command(update, context)

        update.message.reply_text.assert_called_once()
