"""Tests for the @lcp_aware decorator.

Verifies that command handlers with user-authored text trigger
language detection via the LCP (Language Control Plane) in read-only mode.
The decorator uses resolve_readonly() so detection runs for logging/stats
but the user's sticky language is NEVER mutated by command arguments.

Bug context (Item 12): Commands like /remember, /learn called resolve()
which would smart-switch the sticky language based on English command
arguments. A user with sticky='fr' sending '/learn test english pattern'
would have their sticky overwritten to 'en'. Fix: use resolve_readonly().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from presentation.decorators import lcp_aware, _LCP_AWARE_MIN_CHARS


def _make_update(
    text: str,
    user_id: int = 12345,
    chat_id: int = 67890,
) -> MagicMock:
    """Build a minimal mocked Telegram Update for a command message."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = MagicMock()
    update.message.text = text
    return update


class TestLcpAwareDecorator:
    """Tests for the @lcp_aware decorator."""

    @pytest.mark.asyncio
    async def test_readonly_detection_on_clear_language_text(self) -> None:
        """German text triggers readonly detection (no sticky mutation).

        Previously this was a smart-switch scenario; now resolve_readonly()
        is used so sticky is never overwritten by command argument text.
        """
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        # Long enough German text (>= 15 chars)
        german_text = "ich mag Kirschenbaeume und Sonnenblumen"
        update = _make_update(f"/remember {german_text}")
        context = MagicMock()

        # Mock the LanguageResolver.resolve_readonly to track calls
        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock()

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        # Handler must still be called
        handler.assert_called_once_with(update, context)

        # LanguageResolver.resolve_readonly() must have been called with the
        # stripped text (without "/remember " prefix)
        mock_resolver_instance.resolve_readonly.assert_called_once_with(
            12345,  # user_id
            67890,  # chat_id
            german_text,
        )

    @pytest.mark.asyncio
    async def test_keeps_sticky_on_short_text(self) -> None:
        """Text shorter than _LCP_AWARE_MIN_CHARS does not trigger detection.

        Short commands like '/forget abc' should not run detection
        because the backends are unreliable on such short input.
        """
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        # Short text (< 15 chars)
        update = _make_update("/forget mem_42")
        context = MagicMock()

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock()

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        # Handler called
        handler.assert_called_once_with(update, context)

        # Resolver NOT called (text too short)
        mock_resolver_instance.resolve_readonly.assert_not_called()

    @pytest.mark.asyncio
    async def test_keeps_sticky_on_no_text_after_command(self) -> None:
        """Command with no arguments (e.g. bare '/learn') skips detection."""
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        update = _make_update("/learn")
        context = MagicMock()

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock()

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        handler.assert_called_once_with(update, context)
        mock_resolver_instance.resolve_readonly.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_runs_even_if_detection_fails(self) -> None:
        """If LanguageResolver.resolve_readonly() raises, handler still executes.

        The decorator is fire-and-forget: detection errors must not
        block the command handler.
        """
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        long_text = "x" * 20
        update = _make_update(f"/remember {long_text}")
        context = MagicMock()

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock(
            side_effect=RuntimeError("langdetect boom")
        )

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        # Handler MUST still execute despite the resolver error
        handler.assert_called_once_with(update, context)

    @pytest.mark.asyncio
    async def test_no_message_skips_detection(self) -> None:
        """Update without message (e.g. callback query) skips detection."""
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        update = MagicMock()
        update.message = None
        context = MagicMock()

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock()

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        handler.assert_called_once_with(update, context)
        mock_resolver_instance.resolve_readonly.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_command_text_uses_full_text(self) -> None:
        """If message.text does not start with '/', use full text for detection."""
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        # Edge case: somehow a non-command message goes through the decorator
        full_text = "Dies ist ein normaler Satz auf Deutsch"
        update = _make_update(full_text)
        context = MagicMock()

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock()

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        handler.assert_called_once_with(update, context)
        mock_resolver_instance.resolve_readonly.assert_called_once_with(
            12345, 67890, full_text
        )

    @pytest.mark.asyncio
    async def test_exactly_at_min_chars_triggers_detection(self) -> None:
        """Text with exactly _LCP_AWARE_MIN_CHARS characters triggers detection."""
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        # Create text that is exactly _LCP_AWARE_MIN_CHARS long
        user_text = "a" * _LCP_AWARE_MIN_CHARS
        update = _make_update(f"/learn {user_text}")
        context = MagicMock()

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock()

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        handler.assert_called_once_with(update, context)
        mock_resolver_instance.resolve_readonly.assert_called_once_with(
            12345, 67890, user_text
        )

    @pytest.mark.asyncio
    async def test_one_below_min_chars_skips_detection(self) -> None:
        """Text with _LCP_AWARE_MIN_CHARS - 1 characters skips detection."""
        handler = AsyncMock()
        decorated = lcp_aware(handler)

        user_text = "a" * (_LCP_AWARE_MIN_CHARS - 1)
        update = _make_update(f"/learn {user_text}")
        context = MagicMock()

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve_readonly = AsyncMock()

        with patch(
            "presentation.decorators.LanguageResolver",
            return_value=mock_resolver_instance,
        ):
            await decorated(update, context)

        handler.assert_called_once_with(update, context)
        mock_resolver_instance.resolve_readonly.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_functools_wraps(self) -> None:
        """Decorator preserves __name__ and __doc__ of the wrapped function."""

        async def my_handler(update, context):
            """My docstring."""

        decorated = lcp_aware(my_handler)
        assert decorated.__name__ == "my_handler"
        assert decorated.__doc__ == "My docstring."
