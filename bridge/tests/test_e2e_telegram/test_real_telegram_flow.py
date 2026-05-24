"""Legacy E2E Telegram test skeleton (superseded by test_user_journeys.py).

This file is kept for backwards compatibility. The real E2E user-journey
tests are in test_user_journeys.py (10 scenarios).

All tests here remain skipped. They served as Phase 1 placeholders and
are now fully implemented in the user-journeys module.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.e2e_telegram,
    pytest.mark.skipif(
        not os.getenv("TELEGRAM_TEST_API_ID"),
        reason="E2E Telegram tests require TELEGRAM_TEST_API_ID env var",
    ),
]


class TestTelegramE2EFlow:
    """Legacy skeleton. See test_user_journeys.py for real implementations."""

    @pytest.mark.skip(reason="Superseded by test_user_journeys.py")
    async def test_start_command_returns_welcome(self) -> None:
        """Send /start to bot, verify welcome message is returned."""

    @pytest.mark.skip(reason="Superseded by test_user_journeys.py")
    async def test_german_message_gets_german_response(self) -> None:
        """Send German text, verify response is in German."""

    @pytest.mark.skip(reason="Superseded by test_user_journeys.py")
    async def test_lang_command_switches_language(self) -> None:
        """Send /lang en, then English text, verify English response."""

    @pytest.mark.skip(reason="Superseded by test_user_journeys.py")
    async def test_reset_clears_session(self) -> None:
        """Send /reset, verify session state is cleared."""

    @pytest.mark.skip(reason="Superseded by test_user_journeys.py")
    async def test_streaming_response_completeness(self) -> None:
        """Send a question, verify the streamed response is complete."""
