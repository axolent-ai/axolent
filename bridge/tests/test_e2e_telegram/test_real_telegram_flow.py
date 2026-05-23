"""End-to-end Telegram tests using tgintegration.

Spawns a real Telegram bot session and tests user interactions
through the actual Telegram API. Marked @pytest.mark.skip by
default since they require:
  * TELEGRAM_BOT_TOKEN (test bot, not production)
  * TELEGRAM_TEST_API_ID + TELEGRAM_TEST_API_HASH
  * Real Telegram connection

Run with: pytest tests/test_e2e_telegram/ -m e2e --run-e2e

Phase 2 will add actual test methods for:
  * Basic /start flow and welcome message
  * Language detection round-trip (send German, get German response)
  * /lang command switching
  * /reset clears session state
  * /skills listing
  * Streaming response integrity
  * Multi-message conversation memory
  * Rate limiting behavior under rapid messages
  * Privacy guard: PII in message is not leaked in logs
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.skipif(
        not os.getenv("TELEGRAM_TEST_API_ID"),
        reason="E2E Telegram tests require TELEGRAM_TEST_API_ID env var",
    ),
    pytest.mark.skipif(
        not os.getenv("TELEGRAM_TEST_API_HASH"),
        reason="E2E Telegram tests require TELEGRAM_TEST_API_HASH env var",
    ),
    pytest.mark.skipif(
        not os.getenv("TELEGRAM_TEST_BOT_TOKEN"),
        reason="E2E Telegram tests require TELEGRAM_TEST_BOT_TOKEN env var",
    ),
]


class TestTelegramE2EFlow:
    """Skeleton for real Telegram E2E tests.

    These tests will use tgintegration to:
    1. Connect as a real Telegram user (test account)
    2. Send messages to the AXOLENT bot (test instance)
    3. Assert on the bot's responses

    All tests in this class require real Telegram credentials
    and a running bot instance. They are skipped in CI by default.

    Setup for local E2E testing:
        export TELEGRAM_TEST_API_ID=<your_test_api_id>
        export TELEGRAM_TEST_API_HASH=<your_test_api_hash>
        export TELEGRAM_TEST_BOT_TOKEN=<test_bot_token>
        pytest tests/test_e2e_telegram/ -v
    """

    @pytest.mark.skip(reason="Phase 2: implement after tgintegration setup")
    async def test_start_command_returns_welcome(self) -> None:
        """Send /start to bot, verify welcome message is returned."""

    @pytest.mark.skip(reason="Phase 2: implement after tgintegration setup")
    async def test_german_message_gets_german_response(self) -> None:
        """Send German text, verify response is in German."""

    @pytest.mark.skip(reason="Phase 2: implement after tgintegration setup")
    async def test_lang_command_switches_language(self) -> None:
        """Send /lang en, then English text, verify English response."""

    @pytest.mark.skip(reason="Phase 2: implement after tgintegration setup")
    async def test_reset_clears_session(self) -> None:
        """Send /reset, verify session state is cleared."""

    @pytest.mark.skip(reason="Phase 2: implement after tgintegration setup")
    async def test_streaming_response_completeness(self) -> None:
        """Send a question, verify the streamed response is complete."""
