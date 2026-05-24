"""End-to-end Telegram tests using tgintegration.

This package contains tests that spawn a real Telegram bot session
and test user interactions through the actual Telegram API.

Tests are skipped by default unless the E2E environment variables are
configured. See conftest.py for required variables and
docs/E2E_TELEGRAM_TESTS.md for full setup instructions.

Run with:
    pytest -m e2e_telegram --run-e2e -v
"""
