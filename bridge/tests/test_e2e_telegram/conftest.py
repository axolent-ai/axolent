"""E2E Telegram tests via tgintegration.

Skipped unless ALL of the following environment variables are set:
  - TELEGRAM_BOT_TOKEN_TEST: Token for the test bot instance
  - TELEGRAM_API_ID: API ID from my.telegram.org
  - TELEGRAM_API_HASH: API hash from my.telegram.org
  - TELEGRAM_TEST_ACCOUNT_SESSION: Pyrogram session string for the
    test Telegram account (generated via scripts/generate_test_session.py)

These tests require a RUNNING test bot instance (started via the
test_bot_session fixture) and a real Telegram account to send messages.

Run with:
    pytest -m e2e_telegram --run-e2e

See docs/E2E_TELEGRAM_TESTS.md for full setup instructions.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import AsyncGenerator

import pytest

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

_REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN_TEST",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "TELEGRAM_TEST_ACCOUNT_SESSION",
]


def _e2e_env_complete() -> bool:
    """Return True if all required E2E environment variables are set."""
    return all(os.environ.get(k) for k in _REQUIRED_ENV_VARS)


def _missing_env_vars() -> list[str]:
    """Return list of missing environment variable names."""
    return [k for k in _REQUIRED_ENV_VARS if not os.environ.get(k)]


_SKIP_REASON = (
    "E2E Telegram env not configured. "
    "Set TELEGRAM_BOT_TOKEN_TEST, TELEGRAM_API_ID, "
    "TELEGRAM_API_HASH, TELEGRAM_TEST_ACCOUNT_SESSION. "
    "See docs/E2E_TELEGRAM_TESTS.md"
)


# ---------------------------------------------------------------------------
# Auto-skip: applies to ALL tests collected from this directory
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Skip all e2e_telegram tests if environment is not configured."""
    if _e2e_env_complete():
        return

    skip_marker = pytest.mark.skip(reason=_SKIP_REASON)
    for item in items:
        # Only skip tests in this directory
        if "test_e2e_telegram" in str(item.fspath):
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _check_tgintegration():
    """Verify tgintegration is importable before running E2E tests."""
    try:
        import tgintegration  # noqa: F401
    except ImportError:
        pytest.skip(
            "tgintegration not installed. Install with: pip install tgintegration"
        )


@pytest.fixture(scope="session")
def bot_username() -> str:
    """Return the bot username from TELEGRAM_BOT_USERNAME_TEST env var.

    Falls back to 'axolent_test_bot' if not set.
    """
    return os.environ.get("TELEGRAM_BOT_USERNAME_TEST", "axolent_test_bot")


@pytest.fixture(scope="session")
async def test_bot_process() -> AsyncGenerator[subprocess.Popen, None]:
    """Start AXOLENT bot subprocess using TELEGRAM_BOT_TOKEN_TEST.

    Yields the process, kills it after all tests complete.
    The bot is started with the test token so it runs as a separate
    instance from production.
    """
    if not _e2e_env_complete():
        pytest.skip(_SKIP_REASON)

    bridge_dir = Path(__file__).resolve().parents[2]  # bridge/
    main_py = bridge_dir / "main.py"

    if not main_py.exists():
        pytest.skip(f"Bot main.py not found at {main_py}")

    env = os.environ.copy()
    env["TELEGRAM_BOT_TOKEN"] = os.environ["TELEGRAM_BOT_TOKEN_TEST"]
    # Ensure test mode
    env["AXOLENT_ENV"] = "test"
    env["AXOLENT_E2E_MODE"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(main_py)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(bridge_dir),
    )

    # Give the bot time to start and connect to Telegram
    await asyncio.sleep(5)

    if proc.poll() is not None:
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        pytest.fail(
            f"Bot process exited prematurely (code={proc.returncode}).\n"
            f"stdout: {stdout[:500]}\nstderr: {stderr[:500]}"
        )

    yield proc

    # Cleanup: terminate bot
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="session")
async def bot_controller(
    _check_tgintegration, test_bot_process, bot_username
) -> AsyncGenerator:
    """Create and initialize a tgintegration BotController.

    Uses the test Telegram account (via session string) to interact
    with the test bot.
    """
    from pyrogram import Client
    from tgintegration import BotController

    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    session_string = os.environ["TELEGRAM_TEST_ACCOUNT_SESSION"]

    client = Client(
        name="axolent_e2e_test",
        api_id=api_id,
        api_hash=api_hash,
        session_string=session_string,
        in_memory=True,
    )

    controller = BotController(
        client=client,
        peer=f"@{bot_username}",
        max_wait=30.0,  # LLM responses can be slow
        wait_consecutive=3.0,  # Wait for multi-message responses
        raise_no_response=True,
        global_action_delay=1.0,
    )

    async with client:
        await controller.initialize()
        yield controller


@pytest.fixture
async def fresh_chat(bot_controller) -> AsyncGenerator:
    """Ensure a clean state before each test by sending /reset.

    Waits for the bot to acknowledge the reset before yielding.
    """
    from tgintegration import BotController

    controller: BotController = bot_controller

    # Send /reset to clear any existing conversation state
    async with controller.collect(max_wait=15.0) as _reset_ack:  # noqa: F841
        await controller.send_command("reset")

    # Small delay to ensure state is fully cleared
    await asyncio.sleep(1)

    yield controller
