#!/usr/bin/env python3
"""Generate a Pyrogram session string for E2E Telegram tests.

This interactive script helps you create a session string that allows
tgintegration to send messages as a real Telegram user (your test account).

The session string is a portable representation of the Telegram auth session.
Once generated, set it as TELEGRAM_TEST_ACCOUNT_SESSION in your .env.test.

Requirements:
    pip install pyrogram tgcrypto

Usage:
    python scripts/generate_test_session.py

What happens:
    1. You provide API ID + API Hash (from my.telegram.org)
    2. You provide the phone number of your TEST Telegram account
    3. Telegram sends an auth code to that account
    4. You enter the code here
    5. The script outputs the session string

IMPORTANT: Use a SEPARATE test account, not your personal Telegram account!
"""

from __future__ import annotations

import asyncio
import sys


async def main() -> None:
    """Run the interactive session string generator."""
    try:
        from pyrogram import Client
    except ImportError:
        print("ERROR: pyrogram is not installed.")
        print("Install with: pip install pyrogram tgcrypto")
        sys.exit(1)

    print("=" * 60)
    print("  AXOLENT E2E Test Session String Generator")
    print("=" * 60)
    print()
    print("This will generate a Pyrogram session string for your")
    print("TEST Telegram account. You need:")
    print("  1. API ID from my.telegram.org")
    print("  2. API Hash from my.telegram.org")
    print("  3. Phone number of your TEST account")
    print()
    print("IMPORTANT: Use a SEPARATE test account!")
    print("           Do NOT use your personal Telegram account.")
    print()
    print("-" * 60)

    api_id_str = input("API ID: ").strip()
    try:
        api_id = int(api_id_str)
    except ValueError:
        print(f"ERROR: API ID must be a number, got: '{api_id_str}'")
        sys.exit(1)

    api_hash = input("API Hash: ").strip()
    if not api_hash or len(api_hash) < 10:
        print("ERROR: API Hash looks invalid (too short)")
        sys.exit(1)

    print()
    print("Now connecting to Telegram to authenticate...")
    print("You will receive an auth code on your test account.")
    print()

    async with Client(
        name="axolent_session_generator",
        api_id=api_id,
        api_hash=api_hash,
        in_memory=True,
    ) as client:
        session_string = await client.export_session_string()

    print()
    print("=" * 60)
    print("  SESSION STRING GENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Add this to your .env.test file:")
    print()
    print(f"TELEGRAM_TEST_ACCOUNT_SESSION={session_string}")
    print()
    print("-" * 60)
    print()
    print("Full .env.test template:")
    print()
    print(f"TELEGRAM_API_ID={api_id}")
    print(f"TELEGRAM_API_HASH={api_hash}")
    print(f"TELEGRAM_TEST_ACCOUNT_SESSION={session_string}")
    print("TELEGRAM_BOT_TOKEN_TEST=<your_test_bot_token_from_botfather>")
    print("TELEGRAM_BOT_USERNAME_TEST=<your_test_bot_username>")
    print()
    print("Keep this file SECRET. It grants access to your test account!")


if __name__ == "__main__":
    asyncio.run(main())
