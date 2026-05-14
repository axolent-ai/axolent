"""Telegram decorators: whitelist check, privacy guard, and other guards.

Contains decorator functions that run before handler logic.
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Callable

from telegram import Update
from telegram.ext import ContextTypes

log = logging.getLogger(__name__)


def _parse_whitelist() -> set[int]:
    """Parses WHITELIST_USER_IDS, logs invalid entries as critical.

    Returns:
        Set of valid user IDs.
    """
    raw = os.getenv("WHITELIST_USER_IDS", "")
    valid: set[int] = set()
    invalid: list[str] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            valid.add(int(stripped))
        else:
            invalid.append(stripped)
    if invalid:
        log.critical(
            "WHITELIST_USER_IDS contains invalid entries (ignored): %s",
            invalid,
        )
    return valid


# Whitelist configuration (loaded once at import time)
WHITELIST: set[int] = _parse_whitelist()
ALLOW_ALL_USERS: bool = os.getenv("ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")

_PRIVATE_ONLY_MSG = (
    "This command only works in a private chat with the bot. "
    "Please message me directly."
)


def require_whitelist(func: Callable) -> Callable:
    """Decorator: checks whether the user is on the whitelist.

    When ALLOW_ALL_USERS=true, everyone is allowed through.
    Otherwise only users whose ID is in WHITELIST_USER_IDS.

    On rejection: sends an error message and terminates the handler.
    """

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if ALLOW_ALL_USERS:
            return await func(update, context)

        user = update.effective_user
        user_id: int = user.id if user else 0

        if user_id not in WHITELIST:
            username = user.username if user else None
            log.warning(
                "Unauthorized access: user_id=%s username=%s", user_id, username
            )
            if update.message:
                await update.message.reply_text(
                    "You are not authorized to use this bot."
                )
            return

        return await func(update, context)

    return wrapper


def require_private_chat(func: Callable) -> Callable:
    """Decorator: allows command only in private 1:1 chat with the bot.

    In groups/supergroups an error message is sent and the handler is aborted.
    """

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat and update.effective_chat.type != "private":
            if update.message:
                await update.message.reply_text(_PRIVATE_ONLY_MSG)
            return
        return await func(update, context)

    return wrapper
