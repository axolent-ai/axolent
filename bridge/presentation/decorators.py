"""Telegram decorators: whitelist check, privacy guard, LCP-aware, and other guards.

Contains decorator functions that run before handler logic.
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Callable

from telegram import Update
from telegram.ext import ContextTypes

from application.language.resolver import LanguageResolver
from i18n.domain.i18n import t

log = logging.getLogger(__name__)

# Minimum character count for the user-text portion of a command
# before LCP detection is triggered.  Below this threshold the
# detection backends produce unreliable results (e.g. langdetect
# returns "fi" for "Hallo").
_LCP_AWARE_MIN_CHARS: int = 15


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
                await update.message.reply_text(t("errors.not_authorized", "en"))
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


def lcp_aware(func: Callable) -> Callable:
    """Decorator: runs LCP language detection on user text before the handler.

    Commands like /remember, /learn, /explain carry user-authored text
    that may differ from the current sticky language.  This decorator
    uses resolve_readonly() so that detection runs (for logging/stats)
    but the user's sticky language is NEVER mutated by command arguments.

    Behaviour:
        1. Extracts the user-text portion (everything after "/command ").
        2. If the text is >= _LCP_AWARE_MIN_CHARS characters, runs the
           read-only LanguageResolver.resolve_readonly() pipeline which:
           a) detects the language of the text,
           b) returns a LanguageContext for logging/stats purposes,
           c) does NOT perform a smart-switch or write to storage.
        3. The handler then executes as usual; it reads the unchanged
           sticky language for its response.

    This decorator is intentionally fire-and-forget: if detection or
    the resolver raises, the error is logged and the handler proceeds
    with the existing sticky language.
    """

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message and message.text:
            raw = message.text
            # Strip "/command " prefix to get pure user text
            text = raw
            if raw.startswith("/"):
                parts = raw.split(maxsplit=1)
                text = parts[1] if len(parts) > 1 else ""

            if text and len(text) >= _LCP_AWARE_MIN_CHARS:
                user = update.effective_user
                user_id: int = user.id if user else 0
                chat_id: int = update.effective_chat.id if update.effective_chat else 0
                try:
                    resolver = LanguageResolver()
                    await resolver.resolve_readonly(user_id, chat_id, text)
                except Exception:
                    log.debug(
                        "lcp_aware: detection failed for user=%d, proceeding",
                        user.id if user else 0,
                        exc_info=True,
                    )
        return await func(update, context)

    return wrapper
