"""Telegram-Decorators: Whitelist-Check und andere Guards.

Enthält Decorator-Funktionen die vor Handler-Logik ausgeführt werden.
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Callable

from telegram import Update
from telegram.ext import ContextTypes

log = logging.getLogger(__name__)

# Whitelist-Konfiguration (einmal beim Import geladen)
WHITELIST: set[int] = {
    int(uid)
    for uid in os.getenv("WHITELIST_USER_IDS", "").split(",")
    if uid.strip().isdigit()
}
ALLOW_ALL_USERS: bool = os.getenv("ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")


def require_whitelist(func: Callable) -> Callable:
    """Decorator: Prüft ob der User auf der Whitelist steht.

    Wenn ALLOW_ALL_USERS=true, wird jeder durchgelassen.
    Sonst nur User deren ID in WHITELIST_USER_IDS steht.

    Bei Ablehnung: sendet eine Fehlermeldung und beendet den Handler.
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
                "Unautorisierter Zugriff: user_id=%s username=%s", user_id, username
            )
            if update.message:
                await update.message.reply_text(
                    "Du bist nicht autorisiert, diesen Bot zu nutzen."
                )
            return

        return await func(update, context)

    return wrapper
