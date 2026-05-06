"""Conversation Storage: In-memory conversation history per (user_id, chat_id).

Thread-safe via asyncio.Lock. Stores conversation turns and sticky language.
Designed for later migration to SQLite if persistence is needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from domain.conversation import ConversationTurn, MAX_HISTORY_TURNS

log = logging.getLogger(__name__)

# Storage: keyed by (user_id, chat_id)
_histories: dict[tuple[int, int], list[ConversationTurn]] = {}
_languages: dict[tuple[int, int], str] = {}
_lock = asyncio.Lock()


async def save_turn(user_id: int, chat_id: int, turn: ConversationTurn) -> None:
    """Appends a turn to the conversation history. Enforces MAX_HISTORY_TURNS.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        turn: The ConversationTurn to store.
    """
    key = (user_id, chat_id)
    async with _lock:
        if key not in _histories:
            _histories[key] = []
        _histories[key].append(turn)
        # LRU trim: keep only last MAX_HISTORY_TURNS
        if len(_histories[key]) > MAX_HISTORY_TURNS:
            _histories[key] = _histories[key][-MAX_HISTORY_TURNS:]


async def get_history(user_id: int, chat_id: int) -> list[ConversationTurn]:
    """Returns the conversation history for a given user/chat pair.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.

    Returns:
        List of ConversationTurns (may be empty).
    """
    key = (user_id, chat_id)
    async with _lock:
        return list(_histories.get(key, []))


async def reset_conversation(user_id: int, chat_id: int) -> None:
    """Clears conversation history AND sticky language for a user/chat pair.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
    """
    key = (user_id, chat_id)
    async with _lock:
        _histories.pop(key, None)
        _languages.pop(key, None)
    log.info("Conversation reset for user=%d chat=%d", user_id, chat_id)


async def set_language(user_id: int, chat_id: int, lang: str) -> None:
    """Sets the sticky language for a user/chat pair.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        lang: ISO-639-1 language code (e.g. "de", "en").
    """
    key = (user_id, chat_id)
    async with _lock:
        _languages[key] = lang
    log.info("Sticky language set to '%s' for user=%d chat=%d", lang, user_id, chat_id)


async def get_language(user_id: int, chat_id: int) -> Optional[str]:
    """Gets the sticky language for a user/chat pair.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.

    Returns:
        Language code or None if not set yet.
    """
    key = (user_id, chat_id)
    async with _lock:
        return _languages.get(key)
