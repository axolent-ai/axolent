"""Conversation storage: in-memory conversation history per (user_id, chat_id).

Thread-safe via asyncio.Lock. Stores conversation turns and sticky language.
Language is now also persisted to SQLite (survives bot restart).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from domain.conversation import ConversationTurn, MAX_HISTORY_TURNS

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteLanguageStorage

log = logging.getLogger(__name__)

# Storage: indexed by (user_id, chat_id)
_histories: dict[tuple[int, int], list[ConversationTurn]] = {}
_languages: dict[tuple[int, int], str] = {}
_lock = asyncio.Lock()

# Optional SQLite backing store for language persistence
_language_storage: "SqliteLanguageStorage | None" = None


def init_language_storage(storage: "SqliteLanguageStorage") -> None:
    """Initialize the persistent language storage and load existing languages.

    Called once at bot startup after SQLite is initialized.

    Args:
        storage: SqliteLanguageStorage instance.
    """
    global _language_storage
    _language_storage = storage
    # Populate in-memory cache from persistent store
    persisted = storage.load_all()
    _languages.update(persisted)
    if persisted:
        log.info(
            "Loaded %d persistent language preferences from SQLite", len(persisted)
        )


async def save_turn(user_id: int, chat_id: int, turn: ConversationTurn) -> None:
    """Append a turn to the conversation history. Enforces MAX_HISTORY_TURNS.

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
        # LRU trim: keep only the last MAX_HISTORY_TURNS
        if len(_histories[key]) > MAX_HISTORY_TURNS:
            _histories[key] = _histories[key][-MAX_HISTORY_TURNS:]


async def get_history(user_id: int, chat_id: int) -> list[ConversationTurn]:
    """Return the conversation history for a user/chat pair.

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
    """Clear conversation history for a user/chat pair.

    NOTE: Does NOT clear the sticky language anymore (since persistent
    language storage). The user's language preference survives /reset.
    Use /lang to explicitly change the language.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
    """
    key = (user_id, chat_id)
    async with _lock:
        _histories.pop(key, None)
    log.info("Conversation reset for user=%d chat=%d", user_id, chat_id)


async def set_language(user_id: int, chat_id: int, lang: str) -> None:
    """Set the sticky language for a user/chat pair.

    Persists to SQLite if the backing store is initialized.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        lang: ISO-639-1 language code (e.g. "de", "en").
    """
    key = (user_id, chat_id)
    async with _lock:
        _languages[key] = lang
    # Persist to SQLite (survives bot restart)
    if _language_storage is not None:
        try:
            _language_storage.set_language(user_id, chat_id, lang)
        except Exception as e:
            log.warning("Failed to persist language to SQLite: %s", e)
    log.info("Sticky language set to '%s' for user=%d chat=%d", lang, user_id, chat_id)


def _reset_all_for_tests() -> None:
    """For tests: reset the entire conversation storage.

    Clears all histories and languages. ONLY use in tests,
    not in production code.
    """
    _histories.clear()
    _languages.clear()


async def get_language(user_id: int, chat_id: int) -> Optional[str]:
    """Return the sticky language for a user/chat pair.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.

    Returns:
        Language code or None if not yet set.
    """
    key = (user_id, chat_id)
    async with _lock:
        return _languages.get(key)
