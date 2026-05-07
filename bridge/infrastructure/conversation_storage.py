"""Conversation Storage: In-Memory Conversation-History pro (user_id, chat_id).

Thread-safe via asyncio.Lock. Speichert Conversation-Turns und Sticky-Language.
Vorbereitet für spätere Migration auf SQLite falls Persistenz nötig.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from domain.conversation import ConversationTurn, MAX_HISTORY_TURNS

log = logging.getLogger(__name__)

# Storage: indiziert nach (user_id, chat_id)
_histories: dict[tuple[int, int], list[ConversationTurn]] = {}
_languages: dict[tuple[int, int], str] = {}
_lock = asyncio.Lock()


async def save_turn(user_id: int, chat_id: int, turn: ConversationTurn) -> None:
    """Hängt einen Turn an die Conversation-History an. Erzwingt MAX_HISTORY_TURNS.

    Args:
        user_id: Telegram-User-ID.
        chat_id: Telegram-Chat-ID.
        turn: Der zu speichernde ConversationTurn.
    """
    key = (user_id, chat_id)
    async with _lock:
        if key not in _histories:
            _histories[key] = []
        _histories[key].append(turn)
        # LRU-Trim: nur die letzten MAX_HISTORY_TURNS behalten
        if len(_histories[key]) > MAX_HISTORY_TURNS:
            _histories[key] = _histories[key][-MAX_HISTORY_TURNS:]


async def get_history(user_id: int, chat_id: int) -> list[ConversationTurn]:
    """Gibt die Conversation-History für ein User/Chat-Paar zurück.

    Args:
        user_id: Telegram-User-ID.
        chat_id: Telegram-Chat-ID.

    Returns:
        Liste von ConversationTurns (kann leer sein).
    """
    key = (user_id, chat_id)
    async with _lock:
        return list(_histories.get(key, []))


async def reset_conversation(user_id: int, chat_id: int) -> None:
    """Löscht Conversation-History UND Sticky-Language für ein User/Chat-Paar.

    Args:
        user_id: Telegram-User-ID.
        chat_id: Telegram-Chat-ID.
    """
    key = (user_id, chat_id)
    async with _lock:
        _histories.pop(key, None)
        _languages.pop(key, None)
    log.info("Conversation reset for user=%d chat=%d", user_id, chat_id)


async def set_language(user_id: int, chat_id: int, lang: str) -> None:
    """Setzt die Sticky-Language für ein User/Chat-Paar.

    Args:
        user_id: Telegram-User-ID.
        chat_id: Telegram-Chat-ID.
        lang: ISO-639-1-Sprachcode (z.B. "de", "en").
    """
    key = (user_id, chat_id)
    async with _lock:
        _languages[key] = lang
    log.info("Sticky language set to '%s' for user=%d chat=%d", lang, user_id, chat_id)


def _reset_all_for_tests() -> None:
    """Für Tests: setzt das gesamte Conversation-Storage zurück.

    Löscht alle Histories und Languages. NUR in Tests verwenden,
    nicht im Produktionscode.
    """
    _histories.clear()
    _languages.clear()


async def get_language(user_id: int, chat_id: int) -> Optional[str]:
    """Gibt die Sticky-Language für ein User/Chat-Paar zurück.

    Args:
        user_id: Telegram-User-ID.
        chat_id: Telegram-Chat-ID.

    Returns:
        Sprachcode oder None falls noch nicht gesetzt.
    """
    key = (user_id, chat_id)
    async with _lock:
        return _languages.get(key)
