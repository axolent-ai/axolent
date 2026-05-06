"""Bookmark-Service: Use-Case-Koordination für Bookmark-Operationen.

Orchestriert bookmark_storage (Persistenz) und stellt
eine saubere API für die Presentation-Layer bereit.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from infrastructure.bookmark_storage import (
    bookmark_exists,
    delete_bookmark,
    get_bookmark_by_message_id,
    list_recent_bookmarks,
    save_bookmark,
    search_bookmarks,
)

log = logging.getLogger(__name__)


def save_or_toggle_bookmark(
    user_id: int,
    username: Optional[str],
    chat_id: int,
    message_id: int,
    content: str,
) -> tuple[bool, str]:
    """Speichert oder entfernt einen Bookmark (Toggle-Logik).

    Wenn der Bookmark bereits existiert, wird er entfernt.
    Wenn er nicht existiert, wird er gespeichert.

    Args:
        user_id: Telegram User-ID.
        username: Telegram Username.
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID.
        content: Volltext der Nachricht.

    Returns:
        Tuple (was_saved, user_message):
            was_saved=True wenn gespeichert, False wenn entfernt.
            user_message: Bestätigungstext für den User.
    """
    if bookmark_exists(user_id, chat_id, message_id):
        delete_bookmark(user_id, chat_id, message_id)
        log.info(
            "Bookmark entfernt via toggle: user_id=%d chat_id=%d message_id=%d",
            user_id,
            chat_id,
            message_id,
        )
        return False, "Bookmark entfernt"

    save_bookmark(
        user_id=user_id,
        username=username,
        message_id=message_id,
        chat_id=chat_id,
        content=content,
    )
    return True, "Bookmark gespeichert"


def list_bookmarks(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    """Gibt die neuesten Bookmarks eines Users zurück.

    Args:
        user_id: Telegram User-ID.
        limit: Maximale Anzahl.

    Returns:
        Liste von Bookmark-Dicts, neueste zuerst.
    """
    return list_recent_bookmarks(user_id, limit=limit)


def search(user_id: int, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Sucht Bookmarks per Inhalts-Substring.

    Args:
        user_id: Telegram User-ID.
        query: Suchbegriff.
        limit: Max Ergebnisse.

    Returns:
        Liste passender Bookmark-Dicts.
    """
    return search_bookmarks(user_id, query, limit=limit)


def get_bookmark(
    user_id: int, chat_id: int, message_id: int
) -> Optional[dict[str, Any]]:
    """Findet einen einzelnen Bookmark.

    Args:
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID.

    Returns:
        Bookmark-Dict oder None.
    """
    return get_bookmark_by_message_id(user_id, chat_id, message_id)


def remove_bookmark(user_id: int, chat_id: int, message_id: int) -> bool:
    """Löscht einen Bookmark.

    Args:
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID.

    Returns:
        True wenn gelöscht, False wenn nicht gefunden.
    """
    return delete_bookmark(user_id, chat_id, message_id)
