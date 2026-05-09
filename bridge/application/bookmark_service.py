"""Bookmark-Service: Use-Case-Koordination für Bookmark-Operationen.

Orchestriert BookmarkStorage (Persistenz) und stellt
eine saubere API für die Presentation-Layer bereit.

Seit V6: Konstruktor-Injection statt Modul-Global.
BookmarkService bekommt das Storage-Backend via __init__,
identisch zum ChatService-Pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol


log = logging.getLogger(__name__)


class BookmarkStorage(Protocol):
    """Protocol für austauschbare Bookmark-Storage-Backends.

    Implementiert von SqliteBookmarkStorage und den JSONL-Funktionen.
    """

    def save_bookmark(
        self,
        user_id: int,
        username: Optional[str],
        message_id: int,
        chat_id: int,
        content: str,
    ) -> dict[str, Any]: ...

    def list_recent_bookmarks(
        self, user_id: int, limit: int = 10
    ) -> list[dict[str, Any]]: ...

    def search_bookmarks(
        self, user_id: int, query: str, limit: int = 20
    ) -> list[dict[str, Any]]: ...

    def get_bookmark_by_message_id(
        self, user_id: int, chat_id: int, message_id: int
    ) -> Optional[dict[str, Any]]: ...

    def bookmark_exists(self, user_id: int, chat_id: int, message_id: int) -> bool: ...

    def delete_bookmark(self, user_id: int, chat_id: int, message_id: int) -> bool: ...


class BookmarkService:
    """Application-Layer Service für Bookmark-Operationen.

    Bekommt das Storage-Backend via Konstruktor-Injection.
    Enthält Toggle-Logik und Business-Orchestration.
    """

    def __init__(self, storage: BookmarkStorage) -> None:
        self._storage = storage

    def save_or_toggle_bookmark(
        self,
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
        if self._storage.bookmark_exists(user_id, chat_id, message_id):
            self._storage.delete_bookmark(user_id, chat_id, message_id)
            log.info(
                "Bookmark entfernt via toggle: user_id=%d chat_id=%d message_id=%d",
                user_id,
                chat_id,
                message_id,
            )
            return False, "Bookmark entfernt"

        self._storage.save_bookmark(
            user_id=user_id,
            username=username,
            message_id=message_id,
            chat_id=chat_id,
            content=content,
        )
        return True, "Bookmark gespeichert"

    def list_bookmarks(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        """Gibt die neuesten Bookmarks eines Users zurück.

        Args:
            user_id: Telegram User-ID.
            limit: Maximale Anzahl.

        Returns:
            Liste von Bookmark-Dicts, neueste zuerst.
        """
        return self._storage.list_recent_bookmarks(user_id, limit=limit)

    def search(self, user_id: int, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Sucht Bookmarks per Inhalts-Substring.

        Args:
            user_id: Telegram User-ID.
            query: Suchbegriff.
            limit: Max Ergebnisse.

        Returns:
            Liste passender Bookmark-Dicts.
        """
        return self._storage.search_bookmarks(user_id, query, limit=limit)

    def get_bookmark(
        self, user_id: int, chat_id: int, message_id: int
    ) -> Optional[dict[str, Any]]:
        """Findet einen einzelnen Bookmark.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            message_id: Telegram Message-ID.

        Returns:
            Bookmark-Dict oder None.
        """
        return self._storage.get_bookmark_by_message_id(user_id, chat_id, message_id)

    def remove_bookmark(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Löscht einen Bookmark.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            message_id: Telegram Message-ID.

        Returns:
            True wenn gelöscht, False wenn nicht gefunden.
        """
        return self._storage.delete_bookmark(user_id, chat_id, message_id)
