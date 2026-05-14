"""Bookmark service: use-case coordination for bookmark operations.

Orchestrates BookmarkStorage (persistence) and provides
a clean API for the presentation layer.

Since V6: constructor injection instead of module globals.
BookmarkService receives the storage backend via __init__,
identical to the ChatService pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol


log = logging.getLogger(__name__)


class BookmarkStorage(Protocol):
    """Protocol for swappable bookmark storage backends.

    Implemented by SqliteBookmarkStorage and the JSONL functions.
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
    """Application-layer service for bookmark operations.

    Receives the storage backend via constructor injection.
    Contains toggle logic and business orchestration.
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
        """Save or remove a bookmark (toggle logic).

        If the bookmark already exists, it is removed.
        If it does not exist, it is saved.

        Args:
            user_id: Telegram user ID.
            username: Telegram username.
            chat_id: Telegram chat ID.
            message_id: Telegram message ID.
            content: Full text of the message.

        Returns:
            Tuple (was_saved, user_message):
                was_saved=True if saved, False if removed.
                user_message: Confirmation text for the user.
        """
        if self._storage.bookmark_exists(user_id, chat_id, message_id):
            self._storage.delete_bookmark(user_id, chat_id, message_id)
            log.info(
                "Bookmark removed via toggle: user_id=%d chat_id=%d message_id=%d",
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
        """Return the most recent bookmarks for a user.

        Args:
            user_id: Telegram user ID.
            limit: Maximum number of results.

        Returns:
            List of bookmark dicts, newest first.
        """
        return self._storage.list_recent_bookmarks(user_id, limit=limit)

    def search(self, user_id: int, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search bookmarks by content substring.

        Args:
            user_id: Telegram user ID.
            query: Search term.
            limit: Max results.

        Returns:
            List of matching bookmark dicts.
        """
        return self._storage.search_bookmarks(user_id, query, limit=limit)

    def get_bookmark(
        self, user_id: int, chat_id: int, message_id: int
    ) -> Optional[dict[str, Any]]:
        """Find a single bookmark.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            message_id: Telegram message ID.

        Returns:
            Bookmark dict or None.
        """
        return self._storage.get_bookmark_by_message_id(user_id, chat_id, message_id)

    def remove_bookmark(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Delete a bookmark.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            message_id: Telegram message ID.

        Returns:
            True if deleted, False if not found.
        """
        return self._storage.delete_bookmark(user_id, chat_id, message_id)
