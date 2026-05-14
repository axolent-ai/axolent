"""Tests for application.bookmark_service: bookmark use-case orchestration.

Tests toggle logic, user scoping, and search.
Uses tmp_path for isolated test data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from filelock import FileLock

from application.bookmark_service import BookmarkService
from infrastructure.bookmark_storage import JsonlBookmarkStorageAdapter


class TestBookmarkService:
    """Bookmark-Service Use-Case-Tests."""

    @pytest.fixture(autouse=True)
    def _isolate_storage(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage auf tmp_path und erstellt Service."""
        self.bm_path = tmp_path / "bookmarks.jsonl"
        self.lock_path = str(self.bm_path) + ".lock"
        new_lock = FileLock(self.lock_path)

        self._patches = [
            patch("infrastructure.bookmark_storage.BOOKMARKS_PATH", self.bm_path),
            patch("infrastructure.bookmark_storage._BM_LOCK_PATH", self.lock_path),
            patch("infrastructure.bookmark_storage._BM_LOCK", new_lock),
        ]
        for p in self._patches:
            p.start()

        self.svc = BookmarkService(storage=JsonlBookmarkStorageAdapter())

        yield  # type: ignore[misc]

        for p in self._patches:
            p.stop()

    def test_save_or_toggle_creates_then_deletes(self) -> None:
        """Erster Aufruf speichert, zweiter Aufruf löscht (Toggle)."""
        # Erster Aufruf: speichern
        was_saved, msg = self.svc.save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=100, content="Test"
        )
        assert was_saved is True
        assert "saved" in msg.lower()

        # Zweiter Aufruf: löschen
        was_saved, msg = self.svc.save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=100, content="Test"
        )
        assert was_saved is False
        assert "removed" in msg.lower()

    def test_list_returns_user_specific(self) -> None:
        """list_bookmarks gibt nur Bookmarks des angegebenen Users zurück."""
        self.svc.save_or_toggle_bookmark(
            user_id=1, username="a", chat_id=10, message_id=1, content="User 1"
        )
        self.svc.save_or_toggle_bookmark(
            user_id=2, username="b", chat_id=10, message_id=2, content="User 2"
        )

        result = self.svc.list_bookmarks(user_id=1)
        assert len(result) == 1
        assert result[0]["content"] == "User 1"

    def test_search_filters_by_query(self) -> None:
        """search() filtert nach Content-Substring."""
        self.svc.save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=1, content="Python Code"
        )
        self.svc.save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=2, content="Rust Guide"
        )
        self.svc.save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=3, content="Python ML"
        )

        results = self.svc.search(user_id=1, query="Python")
        assert len(results) == 2
        for r in results:
            assert "Python" in r["content"]

    def test_get_bookmark_found(self) -> None:
        """get_bookmark findet einen gespeicherten Bookmark."""
        self.svc.save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=5, content="Find me"
        )
        result = self.svc.get_bookmark(user_id=1, chat_id=10, message_id=5)
        assert result is not None
        assert result["content"] == "Find me"

    def test_get_bookmark_not_found(self) -> None:
        """get_bookmark gibt None für nicht-existierende Bookmarks."""
        result = self.svc.get_bookmark(user_id=1, chat_id=10, message_id=999)
        assert result is None

    def test_remove_bookmark(self) -> None:
        """remove_bookmark löscht und gibt True zurück."""
        self.svc.save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=7, content="Del"
        )
        assert self.svc.remove_bookmark(user_id=1, chat_id=10, message_id=7) is True
        assert self.svc.get_bookmark(user_id=1, chat_id=10, message_id=7) is None
