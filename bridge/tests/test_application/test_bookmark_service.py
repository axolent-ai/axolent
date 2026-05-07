"""Tests für application.bookmark_service: Bookmark Use-Case-Orchestration.

Testet Toggle-Logik, User-Scoping und Suche.
Nutzt tmp_path für isolierte Testdaten.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from filelock import FileLock


class TestBookmarkService:
    """Bookmark-Service Use-Case-Tests."""

    @pytest.fixture(autouse=True)
    def _isolate_storage(self, tmp_path: Path) -> None:
        """Patcht Bookmark-Storage auf tmp_path."""
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

        yield  # type: ignore[misc]

        for p in self._patches:
            p.stop()

    def test_save_or_toggle_creates_then_deletes(self) -> None:
        """Erster Aufruf speichert, zweiter Aufruf loescht (Toggle)."""
        from application.bookmark_service import save_or_toggle_bookmark

        # Erster Aufruf: speichern
        was_saved, msg = save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=100, content="Test"
        )
        assert was_saved is True
        assert "gespeichert" in msg.lower()

        # Zweiter Aufruf: loeschen
        was_saved, msg = save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=100, content="Test"
        )
        assert was_saved is False
        assert "entfernt" in msg.lower()

    def test_list_returns_user_specific(self) -> None:
        """list_bookmarks gibt nur Bookmarks des angegebenen Users zurueck."""
        from application.bookmark_service import list_bookmarks, save_or_toggle_bookmark

        save_or_toggle_bookmark(
            user_id=1, username="a", chat_id=10, message_id=1, content="User 1"
        )
        save_or_toggle_bookmark(
            user_id=2, username="b", chat_id=10, message_id=2, content="User 2"
        )

        result = list_bookmarks(user_id=1)
        assert len(result) == 1
        assert result[0]["content"] == "User 1"

    def test_search_filters_by_query(self) -> None:
        """search() filtert nach Content-Substring."""
        from application.bookmark_service import save_or_toggle_bookmark, search

        save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=1, content="Python Code"
        )
        save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=2, content="Rust Guide"
        )
        save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=3, content="Python ML"
        )

        results = search(user_id=1, query="Python")
        assert len(results) == 2
        for r in results:
            assert "Python" in r["content"]

    def test_get_bookmark_found(self) -> None:
        """get_bookmark findet einen gespeicherten Bookmark."""
        from application.bookmark_service import get_bookmark, save_or_toggle_bookmark

        save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=5, content="Find me"
        )
        result = get_bookmark(user_id=1, chat_id=10, message_id=5)
        assert result is not None
        assert result["content"] == "Find me"

    def test_get_bookmark_not_found(self) -> None:
        """get_bookmark gibt None für nicht-existierende Bookmarks."""
        from application.bookmark_service import get_bookmark

        result = get_bookmark(user_id=1, chat_id=10, message_id=999)
        assert result is None

    def test_remove_bookmark(self) -> None:
        """remove_bookmark loescht und gibt True zurueck."""
        from application.bookmark_service import (
            get_bookmark,
            remove_bookmark,
            save_or_toggle_bookmark,
        )

        save_or_toggle_bookmark(
            user_id=1, username="t", chat_id=10, message_id=7, content="Del"
        )
        assert remove_bookmark(user_id=1, chat_id=10, message_id=7) is True
        assert get_bookmark(user_id=1, chat_id=10, message_id=7) is None
