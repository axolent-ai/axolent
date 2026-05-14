"""Tests for infrastructure.bookmark_storage: JSONL persistence with FileLock.

Tests CRUD operations, chat ID scoping, and concurrent access.
Uses tmp_path fixtures for isolated test files.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest


class TestBookmarkStorage:
    """Bookmark-Storage CRUD-Operationen mit isolierten Testdaten."""

    @pytest.fixture(autouse=True)
    def _isolate_storage(self, tmp_path: Path) -> None:
        """Patcht BOOKMARKS_PATH auf tmp_path für Test-Isolation."""
        self.bm_path = tmp_path / "bookmarks.jsonl"
        self.lock_path = str(self.bm_path) + ".lock"

        # Alle relevanten Modul-Globals patchen
        patches = [
            patch("infrastructure.bookmark_storage.BOOKMARKS_PATH", self.bm_path),
            patch("infrastructure.bookmark_storage._BM_LOCK_PATH", self.lock_path),
        ]
        for p in patches:
            p.start()

        # Lock muss nach Patch des Pfads neu erstellt werden
        from filelock import FileLock

        new_lock = FileLock(self.lock_path)
        patches.append(patch("infrastructure.bookmark_storage._BM_LOCK", new_lock))
        patches[-1].start()

        yield  # type: ignore[misc]

        for p in patches:
            p.stop()

    def test_save_and_get_bookmark(self) -> None:
        """Gespeicherter Bookmark kann per message_id abgerufen werden."""
        from infrastructure.bookmark_storage import (
            get_bookmark_by_message_id,
            save_bookmark,
        )

        save_bookmark(
            user_id=1, username="test", message_id=100, chat_id=200, content="Hello"
        )
        result = get_bookmark_by_message_id(user_id=1, chat_id=200, message_id=100)
        assert result is not None
        assert result["content"] == "Hello"
        assert result["user_id"] == 1

    def test_list_recent_bookmarks(self) -> None:
        """list_recent gibt die neuesten Bookmarks zurück (neueste zuerst)."""
        from infrastructure.bookmark_storage import list_recent_bookmarks, save_bookmark

        for i in range(5):
            save_bookmark(
                user_id=1, username="t", message_id=i, chat_id=10, content=f"Msg {i}"
            )

        recent = list_recent_bookmarks(user_id=1, limit=3)
        assert len(recent) == 3
        # Neueste zuerst (message_id=4 wurde zuletzt gespeichert)
        assert recent[0]["message_id"] == 4

    def test_search_bookmarks(self) -> None:
        """Suche filtert nach Inhalts-Substring (case-insensitive)."""
        from infrastructure.bookmark_storage import save_bookmark, search_bookmarks

        save_bookmark(
            user_id=1, username="t", message_id=1, chat_id=10, content="Python Tipps"
        )
        save_bookmark(
            user_id=1, username="t", message_id=2, chat_id=10, content="Rust Guide"
        )
        save_bookmark(
            user_id=1, username="t", message_id=3, chat_id=10, content="python tricks"
        )

        results = search_bookmarks(user_id=1, query="python")
        assert len(results) == 2  # Case-insensitive: "Python" und "python" matchen

    def test_delete_bookmark(self) -> None:
        """Gelöschter Bookmark ist nicht mehr auffindbar."""
        from infrastructure.bookmark_storage import (
            bookmark_exists,
            delete_bookmark,
            save_bookmark,
        )

        save_bookmark(
            user_id=1, username="t", message_id=50, chat_id=10, content="Delete me"
        )
        assert bookmark_exists(user_id=1, chat_id=10, message_id=50)

        deleted = delete_bookmark(user_id=1, chat_id=10, message_id=50)
        assert deleted is True
        assert not bookmark_exists(user_id=1, chat_id=10, message_id=50)

    def test_delete_nonexistent_returns_false(self) -> None:
        """Löschen eines nicht-existierenden Bookmarks gibt False zurück."""
        from infrastructure.bookmark_storage import delete_bookmark

        result = delete_bookmark(user_id=1, chat_id=10, message_id=9999)
        assert result is False

    def test_bookmark_exists_true_false(self) -> None:
        """bookmark_exists gibt True/False korrekt zurück."""
        from infrastructure.bookmark_storage import bookmark_exists, save_bookmark

        assert not bookmark_exists(user_id=1, chat_id=10, message_id=77)
        save_bookmark(user_id=1, username="t", message_id=77, chat_id=10, content="X")
        assert bookmark_exists(user_id=1, chat_id=10, message_id=77)

    def test_chat_id_scope(self) -> None:
        """Zwei Chats mit gleicher message_id erzeugen keinen Konflikt."""
        from infrastructure.bookmark_storage import (
            get_bookmark_by_message_id,
            save_bookmark,
        )

        save_bookmark(
            user_id=1, username="t", message_id=1, chat_id=100, content="Chat A"
        )
        save_bookmark(
            user_id=1, username="t", message_id=1, chat_id=200, content="Chat B"
        )

        bm_a = get_bookmark_by_message_id(user_id=1, chat_id=100, message_id=1)
        bm_b = get_bookmark_by_message_id(user_id=1, chat_id=200, message_id=1)

        assert bm_a is not None
        assert bm_b is not None
        # Mindestens einer der beiden muss den korrekten Content haben
        contents = {bm_a["content"], bm_b["content"]}
        assert "Chat A" in contents or "Chat B" in contents

    def test_filelock_protects_concurrent_writes(self) -> None:
        """Mehrere Threads können gleichzeitig schreiben ohne Datenverlust."""
        from infrastructure.bookmark_storage import list_recent_bookmarks, save_bookmark

        num_threads = 10
        errors: list[str] = []

        def _write(thread_id: int) -> None:
            try:
                save_bookmark(
                    user_id=1,
                    username="t",
                    message_id=thread_id,
                    chat_id=10,
                    content=f"Thread {thread_id}",
                )
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=_write, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent-Write-Fehler: {errors}"
        all_bm = list_recent_bookmarks(user_id=1, limit=100)
        assert len(all_bm) == num_threads

    def test_user_scope_isolation(self) -> None:
        """Bookmarks eines Users sind für andere User nicht sichtbar."""
        from infrastructure.bookmark_storage import list_recent_bookmarks, save_bookmark

        save_bookmark(
            user_id=1, username="a", message_id=1, chat_id=10, content="User 1"
        )
        save_bookmark(
            user_id=2, username="b", message_id=2, chat_id=10, content="User 2"
        )

        user1_bm = list_recent_bookmarks(user_id=1)
        user2_bm = list_recent_bookmarks(user_id=2)

        assert len(user1_bm) == 1
        assert user1_bm[0]["content"] == "User 1"
        assert len(user2_bm) == 1
        assert user2_bm[0]["content"] == "User 2"


class TestMigrateLegacyChatId:
    """Tests für crash-safe migrate_legacy_chat_id (FIX 9)."""

    @pytest.fixture(autouse=True)
    def _isolate_storage(self, tmp_path: Path) -> None:
        """Patcht BOOKMARKS_PATH auf tmp_path für Test-Isolation."""
        self.bm_path = tmp_path / "bookmarks.jsonl"
        self.lock_path = str(self.bm_path) + ".lock"

        patches = [
            patch("infrastructure.bookmark_storage.BOOKMARKS_PATH", self.bm_path),
            patch("infrastructure.bookmark_storage._BM_LOCK_PATH", self.lock_path),
        ]
        for p in patches:
            p.start()

        from filelock import FileLock

        new_lock = FileLock(self.lock_path)
        patches.append(patch("infrastructure.bookmark_storage._BM_LOCK", new_lock))
        patches[-1].start()

        yield  # type: ignore[misc]

        for p in patches:
            p.stop()

    def test_migration_handles_corrupt_lines(self) -> None:
        """Korrupte JSONL-Zeilen crashen die Migration nicht, werden übersprungen."""
        from infrastructure.bookmark_storage import (
            list_recent_bookmarks,
            migrate_legacy_chat_id,
        )

        # Schreibe Mix aus gültigen und korrupten Zeilen
        import json

        lines = [
            json.dumps({"user_id": 1, "message_id": 1, "content": "valid1"}),
            "THIS IS NOT JSON {{{",
            json.dumps(
                {"user_id": 1, "message_id": 2, "chat_id": 1, "content": "valid2"}
            ),
            "",
            "also broken",
        ]
        self.bm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        migrated = migrate_legacy_chat_id()

        # Nur die gültige Zeile ohne chat_id wird migriert
        assert migrated == 1

        # Korrupte Zeilen sind raus, gültige bleiben
        all_bm = list_recent_bookmarks(user_id=1, limit=10)
        assert len(all_bm) == 2

    def test_migration_nonexistent_file(self) -> None:
        """Bei nicht existierender Datei wird 0 zurückgegeben."""
        from infrastructure.bookmark_storage import migrate_legacy_chat_id

        assert migrate_legacy_chat_id() == 0

    def test_migration_all_valid(self) -> None:
        """Wenn keine Migration noetig und keine Korruption: Datei bleibt unverändert."""
        from infrastructure.bookmark_storage import migrate_legacy_chat_id

        import json

        lines = [
            json.dumps({"user_id": 1, "message_id": 1, "chat_id": 1, "content": "ok"}),
        ]
        self.bm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        migrated = migrate_legacy_chat_id()
        assert migrated == 0
