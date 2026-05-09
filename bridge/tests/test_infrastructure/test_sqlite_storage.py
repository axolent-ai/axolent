"""Tests für SQLite-Storage: Bookmark + Memory + Migration + FTS5.

Testet:
  - SqliteBookmarkStorage: CRUD, User-Isolation, Concurrency
  - SqliteMemoryStorage: CRUD, Layer-Validierung, Search (LIKE + FTS5)
  - JSONL -> SQLite Migration (Idempotenz, Rollback, Corrupt-Handling)
  - FTS5 Volltext-Suche
  - Performance-Smoke-Test
  - Crash-Recovery (DB-Integrität)
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from infrastructure.sqlite_storage import (
    SqliteBookmarkStorage,
    SqliteConnection,
    SqliteMemoryStorage,
    migrate_jsonl_to_sqlite,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporärer DB-Pfad für Test-Isolation."""
    return tmp_path / "test.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    """Frische SQLite-Connection für jeden Test."""
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def bm_storage(conn: SqliteConnection) -> SqliteBookmarkStorage:
    """Bookmark-Storage-Instanz."""
    return SqliteBookmarkStorage(conn)


@pytest.fixture
def mem_storage(conn: SqliteConnection) -> SqliteMemoryStorage:
    """Memory-Storage-Instanz."""
    return SqliteMemoryStorage(conn)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Temporäres data-Verzeichnis für Migration-Tests."""
    d = tmp_path / "data"
    d.mkdir()
    return d


# ──────────────────────────────────────────────────────────────
# SqliteBookmarkStorage Tests
# ──────────────────────────────────────────────────────────────


class TestSqliteBookmarkStorage:
    """Bookmark CRUD-Operationen mit SQLite."""

    def test_save_and_get_bookmark(self, bm_storage: SqliteBookmarkStorage) -> None:
        """Gespeicherter Bookmark kann per message_id abgerufen werden."""
        bm_storage.save_bookmark(
            user_id=1, username="test", message_id=100, chat_id=200, content="Hello"
        )
        result = bm_storage.get_bookmark_by_message_id(
            user_id=1, chat_id=200, message_id=100
        )
        assert result is not None
        assert result["content"] == "Hello"
        assert result["user_id"] == 1

    def test_list_recent_bookmarks(self, bm_storage: SqliteBookmarkStorage) -> None:
        """list_recent gibt die neuesten Bookmarks zurück (neueste zuerst)."""
        for i in range(5):
            bm_storage.save_bookmark(
                user_id=1, username="t", message_id=i, chat_id=10, content=f"Msg {i}"
            )

        recent = bm_storage.list_recent_bookmarks(user_id=1, limit=3)
        assert len(recent) == 3
        # Neueste zuerst (message_id=4 wurde zuletzt gespeichert)
        assert recent[0]["message_id"] == 4

    def test_search_bookmarks(self, bm_storage: SqliteBookmarkStorage) -> None:
        """Suche filtert nach Inhalts-Substring (case-insensitive)."""
        bm_storage.save_bookmark(
            user_id=1, username="t", message_id=1, chat_id=10, content="Python Tipps"
        )
        bm_storage.save_bookmark(
            user_id=1, username="t", message_id=2, chat_id=10, content="Rust Guide"
        )
        bm_storage.save_bookmark(
            user_id=1, username="t", message_id=3, chat_id=10, content="python tricks"
        )

        results = bm_storage.search_bookmarks(user_id=1, query="python")
        assert len(results) == 2

    def test_delete_bookmark(self, bm_storage: SqliteBookmarkStorage) -> None:
        """Gelöschter Bookmark ist nicht mehr auffindbar."""
        bm_storage.save_bookmark(
            user_id=1, username="t", message_id=50, chat_id=10, content="Delete me"
        )
        assert bm_storage.bookmark_exists(user_id=1, chat_id=10, message_id=50)

        deleted = bm_storage.delete_bookmark(user_id=1, chat_id=10, message_id=50)
        assert deleted is True
        assert not bm_storage.bookmark_exists(user_id=1, chat_id=10, message_id=50)

    def test_delete_nonexistent_returns_false(
        self, bm_storage: SqliteBookmarkStorage
    ) -> None:
        """Löschen eines nicht-existierenden Bookmarks gibt False zurück."""
        result = bm_storage.delete_bookmark(user_id=1, chat_id=10, message_id=9999)
        assert result is False

    def test_bookmark_exists_true_false(
        self, bm_storage: SqliteBookmarkStorage
    ) -> None:
        """bookmark_exists gibt True/False korrekt zurück."""
        assert not bm_storage.bookmark_exists(user_id=1, chat_id=10, message_id=77)
        bm_storage.save_bookmark(
            user_id=1, username="t", message_id=77, chat_id=10, content="X"
        )
        assert bm_storage.bookmark_exists(user_id=1, chat_id=10, message_id=77)

    def test_chat_id_scope(self, bm_storage: SqliteBookmarkStorage) -> None:
        """Zwei Chats mit gleicher message_id erzeugen keinen Konflikt."""
        bm_storage.save_bookmark(
            user_id=1, username="t", message_id=1, chat_id=100, content="Chat A"
        )
        bm_storage.save_bookmark(
            user_id=1, username="t", message_id=1, chat_id=200, content="Chat B"
        )

        bm_a = bm_storage.get_bookmark_by_message_id(
            user_id=1, chat_id=100, message_id=1
        )
        bm_b = bm_storage.get_bookmark_by_message_id(
            user_id=1, chat_id=200, message_id=1
        )

        assert bm_a is not None
        assert bm_b is not None
        assert bm_a["content"] == "Chat A"
        assert bm_b["content"] == "Chat B"

    def test_user_scope_isolation(self, bm_storage: SqliteBookmarkStorage) -> None:
        """Bookmarks eines Users sind für andere User nicht sichtbar."""
        bm_storage.save_bookmark(
            user_id=1, username="a", message_id=1, chat_id=10, content="User 1"
        )
        bm_storage.save_bookmark(
            user_id=2, username="b", message_id=2, chat_id=10, content="User 2"
        )

        user1_bm = bm_storage.list_recent_bookmarks(user_id=1)
        user2_bm = bm_storage.list_recent_bookmarks(user_id=2)

        assert len(user1_bm) == 1
        assert user1_bm[0]["content"] == "User 1"
        assert len(user2_bm) == 1
        assert user2_bm[0]["content"] == "User 2"

    def test_concurrent_writes(self, bm_storage: SqliteBookmarkStorage) -> None:
        """Mehrere Threads können gleichzeitig schreiben ohne Datenverlust."""
        num_threads = 10
        errors: list[str] = []

        def _write(thread_id: int) -> None:
            try:
                bm_storage.save_bookmark(
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
        all_bm = bm_storage.list_recent_bookmarks(user_id=1, limit=100)
        assert len(all_bm) == num_threads

    def test_save_bookmark_returns_entry(
        self, bm_storage: SqliteBookmarkStorage
    ) -> None:
        """save_bookmark gibt den gespeicherten Eintrag als Dict zurück."""
        entry = bm_storage.save_bookmark(
            user_id=1, username="test", message_id=42, chat_id=10, content="Test"
        )
        assert entry["user_id"] == 1
        assert entry["message_id"] == 42
        assert entry["content"] == "Test"
        assert "timestamp" in entry

    def test_unicode_content(self, bm_storage: SqliteBookmarkStorage) -> None:
        """Unicode-Zeichen (Umlaute, Emojis) werden korrekt gespeichert."""
        bm_storage.save_bookmark(
            user_id=1,
            username="t",
            message_id=1,
            chat_id=10,
            content="Grüße aus Österreich! 🇦🇹",
        )
        result = bm_storage.get_bookmark_by_message_id(
            user_id=1, chat_id=10, message_id=1
        )
        assert result is not None
        assert "Grüße" in result["content"]
        assert "Österreich" in result["content"]


# ──────────────────────────────────────────────────────────────
# SqliteMemoryStorage Tests
# ──────────────────────────────────────────────────────────────


class TestSqliteMemoryStorageAppend:
    """Tests für append-Operationen."""

    def test_append_episodic(self, mem_storage: SqliteMemoryStorage) -> None:
        """append auf episodic Layer funktioniert."""
        entry = {
            "id": "ep_test1",
            "user_id": 1,
            "content": "Test",
            "timestamp": "2026-01-01T00:00:00",
        }
        mem_storage.append(entry, "episodic")
        result = mem_storage.list_entries(user_id=1, layer="episodic")
        assert len(result) == 1
        assert result[0]["id"] == "ep_test1"

    def test_append_semantic(self, mem_storage: SqliteMemoryStorage) -> None:
        """append auf semantic Layer funktioniert."""
        entry = {
            "id": "sem_test1",
            "user_id": 1,
            "content": "Fakt",
            "category": "fakt",
            "timestamp": "2026-01-01T00:00:00",
        }
        mem_storage.append(entry, "semantic")
        result = mem_storage.list_entries(user_id=1, layer="semantic")
        assert len(result) == 1
        assert result[0]["category"] == "fakt"

    def test_append_procedural(self, mem_storage: SqliteMemoryStorage) -> None:
        """append auf procedural Layer funktioniert."""
        entry = {
            "id": "pro_test1",
            "user_id": 1,
            "content": "Skill",
            "skill_name": "x",
            "timestamp": "2026-01-01T00:00:00",
        }
        mem_storage.append(entry, "procedural")
        result = mem_storage.list_entries(user_id=1, layer="procedural")
        assert len(result) == 1
        assert result[0]["skill_name"] == "x"

    def test_append_invalid_layer_raises(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """Unbekannter Layer wirft ValueError."""
        with pytest.raises(ValueError, match="Unbekannter Layer"):
            mem_storage.append(
                {"id": "x", "content": "", "timestamp": ""}, "invalid_layer"
            )

    def test_multiple_appends(self, mem_storage: SqliteMemoryStorage) -> None:
        """Mehrere appends erzeugen mehrere Einträge."""
        for i in range(5):
            mem_storage.append(
                {
                    "id": f"ep_test{i}",
                    "user_id": 1,
                    "content": f"Entry {i}",
                    "timestamp": f"2026-01-0{i + 1}T00:00:00",
                },
                "episodic",
            )
        result = mem_storage.list_entries(user_id=1, layer="episodic")
        assert len(result) == 5

    def test_append_unicode(self, mem_storage: SqliteMemoryStorage) -> None:
        """Unicode-Zeichen werden korrekt gespeichert."""
        entry = {
            "id": "ep_unicode",
            "user_id": 1,
            "content": "Grüße aus Österreich",
            "timestamp": "2026-01-01T00:00:00",
        }
        mem_storage.append(entry, "episodic")
        result = mem_storage.get_by_id("ep_unicode", "episodic", user_id=1)
        assert result is not None
        assert "Grüße" in result["content"]

    def test_metadata_preserved(self, mem_storage: SqliteMemoryStorage) -> None:
        """Typ-spezifische Felder (context, category etc.) werden in metadata_json
        korrekt gespeichert und beim Lesen zurückgemerged."""
        entry = {
            "id": "ep_meta",
            "user_id": 1,
            "content": "Event mit Kontext",
            "context": {"workspace": "test", "tags": ["important"]},
            "importance": 8,
            "timestamp": "2026-01-01T00:00:00",
        }
        mem_storage.append(entry, "episodic")
        result = mem_storage.get_by_id("ep_meta", "episodic", user_id=1)
        assert result is not None
        assert result["context"] == {"workspace": "test", "tags": ["important"]}
        assert result["importance"] == 8


class TestSqliteMemoryStorageList:
    """Tests für list_entries."""

    def test_list_empty(self, mem_storage: SqliteMemoryStorage) -> None:
        """Leerer Storage gibt leere Liste zurück."""
        result = mem_storage.list_entries(user_id=1, layer="episodic")
        assert result == []

    def test_list_filters_by_user(self, mem_storage: SqliteMemoryStorage) -> None:
        """list_entries filtert nach user_id."""
        mem_storage.append(
            {
                "id": "ep_a",
                "user_id": 1,
                "content": "User 1",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_b",
                "user_id": 2,
                "content": "User 2",
                "timestamp": "2026-01-02T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_c",
                "user_id": 1,
                "content": "User 1 again",
                "timestamp": "2026-01-03T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.list_entries(user_id=1, layer="episodic")
        assert len(result) == 2
        assert all(e["user_id"] == 1 for e in result)

    def test_list_returns_newest_first_by_timestamp(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """list_entries sortiert nach Timestamp absteigend (neueste zuerst)."""
        mem_storage.append(
            {
                "id": "ep_old",
                "user_id": 1,
                "content": "Alt",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_new",
                "user_id": 1,
                "content": "Neu",
                "timestamp": "2026-05-07T12:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_mid",
                "user_id": 1,
                "content": "Mitte",
                "timestamp": "2026-03-15T06:00:00",
            },
            "episodic",
        )

        result = mem_storage.list_entries(user_id=1, layer="episodic")
        assert result[0]["id"] == "ep_new"
        assert result[1]["id"] == "ep_mid"
        assert result[2]["id"] == "ep_old"

    def test_list_respects_limit(self, mem_storage: SqliteMemoryStorage) -> None:
        """list_entries respektiert das Limit."""
        for i in range(20):
            mem_storage.append(
                {
                    "id": f"ep_{i}",
                    "user_id": 1,
                    "content": f"E{i}",
                    "timestamp": f"2026-01-{i + 1:02d}T00:00:00",
                },
                "episodic",
            )

        result = mem_storage.list_entries(user_id=1, layer="episodic", limit=5)
        assert len(result) == 5


class TestSqliteMemoryStorageSearch:
    """Tests für search (LIKE + FTS5)."""

    def test_search_empty(self, mem_storage: SqliteMemoryStorage) -> None:
        """Suche in leerem Storage gibt leere Liste."""
        result = mem_storage.search(user_id=1, query="test", layer="episodic")
        assert result == []

    def test_search_finds_match(self, mem_storage: SqliteMemoryStorage) -> None:
        """Suche findet Matches."""
        mem_storage.append(
            {
                "id": "ep_1",
                "user_id": 1,
                "content": "Jarvis-LITE Architektur",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_2",
                "user_id": 1,
                "content": "Etwas anderes",
                "timestamp": "2026-01-02T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.search(user_id=1, query="Jarvis", layer="episodic")
        assert len(result) == 1
        assert result[0]["id"] == "ep_1"

    def test_search_case_insensitive(self, mem_storage: SqliteMemoryStorage) -> None:
        """Suche ist case-insensitive (via LIKE-Fallback)."""
        mem_storage.append(
            {
                "id": "ep_1",
                "user_id": 1,
                "content": "WICHTIGER Fakt",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )

        # FTS5 MATCH ist case-insensitive per Default-Tokenizer
        result = mem_storage.search(user_id=1, query="wichtiger", layer="episodic")
        assert len(result) == 1

    def test_search_filters_by_user(self, mem_storage: SqliteMemoryStorage) -> None:
        """Suche filtert nach user_id."""
        mem_storage.append(
            {
                "id": "ep_1",
                "user_id": 1,
                "content": "Shared keyword",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_2",
                "user_id": 2,
                "content": "Shared keyword",
                "timestamp": "2026-01-02T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.search(user_id=1, query="keyword", layer="episodic")
        assert len(result) == 1
        assert result[0]["user_id"] == 1

    def test_search_respects_limit(self, mem_storage: SqliteMemoryStorage) -> None:
        """Suche respektiert das Limit."""
        for i in range(10):
            mem_storage.append(
                {
                    "id": f"ep_{i}",
                    "user_id": 1,
                    "content": f"Match {i}",
                    "timestamp": f"2026-01-{i + 1:02d}T00:00:00",
                },
                "episodic",
            )

        result = mem_storage.search(user_id=1, query="Match", layer="episodic", limit=3)
        assert len(result) == 3

    def test_search_returns_newest_first(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """Suche sortiert Treffer nach Timestamp absteigend."""
        mem_storage.append(
            {
                "id": "ep_old",
                "user_id": 1,
                "content": "Keyword old",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_new",
                "user_id": 1,
                "content": "Keyword new",
                "timestamp": "2026-05-07T12:00:00",
            },
            "episodic",
        )

        result = mem_storage.search(user_id=1, query="Keyword", layer="episodic")
        assert result[0]["id"] == "ep_new"
        assert result[1]["id"] == "ep_old"

    def test_search_embedding_mode_raises(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """mode='embedding' raised NotImplementedError (Phase 1+)."""
        with pytest.raises(NotImplementedError, match="Vector-Embedding"):
            mem_storage.search(
                user_id=1, query="test", layer="episodic", mode="embedding"
            )

    def test_search_default_mode_is_substring(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """Default-mode ist 'substring' (bestehendes Verhalten)."""
        mem_storage.append(
            {
                "id": "ep_1",
                "user_id": 1,
                "content": "Suchbegriff hier",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        result = mem_storage.search(
            user_id=1,
            query="Suchbegriff",
            layer="episodic",
            mode="substring",
        )
        assert len(result) == 1


class TestSqliteMemoryStorageDelete:
    """Tests für delete_by_id."""

    def test_delete_existing(self, mem_storage: SqliteMemoryStorage) -> None:
        """delete_by_id entfernt den Entry und gibt True zurück."""
        mem_storage.append(
            {
                "id": "ep_keep",
                "user_id": 1,
                "content": "Keep",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_delete",
                "user_id": 1,
                "content": "Delete me",
                "timestamp": "2026-01-02T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.delete_by_id("ep_delete", "episodic", user_id=1)
        assert result is True

        entries = mem_storage.list_entries(user_id=1, layer="episodic")
        assert len(entries) == 1
        assert entries[0]["id"] == "ep_keep"

    def test_delete_nonexistent(self, mem_storage: SqliteMemoryStorage) -> None:
        """delete_by_id gibt False zurück wenn Entry nicht existiert."""
        result = mem_storage.delete_by_id("ep_ghost", "episodic", user_id=1)
        assert result is False

    def test_delete_wrong_user(self, mem_storage: SqliteMemoryStorage) -> None:
        """delete_by_id verweigert Löschung wenn user_id nicht passt."""
        mem_storage.append(
            {
                "id": "ep_owned",
                "user_id": 1,
                "content": "Owned by 1",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.delete_by_id("ep_owned", "episodic", user_id=999)
        assert result is False

        entries = mem_storage.list_entries(user_id=1, layer="episodic")
        assert len(entries) == 1


class TestSqliteMemoryStorageGetById:
    """Tests für get_by_id."""

    def test_get_existing(self, mem_storage: SqliteMemoryStorage) -> None:
        """get_by_id findet existierenden Entry."""
        mem_storage.append(
            {
                "id": "ep_find",
                "user_id": 1,
                "content": "Findbar",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.get_by_id("ep_find", "episodic", user_id=1)
        assert result is not None
        assert result["content"] == "Findbar"

    def test_get_nonexistent(self, mem_storage: SqliteMemoryStorage) -> None:
        """get_by_id gibt None zurück wenn nicht gefunden."""
        result = mem_storage.get_by_id("ep_nope", "episodic", user_id=1)
        assert result is None


class TestSqliteMemoryStorageConcurrency:
    """Tests für concurrent writes."""

    def test_concurrent_appends(self, mem_storage: SqliteMemoryStorage) -> None:
        """Parallele Writes via Threads funktionieren korrekt."""
        num_threads = 10
        entries_per_thread = 20
        errors: list[str] = []

        def writer(thread_id: int) -> None:
            for i in range(entries_per_thread):
                try:
                    mem_storage.append(
                        {
                            "id": f"ep_t{thread_id}_{i}",
                            "user_id": 1,
                            "content": f"Thread {thread_id} Entry {i}",
                            "timestamp": f"2026-01-01T{thread_id:02d}:{i:02d}:00",
                        },
                        "episodic",
                    )
                except Exception as e:
                    errors.append(str(e))

        threads = [
            threading.Thread(target=writer, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent-Write-Fehler: {errors}"
        all_entries = mem_storage.list_entries(user_id=1, layer="episodic", limit=1000)
        assert len(all_entries) == num_threads * entries_per_thread


# ──────────────────────────────────────────────────────────────
# FTS5 Tests
# ──────────────────────────────────────────────────────────────


class TestFTS5Search:
    """Tests für FTS5-Volltext-Suche."""

    def test_fts_finds_single_word(self, mem_storage: SqliteMemoryStorage) -> None:
        """FTS5 findet einzelne Wörter."""
        mem_storage.append(
            {
                "id": "ep_1",
                "user_id": 1,
                "content": "Python Programmierung lernen",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_2",
                "user_id": 1,
                "content": "Rust für Anfänger",
                "timestamp": "2026-01-02T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.search(user_id=1, query="Python", layer="episodic")
        assert len(result) == 1
        assert result[0]["id"] == "ep_1"

    def test_fts_multilingual(self, mem_storage: SqliteMemoryStorage) -> None:
        """FTS5 findet auch deutsche Wörter mit Umlauten."""
        mem_storage.append(
            {
                "id": "ep_de",
                "user_id": 1,
                "content": "Über die Brücke gehen",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )

        # FTS5 mit Default-Tokenizer sollte Unicode-Wörter finden
        result = mem_storage.search(user_id=1, query="Brücke", layer="episodic")
        assert len(result) == 1

    def test_fts_does_not_leak_across_users(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """FTS5-Suche respektiert User-Isolation."""
        mem_storage.append(
            {
                "id": "ep_u1",
                "user_id": 1,
                "content": "Geheimes Passwort",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_u2",
                "user_id": 2,
                "content": "Geheimes Passwort",
                "timestamp": "2026-01-02T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.search(user_id=1, query="Passwort", layer="episodic")
        assert len(result) == 1
        assert result[0]["user_id"] == 1


# ──────────────────────────────────────────────────────────────
# Migration Tests
# ──────────────────────────────────────────────────────────────


class TestMigration:
    """Tests für JSONL -> SQLite Migration."""

    def test_migrate_bookmarks(self, data_dir: Path) -> None:
        """Bookmark-JSONL wird korrekt migriert und als .bak umbenannt."""
        bm_path = data_dir / "bookmarks.jsonl"
        entries = [
            {
                "user_id": 1,
                "username": "a",
                "chat_id": 10,
                "message_id": 1,
                "content": "Test 1",
                "timestamp": "2026-01-01T00:00:00",
            },
            {
                "user_id": 2,
                "username": "b",
                "chat_id": 20,
                "message_id": 2,
                "content": "Test 2",
                "timestamp": "2026-01-02T00:00:00",
            },
        ]
        with open(bm_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        conn = SqliteConnection(data_dir / "test.db")
        try:
            stats = migrate_jsonl_to_sqlite(conn, data_dir)

            assert stats["bookmarks"] == 2
            assert not bm_path.exists()
            assert (data_dir / "bookmarks.jsonl.bak").exists()

            # Daten sind in SQLite
            bm_storage = SqliteBookmarkStorage(conn)
            assert len(bm_storage.list_recent_bookmarks(user_id=1)) == 1
            assert len(bm_storage.list_recent_bookmarks(user_id=2)) == 1
        finally:
            conn.close()

    def test_migrate_memory(self, data_dir: Path) -> None:
        """Memory-JSONL wird korrekt migriert."""
        ep_path = data_dir / "memory_episodic.jsonl"
        entries = [
            {
                "id": "ep_1",
                "user_id": 1,
                "content": "Event 1",
                "context": {"tag": "test"},
                "importance": 7,
                "timestamp": "2026-01-01T00:00:00",
            },
            {
                "id": "ep_2",
                "user_id": 1,
                "content": "Event 2",
                "importance": 5,
                "timestamp": "2026-01-02T00:00:00",
            },
        ]
        with open(ep_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        conn = SqliteConnection(data_dir / "test.db")
        try:
            stats = migrate_jsonl_to_sqlite(conn, data_dir)

            assert stats["memory_episodic"] == 2
            assert not ep_path.exists()
            assert (data_dir / "memory_episodic.jsonl.bak").exists()

            # Daten sind in SQLite mit Metadata
            mem_storage = SqliteMemoryStorage(conn)
            entries_db = mem_storage.list_entries(user_id=1, layer="episodic")
            assert len(entries_db) == 2
            # Metadata (context) muss erhalten sein
            entry_with_ctx = next(e for e in entries_db if e["id"] == "ep_1")
            assert entry_with_ctx["context"] == {"tag": "test"}
        finally:
            conn.close()

    def test_migration_idempotent(self, data_dir: Path) -> None:
        """Zweite Migration ändert nichts wenn DB schon Daten hat."""
        bm_path = data_dir / "bookmarks.jsonl"
        entries = [
            {
                "user_id": 1,
                "username": "a",
                "chat_id": 10,
                "message_id": 1,
                "content": "Test",
                "timestamp": "2026-01-01T00:00:00",
            },
        ]
        with open(bm_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        conn = SqliteConnection(data_dir / "test.db")
        try:
            stats1 = migrate_jsonl_to_sqlite(conn, data_dir)
            assert stats1["bookmarks"] == 1

            # Schreibe neue JSONL (simuliert Rollback-Szenario)
            with open(bm_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(entries[0], ensure_ascii=False) + "\n")

            # Zweite Migration: DB ist nicht leer -> skip
            stats2 = migrate_jsonl_to_sqlite(conn, data_dir)
            assert "bookmarks" not in stats2

            # Nur 1 Eintrag in DB (nicht verdoppelt)
            bm_storage = SqliteBookmarkStorage(conn)
            assert len(bm_storage.list_recent_bookmarks(user_id=1)) == 1
        finally:
            conn.close()

    def test_migration_handles_corrupt_lines(self, data_dir: Path) -> None:
        """Korrupte JSONL-Zeilen werden übersprungen."""
        bm_path = data_dir / "bookmarks.jsonl"
        with open(bm_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "user_id": 1,
                        "username": "a",
                        "chat_id": 10,
                        "message_id": 1,
                        "content": "valid",
                        "timestamp": "2026-01-01T00:00:00",
                    }
                )
                + "\n"
            )
            f.write("THIS IS NOT JSON {{{\n")
            f.write(
                json.dumps(
                    {
                        "user_id": 1,
                        "username": "b",
                        "chat_id": 10,
                        "message_id": 2,
                        "content": "also valid",
                        "timestamp": "2026-01-02T00:00:00",
                    }
                )
                + "\n"
            )

        conn = SqliteConnection(data_dir / "test.db")
        try:
            stats = migrate_jsonl_to_sqlite(conn, data_dir)
            assert stats["bookmarks"] == 2  # 2 valide Zeilen

            bm_storage = SqliteBookmarkStorage(conn)
            assert len(bm_storage.list_recent_bookmarks(user_id=1)) == 2
        finally:
            conn.close()

    def test_migration_no_jsonl_files(self, data_dir: Path) -> None:
        """Migration ohne JSONL-Files gibt leere Stats zurück."""
        conn = SqliteConnection(data_dir / "test.db")
        try:
            stats = migrate_jsonl_to_sqlite(conn, data_dir)
            assert stats == {}
        finally:
            conn.close()


# ──────────────────────────────────────────────────────────────
# User-Isolation Tests (Security)
# ──────────────────────────────────────────────────────────────


class TestUserIsolation:
    """Tests für Cross-User-Isolation."""

    def test_bookmark_user_isolation(self, bm_storage: SqliteBookmarkStorage) -> None:
        """User A sieht nicht User B's Bookmarks."""
        bm_storage.save_bookmark(
            user_id=100,
            username="alice",
            message_id=1,
            chat_id=10,
            content="Alice secret",
        )
        bm_storage.save_bookmark(
            user_id=200, username="bob", message_id=2, chat_id=10, content="Bob secret"
        )

        alice_bm = bm_storage.list_recent_bookmarks(user_id=100)
        bob_bm = bm_storage.list_recent_bookmarks(user_id=200)

        assert len(alice_bm) == 1
        assert alice_bm[0]["content"] == "Alice secret"
        assert len(bob_bm) == 1
        assert bob_bm[0]["content"] == "Bob secret"

    def test_memory_user_isolation(self, mem_storage: SqliteMemoryStorage) -> None:
        """User A sieht nicht User B's Memory-Entries."""
        mem_storage.append(
            {
                "id": "ep_alice",
                "user_id": 100,
                "content": "Alice memory",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        mem_storage.append(
            {
                "id": "ep_bob",
                "user_id": 200,
                "content": "Bob memory",
                "timestamp": "2026-01-02T00:00:00",
            },
            "episodic",
        )

        alice_mem = mem_storage.list_entries(user_id=100, layer="episodic")
        bob_mem = mem_storage.list_entries(user_id=200, layer="episodic")

        assert len(alice_mem) == 1
        assert alice_mem[0]["content"] == "Alice memory"
        assert len(bob_mem) == 1
        assert bob_mem[0]["content"] == "Bob memory"

    def test_memory_delete_respects_ownership(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """User B kann User A's Entries nicht löschen."""
        mem_storage.append(
            {
                "id": "ep_owned",
                "user_id": 100,
                "content": "Only mine",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.delete_by_id("ep_owned", "episodic", user_id=200)
        assert result is False

        # Entry ist noch da
        entry = mem_storage.get_by_id("ep_owned", "episodic", user_id=100)
        assert entry is not None

    def test_memory_get_by_id_respects_ownership(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """User B kann User A's Entry nicht per get_by_id lesen."""
        mem_storage.append(
            {
                "id": "ep_private",
                "user_id": 100,
                "content": "Private",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )

        result = mem_storage.get_by_id("ep_private", "episodic", user_id=200)
        assert result is None

    def test_bookmark_search_respects_user(
        self, bm_storage: SqliteBookmarkStorage
    ) -> None:
        """Bookmark-Suche zeigt nur eigene Ergebnisse."""
        bm_storage.save_bookmark(
            user_id=100,
            username="alice",
            message_id=1,
            chat_id=10,
            content="Shared keyword",
        )
        bm_storage.save_bookmark(
            user_id=200,
            username="bob",
            message_id=2,
            chat_id=10,
            content="Shared keyword",
        )

        results = bm_storage.search_bookmarks(user_id=100, query="keyword")
        assert len(results) == 1
        assert results[0]["user_id"] == 100


# ──────────────────────────────────────────────────────────────
# DB-Integrität Tests
# ──────────────────────────────────────────────────────────────


class TestDbIntegrity:
    """Tests für DB-Integrität und WAL-Mode."""

    def test_wal_mode_enabled(self, conn: SqliteConnection) -> None:
        """WAL-Mode ist aktiv."""
        row = conn.fetchone("PRAGMA journal_mode")
        assert row is not None
        assert dict(row)["journal_mode"] == "wal"

    def test_schema_idempotent(self, db_path: Path) -> None:
        """Schema-Initialisierung kann mehrfach aufgerufen werden."""
        conn1 = SqliteConnection(db_path)
        conn1.close()

        conn2 = SqliteConnection(db_path)
        # Zweite Initialisierung darf nicht crashen
        row = conn2.fetchone("SELECT COUNT(*) as cnt FROM bookmarks")
        assert row is not None
        conn2.close()

    def test_connection_close_and_reopen(self, db_path: Path) -> None:
        """Daten überleben Connection-Close + Reopen."""
        conn1 = SqliteConnection(db_path)
        bm1 = SqliteBookmarkStorage(conn1)
        bm1.save_bookmark(
            user_id=1, username="t", message_id=1, chat_id=10, content="Persist"
        )
        conn1.close()

        conn2 = SqliteConnection(db_path)
        bm2 = SqliteBookmarkStorage(conn2)
        result = bm2.get_bookmark_by_message_id(user_id=1, chat_id=10, message_id=1)
        assert result is not None
        assert result["content"] == "Persist"
        conn2.close()


# ──────────────────────────────────────────────────────────────
# Performance-Smoke-Test
# ──────────────────────────────────────────────────────────────


class TestPerformanceSmokeTest:
    """Performance-Smoke-Test: Vergleich SQLite vs. JSONL Größenordnung."""

    def test_bookmark_bulk_insert_and_lookup(
        self, bm_storage: SqliteBookmarkStorage
    ) -> None:
        """1000 Bookmarks einfügen, Single-User-Lookup <50ms."""
        # Bulk Insert
        t0 = time.monotonic()
        for i in range(1000):
            bm_storage.save_bookmark(
                user_id=1,
                username="perf",
                message_id=i,
                chat_id=10,
                content=f"Bookmark content number {i}",
            )
        _ = time.monotonic() - t0

        # Single Lookup
        t0 = time.monotonic()
        result = bm_storage.get_bookmark_by_message_id(
            user_id=1, chat_id=10, message_id=500
        )
        lookup_time = time.monotonic() - t0

        assert result is not None
        assert lookup_time < 0.05, f"Lookup dauerte {lookup_time:.3f}s (>50ms)"

        # List recent
        t0 = time.monotonic()
        recent = bm_storage.list_recent_bookmarks(user_id=1, limit=10)
        list_time = time.monotonic() - t0

        assert len(recent) == 10
        assert list_time < 0.05, f"List dauerte {list_time:.3f}s (>50ms)"

    def test_memory_bulk_insert_and_search(
        self, mem_storage: SqliteMemoryStorage
    ) -> None:
        """5000 Memory-Entries einfügen, FTS-Suche <100ms."""
        # Bulk Insert
        t0 = time.monotonic()
        for i in range(5000):
            mem_storage.append(
                {
                    "id": f"ep_perf_{i}",
                    "user_id": 1,
                    "content": f"Memory entry about topic number {i} with details",
                    "importance": i % 10,
                    "timestamp": f"2026-01-{(i % 28) + 1:02d}T{i % 24:02d}:00:00",
                },
                "episodic",
            )
        _ = time.monotonic() - t0

        # FTS Search
        t0 = time.monotonic()
        results = mem_storage.search(
            user_id=1, query="topic", layer="episodic", limit=20
        )
        search_time = time.monotonic() - t0

        assert len(results) > 0
        assert search_time < 0.1, f"FTS-Suche dauerte {search_time:.3f}s (>100ms)"

        # Single Lookup
        t0 = time.monotonic()
        entry = mem_storage.get_by_id("ep_perf_2500", "episodic", user_id=1)
        get_time = time.monotonic() - t0

        assert entry is not None
        assert get_time < 0.05, f"get_by_id dauerte {get_time:.3f}s (>50ms)"

    def test_multi_user_isolation_at_scale(
        self, bm_storage: SqliteBookmarkStorage
    ) -> None:
        """User-Isolation funktioniert auch bei 500 Bookmarks pro User."""
        num_users = 5
        bm_per_user = 100

        for uid in range(num_users):
            for i in range(bm_per_user):
                bm_storage.save_bookmark(
                    user_id=uid,
                    username=f"user_{uid}",
                    message_id=i,
                    chat_id=uid * 100,
                    content=f"User {uid} bookmark {i}",
                )

        # Jeder User sieht nur seine Bookmarks
        for uid in range(num_users):
            t0 = time.monotonic()
            bms = bm_storage.list_recent_bookmarks(user_id=uid, limit=200)
            query_time = time.monotonic() - t0

            assert len(bms) == bm_per_user
            assert all(b["user_id"] == uid for b in bms)
            assert query_time < 0.05, (
                f"User {uid} Query dauerte {query_time:.3f}s (>50ms)"
            )
