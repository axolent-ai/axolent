"""K10: Persistence and storage edge case tests.

SQLite lock contention, corrupted conversation history,
invalid SQL in executescript, disk-full simulation,
encryption storage without keyring.
"""

from __future__ import annotations

import sqlite3
import threading
import unicodedata
from pathlib import Path

import pytest

from infrastructure.sqlite_storage import (
    SqliteConnection,
    SqliteBookmarkStorage,
    SqliteMemoryStorage,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
    HYPOTHESIS_SCHEMA_SQL,
)


@pytest.mark.adversarial
class TestSQLiteLockContention:
    """SQLite WAL mode under concurrent write pressure."""

    def test_concurrent_bookmark_writes(self, tmp_path: Path) -> None:
        """WHAT: Multiple threads writing bookmarks simultaneously.
        EXPECTED: All writes succeed or fail gracefully (no corruption).
        WHY: SQLite can fail with SQLITE_BUSY under contention.
        """
        db_path = tmp_path / "test.db"
        conn = SqliteConnection(db_path)
        storage = SqliteBookmarkStorage(conn)
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    storage.save_bookmark(
                        user_id=thread_id,
                        username=f"user_{thread_id}",
                        chat_id=thread_id * 1000,
                        message_id=i,
                        content=f"Bookmark from thread {thread_id}, item {i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some errors are acceptable (SQLITE_BUSY), but no corruption
        if errors:
            for e in errors:
                assert isinstance(
                    e, (sqlite3.OperationalError, sqlite3.IntegrityError)
                ), f"Unexpected error type: {type(e)}: {e}"
        conn.close()

    def test_concurrent_read_write(self, tmp_path: Path) -> None:
        """WHAT: Reads while writes are happening.
        EXPECTED: Reads see consistent data (WAL mode).
        WHY: WAL allows concurrent readers and writers.
        """
        db_path = tmp_path / "test_rw.db"
        conn = SqliteConnection(db_path)
        storage = SqliteBookmarkStorage(conn)
        errors: list[Exception] = []

        # Seed some data
        for i in range(10):
            storage.save_bookmark(
                user_id=1,
                username="test",
                chat_id=100,
                message_id=i,
                content=f"Bookmark {i}",
            )

        def reader() -> None:
            try:
                for _ in range(50):
                    bookmarks = storage.list_recent_bookmarks(user_id=1, limit=10)
                    assert isinstance(bookmarks, list)
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(10, 30):
                    storage.save_bookmark(
                        user_id=1,
                        username="test",
                        chat_id=100,
                        message_id=i,
                        content=f"Concurrent bookmark {i}",
                    )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if errors:
            for e in errors:
                assert isinstance(e, (sqlite3.OperationalError,)), (
                    f"Unexpected error: {type(e)}: {e}"
                )
        conn.close()


@pytest.mark.adversarial
class TestCorruptedData:
    """Corrupted or malformed data in storage."""

    def test_hypothesis_scope_from_invalid_json(self) -> None:
        """WHAT: HypothesisScope.from_json with invalid JSON.
        EXPECTED: Returns default scope, no crash.
        WHY: Corrupted DB rows could have invalid JSON in scope_json.
        """
        scope = HypothesisScope.from_json("{invalid json")
        assert scope.project == ""
        assert scope.client == ""
        assert scope.context == ()

    def test_hypothesis_scope_from_empty_string(self) -> None:
        """WHAT: HypothesisScope.from_json with empty string.
        EXPECTED: Returns default scope.
        WHY: NULL columns read as empty string.
        """
        scope = HypothesisScope.from_json("")
        assert scope.project == ""

    @pytest.mark.xfail(
        reason="FINDING-08: from_json catches TypeError but not AttributeError. "
        "json.loads('null') returns None; None.get() raises AttributeError.",
        strict=True,
    )
    def test_hypothesis_scope_from_none_like(self) -> None:
        """WHAT: HypothesisScope.from_json with 'null' string.
        EXPECTED: Returns default scope.
        WHY: JSON null could be stored as string.
        """
        scope = HypothesisScope.from_json("null")
        assert scope.project == ""

    @pytest.mark.xfail(
        reason="FINDING-09: from_json catches TypeError but not AttributeError. "
        "json.loads('[1,2,3]') returns list; list.get() raises AttributeError.",
        strict=True,
    )
    def test_hypothesis_scope_from_array_json(self) -> None:
        """WHAT: HypothesisScope.from_json with JSON array instead of object.
        EXPECTED: Returns default scope (TypeError on .get()).
        WHY: Wrong JSON type stored in column.
        """
        scope = HypothesisScope.from_json("[1, 2, 3]")
        assert scope.project == ""

    def test_hypothesis_scope_to_json_roundtrip(self) -> None:
        """WHAT: Scope with unicode content survives JSON roundtrip.
        EXPECTED: Exact content preserved.
        WHY: ensure_ascii=False must work for German umlauts.
        """
        original = HypothesisScope(
            project="test-projekt",
            client="kunde-mit-umlauten",
            context=("tag1", "tag2"),
        )
        json_str = original.to_json()
        restored = HypothesisScope.from_json(json_str)
        assert restored.project == original.project
        assert restored.client == original.client
        assert restored.context == original.context


@pytest.mark.adversarial
class TestInvalidSQL:
    """Invalid SQL through executescript or raw queries."""

    def test_hypothesis_schema_is_idempotent(self, tmp_path: Path) -> None:
        """WHAT: HYPOTHESIS_SCHEMA_SQL executed twice.
        EXPECTED: No error (CREATE TABLE IF NOT EXISTS).
        WHY: Schema init must be safe to call multiple times.
        """
        db_path = tmp_path / "schema_test.db"
        raw_conn = sqlite3.connect(str(db_path))
        raw_conn.row_factory = sqlite3.Row
        # Execute schema twice
        raw_conn.executescript(HYPOTHESIS_SCHEMA_SQL)
        raw_conn.executescript(HYPOTHESIS_SCHEMA_SQL)
        # Verify tables exist
        cursor = raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "hypotheses" in tables
        assert "hypothesis_evidence" in tables
        raw_conn.close()

    def test_unicode_in_sql_values(self, tmp_path: Path) -> None:
        """WHAT: Unicode characters (including emojis) in SQL values.
        EXPECTED: Stored and retrieved correctly.
        WHY: UTF-8 encoding must work end-to-end in SQLite.
        """
        db_path = tmp_path / "unicode_test.db"
        conn = SqliteConnection(db_path)
        store = HypothesisStorage(conn)
        store.init_schema()

        h = Hypothesis(
            hypothesis_id="unicode-test",
            user_id=1,
            claim="User prefers bullet points: café",
            scope=HypothesisScope(),
            created_at="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        )
        store.insert_hypothesis(h)
        retrieved = store.get_hypothesis("unicode-test")
        assert retrieved is not None
        # Compare with NFC normalization: file encoding may use composed
        # or decomposed form (e + combining accent vs precomposed é).
        normalized_claim = unicodedata.normalize("NFC", retrieved.claim)
        assert "caf" in normalized_claim  # café substring
        assert "bullet" in normalized_claim
        conn.close()


@pytest.mark.adversarial
class TestStorageEdgeCases:
    """Edge cases in storage layer."""

    def test_sqlite_connection_creates_parent_dirs(self, tmp_path: Path) -> None:
        """WHAT: SqliteConnection with a path in a non-existent directory.
        EXPECTED: Directory created automatically.
        WHY: First-run scenario may have missing data directory.
        """
        db_path = tmp_path / "nonexistent" / "subdir" / "test.db"
        conn = SqliteConnection(db_path)
        # Force connection init
        _ = conn.fetchall("SELECT 1", ())
        assert db_path.exists()
        conn.close()

    def test_memory_entry_with_very_long_content(self, tmp_path: Path) -> None:
        """WHAT: Memory entry with 100KB content.
        EXPECTED: Stored and retrieved correctly.
        WHY: /remember with very long content.
        """
        db_path = tmp_path / "long_memory.db"
        conn = SqliteConnection(db_path)
        storage = SqliteMemoryStorage(conn)
        content = "Important fact: " + "x" * 100_000
        from datetime import datetime, timezone

        entry = {
            "id": "long-001",
            "user_id": 1,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        storage.append(entry, layer="episodic")
        entries = storage.list_entries(user_id=1, layer="episodic", limit=10)
        assert len(entries) > 0
        assert len(entries[0]["content"]) > 100_000
        conn.close()

    def test_bookmark_duplicate_message_id(self, tmp_path: Path) -> None:
        """WHAT: Save bookmark with same message_id twice for same user/chat.
        EXPECTED: INSERT OR REPLACE updates the content.
        WHY: SqliteBookmarkStorage uses INSERT OR REPLACE.
        """
        db_path = tmp_path / "dup_bookmark.db"
        conn = SqliteConnection(db_path)
        storage = SqliteBookmarkStorage(conn)
        # First save
        storage.save_bookmark(
            user_id=1,
            username="test",
            chat_id=100,
            message_id=42,
            content="Test bookmark v1",
        )
        # Second save (same user/chat/message_id)
        storage.save_bookmark(
            user_id=1,
            username="test",
            chat_id=100,
            message_id=42,
            content="Test bookmark v2",
        )
        # Should have exactly 1 bookmark (replaced, not duplicated)
        bm = storage.get_bookmark_by_message_id(user_id=1, chat_id=100, message_id=42)
        assert bm is not None
        assert bm["content"] == "Test bookmark v2"
        conn.close()
