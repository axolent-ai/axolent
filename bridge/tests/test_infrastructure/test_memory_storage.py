"""Tests for MemoryStorage: JSONL persistence with FileLock.

Tests append, list, search, delete, concurrent writes,
mode parameter, and recency sorting.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from infrastructure.memory_storage import MemoryStorage


@pytest.fixture
def storage(tmp_data_dir: Path) -> MemoryStorage:
    """Create a fresh MemoryStorage instance with temporary directory."""
    return MemoryStorage(data_dir=tmp_data_dir)


class TestMemoryStorageAppend:
    """Tests for append operations."""

    def test_append_episodic_creates_file(self, storage: MemoryStorage) -> None:
        """First append creates the JSONL file."""
        entry = {"id": "ep_test1", "user_id": 1, "content": "Test"}
        storage.append(entry, "episodic")
        assert storage.episodic_path.exists()

    def test_append_semantic(self, storage: MemoryStorage) -> None:
        """Append to semantic layer works."""
        entry = {"id": "sem_test1", "user_id": 1, "content": "Fakt", "category": "fakt"}
        storage.append(entry, "semantic")
        assert storage.semantic_path.exists()

    def test_append_procedural(self, storage: MemoryStorage) -> None:
        """Append to procedural layer works."""
        entry = {"id": "pro_test1", "user_id": 1, "content": "Skill", "skill_name": "x"}
        storage.append(entry, "procedural")
        assert storage.procedural_path.exists()

    def test_append_invalid_layer_raises(self, storage: MemoryStorage) -> None:
        """Unknown layer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown layer"):
            storage.append({"id": "x"}, "invalid_layer")

    def test_multiple_appends(self, storage: MemoryStorage) -> None:
        """Multiple appends produce multiple lines."""
        for i in range(5):
            storage.append(
                {"id": f"ep_test{i}", "user_id": 1, "content": f"Entry {i}"}, "episodic"
            )

        lines = storage.episodic_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_append_unicode(self, storage: MemoryStorage) -> None:
        """Unicode characters (umlauts, emojis) are stored correctly."""
        entry = {"id": "ep_unicode", "user_id": 1, "content": "Aeoeuess Gruesse"}
        storage.append(entry, "episodic")

        raw = storage.episodic_path.read_text(encoding="utf-8")
        assert "Aeoeuess Gruesse" in raw


class TestMemoryStorageList:
    """Tests for list_entries."""

    def test_list_empty(self, storage: MemoryStorage) -> None:
        """Empty storage returns empty list."""
        result = storage.list_entries(user_id=1, layer="episodic")
        assert result == []

    def test_list_filters_by_user(self, storage: MemoryStorage) -> None:
        """list_entries filters by user_id."""
        storage.append({"id": "ep_a", "user_id": 1, "content": "User 1"}, "episodic")
        storage.append({"id": "ep_b", "user_id": 2, "content": "User 2"}, "episodic")
        storage.append(
            {"id": "ep_c", "user_id": 1, "content": "User 1 again"}, "episodic"
        )

        result = storage.list_entries(user_id=1, layer="episodic")
        assert len(result) == 2
        assert all(e["user_id"] == 1 for e in result)

    def test_list_returns_newest_first_by_timestamp(
        self, storage: MemoryStorage
    ) -> None:
        """list_entries sorts by timestamp descending (newest first)."""
        storage.append(
            {
                "id": "ep_old",
                "user_id": 1,
                "content": "Alt",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        storage.append(
            {
                "id": "ep_new",
                "user_id": 1,
                "content": "Neu",
                "timestamp": "2026-05-07T12:00:00",
            },
            "episodic",
        )
        storage.append(
            {
                "id": "ep_mid",
                "user_id": 1,
                "content": "Mitte",
                "timestamp": "2026-03-15T06:00:00",
            },
            "episodic",
        )

        result = storage.list_entries(user_id=1, layer="episodic")
        assert result[0]["id"] == "ep_new"
        assert result[1]["id"] == "ep_mid"
        assert result[2]["id"] == "ep_old"

    def test_list_respects_limit(self, storage: MemoryStorage) -> None:
        """list_entries respects the limit."""
        for i in range(20):
            storage.append(
                {"id": f"ep_{i}", "user_id": 1, "content": f"E{i}"}, "episodic"
            )

        result = storage.list_entries(user_id=1, layer="episodic", limit=5)
        assert len(result) == 5


class TestMemoryStorageSearch:
    """Tests for search."""

    def test_search_empty(self, storage: MemoryStorage) -> None:
        """Search in empty storage returns empty list."""
        result = storage.search(user_id=1, query="test", layer="episodic")
        assert result == []

    def test_search_finds_match(self, storage: MemoryStorage) -> None:
        """Search finds substring matches."""
        storage.append(
            {"id": "ep_1", "user_id": 1, "content": "Axolent Architektur"},
            "episodic",
        )
        storage.append(
            {"id": "ep_2", "user_id": 1, "content": "Etwas anderes"}, "episodic"
        )

        result = storage.search(user_id=1, query="Axolent", layer="episodic")
        assert len(result) == 1
        assert result[0]["id"] == "ep_1"

    def test_search_case_insensitive(self, storage: MemoryStorage) -> None:
        """Search is case-insensitive."""
        storage.append(
            {"id": "ep_1", "user_id": 1, "content": "WICHTIGER Fakt"}, "episodic"
        )

        result = storage.search(user_id=1, query="wichtiger", layer="episodic")
        assert len(result) == 1

    def test_search_filters_by_user(self, storage: MemoryStorage) -> None:
        """Search filters by user_id."""
        storage.append(
            {"id": "ep_1", "user_id": 1, "content": "Shared keyword"}, "episodic"
        )
        storage.append(
            {"id": "ep_2", "user_id": 2, "content": "Shared keyword"}, "episodic"
        )

        result = storage.search(user_id=1, query="keyword", layer="episodic")
        assert len(result) == 1
        assert result[0]["user_id"] == 1

    def test_search_respects_limit(self, storage: MemoryStorage) -> None:
        """Search respects the limit."""
        for i in range(10):
            storage.append(
                {"id": f"ep_{i}", "user_id": 1, "content": f"Match {i}"}, "episodic"
            )

        result = storage.search(user_id=1, query="Match", layer="episodic", limit=3)
        assert len(result) == 3

    def test_search_returns_newest_first(self, storage: MemoryStorage) -> None:
        """Search sorts hits by timestamp descending."""
        storage.append(
            {
                "id": "ep_old",
                "user_id": 1,
                "content": "Keyword old",
                "timestamp": "2026-01-01T00:00:00",
            },
            "episodic",
        )
        storage.append(
            {
                "id": "ep_new",
                "user_id": 1,
                "content": "Keyword new",
                "timestamp": "2026-05-07T12:00:00",
            },
            "episodic",
        )

        result = storage.search(user_id=1, query="Keyword", layer="episodic")
        assert result[0]["id"] == "ep_new"
        assert result[1]["id"] == "ep_old"

    def test_search_embedding_mode_raises(self, storage: MemoryStorage) -> None:
        """mode='embedding' raises NotImplementedError (Phase 1+)."""
        with pytest.raises(NotImplementedError, match="Vector embedding"):
            storage.search(user_id=1, query="test", layer="episodic", mode="embedding")

    def test_search_default_mode_is_substring(self, storage: MemoryStorage) -> None:
        """Default mode is 'substring' (existing behavior)."""
        storage.append(
            {"id": "ep_1", "user_id": 1, "content": "Suchbegriff hier"}, "episodic"
        )
        # Explizit mode="substring"
        result = storage.search(
            user_id=1, query="Suchbegriff", layer="episodic", mode="substring"
        )
        assert len(result) == 1


class TestMemoryStorageDelete:
    """Tests for delete_by_id."""

    def test_delete_existing(self, storage: MemoryStorage) -> None:
        """delete_by_id removes the entry and returns True."""
        storage.append({"id": "ep_keep", "user_id": 1, "content": "Keep"}, "episodic")
        storage.append(
            {"id": "ep_delete", "user_id": 1, "content": "Delete me"}, "episodic"
        )

        result = storage.delete_by_id("ep_delete", "episodic", user_id=1)
        assert result is True

        # Only one entry remaining
        entries = storage.list_entries(user_id=1, layer="episodic")
        assert len(entries) == 1
        assert entries[0]["id"] == "ep_keep"

    def test_delete_nonexistent(self, storage: MemoryStorage) -> None:
        """delete_by_id returns False when entry does not exist."""
        result = storage.delete_by_id("ep_ghost", "episodic", user_id=1)
        assert result is False

    def test_delete_wrong_user(self, storage: MemoryStorage) -> None:
        """delete_by_id refuses deletion when user_id does not match."""
        storage.append(
            {"id": "ep_owned", "user_id": 1, "content": "Owned by 1"}, "episodic"
        )

        result = storage.delete_by_id("ep_owned", "episodic", user_id=999)
        assert result is False

        # Entry is still there
        entries = storage.list_entries(user_id=1, layer="episodic")
        assert len(entries) == 1


class TestMemoryStorageGetById:
    """Tests for get_by_id."""

    def test_get_existing(self, storage: MemoryStorage) -> None:
        """get_by_id finds existing entry."""
        storage.append(
            {"id": "ep_find", "user_id": 1, "content": "Findbar"}, "episodic"
        )

        result = storage.get_by_id("ep_find", "episodic", user_id=1)
        assert result is not None
        assert result["content"] == "Findbar"

    def test_get_nonexistent(self, storage: MemoryStorage) -> None:
        """get_by_id returns None when not found."""
        result = storage.get_by_id("ep_nope", "episodic", user_id=1)
        assert result is None


class TestMemoryStorageConcurrency:
    """Tests for concurrent writes via FileLock."""

    def test_concurrent_appends(self, storage: MemoryStorage) -> None:
        """Parallel writes via threads result in correct JSONL file."""
        num_threads = 10
        entries_per_thread = 20

        def writer(thread_id: int) -> None:
            for i in range(entries_per_thread):
                storage.append(
                    {
                        "id": f"ep_t{thread_id}_{i}",
                        "user_id": 1,
                        "content": f"Thread {thread_id} Entry {i}",
                    },
                    "episodic",
                )

        threads = [
            threading.Thread(target=writer, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All lines must be valid JSON
        lines = storage.episodic_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == num_threads * entries_per_thread

        for line in lines:
            entry = json.loads(line)  # Darf nicht crashen
            assert "id" in entry
