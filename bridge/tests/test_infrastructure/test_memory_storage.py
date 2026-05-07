"""Tests fuer MemoryStorage: JSONL-Persistierung mit FileLock.

Testet append, list, search, delete und concurrent writes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from infrastructure.memory_storage import MemoryStorage


@pytest.fixture
def storage(tmp_data_dir: Path) -> MemoryStorage:
    """Erstellt eine frische MemoryStorage-Instanz mit temporaerem Verzeichnis."""
    return MemoryStorage(data_dir=tmp_data_dir)


class TestMemoryStorageAppend:
    """Tests fuer append-Operationen."""

    def test_append_episodic_creates_file(self, storage: MemoryStorage) -> None:
        """Erster append erstellt die JSONL-Datei."""
        entry = {"id": "ep_test1", "user_id": 1, "content": "Test"}
        storage.append(entry, "episodic")
        assert storage.episodic_path.exists()

    def test_append_semantic(self, storage: MemoryStorage) -> None:
        """append auf semantic Layer funktioniert."""
        entry = {"id": "sem_test1", "user_id": 1, "content": "Fakt", "category": "fakt"}
        storage.append(entry, "semantic")
        assert storage.semantic_path.exists()

    def test_append_procedural(self, storage: MemoryStorage) -> None:
        """append auf procedural Layer funktioniert."""
        entry = {"id": "pro_test1", "user_id": 1, "content": "Skill", "skill_name": "x"}
        storage.append(entry, "procedural")
        assert storage.procedural_path.exists()

    def test_append_invalid_layer_raises(self, storage: MemoryStorage) -> None:
        """Unbekannter Layer wirft ValueError."""
        with pytest.raises(ValueError, match="Unbekannter Layer"):
            storage.append({"id": "x"}, "invalid_layer")

    def test_multiple_appends(self, storage: MemoryStorage) -> None:
        """Mehrere appends erzeugen mehrere Zeilen."""
        for i in range(5):
            storage.append(
                {"id": f"ep_test{i}", "user_id": 1, "content": f"Entry {i}"}, "episodic"
            )

        lines = storage.episodic_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_append_unicode(self, storage: MemoryStorage) -> None:
        """Unicode-Zeichen (Umlaute, Emojis) werden korrekt gespeichert."""
        entry = {"id": "ep_unicode", "user_id": 1, "content": "Äöü ß Grüße"}
        storage.append(entry, "episodic")

        raw = storage.episodic_path.read_text(encoding="utf-8")
        assert "Äöü ß Grüße" in raw


class TestMemoryStorageList:
    """Tests fuer list_entries."""

    def test_list_empty(self, storage: MemoryStorage) -> None:
        """Leerer Storage gibt leere Liste zurueck."""
        result = storage.list_entries(user_id=1, layer="episodic")
        assert result == []

    def test_list_filters_by_user(self, storage: MemoryStorage) -> None:
        """list_entries filtert nach user_id."""
        storage.append({"id": "ep_a", "user_id": 1, "content": "User 1"}, "episodic")
        storage.append({"id": "ep_b", "user_id": 2, "content": "User 2"}, "episodic")
        storage.append(
            {"id": "ep_c", "user_id": 1, "content": "User 1 again"}, "episodic"
        )

        result = storage.list_entries(user_id=1, layer="episodic")
        assert len(result) == 2
        assert all(e["user_id"] == 1 for e in result)

    def test_list_returns_newest_first(self, storage: MemoryStorage) -> None:
        """list_entries liefert neueste Eintraege zuerst."""
        for i in range(5):
            storage.append(
                {"id": f"ep_{i}", "user_id": 1, "content": f"Entry {i}"}, "episodic"
            )

        result = storage.list_entries(user_id=1, layer="episodic")
        # Neueste zuerst = letzter geschriebener ist erster im Result
        assert result[0]["id"] == "ep_4"
        assert result[-1]["id"] == "ep_0"

    def test_list_respects_limit(self, storage: MemoryStorage) -> None:
        """list_entries respektiert das Limit."""
        for i in range(20):
            storage.append(
                {"id": f"ep_{i}", "user_id": 1, "content": f"E{i}"}, "episodic"
            )

        result = storage.list_entries(user_id=1, layer="episodic", limit=5)
        assert len(result) == 5


class TestMemoryStorageSearch:
    """Tests fuer search."""

    def test_search_empty(self, storage: MemoryStorage) -> None:
        """Suche in leerem Storage gibt leere Liste."""
        result = storage.search(user_id=1, query="test", layer="episodic")
        assert result == []

    def test_search_finds_match(self, storage: MemoryStorage) -> None:
        """Suche findet substring-Matches."""
        storage.append(
            {"id": "ep_1", "user_id": 1, "content": "Jarvis-LITE Architektur"},
            "episodic",
        )
        storage.append(
            {"id": "ep_2", "user_id": 1, "content": "Etwas anderes"}, "episodic"
        )

        result = storage.search(user_id=1, query="Jarvis", layer="episodic")
        assert len(result) == 1
        assert result[0]["id"] == "ep_1"

    def test_search_case_insensitive(self, storage: MemoryStorage) -> None:
        """Suche ist case-insensitive."""
        storage.append(
            {"id": "ep_1", "user_id": 1, "content": "WICHTIGER Fakt"}, "episodic"
        )

        result = storage.search(user_id=1, query="wichtiger", layer="episodic")
        assert len(result) == 1

    def test_search_filters_by_user(self, storage: MemoryStorage) -> None:
        """Suche filtert nach user_id."""
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
        """Suche respektiert das Limit."""
        for i in range(10):
            storage.append(
                {"id": f"ep_{i}", "user_id": 1, "content": f"Match {i}"}, "episodic"
            )

        result = storage.search(user_id=1, query="Match", layer="episodic", limit=3)
        assert len(result) == 3


class TestMemoryStorageDelete:
    """Tests fuer delete_by_id."""

    def test_delete_existing(self, storage: MemoryStorage) -> None:
        """delete_by_id entfernt den Entry und gibt True zurueck."""
        storage.append({"id": "ep_keep", "user_id": 1, "content": "Keep"}, "episodic")
        storage.append(
            {"id": "ep_delete", "user_id": 1, "content": "Delete me"}, "episodic"
        )

        result = storage.delete_by_id("ep_delete", "episodic", user_id=1)
        assert result is True

        # Nur noch ein Entry uebrig
        entries = storage.list_entries(user_id=1, layer="episodic")
        assert len(entries) == 1
        assert entries[0]["id"] == "ep_keep"

    def test_delete_nonexistent(self, storage: MemoryStorage) -> None:
        """delete_by_id gibt False zurueck wenn Entry nicht existiert."""
        result = storage.delete_by_id("ep_ghost", "episodic", user_id=1)
        assert result is False

    def test_delete_wrong_user(self, storage: MemoryStorage) -> None:
        """delete_by_id verweigert Loeschung wenn user_id nicht passt."""
        storage.append(
            {"id": "ep_owned", "user_id": 1, "content": "Owned by 1"}, "episodic"
        )

        result = storage.delete_by_id("ep_owned", "episodic", user_id=999)
        assert result is False

        # Entry ist noch da
        entries = storage.list_entries(user_id=1, layer="episodic")
        assert len(entries) == 1


class TestMemoryStorageGetById:
    """Tests fuer get_by_id."""

    def test_get_existing(self, storage: MemoryStorage) -> None:
        """get_by_id findet existierenden Entry."""
        storage.append(
            {"id": "ep_find", "user_id": 1, "content": "Findbar"}, "episodic"
        )

        result = storage.get_by_id("ep_find", "episodic", user_id=1)
        assert result is not None
        assert result["content"] == "Findbar"

    def test_get_nonexistent(self, storage: MemoryStorage) -> None:
        """get_by_id gibt None zurueck wenn nicht gefunden."""
        result = storage.get_by_id("ep_nope", "episodic", user_id=1)
        assert result is None


class TestMemoryStorageConcurrency:
    """Tests fuer concurrent writes via FileLock."""

    def test_concurrent_appends(self, storage: MemoryStorage) -> None:
        """Parallele Writes via Threads fuehren zu korrekter JSONL-Datei."""
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

        # Alle Zeilen muessen valides JSON sein
        lines = storage.episodic_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == num_threads * entries_per_thread

        for line in lines:
            entry = json.loads(line)  # Darf nicht crashen
            assert "id" in entry
