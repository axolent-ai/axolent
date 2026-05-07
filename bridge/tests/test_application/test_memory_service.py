"""Tests für MemoryService: CRUD-Flows ueber alle drei Layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from application.memory_service import MemoryService
from infrastructure.memory_storage import MemoryStorage


@pytest.fixture
def memory_service(tmp_data_dir: Path) -> MemoryService:
    """Erstellt einen frischen MemoryService mit temporaerem Storage."""
    storage = MemoryStorage(data_dir=tmp_data_dir)
    return MemoryService(storage=storage)


class TestRememberEpisodic:
    """Tests für remember_episodic."""

    def test_remember_returns_id(self, memory_service: MemoryService) -> None:
        """remember_episodic gibt eine ep_-ID zurueck."""
        entry_id = memory_service.remember_episodic(user_id=1, content="Test Event")
        assert entry_id.startswith("ep_")

    def test_remember_persists(self, memory_service: MemoryService) -> None:
        """Gespeicherter Entry ist via list_recent abrufbar."""
        memory_service.remember_episodic(user_id=1, content="Persistiert")
        entries = memory_service.list_recent(user_id=1, layer="episodic")
        assert len(entries) == 1
        assert entries[0]["content"] == "Persistiert"

    def test_remember_with_importance(self, memory_service: MemoryService) -> None:
        """Importance wird korrekt gespeichert."""
        memory_service.remember_episodic(user_id=1, content="Wichtig", importance=9)
        entries = memory_service.list_recent(user_id=1, layer="episodic")
        assert entries[0]["importance"] == 9


class TestRememberSemantic:
    """Tests für remember_semantic."""

    def test_remember_with_category(self, memory_service: MemoryService) -> None:
        """Semantic Entry mit Kategorie wird korrekt gespeichert."""
        entry_id = memory_service.remember_semantic(
            user_id=1, content="User mag kurze Antworten", category="praeferenz"
        )
        assert entry_id.startswith("sem_")

        entries = memory_service.list_recent(user_id=1, layer="semantic")
        assert len(entries) == 1
        assert entries[0]["category"] == "praeferenz"


class TestRememberProcedural:
    """Tests für remember_procedural."""

    def test_remember_with_skill_name(self, memory_service: MemoryService) -> None:
        """Procedural Entry mit skill_name wird korrekt gespeichert."""
        entry_id = memory_service.remember_procedural(
            user_id=1,
            content="Codeblock verwenden bei Code-Fragen",
            skill_name="code_format",
        )
        assert entry_id.startswith("pro_")

        entries = memory_service.list_recent(user_id=1, layer="procedural")
        assert len(entries) == 1
        assert entries[0]["skill_name"] == "code_format"


class TestRecall:
    """Tests für recall (Suche)."""

    def test_recall_finds_match(self, memory_service: MemoryService) -> None:
        """recall findet Entries die den Suchbegriff enthalten."""
        memory_service.remember_episodic(
            user_id=1, content="Jarvis-LITE Bridge Architektur"
        )
        memory_service.remember_episodic(user_id=1, content="Etwas voellig anderes")

        results = memory_service.recall(user_id=1, query="Jarvis")
        assert len(results) == 1
        assert "Jarvis" in results[0]["content"]

    def test_recall_no_match(self, memory_service: MemoryService) -> None:
        """recall gibt leere Liste bei keinem Treffer."""
        memory_service.remember_episodic(user_id=1, content="ABC")
        results = memory_service.recall(user_id=1, query="XYZ")
        assert results == []


class TestForget:
    """Tests für forget."""

    def test_forget_existing(self, memory_service: MemoryService) -> None:
        """forget loescht Entry und gibt True zurueck."""
        entry_id = memory_service.remember_episodic(user_id=1, content="Vergessen mich")
        assert memory_service.forget(user_id=1, entry_id=entry_id) is True

        # Nicht mehr auffindbar
        entries = memory_service.list_recent(user_id=1, layer="episodic")
        assert len(entries) == 0

    def test_forget_nonexistent(self, memory_service: MemoryService) -> None:
        """forget gibt False zurueck wenn Entry nicht existiert."""
        assert memory_service.forget(user_id=1, entry_id="ep_ghost123456") is False

    def test_forget_wrong_user(self, memory_service: MemoryService) -> None:
        """forget verweigert Loeschung bei falschem User."""
        entry_id = memory_service.remember_episodic(user_id=1, content="Gehoert User 1")
        assert memory_service.forget(user_id=999, entry_id=entry_id) is False

    def test_forget_semantic(self, memory_service: MemoryService) -> None:
        """forget funktioniert auch für semantic Layer (via ID-Prefix)."""
        entry_id = memory_service.remember_semantic(
            user_id=1, content="Fakt", category="fakt"
        )
        assert memory_service.forget(user_id=1, entry_id=entry_id) is True

    def test_forget_procedural(self, memory_service: MemoryService) -> None:
        """forget funktioniert auch für procedural Layer (via ID-Prefix)."""
        entry_id = memory_service.remember_procedural(
            user_id=1, content="Skill", skill_name="test_skill"
        )
        assert memory_service.forget(user_id=1, entry_id=entry_id) is True

    def test_forget_unknown_prefix(self, memory_service: MemoryService) -> None:
        """forget gibt False zurueck bei unbekanntem ID-Prefix."""
        assert memory_service.forget(user_id=1, entry_id="unknown_abc123") is False


class TestGetEntry:
    """Tests für get_entry."""

    def test_get_existing(self, memory_service: MemoryService) -> None:
        """get_entry findet vorhandenen Entry."""
        entry_id = memory_service.remember_episodic(user_id=1, content="Findbar")
        result = memory_service.get_entry(user_id=1, entry_id=entry_id)
        assert result is not None
        assert result["content"] == "Findbar"

    def test_get_nonexistent(self, memory_service: MemoryService) -> None:
        """get_entry gibt None zurueck bei unbekannter ID."""
        result = memory_service.get_entry(user_id=1, entry_id="ep_nope12345678")
        assert result is None


class TestLayerFromId:
    """Tests für _layer_from_id."""

    def test_episodic_prefix(self) -> None:
        assert MemoryService._layer_from_id("ep_abc123") == "episodic"

    def test_semantic_prefix(self) -> None:
        assert MemoryService._layer_from_id("sem_abc123") == "semantic"

    def test_procedural_prefix(self) -> None:
        assert MemoryService._layer_from_id("pro_abc123") == "procedural"

    def test_unknown_prefix(self) -> None:
        assert MemoryService._layer_from_id("xyz_abc123") is None

    def test_empty_string(self) -> None:
        assert MemoryService._layer_from_id("") is None
