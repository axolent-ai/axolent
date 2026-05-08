"""Tests für SemanticEntry: Erstellung, Serialisierung, Deserialisierung."""

from __future__ import annotations

from domain.memory.semantic import SemanticEntry


class TestSemanticEntry:
    """Tests für SemanticEntry Datenklasse."""

    def test_default_creation(self) -> None:
        """Entry mit Default-Werten hat korrekte Struktur."""
        entry = SemanticEntry()
        assert entry.id.startswith("sem_")
        assert len(entry.id) == 16  # sem_ + 12 hex chars
        assert entry.user_id == 0
        assert entry.content == ""
        assert entry.category == "fakt"
        assert entry.importance == 5

    def test_creation_with_category(self) -> None:
        """Entry mit expliziter Kategorie."""
        entry = SemanticEntry(
            user_id=123,
            content="User bevorzugt kurze Antworten",
            category="praeferenz",
            importance=8,
        )
        assert entry.category == "praeferenz"
        assert entry.content == "User bevorzugt kurze Antworten"

    def test_to_dict_includes_category(self) -> None:
        """to_dict enthaelt category-Feld."""
        entry = SemanticEntry(category="person", content="Jessica, Selbststaendig")
        d = entry.to_dict()
        assert d["category"] == "person"
        assert "category" in d

    def test_from_dict(self) -> None:
        """from_dict deserialisiert korrekt inklusive category."""
        data = {
            "id": "sem_abc123def456",
            "user_id": 42,
            "content": "Fakt über User",
            "category": "projekt",
            "context": {},
            "timestamp": "2026-05-07T02:00:00+00:00",
            "importance": 6,
        }
        entry = SemanticEntry.from_dict(data)
        assert entry.category == "projekt"
        assert entry.user_id == 42

    def test_roundtrip(self) -> None:
        """to_dict -> from_dict ergibt gleiche Daten."""
        original = SemanticEntry(
            user_id=555,
            content="Semantischer Fakt",
            category="praeferenz",
            importance=4,
        )
        reconstructed = SemanticEntry.from_dict(original.to_dict())
        assert reconstructed.id == original.id
        assert reconstructed.category == original.category
        assert reconstructed.content == original.content

    def test_id_prefix(self) -> None:
        """Jede neue ID beginnt mit sem_."""
        entries = [SemanticEntry() for _ in range(10)]
        for entry in entries:
            assert entry.id.startswith("sem_")

    def test_unique_ids(self) -> None:
        """Jede neue Entry bekommt eine eindeutige ID."""
        ids = {SemanticEntry().id for _ in range(100)}
        assert len(ids) == 100
