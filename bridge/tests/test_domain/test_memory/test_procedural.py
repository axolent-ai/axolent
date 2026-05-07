"""Tests fuer ProceduralEntry: Erstellung, Serialisierung, Deserialisierung."""

from __future__ import annotations

from domain.memory.procedural import ProceduralEntry


class TestProceduralEntry:
    """Tests fuer ProceduralEntry Datenklasse."""

    def test_default_creation(self) -> None:
        """Entry mit Default-Werten hat korrekte Struktur."""
        entry = ProceduralEntry()
        assert entry.id.startswith("pro_")
        assert len(entry.id) == 16  # pro_ + 12 hex chars
        assert entry.user_id == 0
        assert entry.content == ""
        assert entry.skill_name == ""
        assert entry.usage_count == 0
        assert entry.importance == 5

    def test_creation_with_skill(self) -> None:
        """Entry mit Skill-Name und Usage-Count."""
        entry = ProceduralEntry(
            user_id=123,
            content="Wenn User nach Code fragt, Codeblock verwenden",
            skill_name="code_format",
            usage_count=5,
            importance=9,
        )
        assert entry.skill_name == "code_format"
        assert entry.usage_count == 5
        assert entry.importance == 9

    def test_to_dict_includes_skill_fields(self) -> None:
        """to_dict enthaelt skill_name und usage_count."""
        entry = ProceduralEntry(skill_name="kurze_antworten", usage_count=3)
        d = entry.to_dict()
        assert d["skill_name"] == "kurze_antworten"
        assert d["usage_count"] == 3

    def test_from_dict(self) -> None:
        """from_dict deserialisiert korrekt inklusive Skill-Felder."""
        data = {
            "id": "pro_abc123def456",
            "user_id": 42,
            "content": "Skill-Beschreibung",
            "skill_name": "debug_flow",
            "usage_count": 12,
            "context": {},
            "timestamp": "2026-05-07T02:00:00+00:00",
            "importance": 7,
        }
        entry = ProceduralEntry.from_dict(data)
        assert entry.skill_name == "debug_flow"
        assert entry.usage_count == 12

    def test_roundtrip(self) -> None:
        """to_dict -> from_dict ergibt gleiche Daten."""
        original = ProceduralEntry(
            user_id=999,
            content="Voyager-Pattern: neue Skills extrahieren",
            skill_name="skill_extraction",
            usage_count=0,
            importance=10,
        )
        reconstructed = ProceduralEntry.from_dict(original.to_dict())
        assert reconstructed.id == original.id
        assert reconstructed.skill_name == original.skill_name
        assert reconstructed.usage_count == original.usage_count

    def test_id_prefix(self) -> None:
        """Jede neue ID beginnt mit pro_."""
        entries = [ProceduralEntry() for _ in range(10)]
        for entry in entries:
            assert entry.id.startswith("pro_")

    def test_unique_ids(self) -> None:
        """Jede neue Entry bekommt eine eindeutige ID."""
        ids = {ProceduralEntry().id for _ in range(100)}
        assert len(ids) == 100
