"""Tests für EpisodicEntry: Erstellung, Serialisierung, Deserialisierung."""

from __future__ import annotations

from domain.memory.episodic import EpisodicEntry


class TestEpisodicEntry:
    """Tests für EpisodicEntry Datenklasse."""

    def test_default_creation(self) -> None:
        """Entry mit Default-Werten hat korrekte Struktur."""
        entry = EpisodicEntry()
        assert entry.id.startswith("ep_")
        assert len(entry.id) == 15  # ep_ + 12 hex chars
        assert entry.user_id == 0
        assert entry.content == ""
        assert entry.context == {}
        assert entry.importance == 5
        assert entry.timestamp != ""

    def test_creation_with_values(self) -> None:
        """Entry mit expliziten Werten übernimmt alle Felder."""
        entry = EpisodicEntry(
            user_id=12345,
            content="User hat nach RAG gefragt",
            importance=8,
            context={"source": "telegram"},
        )
        assert entry.user_id == 12345
        assert entry.content == "User hat nach RAG gefragt"
        assert entry.importance == 8
        assert entry.context == {"source": "telegram"}

    def test_to_dict(self) -> None:
        """to_dict serialisiert alle Felder korrekt."""
        entry = EpisodicEntry(
            id="ep_test123456ab",
            user_id=99,
            content="Testinhalt",
            context={"key": "value"},
            timestamp="2026-05-07T01:00:00+00:00",
            importance=7,
        )
        d = entry.to_dict()
        assert d["id"] == "ep_test123456ab"
        assert d["user_id"] == 99
        assert d["content"] == "Testinhalt"
        assert d["context"] == {"key": "value"}
        assert d["timestamp"] == "2026-05-07T01:00:00+00:00"
        assert d["importance"] == 7

    def test_from_dict(self) -> None:
        """from_dict deserialisiert korrekt."""
        data = {
            "id": "ep_abc123def456",
            "user_id": 42,
            "content": "Deserialisiert",
            "context": {},
            "timestamp": "2026-05-07T02:00:00+00:00",
            "importance": 3,
        }
        entry = EpisodicEntry.from_dict(data)
        assert entry.id == "ep_abc123def456"
        assert entry.user_id == 42
        assert entry.content == "Deserialisiert"
        assert entry.importance == 3

    def test_roundtrip(self) -> None:
        """to_dict -> from_dict ergibt gleiche Daten."""
        original = EpisodicEntry(
            user_id=777,
            content="Roundtrip-Test mit Umlauten: äöüß",
            importance=9,
        )
        reconstructed = EpisodicEntry.from_dict(original.to_dict())
        assert reconstructed.id == original.id
        assert reconstructed.user_id == original.user_id
        assert reconstructed.content == original.content
        assert reconstructed.importance == original.importance

    def test_id_prefix(self) -> None:
        """Jede neue ID beginnt mit ep_."""
        entries = [EpisodicEntry() for _ in range(10)]
        for entry in entries:
            assert entry.id.startswith("ep_")

    def test_unique_ids(self) -> None:
        """Jede neue Entry bekommt eine eindeutige ID."""
        ids = {EpisodicEntry().id for _ in range(100)}
        assert len(ids) == 100
