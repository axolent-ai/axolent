"""Tests für domain.bookmark: Bookmark-Dataclass und Serialisierung.

Testet Erstellung, Serialisierung und Deserialisierung der Bookmark-Entity.
"""

from domain.bookmark import Bookmark, format_bookmark_preview


class TestBookmarkDataclass:
    """Bookmark-Dataclass Grundfunktionalitaet."""

    def test_bookmark_dataclass_creation(self) -> None:
        """Bookmark kann mit allen Pflichtfeldern erstellt werden."""
        bm = Bookmark(
            user_id=123,
            chat_id=456,
            message_id=789,
            content="Test content",
        )
        assert bm.user_id == 123
        assert bm.chat_id == 456
        assert bm.message_id == 789
        assert bm.content == "Test content"
        assert bm.timestamp  # auto-generiert
        assert bm.username is None  # Optional

    def test_bookmark_frozen(self) -> None:
        """Bookmark ist frozen (immutable)."""
        bm = Bookmark(user_id=1, chat_id=2, message_id=3, content="x")
        try:
            bm.content = "changed"  # type: ignore[misc]
            assert False, "FrozenInstanceError erwartet"
        except Exception:
            pass  # Korrekt: frozen dataclass wirft Fehler

    def test_bookmark_serialization(self) -> None:
        """to_dict erzeugt alle erwarteten Keys."""
        bm = Bookmark(
            user_id=123,
            chat_id=456,
            message_id=789,
            content="Hallo Welt",
            timestamp="2026-05-06T12:00:00+00:00",
            username="testuser",
        )
        d = bm.to_dict()
        assert d["user_id"] == 123
        assert d["chat_id"] == 456
        assert d["message_id"] == 789
        assert d["content"] == "Hallo Welt"
        assert d["timestamp"] == "2026-05-06T12:00:00+00:00"
        assert d["username"] == "testuser"

    def test_bookmark_deserialization(self) -> None:
        """from_dict rekonstruiert den Bookmark korrekt."""
        data = {
            "user_id": 111,
            "chat_id": 222,
            "message_id": 333,
            "content": "Bookmark Inhalt",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "username": "jess",
        }
        bm = Bookmark.from_dict(data)
        assert bm.user_id == 111
        assert bm.chat_id == 222
        assert bm.message_id == 333
        assert bm.content == "Bookmark Inhalt"
        assert bm.username == "jess"

    def test_bookmark_from_dict_missing_fields(self) -> None:
        """from_dict mit fehlenden Feldern nutzt Defaults (kein Crash)."""
        bm = Bookmark.from_dict({})
        assert bm.user_id == 0
        assert bm.chat_id == 0
        assert bm.message_id == 0
        assert bm.content == ""

    def test_bookmark_roundtrip(self) -> None:
        """to_dict -> from_dict muss idempotent sein."""
        original = Bookmark(
            user_id=42,
            chat_id=99,
            message_id=7,
            content="Roundtrip-Test",
            timestamp="2026-05-06T10:00:00+00:00",
            username="round",
        )
        reconstructed = Bookmark.from_dict(original.to_dict())
        assert reconstructed.user_id == original.user_id
        assert reconstructed.content == original.content
        assert reconstructed.timestamp == original.timestamp


class TestFormatBookmarkPreview:
    """Preview-Formatierung für Bookmark-Auflistung."""

    def test_format_bookmark_preview_short(self) -> None:
        """Kurze Bookmarks werden vollstaendig angezeigt."""
        bm = {
            "timestamp": "2026-05-06T12:00:00+00:00",
            "content": "Kurzer Text",
        }
        result = format_bookmark_preview(bm, 1)
        assert "1." in result
        assert "Kurzer Text" in result
        assert "06.05.2026" in result

    def test_format_bookmark_preview_truncated(self) -> None:
        """Lange Bookmarks werden nach 200 Zeichen mit '...' abgeschnitten."""
        long_content = "A" * 300
        bm = {
            "timestamp": "2026-05-06T12:00:00+00:00",
            "content": long_content,
        }
        result = format_bookmark_preview(bm, 1)
        assert "..." in result
        assert len(result) < 300  # Wurde gekuerzt
