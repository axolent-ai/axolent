"""Bookmark-Entity als Dataclass.

Repräsentiert einen gespeicherten Telegram-Bot-Antwort-Bookmark.
Reine Datenstruktur ohne I/O, ohne Persistenz-Logik.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Bookmark:
    """Einzelner Bookmark-Eintrag.

    Attributes:
        user_id: Telegram User-ID des Besitzers.
        chat_id: Telegram Chat-ID in dem die Nachricht gesendet wurde.
        message_id: Telegram Message-ID der gebookmarkten Bot-Antwort.
        content: Volltext der Bot-Antwort.
        timestamp: ISO-8601 Zeitstempel der Erstellung.
        username: Telegram Username (optional).
    """

    user_id: int
    chat_id: int
    message_id: int
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    username: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den Bookmark zu einem Dictionary für JSONL-Storage."""
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "username": self.username,
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bookmark:
        """Deserialisiert einen Bookmark aus einem Dictionary."""
        return cls(
            user_id=data.get("user_id", 0),
            chat_id=data.get("chat_id", 0),
            message_id=data.get("message_id", 0),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", ""),
            username=data.get("username"),
        )


BOOKMARK_PREVIEW_LEN: int = 200


def format_bookmark_preview(bm: dict[str, Any], index: int) -> str:
    """Formatiert einen Bookmark als Kurzvorschau-Zeile.

    Args:
        bm: Bookmark-Dict (aus JSONL gelesen).
        index: Anzeige-Index (1-basiert).

    Returns:
        Formatierte Preview-Zeile mit Datum und Content-Ausschnitt.
    """
    ts = bm.get("timestamp", "?")
    try:
        dt = datetime.fromisoformat(ts)
        ts_display = dt.strftime("%d.%m.%Y %H:%M")
    except (ValueError, TypeError):
        ts_display = ts

    content = bm.get("content", "")
    preview = content[:BOOKMARK_PREVIEW_LEN]
    if len(content) > BOOKMARK_PREVIEW_LEN:
        preview += "..."

    return f"{index}. [{ts_display}]\n{preview}"
