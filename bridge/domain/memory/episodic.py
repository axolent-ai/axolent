"""Episodic Memory: Was passiert ist.

Konkrete Events, Gesprächsverläufe, Beobachtungen mit Zeitstempel und Kontext.
Beispiel: "User hat am 2026-05-07 nach Acquirern gefragt"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class EpisodicEntry:
    """Ein einzelnes Event in der Episodic Memory.

    Attributes:
        id: Eindeutige ID mit Prefix ep_ fuer Layer-Erkennung.
        user_id: Telegram-User-ID.
        content: Beschreibung des Events.
        context: Optionaler Kontext (Workspace, Tags, Quelle).
        timestamp: ISO 8601 UTC.
        importance: 1-10, von User vergeben oder Auto-Score.
    """

    id: str = field(default_factory=lambda: f"ep_{uuid4().hex[:12]}")
    user_id: int = 0
    content: str = ""
    context: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    importance: int = 5

    def to_dict(self) -> dict:
        """Serialisiert den Entry als Dict fuer JSONL-Persistierung."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "context": self.context,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EpisodicEntry:
        """Deserialisiert einen Entry aus einem Dict."""
        return cls(
            id=data.get("id", f"ep_{uuid4().hex[:12]}"),
            user_id=data.get("user_id", 0),
            content=data.get("content", ""),
            context=data.get("context", {}),
            timestamp=data.get("timestamp", ""),
            importance=data.get("importance", 5),
        )
