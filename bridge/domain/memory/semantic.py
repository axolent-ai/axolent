"""Semantic Memory: Generalisierte Fakten.

Abstrahiertes Wissen, das aus episodischen Erfahrungen extrahiert wurde.
Beispiel: "User bevorzugt kurze Antworten", "User arbeitet an Jarvis-LITE"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class SemanticEntry:
    """Ein Fakt in der Semantic Memory.

    Attributes:
        id: Eindeutige ID mit Prefix sem_ für Layer-Erkennung.
        user_id: Telegram-User-ID.
        content: Der generalisierte Fakt.
        category: Klassifizierung (z.B. "fakt", "person", "präferenz", "projekt").
        context: Optionaler Kontext (Quelle, Confidence).
        timestamp: ISO 8601 UTC.
        importance: 1-10, Auto-Score oder User-vergeben.
    """

    id: str = field(default_factory=lambda: f"sem_{uuid4().hex[:12]}")
    user_id: int = 0
    content: str = ""
    category: str = "fakt"
    context: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    importance: int = 5

    def to_dict(self) -> dict:
        """Serialisiert den Entry als Dict für JSONL-Persistierung."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "category": self.category,
            "context": self.context,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SemanticEntry:
        """Deserialisiert einen Entry aus einem Dict."""
        return cls(
            id=data.get("id", f"sem_{uuid4().hex[:12]}"),
            user_id=data.get("user_id", 0),
            content=data.get("content", ""),
            category=data.get("category", "fakt"),
            context=data.get("context", {}),
            timestamp=data.get("timestamp", ""),
            importance=data.get("importance", 5),
        )
