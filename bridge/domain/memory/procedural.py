"""Procedural Memory: Skills und Patterns.

Wiederholbare Handlungsmuster, die der Bot gelernt hat.
Beispiel: "Wenn User nach Code fragt, immer mit Codeblock antworten"
Phase 1+: Voyager-Pattern Skill-Compression füllt diesen Layer automatisch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ProceduralEntry:
    """Ein Skill/Pattern in der Procedural Memory.

    Attributes:
        id: Eindeutige ID mit Prefix pro_ für Layer-Erkennung.
        user_id: Telegram-User-ID.
        content: Beschreibung des Skills/Patterns.
        skill_name: Kurzname des Skills (z.B. "code_format", "kurze_antworten").
        usage_count: Wie oft der Skill angewendet wurde.
        context: Optionaler Kontext.
        timestamp: ISO 8601 UTC.
        importance: 1-10.
    """

    id: str = field(default_factory=lambda: f"pro_{uuid4().hex[:12]}")
    user_id: int = 0
    content: str = ""
    skill_name: str = ""
    usage_count: int = 0
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
            "skill_name": self.skill_name,
            "usage_count": self.usage_count,
            "context": self.context,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProceduralEntry:
        """Deserialisiert einen Entry aus einem Dict."""
        return cls(
            id=data.get("id", f"pro_{uuid4().hex[:12]}"),
            user_id=data.get("user_id", 0),
            content=data.get("content", ""),
            skill_name=data.get("skill_name", ""),
            usage_count=data.get("usage_count", 0),
            context=data.get("context", {}),
            timestamp=data.get("timestamp", ""),
            importance=data.get("importance", 5),
        )
