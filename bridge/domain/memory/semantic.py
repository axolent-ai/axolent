"""Semantic memory: generalized facts.

Abstracted knowledge extracted from episodic experiences.
Example: "User prefers short answers", "User works on Axolent"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class SemanticEntry:
    """A fact in semantic memory.

    Attributes:
        id: Unique ID with prefix sem_ for layer detection.
        user_id: Telegram user ID.
        content: The generalized fact.
        category: Classification (e.g. "fakt", "person", "preference", "projekt").
        context: Optional context (source, confidence).
        timestamp: ISO 8601 UTC.
        importance: 1-10, auto-scored or user-assigned.
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
        """Serialize the entry as dict for JSONL persistence."""
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
        """Deserialize an entry from a dict."""
        return cls(
            id=data.get("id", f"sem_{uuid4().hex[:12]}"),
            user_id=data.get("user_id", 0),
            content=data.get("content", ""),
            category=data.get("category", "fakt"),
            context=data.get("context", {}),
            timestamp=data.get("timestamp", ""),
            importance=data.get("importance", 5),
        )
