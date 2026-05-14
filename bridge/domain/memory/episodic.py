"""Episodic memory: what happened.

Concrete events, conversation flows, observations with timestamps and context.
Example: "User asked about acquirers on 2026-05-07"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class EpisodicEntry:
    """A single event in episodic memory.

    Attributes:
        id: Unique ID with prefix ep_ for layer detection.
        user_id: Telegram user ID.
        content: Description of the event.
        context: Optional context (workspace, tags, source).
        timestamp: ISO 8601 UTC.
        importance: 1-10, assigned by user or auto-scored.
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
        """Serialize the entry as dict for JSONL persistence."""
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
        """Deserialize an entry from a dict."""
        return cls(
            id=data.get("id", f"ep_{uuid4().hex[:12]}"),
            user_id=data.get("user_id", 0),
            content=data.get("content", ""),
            context=data.get("context", {}),
            timestamp=data.get("timestamp", ""),
            importance=data.get("importance", 5),
        )
