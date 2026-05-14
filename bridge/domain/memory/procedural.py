"""Procedural memory: skills and patterns.

Repeatable action patterns the bot has learned.
Example: "When user asks for code, always respond with a code block"
Phase 1+: Voyager-pattern skill compression will fill this layer automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ProceduralEntry:
    """A skill/pattern in procedural memory.

    Attributes:
        id: Unique ID with prefix pro_ for layer detection.
        user_id: Telegram user ID.
        content: Description of the skill/pattern.
        skill_name: Short name of the skill (e.g. "code_format", "short_answers").
        usage_count: How often the skill has been applied.
        context: Optional context.
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
        """Serialize the entry as dict for JSONL persistence."""
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
        """Deserialize an entry from a dict."""
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
