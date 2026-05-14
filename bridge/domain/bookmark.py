"""Bookmark entity as dataclass.

Represents a saved Telegram bot response bookmark.
Pure data structure without I/O, without persistence logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Bookmark:
    """Single bookmark entry.

    Attributes:
        user_id: Telegram user ID of the owner.
        chat_id: Telegram chat ID where the message was sent.
        message_id: Telegram message ID of the bookmarked bot response.
        content: Full text of the bot response.
        timestamp: ISO-8601 timestamp of creation.
        username: Telegram username (optional).
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
        """Serialize the bookmark to a dictionary for JSONL storage."""
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
        """Deserialize a bookmark from a dictionary."""
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
    """Format a bookmark as a short preview line.

    Args:
        bm: Bookmark dict (read from JSONL).
        index: Display index (1-based).

    Returns:
        Formatted preview line with date and content excerpt.
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
