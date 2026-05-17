"""RequestEnvelope: immutable representation of a raw user request.

Channel-specific data (Telegram, Desktop, CLI) is normalized here.
After this point, the application layer works only with RequestEnvelope.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """Immutable envelope representing a single user request.

    Created at the edge (handler/channel adapter) and passed
    through the entire execution pipeline unchanged.

    Attributes:
        request_id: Unique identifier for this request (audit correlation).
        user_id: Numeric user ID from the channel.
        chat_id: Numeric chat/conversation ID.
        channel: Origin channel identifier.
        raw_text: Original unmodified user text.
        command: Extracted command (e.g. "debate") or None for plain messages.
        args: Command arguments (empty list for plain messages).
        timestamp_utc: When the request was received (UTC).
        username: Optional display name / username from the channel.
        reply_to_text: Text of the message being replied to (if any).
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: int = 0
    chat_id: int = 0
    channel: Literal["telegram", "desktop", "mini_app", "cli", "webhook"] = "telegram"
    raw_text: str = ""
    command: Optional[str] = None
    args: tuple[str, ...] = field(default_factory=tuple)
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    username: Optional[str] = None
    reply_to_text: Optional[str] = None

    @classmethod
    def from_telegram(
        cls,
        user_id: int,
        chat_id: int,
        text: str,
        username: Optional[str] = None,
        reply_to_text: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[list[str]] = None,
    ) -> "RequestEnvelope":
        """Create an envelope from Telegram message data.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            text: Message text.
            username: Telegram username.
            reply_to_text: Reply-to message text.
            command: Extracted command name (without /).
            args: Command arguments.

        Returns:
            Frozen RequestEnvelope instance.
        """
        return cls(
            user_id=user_id,
            chat_id=chat_id,
            channel="telegram",
            raw_text=text,
            command=command,
            args=tuple(args) if args else (),
            username=username,
            reply_to_text=reply_to_text,
        )

    @classmethod
    def from_debate_command(
        cls,
        user_id: int,
        chat_id: int,
        question: str,
        username: Optional[str] = None,
    ) -> "RequestEnvelope":
        """Create an envelope for a /debate command.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            question: The debate question text.
            username: Telegram username.

        Returns:
            Frozen RequestEnvelope with command="debate".
        """
        return cls(
            user_id=user_id,
            chat_id=chat_id,
            channel="telegram",
            raw_text=question,
            command="debate",
            args=(question,),
            username=username,
        )
