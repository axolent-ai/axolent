"""Conversation domain model.

Defines ConversationTurn dataclass and history building logic.
Pure domain: no I/O, no storage, no framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """A single turn in the conversation (user message or bot response).

    Attributes:
        role: "user" or "assistant".
        content: The message text.
        timestamp: UTC timestamp of the turn.
    """

    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


MAX_HISTORY_TURNS: int = 20


def build_context_block(history: list[ConversationTurn], current_message: str) -> str:
    """Build the conversation context string for Claude.

    Format:
        [CONVERSATION HISTORY]
        User: ...
        Axolent: ...

        [CURRENT MESSAGE]
        <current message>

    With empty history, only the current message is returned (no wrapper).

    Args:
        history: List of previous ConversationTurns (already trimmed to max).
        current_message: The current user message.

    Returns:
        Formatted context string for the Claude prompt.
    """
    if not history:
        return current_message

    lines: list[str] = ["[CONVERSATION HISTORY]"]
    for turn in history:
        label = "User" if turn.role == "user" else "Axolent"
        lines.append(f"{label}: {turn.content}")

    lines.append("")
    lines.append("[CURRENT MESSAGE]")
    lines.append(current_message)

    return "\n".join(lines)
