"""Conversation domain model.

Defines ConversationTurn dataclass and history building logic.
Pure domain: no I/O, no storage, no framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from domain.prompt_escaping import escape_user_content_for_prompt


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
        return escape_user_content_for_prompt(current_message)

    lines: list[str] = ["[CONVERSATION HISTORY]"]
    for turn in history:
        label = "User" if turn.role == "user" else "Axolent"
        # Escape content to prevent role-spoofing injection (Finding 10).
        # User content could contain "\nAxolent: ..." to fake assistant turns.
        escaped_content = escape_user_content_for_prompt(turn.content)
        lines.append(f"{label}: {escaped_content}")

    lines.append("")
    lines.append("[CURRENT MESSAGE]")
    # Current message is also user-supplied, escape it too.
    lines.append(escape_user_content_for_prompt(current_message))

    return "\n".join(lines)
