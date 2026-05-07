"""Conversation Domain Model.

Definiert ConversationTurn-Dataclass und History-Building-Logik.
Pure Domain: kein I/O, kein Storage, keine Framework-Abhängigkeiten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Ein einzelner Turn in der Konversation (User-Nachricht oder Bot-Antwort).

    Attributes:
        role: "user" oder "assistant".
        content: Der Nachrichtentext.
        timestamp: UTC-Zeitstempel des Turns.
    """

    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


MAX_HISTORY_TURNS: int = 20


def build_context_block(history: list[ConversationTurn], current_message: str) -> str:
    """Baut den Konversations-Kontext-String für Claude.

    Format:
        [VERLAUF DER UNTERHALTUNG]
        User: ...
        Jarvis: ...

        [AKTUELLE NACHRICHT]
        <current message>

    Bei leerer History wird nur die aktuelle Nachricht zurückgegeben (kein Wrapper).

    Args:
        history: Liste vorheriger ConversationTurns (bereits auf Max getrimmt).
        current_message: Die aktuelle User-Nachricht.

    Returns:
        Formatierter Kontext-String für den Claude-Prompt.
    """
    if not history:
        return current_message

    lines: list[str] = ["[VERLAUF DER UNTERHALTUNG]"]
    for turn in history:
        label = "User" if turn.role == "user" else "Jarvis"
        lines.append(f"{label}: {turn.content}")

    lines.append("")
    lines.append("[AKTUELLE NACHRICHT]")
    lines.append(current_message)

    return "\n".join(lines)
