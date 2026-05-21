"""Conversation source protocol and data structures for import.

Defines the ConversationSource protocol that all parsers implement,
and the ParsedConversation dataclass that carries extracted messages.

All parsers are local-only (Mode B). No cloud API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ParsedConversation:
    """A single parsed conversation from an external source.

    Attributes:
        source_path: Absolute path to the source file.
        source_type: Parser identifier ("chatgpt" / "claude" / "markdown" / "plaintext").
        user_messages: Extracted user messages (immutable tuple).
        assistant_messages: Extracted assistant messages (immutable tuple).
        parsed_at: ISO-8601 UTC timestamp of when parsing occurred.
    """

    source_path: str
    source_type: str
    user_messages: tuple[str, ...]
    assistant_messages: tuple[str, ...]
    parsed_at: str


def make_parsed_conversation(
    *,
    source_path: str,
    source_type: str,
    user_messages: list[str] | tuple[str, ...],
    assistant_messages: list[str] | tuple[str, ...],
) -> ParsedConversation:
    """Factory for creating ParsedConversation with current timestamp.

    Args:
        source_path: Absolute path to the source file.
        source_type: Parser identifier.
        user_messages: User messages (will be converted to tuple).
        assistant_messages: Assistant messages (will be converted to tuple).

    Returns:
        A new ParsedConversation instance.
    """
    return ParsedConversation(
        source_path=source_path,
        source_type=source_type,
        user_messages=tuple(user_messages),
        assistant_messages=tuple(assistant_messages),
        parsed_at=datetime.now(timezone.utc).isoformat(),
    )


@runtime_checkable
class ConversationSource(Protocol):
    """Protocol for conversation import sources.

    Each parser implements this protocol to handle a specific file
    format. The orchestrator iterates over registered sources and
    delegates to the first one that can handle a given path.
    """

    def can_handle(self, path: Path) -> bool:
        """Check whether this source can parse the given file.

        Args:
            path: Path to the file to check.

        Returns:
            True if this parser can handle the file.
        """
        ...

    def parse(self, path: Path) -> Iterator[ParsedConversation]:
        """Parse a file and yield conversation records.

        A single file may contain multiple conversations (e.g. ChatGPT
        export with hundreds of threads). Each is yielded separately.

        Args:
            path: Path to the file to parse.

        Yields:
            ParsedConversation instances.
        """
        ...
