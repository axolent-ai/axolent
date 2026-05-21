"""Plain text conversation importer for Skill-Compression.

Parses generic text files that contain conversation-style content
using heuristic detection of User:/Assistant: (or similar) labels.

This is the fallback importer for files that do not match any
structured format. It uses simple line-by-line heuristic parsing.

Supported label patterns (case-insensitive):
  - User: / Assistant:
  - Human: / AI:
  - Q: / A:
  - Me: / Bot:
  - Ich: / Assistent:

No external dependencies. Pure string parsing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from application.skill_compression.conversation_import.conversation_source import (
    ParsedConversation,
    make_parsed_conversation,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------

# Role label at start of line
_USER_LABEL = re.compile(
    r"^(User|Human|Me|Ich|Benutzer|Q)\s*:\s*",
    re.IGNORECASE,
)
_ASSISTANT_LABEL = re.compile(
    r"^(Assistant|AI|Bot|Assistent|ChatGPT|Claude|GPT|A)\s*:\s*",
    re.IGNORECASE,
)

# Combined detection: does the file contain at least one user + one assistant label?
_HAS_USER_LABEL = re.compile(
    r"^(User|Human|Me|Ich|Benutzer|Q)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_HAS_ASSISTANT_LABEL = re.compile(
    r"^(Assistant|AI|Bot|Assistent|ChatGPT|Claude|GPT|A)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# Minimum file size to consider
_MIN_CONTENT_LENGTH = 20

# Maximum file size (10 MB for plain text)
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class PlaintextImporter:
    """Parses plain text files with User:/Assistant: heuristics.

    Implements ConversationSource protocol.
    This is the most permissive parser and serves as fallback.
    """

    def can_handle(self, path: Path) -> bool:
        """Check if the file is a text file with conversation labels.

        Args:
            path: File path to check.

        Returns:
            True if .txt extension and contains role labels.
        """
        if path.suffix.lower() != ".txt":
            return False

        try:
            size = path.stat().st_size
            if size > _MAX_FILE_SIZE_BYTES:
                return False

            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

        if len(text.strip()) < _MIN_CONTENT_LENGTH:
            return False

        # Must have at least one of each role
        has_user = bool(_HAS_USER_LABEL.search(text))
        has_assistant = bool(_HAS_ASSISTANT_LABEL.search(text))
        return has_user and has_assistant

    def parse(self, path: Path) -> Iterator[ParsedConversation]:
        """Parse a plain text file and yield conversations.

        Splits the file into conversation blocks. A new conversation
        starts when a user label appears after a gap (empty line after
        an assistant message).

        Args:
            path: Path to the .txt file.

        Yields:
            ParsedConversation instances.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log.warning("Cannot read file: %s", path)
            return

        if len(text.strip()) < _MIN_CONTENT_LENGTH:
            return

        conversations = self._split_conversations(text)
        source_path = str(path.resolve())

        for user_msgs, assistant_msgs in conversations:
            if user_msgs or assistant_msgs:
                yield make_parsed_conversation(
                    source_path=source_path,
                    source_type="plaintext",
                    user_messages=user_msgs,
                    assistant_messages=assistant_msgs,
                )

    def _split_conversations(
        self,
        text: str,
    ) -> list[tuple[list[str], list[str]]]:
        """Split text into conversation blocks by role labels.

        Parses line by line: when a role label is found, starts
        collecting content for that role. Content continues until
        the next role label.

        Args:
            text: Full file text.

        Returns:
            List of (user_messages, assistant_messages) tuples.
        """
        user_messages: list[str] = []
        assistant_messages: list[str] = []
        current_role: str | None = None
        current_content: list[str] = []

        for line in text.splitlines():
            user_match = _USER_LABEL.match(line)
            assistant_match = _ASSISTANT_LABEL.match(line)

            if user_match:
                # Flush previous content
                self._flush_content(
                    current_role,
                    current_content,
                    user_messages,
                    assistant_messages,
                )
                current_role = "user"
                # Content after label on same line
                remainder = line[user_match.end() :].strip()
                current_content = [remainder] if remainder else []

            elif assistant_match:
                # Flush previous content
                self._flush_content(
                    current_role,
                    current_content,
                    user_messages,
                    assistant_messages,
                )
                current_role = "assistant"
                remainder = line[assistant_match.end() :].strip()
                current_content = [remainder] if remainder else []

            elif current_role is not None:
                # Continuation line for current role
                current_content.append(line)

        # Flush final content
        self._flush_content(
            current_role,
            current_content,
            user_messages,
            assistant_messages,
        )

        if not user_messages and not assistant_messages:
            return []

        return [(user_messages, assistant_messages)]

    @staticmethod
    def _flush_content(
        role: str | None,
        content_lines: list[str],
        user_messages: list[str],
        assistant_messages: list[str],
    ) -> None:
        """Flush accumulated content lines into the appropriate list.

        Args:
            role: Current role ("user" or "assistant") or None.
            content_lines: Accumulated content lines.
            user_messages: User message accumulator (mutated).
            assistant_messages: Assistant message accumulator (mutated).
        """
        if role is None or not content_lines:
            return

        text = "\n".join(content_lines).strip()
        if not text:
            return

        if role == "user":
            user_messages.append(text)
        elif role == "assistant":
            assistant_messages.append(text)
