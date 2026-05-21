"""Markdown conversation importer for Skill-Compression.

Parses .md files that contain conversation-style content, typically
from Obsidian or similar note-taking apps. Looks for alternating
user/assistant blocks marked by common heading or label patterns.

Supported patterns:
  - ## User / ## Assistant headings
  - **User:** / **Assistant:** bold labels
  - > User: / > Assistant: blockquote labels
  - Human: / Assistant: plain labels (Claude-style)

No external dependencies. Pure regex-based parsing.
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
# Detection patterns for user/assistant blocks
# ---------------------------------------------------------------

# Heading-style: ## User, ## Human, ## Assistant, ## AI, ## Bot
_HEADING_PATTERN = re.compile(
    r"^#{1,4}\s+(User|Human|Me|Ich|Benutzer)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_HEADING_ASSISTANT_PATTERN = re.compile(
    r"^#{1,4}\s+(Assistant|AI|Bot|ChatGPT|Claude|GPT)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Bold-label: **User:** or **Human:**
_BOLD_LABEL_PATTERN = re.compile(
    r"^\*\*(User|Human|Me|Ich|Benutzer)\*\*:\s*",
    re.IGNORECASE | re.MULTILINE,
)
_BOLD_ASSISTANT_PATTERN = re.compile(
    r"^\*\*(Assistant|AI|Bot|ChatGPT|Claude|GPT)\*\*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Plain-label: User: or Human: or Assistant:
_PLAIN_LABEL_PATTERN = re.compile(
    r"^(User|Human|Me|Ich|Benutzer):\s*",
    re.IGNORECASE | re.MULTILINE,
)
_PLAIN_ASSISTANT_LABEL = re.compile(
    r"^(Assistant|AI|Bot|ChatGPT|Claude|GPT):\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Combined splitter: splits on any user/assistant marker
# Matches headings (## User), bold labels (**User:**), and plain labels (User:).
# The marker line may have trailing content (bold-label style: **User:** text...).
# Three explicit alternatives to avoid greedy/ambiguous matching:
#   1. ## User        (heading style)
#   2. **User**:      (bold-label style)
#   3. User:          (plain label, with optional > for blockquotes)
_ROLE_SPLIT = re.compile(
    r"^(?:"
    r"#{1,4}\s+"
    r"(User|Human|Me|Ich|Benutzer|Assistant|AI|Bot|ChatGPT|Claude|GPT)"
    r"\s*"
    r"|"
    r"\*\*"
    r"(User|Human|Me|Ich|Benutzer|Assistant|AI|Bot|ChatGPT|Claude|GPT)"
    r":?\*\*:?\s*"
    r"|"
    r">?\s*"
    r"(User|Human|Me|Ich|Benutzer|Assistant|AI|Bot|ChatGPT|Claude|GPT)"
    r":\s*"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_USER_ROLES = frozenset({"user", "human", "me", "ich", "benutzer"})
_ASSISTANT_ROLES = frozenset({"assistant", "ai", "bot", "chatgpt", "claude", "gpt"})

# Minimum content length to consider a file as having conversations
_MIN_CONTENT_LENGTH = 20


class MarkdownImporter:
    """Parses Markdown files for conversation-style user/assistant blocks.

    Implements ConversationSource protocol.
    """

    def can_handle(self, path: Path) -> bool:
        """Check if the file is a .md file with conversation markers.

        Args:
            path: File path to check.

        Returns:
            True if .md extension and file contains conversation markers.
        """
        if path.suffix.lower() != ".md":
            return False

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

        # Must have at least one role marker
        return bool(_ROLE_SPLIT.search(text))

    def parse(self, path: Path) -> Iterator[ParsedConversation]:
        """Parse a Markdown file and yield conversations.

        A single .md file produces one ParsedConversation (unless it
        has clear conversation separators like ---, in which case
        each section becomes a separate conversation).

        Args:
            path: Path to the .md file.

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

        # Split into sections by horizontal rules (---, ***, ___)
        sections = re.split(r"\n(?:---+|\*\*\*+|___+)\n", text)

        for section in sections:
            conversation = self._parse_section(section, str(path.resolve()))
            if conversation is not None:
                yield conversation

    def _parse_section(
        self,
        text: str,
        source_path: str,
    ) -> ParsedConversation | None:
        """Parse a single section of Markdown into a conversation.

        Args:
            text: Section text.
            source_path: Original file path.

        Returns:
            ParsedConversation or None if no valid conversation found.
        """
        user_messages: list[str] = []
        assistant_messages: list[str] = []

        # Find all role markers and their positions
        markers: list[tuple[int, int, str]] = []  # (start, end, role)
        for match in _ROLE_SPLIT.finditer(text):
            # Extract role from whichever group matched (3 alternatives)
            role_raw = (
                match.group(1) or match.group(2) or match.group(3) or ""
            ).lower()
            if role_raw in _USER_ROLES:
                markers.append((match.start(), match.end(), "user"))
            elif role_raw in _ASSISTANT_ROLES:
                markers.append((match.start(), match.end(), "assistant"))

        if len(markers) < 2:
            return None

        # Extract content between markers
        for i, (_, end, role) in enumerate(markers):
            if i + 1 < len(markers):
                next_start = markers[i + 1][0]
                content = text[end:next_start].strip()
            else:
                content = text[end:].strip()

            if not content:
                continue

            if role == "user":
                user_messages.append(content)
            else:
                assistant_messages.append(content)

        if not user_messages and not assistant_messages:
            return None

        return make_parsed_conversation(
            source_path=source_path,
            source_type="markdown",
            user_messages=user_messages,
            assistant_messages=assistant_messages,
        )
