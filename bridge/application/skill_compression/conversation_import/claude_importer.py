"""Claude JSON/JSONL export importer for Skill-Compression.

Parses the official Claude data export format (conversations.jsonl).
The export is a JSONL file where each line is a complete conversation
object, or alternatively a JSON array of conversation objects.

Schema reference (Anthropic Data Export, 2024+):
  Each conversation object:
  {
    "uuid": "...",
    "name": "Conversation Name",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "chat_messages": [
      {
        "uuid": "...",
        "text": "Message text",
        "sender": "human" | "assistant",
        "created_at": "..."
      }
    ]
  }

No external dependencies. Pure json parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from application.skill_compression.conversation_import.conversation_source import (
    ParsedConversation,
    make_parsed_conversation,
)

log = logging.getLogger(__name__)

# Maximum file size to attempt parsing (500 MB safety limit)
_MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024


class ClaudeImporter:
    """Parses Claude JSONL/JSON export files (conversations.jsonl).

    Supports both JSONL format (one conversation per line) and
    plain JSON array format.

    Implements ConversationSource protocol.
    """

    def can_handle(self, path: Path) -> bool:
        """Check if the file is a Claude export.

        Detects by:
          1. .jsonl or .json extension
          2. Contains 'chat_messages' key (Claude-specific)
          3. Messages use 'sender' field with 'human'/'assistant'

        Args:
            path: File path to check.

        Returns:
            True if this looks like a Claude conversations export.
        """
        suffix = path.suffix.lower()
        if suffix not in (".jsonl", ".json"):
            return False

        try:
            size = path.stat().st_size
            if size > _MAX_FILE_SIZE_BYTES:
                log.warning(
                    "File too large for Claude import: %s (%d bytes)",
                    path,
                    size,
                )
                return False

            # Read first 4KB to detect format
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                header = fh.read(4096)

            # Claude exports use 'chat_messages' and 'sender'
            return '"chat_messages"' in header and '"sender"' in header

        except OSError:
            return False

    def parse(self, path: Path) -> Iterator[ParsedConversation]:
        """Parse a Claude export file and yield conversations.

        Handles both JSONL (one object per line) and JSON array formats.

        Args:
            path: Path to conversations.jsonl or .json.

        Yields:
            ParsedConversation instances.
        """
        source_path = str(path.resolve())

        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as exc:
            log.warning("Failed to read Claude export: %s: %s", path, exc)
            return

        stripped = content.strip()

        # Detect format: JSONL (lines) or JSON array
        if stripped.startswith("["):
            # JSON array format
            yield from self._parse_json_array(stripped, source_path)
        else:
            # JSONL format: one JSON object per line
            yield from self._parse_jsonl(stripped, source_path)

    def _parse_json_array(
        self,
        content: str,
        source_path: str,
    ) -> Iterator[ParsedConversation]:
        """Parse a JSON array of conversation objects.

        Args:
            content: Full file content.
            source_path: Original file path.

        Yields:
            ParsedConversation instances.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            log.warning("Invalid JSON in Claude export: %s", exc)
            return

        if not isinstance(data, list):
            return

        for entry in data:
            conversation = self._parse_conversation(entry, source_path)
            if conversation is not None:
                yield conversation

    def _parse_jsonl(
        self,
        content: str,
        source_path: str,
    ) -> Iterator[ParsedConversation]:
        """Parse JSONL format (one JSON object per line).

        Args:
            content: Full file content.
            source_path: Original file path.

        Yields:
            ParsedConversation instances.
        """
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                log.debug("Skipping invalid JSON at line %d", line_num)
                continue

            conversation = self._parse_conversation(entry, source_path)
            if conversation is not None:
                yield conversation

    def _parse_conversation(
        self,
        entry: dict[str, Any],
        source_path: str,
    ) -> ParsedConversation | None:
        """Parse a single Claude conversation object.

        Args:
            entry: Single conversation dict.
            source_path: Original file path.

        Returns:
            ParsedConversation or None if empty/invalid.
        """
        if not isinstance(entry, dict):
            return None

        messages = entry.get("chat_messages")
        if not isinstance(messages, list):
            return None

        user_messages: list[str] = []
        assistant_messages: list[str] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            sender = msg.get("sender", "")
            text = msg.get("text", "")

            # Fallback: some exports use 'content' instead of 'text'
            if not text and "content" in msg:
                content = msg["content"]
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text_parts = [
                        str(p) for p in content if isinstance(p, str) and p.strip()
                    ]
                    text = "\n".join(text_parts)

            if not isinstance(text, str) or not text.strip():
                continue

            if sender == "human":
                user_messages.append(text.strip())
            elif sender == "assistant":
                assistant_messages.append(text.strip())

        if not user_messages and not assistant_messages:
            return None

        return make_parsed_conversation(
            source_path=source_path,
            source_type="claude",
            user_messages=user_messages,
            assistant_messages=assistant_messages,
        )
