"""ChatGPT JSON export importer for Skill-Compression.

Parses the official ChatGPT data export format (conversations.json).
The export contains an array of conversation objects, each with a
'mapping' dict that holds the message tree.

Schema reference (OpenAI Data Export, 2024+):
  [
    {
      "title": "...",
      "mapping": {
        "<uuid>": {
          "message": {
            "author": {"role": "user" | "assistant" | "system"},
            "content": {"parts": ["text..."]}
          },
          "parent": "<uuid>" | null,
          "children": ["<uuid>", ...]
        }
      }
    },
    ...
  ]

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


class ChatGPTImporter:
    """Parses ChatGPT JSON export files (conversations.json).

    Implements ConversationSource protocol.
    """

    def can_handle(self, path: Path) -> bool:
        """Check if the file is a ChatGPT export JSON.

        Detects by:
          1. .json extension
          2. File starts with '[' (array)
          3. First object has 'mapping' key (ChatGPT-specific)

        Args:
            path: File path to check.

        Returns:
            True if this looks like a ChatGPT conversations.json.
        """
        if path.suffix.lower() != ".json":
            return False

        try:
            size = path.stat().st_size
            if size > _MAX_FILE_SIZE_BYTES:
                log.warning(
                    "File too large for ChatGPT import: %s (%d bytes)",
                    path,
                    size,
                )
                return False

            # Read first 4KB to detect format without loading entire file
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                header = fh.read(4096).lstrip()

            if not header.startswith("["):
                return False

            # Quick heuristic: check for "mapping" key in header
            return '"mapping"' in header

        except OSError:
            return False

    def parse(self, path: Path) -> Iterator[ParsedConversation]:
        """Parse a ChatGPT conversations.json and yield conversations.

        Each conversation object in the array becomes one ParsedConversation.
        System messages are skipped. Empty conversations are skipped.

        Args:
            path: Path to conversations.json.

        Yields:
            ParsedConversation instances.
        """
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Failed to parse ChatGPT export: %s: %s", path, exc)
            return

        if not isinstance(data, list):
            log.warning("ChatGPT export is not a JSON array: %s", path)
            return

        source_path = str(path.resolve())

        for entry in data:
            conversation = self._parse_conversation(entry, source_path)
            if conversation is not None:
                yield conversation

    def _parse_conversation(
        self,
        entry: dict[str, Any],
        source_path: str,
    ) -> ParsedConversation | None:
        """Parse a single ChatGPT conversation object.

        Walks the message tree via 'mapping' to extract user and
        assistant messages in order.

        Args:
            entry: Single conversation dict from the export.
            source_path: Original file path.

        Returns:
            ParsedConversation or None if empty/invalid.
        """
        if not isinstance(entry, dict):
            return None

        mapping = entry.get("mapping")
        if not isinstance(mapping, dict):
            return None

        # Build ordered message list by following parent->children
        messages = self._extract_messages_ordered(mapping)

        user_messages: list[str] = []
        assistant_messages: list[str] = []

        for role, text in messages:
            if role == "user":
                user_messages.append(text)
            elif role == "assistant":
                assistant_messages.append(text)

        if not user_messages and not assistant_messages:
            return None

        return make_parsed_conversation(
            source_path=source_path,
            source_type="chatgpt",
            user_messages=user_messages,
            assistant_messages=assistant_messages,
        )

    @staticmethod
    def _extract_messages_ordered(
        mapping: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Extract messages from ChatGPT mapping in conversation order.

        Finds the root node (no parent), then walks children depth-first
        to reconstruct the conversation order.

        Args:
            mapping: The 'mapping' dict from a ChatGPT conversation.

        Returns:
            List of (role, text) tuples in conversation order.
        """
        # Find root: node whose parent is None or not in mapping
        root_id = None
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent is None or parent not in mapping:
                root_id = node_id
                break

        if root_id is None:
            return []

        # BFS/DFS from root following children
        messages: list[tuple[str, str]] = []
        queue = [root_id]

        while queue:
            current_id = queue.pop(0)
            node = mapping.get(current_id)
            if node is None:
                continue

            # Extract message if present
            msg = node.get("message")
            if msg is not None and isinstance(msg, dict):
                author = msg.get("author", {})
                role = author.get("role", "") if isinstance(author, dict) else ""
                content = msg.get("content", {})

                text = ""
                if isinstance(content, dict):
                    parts = content.get("parts", [])
                    if isinstance(parts, list):
                        text_parts = [
                            str(p) for p in parts if isinstance(p, str) and p.strip()
                        ]
                        text = "\n".join(text_parts)
                elif isinstance(content, str):
                    text = content

                if text.strip() and role in ("user", "assistant"):
                    messages.append((role, text.strip()))

            # Add children to queue
            children = node.get("children", [])
            if isinstance(children, list):
                queue.extend(children)

        return messages
