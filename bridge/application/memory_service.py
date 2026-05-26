"""Memory service: CRUD coordinator for Trinity Memory.

Coordinates between domain (entry classes) and infrastructure (MemoryStorage).
User-facing API for Telegram handlers.

Phase 1: Manual save/recall via commands.
Phase 1+: Auto-memory loading in chat service, consolidation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from application.security.secret_scanner import SecretBlockedError, SecretScanner
from domain.memory.episodic import EpisodicEntry
from domain.memory.procedural import ProceduralEntry
from domain.memory.semantic import SemanticEntry
from infrastructure.memory_storage import MemoryStorage

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteMemoryStorage

log = logging.getLogger(__name__)

# Module-level scanner instance (stateless, safe to share).
_secret_scanner = SecretScanner()


class MemoryService:
    """Coordinator for all memory operations.

    Provides a clean API used by Telegram handlers.
    Handles entry creation, validation, and storage delegation.
    """

    def __init__(self, storage: Union[MemoryStorage, SqliteMemoryStorage]) -> None:
        """Initialize the service with a storage adapter.

        Args:
            storage: MemoryStorage (JSONL) or SqliteMemoryStorage (SQLite).
        """
        self.storage = storage

    @staticmethod
    def _scan_and_raise(content: str, user_id: int, layer: str) -> None:
        """Defense-in-depth gate: scan content for secrets before storage.

        Shared by all three remember_* methods so that every memory
        write path is protected, not just remember_episodic.

        Args:
            content: Text to scan.
            user_id: Telegram user ID (for logging).
            layer: Memory layer name (for logging).

        Raises:
            SecretBlockedError: If secrets are detected.
        """
        matches = _secret_scanner.scan(content)
        if matches:
            log.warning(
                "remember_%s blocked: user=%d pattern=%s layer=%d",
                layer,
                user_id,
                matches[0].pattern_name,
                matches[0].layer,
            )
            raise SecretBlockedError(matches)

    def remember_episodic(
        self,
        user_id: int,
        content: str,
        importance: int = 5,
        context: Optional[dict] = None,
    ) -> str:
        """Store an episodic event.

        Defense-in-depth: scans content for secrets/PII before storage.
        Raises SecretBlockedError if secrets are detected, ensuring that
        ALL callers (handler, auto-promotion, admin CLI, tests) are
        protected, not just the Telegram /remember handler.

        Args:
            user_id: Telegram user ID.
            content: Description of the event.
            importance: Importance 1-10.
            context: Optional context.

        Returns:
            ID of the new entry (ep_...).

        Raises:
            SecretBlockedError: If content contains detected secrets.
        """
        # BL-3: Defense-in-depth gate. Every caller goes through this.
        self._scan_and_raise(content, user_id, "episodic")

        entry = EpisodicEntry(
            user_id=user_id,
            content=content,
            importance=importance,
            context=context or {},
        )
        self.storage.append(entry.to_dict(), "episodic")
        log.info("Episodic memory saved: user=%d id=%s", user_id, entry.id)
        return entry.id

    def remember_semantic(
        self,
        user_id: int,
        content: str,
        category: str = "fakt",
        importance: int = 5,
        context: Optional[dict] = None,
    ) -> str:
        """Store a semantic fact.

        Defense-in-depth: scans content for secrets/PII before storage.
        Raises SecretBlockedError if secrets are detected.

        Args:
            user_id: Telegram user ID.
            content: The generalized fact.
            category: Classification (fakt, person, praeferenz, projekt).  # noqa: fake-umlaut
                      Note: values are ASCII keys (DB schema, not localized).
            importance: Importance 1-10.
            context: Optional context.

        Returns:
            ID of the new entry (sem_...).

        Raises:
            SecretBlockedError: If content contains detected secrets.
        """
        self._scan_and_raise(content, user_id, "semantic")
        entry = SemanticEntry(
            user_id=user_id,
            content=content,
            category=category,
            importance=importance,
            context=context or {},
        )
        self.storage.append(entry.to_dict(), "semantic")
        log.info("Semantic memory saved: user=%d id=%s", user_id, entry.id)
        return entry.id

    def remember_procedural(
        self,
        user_id: int,
        content: str,
        skill_name: str,
        importance: int = 5,
        context: Optional[dict] = None,
    ) -> str:
        """Store a skill/pattern.

        Defense-in-depth: scans content for secrets/PII before storage.
        Raises SecretBlockedError if secrets are detected.

        Args:
            user_id: Telegram user ID.
            content: Description of the skill.
            skill_name: Short name (e.g. "code_format").
            importance: Importance 1-10.
            context: Optional context.

        Returns:
            ID of the new entry (pro_...).

        Raises:
            SecretBlockedError: If content contains detected secrets.
        """
        self._scan_and_raise(content, user_id, "procedural")
        entry = ProceduralEntry(
            user_id=user_id,
            content=content,
            skill_name=skill_name,
            importance=importance,
            context=context or {},
        )
        self.storage.append(entry.to_dict(), "procedural")
        log.info("Procedural memory saved: user=%d id=%s", user_id, entry.id)
        return entry.id

    def recall(
        self,
        user_id: int,
        query: str,
        layer: str = "episodic",
        limit: int = 20,
    ) -> list[dict]:
        """Search memory for a term.

        Args:
            user_id: Telegram user ID.
            query: Search term (substring match).
            layer: Layer to search.
            limit: Maximum hits.

        Returns:
            List of matching entry dicts.
        """
        return self.storage.search(user_id, query, layer, limit)

    def list_recent(
        self, user_id: int, layer: str = "episodic", limit: int = 50
    ) -> list[dict]:
        """List the most recent entries of a layer.

        Args:
            user_id: Telegram user ID.
            layer: Layer to query.
            limit: Maximum count.

        Returns:
            List of entry dicts, newest first.
        """
        return self.storage.list_entries(user_id, layer, limit)

    def forget(self, user_id: int, entry_id: str) -> bool:
        """Delete a memory entry (with ownership check).

        Detects the layer automatically from the ID prefix.

        Args:
            user_id: Telegram user ID.
            entry_id: ID of the entry (ep_..., sem_..., pro_...).

        Returns:
            True if deleted, False if not found.
        """
        layer = self._layer_from_id(entry_id)
        if layer is None:
            log.warning("Unknown ID prefix for forget: %s", entry_id)
            return False
        return self.storage.delete_by_id(entry_id, layer, user_id)

    def get_entry(self, user_id: int, entry_id: str) -> Optional[dict]:
        """Read a single entry by its ID.

        Args:
            user_id: Telegram user ID.
            entry_id: Requested ID.

        Returns:
            Entry dict or None.
        """
        layer = self._layer_from_id(entry_id)
        if layer is None:
            return None
        return self.storage.get_by_id(entry_id, layer, user_id)

    @staticmethod
    def _layer_from_id(entry_id: str) -> Optional[str]:
        """Detect the layer from the ID prefix.

        Args:
            entry_id: Entry ID with prefix (ep_, sem_, pro_).

        Returns:
            Layer name or None for unknown prefix.
        """
        if entry_id.startswith("ep_"):
            return "episodic"
        elif entry_id.startswith("sem_"):
            return "semantic"
        elif entry_id.startswith("pro_"):
            return "procedural"
        return None
