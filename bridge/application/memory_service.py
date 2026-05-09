"""Memory-Service: CRUD-Coordinator für Trinity-Memory.

Koordiniert zwischen Domain (Entry-Klassen) und Infrastructure (MemoryStorage).
User-facing API für Telegram-Handler.

Phase 1: Manuelles Speichern/Abrufen via Commands.
Phase 1+: Auto-Memory-Loading in Chat-Service, Konsolidierung.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from domain.memory.episodic import EpisodicEntry
from domain.memory.procedural import ProceduralEntry
from domain.memory.semantic import SemanticEntry
from infrastructure.memory_storage import MemoryStorage

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteMemoryStorage

log = logging.getLogger(__name__)


class MemoryService:
    """Koordinator für alle Memory-Operationen.

    Bietet eine saubere API die von Telegram-Handlern genutzt wird.
    Kümmert sich um Entry-Erstellung, Validierung und Storage-Delegation.
    """

    def __init__(self, storage: Union[MemoryStorage, SqliteMemoryStorage]) -> None:
        """Initialisiert den Service mit einem Storage-Adapter.

        Args:
            storage: MemoryStorage (JSONL) oder SqliteMemoryStorage (SQLite).
        """
        self.storage = storage

    def remember_episodic(
        self,
        user_id: int,
        content: str,
        importance: int = 5,
        context: Optional[dict] = None,
    ) -> str:
        """Speichert ein episodisches Event.

        Args:
            user_id: Telegram-User-ID.
            content: Beschreibung des Events.
            importance: Wichtigkeit 1-10.
            context: Optionaler Kontext.

        Returns:
            ID des neuen Entries (ep_...).
        """
        entry = EpisodicEntry(
            user_id=user_id,
            content=content,
            importance=importance,
            context=context or {},
        )
        self.storage.append(entry.to_dict(), "episodic")
        log.info("Episodic Memory gespeichert: user=%d id=%s", user_id, entry.id)
        return entry.id

    def remember_semantic(
        self,
        user_id: int,
        content: str,
        category: str = "fakt",
        importance: int = 5,
        context: Optional[dict] = None,
    ) -> str:
        """Speichert einen semantischen Fakt.

        Args:
            user_id: Telegram-User-ID.
            content: Der generalisierte Fakt.
            category: Klassifizierung (fakt, person, praeferenz, projekt).
            importance: Wichtigkeit 1-10.
            context: Optionaler Kontext.

        Returns:
            ID des neuen Entries (sem_...).
        """
        entry = SemanticEntry(
            user_id=user_id,
            content=content,
            category=category,
            importance=importance,
            context=context or {},
        )
        self.storage.append(entry.to_dict(), "semantic")
        log.info("Semantic Memory gespeichert: user=%d id=%s", user_id, entry.id)
        return entry.id

    def remember_procedural(
        self,
        user_id: int,
        content: str,
        skill_name: str,
        importance: int = 5,
        context: Optional[dict] = None,
    ) -> str:
        """Speichert einen Skill/ein Pattern.

        Args:
            user_id: Telegram-User-ID.
            content: Beschreibung des Skills.
            skill_name: Kurzname (z.B. "code_format").
            importance: Wichtigkeit 1-10.
            context: Optionaler Kontext.

        Returns:
            ID des neuen Entries (pro_...).
        """
        entry = ProceduralEntry(
            user_id=user_id,
            content=content,
            skill_name=skill_name,
            importance=importance,
            context=context or {},
        )
        self.storage.append(entry.to_dict(), "procedural")
        log.info("Procedural Memory gespeichert: user=%d id=%s", user_id, entry.id)
        return entry.id

    def recall(
        self,
        user_id: int,
        query: str,
        layer: str = "episodic",
        limit: int = 20,
    ) -> list[dict]:
        """Durchsucht Memory nach einem Begriff.

        Args:
            user_id: Telegram-User-ID.
            query: Suchbegriff (substring match).
            layer: Zu durchsuchender Layer.
            limit: Maximale Treffer.

        Returns:
            Liste von matching Entry-Dicts.
        """
        return self.storage.search(user_id, query, layer, limit)

    def list_recent(
        self, user_id: int, layer: str = "episodic", limit: int = 50
    ) -> list[dict]:
        """Listet die neuesten Entries eines Layers.

        Args:
            user_id: Telegram-User-ID.
            layer: Abzufragender Layer.
            limit: Maximale Anzahl.

        Returns:
            Liste von Entry-Dicts, neueste zuerst.
        """
        return self.storage.list_entries(user_id, layer, limit)

    def forget(self, user_id: int, entry_id: str) -> bool:
        """Löscht einen Memory-Entry (mit Ownership-Check).

        Erkennt den Layer automatisch anhand des ID-Prefix.

        Args:
            user_id: Telegram-User-ID.
            entry_id: ID des Entries (ep_..., sem_..., pro_...).

        Returns:
            True wenn gelöscht, False wenn nicht gefunden.
        """
        layer = self._layer_from_id(entry_id)
        if layer is None:
            log.warning("Unbekanntes ID-Prefix für forget: %s", entry_id)
            return False
        return self.storage.delete_by_id(entry_id, layer, user_id)

    def get_entry(self, user_id: int, entry_id: str) -> Optional[dict]:
        """Liest einen einzelnen Entry anhand seiner ID.

        Args:
            user_id: Telegram-User-ID.
            entry_id: Gesuchte ID.

        Returns:
            Entry-Dict oder None.
        """
        layer = self._layer_from_id(entry_id)
        if layer is None:
            return None
        return self.storage.get_by_id(entry_id, layer, user_id)

    @staticmethod
    def _layer_from_id(entry_id: str) -> Optional[str]:
        """Erkennt den Layer anhand des ID-Prefix.

        Args:
            entry_id: Entry-ID mit Prefix (ep_, sem_, pro_).

        Returns:
            Layer-Name oder None bei unbekanntem Prefix.
        """
        if entry_id.startswith("ep_"):
            return "episodic"
        elif entry_id.startswith("sem_"):
            return "semantic"
        elif entry_id.startswith("pro_"):
            return "procedural"
        return None
