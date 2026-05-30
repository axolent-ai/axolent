"""Memory consolidation: dedup and aging for memory entries.

Provides:
  * Exact dedup: entries with identical text + same user are merged
  * Aging: entries older than AGING_THRESHOLD_DAYS without re-reference
    are marked as low_relevance (not deleted)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Protocol

log = logging.getLogger(__name__)

# Entries older than this without re-reference get low_relevance tag
AGING_THRESHOLD_DAYS = 180


class MemoryStoragePort(Protocol):
    """Minimal interface for memory storage needed by consolidator."""

    def list_entries(self, user_id: int, layer: str, limit: int = 50) -> list[dict]: ...

    def delete_by_id(self, entry_id: str, layer: str, user_id: int) -> bool: ...

    def update_metadata(
        self, entry_id: str, layer: str, user_id: int, metadata: dict
    ) -> bool: ...


class MemoryConsolidator:
    """Memory consolidation: episodic dedup + aging.

    Phase 1: exact dedup (identical text + same user) and age-based
    low_relevance marking. Phase 1+: embedding-based near-dup detection.
    """

    def __init__(self, storage: MemoryStoragePort | None = None) -> None:
        self._storage = storage

    def consolidate(
        self,
        user_id: int | None = None,
        since_iso: str | None = None,
        max_entries: int = 100,
    ) -> int:
        """Run one consolidation round (dedup + aging).

        Args:
            user_id: If set, consolidate only this user's memory.
                     None = no-op (multi-user not yet supported).
            since_iso: If set, only consider entries from this timestamp onward.
            max_entries: Upper bound for entries consolidated per run.

        Returns:
            Number of consolidated (removed/marked) entries.
        """
        if self._storage is None or user_id is None:
            return 0

        consolidated = 0
        for layer in ("episodic", "semantic", "procedural"):
            entries = self._storage.list_entries(user_id, layer, limit=max_entries)
            consolidated += self._dedup_entries(entries, layer, user_id)
            consolidated += self._age_entries(entries, layer, user_id)

        if consolidated > 0:
            log.info(
                "Consolidation for user=%d: %d entries consolidated",
                user_id,
                consolidated,
            )
        return consolidated

    def _dedup_entries(self, entries: list[dict], layer: str, user_id: int) -> int:
        """Remove exact duplicate entries (same text, same user).

        Keeps the newest entry (by timestamp), deletes older duplicates.

        Returns:
            Number of deleted duplicates.
        """
        if not entries or self._storage is None:
            return 0

        # Group by normalized content
        seen: dict[str, dict] = {}
        to_delete: list[str] = []

        for entry in entries:
            content = (entry.get("content") or "").strip().lower()
            if not content:
                continue
            entry_id = entry.get("id", "")
            if content in seen:
                # Keep the newer one (by timestamp), delete the older
                existing = seen[content]
                existing_ts = existing.get("timestamp", "")
                entry_ts = entry.get("timestamp", "")
                if entry_ts > existing_ts:
                    # New entry is newer: delete the old one
                    to_delete.append(existing.get("id", ""))
                    seen[content] = entry
                else:
                    # Old entry is newer: delete this one
                    to_delete.append(entry_id)
            else:
                seen[content] = entry

        deleted = 0
        for eid in to_delete:
            if self._storage.delete_by_id(eid, layer, user_id):
                deleted += 1

        if deleted > 0:
            log.debug(
                "Dedup: removed %d duplicates in layer=%s user=%d",
                deleted,
                layer,
                user_id,
            )
        return deleted

    def _age_entries(self, entries: list[dict], layer: str, user_id: int) -> int:
        """Mark old entries as low_relevance.

        Entries older than AGING_THRESHOLD_DAYS that are not already
        marked get a low_relevance flag in their metadata.

        Returns:
            Number of entries marked.
        """
        if not entries or self._storage is None:
            return 0

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=AGING_THRESHOLD_DAYS)
        ).isoformat()

        marked = 0
        for entry in entries:
            ts = entry.get("timestamp", "")
            if not ts or ts >= cutoff:
                continue

            # Check if already marked
            metadata = entry.get("metadata") or {}
            if isinstance(metadata, str):
                import json

                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            if metadata.get("low_relevance"):
                continue

            metadata["low_relevance"] = True
            entry_id = entry.get("id", "")
            if entry_id and self._storage.update_metadata(
                entry_id, layer, user_id, metadata
            ):
                marked += 1

        if marked > 0:
            log.debug(
                "Aging: marked %d entries as low_relevance in layer=%s user=%d",
                marked,
                layer,
                user_id,
            )
        return marked
