"""Memory storage: JSONL adapter with FileLock.

Three files:
  data/memory_episodic.jsonl
  data/memory_semantic.jsonl
  data/memory_procedural.jsonl

Thread-safe via filelock. Reads/writes UTF-8.
Later Phase 1+: migration to SQLite with vector index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

from filelock import FileLock

from infrastructure.encoding import append_jsonl_utf8, open_utf8

log = logging.getLogger(__name__)

# Valid layer names
VALID_LAYERS: set[str] = {"episodic", "semantic", "procedural"}

# Search modes: "substring" (current), "embedding" (Phase 1+)
SearchMode = Literal["substring", "embedding"]


class MemoryStorage:
    """JSONL adapter for Trinity Memory persistence.

    Each layer gets its own JSONL file.
    All operations are atomically protected via FileLock.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize storage with the data directory.

        Args:
            data_dir: Path to the data/ folder (created if needed).
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.episodic_path = data_dir / "memory_episodic.jsonl"
        self.semantic_path = data_dir / "memory_semantic.jsonl"
        self.procedural_path = data_dir / "memory_procedural.jsonl"

        self._lock = FileLock(str(data_dir / "memory.lock"))

    def _path_for_layer(self, layer: str) -> Path:
        """Return the file path for a layer.

        Args:
            layer: "episodic", "semantic", or "procedural".

        Returns:
            Path to the JSONL file.

        Raises:
            ValueError: For unknown layer.
        """
        if layer not in VALID_LAYERS:
            raise ValueError(f"Unknown layer: '{layer}'. Allowed: {VALID_LAYERS}")
        path_map = {
            "episodic": self.episodic_path,
            "semantic": self.semantic_path,
            "procedural": self.procedural_path,
        }
        return path_map[layer]

    def append(self, entry: dict, layer: str) -> None:
        """Append an entry to the corresponding layer.

        Args:
            entry: Serialized entry dict.
            layer: Target layer.
        """
        path = self._path_for_layer(layer)
        with self._lock:
            append_jsonl_utf8(entry, path)
        log.debug("Memory entry appended: layer=%s id=%s", layer, entry.get("id"))

    def _read_filtered(self, path: Path, user_id: int, limit: int) -> list[dict]:
        """Read JSONL, filter by user_id, sort by timestamp descending.

        Args:
            path: Path to the JSONL file.
            user_id: Telegram user ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of entry dicts, newest first, limited.
        """
        if not path.exists():
            return []
        entries: list[dict] = []
        with open_utf8(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("user_id") == user_id:
                        entries.append(entry)
                except json.JSONDecodeError:
                    log.warning("Corrupt JSONL line in %s skipped", path)
                    continue
        # Newest first (by timestamp), then limit
        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return entries[:limit]

    def list_entries(self, user_id: int, layer: str, limit: int = 50) -> list[dict]:
        """Read entries for a user, newest first.

        Args:
            user_id: Telegram user ID.
            layer: Layer to read.
            limit: Maximum number of entries.

        Returns:
            List of entry dicts, newest first (sorted by timestamp).
        """
        path = self._path_for_layer(layer)
        with self._lock:
            return self._read_filtered(path, user_id, limit)

    def search(
        self,
        user_id: int,
        query: str,
        layer: str = "episodic",
        limit: int = 20,
        mode: SearchMode = "substring",
    ) -> list[dict]:
        """Search memory entries for a user.

        Args:
            user_id: Telegram user ID.
            query: Search term.
            layer: Layer to search.
            limit: Maximum hits.
            mode: "substring" (default, current logic) or "embedding"
                  (Phase 1+, not yet implemented).

        Returns:
            List of matching entry dicts, newest hits first.

        Raises:
            NotImplementedError: For mode="embedding" (Phase 1+).
        """
        if mode == "embedding":
            raise NotImplementedError(
                "Vector embedding search is Phase 1+. Currently only 'substring'."
            )

        path = self._path_for_layer(layer)
        if not path.exists():
            return []

        query_lower = query.lower()
        results: list[dict] = []

        with self._lock:
            with open_utf8(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("user_id") != user_id:
                            continue
                        if query_lower in entry.get("content", "").lower():
                            results.append(entry)
                    except json.JSONDecodeError:
                        continue

        # Newest hits first, then limit
        results.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return results[:limit]

    def delete_by_id(self, entry_id: str, layer: str, user_id: int) -> bool:
        """Delete an entry by its ID (atomic via read-filter-rewrite).

        Verifies ownership: entry must belong to the user.

        Args:
            entry_id: ID of the entry to delete.
            layer: Layer to search in.
            user_id: User ID for ownership check.

        Returns:
            True if entry was found and deleted, False if not found.
        """
        path = self._path_for_layer(layer)
        if not path.exists():
            return False

        found = False
        remaining: list[str] = []

        with self._lock:
            with open_utf8(path, "r") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    try:
                        entry = json.loads(line_stripped)
                        if (
                            entry.get("id") == entry_id
                            and entry.get("user_id") == user_id
                        ):
                            found = True
                            continue  # Skip (delete)
                        remaining.append(line_stripped)
                    except json.JSONDecodeError:
                        remaining.append(line_stripped)

            if found:
                tmp_path = path.with_suffix(".jsonl.tmp")
                with open_utf8(tmp_path, "w") as f:
                    for line_content in remaining:
                        f.write(line_content + "\n")
                tmp_path.replace(path)
                log.info("Memory entry deleted: id=%s layer=%s", entry_id, layer)

        return found

    def get_by_id(self, entry_id: str, layer: str, user_id: int) -> Optional[dict]:
        """Read a single entry by its ID.

        Args:
            entry_id: Requested entry ID.
            layer: Layer to search in.
            user_id: User ID for ownership check.

        Returns:
            Entry dict or None if not found.
        """
        path = self._path_for_layer(layer)
        if not path.exists():
            return None

        with self._lock:
            with open_utf8(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if (
                            entry.get("id") == entry_id
                            and entry.get("user_id") == user_id
                        ):
                            return entry
                    except json.JSONDecodeError:
                        continue
        return None
