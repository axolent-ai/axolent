"""Memory-Storage: JSONL-Adapter mit FileLock.

Drei Dateien:
  data/memory_episodic.jsonl
  data/memory_semantic.jsonl
  data/memory_procedural.jsonl

Thread-safe via filelock. Liest/schreibt UTF-8.
Später Phase 1+: Migration auf SQLite mit Vector-Index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from filelock import FileLock

from infrastructure.encoding import append_jsonl_utf8, open_utf8

log = logging.getLogger(__name__)

# Valide Layer-Namen
VALID_LAYERS: set[str] = {"episodic", "semantic", "procedural"}


class MemoryStorage:
    """JSONL-Adapter fuer Trinity-Memory-Persistierung.

    Jeder Layer bekommt eine eigene JSONL-Datei.
    Alle Operationen sind atomar via FileLock geschuetzt.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialisiert den Storage mit dem data-Verzeichnis.

        Args:
            data_dir: Pfad zum data/-Ordner (wird bei Bedarf erstellt).
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.episodic_path = data_dir / "memory_episodic.jsonl"
        self.semantic_path = data_dir / "memory_semantic.jsonl"
        self.procedural_path = data_dir / "memory_procedural.jsonl"

        self._lock = FileLock(str(data_dir / "memory.lock"))

    def _path_for_layer(self, layer: str) -> Path:
        """Gibt den Dateipfad fuer einen Layer zurueck.

        Args:
            layer: "episodic", "semantic" oder "procedural".

        Returns:
            Path zum JSONL-File.

        Raises:
            ValueError: Bei unbekanntem Layer.
        """
        if layer not in VALID_LAYERS:
            raise ValueError(f"Unbekannter Layer: '{layer}'. Erlaubt: {VALID_LAYERS}")
        path_map = {
            "episodic": self.episodic_path,
            "semantic": self.semantic_path,
            "procedural": self.procedural_path,
        }
        return path_map[layer]

    def append(self, entry: dict, layer: str) -> None:
        """Haengt einen Entry an den entsprechenden Layer an.

        Args:
            entry: Serialisiertes Entry-Dict.
            layer: Ziel-Layer.
        """
        path = self._path_for_layer(layer)
        with self._lock:
            append_jsonl_utf8(entry, path)
        log.debug("Memory-Entry angehaengt: layer=%s id=%s", layer, entry.get("id"))

    def list_entries(self, user_id: int, layer: str, limit: int = 50) -> list[dict]:
        """Liest Entries fuer einen User, neueste zuerst.

        Args:
            user_id: Telegram-User-ID.
            layer: Zu lesender Layer.
            limit: Maximale Anzahl Eintraege.

        Returns:
            Liste von Entry-Dicts, neueste zuerst (reversed).
        """
        path = self._path_for_layer(layer)
        if not path.exists():
            return []

        entries: list[dict] = []
        with self._lock:
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
                        log.warning("Korrupte JSONL-Zeile in %s uebersprungen", path)
                        continue

        # Neueste zuerst, limitiert
        entries.reverse()
        return entries[:limit]

    def search(
        self, user_id: int, query: str, layer: str = "episodic", limit: int = 20
    ) -> list[dict]:
        """Substring-Suche ueber Memory-Entries eines Users.

        Einfache Implementierung: case-insensitive substring match auf content.
        Phase 1+: Vector-Embeddings fuer semantische Suche.

        Args:
            user_id: Telegram-User-ID.
            query: Suchbegriff.
            layer: Zu durchsuchender Layer.
            limit: Maximale Treffer.

        Returns:
            Liste von matching Entry-Dicts.
        """
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
                            if len(results) >= limit:
                                break
                    except json.JSONDecodeError:
                        continue

        return results

    def delete_by_id(self, entry_id: str, layer: str, user_id: int) -> bool:
        """Loescht einen Entry anhand seiner ID (atomar via Read-Filter-Rewrite).

        Verifiziert Ownership: Entry muss dem User gehoeren.

        Args:
            entry_id: ID des zu loeschenden Entries.
            layer: Layer in dem gesucht wird.
            user_id: User-ID fuer Ownership-Check.

        Returns:
            True wenn Entry gefunden und geloescht, False wenn nicht gefunden.
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
                with open_utf8(path, "w") as f:
                    for line_content in remaining:
                        f.write(line_content + "\n")
                log.info("Memory-Entry geloescht: id=%s layer=%s", entry_id, layer)

        return found

    def get_by_id(self, entry_id: str, layer: str, user_id: int) -> Optional[dict]:
        """Liest einen einzelnen Entry anhand seiner ID.

        Args:
            entry_id: Gesuchte Entry-ID.
            layer: Layer in dem gesucht wird.
            user_id: User-ID fuer Ownership-Check.

        Returns:
            Entry-Dict oder None wenn nicht gefunden.
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
