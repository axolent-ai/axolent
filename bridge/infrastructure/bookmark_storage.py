"""Bookmark-Storage: JSONL-Adapter mit FileLock + SQLite-Backend-Switch.

Persistiert Bookmarks als append-only JSONL-Datei (Legacy) oder via SQLite.
Thread-safe via filelock (JSONL) bzw. threading.Lock (SQLite).

Backend-Switch: Wenn use_sqlite_backend() aufgerufen wird, delegieren
alle Modul-Level-Funktionen an die SqliteBookmarkStorage-Instanz.
Application-Layer bleibt unverändert.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from filelock import FileLock

from infrastructure.encoding import append_jsonl_utf8, open_utf8

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteBookmarkStorage

log = logging.getLogger(__name__)

BOOKMARKS_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "bookmarks.jsonl"
)
_BM_LOCK_PATH = str(BOOKMARKS_PATH) + ".lock"
_BM_LOCK = FileLock(_BM_LOCK_PATH)

# Backend-Switch: None = JSONL (Legacy), sonst SQLite-Instanz
_sqlite_backend: Optional[SqliteBookmarkStorage] = None


def use_sqlite_backend(backend: SqliteBookmarkStorage) -> None:
    """Aktiviert SQLite als Bookmark-Backend.

    Nach diesem Aufruf delegieren alle Modul-Level-Funktionen
    an die übergebene SqliteBookmarkStorage-Instanz.

    Args:
        backend: Initialisierte SqliteBookmarkStorage-Instanz.
    """
    global _sqlite_backend  # noqa: PLW0603
    _sqlite_backend = backend
    log.info("Bookmark-Storage: SQLite-Backend aktiviert")


def migrate_legacy_chat_id() -> int:
    """Schreibt fehlende chat_id in alte Bookmarks (idempotent, crash-safe).

    Annahme: bei DMs in Telegram ist chat_id == user_id.
    Korrupte JSONL-Zeilen werden übersprungen und geloggt statt den Start zu crashen.

    Returns:
        Anzahl migrierter Einträge.
    """
    if not BOOKMARKS_PATH.exists():
        return 0
    migrated = 0
    valid_lines: list[dict] = []
    corrupt_count = 0
    with _BM_LOCK:
        with open_utf8(BOOKMARKS_PATH, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "chat_id" not in entry or entry.get("chat_id") is None:
                        entry["chat_id"] = entry.get("user_id", 0)
                        migrated += 1
                    valid_lines.append(entry)
                except json.JSONDecodeError as e:
                    corrupt_count += 1
                    log.warning(
                        "Migration: korrupte Zeile %d übersprungen: %s",
                        line_num,
                        e,
                    )
        if migrated > 0 or corrupt_count > 0:
            # Atomarer Rewrite
            tmp_path = BOOKMARKS_PATH.with_suffix(".jsonl.tmp")
            with open_utf8(tmp_path, "w") as f:
                for entry in valid_lines:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            tmp_path.replace(BOOKMARKS_PATH)
        if corrupt_count > 0:
            log.info(
                "Migration: %d korrupte Zeile(n) entfernt, %d Eintraege migriert",
                corrupt_count,
                migrated,
            )
    return migrated


def save_bookmark(
    user_id: int,
    username: Optional[str],
    message_id: int,
    chat_id: int,
    content: str,
) -> dict[str, Any]:
    """Speichert einen Bookmark-Eintrag.

    Delegiert an SQLite-Backend falls aktiviert, sonst JSONL.

    Args:
        user_id: Telegram User-ID.
        username: Telegram Username (kann None sein).
        message_id: Telegram Message-ID der Bot-Antwort.
        chat_id: Telegram Chat-ID.
        content: Volltext der Bot-Antwort.

    Returns:
        Der gespeicherte Bookmark-Eintrag als Dict.
    """
    if _sqlite_backend is not None:
        return _sqlite_backend.save_bookmark(
            user_id, username, message_id, chat_id, content
        )

    from datetime import datetime, timezone

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "username": username,
        "message_id": message_id,
        "chat_id": chat_id,
        "content": content,
    }
    with _BM_LOCK:
        append_jsonl_utf8(entry, BOOKMARKS_PATH)
    log.info(
        "Bookmark gespeichert: user=%s chat_id=%d message_id=%d content_len=%d",
        username,
        chat_id,
        message_id,
        len(content),
    )
    return entry


def _read_all_bookmarks(user_id: int) -> list[dict[str, Any]]:
    """Liest alle Bookmarks eines bestimmten Users (stream-basiert, JSONL-only).

    Args:
        user_id: Telegram User-ID zum Filtern.

    Returns:
        Liste von Bookmark-Dicts für diesen User, neueste zuerst.
    """
    if not BOOKMARKS_PATH.exists():
        return []

    bookmarks: list[dict[str, Any]] = []
    with _BM_LOCK:
        with open_utf8(BOOKMARKS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("user_id") == user_id:
                        bookmarks.append(entry)
                except json.JSONDecodeError:
                    log.warning("Korrupte JSONL-Zeile übersprungen: %s", line[:80])
                    continue

    # Neueste zuerst
    bookmarks.reverse()
    return bookmarks


def list_recent_bookmarks(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    """Gibt die neuesten Bookmarks eines Users zurück.

    Args:
        user_id: Telegram User-ID.
        limit: Maximale Anzahl zurückzugebender Bookmarks.

    Returns:
        Liste von Bookmark-Dicts, neueste zuerst, max `limit` Einträge.
    """
    if _sqlite_backend is not None:
        return _sqlite_backend.list_recent_bookmarks(user_id, limit)

    all_bm = _read_all_bookmarks(user_id)
    return all_bm[:limit]


def search_bookmarks(user_id: int, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Sucht Bookmarks per Inhalts-Substring (case-insensitive).

    Args:
        user_id: Telegram User-ID.
        query: Suchbegriff für den Bookmark-Inhalt.
        limit: Maximale Anzahl Ergebnisse.

    Returns:
        Liste passender Bookmark-Dicts, neueste zuerst.
    """
    if _sqlite_backend is not None:
        return _sqlite_backend.search_bookmarks(user_id, query, limit)

    query_lower = query.lower()
    all_bm = _read_all_bookmarks(user_id)
    results: list[dict[str, Any]] = []
    for bm in all_bm:
        if query_lower in bm.get("content", "").lower():
            results.append(bm)
            if len(results) >= limit:
                break
    return results


def get_bookmark_by_message_id(
    user_id: int, chat_id: int, message_id: int
) -> Optional[dict[str, Any]]:
    """Findet einen Bookmark per chat_id + message_id.

    Args:
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID zum Suchen.

    Returns:
        Bookmark-Dict oder None falls nicht gefunden.
    """
    if _sqlite_backend is not None:
        return _sqlite_backend.get_bookmark_by_message_id(user_id, chat_id, message_id)

    all_bm = _read_all_bookmarks(user_id)
    for bm in all_bm:
        if bm.get("message_id") == message_id and bm.get("chat_id") == chat_id:
            return bm
    return None


def bookmark_exists(user_id: int, chat_id: int, message_id: int) -> bool:
    """Prüft ob ein Bookmark mit dieser chat_id + message_id für diesen User existiert.

    Args:
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID zum Prüfen.

    Returns:
        True wenn der Bookmark existiert, False sonst.
    """
    if _sqlite_backend is not None:
        return _sqlite_backend.bookmark_exists(user_id, chat_id, message_id)

    return get_bookmark_by_message_id(user_id, chat_id, message_id) is not None


def delete_bookmark(user_id: int, chat_id: int, message_id: int) -> bool:
    """Löscht einen Bookmark per chat_id + message_id.

    Args:
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID zum Entfernen.

    Returns:
        True wenn ein Bookmark gelöscht wurde, False falls nicht gefunden.
    """
    if _sqlite_backend is not None:
        return _sqlite_backend.delete_bookmark(user_id, chat_id, message_id)

    if not BOOKMARKS_PATH.exists():
        return False

    lines_to_keep: list[str] = []
    found: bool = False

    with _BM_LOCK:
        with open_utf8(BOOKMARKS_PATH, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    lines_to_keep.append(stripped)
                    continue
                if (
                    entry.get("user_id") == user_id
                    and entry.get("message_id") == message_id
                    and entry.get("chat_id") == chat_id
                ):
                    found = True
                    log.info(
                        "Bookmark gelöscht: user_id=%d chat_id=%d message_id=%d",
                        user_id,
                        chat_id,
                        message_id,
                    )
                    continue
                lines_to_keep.append(stripped)

        if found:
            tmp_path = BOOKMARKS_PATH.with_suffix(".jsonl.tmp")
            with open_utf8(tmp_path, "w") as f:
                for kept_line in lines_to_keep:
                    f.write(kept_line + "\n")
            tmp_path.replace(BOOKMARKS_PATH)

    return found


class JsonlBookmarkStorageAdapter:
    """Adapter-Klasse die JSONL-Modul-Funktionen als BookmarkStorage-Protocol bereitstellt.

    Wird nur im JSONL-Legacy-Modus verwendet (USE_SQLITE_STORAGE=false).
    Delegiert an die Modul-Level-Funktionen in diesem Modul.
    """

    def save_bookmark(
        self,
        user_id: int,
        username: Optional[str],
        message_id: int,
        chat_id: int,
        content: str,
    ) -> dict[str, Any]:
        """Delegiert an Modul-Level save_bookmark."""
        return save_bookmark(user_id, username, message_id, chat_id, content)

    def list_recent_bookmarks(
        self, user_id: int, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Delegiert an Modul-Level list_recent_bookmarks."""
        return list_recent_bookmarks(user_id, limit=limit)

    def search_bookmarks(
        self, user_id: int, query: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Delegiert an Modul-Level search_bookmarks."""
        return search_bookmarks(user_id, query, limit=limit)

    def get_bookmark_by_message_id(
        self, user_id: int, chat_id: int, message_id: int
    ) -> Optional[dict[str, Any]]:
        """Delegiert an Modul-Level get_bookmark_by_message_id."""
        return get_bookmark_by_message_id(user_id, chat_id, message_id)

    def bookmark_exists(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Delegiert an Modul-Level bookmark_exists."""
        return bookmark_exists(user_id, chat_id, message_id)

    def delete_bookmark(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Delegiert an Modul-Level delete_bookmark."""
        return delete_bookmark(user_id, chat_id, message_id)
