"""Bookmark-Storage: JSONL-Adapter mit FileLock.

Persistiert Bookmarks als append-only JSONL-Datei.
Thread-safe via filelock. Liest/schreibt UTF-8.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock

from infrastructure.encoding import append_jsonl_utf8, open_utf8

log = logging.getLogger(__name__)

BOOKMARKS_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "bookmarks.jsonl"
)
_BM_LOCK_PATH = str(BOOKMARKS_PATH) + ".lock"
_BM_LOCK = FileLock(_BM_LOCK_PATH)


def save_bookmark(
    user_id: int,
    username: Optional[str],
    message_id: int,
    chat_id: int,
    content: str,
) -> dict[str, Any]:
    """Speichert einen Bookmark-Eintrag in bookmarks.jsonl.

    Args:
        user_id: Telegram User-ID.
        username: Telegram Username (kann None sein).
        message_id: Telegram Message-ID der Bot-Antwort.
        chat_id: Telegram Chat-ID.
        content: Volltext der Bot-Antwort.

    Returns:
        Der gespeicherte Bookmark-Eintrag als Dict.
    """
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
    """Liest alle Bookmarks eines bestimmten Users (stream-basiert).

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
    all_bm = _read_all_bookmarks(user_id)
    for bm in all_bm:
        # Backward compat: alte Bookmarks haben evtl. keine chat_id
        bm_chat_id = bm.get("chat_id")
        if bm.get("message_id") == message_id:
            if bm_chat_id is None or bm_chat_id == chat_id:
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
    return get_bookmark_by_message_id(user_id, chat_id, message_id) is not None


def delete_bookmark(user_id: int, chat_id: int, message_id: int) -> bool:
    """Löscht einen Bookmark per chat_id + message_id (schreibt JSONL ohne den Eintrag neu).

    Args:
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID zum Entfernen.

    Returns:
        True wenn ein Bookmark gelöscht wurde, False falls nicht gefunden.
    """
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
                bm_chat_id = entry.get("chat_id")
                if (
                    entry.get("user_id") == user_id
                    and entry.get("message_id") == message_id
                    and (bm_chat_id is None or bm_chat_id == chat_id)
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
            with open_utf8(BOOKMARKS_PATH, "w") as f:
                for kept_line in lines_to_keep:
                    f.write(kept_line + "\n")

    return found
