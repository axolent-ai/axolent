"""Bookmark storage: JSONL backend (legacy) with FileLock.

Persists bookmarks as an append-only JSONL file.
Thread-safe via filelock.

Since V6, the SQLite backend is used exclusively via BookmarkService
(constructor injection). This module remains as JSONL legacy backend
and for the JsonlBookmarkStorageAdapter class.
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


def migrate_legacy_chat_id() -> int:
    """Write missing chat_id into old bookmarks (idempotent, crash-safe).

    Assumption: for DMs in Telegram, chat_id == user_id.
    Corrupt JSONL lines are skipped and logged instead of crashing startup.

    Returns:
        Number of migrated entries.
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
                        "Migration: corrupt line %d skipped: %s",
                        line_num,
                        e,
                    )
        if migrated > 0 or corrupt_count > 0:
            # Atomic rewrite
            tmp_path = BOOKMARKS_PATH.with_suffix(".jsonl.tmp")
            with open_utf8(tmp_path, "w") as f:
                for entry in valid_lines:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            tmp_path.replace(BOOKMARKS_PATH)
        if corrupt_count > 0:
            log.info(
                "Migration: %d corrupt line(s) removed, %d entries migrated",
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
    """Save a bookmark entry.

    Args:
        user_id: Telegram user ID.
        username: Telegram username (can be None).
        message_id: Telegram message ID of the bot response.
        chat_id: Telegram chat ID.
        content: Full text of the bot response.

    Returns:
        The saved bookmark entry as dict.
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
        "Bookmark saved: user=%s chat_id=%d message_id=%d content_len=%d",
        username,
        chat_id,
        message_id,
        len(content),
    )
    return entry


def _read_all_bookmarks(user_id: int) -> list[dict[str, Any]]:
    """Read all bookmarks of a specific user (stream-based, JSONL only).

    Args:
        user_id: Telegram user ID to filter by.

    Returns:
        List of bookmark dicts for this user, newest first.
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
                    log.warning("Corrupt JSONL line skipped: %s", line[:80])
                    continue

    # Newest first
    bookmarks.reverse()
    return bookmarks


def list_recent_bookmarks(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent bookmarks of a user.

    Args:
        user_id: Telegram user ID.
        limit: Maximum number of bookmarks to return.

    Returns:
        List of bookmark dicts, newest first, max `limit` entries.
    """
    all_bm = _read_all_bookmarks(user_id)
    return all_bm[:limit]


def search_bookmarks(user_id: int, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search bookmarks by content substring (case-insensitive).

    Args:
        user_id: Telegram user ID.
        query: Search term for the bookmark content.
        limit: Maximum number of results.

    Returns:
        List of matching bookmark dicts, newest first.
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
    """Find a bookmark by chat_id + message_id.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        message_id: Telegram message ID to search for.

    Returns:
        Bookmark dict or None if not found.
    """
    all_bm = _read_all_bookmarks(user_id)
    for bm in all_bm:
        if bm.get("message_id") == message_id and bm.get("chat_id") == chat_id:
            return bm
    return None


def bookmark_exists(user_id: int, chat_id: int, message_id: int) -> bool:
    """Check if a bookmark with this chat_id + message_id exists for this user.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        message_id: Telegram message ID to check.

    Returns:
        True if the bookmark exists, False otherwise.
    """
    return get_bookmark_by_message_id(user_id, chat_id, message_id) is not None


def delete_bookmark(user_id: int, chat_id: int, message_id: int) -> bool:
    """Delete a bookmark by chat_id + message_id.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        message_id: Telegram message ID to remove.

    Returns:
        True if a bookmark was deleted, False if not found.
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
                if (
                    entry.get("user_id") == user_id
                    and entry.get("message_id") == message_id
                    and entry.get("chat_id") == chat_id
                ):
                    found = True
                    log.info(
                        "Bookmark deleted: user_id=%d chat_id=%d message_id=%d",
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
    """Adapter class providing JSONL module functions as BookmarkStorage protocol.

    Only used in JSONL legacy mode (USE_SQLITE_STORAGE=false).
    Delegates to module-level functions in this module.
    """

    def save_bookmark(
        self,
        user_id: int,
        username: Optional[str],
        message_id: int,
        chat_id: int,
        content: str,
    ) -> dict[str, Any]:
        """Delegate to module-level save_bookmark."""
        return save_bookmark(user_id, username, message_id, chat_id, content)

    def list_recent_bookmarks(
        self, user_id: int, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Delegate to module-level list_recent_bookmarks."""
        return list_recent_bookmarks(user_id, limit=limit)

    def search_bookmarks(
        self, user_id: int, query: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Delegate to module-level search_bookmarks."""
        return search_bookmarks(user_id, query, limit=limit)

    def get_bookmark_by_message_id(
        self, user_id: int, chat_id: int, message_id: int
    ) -> Optional[dict[str, Any]]:
        """Delegate to module-level get_bookmark_by_message_id."""
        return get_bookmark_by_message_id(user_id, chat_id, message_id)

    def bookmark_exists(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Delegate to module-level bookmark_exists."""
        return bookmark_exists(user_id, chat_id, message_id)

    def delete_bookmark(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Delegate to module-level delete_bookmark."""
        return delete_bookmark(user_id, chat_id, message_id)
