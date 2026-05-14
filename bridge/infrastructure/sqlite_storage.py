"""SQLite storage adapter for bookmarks and memory.

Replaces JSONL backends with SQLite for:
  * O(log N) lookups instead of O(N) full scans
  * Cross-user isolation via WHERE clause
  * FTS5 for semantic substring search
  * Preparation for sqlite-vec (Phase 1+)

Uses sqlite3 (standard library), WAL mode for concurrency.
Thread-safe via check_same_thread=False + threading.Lock.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

log = logging.getLogger(__name__)

# Search modes (compatible with JSONL MemoryStorage)
SearchMode = Literal["substring", "embedding"]

# Default DB path
DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "axolent.db"

_SCHEMA_SQL = """
-- Bookmarks
CREATE TABLE IF NOT EXISTS bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    username TEXT,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bookmarks_user_chat
    ON bookmarks(user_id, chat_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bookmarks_user_chat_msg
    ON bookmarks(user_id, chat_id, message_id);

-- Memory entries (all three layers in one table, type column differentiates)
CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER,
    timestamp TEXT NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_user_type_time
    ON memory_entries(user_id, type, timestamp DESC);

-- User profiles for rate limiting (persistent across bot restart)
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    profile TEXT NOT NULL DEFAULT 'normal',
    set_at TEXT NOT NULL,
    PRIMARY KEY (user_id)
);

-- User model overrides (Phase 1: slot='global', Phase 2+: 'chat', 'code', etc.)
CREATE TABLE IF NOT EXISTS user_slot_models (
    user_id INTEGER NOT NULL,
    slot TEXT NOT NULL,
    model_id TEXT NOT NULL,
    set_at TEXT NOT NULL,
    PRIMARY KEY (user_id, slot)
);

-- FTS5 for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memory_entries',
    content_rowid='rowid'
);
"""

# FTS triggers are created separately (VIRTUAL TABLE triggers
# need the table first)
_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory_entries BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory_entries BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory_entries BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

# Valid layer names (compatible with JSONL MemoryStorage)
VALID_LAYERS: set[str] = {"episodic", "semantic", "procedural"}


class SqliteConnection:
    """Thread-safe SQLite connection manager with WAL mode.

    Uses a persistent connection with explicit lock instead of
    connection-per-request. Reason: WAL mode benefits from a
    long-lived connection, and the bot is single-process.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_connection(self) -> sqlite3.Connection:
        """Create the connection lazily on first access."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit, we control transactions manually
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialize schema + FTS triggers (idempotent)."""
        conn = self._conn
        if conn is None:  # pragma: no cover
            raise RuntimeError("_init_schema called before connection established")
        conn.executescript(_SCHEMA_SQL)
        conn.executescript(_TRIGGER_SQL)
        log.debug("SQLite schema initialized: %s", self._db_path)

    def execute(
        self,
        sql: str,
        params: tuple | dict = (),
        *,
        many: bool = False,
        data: list[tuple] | None = None,
    ) -> sqlite3.Cursor:
        """Thread-safe SQL execution with lock.

        Args:
            sql: SQL statement with ? placeholders.
            params: Parameter tuple or dict.
            many: If True, executemany with data.
            data: Data list for executemany.

        Returns:
            sqlite3.Cursor with result.
        """
        with self._lock:
            conn = self._ensure_connection()
            if many and data is not None:
                return conn.executemany(sql, data)
            return conn.execute(sql, params)

    def execute_in_transaction(self, operations: list[tuple[str, tuple]]) -> None:
        """Execute multiple statements in a transaction.

        Args:
            operations: List of (sql, params) tuples.
        """
        with self._lock:
            conn = self._ensure_connection()
            conn.execute("BEGIN")
            try:
                for sql, params in operations:
                    conn.execute(sql, params)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def fetchall(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        """Thread-safe query with fetchall.

        Args:
            sql: SELECT statement.
            params: Parameters.

        Returns:
            List of sqlite3.Row objects.
        """
        with self._lock:
            conn = self._ensure_connection()
            return conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple | dict = ()) -> Optional[sqlite3.Row]:
        """Thread-safe query with fetchone.

        Args:
            sql: SELECT statement.
            params: Parameters.

        Returns:
            sqlite3.Row or None.
        """
        with self._lock:
            conn = self._ensure_connection()
            return conn.execute(sql, params).fetchone()

    def close(self) -> None:
        """Close the connection (for graceful shutdown)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ──────────────────────────────────────────────────────────────
# Bookmark Storage (SQLite)
# ──────────────────────────────────────────────────────────────


class SqliteBookmarkStorage:
    """SQLite adapter for bookmarks.

    Drop-in replacement for the JSONL-based bookmark functions.
    API is identical to infrastructure.bookmark_storage.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn

    def save_bookmark(
        self,
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
            The saved bookmark entry as a dict.
        """
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO bookmarks
               (user_id, username, chat_id, message_id, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, username, chat_id, message_id, content, ts),
        )
        entry = {
            "timestamp": ts,
            "user_id": user_id,
            "username": username,
            "message_id": message_id,
            "chat_id": chat_id,
            "content": content,
        }
        log.info(
            "Bookmark saved: user=%s chat_id=%d message_id=%d content_len=%d",
            username,
            chat_id,
            message_id,
            len(content),
        )
        return entry

    def list_recent_bookmarks(
        self, user_id: int, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return the most recent bookmarks for a user.

        Args:
            user_id: Telegram user ID.
            limit: Maximum number of bookmarks to return.

        Returns:
            List of bookmark dicts, newest first, at most `limit` entries.
        """
        rows = self._conn.fetchall(
            """SELECT user_id, username, chat_id, message_id, content,
                      created_at as timestamp
               FROM bookmarks
               WHERE user_id = ?
               ORDER BY created_at DESC, rowid DESC
               LIMIT ?""",
            (user_id, limit),
        )
        return [dict(r) for r in rows]

    def search_bookmarks(
        self, user_id: int, query: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search bookmarks by content substring (case-insensitive).

        Args:
            user_id: Telegram user ID.
            query: Search term for bookmark content.
            limit: Maximum number of results.

        Returns:
            List of matching bookmark dicts, newest first.
        """
        # SQLite LIKE is case-insensitive by default for ASCII.
        # For Unicode correctness we use LOWER().
        rows = self._conn.fetchall(
            """SELECT user_id, username, chat_id, message_id, content,
                      created_at as timestamp
               FROM bookmarks
               WHERE user_id = ? AND LOWER(content) LIKE LOWER(?)
               ORDER BY created_at DESC, rowid DESC
               LIMIT ?""",
            (user_id, f"%{query}%", limit),
        )
        return [dict(r) for r in rows]

    def get_bookmark_by_message_id(
        self, user_id: int, chat_id: int, message_id: int
    ) -> Optional[dict[str, Any]]:
        """Find a bookmark by chat_id + message_id.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            message_id: Telegram message ID to search for.

        Returns:
            Bookmark dict or None if not found.
        """
        row = self._conn.fetchone(
            """SELECT user_id, username, chat_id, message_id, content,
                      created_at as timestamp
               FROM bookmarks
               WHERE user_id = ? AND chat_id = ? AND message_id = ?""",
            (user_id, chat_id, message_id),
        )
        return dict(row) if row else None

    def bookmark_exists(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Check whether a bookmark exists.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            message_id: Telegram message ID to check.

        Returns:
            True if the bookmark exists, False otherwise.
        """
        row = self._conn.fetchone(
            """SELECT 1 FROM bookmarks
               WHERE user_id = ? AND chat_id = ? AND message_id = ?""",
            (user_id, chat_id, message_id),
        )
        return row is not None

    def delete_bookmark(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Delete a bookmark by chat_id + message_id.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            message_id: Telegram message ID to remove.

        Returns:
            True if a bookmark was deleted, False if not found.
        """
        cursor = self._conn.execute(
            """DELETE FROM bookmarks
               WHERE user_id = ? AND chat_id = ? AND message_id = ?""",
            (user_id, chat_id, message_id),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            log.info(
                "Bookmark deleted: user_id=%d chat_id=%d message_id=%d",
                user_id,
                chat_id,
                message_id,
            )
        return deleted


# ──────────────────────────────────────────────────────────────
# Memory Storage (SQLite)
# ──────────────────────────────────────────────────────────────


class SqliteMemoryStorage:
    """SQLite adapter for Trinity Memory persistence.

    Drop-in replacement for MemoryStorage (JSONL).
    All three layers (episodic, semantic, procedural) live in one
    table with type column. Layer-specific fields are stored in
    metadata_json.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn

    @staticmethod
    def _validate_layer(layer: str) -> None:
        """Validate that the layer name is known.

        Raises:
            ValueError: If the layer is unknown.
        """
        if layer not in VALID_LAYERS:
            raise ValueError(f"Unknown layer: '{layer}'. Allowed: {VALID_LAYERS}")

    @staticmethod
    def _entry_to_row(entry: dict, layer: str) -> tuple:
        """Convert an entry dict to a SQLite row tuple.

        Type-specific fields (context, category, skill_name, usage_count)
        are packed into metadata_json.

        Args:
            entry: Serialized entry dict.
            layer: Target layer.

        Returns:
            Tuple for INSERT.
        """
        # All fields not in the base schema go into metadata
        base_keys = {"id", "user_id", "content", "importance", "timestamp", "type"}
        metadata = {k: v for k, v in entry.items() if k not in base_keys}

        return (
            entry.get("id", ""),
            entry.get("user_id", 0),
            layer,
            entry.get("content", ""),
            entry.get("importance"),
            entry.get("timestamp", ""),
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
        )

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> dict:
        """Convert a SQLite row back into an entry dict.

        Merges metadata_json back into the main dict.

        Args:
            row: sqlite3.Row object.

        Returns:
            Entry dict (compatible with JSONL format).
        """
        entry: dict[str, Any] = {
            "id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "timestamp": row["timestamp"],
        }
        if row["importance"] is not None:
            entry["importance"] = row["importance"]

        metadata_raw = row["metadata_json"]
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
                entry.update(metadata)
            except json.JSONDecodeError:
                log.warning("Corrupt metadata_json for entry %s", row["id"])

        return entry

    def append(self, entry: dict, layer: str) -> None:
        """Append an entry to the specified layer.

        Args:
            entry: Serialized entry dict.
            layer: Target layer.
        """
        self._validate_layer(layer)
        row_data = self._entry_to_row(entry, layer)
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_entries
               (id, user_id, type, content, importance, timestamp, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            row_data,
        )
        log.debug("Memory entry appended: layer=%s id=%s", layer, entry.get("id"))

    def list_entries(self, user_id: int, layer: str, limit: int = 50) -> list[dict]:
        """Read entries for a user, newest first.

        Args:
            user_id: Telegram user ID.
            layer: Layer to read from.
            limit: Maximum number of entries.

        Returns:
            List of entry dicts, newest first (sorted by timestamp).
        """
        self._validate_layer(layer)
        rows = self._conn.fetchall(
            """SELECT id, user_id, type, content, importance, timestamp, metadata_json
               FROM memory_entries
               WHERE user_id = ? AND type = ?
               ORDER BY timestamp DESC, rowid DESC
               LIMIT ?""",
            (user_id, layer, limit),
        )
        return [self._row_to_entry(r) for r in rows]

    def search(
        self,
        user_id: int,
        query: str,
        layer: str = "episodic",
        limit: int = 20,
        mode: SearchMode = "substring",
    ) -> list[dict]:
        """Search memory entries for a user.

        Supports two modes:
          - "substring": SQLite LIKE (default, compatible with JSONL)
          - "embedding": Phase 1+, not yet implemented

        When FTS5 index is available AND mode="substring", FTS5 is used
        for better performance.

        Args:
            user_id: Telegram user ID.
            query: Search term.
            layer: Layer to search.
            limit: Maximum number of hits.
            mode: "substring" or "embedding".

        Returns:
            List of matching entry dicts, newest hits first.

        Raises:
            NotImplementedError: For mode="embedding".
        """
        if mode == "embedding":
            raise NotImplementedError(
                "Vector embedding search is Phase 1+. Currently only 'substring'."
            )

        self._validate_layer(layer)

        # Try FTS5 first (faster for large datasets)
        try:
            # Remove quotes from query (avoid FTS5 syntax errors)
            fts_query = query.replace('"', "")
            if fts_query.strip():
                rows = self._conn.fetchall(
                    """SELECT me.id, me.user_id, me.type, me.content,
                              me.importance, me.timestamp, me.metadata_json
                       FROM memory_entries me
                       JOIN memory_fts fts ON me.rowid = fts.rowid
                       WHERE me.user_id = ? AND me.type = ?
                         AND memory_fts MATCH ?
                       ORDER BY me.timestamp DESC, me.rowid DESC
                       LIMIT ?""",
                    (user_id, layer, f'"{fts_query}"', limit),
                )
                if rows:
                    return [self._row_to_entry(r) for r in rows]
                # FTS5 returned 0 hits: fall back to LIKE
                # (FTS5 tokenizes, does not find token interiors like "Super" in "Superword")
                log.debug("FTS5: 0 hits for '%s', falling back to LIKE", query)
        except sqlite3.OperationalError:
            # FTS table broken or missing: LIKE fallback
            log.debug("FTS5 search failed, falling back to LIKE")

        rows = self._conn.fetchall(
            """SELECT id, user_id, type, content, importance,
                      timestamp, metadata_json
               FROM memory_entries
               WHERE user_id = ? AND type = ?
                 AND LOWER(content) LIKE LOWER(?)
               ORDER BY timestamp DESC, rowid DESC
               LIMIT ?""",
            (user_id, layer, f"%{query}%", limit),
        )
        return [self._row_to_entry(r) for r in rows]

    def delete_by_id(self, entry_id: str, layer: str, user_id: int) -> bool:
        """Delete an entry by its ID.

        Verifies ownership: the entry must belong to the user.

        Args:
            entry_id: ID of the entry to delete.
            layer: Layer to search in.
            user_id: User ID for ownership check.

        Returns:
            True if the entry was found and deleted, False if not found.
        """
        self._validate_layer(layer)
        cursor = self._conn.execute(
            """DELETE FROM memory_entries
               WHERE id = ? AND type = ? AND user_id = ?""",
            (entry_id, layer, user_id),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            log.info("Memory entry deleted: id=%s layer=%s", entry_id, layer)
        return deleted

    def get_by_id(self, entry_id: str, layer: str, user_id: int) -> Optional[dict]:
        """Read a single entry by its ID.

        Args:
            entry_id: Entry ID to look up.
            layer: Layer to search in.
            user_id: User ID for ownership check.

        Returns:
            Entry dict or None if not found.
        """
        self._validate_layer(layer)
        row = self._conn.fetchone(
            """SELECT id, user_id, type, content, importance,
                      timestamp, metadata_json
               FROM memory_entries
               WHERE id = ? AND type = ? AND user_id = ?""",
            (entry_id, layer, user_id),
        )
        return self._row_to_entry(row) if row else None


# ──────────────────────────────────────────────────────────────
# Profile Storage (SQLite)
# ──────────────────────────────────────────────────────────────


class SqliteProfileStorage:
    """SQLite adapter for rate limit profiles.

    Replaces JSONL-based profile persistence in rate_limiter.py.
    Stores the active profile per user_id.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn

    def load_all(self) -> dict[int, str]:
        """Load all user profiles.

        Returns:
            Dict: user_id -> profile_name.
        """
        rows = self._conn.fetchall("SELECT user_id, profile FROM user_profiles")
        return {int(row["user_id"]): row["profile"] for row in rows}

    def save(self, user_id: int, chat_id: int, profile: str) -> None:
        """Save or update a user profile.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            profile: Profile name (light, normal, power, unlimited).
        """
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO user_profiles
               (user_id, chat_id, profile, set_at)
               VALUES (?, ?, ?, ?)""",
            (user_id, chat_id, profile, ts),
        )
        log.debug("Profile saved: user_id=%d profile=%s", user_id, profile)


# ──────────────────────────────────────────────────────────────
# Model Storage (SQLite)
# ──────────────────────────────────────────────────────────────


class SqliteModelStorage:
    """SQLite adapter for user model overrides.

    Stores the chosen model per (user_id, slot).
    Phase 1: only slot='global'. Phase 2+: 'chat', 'code', etc.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn

    def get_model(self, user_id: int, slot: str = "global") -> Optional[str]:
        """Read the active model override for a user and slot.

        Args:
            user_id: Telegram user ID.
            slot: Slot name (default: 'global').

        Returns:
            Model ID as string or None if no override is set.
        """
        row = self._conn.fetchone(
            "SELECT model_id FROM user_slot_models WHERE user_id = ? AND slot = ?",
            (user_id, slot),
        )
        return row["model_id"] if row else None

    def set_model(self, user_id: int, model_id: str, slot: str = "global") -> None:
        """Set or update a model override.

        Args:
            user_id: Telegram user ID.
            model_id: Full model ID (e.g. 'claude-opus-4-7').
            slot: Slot name (default: 'global').
        """
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO user_slot_models
               (user_id, slot, model_id, set_at)
               VALUES (?, ?, ?, ?)""",
            (user_id, slot, model_id, ts),
        )
        log.debug(
            "Model override saved: user_id=%d slot=%s model=%s",
            user_id,
            slot,
            model_id,
        )

    def delete_model(self, user_id: int, slot: str = "global") -> bool:
        """Remove a model override (reset to default).

        Args:
            user_id: Telegram user ID.
            slot: Slot name (default: 'global').

        Returns:
            True if an override was deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM user_slot_models WHERE user_id = ? AND slot = ?",
            (user_id, slot),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            log.debug("Model override deleted: user_id=%d slot=%s", user_id, slot)
        return deleted

    def get_all_models(self, user_id: int) -> dict[str, str]:
        """Read all slot overrides for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            Dict of slot_name -> model_id for all set overrides.
        """
        rows = self._conn.fetchall(
            "SELECT slot, model_id FROM user_slot_models WHERE user_id = ?",
            (user_id,),
        )
        return {row["slot"]: row["model_id"] for row in rows}

    def delete_all_models(self, user_id: int) -> int:
        """Remove all model overrides for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            Number of deleted entries.
        """
        cursor = self._conn.execute(
            "DELETE FROM user_slot_models WHERE user_id = ?",
            (user_id,),
        )
        count = cursor.rowcount
        if count > 0:
            log.debug(
                "All model overrides deleted: user_id=%d count=%d",
                user_id,
                count,
            )
        return count

    def _reset_all_for_tests(self) -> None:
        """Delete all model overrides (test-only).

        Consistency pattern: analogous to conversation_storage._reset_all_for_tests.
        """
        self._conn.execute("DELETE FROM user_slot_models", ())


# ──────────────────────────────────────────────────────────────
# JSONL → SQLite Migration
# ──────────────────────────────────────────────────────────────


def migrate_jsonl_to_sqlite(
    conn: SqliteConnection,
    data_dir: Path,
) -> dict[str, int]:
    """Migrate existing JSONL data into SQLite (idempotent).

    Workflow:
      1. Schema is already initialized (via SqliteConnection)
      2. If bookmarks.jsonl exists AND bookmarks table is empty:
         import all lines
      3. Same for memory_*.jsonl
      4. Rename migrated JSONL files to .bak

    Args:
        conn: Initialized SqliteConnection.
        data_dir: Path to the data/ directory containing JSONL files.

    Returns:
        Dict with migration statistics:
          {"bookmarks": N, "memory_episodic": N, ...}
    """
    stats: dict[str, int] = {}

    # Bookmark-Migration
    bm_path = data_dir / "bookmarks.jsonl"
    if bm_path.exists():
        row = conn.fetchone("SELECT COUNT(*) as cnt FROM bookmarks")
        if row and row["cnt"] == 0:
            count = _migrate_bookmarks_jsonl(conn, bm_path)
            stats["bookmarks"] = count
            if count > 0:
                bak_path = bm_path.with_suffix(".jsonl.bak")
                bm_path.rename(bak_path)
                log.info(
                    "Bookmark migration: %d entries migrated, %s -> %s",
                    count,
                    bm_path.name,
                    bak_path.name,
                )
        else:
            log.debug("Bookmark migration skipped: table not empty")

    # Memory-Migration (alle drei Layer)
    layer_files = {
        "episodic": data_dir / "memory_episodic.jsonl",
        "semantic": data_dir / "memory_semantic.jsonl",
        "procedural": data_dir / "memory_procedural.jsonl",
    }

    for layer, jsonl_path in layer_files.items():
        if jsonl_path.exists():
            row = conn.fetchone(
                "SELECT COUNT(*) as cnt FROM memory_entries WHERE type = ?",
                (layer,),
            )
            if row and row["cnt"] == 0:
                count = _migrate_memory_jsonl(conn, jsonl_path, layer)
                stats[f"memory_{layer}"] = count
                if count > 0:
                    bak_path = jsonl_path.with_suffix(".jsonl.bak")
                    jsonl_path.rename(bak_path)
                    log.info(
                        "Memory migration (%s): %d entries migrated, %s -> %s",
                        layer,
                        count,
                        jsonl_path.name,
                        bak_path.name,
                    )
            else:
                log.debug(
                    "Memory migration (%s) skipped: table not empty",
                    layer,
                )

    return stats


def _migrate_bookmarks_jsonl(conn: SqliteConnection, path: Path) -> int:
    """Read bookmarks.jsonl and write all entries into SQLite.

    Args:
        conn: SQLite connection.
        path: Path to bookmarks.jsonl.

    Returns:
        Number of migrated entries.
    """
    entries: list[tuple] = []
    corrupt = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(
                    (
                        data.get("user_id", 0),
                        data.get("username"),
                        data.get("chat_id", data.get("user_id", 0)),
                        data.get("message_id", 0),
                        data.get("content", ""),
                        data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    )
                )
            except json.JSONDecodeError as e:
                corrupt += 1
                log.warning(
                    "Migration: corrupt bookmark line %d skipped: %s",
                    line_num,
                    e,
                )

    if corrupt > 0:
        log.info("Migration: %d corrupt bookmark lines skipped", corrupt)

    if entries:
        conn.execute_in_transaction(
            [
                (
                    """INSERT OR IGNORE INTO bookmarks
                       (user_id, username, chat_id, message_id, content, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    row,
                )
                for row in entries
            ]
        )

    return len(entries)


def _migrate_memory_jsonl(conn: SqliteConnection, path: Path, layer: str) -> int:
    """Read a memory_*.jsonl file and write all entries into SQLite.

    Args:
        conn: SQLite connection.
        path: Path to the JSONL file.
        layer: Memory layer (episodic/semantic/procedural).

    Returns:
        Number of migrated entries.
    """
    entries: list[tuple] = []
    corrupt = 0
    base_keys = {"id", "user_id", "content", "importance", "timestamp", "type"}

    with open(path, encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                metadata = {k: v for k, v in data.items() if k not in base_keys}
                entries.append(
                    (
                        data.get("id", ""),
                        data.get("user_id", 0),
                        layer,
                        data.get("content", ""),
                        data.get("importance"),
                        data.get("timestamp", ""),
                        json.dumps(metadata, ensure_ascii=False) if metadata else None,
                    )
                )
            except json.JSONDecodeError as e:
                corrupt += 1
                log.warning(
                    "Migration: corrupt memory line %d in %s skipped: %s",
                    line_num,
                    path.name,
                    e,
                )

    if corrupt > 0:
        log.info(
            "Migration: %d corrupt lines in %s skipped",
            corrupt,
            path.name,
        )

    if entries:
        conn.execute_in_transaction(
            [
                (
                    """INSERT OR IGNORE INTO memory_entries
                       (id, user_id, type, content, importance, timestamp, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
                for row in entries
            ]
        )

    return len(entries)
