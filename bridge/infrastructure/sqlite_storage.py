"""SQLite-Storage-Adapter für Bookmarks und Memory.

Ersetzt JSONL-Backends mit SQLite für:
  - O(log N) Lookups statt O(N) Vollscans
  - Cross-User-Isolation per WHERE-Clause
  - FTS5 für semantische Substring-Suche
  - Vorbereitung für sqlite-vec (Phase 1+)

Nutzt sqlite3 (Standard-Library), WAL-Mode für Concurrency.
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

# Such-Modi (kompatibel mit JSONL-MemoryStorage)
SearchMode = Literal["substring", "embedding"]

# Default DB-Pfad
DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "jarvis.db"

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

-- Memory entries (alle drei Layer in einer Tabelle, type-Spalte unterscheidet)
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

-- User-Profile für Rate-Limiting (persistent über Bot-Restart)
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    profile TEXT NOT NULL DEFAULT 'normal',
    set_at TEXT NOT NULL,
    PRIMARY KEY (user_id)
);

-- User-Modell-Overrides (Phase 1: slot='global', Phase 2+: 'chat', 'code', etc.)
CREATE TABLE IF NOT EXISTS user_slot_models (
    user_id INTEGER NOT NULL,
    slot TEXT NOT NULL,
    model_id TEXT NOT NULL,
    set_at TEXT NOT NULL,
    PRIMARY KEY (user_id, slot)
);

-- FTS5 für Volltext-Suche
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memory_entries',
    content_rowid='rowid'
);
"""

# FTS-Trigger werden separat angelegt (VIRTUAL TABLE Triggers
# brauchen erst die Tabelle)
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

# Valide Layer-Namen (kompatibel mit JSONL-MemoryStorage)
VALID_LAYERS: set[str] = {"episodic", "semantic", "procedural"}


class SqliteConnection:
    """Thread-safe SQLite-Connection-Manager mit WAL-Mode.

    Verwendet eine persistente Connection mit explizitem Lock
    statt Connection-per-Request. Grund: WAL-Mode profitiert
    von einer langlebigen Connection, und der Bot ist single-process.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_connection(self) -> sqlite3.Connection:
        """Erstellt die Connection lazy beim ersten Zugriff."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit, wir steuern Transaktionen manuell
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialisiert Schema + FTS-Trigger (idempotent)."""
        conn = self._conn
        if conn is None:  # pragma: no cover
            raise RuntimeError("_init_schema called before connection established")
        conn.executescript(_SCHEMA_SQL)
        conn.executescript(_TRIGGER_SQL)
        log.debug("SQLite-Schema initialisiert: %s", self._db_path)

    def execute(
        self,
        sql: str,
        params: tuple | dict = (),
        *,
        many: bool = False,
        data: list[tuple] | None = None,
    ) -> sqlite3.Cursor:
        """Thread-safe SQL-Ausführung mit Lock.

        Args:
            sql: SQL-Statement mit ?-Platzhaltern.
            params: Parameter-Tupel oder -Dict.
            many: Wenn True, executemany mit data.
            data: Datenliste für executemany.

        Returns:
            sqlite3.Cursor mit Ergebnis.
        """
        with self._lock:
            conn = self._ensure_connection()
            if many and data is not None:
                return conn.executemany(sql, data)
            return conn.execute(sql, params)

    def execute_in_transaction(self, operations: list[tuple[str, tuple]]) -> None:
        """Führt mehrere Statements in einer Transaktion aus.

        Args:
            operations: Liste von (sql, params) Tupeln.
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
        """Thread-safe Query mit fetchall.

        Args:
            sql: SELECT-Statement.
            params: Parameter.

        Returns:
            Liste von sqlite3.Row-Objekten.
        """
        with self._lock:
            conn = self._ensure_connection()
            return conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple | dict = ()) -> Optional[sqlite3.Row]:
        """Thread-safe Query mit fetchone.

        Args:
            sql: SELECT-Statement.
            params: Parameter.

        Returns:
            sqlite3.Row oder None.
        """
        with self._lock:
            conn = self._ensure_connection()
            return conn.execute(sql, params).fetchone()

    def close(self) -> None:
        """Schließt die Connection (für graceful shutdown)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ──────────────────────────────────────────────────────────────
# Bookmark Storage (SQLite)
# ──────────────────────────────────────────────────────────────


class SqliteBookmarkStorage:
    """SQLite-Adapter für Bookmarks.

    Drop-in-Replacement für die JSONL-basierten Bookmark-Funktionen.
    API ist identisch zu infrastructure.bookmark_storage.
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
        """Speichert einen Bookmark-Eintrag.

        Args:
            user_id: Telegram User-ID.
            username: Telegram Username (kann None sein).
            message_id: Telegram Message-ID der Bot-Antwort.
            chat_id: Telegram Chat-ID.
            content: Volltext der Bot-Antwort.

        Returns:
            Der gespeicherte Bookmark-Eintrag als Dict.
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
            "Bookmark gespeichert: user=%s chat_id=%d message_id=%d content_len=%d",
            username,
            chat_id,
            message_id,
            len(content),
        )
        return entry

    def list_recent_bookmarks(
        self, user_id: int, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Gibt die neuesten Bookmarks eines Users zurück.

        Args:
            user_id: Telegram User-ID.
            limit: Maximale Anzahl zurückzugebender Bookmarks.

        Returns:
            Liste von Bookmark-Dicts, neueste zuerst, max `limit` Einträge.
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
        """Sucht Bookmarks per Inhalts-Substring (case-insensitive).

        Args:
            user_id: Telegram User-ID.
            query: Suchbegriff für den Bookmark-Inhalt.
            limit: Maximale Anzahl Ergebnisse.

        Returns:
            Liste passender Bookmark-Dicts, neueste zuerst.
        """
        # SQLite LIKE ist per Default case-insensitive für ASCII.
        # Für Unicode-Korrektheit verwenden wir LOWER().
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
        """Findet einen Bookmark per chat_id + message_id.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            message_id: Telegram Message-ID zum Suchen.

        Returns:
            Bookmark-Dict oder None falls nicht gefunden.
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
        """Prüft ob ein Bookmark existiert.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            message_id: Telegram Message-ID zum Prüfen.

        Returns:
            True wenn der Bookmark existiert, False sonst.
        """
        row = self._conn.fetchone(
            """SELECT 1 FROM bookmarks
               WHERE user_id = ? AND chat_id = ? AND message_id = ?""",
            (user_id, chat_id, message_id),
        )
        return row is not None

    def delete_bookmark(self, user_id: int, chat_id: int, message_id: int) -> bool:
        """Löscht einen Bookmark per chat_id + message_id.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            message_id: Telegram Message-ID zum Entfernen.

        Returns:
            True wenn ein Bookmark gelöscht wurde, False falls nicht gefunden.
        """
        cursor = self._conn.execute(
            """DELETE FROM bookmarks
               WHERE user_id = ? AND chat_id = ? AND message_id = ?""",
            (user_id, chat_id, message_id),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            log.info(
                "Bookmark gelöscht: user_id=%d chat_id=%d message_id=%d",
                user_id,
                chat_id,
                message_id,
            )
        return deleted


# ──────────────────────────────────────────────────────────────
# Memory Storage (SQLite)
# ──────────────────────────────────────────────────────────────


class SqliteMemoryStorage:
    """SQLite-Adapter für Trinity-Memory-Persistierung.

    Drop-in-Replacement für MemoryStorage (JSONL).
    Alle drei Layer (episodic, semantic, procedural) leben in einer
    Tabelle mit type-Spalte. Layer-spezifische Felder werden in
    metadata_json gespeichert.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn

    @staticmethod
    def _validate_layer(layer: str) -> None:
        """Prüft ob der Layer valide ist.

        Raises:
            ValueError: Bei unbekanntem Layer.
        """
        if layer not in VALID_LAYERS:
            raise ValueError(f"Unbekannter Layer: '{layer}'. Erlaubt: {VALID_LAYERS}")

    @staticmethod
    def _entry_to_row(entry: dict, layer: str) -> tuple:
        """Konvertiert ein Entry-Dict in ein SQLite-Row-Tupel.

        Typ-spezifische Felder (context, category, skill_name, usage_count)
        werden in metadata_json gepackt.

        Args:
            entry: Serialisiertes Entry-Dict.
            layer: Ziel-Layer.

        Returns:
            Tupel für INSERT.
        """
        # Alle Felder die nicht zum Basis-Schema gehören -> metadata
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
        """Konvertiert eine SQLite-Row zurück in ein Entry-Dict.

        Merged metadata_json zurück in das Haupt-Dict.

        Args:
            row: sqlite3.Row-Objekt.

        Returns:
            Entry-Dict (kompatibel mit JSONL-Format).
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
                log.warning("Korruptes metadata_json für entry %s", row["id"])

        return entry

    def append(self, entry: dict, layer: str) -> None:
        """Hängt einen Entry an den entsprechenden Layer an.

        Args:
            entry: Serialisiertes Entry-Dict.
            layer: Ziel-Layer.
        """
        self._validate_layer(layer)
        row_data = self._entry_to_row(entry, layer)
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_entries
               (id, user_id, type, content, importance, timestamp, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            row_data,
        )
        log.debug("Memory-Entry angehängt: layer=%s id=%s", layer, entry.get("id"))

    def list_entries(self, user_id: int, layer: str, limit: int = 50) -> list[dict]:
        """Liest Entries für einen User, neueste zuerst.

        Args:
            user_id: Telegram-User-ID.
            layer: Zu lesender Layer.
            limit: Maximale Anzahl Einträge.

        Returns:
            Liste von Entry-Dicts, neueste zuerst (nach Timestamp sortiert).
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
        """Durchsucht Memory-Entries eines Users.

        Unterstützt zwei Modi:
          - "substring": SQLite LIKE (default, kompatibel mit JSONL)
          - "embedding": Phase 1+, noch nicht implementiert

        Wenn FTS5-Index vorhanden ist UND mode="substring", wird
        FTS5 für bessere Performance genutzt.

        Args:
            user_id: Telegram-User-ID.
            query: Suchbegriff.
            layer: Zu durchsuchender Layer.
            limit: Maximale Treffer.
            mode: "substring" oder "embedding".

        Returns:
            Liste von matching Entry-Dicts, neueste Treffer zuerst.

        Raises:
            NotImplementedError: Bei mode="embedding".
        """
        if mode == "embedding":
            raise NotImplementedError(
                "Vector-Embedding-Suche ist Phase 1+. Heute nur 'substring'."
            )

        self._validate_layer(layer)

        # Versuche FTS5 zuerst (schneller bei großen Datenmengen)
        try:
            # Anführungszeichen aus Query entfernen (FTS5-Syntax-Fehler vermeiden)
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
                # FTS5 lieferte 0 Treffer: auf LIKE zurückfallen
                # (FTS5 tokenisiert, findet keine Token-Inneren wie "Super" in "Superword")
                log.debug("FTS5: 0 Treffer für '%s', Fallback auf LIKE", query)
        except sqlite3.OperationalError:
            # FTS-Tabelle kaputt oder nicht vorhanden: LIKE-Fallback
            log.debug("FTS5-Suche fehlgeschlagen, Fallback auf LIKE")

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
        """Löscht einen Entry anhand seiner ID.

        Verifiziert Ownership: Entry muss dem User gehören.

        Args:
            entry_id: ID des zu löschenden Entries.
            layer: Layer in dem gesucht wird.
            user_id: User-ID für Ownership-Check.

        Returns:
            True wenn Entry gefunden und gelöscht, False wenn nicht gefunden.
        """
        self._validate_layer(layer)
        cursor = self._conn.execute(
            """DELETE FROM memory_entries
               WHERE id = ? AND type = ? AND user_id = ?""",
            (entry_id, layer, user_id),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            log.info("Memory-Entry gelöscht: id=%s layer=%s", entry_id, layer)
        return deleted

    def get_by_id(self, entry_id: str, layer: str, user_id: int) -> Optional[dict]:
        """Liest einen einzelnen Entry anhand seiner ID.

        Args:
            entry_id: Gesuchte Entry-ID.
            layer: Layer in dem gesucht wird.
            user_id: User-ID für Ownership-Check.

        Returns:
            Entry-Dict oder None wenn nicht gefunden.
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
    """SQLite-Adapter für Rate-Limit-Profile.

    Ersetzt JSONL-basierte Profile-Persistierung in rate_limiter.py.
    Speichert pro user_id das aktive Profil.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn

    def load_all(self) -> dict[int, str]:
        """Lädt alle User-Profile.

        Returns:
            Dict: user_id -> profile_name.
        """
        rows = self._conn.fetchall("SELECT user_id, profile FROM user_profiles")
        return {int(row["user_id"]): row["profile"] for row in rows}

    def save(self, user_id: int, chat_id: int, profile: str) -> None:
        """Speichert oder aktualisiert ein User-Profil.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            profile: Profilname (light, normal, power, unlimited).
        """
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO user_profiles
               (user_id, chat_id, profile, set_at)
               VALUES (?, ?, ?, ?)""",
            (user_id, chat_id, profile, ts),
        )
        log.debug("Profil gespeichert: user_id=%d profile=%s", user_id, profile)


# ──────────────────────────────────────────────────────────────
# Model Storage (SQLite)
# ──────────────────────────────────────────────────────────────


class SqliteModelStorage:
    """SQLite-Adapter für User-Modell-Overrides.

    Speichert pro (user_id, slot) das gewählte Modell.
    Phase 1: nur slot='global'. Phase 2+: 'chat', 'code', etc.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn

    def get_model(self, user_id: int, slot: str = "global") -> Optional[str]:
        """Liest das aktive Modell-Override für einen User und Slot.

        Args:
            user_id: Telegram-User-ID.
            slot: Slot-Name (default: 'global').

        Returns:
            Modell-ID als String oder None wenn kein Override gesetzt.
        """
        row = self._conn.fetchone(
            "SELECT model_id FROM user_slot_models WHERE user_id = ? AND slot = ?",
            (user_id, slot),
        )
        return row["model_id"] if row else None

    def set_model(self, user_id: int, model_id: str, slot: str = "global") -> None:
        """Setzt oder aktualisiert das Modell-Override.

        Args:
            user_id: Telegram-User-ID.
            model_id: Volle Modell-ID (z.B. 'claude-opus-4-7').
            slot: Slot-Name (default: 'global').
        """
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO user_slot_models
               (user_id, slot, model_id, set_at)
               VALUES (?, ?, ?, ?)""",
            (user_id, slot, model_id, ts),
        )
        log.debug(
            "Modell-Override gespeichert: user_id=%d slot=%s model=%s",
            user_id,
            slot,
            model_id,
        )

    def delete_model(self, user_id: int, slot: str = "global") -> bool:
        """Entfernt ein Modell-Override (Reset auf Default).

        Args:
            user_id: Telegram-User-ID.
            slot: Slot-Name (default: 'global').

        Returns:
            True wenn ein Override gelöscht wurde.
        """
        cursor = self._conn.execute(
            "DELETE FROM user_slot_models WHERE user_id = ? AND slot = ?",
            (user_id, slot),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            log.debug("Modell-Override gelöscht: user_id=%d slot=%s", user_id, slot)
        return deleted

    def _reset_all_for_tests(self) -> None:
        """Löscht alle Modell-Overrides (nur für Tests).

        Konsistenz-Pattern: analog zu conversation_storage._reset_all_for_tests.
        """
        self._conn.execute("DELETE FROM user_slot_models", ())


# ──────────────────────────────────────────────────────────────
# JSONL → SQLite Migration
# ──────────────────────────────────────────────────────────────


def migrate_jsonl_to_sqlite(
    conn: SqliteConnection,
    data_dir: Path,
) -> dict[str, int]:
    """Migriert bestehende JSONL-Daten in SQLite (idempotent).

    Ablauf:
      1. Schema ist bereits initialisiert (via SqliteConnection)
      2. Wenn bookmarks.jsonl existiert UND bookmarks-Tabelle leer:
         alle Zeilen importieren
      3. Analog für memory_*.jsonl
      4. Migrierte JSONL-Files als .bak umbenennen

    Args:
        conn: Initialisierte SqliteConnection.
        data_dir: Pfad zum data/-Ordner mit JSONL-Dateien.

    Returns:
        Dict mit Migrations-Statistiken:
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
                    "Bookmark-Migration: %d Einträge migriert, %s -> %s",
                    count,
                    bm_path.name,
                    bak_path.name,
                )
        else:
            log.debug("Bookmark-Migration übersprungen: Tabelle nicht leer")

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
                        "Memory-Migration (%s): %d Einträge migriert, %s -> %s",
                        layer,
                        count,
                        jsonl_path.name,
                        bak_path.name,
                    )
            else:
                log.debug(
                    "Memory-Migration (%s) übersprungen: Tabelle nicht leer",
                    layer,
                )

    return stats


def _migrate_bookmarks_jsonl(conn: SqliteConnection, path: Path) -> int:
    """Liest bookmarks.jsonl und schreibt alle Einträge in SQLite.

    Args:
        conn: SQLite-Connection.
        path: Pfad zur bookmarks.jsonl.

    Returns:
        Anzahl migrierter Einträge.
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
                    "Migration: korrupte Bookmark-Zeile %d übersprungen: %s",
                    line_num,
                    e,
                )

    if corrupt > 0:
        log.info("Migration: %d korrupte Bookmark-Zeilen übersprungen", corrupt)

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
    """Liest eine memory_*.jsonl und schreibt alle Einträge in SQLite.

    Args:
        conn: SQLite-Connection.
        path: Pfad zur JSONL-Datei.
        layer: Memory-Layer (episodic/semantic/procedural).

    Returns:
        Anzahl migrierter Einträge.
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
                    "Migration: korrupte Memory-Zeile %d in %s übersprungen: %s",
                    line_num,
                    path.name,
                    e,
                )

    if corrupt > 0:
        log.info(
            "Migration: %d korrupte Zeilen in %s übersprungen",
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
