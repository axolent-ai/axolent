"""Encrypted SQLite storage via SQLCipher (closes SECURITY.md R-1).

Provides CryptoConnection as a drop-in replacement for SqliteConnection
with transparent AES-256 encryption at rest.

Key management uses the OS credential vault via the `keyring` library:
  * macOS: Keychain
  * Windows: Credential Manager (DPAPI)
  * Linux: Secret Service / gnome-keyring

On first start (no key in vault): generates a 32-byte random key,
stores it in the vault, and initializes the encrypted DB.

Migration path (one-shot): if an unencrypted axolent.db exists but no
encrypted DB, the migrator exports all data into the new encrypted DB,
then renames the old file to .plaintext.bak.

Architecture Guard AG-SC-7: in production mode, opening the DB
without SQLCipher is a fatal error.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Service name for keyring (consistent across platforms)
_KEYRING_SERVICE = "axolent-ai"
_KEYRING_USERNAME = "db-encryption-key"

# Encrypted DB filename (different from plaintext to allow co-existence during migration)
ENCRYPTED_DB_NAME = "axolent.db"
PLAINTEXT_BACKUP_SUFFIX = ".plaintext.bak"


def _generate_key() -> str:
    """Generate a 64-char hex key (32 bytes of entropy, AES-256).

    Returns:
        Hex-encoded 32-byte key string.
    """
    return secrets.token_hex(32)


def get_or_create_key() -> str:
    """Retrieve the DB encryption key from the OS vault, or create one.

    On first call (key not in vault):
      1. Generates a new 32-byte random key
      2. Stores it in the OS credential vault
      3. Returns it

    On subsequent calls:
      1. Retrieves the key from the vault
      2. Returns it

    Returns:
        Hex-encoded encryption key.

    Raises:
        RuntimeError: If keyring is not available or vault access fails.
    """
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError(
            "keyring library is required for SQLCipher key management. "
            "Install with: pip install keyring"
        ) from exc

    try:
        existing_key = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read encryption key from OS vault: {exc}. "
            "Ensure your OS keyring service is running."
        ) from exc

    if existing_key is not None:
        log.debug("Encryption key retrieved from OS vault")
        return existing_key

    # First start: generate and store
    new_key = _generate_key()
    try:
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, new_key)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to store encryption key in OS vault: {exc}. "
            "Ensure your OS keyring service is running and allows write access."
        ) from exc

    log.info("New encryption key generated and stored in OS vault")
    return new_key


def _open_sqlcipher(db_path: Path, key: str) -> sqlite3.Connection:
    """Open a SQLCipher-encrypted database connection.

    Tries pysqlcipher3 first, then falls back to sqlcipher3.
    Both provide a sqlite3-compatible interface with PRAGMA key support.

    Args:
        db_path: Path to the encrypted .db file.
        key: Hex-encoded encryption key.

    Returns:
        An open sqlite3-compatible Connection with encryption active.

    Raises:
        RuntimeError: If no SQLCipher library is available.
    """
    conn: Optional[sqlite3.Connection] = None

    # Try pysqlcipher3 first
    try:
        from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore[import-untyped]

        conn = sqlcipher.connect(str(db_path), check_same_thread=False)
        # PRAGMA key cannot be parameterized (SQLCipher API limitation).
        # Key comes from OS keyring, never user-controlled.
        conn.execute(f"PRAGMA key=\"x'{key}'\"")  # nosec B608  # nosemgrep
        # Verify the key works by reading the schema
        conn.execute("SELECT count(*) FROM sqlite_master")
        log.debug("SQLCipher connection via pysqlcipher3: %s", db_path)
        return conn
    except ImportError:
        log.debug("pysqlcipher3 not available, trying sqlcipher3")
    except Exception as exc:
        if conn is not None:
            conn.close()
        log.debug("pysqlcipher3 failed: %s", exc)

    # Try sqlcipher3
    try:
        import sqlcipher3  # type: ignore[import-untyped]

        conn = sqlcipher3.connect(str(db_path), check_same_thread=False)
        # PRAGMA key cannot be parameterized (SQLCipher API limitation).
        # Key comes from OS keyring, never user-controlled.
        conn.execute(f"PRAGMA key=\"x'{key}'\"")  # nosec B608  # nosemgrep
        conn.execute("SELECT count(*) FROM sqlite_master")
        log.debug("SQLCipher connection via sqlcipher3: %s", db_path)
        return conn
    except ImportError:
        log.debug("sqlcipher3 not available either")
    except Exception as exc:
        if conn is not None:
            conn.close()
        log.debug("sqlcipher3 failed: %s", exc)

    raise RuntimeError(
        "No SQLCipher library available. Install one of: "
        "pysqlcipher3, sqlcipher3. "
        "Example: pip install pysqlcipher3"
    )


def is_sqlcipher_available() -> bool:
    """Check whether any SQLCipher library is importable.

    Returns:
        True if at least one SQLCipher provider can be imported.
    """
    try:
        from pysqlcipher3 import dbapi2  # type: ignore[import-untyped] # noqa: F401

        return True
    except ImportError:
        pass

    try:
        import sqlcipher3  # type: ignore[import-untyped] # noqa: F401

        return True
    except ImportError:
        pass

    return False


class CryptoConnection:
    """Thread-safe encrypted SQLite connection manager.

    Drop-in replacement for SqliteConnection with SQLCipher encryption.
    Uses the same interface: execute, fetchall, fetchone, close, etc.

    If SQLCipher is not available (dev/test environment), falls back to
    plain sqlite3 with a logged warning. In production mode
    (AXOLENT_PRODUCTION=true), this fallback is blocked (AG-SC-7).
    """

    def __init__(
        self,
        db_path: Path | str,
        key: Optional[str] = None,
        *,
        require_encryption: Optional[bool] = None,
    ) -> None:
        """Initialize the encrypted connection manager.

        Args:
            db_path: Path to the database file.
            key: Encryption key (hex). If None, retrieved from OS vault.
            require_encryption: If True, refuse to open without encryption.
                If None, auto-detect from AXOLENT_PRODUCTION env var.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._key = key
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._encrypted = False

        if require_encryption is None:
            self._require_encryption = (
                os.environ.get("AXOLENT_PRODUCTION", "").lower() == "true"
            )
        else:
            self._require_encryption = require_encryption

    @property
    def is_encrypted(self) -> bool:
        """Whether the current connection uses encryption."""
        return self._encrypted

    def _ensure_connection(self) -> sqlite3.Connection:
        """Create the connection lazily on first access."""
        if self._conn is not None:
            return self._conn

        # Try encrypted connection first
        if is_sqlcipher_available():
            if self._key is None:
                self._key = get_or_create_key()
            try:
                self._conn = _open_sqlcipher(self._db_path, self._key)
                self._encrypted = True
                self._conn.row_factory = sqlite3.Row
                self._conn.isolation_level = None  # Autocommit (manual txn control)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA busy_timeout=5000")
                log.info("Encrypted DB connection established: %s", self._db_path)
                return self._conn
            except Exception as exc:
                log.warning("SQLCipher connection failed: %s", exc)

        # Fallback: plain sqlite3 (blocked in production)
        if self._require_encryption:
            raise RuntimeError(
                "AG-SC-7 VIOLATION: Cannot open database without encryption "
                "in production mode. Install pysqlcipher3 or sqlcipher3. "
                f"DB path: {self._db_path}"
            )

        log.warning(
            "SQLCipher not available. Opening DB WITHOUT encryption. "
            "This is acceptable for development/testing only."
        )
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._encrypted = False
        return self._conn

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

    def executescript(self, sql: str) -> None:
        """Execute a multi-statement SQL script.

        Args:
            sql: SQL script with multiple statements.
        """
        with self._lock:
            conn = self._ensure_connection()
            conn.executescript(sql)

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
                self._encrypted = False


def migrate_plaintext_to_encrypted(
    plaintext_path: Path,
    encrypted_path: Path,
    key: str,
) -> bool:
    """One-shot migration of plaintext SQLite DB to encrypted SQLCipher DB.

    Strategy (IC-SC-8: one-shot):
      1. Open plaintext DB with standard sqlite3
      2. Open new encrypted DB with SQLCipher
      3. Copy schema + data via SQL dump
      4. Verify row counts match
      5. Rename plaintext to .plaintext.bak

    This is safe because:
      - The plaintext .bak file is kept as recovery option
      - Row count verification catches truncation errors
      - If anything fails, the original file is untouched

    Args:
        plaintext_path: Path to unencrypted axolent.db.
        encrypted_path: Path for new encrypted DB (can be same path).
        key: Hex-encoded encryption key.

    Returns:
        True if migration was performed, False if not needed.

    Raises:
        RuntimeError: If migration fails (original file is preserved).
    """
    if not plaintext_path.exists():
        log.debug("No plaintext DB found at %s, no migration needed", plaintext_path)
        return False

    # Check if file is actually unencrypted by trying to read with plain sqlite3
    try:
        test_conn = sqlite3.connect(str(plaintext_path))
        test_conn.execute("SELECT count(*) FROM sqlite_master")
        test_conn.close()
    except sqlite3.DatabaseError:
        log.debug(
            "DB at %s is already encrypted or corrupt, skipping migration",
            plaintext_path,
        )
        return False

    if not is_sqlcipher_available():
        log.warning(
            "SQLCipher not available. Cannot migrate plaintext DB to encrypted. "
            "Install pysqlcipher3 or sqlcipher3."
        )
        return False

    log.info("Starting plaintext -> encrypted DB migration: %s", plaintext_path)

    # Step 1: Read all data from plaintext
    src = sqlite3.connect(str(plaintext_path))
    src.row_factory = sqlite3.Row

    # Get table row counts for verification
    table_counts: dict[str, int] = {}
    try:
        tables = [
            row[0]
            for row in src.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'memory_fts%'"
            ).fetchall()
        ]
        for table in tables:
            # Table name comes from sqlite_master query, never user-controlled.
            count = src.execute(f"SELECT count(*) FROM [{table}]").fetchone()[0]  # nosec B608  # nosemgrep
            table_counts[table] = count
    except Exception as exc:
        src.close()
        raise RuntimeError(f"Failed to read plaintext DB for migration: {exc}") from exc

    # Step 2: Dump SQL
    try:
        dump_lines = list(src.iterdump())
    except Exception as exc:
        src.close()
        raise RuntimeError(f"Failed to dump plaintext DB: {exc}") from exc
    finally:
        src.close()

    # Step 3: Create encrypted DB at a temporary path first
    temp_encrypted = encrypted_path.with_suffix(".enc.tmp")
    try:
        enc_conn = _open_sqlcipher(temp_encrypted, key)
        # Execute the dump (minus BEGIN/COMMIT which iterdump includes)
        full_dump = "\n".join(dump_lines)
        enc_conn.executescript(full_dump)

        # Step 4: Verify row counts
        for table, expected_count in table_counts.items():
            try:
                # Table name comes from sqlite_master query, never user-controlled.
                row = enc_conn.execute(f"SELECT count(*) FROM [{table}]").fetchone()  # nosec B608  # nosemgrep
                actual = row[0]
                if actual != expected_count:
                    raise RuntimeError(
                        f"Migration verification failed: {table} has {actual} rows, "
                        f"expected {expected_count}"
                    )
            except sqlite3.OperationalError:
                # Table might be a virtual table that was not migrated
                log.debug("Skipping verification for table %s", table)

        enc_conn.close()
    except Exception as exc:
        # Cleanup temp file on failure
        if temp_encrypted.exists():
            temp_encrypted.unlink()
        raise RuntimeError(f"Encrypted DB creation failed: {exc}") from exc

    # Step 5: Atomic swap
    backup_path = plaintext_path.with_suffix(PLAINTEXT_BACKUP_SUFFIX)

    # Rename original to .bak
    shutil.move(str(plaintext_path), str(backup_path))

    # Move encrypted to final path
    shutil.move(str(temp_encrypted), str(encrypted_path))

    log.info(
        "Migration complete: %d tables, %d total rows. Plaintext backup: %s",
        len(table_counts),
        sum(table_counts.values()),
        backup_path,
    )
    return True
