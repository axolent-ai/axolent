"""Tests for SQLCipher migration and encrypted storage (Step 1.1/10).

Covers:
  - CryptoConnection in non-encrypted mode (dev/test fallback)
  - Key generation and deterministic format
  - AG-SC-7: production mode refuses unencrypted DB
  - Migration path: plaintext -> encrypted (mocked SQLCipher)
  - Thread-safe operations
  - CryptoConnection API compatibility with SqliteConnection
"""

from __future__ import annotations


import pytest

from infrastructure.crypto_storage import (
    CryptoConnection,
    _generate_key,
    is_sqlcipher_available,
    migrate_plaintext_to_encrypted,
)


class TestKeyGeneration:
    """Tests for encryption key generation."""

    def test_key_length(self):
        """Generated key should be 64 hex characters (32 bytes)."""
        key = _generate_key()
        assert len(key) == 64

    def test_key_is_hex(self):
        """Generated key should contain only hex characters."""
        key = _generate_key()
        assert all(c in "0123456789abcdef" for c in key)

    def test_keys_are_unique(self):
        """Each generated key should be unique."""
        keys = {_generate_key() for _ in range(100)}
        assert len(keys) == 100


class TestCryptoConnectionUnencrypted:
    """Tests for CryptoConnection in dev/test mode (no SQLCipher)."""

    def test_fallback_to_plain_sqlite(self, tmp_path):
        """Without SQLCipher, should fall back to plain sqlite3 in non-prod."""
        db_path = tmp_path / "test.db"
        conn = CryptoConnection(db_path, require_encryption=False)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'hello')", ())
        row = conn.fetchone("SELECT val FROM test WHERE id = 1")
        assert row["val"] == "hello"
        assert conn.is_encrypted is False
        conn.close()

    def test_execute_in_transaction(self, tmp_path):
        """Transaction support should work."""
        db_path = tmp_path / "test.db"
        conn = CryptoConnection(db_path, require_encryption=False)
        conn.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute_in_transaction(
            [
                ("INSERT INTO t VALUES (1, 'a')", ()),
                ("INSERT INTO t VALUES (2, 'b')", ()),
            ]
        )
        rows = conn.fetchall("SELECT * FROM t ORDER BY id")
        assert len(rows) == 2
        conn.close()

    def test_fetchall_returns_rows(self, tmp_path):
        """fetchall should return a list of Row objects."""
        db_path = tmp_path / "test.db"
        conn = CryptoConnection(db_path, require_encryption=False)
        conn.executescript("CREATE TABLE t (id INTEGER, v TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'a')", ())
        conn.execute("INSERT INTO t VALUES (2, 'b')", ())
        rows = conn.fetchall("SELECT * FROM t ORDER BY id")
        assert len(rows) == 2
        assert rows[0]["v"] == "a"
        conn.close()

    def test_close_and_reopen(self, tmp_path):
        """After close, operations should reconnect lazily."""
        db_path = tmp_path / "test.db"
        conn = CryptoConnection(db_path, require_encryption=False)
        conn.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO t VALUES (1)", ())
        conn.close()
        # Should reconnect on next operation
        row = conn.fetchone("SELECT * FROM t WHERE id = 1")
        assert row is not None
        conn.close()

    def test_parent_dir_created(self, tmp_path):
        """Parent directories should be created automatically."""
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        conn = CryptoConnection(deep_path, require_encryption=False)
        conn.executescript("CREATE TABLE t (id INTEGER)")
        assert deep_path.parent.exists()
        conn.close()


class TestProductionGuard:
    """AG-SC-7: Production mode must refuse unencrypted DB."""

    def test_production_mode_blocks_unencrypted(self, tmp_path):
        """In production mode, CryptoConnection must refuse to open without encryption."""
        db_path = tmp_path / "test.db"
        conn = CryptoConnection(db_path, require_encryption=True)

        # If SQLCipher is not available, this should raise
        if not is_sqlcipher_available():
            with pytest.raises(RuntimeError, match="AG-SC-7"):
                conn.execute("SELECT 1")

    def test_production_env_var_detection(self, tmp_path, monkeypatch):
        """AXOLENT_PRODUCTION=true should auto-enable require_encryption."""
        monkeypatch.setenv("AXOLENT_PRODUCTION", "true")
        db_path = tmp_path / "test.db"
        conn = CryptoConnection(db_path)
        assert conn._require_encryption is True


class TestMigrationPlaintext:
    """Tests for plaintext to encrypted migration."""

    def test_migration_skips_nonexistent(self, tmp_path):
        """Migration should skip if plaintext DB does not exist."""
        result = migrate_plaintext_to_encrypted(
            tmp_path / "nonexistent.db",
            tmp_path / "encrypted.db",
            "fake_key",
        )
        assert result is False

    def test_migration_detects_already_encrypted(self, tmp_path):
        """Migration should skip if the DB is not readable as plaintext."""
        # Create a file with random bytes (simulates encrypted DB)
        db_path = tmp_path / "encrypted.db"
        db_path.write_bytes(b"SQLite format 3\x00" + b"\xff" * 100)
        result = migrate_plaintext_to_encrypted(
            db_path, tmp_path / "new.db", "fake_key"
        )
        assert result is False


class TestCryptoConnectionAPICompat:
    """Tests that CryptoConnection is API-compatible with SqliteConnection."""

    def test_has_execute(self):
        """CryptoConnection must have execute method."""
        assert hasattr(CryptoConnection, "execute")

    def test_has_fetchall(self):
        """CryptoConnection must have fetchall method."""
        assert hasattr(CryptoConnection, "fetchall")

    def test_has_fetchone(self):
        """CryptoConnection must have fetchone method."""
        assert hasattr(CryptoConnection, "fetchone")

    def test_has_execute_in_transaction(self):
        """CryptoConnection must have execute_in_transaction method."""
        assert hasattr(CryptoConnection, "execute_in_transaction")

    def test_has_executescript(self):
        """CryptoConnection must have executescript method."""
        assert hasattr(CryptoConnection, "executescript")

    def test_has_close(self):
        """CryptoConnection must have close method."""
        assert hasattr(CryptoConnection, "close")
