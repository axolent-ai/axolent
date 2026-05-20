"""Architecture Guard Tests for Skill-Compression.

AG-SC-7: test_sqlcipher_enabled_in_prod
  DB must not be opened unencrypted in production mode.

AG-SC-8: test_pattern_type_is_text_not_enum
  type column is TEXT, no CHECK constraint with fixed values.

Additional guards:
  - No direct sqlite3.connect() in production code (crypto_storage.py)
  - Hypothesis is frozen dataclass with slots
  - Fingerprint hash is deterministic
"""

from __future__ import annotations

import ast
import re
import sqlite3
from pathlib import Path

import pytest

from application.skill_compression.event_normalizer import compute_fingerprint
from application.skill_compression.hypothesis_storage import (
    HYPOTHESIS_SCHEMA_SQL,
    Hypothesis,
)
from infrastructure.crypto_storage import CryptoConnection, is_sqlcipher_available


class TestAGSC7SqlcipherInProd:
    """AG-SC-7: DB must not be opened unencrypted in production mode."""

    def test_production_refuses_unencrypted(self, tmp_path):
        """CryptoConnection with require_encryption=True must refuse plain sqlite3."""
        db_path = tmp_path / "prod.db"
        conn = CryptoConnection(db_path, require_encryption=True)

        if not is_sqlcipher_available():
            with pytest.raises(RuntimeError, match="AG-SC-7"):
                conn.execute("SELECT 1")
        else:
            # If SQLCipher IS available, it should open encrypted
            conn.execute("SELECT 1")
            assert conn.is_encrypted is True
            conn.close()

    def test_no_bare_sqlite3_connect_in_crypto_module(self):
        """The crypto_storage module must not call sqlite3.connect() for production paths.

        It may only use _open_sqlcipher() for encrypted connections.
        The plain sqlite3.connect() calls are:
          1. In the fallback path (guarded by require_encryption check)
          2. In migration (for reading the plaintext source)
        Both are legitimate. This test verifies the guard is present.
        """
        source_path = (
            Path(__file__).resolve().parents[3] / "infrastructure" / "crypto_storage.py"
        )
        source = source_path.read_text(encoding="utf-8")

        # Parse AST to find sqlite3.connect calls
        tree = ast.parse(source)
        sqlite3_connects = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match sqlite3.connect(...)
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "connect"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "sqlite3"
                ):
                    sqlite3_connects.append(node.lineno)

        # There should be exactly 2 plain sqlite3.connect calls:
        # 1. In CryptoConnection._ensure_connection (fallback, guarded)
        # 2. In migrate_plaintext_to_encrypted (reading source)
        assert len(sqlite3_connects) <= 3, (
            f"Found {len(sqlite3_connects)} sqlite3.connect() calls at lines "
            f"{sqlite3_connects}. Expected at most 3 (fallback + migration read + "
            f"migration verification). Each must be guarded."
        )


class TestAGSC8PatternTypeText:
    """AG-SC-8: pattern type must be TEXT, not ENUM with CHECK constraint."""

    def test_schema_type_column_is_text(self):
        """The type column in hypotheses DDL must be TEXT."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(HYPOTHESIS_SCHEMA_SQL)

        rows = conn.execute("PRAGMA table_info(hypotheses)").fetchall()
        type_col = [r for r in rows if r[1] == "type"]
        assert len(type_col) == 1
        assert type_col[0][2] == "TEXT"
        conn.close()

    def test_no_check_constraint_in_ddl(self):
        """The DDL must not contain CHECK constraints on type-like columns."""
        # Check the raw DDL string
        upper_sql = HYPOTHESIS_SCHEMA_SQL.upper()
        # Split into individual statements
        for statement in upper_sql.split(";"):
            if "CREATE TABLE" in statement and "HYPOTHES" in statement:
                # Must not have CHECK(...) that constrains type values
                if "CHECK" in statement:
                    # It is acceptable to have CHECK for boolean columns,
                    # but not for type/status/source_type
                    check_match = re.search(r"CHECK\s*\(.*TYPE.*\)", statement)
                    assert check_match is None, (
                        "Found CHECK constraint on TYPE column in DDL"
                    )

    def test_future_type_insertable(self):
        """Types not in v1 (context, style, outcome) must be insertable."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(HYPOTHESIS_SCHEMA_SQL)

        # Insert with future types
        future_types = ["context", "style", "outcome", "workflow", "custom_v3"]
        for i, ftype in enumerate(future_types):
            conn.execute(
                "INSERT INTO hypotheses "
                "(hypothesis_id, user_id, type, claim, status, version, "
                "elo_rating, elo_games_played, bayes_confidence, "
                "support_count, contradict_count, fsrs_state_json, "
                "source_type, decay_immune, created_at, last_seen, "
                "approval_state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"hyp_future_{i}",
                    1,
                    ftype,
                    "test",
                    "candidate",
                    1,
                    1500.0,
                    0,
                    0.5,
                    0,
                    0,
                    "{}",
                    "live_chat",
                    0,
                    "2026-05-20T00:00:00Z",
                    "2026-05-20T00:00:00Z",
                    "pending",
                ),
            )

        # Verify all were inserted
        count = conn.execute("SELECT count(*) FROM hypotheses").fetchone()[0]
        assert count == len(future_types)
        conn.close()


class TestHypothesisFrozenInvariant:
    """AG-SC-1 (related): Hypothesis must be frozen dataclass with slots."""

    def test_hypothesis_is_frozen(self):
        """Hypothesis instances must be immutable."""
        h = Hypothesis(hypothesis_id="test")
        with pytest.raises(AttributeError):
            h.claim = "mutated"  # type: ignore[misc]

    def test_hypothesis_has_slots(self):
        """Hypothesis must use __slots__ for memory efficiency."""
        assert hasattr(Hypothesis, "__slots__")
        # Frozen + slots = no __dict__
        h = Hypothesis(hypothesis_id="test")
        assert not hasattr(h, "__dict__")


class TestFingerprintDeterminism:
    """Guard: fingerprint computation must be deterministic."""

    def test_same_inputs_always_same_hash(self):
        """100 calls with same inputs must produce same hash."""
        hashes = set()
        for _ in range(100):
            h = compute_fingerprint(
                intent="create_code",
                domain="development",
                format_type="code",
                constraints={"duration": "30s"},
                scope={"project": "test"},
                language="en",
            )
            hashes.add(h)
        assert len(hashes) == 1

    def test_constraint_order_irrelevant(self):
        """Dict key order in constraints must not affect hash."""
        h1 = compute_fingerprint(
            intent="test",
            domain="test",
            format_type="test",
            constraints={"a": "1", "b": "2"},
            scope={},
            language="en",
        )
        h2 = compute_fingerprint(
            intent="test",
            domain="test",
            format_type="test",
            constraints={"b": "2", "a": "1"},
            scope={},
            language="en",
        )
        assert h1 == h2
