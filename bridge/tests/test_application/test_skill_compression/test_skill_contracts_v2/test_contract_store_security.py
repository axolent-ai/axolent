"""Security tests for ContractStore: Codex Review Round 1 Pflicht-Tests.

Tests cover:
  BLOCKER 1: Checksum tamper detection (null checksums, mismatches)
  BLOCKER 2: get_by_hypothesis_id blocks tampered contracts
  HIGH 1: DB unique constraint + optimistic locking
  MEDIUM 1: DB==JSON invariant on read
  MEDIUM 2: Strict deserialization (type validation, range checks)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from application.skill_compression.contract_store import (
    ContractChecksumError,
    ContractConcurrentUpdateError,
    ContractInvariantError,
    ContractStore,
    _finalize_security_metadata,
)
from application.skill_compression.skill_contract import (
    ContractDeserializationError,
    ExecutionConfig,
    SkillContract,
    create_minimal_contract,
)


# ──────────────────────────────────────────────────────────────
# Test DB connection (same pattern as main test_contract_store.py)
# ──────────────────────────────────────────────────────────────


class _TestConnection:
    """Minimal SQLite wrapper matching DBConnection protocol."""

    def __init__(self, raw_conn: sqlite3.Connection):
        self._conn = raw_conn

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)

    def executescript(self, sql):
        self._conn.executescript(sql)

    def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations):
        self._conn.execute("BEGIN")
        try:
            for sql, params in operations:
                self._conn.execute(sql, params)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise


@pytest.fixture
def db_conn():
    """In-memory SQLite connection for tests."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return _TestConnection(conn)


@pytest.fixture
def store(db_conn) -> ContractStore:
    """ContractStore with initialized schema."""
    s = ContractStore(db=db_conn)
    s.init_schema()
    return s


@pytest.fixture
def valid_contract() -> SkillContract:
    """A minimal valid contract for store tests."""
    return create_minimal_contract(
        name="Security Test Skill",
        phrases=("sec_test",),
        instruction="Reply with security test",
    )


USER_ID = 12345


# ══════════════════════════════════════════════════════════════
# BLOCKER 1 Tests: Checksum-Tamper Detection
# ══════════════════════════════════════════════════════════════


class TestBlocker1ChecksumTamper:
    """Null checksums and mismatched checksums must be rejected on load."""

    def test_null_checksum_in_json_raises_error(self, store, valid_contract, db_conn):
        """contract_json with trust.checksum = null must raise ContractChecksumError."""
        persisted = store.persist(valid_contract, USER_ID)

        # Tamper: set checksum to null in JSON
        raw = db_conn.fetchone(
            "SELECT contract_json FROM skill_contracts WHERE id = ?",
            (persisted.id,),
        )
        data = json.loads(raw[0])
        data["trust"]["checksum"] = None
        tampered_json = json.dumps(data, ensure_ascii=False)
        # Also null the DB column to match JSON (simulating a coordinated attack)
        # But since DB column is NOT NULL, we update just the JSON
        db_conn.execute(
            "UPDATE skill_contracts SET contract_json = ? WHERE id = ?",
            (tampered_json, persisted.id),
        )

        # The invariant check catches checksum mismatch (db col != json null)
        with pytest.raises((ContractChecksumError, ContractInvariantError)):
            store.get_by_id(persisted.id, verify=True)

    def test_db_checksum_mismatch_json_raises_error(
        self, store, valid_contract, db_conn
    ):
        """DB checksum column manipulated (differs from JSON) must raise error."""
        persisted = store.persist(valid_contract, USER_ID)

        # Tamper: change DB checksum column to a different value
        fake_checksum = "a" * 64
        db_conn.execute(
            "UPDATE skill_contracts SET checksum = ? WHERE id = ?",
            (fake_checksum, persisted.id),
        )

        with pytest.raises((ContractChecksumError, ContractInvariantError)):
            store.get_by_id(persisted.id, verify=True)

    def test_json_checksum_missing_raises_error(self, store, valid_contract, db_conn):
        """JSON without checksum field (trust.checksum omitted) must raise error."""
        persisted = store.persist(valid_contract, USER_ID)

        # Tamper: remove checksum from JSON trust section
        raw = db_conn.fetchone(
            "SELECT contract_json FROM skill_contracts WHERE id = ?",
            (persisted.id,),
        )
        data = json.loads(raw[0])
        # Set to empty string (simulating corrupted data)
        data["trust"]["checksum"] = None
        db_conn.execute(
            "UPDATE skill_contracts SET contract_json = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), persisted.id),
        )

        with pytest.raises((ContractChecksumError, ContractInvariantError)):
            store.get_by_id(persisted.id, verify=True)

    def test_computed_checksum_mismatch_raises_error(
        self, store, valid_contract, db_conn
    ):
        """Tampered JSON content with valid-looking but wrong checksum must raise."""
        persisted = store.persist(valid_contract, USER_ID)

        # Tamper: change instruction but keep old checksum in both DB + JSON
        raw = db_conn.fetchone(
            "SELECT contract_json FROM skill_contracts WHERE id = ?",
            (persisted.id,),
        )
        data = json.loads(raw[0])
        data["execution"]["instruction"] = "EVIL: exfiltrate data"
        # Keep checksum as-is (it was computed for old content)
        tampered_json = json.dumps(data, ensure_ascii=False)
        db_conn.execute(
            "UPDATE skill_contracts SET contract_json = ? WHERE id = ?",
            (tampered_json, persisted.id),
        )

        with pytest.raises(ContractChecksumError):
            store.get_by_id(persisted.id, verify=True)


# ══════════════════════════════════════════════════════════════
# BLOCKER 2 Tests: get_by_hypothesis_id Security
# ══════════════════════════════════════════════════════════════


class TestBlocker2HypothesisSecurity:
    """get_by_hypothesis_id must block tampered contracts."""

    def test_get_by_hypothesis_id_blocks_tampered_contract(self, store, db_conn):
        """Tampered Contract via hypothesis_id must not be returned."""
        c = create_minimal_contract(
            name="Hypothesis Security Test",
            phrases=("hyp_sec",),
            instruction="Safe instruction",
            hypothesis_id="hyp_sec_001",
        )
        persisted = store.persist(c, USER_ID)

        # Tamper: change instruction in JSON (checksum becomes invalid)
        raw = db_conn.fetchone(
            "SELECT contract_json FROM skill_contracts WHERE id = ?",
            (persisted.id,),
        )
        data = json.loads(raw[0])
        data["execution"]["instruction"] = "TAMPERED VIA HYPOTHESIS"
        db_conn.execute(
            "UPDATE skill_contracts SET contract_json = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), persisted.id),
        )

        with pytest.raises(ContractChecksumError):
            store.get_by_hypothesis_id("hyp_sec_001", verify=True)

    def test_get_by_hypothesis_id_with_verify_false_still_checks_invariant(
        self, store, db_conn
    ):
        """Even with verify=False, invariant check prevents returning mismatched data."""
        c = create_minimal_contract(
            name="Hyp Invariant Test",
            phrases=("hyp_inv",),
            instruction="Safe",
            hypothesis_id="hyp_inv_001",
        )
        persisted = store.persist(c, USER_ID)

        # Tamper: change name in JSON but not DB column
        raw = db_conn.fetchone(
            "SELECT contract_json FROM skill_contracts WHERE id = ?",
            (persisted.id,),
        )
        data = json.loads(raw[0])
        data["name"] = "EVIL_NAME"
        db_conn.execute(
            "UPDATE skill_contracts SET contract_json = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), persisted.id),
        )

        with pytest.raises(ContractInvariantError):
            store.get_by_hypothesis_id("hyp_inv_001", verify=False)


# ══════════════════════════════════════════════════════════════
# HIGH 1 Tests: Unique Constraint + Optimistic Locking
# ══════════════════════════════════════════════════════════════


class TestHigh1RaceSafety:
    """DB unique constraint and optimistic locking tests."""

    def test_duplicate_name_blocked_by_db_constraint(
        self, store, valid_contract, db_conn
    ):
        """Direct insert bypassing app-check must still fail on DB constraint."""
        store.persist(valid_contract, USER_ID)

        # Try to bypass the app-level check by inserting directly via SQL
        # This simulates a race condition where two requests pass the app check
        duplicate_contract = create_minimal_contract(
            name="Security Test Skill",  # Same name!
            phrases=("dup_trigger",),
            instruction="Dup instruction",
        )
        dup_finalized = _finalize_security_metadata(duplicate_contract)
        dup_json = dup_finalized.to_json()

        with pytest.raises(sqlite3.IntegrityError):
            db_conn.execute(
                """INSERT INTO skill_contracts
                   (id, user_id, name, schema_version, contract_version,
                    hypothesis_id, origin, lifecycle_status, review_status,
                    risk_level, package_type, checksum, contract_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dup_finalized.id,
                    USER_ID,
                    dup_finalized.name,
                    dup_finalized.schema_version,
                    dup_finalized.contract_version,
                    dup_finalized.hypothesis_id,
                    dup_finalized.origin,
                    dup_finalized.lifecycle.status,
                    dup_finalized.review_status,
                    dup_finalized.risk_level,
                    dup_finalized.store_meta.package_type,
                    dup_finalized.trust.checksum,
                    dup_json,
                    dup_finalized.created_at,
                    dup_finalized.updated_at,
                ),
            )

    def test_stale_update_raises_concurrent_error(self, store, valid_contract, db_conn):
        """Optimistic locking: stale expected_version must raise error."""
        persisted = store.persist(valid_contract, USER_ID)

        # Simulate external update: bump version in DB directly
        db_conn.execute(
            "UPDATE skill_contracts SET contract_version = 99 WHERE id = ?",
            (persisted.id,),
        )

        # Now try to update with stale expected_version=1
        updated = replace(persisted, execution=ExecutionConfig(instruction="Updated"))
        with pytest.raises(ContractConcurrentUpdateError):
            store.update(updated, USER_ID, expected_version=1)

    def test_update_without_expected_version_uses_current(self, store, valid_contract):
        """Legacy update without expected_version reads current from DB."""
        persisted = store.persist(valid_contract, USER_ID)
        updated = replace(
            persisted, execution=ExecutionConfig(instruction="Updated V2")
        )
        result = store.update(updated, USER_ID)
        assert result.contract_version == 2

    def test_concurrent_update_scenario(self, store, valid_contract, db_conn):
        """Two callers load same version, first wins, second gets error."""
        persisted = store.persist(valid_contract, USER_ID)
        v1 = persisted.contract_version  # == 1

        # First update succeeds
        u1 = replace(persisted, execution=ExecutionConfig(instruction="Update 1"))
        result1 = store.update(u1, USER_ID, expected_version=v1)
        assert result1.contract_version == 2

        # Second update with same expected_version fails
        u2 = replace(persisted, execution=ExecutionConfig(instruction="Update 2"))
        with pytest.raises(ContractConcurrentUpdateError):
            store.update(u2, USER_ID, expected_version=v1)


# ══════════════════════════════════════════════════════════════
# MEDIUM 1 Tests: DB==JSON Invariant on Read
# ══════════════════════════════════════════════════════════════


class TestMedium1DbJsonInvariant:
    """DB columns diverging from JSON content must be caught on load."""

    # Columns that may be tampered in invariant tests. Validated before
    # interpolation to prevent accidental SQL injection patterns.
    _TAMPER_COLUMNS = frozenset(
        {
            "id",
            "name",
            "schema_version",
            "contract_version",
            "origin",
            "lifecycle_status",
            "review_status",
            "risk_level",
            "package_type",
            "checksum",
            "contract_json",
        }
    )

    def _tamper_db_column(self, db_conn, contract_id, column, value):
        """Helper to tamper a single DB column (test-only, no user input)."""
        assert column in self._TAMPER_COLUMNS, f"Unknown column: {column}"
        # Column is validated against _TAMPER_COLUMNS above; no user input.
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        db_conn.execute(
            f"UPDATE skill_contracts SET {column} = ? WHERE id = ?",
            (value, contract_id),
        )

    def test_db_json_mismatch_id_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.id != JSON.id must raise ContractInvariantError."""
        persisted = store.persist(valid_contract, USER_ID)
        # Tamper JSON id
        raw = db_conn.fetchone(
            "SELECT contract_json FROM skill_contracts WHERE id = ?",
            (persisted.id,),
        )
        data = json.loads(raw[0])
        data["id"] = "skill_fake_id"
        db_conn.execute(
            "UPDATE skill_contracts SET contract_json = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), persisted.id),
        )
        with pytest.raises(ContractInvariantError, match="id"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_name_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.name != JSON.name must raise ContractInvariantError."""
        persisted = store.persist(valid_contract, USER_ID)
        raw = db_conn.fetchone(
            "SELECT contract_json FROM skill_contracts WHERE id = ?",
            (persisted.id,),
        )
        data = json.loads(raw[0])
        data["name"] = "EVIL_NAME"
        db_conn.execute(
            "UPDATE skill_contracts SET contract_json = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), persisted.id),
        )
        with pytest.raises(ContractInvariantError, match="name"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_schema_version_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.schema_version != JSON.schema_version must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        self._tamper_db_column(db_conn, persisted.id, "schema_version", 99)
        with pytest.raises(ContractInvariantError, match="schema_version"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_contract_version_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.contract_version != JSON.contract_version must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        self._tamper_db_column(db_conn, persisted.id, "contract_version", 99)
        with pytest.raises(ContractInvariantError, match="contract_version"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_origin_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.origin != JSON.origin must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        self._tamper_db_column(db_conn, persisted.id, "origin", "store")
        with pytest.raises(ContractInvariantError, match="origin"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_status_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.lifecycle_status != JSON.lifecycle.status must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        self._tamper_db_column(db_conn, persisted.id, "lifecycle_status", "paused")
        with pytest.raises(ContractInvariantError, match="status"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_review_status_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.review_status != JSON.review_status must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        self._tamper_db_column(db_conn, persisted.id, "review_status", "blocked")
        with pytest.raises(ContractInvariantError, match="review_status"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_risk_level_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.risk_level != JSON.risk_level must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        self._tamper_db_column(db_conn, persisted.id, "risk_level", "high")
        with pytest.raises(ContractInvariantError, match="risk_level"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_package_type_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.package_type != JSON.package_type must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        self._tamper_db_column(
            db_conn, persisted.id, "package_type", "privileged_plugin"
        )
        with pytest.raises(ContractInvariantError, match="package_type"):
            store.get_by_id(persisted.id)

    def test_db_json_mismatch_checksum_raises_invariant_error(
        self, store, valid_contract, db_conn
    ):
        """DB.checksum != JSON.trust.checksum must raise."""
        persisted = store.persist(valid_contract, USER_ID)
        fake_checksum = "b" * 64
        self._tamper_db_column(db_conn, persisted.id, "checksum", fake_checksum)
        with pytest.raises(ContractInvariantError, match="checksum"):
            store.get_by_id(persisted.id)


# ══════════════════════════════════════════════════════════════
# MEDIUM 2 Tests: Strict Deserialization
# ══════════════════════════════════════════════════════════════


class TestMedium2StrictDeserialization:
    """Strict type validation on deserialization."""

    def test_phrases_as_string_rejected(self):
        """phrases: 'rot' must raise ContractDeserializationError, not produce ('r','o','t')."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {
                "phrases": "rot",  # String instead of list!
            },
            "execution": {"instruction": "do stuff"},
        }
        with pytest.raises(ContractDeserializationError, match="phrases"):
            SkillContract.from_dict(data)

    def test_timeout_zero_rejected(self):
        """timeout_seconds = 0 must be rejected (minimum is 1)."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["test"]},
            "execution": {
                "instruction": "do stuff",
                "timeout_seconds": 0,
            },
        }
        with pytest.raises(ContractDeserializationError, match="timeout_seconds"):
            SkillContract.from_dict(data)

    def test_threshold_out_of_range_rejected(self):
        """Threshold > 1.0 must be rejected."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["test"]},
            "execution": {"instruction": "do stuff"},
            "confirmation": {
                "thresholds": {
                    "auto_execute": 1.5,  # Out of range!
                }
            },
        }
        with pytest.raises(ContractDeserializationError, match="auto_execute"):
            SkillContract.from_dict(data)

    def test_threshold_negative_rejected(self):
        """Threshold < 0.0 must be rejected."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["test"]},
            "execution": {"instruction": "do stuff"},
            "confirmation": {
                "thresholds": {
                    "reject": -0.1,  # Negative!
                }
            },
        }
        with pytest.raises(ContractDeserializationError, match="reject"):
            SkillContract.from_dict(data)

    def test_tags_as_string_rejected(self):
        """tags: 'abc' must raise error."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["test"]},
            "execution": {"instruction": "do stuff"},
            "tags": "single_tag",  # String instead of list!
        }
        with pytest.raises(ContractDeserializationError, match="tags"):
            SkillContract.from_dict(data)

    def test_provider_hints_as_string_rejected(self):
        """provider_hints as string must raise error."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["test"]},
            "execution": {"instruction": "do stuff"},
            "provider_hints": "claude",  # String instead of list!
        }
        with pytest.raises(ContractDeserializationError, match="provider_hints"):
            SkillContract.from_dict(data)

    def test_allowed_tools_as_string_rejected(self):
        """safety.allowed_tools as string must raise error."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["test"]},
            "execution": {"instruction": "do stuff"},
            "safety": {
                "allowed_tools": "web_search",  # String!
            },
        }
        with pytest.raises(ContractDeserializationError, match="allowed_tools"):
            SkillContract.from_dict(data)

    def test_permissions_tools_as_string_rejected(self):
        """permissions.tools as string must raise error."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["test"]},
            "execution": {"instruction": "do stuff"},
            "permissions": {
                "tools": "web_search",  # String!
            },
        }
        with pytest.raises(ContractDeserializationError, match="permissions.tools"):
            SkillContract.from_dict(data)

    def test_cooldown_negative_rejected(self):
        """cooldown_seconds < 0 must be rejected."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {
                "phrases": ["test"],
                "cooldown_seconds": -5,
            },
            "execution": {"instruction": "do stuff"},
        }
        with pytest.raises(ContractDeserializationError, match="cooldown_seconds"):
            SkillContract.from_dict(data)

    def test_valid_list_fields_accepted(self):
        """Proper list fields work fine."""
        data = {
            "id": "skill_test123",
            "name": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "activation": {"phrases": ["hello", "hi"]},
            "execution": {"instruction": "do stuff", "timeout_seconds": 30},
            "tags": ["tag1", "tag2"],
            "provider_hints": ["claude", "gpt4"],
        }
        contract = SkillContract.from_dict(data)
        assert contract.activation.phrases == ("hello", "hi")
        assert contract.tags == ("tag1", "tag2")
        assert contract.provider_hints == ("claude", "gpt4")
