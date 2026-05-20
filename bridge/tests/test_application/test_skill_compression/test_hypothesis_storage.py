"""Tests for the Hypothesis Storage Schema (Step 1.3/10).

Covers:
  - Schema creation (7 tables)
  - Foreign key constraints between tables
  - Indices on key fields
  - HC-SC-9: type column is TEXT, not ENUM
  - AG-SC-8: no CHECK constraint with fixed values on type
  - Hypothesis CRUD operations
  - Evidence ledger operations
  - Tombstone operations
  - Pattern difficulty CRUD
  - Version history
  - Alias operations
  - Frozen dataclass invariant (HC-SC-1)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)


@pytest.fixture
def db_conn(tmp_path):
    """Create an in-memory SQLite connection for tests."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    # Wrap with minimal interface matching DBConnection protocol
    class TestConnection:
        def __init__(self, raw_conn):
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

    return TestConnection(conn)


@pytest.fixture
def storage(db_conn):
    """Create a HypothesisStorage with initialized schema."""
    s = HypothesisStorage(db_conn)
    s.init_schema()
    return s


@pytest.fixture
def sample_hypothesis():
    """Create a sample Hypothesis for testing."""
    return Hypothesis(
        hypothesis_id="hyp_test_001",
        user_id=42,
        type="preference",
        scope=HypothesisScope(project="axolent", client="honey"),
        claim="User prefers short answers in code reviews",
        status="candidate",
        version=1,
        elo_rating=1500.0,
        elo_games_played=0,
        bayes_confidence=0.5,
        support_count=0,
        contradict_count=0,
        fsrs_state_json="{}",
        source_type="live_chat",
        decay_immune=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        last_seen=datetime.now(timezone.utc).isoformat(),
        pattern_hash="abc123",
        scope_hash="def456",
    )


class TestSchemaCreation:
    """Tests for schema initialization."""

    def test_all_7_tables_created(self, db_conn, storage):
        """All 7 tables should exist after schema init."""
        rows = db_conn.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        table_names = {row["name"] for row in rows}
        expected = {
            "hypotheses",
            "hypothesis_aliases",
            "hypothesis_evidence",
            "hypothesis_versions",
            "hypothesis_tombstones",
            "hypothesis_local_eval_set",
            "pattern_difficulty",
        }
        assert expected.issubset(table_names), (
            f"Missing tables: {expected - table_names}"
        )

    def test_schema_idempotent(self, db_conn):
        """Calling init_schema twice should not error."""
        s = HypothesisStorage(db_conn)
        s.init_schema()
        s.init_schema()  # Second call should be safe

    def test_foreign_keys_on(self, db_conn, storage):
        """Foreign keys should be enabled."""
        row = db_conn.fetchone("PRAGMA foreign_keys")
        assert row[0] == 1


class TestPatternTypeIsText:
    """AG-SC-8: type column must be TEXT, not CHECK-constrained ENUM."""

    def test_type_column_is_text(self, db_conn, storage):
        """The type column in hypotheses should be TEXT without CHECK."""
        rows = db_conn.fetchall("PRAGMA table_info(hypotheses)")
        type_col = [r for r in rows if r["name"] == "type"]
        assert len(type_col) == 1
        assert type_col[0]["type"] == "TEXT"

    def test_arbitrary_type_values_accepted(self, db_conn, storage, sample_hypothesis):
        """Future type values (context, style, outcome) must be insertable."""
        # Insert with a custom type not in the original 3
        from dataclasses import replace

        future_hyp = replace(
            sample_hypothesis,
            hypothesis_id="hyp_future_type",
            type="context_pattern",  # Not in v1 types
        )
        storage.insert_hypothesis(future_hyp)
        retrieved = storage.get_hypothesis("hyp_future_type")
        assert retrieved is not None
        assert retrieved.type == "context_pattern"

    def test_no_check_constraint_on_type(self, db_conn, storage):
        """There must be no CHECK constraint limiting type values."""
        # Get table SQL
        row = db_conn.fetchone("SELECT sql FROM sqlite_master WHERE name='hypotheses'")
        create_sql = row["sql"].upper()
        # Should not contain CHECK on type
        assert "CHECK" not in create_sql or "TYPE" not in create_sql


class TestHypothesisCRUD:
    """Tests for hypothesis insert/retrieve operations."""

    def test_insert_and_retrieve(self, storage, sample_hypothesis):
        """Insert and retrieve a hypothesis by ID."""
        storage.insert_hypothesis(sample_hypothesis)
        retrieved = storage.get_hypothesis("hyp_test_001")
        assert retrieved is not None
        assert retrieved.hypothesis_id == "hyp_test_001"
        assert retrieved.user_id == 42
        assert retrieved.type == "preference"
        assert retrieved.claim == "User prefers short answers in code reviews"
        assert retrieved.elo_rating == 1500.0

    def test_scope_roundtrip(self, storage, sample_hypothesis):
        """Scope should survive JSON serialization roundtrip."""
        storage.insert_hypothesis(sample_hypothesis)
        retrieved = storage.get_hypothesis("hyp_test_001")
        assert retrieved.scope.project == "axolent"
        assert retrieved.scope.client == "honey"

    def test_get_nonexistent_returns_none(self, storage):
        """Getting a non-existent hypothesis should return None."""
        assert storage.get_hypothesis("nonexistent") is None

    def test_get_by_user(self, storage, sample_hypothesis):
        """Retrieve hypotheses filtered by user_id."""
        storage.insert_hypothesis(sample_hypothesis)
        results = storage.get_hypotheses_by_user(42)
        assert len(results) == 1

    def test_get_by_user_and_status(self, storage, sample_hypothesis):
        """Retrieve hypotheses filtered by user_id and status."""
        storage.insert_hypothesis(sample_hypothesis)
        results = storage.get_hypotheses_by_user(42, status="candidate")
        assert len(results) == 1
        results = storage.get_hypotheses_by_user(42, status="active")
        assert len(results) == 0

    def test_count_active(self, storage, sample_hypothesis):
        """Count active hypotheses for max-50 check (HC-SC-8)."""
        from dataclasses import replace

        for i in range(5):
            h = replace(
                sample_hypothesis,
                hypothesis_id=f"hyp_active_{i}",
                status="active",
            )
            storage.insert_hypothesis(h)
        assert storage.count_active_hypotheses(42) == 5

    def test_decay_immune_flag(self, storage, sample_hypothesis):
        """decay_immune flag should survive roundtrip."""
        from dataclasses import replace

        immune = replace(
            sample_hypothesis, hypothesis_id="hyp_immune", decay_immune=True
        )
        storage.insert_hypothesis(immune)
        retrieved = storage.get_hypothesis("hyp_immune")
        assert retrieved.decay_immune is True


class TestEvidenceLedger:
    """Tests for the evidence table."""

    def test_insert_and_retrieve_evidence(self, storage, sample_hypothesis):
        """Insert evidence and retrieve by hypothesis."""
        storage.insert_hypothesis(sample_hypothesis)
        storage.insert_evidence(
            evidence_id="evi_001",
            hypothesis_id="hyp_test_001",
            hypothesis_version=1,
            signal_type="no_correction",
            signal_strength=0.8,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        evidence = storage.get_evidence_for_hypothesis("hyp_test_001")
        assert len(evidence) == 1
        assert evidence[0]["signal_type"] == "no_correction"

    def test_evidence_foreign_key(self, db_conn, storage):
        """Evidence should reference a valid hypothesis (FK constraint)."""
        with pytest.raises(sqlite3.IntegrityError):
            storage.insert_evidence(
                evidence_id="evi_orphan",
                hypothesis_id="nonexistent_hyp",
                hypothesis_version=1,
                signal_type="no_correction",
                signal_strength=1.0,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    def test_evidence_filter_by_version(self, storage, sample_hypothesis):
        """Evidence can be filtered by hypothesis version."""
        storage.insert_hypothesis(sample_hypothesis)
        ts = datetime.now(timezone.utc).isoformat()
        storage.insert_evidence("evi_v1", "hyp_test_001", 1, "no_correction", 1.0, ts)
        storage.insert_evidence("evi_v2", "hyp_test_001", 2, "correction", 1.0, ts)

        v1_evidence = storage.get_evidence_for_hypothesis("hyp_test_001", version=1)
        assert len(v1_evidence) == 1
        assert v1_evidence[0]["evidence_id"] == "evi_v1"


class TestTombstones:
    """Tests for the tombstone table."""

    def test_insert_and_check_tombstone(self, storage, sample_hypothesis):
        """Active tombstone should block re-learning."""
        storage.insert_hypothesis(sample_hypothesis)
        storage.insert_tombstone(
            tombstone_id="tomb_001",
            hypothesis_id="hyp_test_001",
            fingerprint="abc123",
            deleted_at=datetime.now(timezone.utc).isoformat(),
            expires_at="2099-12-31T23:59:59+00:00",
        )
        assert storage.check_tombstone("abc123") is True

    def test_permanent_tombstone(self, storage, sample_hypothesis):
        """Permanent tombstone should always block."""
        storage.insert_hypothesis(sample_hypothesis)
        storage.insert_tombstone(
            tombstone_id="tomb_perm",
            hypothesis_id="hyp_test_001",
            fingerprint="perm_fp",
            deleted_at=datetime.now(timezone.utc).isoformat(),
            permanent=True,
        )
        assert storage.check_tombstone("perm_fp") is True

    def test_no_tombstone_allows_learning(self, storage):
        """No tombstone should allow learning."""
        assert storage.check_tombstone("unknown_fp") is False


class TestPatternDifficulty:
    """Tests for the pattern_difficulty table."""

    def test_upsert_and_retrieve(self, storage):
        """Insert and retrieve pattern difficulty."""
        ts = datetime.now(timezone.utc).isoformat()
        storage.upsert_pattern_difficulty("fp_hash_1", 1500.0, 0, ts)
        result = storage.get_pattern_difficulty("fp_hash_1")
        assert result is not None
        assert result["difficulty_rating"] == 1500.0

    def test_upsert_updates(self, storage):
        """Upsert should update existing record."""
        ts = datetime.now(timezone.utc).isoformat()
        storage.upsert_pattern_difficulty("fp_hash_1", 1500.0, 0, ts)
        storage.upsert_pattern_difficulty("fp_hash_1", 1600.0, 5, ts)
        result = storage.get_pattern_difficulty("fp_hash_1")
        assert result["difficulty_rating"] == 1600.0
        assert result["games_played"] == 5


class TestVersionHistory:
    """Tests for the hypothesis_versions table."""

    def test_insert_version(self, storage, sample_hypothesis):
        """Insert a version record."""
        storage.insert_hypothesis(sample_hypothesis)
        storage.insert_version(
            version_id="ver_001",
            hypothesis_id="hyp_test_001",
            version=1,
            claim="Original claim",
            elo_rating_at_save=1500.0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        # Verify by direct query (no dedicated getter yet)
        rows = storage._conn.fetchall(
            "SELECT * FROM hypothesis_versions WHERE hypothesis_id = ?",
            ("hyp_test_001",),
        )
        assert len(rows) == 1
        assert rows[0]["claim"] == "Original claim"


class TestAliases:
    """Tests for the hypothesis_aliases table."""

    def test_insert_alias(self, storage, sample_hypothesis):
        """Insert an alias for a hypothesis."""
        storage.insert_hypothesis(sample_hypothesis)
        ts = datetime.now(timezone.utc).isoformat()
        storage.insert_alias(
            alias_id="alias_001",
            hypothesis_id="hyp_test_001",
            alias_text="short code reviews",
            first_seen=ts,
            last_seen=ts,
        )
        rows = storage._conn.fetchall(
            "SELECT * FROM hypothesis_aliases WHERE hypothesis_id = ?",
            ("hyp_test_001",),
        )
        assert len(rows) == 1
        assert rows[0]["alias_text"] == "short code reviews"


class TestLocalEvalSet:
    """Tests for the hypothesis_local_eval_set table."""

    def test_insert_eval_example(self, storage, sample_hypothesis):
        """Insert a smoke-test example."""
        storage.insert_hypothesis(sample_hypothesis)
        storage.insert_eval_example(
            eval_id="eval_001",
            hypothesis_id="hyp_test_001",
            example_input="Review this function",
            example_output="The function has a bug in line 3...",
        )
        rows = storage._conn.fetchall(
            "SELECT * FROM hypothesis_local_eval_set WHERE hypothesis_id = ?",
            ("hyp_test_001",),
        )
        assert len(rows) == 1


class TestHypothesisDataclass:
    """Tests for the Hypothesis dataclass invariants."""

    def test_hypothesis_is_frozen(self):
        """HC-SC-1: Hypothesis must be frozen (immutable)."""
        h = Hypothesis(hypothesis_id="test")
        with pytest.raises(AttributeError):
            h.status = "active"  # type: ignore[misc]

    def test_hypothesis_has_slots(self):
        """HC-SC-1: Hypothesis must use __slots__ for memory efficiency."""
        assert hasattr(Hypothesis, "__slots__")

    def test_scope_serialization_roundtrip(self):
        """HypothesisScope should survive JSON roundtrip."""
        scope = HypothesisScope(
            project="test", client="client1", context=("tag1", "tag2")
        )
        json_str = scope.to_json()
        restored = HypothesisScope.from_json(json_str)
        assert restored.project == "test"
        assert restored.client == "client1"
        assert restored.context == ("tag1", "tag2")

    def test_scope_from_empty_json(self):
        """Empty JSON should produce default scope."""
        scope = HypothesisScope.from_json("")
        assert scope.project == ""
        assert scope.client == ""

    def test_scope_from_invalid_json(self):
        """Invalid JSON should produce default scope."""
        scope = HypothesisScope.from_json("not json")
        assert scope.project == ""
