"""Tests for Skill Versioning (HC-SC-12), Commit 4.2.

Covers:
  - create_new_version creates new Hypothesis with version+1
  - Old hypothesis archived in hypothesis_versions with deprecated_at
  - Old evidence stays with old version (no move)
  - predecessor_context stored as reference
  - New version starts with elo=1500, support=0
  - New version status = 'suggested'
  - get_version_history returns ordered history
"""

from __future__ import annotations

import sqlite3

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


class FakeDBConnection:
    """In-memory SQLite connection for tests."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql, params=(), **kwargs):
        return self._conn.execute(sql, params)

    def executescript(self, sql):
        self._conn.executescript(sql)

    def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations):
        for sql, params in operations:
            self._conn.execute(sql, params)
        self._conn.commit()


def _setup_storage() -> HypothesisStorage:
    """Create an in-memory HypothesisStorage with schema."""
    conn = FakeDBConnection()
    storage = HypothesisStorage(conn)
    storage.init_schema()
    return storage


def _make_hypothesis(
    *,
    hypothesis_id: str = "hyp-ver-001",
    user_id: int = 42,
    status: str = "confirmed",
    claim: str = "30s Retargeting Drehkonzept",
    version: int = 1,
    elo_rating: float = 1700.0,
    support_count: int = 5,
    contradict_count: int = 1,
) -> Hypothesis:
    """Create a test hypothesis for versioning."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=user_id,
        type="preference",
        scope=HypothesisScope(project="client_ads", client="honey-brand"),
        claim=claim,
        status=status,
        version=version,
        elo_rating=elo_rating,
        support_count=support_count,
        contradict_count=contradict_count,
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T12:00:00+00:00",
    )


# ---------------------------------------------------------------
# Tests: create_new_version
# ---------------------------------------------------------------


class TestCreateNewVersion:
    """Tests for HypothesisStorage.create_new_version."""

    def test_creates_version_plus_one(self) -> None:
        """New version should have version = old_version + 1."""
        storage = _setup_storage()
        hyp = _make_hypothesis(version=1)
        storage.insert_hypothesis(hyp)

        new_hyp = storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="45s Brand Awareness Drehkonzept",
            change_reason="3 Korrekturen Richtung 60s",
        )

        assert new_hyp is not None
        assert new_hyp.version == 2

    def test_new_version_has_new_claim(self) -> None:
        """New version should use the provided new_claim."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        new_hyp = storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="45s Brand Awareness",
            change_reason="User corrected",
        )

        assert new_hyp is not None
        assert new_hyp.claim == "45s Brand Awareness"

    def test_new_version_resets_elo(self) -> None:
        """New version must start with elo=1500 (fresh confidence)."""
        storage = _setup_storage()
        hyp = _make_hypothesis(elo_rating=1800.0)
        storage.insert_hypothesis(hyp)

        new_hyp = storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="Updated claim",
            change_reason="test",
        )

        assert new_hyp is not None
        assert new_hyp.elo_rating == 1500.0

    def test_new_version_resets_support_count(self) -> None:
        """New version must start with support_count=0."""
        storage = _setup_storage()
        hyp = _make_hypothesis(support_count=10)
        storage.insert_hypothesis(hyp)

        new_hyp = storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="Updated claim",
            change_reason="test",
        )

        assert new_hyp is not None
        assert new_hyp.support_count == 0
        assert new_hyp.contradict_count == 0

    def test_new_version_status_is_suggested(self) -> None:
        """New version status = 'suggested' (IC-VERSION-1: normal lifecycle)."""
        storage = _setup_storage()
        hyp = _make_hypothesis(status="active")
        storage.insert_hypothesis(hyp)

        new_hyp = storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="Updated",
            change_reason="test",
        )

        assert new_hyp is not None
        assert new_hyp.status == "suggested"

    def test_old_version_archived_in_versions_table(self) -> None:
        """Old version must be stored in hypothesis_versions."""
        storage = _setup_storage()
        hyp = _make_hypothesis(elo_rating=1750.0)
        storage.insert_hypothesis(hyp)

        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="New claim",
            change_reason="Drift detected",
        )

        history = storage.get_version_history("hyp-ver-001")
        assert len(history) == 1
        assert history[0]["version"] == 1
        assert history[0]["claim"] == "30s Retargeting Drehkonzept"
        assert history[0]["elo_rating_at_save"] == 1750.0

    def test_old_version_has_deprecated_at(self) -> None:
        """Archived version must have deprecated_at set to current time."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="New",
            change_reason="test",
        )

        history = storage.get_version_history("hyp-ver-001")
        assert len(history) == 1
        assert history[0]["deprecated_at"] is not None
        # ISO format check
        assert "2026-" in history[0]["deprecated_at"]

    def test_predecessor_context_stored(self) -> None:
        """predecessor_context must be stored with the archived version."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="New claim",
            change_reason="Explicit context test",
            predecessor_context="v1 hatte 5 Belege aus 3 Sessions",
        )

        history = storage.get_version_history("hyp-ver-001")
        assert len(history) == 1
        assert history[0]["predecessor_context"] == "v1 hatte 5 Belege aus 3 Sessions"

    def test_auto_generated_predecessor_context(self) -> None:
        """When predecessor_context is None, it should be auto-generated."""
        storage = _setup_storage()
        hyp = _make_hypothesis(elo_rating=1700.0, support_count=5)
        storage.insert_hypothesis(hyp)

        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="New",
            change_reason="test",
            predecessor_context=None,
        )

        history = storage.get_version_history("hyp-ver-001")
        ctx = history[0]["predecessor_context"]
        assert ctx is not None
        assert "v1" in ctx
        assert "1700" in ctx

    def test_evidence_stays_with_old_version(self) -> None:
        """Old evidence must remain linked to old version (HC-SC-12)."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        # Add evidence for version 1
        storage.insert_evidence(
            evidence_id="ev-001",
            hypothesis_id="hyp-ver-001",
            hypothesis_version=1,
            signal_type="no_correction",
            signal_strength=1.0,
            created_at="2026-05-20T11:00:00+00:00",
        )

        # Create new version
        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="New claim",
            change_reason="test",
        )

        # Evidence still references version 1, NOT version 2
        evidence = storage.get_evidence_for_hypothesis("hyp-ver-001", version=1)
        assert len(evidence) == 1
        assert evidence[0]["hypothesis_version"] == 1

        # No evidence for version 2 yet
        evidence_v2 = storage.get_evidence_for_hypothesis("hyp-ver-001", version=2)
        assert len(evidence_v2) == 0

    def test_nonexistent_hypothesis_returns_none(self) -> None:
        """create_new_version on non-existent ID returns None."""
        storage = _setup_storage()
        result = storage.create_new_version(
            hypothesis_id="nonexistent",
            new_claim="test",
            change_reason="test",
        )
        assert result is None

    def test_multiple_versions(self) -> None:
        """Multiple version bumps should accumulate in history."""
        storage = _setup_storage()
        hyp = _make_hypothesis(version=1)
        storage.insert_hypothesis(hyp)

        # v1 -> v2
        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="v2 claim",
            change_reason="first update",
        )

        # v2 -> v3
        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="v3 claim",
            change_reason="second update",
        )

        current = storage.get_hypothesis("hyp-ver-001")
        assert current is not None
        assert current.version == 3
        assert current.claim == "v3 claim"

        # History should have 2 entries (v1 and v2 archived)
        history = storage.get_version_history("hyp-ver-001")
        assert len(history) == 2
        versions_in_history = [h["version"] for h in history]
        assert 1 in versions_in_history
        assert 2 in versions_in_history


# ---------------------------------------------------------------
# Tests: get_version_history
# ---------------------------------------------------------------


class TestGetVersionHistory:
    """Tests for version history retrieval."""

    def test_empty_history_for_new_hypothesis(self) -> None:
        """Fresh hypothesis should have no version history."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        history = storage.get_version_history("hyp-ver-001")
        assert history == []

    def test_history_ordered_by_version_desc(self) -> None:
        """History should be ordered newest version first."""
        storage = _setup_storage()
        hyp = _make_hypothesis(version=1)
        storage.insert_hypothesis(hyp)

        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="v2",
            change_reason="update 1",
        )
        storage.create_new_version(
            hypothesis_id="hyp-ver-001",
            new_claim="v3",
            change_reason="update 2",
        )

        history = storage.get_version_history("hyp-ver-001")
        assert len(history) == 2
        # Newest first (version DESC)
        assert history[0]["version"] >= history[1]["version"]
