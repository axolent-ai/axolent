"""Tests for SkillExplainer (Step 6).

Covers all 8 question types (HC-SC-18, HC-EXPLAIN-1):
  1. WHAT_RECOGNIZED   - basic pattern info
  2. WHY_NOT_SKILL     - 5-Why for non-promotion
  3. WHY_PROMOTED      - promotion rationale
  4. WHEN_DRIFT        - contradiction timeline
  5. WHAT_NEEDED       - recommended actions
  6. LESSONS_LEARNED   - version history + lessons
  7. SCOPE_BOUNDARIES  - where skill does NOT apply
  8. COUNTER_EVIDENCE  - negative evidence listing

Plus:
  - "No data" responses when evidence is empty (HC-EXPLAIN-1)
  - Hypothesis not found
  - list_question_types returns all 8
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
from application.skill_compression.skill_explainer import (
    ExplainerQuestionType,
    ExplainerResponse,
    SkillExplainer,
    _describe_scope,
    _describe_status,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_storage() -> HypothesisStorage:
    """Create an in-memory HypothesisStorage with schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Wrap with a minimal compatible API
    storage = HypothesisStorage(_FakeConn(conn))
    storage.init_schema()
    return storage


class _FakeConn:
    """Minimal DB connection wrapper for sqlite3 in tests."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql, params=(), **kwargs):
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur

    def executescript(self, sql):
        self._conn.executescript(sql)
        self._conn.commit()

    def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations):
        for sql, params in operations:
            self._conn.execute(sql, params)
        self._conn.commit()


def _insert_test_hypothesis(
    storage: HypothesisStorage,
    *,
    hyp_id: str = "hyp_test_001",
    claim: str = "User bevorzugt Bulletpoints in Zusammenfassungen",
    status: str = "confirmed",
    hyp_type: str = "preference",
    elo: float = 1700.0,
    support: int = 5,
    contradict: int = 1,
    source_type: str = "live_chat",
    scope: HypothesisScope | None = None,
) -> Hypothesis:
    """Insert a test hypothesis and return it."""
    now = datetime.now(timezone.utc).isoformat()
    if scope is None:
        scope = HypothesisScope()

    hyp = Hypothesis(
        hypothesis_id=hyp_id,
        user_id=1,
        type=hyp_type,
        scope=scope,
        claim=claim,
        status=status,
        version=1,
        elo_rating=elo,
        bayes_confidence=0.75,
        support_count=support,
        contradict_count=contradict,
        source_type=source_type,
        decay_immune=source_type == "learn_command",
        created_at=now,
        last_seen=now,
    )
    storage.insert_hypothesis(hyp)
    return hyp


def _add_evidence(
    storage: HypothesisStorage,
    hyp_id: str,
    signal_type: str = "no_correction",
    count: int = 1,
) -> None:
    """Add evidence records to a hypothesis."""
    for i in range(count):
        now = datetime.now(timezone.utc).isoformat()
        storage.insert_evidence(
            evidence_id=f"ev_{hyp_id}_{signal_type}_{i}",
            hypothesis_id=hyp_id,
            hypothesis_version=1,
            signal_type=signal_type,
            signal_strength=1.0,
            created_at=now,
            episode_id=f"session_{i % 3}",
        )


# ---------------------------------------------------------------
# Tests: 8 question types
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestExplainerQuestionTypes:
    """All 8 question types return meaningful responses."""

    def test_type1_what_recognized(self) -> None:
        """Type 1: describes what the pattern recognized."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        _add_evidence(storage, "hyp_test_001", "no_correction", 3)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHAT_RECOGNIZED)

        assert resp.has_data is True
        assert resp.question_type == ExplainerQuestionType.WHAT_RECOGNIZED
        assert "Bulletpoints" in resp.explanation
        assert "preference" in resp.explanation
        assert resp.evidence_count == 3

    def test_type2_why_not_skill_candidate(self) -> None:
        """Type 2: explains why a candidate is not yet a skill."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, status="candidate", elo=1400.0, support=1)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHY_NOT_SKILL)

        assert resp.has_data is True
        assert "Zu wenige Belege" in resp.explanation or "Elo" in resp.explanation

    def test_type2_why_not_skill_already_confirmed(self) -> None:
        """Type 2: if already a skill, says so."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, status="confirmed")

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHY_NOT_SKILL)

        assert resp.has_data is True
        assert "Bereits ein Skill" in resp.title

    def test_type3_why_promoted(self) -> None:
        """Type 3: shows promotion rationale for confirmed skill."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, status="confirmed")
        _add_evidence(storage, "hyp_test_001", "no_correction", 5)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHY_PROMOTED)

        assert resp.has_data is True
        assert "Positive Belege" in resp.explanation
        assert "Schwelle" in resp.explanation

    def test_type3_why_promoted_not_yet(self) -> None:
        """Type 3: returns no-data for candidate that was never promoted."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, status="candidate")

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHY_PROMOTED)

        assert resp.has_data is False

    def test_type4_when_drift(self) -> None:
        """Type 4: shows contradiction timeline."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, status="needs_review")
        _add_evidence(storage, "hyp_test_001", "no_correction", 3)
        _add_evidence(storage, "hyp_test_001", "correction", 2)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHEN_DRIFT)

        assert resp.has_data is True
        assert "Drift-Timeline" in resp.explanation
        assert "Korrektur" in resp.explanation or "correction" in resp.explanation

    def test_type4_when_drift_no_contradictions(self) -> None:
        """Type 4: returns no-data when no contradictions exist."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        _add_evidence(storage, "hyp_test_001", "no_correction", 5)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHEN_DRIFT)

        assert resp.has_data is False
        assert "Kein Drift" in resp.explanation

    def test_type5_what_needed(self) -> None:
        """Type 5: recommends concrete actions."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, status="candidate", elo=1400.0, support=1)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHAT_NEEDED)

        assert resp.has_data is True
        assert "Massnahmen" in resp.explanation or "nötig" in resp.explanation.lower()

    def test_type6_lessons_learned(self) -> None:
        """Type 6: shows lessons from version history."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, source_type="learn_command")
        _add_evidence(storage, "hyp_test_001", "correction", 2)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.LESSONS_LEARNED)

        assert resp.has_data is True
        assert "/learn" in resp.explanation

    def test_type6_lessons_no_data(self) -> None:
        """Type 6: returns no-data for brand new pattern without lessons."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, source_type="live_chat")

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.LESSONS_LEARNED)

        # live_chat with no contradictions and no versions = no lessons
        assert resp.has_data is False

    def test_type7_scope_boundaries(self) -> None:
        """Type 7: describes scope boundaries."""
        storage = _make_storage()
        scope = HypothesisScope(project="ads", client="honey")
        _insert_test_hypothesis(storage, scope=scope)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.SCOPE_BOUNDARIES)

        assert resp.has_data is True
        assert "honey" in resp.explanation
        assert "ads" in resp.explanation
        assert "NICHT" in resp.explanation

    def test_type7_scope_global(self) -> None:
        """Type 7: global scope says no restrictions."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, scope=HypothesisScope())

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.SCOPE_BOUNDARIES)

        assert resp.has_data is True
        assert "global" in resp.explanation.lower()

    def test_type8_counter_evidence(self) -> None:
        """Type 8: lists negative evidence."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        _add_evidence(storage, "hyp_test_001", "no_correction", 3)
        _add_evidence(storage, "hyp_test_001", "correction", 2)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.COUNTER_EVIDENCE)

        assert resp.has_data is True
        assert "Gegenbelege" in resp.title
        assert "2 von 5" in resp.explanation

    def test_type8_counter_evidence_none(self) -> None:
        """Type 8: returns no-data when all evidence is positive."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        _add_evidence(storage, "hyp_test_001", "no_correction", 5)

        explainer = SkillExplainer(storage)
        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.COUNTER_EVIDENCE)

        assert resp.has_data is False
        assert "Keine Gegenbelege" in resp.explanation


# ---------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestExplainerEdgeCases:
    """Edge cases and no-data scenarios (HC-EXPLAIN-1)."""

    def test_hypothesis_not_found(self) -> None:
        """Unknown hypothesis ID returns not-found response."""
        storage = _make_storage()
        explainer = SkillExplainer(storage)

        resp = explainer.explain("nonexistent", ExplainerQuestionType.WHAT_RECOGNIZED)

        assert resp.has_data is False
        assert "nicht gefunden" in resp.explanation

    def test_all_8_types_callable(self) -> None:
        """All 8 ExplainerQuestionType values can be passed to explain()."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        explainer = SkillExplainer(storage)

        for qt in ExplainerQuestionType:
            resp = explainer.explain("hyp_test_001", qt)
            assert isinstance(resp, ExplainerResponse)
            assert resp.hypothesis_id == "hyp_test_001"

    def test_list_question_types_returns_8(self) -> None:
        """list_question_types returns exactly 8 entries."""
        storage = _make_storage()
        explainer = SkillExplainer(storage)

        types = explainer.list_question_types()
        assert len(types) == 8

    def test_empty_claim_what_recognized(self) -> None:
        """Type 1 with empty claim returns no-data."""
        storage = _make_storage()
        _insert_test_hypothesis(storage, claim="")
        explainer = SkillExplainer(storage)

        resp = explainer.explain("hyp_test_001", ExplainerQuestionType.WHAT_RECOGNIZED)
        assert resp.has_data is False


# ---------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestExplainerHelpers:
    """Test helper functions."""

    def test_describe_scope_global(self) -> None:
        """Global scope described correctly."""
        assert "Global" in _describe_scope(HypothesisScope())

    def test_describe_scope_specific(self) -> None:
        """Specific scope includes client and project."""
        desc = _describe_scope(HypothesisScope(project="ads", client="honey"))
        assert "honey" in desc
        assert "ads" in desc

    def test_describe_status_known(self) -> None:
        """Known statuses return German descriptions."""
        desc = _describe_status("confirmed")
        assert "Bestätigt" in desc

    def test_describe_status_unknown(self) -> None:
        """Unknown status returns the raw string."""
        assert _describe_status("xyzzy") == "xyzzy"
