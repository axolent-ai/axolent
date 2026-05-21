"""Tests for LocalEvalSet (Step 6).

Covers:
  - add_example + evaluate round-trip
  - Smoke-test failed -> status not promoted
  - Max 5 examples (HC-EVAL-1)
  - get_examples returns stored examples
  - evaluate with no examples returns passed=True (skip)
  - evaluate with hypothesis not found returns passed=True (skip)
  - remove_example works
  - _basic_match heuristic
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
from application.skill_compression.local_eval_set import (
    MAX_EXAMPLES_PER_HYPOTHESIS,
    LocalEvalSet,
    _basic_match,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


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


def _make_storage() -> HypothesisStorage:
    """Create an in-memory HypothesisStorage with schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    storage = HypothesisStorage(_FakeConn(conn))
    storage.init_schema()
    return storage


def _insert_test_hypothesis(
    storage: HypothesisStorage,
    hyp_id: str = "hyp_eval_001",
    claim: str = "User bevorzugt Bulletpoints in Zusammenfassungen",
) -> Hypothesis:
    """Insert a test hypothesis."""
    now = datetime.now(timezone.utc).isoformat()
    hyp = Hypothesis(
        hypothesis_id=hyp_id,
        user_id=1,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status="confirmed",
        version=1,
        elo_rating=1700.0,
        bayes_confidence=0.75,
        support_count=5,
        created_at=now,
        last_seen=now,
    )
    storage.insert_hypothesis(hyp)
    return hyp


# ---------------------------------------------------------------
# Tests: add + evaluate round-trip
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestLocalEvalSetRoundTrip:
    """add_example + evaluate should work end-to-end."""

    def test_add_and_evaluate_passing(self) -> None:
        """Adding a matching example produces a passing eval."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        eval_id = eval_set.add_example(
            "hyp_eval_001",
            "Fasse den Text zusammen",
            "Bulletpoints in Zusammenfassungen",
        )

        assert eval_id is not None

        result = eval_set.evaluate("hyp_eval_001")
        assert result.passed is True
        assert result.total_examples == 1
        assert result.passed_count == 1
        assert result.failed_count == 0

    def test_add_and_evaluate_failing(self) -> None:
        """Adding a non-matching example produces a failing eval."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        eval_set.add_example(
            "hyp_eval_001",
            "Some unrelated input",
            "Completely unrelated expected output xyz123",
        )

        result = eval_set.evaluate("hyp_eval_001")
        assert result.passed is False
        assert result.failed_count == 1

    def test_mixed_results(self) -> None:
        """One passing and one failing example -> overall fail."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        eval_set.add_example(
            "hyp_eval_001", "input1", "Bulletpoints in Zusammenfassungen"
        )
        eval_set.add_example("hyp_eval_001", "input2", "completely unrelated xyz999")

        result = eval_set.evaluate("hyp_eval_001")
        assert result.passed is False
        assert result.passed_count == 1
        assert result.failed_count == 1
        assert result.total_examples == 2


# ---------------------------------------------------------------
# Tests: Max 5 examples (HC-EVAL-1)
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestLocalEvalSetMaxExamples:
    """HC-EVAL-1: Max 5 examples per hypothesis."""

    def test_max_5_enforced(self) -> None:
        """Adding a 6th example returns None."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        for i in range(MAX_EXAMPLES_PER_HYPOTHESIS):
            result = eval_set.add_example("hyp_eval_001", f"input_{i}", f"output_{i}")
            assert result is not None

        # 6th should fail
        result = eval_set.add_example("hyp_eval_001", "input_6", "output_6")
        assert result is None

    def test_count_examples(self) -> None:
        """count_examples returns correct count."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        assert eval_set.count_examples("hyp_eval_001") == 0

        eval_set.add_example("hyp_eval_001", "in1", "out1")
        assert eval_set.count_examples("hyp_eval_001") == 1

        eval_set.add_example("hyp_eval_001", "in2", "out2")
        assert eval_set.count_examples("hyp_eval_001") == 2


# ---------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestLocalEvalSetEdgeCases:
    """Edge cases for evaluation."""

    def test_evaluate_no_examples_passes(self) -> None:
        """No examples in eval set -> passed=True (skip)."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        result = eval_set.evaluate("hyp_eval_001")
        assert result.passed is True
        assert result.skip_reason == "No examples in eval set"

    def test_evaluate_hypothesis_not_found(self) -> None:
        """Hypothesis not found -> passed=True (skip)."""
        storage = _make_storage()
        eval_set = LocalEvalSet(storage)

        result = eval_set.evaluate("nonexistent")
        assert result.passed is True
        assert result.skip_reason == "Hypothesis not found"

    def test_get_examples_returns_stored(self) -> None:
        """get_examples returns all stored examples."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        eval_set.add_example("hyp_eval_001", "in1", "out1")
        eval_set.add_example("hyp_eval_001", "in2", "out2")

        examples = eval_set.get_examples("hyp_eval_001")
        assert len(examples) == 2
        inputs = {e.example_input for e in examples}
        assert inputs == {"in1", "in2"}

    def test_remove_example(self) -> None:
        """remove_example deletes from DB."""
        storage = _make_storage()
        _insert_test_hypothesis(storage)
        eval_set = LocalEvalSet(storage)

        eval_id = eval_set.add_example("hyp_eval_001", "in1", "out1")
        assert eval_set.count_examples("hyp_eval_001") == 1

        eval_set.remove_example(eval_id)
        assert eval_set.count_examples("hyp_eval_001") == 0


# ---------------------------------------------------------------
# Tests: _basic_match heuristic
# ---------------------------------------------------------------


@pytest.mark.skill_compression
class TestBasicMatch:
    """Unit tests for the _basic_match heuristic."""

    def test_substring_match(self) -> None:
        """Substring in either direction matches."""
        assert _basic_match("Bulletpoints", "Bulletpoints in Zusammenfassungen") is True
        assert _basic_match("Bulletpoints in Zusammenfassungen", "Bulletpoints") is True

    def test_word_overlap_above_threshold(self) -> None:
        """Word overlap > 30% matches."""
        assert (
            _basic_match(
                "User bevorzugt Bulletpoints",
                "Bulletpoints bevorzugt in Listen",
            )
            is True
        )

    def test_no_match(self) -> None:
        """Completely different strings do not match."""
        assert (
            _basic_match(
                "completely different xyz123",
                "nothing in common abc789",
            )
            is False
        )

    def test_empty_strings(self) -> None:
        """Empty strings do not match."""
        assert _basic_match("", "something") is False
        assert _basic_match("something", "") is False
        assert _basic_match("", "") is False
