"""Tests for the Elo Rating System (Step 2.3/10).

Covers:
  - Expected outcome calculation
  - Rating update mechanics (win/lose/draw)
  - HC-SC-4: initial rating 1500
  - Pattern with 1800 loses against 1600 = rating drops significantly
  - Symmetry: win and loss updates are inverse
  - Difficulty update (inverse of pattern update)
  - Rating confidence levels
  - K-factor behavior
  - EloUpdate dataclass immutability
  - Elo integration with HypothesisStorage
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from application.skill_compression.elo_rating import (
    DEFAULT_K_FACTOR,
    INITIAL_RATING,
    EloUpdate,
    compute_elo_update,
    expected_outcome,
    rating_confidence_level,
    update_difficulty,
    update_elo,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisStorage,
)


# ---------------------------------------------------------------
# Expected outcome tests
# ---------------------------------------------------------------


class TestExpectedOutcome:
    """Tests for the expected outcome formula."""

    def test_equal_ratings_gives_half(self):
        """Equal ratings should give expected outcome of 0.5."""
        result = expected_outcome(1500.0, 1500.0)
        assert abs(result - 0.5) < 0.001

    def test_higher_pattern_expects_win(self):
        """Higher pattern rating should expect win (E > 0.5)."""
        result = expected_outcome(1800.0, 1500.0)
        assert result > 0.5

    def test_lower_pattern_expects_loss(self):
        """Lower pattern rating should expect loss (E < 0.5)."""
        result = expected_outcome(1200.0, 1500.0)
        assert result < 0.5

    def test_extreme_difference(self):
        """Very high difference should approach 0 or 1."""
        high = expected_outcome(2500.0, 1000.0)
        low = expected_outcome(1000.0, 2500.0)
        assert high > 0.99
        assert low < 0.01

    def test_output_bounded_zero_one(self):
        """Expected outcome should always be in [0, 1]."""
        for pr in [0, 500, 1000, 1500, 2000, 3000, 5000]:
            for rd in [0, 500, 1000, 1500, 2000, 3000, 5000]:
                result = expected_outcome(float(pr), float(rd))
                assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------


class TestUpdateElo:
    """Tests for the core Elo update function."""

    def test_initial_rating_is_1500(self):
        """HC-SC-4: initial rating must be 1500."""
        assert INITIAL_RATING == 1500.0

    def test_win_increases_rating(self):
        """Winning should increase pattern rating."""
        new = update_elo(1500.0, 1500.0, "pattern_wins")
        assert new > 1500.0

    def test_loss_decreases_rating(self):
        """Losing should decrease pattern rating."""
        new = update_elo(1500.0, 1500.0, "pattern_loses")
        assert new < 1500.0

    def test_draw_no_change_at_equal(self):
        """Draw at equal ratings should not change rating."""
        new = update_elo(1500.0, 1500.0, "draw")
        assert abs(new - 1500.0) < 0.01

    def test_1800_loses_to_1600_drops_significantly(self):
        """Spec test: pattern with rating 1800 loses against difficulty
        1600. This is an upset (pattern expected to win), so rating
        should drop significantly.
        """
        new = update_elo(1800.0, 1600.0, "pattern_loses")
        drop = 1800.0 - new
        # Expected outcome for 1800 vs 1600 is ~0.76
        # Loss means actual=0.0, delta = 32 * (0.0 - 0.76) = -24.3
        assert drop > 20.0, f"Expected significant drop, got {drop:.1f}"
        assert drop < 32.0, f"Drop should not exceed K-factor, got {drop:.1f}"

    def test_easy_request_wrong_big_loss(self):
        """Easy request (low difficulty) answered wrong = big loss.

        Spec: "easy request wrong = big loss". Pattern at 1800
        loses to difficulty 1200 = very big drop.
        """
        new_easy_loss = update_elo(1800.0, 1200.0, "pattern_loses")
        new_hard_loss = update_elo(1800.0, 2200.0, "pattern_loses")
        drop_easy = 1800.0 - new_easy_loss
        drop_hard = 1800.0 - new_hard_loss
        assert drop_easy > drop_hard, (
            f"Easy loss ({drop_easy:.1f}) should hurt more than "
            f"hard loss ({drop_hard:.1f})"
        )

    def test_hard_request_wrong_small_loss(self):
        """Hard request (high difficulty) answered wrong = small loss.

        Pattern at 1500 losing to difficulty 2000 should not drop much
        because the loss was expected.
        """
        new = update_elo(1500.0, 2000.0, "pattern_loses")
        drop = 1500.0 - new
        assert drop < 10.0, f"Expected small drop, got {drop:.1f}"


class TestEloSymmetry:
    """Tests for the symmetry property of Elo updates."""

    def test_win_and_loss_are_inverse(self):
        """Win and loss updates should be inverse (symmetry).

        If a pattern at 1500 plays against difficulty 1500:
        - Win: delta = +16.0
        - Loss: delta = -16.0
        The magnitudes should be equal.
        """
        win_result = update_elo(1500.0, 1500.0, "pattern_wins")
        loss_result = update_elo(1500.0, 1500.0, "pattern_loses")
        win_delta = win_result - 1500.0
        loss_delta = 1500.0 - loss_result
        assert abs(win_delta - loss_delta) < 0.01, (
            f"Win delta ({win_delta:.2f}) and loss delta ({loss_delta:.2f}) "
            f"should be equal at equal ratings"
        )

    def test_symmetry_at_different_ratings(self):
        """Symmetry should hold at any rating pair."""
        for pattern_r in [1200, 1500, 1800]:
            for diff_r in [1200, 1500, 1800]:
                win = update_elo(float(pattern_r), float(diff_r), "pattern_wins")
                loss = update_elo(float(pattern_r), float(diff_r), "pattern_loses")
                win_delta = win - pattern_r
                loss_delta = pattern_r - loss
                # Win + loss deltas should sum to K
                assert abs(win_delta + loss_delta - DEFAULT_K_FACTOR) < 0.01


class TestKFactor:
    """Tests for K-factor behavior."""

    def test_default_k_factor_is_32(self):
        """IC-ELO-1: default K-factor should be 32."""
        assert DEFAULT_K_FACTOR == 32.0

    def test_higher_k_bigger_changes(self):
        """Higher K-factor should produce bigger rating changes."""
        small_k = update_elo(1500.0, 1500.0, "pattern_wins", k_factor=16.0)
        big_k = update_elo(1500.0, 1500.0, "pattern_wins", k_factor=64.0)
        delta_small = small_k - 1500.0
        delta_big = big_k - 1500.0
        assert delta_big > delta_small


# ---------------------------------------------------------------
# Difficulty update tests
# ---------------------------------------------------------------


class TestUpdateDifficulty:
    """Tests for request difficulty updates."""

    def test_pattern_wins_difficulty_drops(self):
        """When pattern wins, difficulty should drop (request was easy)."""
        new_diff = update_difficulty(1500.0, 1500.0, "pattern_wins")
        assert new_diff < 1500.0

    def test_pattern_loses_difficulty_rises(self):
        """When pattern loses, difficulty should rise (request was hard)."""
        new_diff = update_difficulty(1500.0, 1500.0, "pattern_loses")
        assert new_diff > 1500.0

    def test_difficulty_inverse_of_pattern(self):
        """Difficulty change should be inverse of pattern change."""
        new_pattern = update_elo(1500.0, 1500.0, "pattern_wins")
        new_difficulty = update_difficulty(1500.0, 1500.0, "pattern_wins")
        pattern_delta = new_pattern - 1500.0
        diff_delta = 1500.0 - new_difficulty
        assert abs(pattern_delta - diff_delta) < 0.01


# ---------------------------------------------------------------
# Compute Elo Update (structured result) tests
# ---------------------------------------------------------------


class TestComputeEloUpdate:
    """Tests for the compute_elo_update convenience function."""

    def test_returns_elo_update_record(self):
        """Should return a properly populated EloUpdate."""
        result = compute_elo_update(
            pattern_id="hyp_123",
            pattern_rating=1500.0,
            request_difficulty=1500.0,
            outcome="pattern_wins",
        )
        assert isinstance(result, EloUpdate)
        assert result.pattern_id == "hyp_123"
        assert result.old_rating == 1500.0
        assert result.new_rating > 1500.0
        assert result.outcome == "pattern_wins"
        assert result.k_factor == DEFAULT_K_FACTOR

    def test_old_and_new_rating_differ(self):
        """Old and new rating should differ after win/loss."""
        result = compute_elo_update("hyp_1", 1500.0, 1500.0, "pattern_wins")
        assert result.old_rating != result.new_rating


# ---------------------------------------------------------------
# Rating confidence level tests
# ---------------------------------------------------------------


class TestRatingConfidenceLevel:
    """Tests for the human-readable confidence classifier."""

    def test_high_confidence(self):
        assert rating_confidence_level(1800.0) == "high"
        assert rating_confidence_level(2000.0) == "high"

    def test_medium_confidence(self):
        assert rating_confidence_level(1650.0) == "medium"
        assert rating_confidence_level(1750.0) == "medium"

    def test_neutral_confidence(self):
        assert rating_confidence_level(1500.0) == "neutral"
        assert rating_confidence_level(1600.0) == "neutral"

    def test_low_confidence(self):
        assert rating_confidence_level(1350.0) == "low"
        assert rating_confidence_level(1400.0) == "low"

    def test_very_low_confidence(self):
        assert rating_confidence_level(1200.0) == "very_low"
        assert rating_confidence_level(1000.0) == "very_low"


# ---------------------------------------------------------------
# EloUpdate immutability tests
# ---------------------------------------------------------------


class TestEloUpdateFrozen:
    """Guard: EloUpdate must be immutable."""

    def test_elo_update_is_frozen(self):
        """EloUpdate should be frozen (immutable)."""
        u = EloUpdate(
            pattern_id="test",
            old_rating=1500.0,
            new_rating=1516.0,
            request_difficulty=1500.0,
            outcome="pattern_wins",
            k_factor=32.0,
        )
        with pytest.raises(AttributeError):
            u.new_rating = 9999.0  # type: ignore[misc]

    def test_elo_update_has_slots(self):
        """EloUpdate should use __slots__."""
        assert hasattr(EloUpdate, "__slots__")


# ---------------------------------------------------------------
# Integration: Elo + HypothesisStorage
# ---------------------------------------------------------------


class TestEloStorageIntegration:
    """Tests for Elo rating updates persisted via HypothesisStorage."""

    @pytest.fixture
    def db_conn(self, tmp_path):
        """Create an in-memory SQLite connection for tests."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")

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
    def storage(self, db_conn):
        """Create a HypothesisStorage with initialized schema."""
        s = HypothesisStorage(db_conn)
        s.init_schema()
        return s

    def test_update_hypothesis_elo(self, storage):
        """Elo rating update should persist in DB."""
        ts = datetime.now(timezone.utc).isoformat()
        h = Hypothesis(
            hypothesis_id="hyp_elo_test",
            user_id=42,
            type="preference",
            claim="Test Elo",
            elo_rating=1500.0,
            elo_games_played=0,
            created_at=ts,
            last_seen=ts,
        )
        storage.insert_hypothesis(h)

        # Simulate a win
        new_rating = update_elo(1500.0, 1500.0, "pattern_wins")
        storage.update_hypothesis_elo("hyp_elo_test", new_rating)

        retrieved = storage.get_hypothesis("hyp_elo_test")
        assert retrieved is not None
        assert retrieved.elo_rating == new_rating
        assert retrieved.elo_games_played == 1

    def test_multiple_elo_updates(self, storage):
        """Multiple Elo updates should accumulate correctly."""
        ts = datetime.now(timezone.utc).isoformat()
        h = Hypothesis(
            hypothesis_id="hyp_multi_elo",
            user_id=42,
            type="preference",
            claim="Test multiple Elo",
            elo_rating=1500.0,
            elo_games_played=0,
            created_at=ts,
            last_seen=ts,
        )
        storage.insert_hypothesis(h)

        # 3 wins
        rating = 1500.0
        for _ in range(3):
            rating = update_elo(rating, 1500.0, "pattern_wins")
            storage.update_hypothesis_elo("hyp_multi_elo", rating)

        retrieved = storage.get_hypothesis("hyp_multi_elo")
        assert retrieved is not None
        assert retrieved.elo_rating > 1500.0
        assert retrieved.elo_games_played == 3

    def test_pattern_difficulty_elo_update(self, storage):
        """Pattern difficulty should update via upsert."""
        ts = datetime.now(timezone.utc).isoformat()
        storage.upsert_pattern_difficulty("fp_test", 1500.0, 0, ts)

        new_diff = update_difficulty(1500.0, 1500.0, "pattern_wins")
        storage.upsert_pattern_difficulty("fp_test", new_diff, 1, ts)

        result = storage.get_pattern_difficulty("fp_test")
        assert result is not None
        assert result["difficulty_rating"] == new_diff
        assert result["games_played"] == 1

    def test_update_hypothesis_support(self, storage):
        """Support/contradict counts should update correctly."""
        ts = datetime.now(timezone.utc).isoformat()
        h = Hypothesis(
            hypothesis_id="hyp_support_test",
            user_id=42,
            type="preference",
            claim="Test support",
            created_at=ts,
            last_seen=ts,
        )
        storage.insert_hypothesis(h)

        storage.update_hypothesis_support(
            "hyp_support_test",
            increment_support=3,
            last_seen=ts,
        )
        retrieved = storage.get_hypothesis("hyp_support_test")
        assert retrieved.support_count == 3

        storage.update_hypothesis_support(
            "hyp_support_test",
            increment_contradict=1,
            last_contradiction_at=ts,
        )
        retrieved = storage.get_hypothesis("hyp_support_test")
        assert retrieved.contradict_count == 1
        assert retrieved.last_contradiction_at == ts

    def test_update_hypothesis_status(self, storage):
        """Status update should persist."""
        ts = datetime.now(timezone.utc).isoformat()
        h = Hypothesis(
            hypothesis_id="hyp_status_test",
            user_id=42,
            type="preference",
            claim="Test status",
            status="candidate",
            created_at=ts,
            last_seen=ts,
        )
        storage.insert_hypothesis(h)

        storage.update_hypothesis_status("hyp_status_test", "suggested")
        retrieved = storage.get_hypothesis("hyp_status_test")
        assert retrieved.status == "suggested"
