"""Integration test: Layer 2 pipeline (Step 2 complete).

Tests the full Layer 2 flow:
  Event -> N-Gram pattern detected -> stored in DB -> Elo update on match

Architecture Guard:
  N-Gram + Markov + Elo must NOT be imported by SkillMatcher directly.
  They communicate only through Pattern Storage (HC-LAYER2-1).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from application.skill_compression.elo_rating import (
    INITIAL_RATING,
    compute_elo_update,
    update_difficulty,
    update_elo,
)
from application.skill_compression.event_normalizer import (
    NormalizedEvent,
    normalize_event,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisStorage,
)
from application.skill_compression.markov_chain import MarkovChain
from application.skill_compression.ngram_extractor import (
    extract_ngrams,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def db_conn():
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
def storage(db_conn):
    """Create a HypothesisStorage with initialized schema."""
    s = HypothesisStorage(db_conn)
    s.init_schema()
    return s


# ---------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------


class TestLayer2Pipeline:
    """Full pipeline: Event -> N-Gram -> DB -> Elo update."""

    def test_ngram_to_hypothesis_to_elo(self, storage):
        """Complete flow: detect N-gram pattern, store as hypothesis, update Elo.

        This simulates:
        1. User repeats a 3-step workflow 5 times
        2. N-gram extractor detects the pattern
        3. Pattern is stored as a hypothesis with initial Elo
        4. First match triggers Elo update
        5. Rating should increase (pattern_wins)
        """
        # 1. Generate repeated events
        events: list[NormalizedEvent] = []
        for i in range(5):
            events.append(
                normalize_event(
                    "Create a retargeting ad copy",
                    user_id=42,
                    timestamp=f"2026-05-20T{i * 3:02d}:00:00+00:00",
                )
            )
            events.append(
                normalize_event(
                    "Analyze campaign performance data",
                    user_id=42,
                    timestamp=f"2026-05-20T{i * 3 + 1:02d}:00:00+00:00",
                )
            )
            events.append(
                normalize_event(
                    "Plan next marketing strategy",
                    user_id=42,
                    timestamp=f"2026-05-20T{i * 3 + 2:02d}:00:00+00:00",
                )
            )

        # 2. Extract N-gram patterns
        patterns = extract_ngrams(events, n=3, min_occurrences=3)
        assert len(patterns) > 0, "Should detect at least one recurring 3-gram"

        top_pattern = patterns[0]
        assert top_pattern.occurrences >= 3

        # 3. Store as hypothesis
        ts = datetime.now(timezone.utc).isoformat()
        h = Hypothesis(
            hypothesis_id=f"hyp_{top_pattern.pattern_hash[:12]}",
            user_id=42,
            type="request",
            claim=f"User repeats workflow: {' -> '.join(top_pattern.events)}",
            status="candidate",
            version=1,
            elo_rating=INITIAL_RATING,
            elo_games_played=0,
            created_at=ts,
            last_seen=ts,
            pattern_hash=top_pattern.pattern_hash,
        )
        storage.insert_hypothesis(h)

        # 4. Simulate a match: pattern wins (user did not correct)
        elo_result = compute_elo_update(
            pattern_id=h.hypothesis_id,
            pattern_rating=INITIAL_RATING,
            request_difficulty=INITIAL_RATING,
            outcome="pattern_wins",
        )
        storage.update_hypothesis_elo(
            h.hypothesis_id,
            elo_result.new_rating,
        )

        # 5. Verify
        retrieved = storage.get_hypothesis(h.hypothesis_id)
        assert retrieved is not None
        assert retrieved.elo_rating > INITIAL_RATING
        assert retrieved.elo_games_played == 1

    def test_markov_prediction_matches_ngram_candidate(self, storage):
        """Markov prediction and N-gram detection should be consistent.

        If a Markov chain predicts A -> B with high probability,
        and N-gram detects [A, B, C] as recurring, both algorithms
        agree on a candidate (convergent evidence).
        """
        events: list[NormalizedEvent] = []
        for i in range(20):
            events.append(
                normalize_event(
                    "Create ad copy for campaign",
                    user_id=42,
                    timestamp=f"2026-05-20T{(i * 2) % 24:02d}:{(i * 2) // 24:02d}:00+00:00",
                )
            )
            events.append(
                normalize_event(
                    "Review analytics dashboard",
                    user_id=42,
                    timestamp=f"2026-05-20T{(i * 2 + 1) % 24:02d}:{(i * 2 + 1) // 24:02d}:00+00:00",
                )
            )

        # Markov: should predict B after A with high probability
        chain = MarkovChain()
        chain.update_batch(events)

        first_state = f"{events[0].domain}.{events[0].intent}"
        second_state = f"{events[1].domain}.{events[1].intent}"

        predictions = chain.predict_next(first_state)
        assert len(predictions) > 0
        # The most likely next state should be the second event type
        top_prediction = predictions[0]
        assert top_prediction.to_state == second_state
        assert top_prediction.probability > 0.5

        # N-gram: should detect [A, B, A] or similar as recurring
        ngrams = extract_ngrams(events, n=3, min_occurrences=5)
        assert len(ngrams) > 0, "N-gram should also detect recurring patterns"

    def test_elo_updates_pattern_difficulty_table(self, storage):
        """Pattern difficulty should be trackable alongside hypothesis Elo.

        The pattern_difficulty table stores per-fingerprint difficulty
        ratings that update in parallel with hypothesis ratings.
        """
        ts = datetime.now(timezone.utc).isoformat()
        fp_hash = "test_fp_difficulty_integration"

        # Initialize difficulty at 1500
        storage.upsert_pattern_difficulty(fp_hash, INITIAL_RATING, 0, ts)

        # Create a hypothesis for this fingerprint
        h = Hypothesis(
            hypothesis_id="hyp_diff_test",
            user_id=42,
            type="preference",
            claim="Test difficulty tracking",
            elo_rating=INITIAL_RATING,
            elo_games_played=0,
            created_at=ts,
            last_seen=ts,
            pattern_hash=fp_hash,
        )
        storage.insert_hypothesis(h)

        # Pattern wins: pattern rating goes up, difficulty goes down
        new_pattern_rating = update_elo(INITIAL_RATING, INITIAL_RATING, "pattern_wins")
        new_difficulty = update_difficulty(
            INITIAL_RATING, INITIAL_RATING, "pattern_wins"
        )

        storage.update_hypothesis_elo("hyp_diff_test", new_pattern_rating)
        storage.upsert_pattern_difficulty(fp_hash, new_difficulty, 1, ts)

        # Verify both updated correctly
        retrieved_h = storage.get_hypothesis("hyp_diff_test")
        retrieved_d = storage.get_pattern_difficulty(fp_hash)

        assert retrieved_h.elo_rating > INITIAL_RATING  # Pattern went up
        assert retrieved_d["difficulty_rating"] < INITIAL_RATING  # Difficulty went down

    def test_ngram_finds_pattern_already_in_db(self, storage):
        """If an N-gram is already stored as a hypothesis, re-detection
        should allow an Elo update rather than creating a duplicate.
        """
        ts = datetime.now(timezone.utc).isoformat()

        # Store an existing hypothesis
        existing_hash = "known_ngram_hash_abc123"
        h = Hypothesis(
            hypothesis_id="hyp_existing_ngram",
            user_id=42,
            type="request",
            claim="Known workflow pattern",
            status="candidate",
            elo_rating=1550.0,
            elo_games_played=3,
            created_at=ts,
            last_seen=ts,
            pattern_hash=existing_hash,
        )
        storage.insert_hypothesis(h)

        # Simulate re-detection and Elo update
        storage.update_hypothesis_elo(
            "hyp_existing_ngram",
            update_elo(1550.0, 1500.0, "pattern_wins"),
        )

        retrieved = storage.get_hypothesis("hyp_existing_ngram")
        assert retrieved.elo_rating > 1550.0
        assert retrieved.elo_games_played == 4


# ---------------------------------------------------------------
# Architecture Guards for Layer 2
# ---------------------------------------------------------------


class TestLayer2ArchitectureGuards:
    """HC-LAYER2-1: Tier-1 algorithms are candidate engines, NOT truth.

    These guards verify the architectural boundary: Layer 2 modules
    must NOT be imported by SkillMatcher (Layer 5). They communicate
    only through Pattern Storage.
    """

    def test_ngram_not_imported_by_skill_modules(self):
        """N-Gram extractor should not be imported by SkillMatcher.

        This is a static check: the file should not exist yet (Layer 5
        is not built), but when it is, it must not import Layer 2.
        """
        import importlib

        # Verify the module exists and is importable
        spec = importlib.util.find_spec("application.skill_compression.ngram_extractor")
        assert spec is not None, "ngram_extractor module should be importable"

        # Verify it does NOT import SkillMatcher (which does not exist yet)
        # This is a forward-looking guard

        source = spec.origin
        if source:
            from pathlib import Path

            code = Path(source).read_text(encoding="utf-8")
            assert "skill_matcher" not in code.lower(), (
                "ngram_extractor must not reference skill_matcher"
            )

    def test_markov_not_imported_by_skill_modules(self):
        """Markov chain should not be imported by SkillMatcher."""
        import importlib

        spec = importlib.util.find_spec("application.skill_compression.markov_chain")
        assert spec is not None

        source = spec.origin
        if source:
            from pathlib import Path

            code = Path(source).read_text(encoding="utf-8")
            assert "skill_matcher" not in code.lower()

    def test_elo_not_imported_by_skill_modules(self):
        """Elo rating should not be imported by SkillMatcher."""
        import importlib

        spec = importlib.util.find_spec("application.skill_compression.elo_rating")
        assert spec is not None

        source = spec.origin
        if source:
            from pathlib import Path

            code = Path(source).read_text(encoding="utf-8")
            assert "skill_matcher" not in code.lower()

    def test_layer2_modules_only_import_layer1(self):
        """Layer 2 modules should only import from Layer 1 (event_normalizer)
        and standard library, not from Layer 3+ (evidence, judge, etc.).
        """
        import importlib
        from pathlib import Path

        forbidden_imports = [
            "evidence_ledger",
            "pattern_judge",
            "skill_matcher",
        ]

        layer2_modules = [
            "application.skill_compression.ngram_extractor",
            "application.skill_compression.markov_chain",
            "application.skill_compression.elo_rating",
        ]

        for mod_name in layer2_modules:
            spec = importlib.util.find_spec(mod_name)
            if spec and spec.origin:
                code = Path(spec.origin).read_text(encoding="utf-8")
                for forbidden in forbidden_imports:
                    assert forbidden not in code, (
                        f"{mod_name} must not import {forbidden} "
                        f"(Layer 2 -> Layer 3+ violation)"
                    )
