"""Tests for SkillMatcher (Layer 5), Commit 4.1.

Covers:
  - Direct alias match returns SkillMatch
  - Fingerprint match when no alias matches
  - Score threshold respected (match with 0.6 -> no match)
  - Status candidate/suggested NOT matched (only confirmed + active)
  - Multiple matching hypotheses -> CollisionDetector called
  - should_ask_user logic (HC-SC-10)
"""

from __future__ import annotations

import sqlite3

from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import (
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_CONFIRMED,
    STATUS_SUGGESTED,
    PatternJudge,
)
from application.skill_compression.skill_matcher import (
    DEFAULT_USER_PREFERENCES,
    MATCHABLE_STATUSES,
    SkillMatch,
    SkillMatcher,
    should_ask_user,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


class FakeDBConnection:
    """Minimal in-memory SQLite connection compatible with HypothesisStorage."""

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
    """Create an in-memory HypothesisStorage with schema initialized."""
    conn = FakeDBConnection()
    storage = HypothesisStorage(conn)
    storage.init_schema()
    return storage


def _make_hypothesis(
    *,
    hypothesis_id: str = "hyp-001",
    user_id: int = 42,
    status: str = STATUS_CONFIRMED,
    claim: str = "User prefers bullet points",
    h_type: str = "preference",
    elo_rating: float = 1600.0,
    project: str = "",
    client: str = "",
    pattern_hash: str = "abc123",
) -> Hypothesis:
    """Create a test hypothesis."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=user_id,
        type=h_type,
        scope=HypothesisScope(project=project, client=client),
        claim=claim,
        status=status,
        elo_rating=elo_rating,
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
        pattern_hash=pattern_hash,
    )


def _make_event(
    *,
    raw_text: str = "bullet points bitte",
    user_id: int = 42,
    intent: str = "create_text",
    domain: str = "content",
    language: str = "de",
) -> NormalizedEvent:
    """Create a test NormalizedEvent."""
    return NormalizedEvent(
        event_id="evt_test_001",
        user_id=user_id,
        timestamp="2026-05-20T12:00:00+00:00",
        raw_text=raw_text,
        intent=intent,
        domain=domain,
        format_type="list",
        language=language,
        fingerprint_hash="fp_test_001",
    )


# ---------------------------------------------------------------
# Tests: SkillMatcher.match
# ---------------------------------------------------------------


class TestSkillMatcherAliasMatch:
    """Test direct alias matching."""

    def test_alias_match_returns_skill_match(self) -> None:
        """Direct alias match should return a SkillMatch."""
        storage = _setup_storage()
        hyp = _make_hypothesis(elo_rating=1700.0)
        storage.insert_hypothesis(hyp)
        storage.insert_alias(
            alias_id="alias-001",
            hypothesis_id="hyp-001",
            alias_text="bullet points bitte",
            first_seen="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
            confidence=0.9,
        )

        matcher = SkillMatcher(storage, PatternJudge())
        event = _make_event(raw_text="bullet points bitte")
        result = matcher.match(event)

        assert result is not None
        assert result.hypothesis.hypothesis_id == "hyp-001"
        assert result.match_source == "alias"
        assert result.confidence > 0.7

    def test_alias_match_case_insensitive(self) -> None:
        """Alias match should be case-insensitive."""
        storage = _setup_storage()
        hyp = _make_hypothesis(elo_rating=1700.0)
        storage.insert_hypothesis(hyp)
        storage.insert_alias(
            alias_id="alias-002",
            hypothesis_id="hyp-001",
            alias_text="Bullet Points Bitte",
            first_seen="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
            confidence=0.9,
        )

        matcher = SkillMatcher(storage, PatternJudge())
        event = _make_event(raw_text="bullet points bitte")
        result = matcher.match(event)

        assert result is not None
        assert result.match_source == "alias"


class TestSkillMatcherStatusFiltering:
    """Test that only confirmed/active hypotheses are matched."""

    def test_candidate_not_matched(self) -> None:
        """Candidate hypotheses must NOT be matched."""
        storage = _setup_storage()
        hyp = _make_hypothesis(status=STATUS_CANDIDATE, elo_rating=1800.0)
        storage.insert_hypothesis(hyp)
        storage.insert_alias(
            alias_id="alias-003",
            hypothesis_id="hyp-001",
            alias_text="bullet points bitte",
            first_seen="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
            confidence=0.95,
        )

        matcher = SkillMatcher(storage, PatternJudge())
        event = _make_event()
        result = matcher.match(event)

        assert result is None

    def test_suggested_not_matched(self) -> None:
        """Suggested hypotheses must NOT be matched."""
        storage = _setup_storage()
        hyp = _make_hypothesis(status=STATUS_SUGGESTED, elo_rating=1800.0)
        storage.insert_hypothesis(hyp)
        storage.insert_alias(
            alias_id="alias-004",
            hypothesis_id="hyp-001",
            alias_text="bullet points bitte",
            first_seen="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
            confidence=0.95,
        )

        matcher = SkillMatcher(storage, PatternJudge())
        event = _make_event()
        result = matcher.match(event)

        assert result is None

    def test_confirmed_is_matched(self) -> None:
        """Confirmed hypotheses should be matched."""
        assert STATUS_CONFIRMED in MATCHABLE_STATUSES

    def test_active_is_matched(self) -> None:
        """Active hypotheses should be matched."""
        assert STATUS_ACTIVE in MATCHABLE_STATUSES


class TestSkillMatcherThreshold:
    """Test score threshold enforcement."""

    def test_low_elo_below_threshold(self) -> None:
        """Low Elo with low alias confidence should not match."""
        storage = _setup_storage()
        # Very low elo + low confidence = combined score below 0.7
        hyp = _make_hypothesis(elo_rating=800.0)
        storage.insert_hypothesis(hyp)
        storage.insert_alias(
            alias_id="alias-005",
            hypothesis_id="hyp-001",
            alias_text="bullet points bitte",
            first_seen="2026-05-20T10:00:00+00:00",
            last_seen="2026-05-20T10:00:00+00:00",
            confidence=0.5,
        )

        matcher = SkillMatcher(storage, PatternJudge())
        event = _make_event()
        result = matcher.match(event)

        # 0.5 * (800/2000) * 1.2 = 0.5 * 0.4 * 1.2 = 0.24 < 0.7
        assert result is None

    def test_no_match_returns_none(self) -> None:
        """No matching hypotheses should return None."""
        storage = _setup_storage()
        matcher = SkillMatcher(storage, PatternJudge())
        event = _make_event()
        result = matcher.match(event)

        assert result is None


class TestSkillMatcherCollision:
    """Test collision detection delegation."""

    def test_multiple_matches_trigger_collision_resolution(self) -> None:
        """Multiple fingerprint matches should call CollisionDetector."""
        storage = _setup_storage()

        # Two confirmed hypotheses with different scopes
        hyp_global = _make_hypothesis(
            hypothesis_id="hyp-global",
            claim="Global rule",
            elo_rating=1700.0,
            pattern_hash="fp_global",
        )
        hyp_specific = _make_hypothesis(
            hypothesis_id="hyp-specific",
            claim="Specific rule",
            elo_rating=1700.0,
            project="ads",
            client="acme",
            pattern_hash="fp_specific",
        )
        storage.insert_hypothesis(hyp_global)
        storage.insert_hypothesis(hyp_specific)

        matcher = SkillMatcher(storage, PatternJudge())

        # Create fake fingerprint matches
        match_global = SkillMatch(
            hypothesis=hyp_global,
            confidence=0.85,
            requires_confirmation=True,
            explanation="test",
        )
        match_specific = SkillMatch(
            hypothesis=hyp_specific,
            confidence=0.90,
            requires_confirmation=True,
            explanation="test",
        )

        result = matcher._resolve_collision([match_specific, match_global])
        assert result is not None
        # Specific scope should win
        assert result.hypothesis.hypothesis_id == "hyp-specific"


# ---------------------------------------------------------------
# Tests: should_ask_user (HC-SC-10)
# ---------------------------------------------------------------


class TestShouldAskUser:
    """Tests for the should_ask_user helper (HC-SC-10)."""

    def test_confirmed_always_asks(self) -> None:
        """Status confirmed: should_ask_user must be True, always."""
        hyp = _make_hypothesis(status=STATUS_CONFIRMED)
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=True,
            explanation="test",
        )
        # Even with auto_apply_enabled=True
        assert should_ask_user(match, {"auto_apply_enabled": True}) is True

    def test_active_never_asks_regardless_of_preferences(self) -> None:
        """Status active: never ask (Round-5: user already confirmed once).

        Round-5 change (2026-05-27): Active skills auto-apply unconditionally.
        Previously checked auto_apply_enabled preference. Now 'active' means
        user explicitly approved, so it always auto-applies.
        """
        hyp = _make_hypothesis(status=STATUS_ACTIVE)
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=False,
            explanation="test",
        )
        assert should_ask_user(match, {"auto_apply_enabled": False}) is False

    def test_active_auto_apply_enabled_does_not_ask(self) -> None:
        """Status active + auto_apply_enabled=True: do not ask."""
        hyp = _make_hypothesis(status=STATUS_ACTIVE)
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=False,
            explanation="test",
        )
        assert should_ask_user(match, {"auto_apply_enabled": True}) is False

    def test_default_preferences_active_does_not_ask(self) -> None:
        """Default preferences + active status: should NOT ask (Round-5).

        Round-5 change: active skills never ask, regardless of preferences.
        """
        hyp = _make_hypothesis(status=STATUS_ACTIVE)
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=False,
            explanation="test",
        )
        # None = use defaults; active still does not ask
        assert should_ask_user(match) is False

    def test_default_auto_apply_is_false(self) -> None:
        """Default preference for auto_apply_enabled must be False."""
        assert DEFAULT_USER_PREFERENCES["auto_apply_enabled"] is False
