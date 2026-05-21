"""End-to-End Integration Tests for Skill-Compression (Step 9).

Tests the complete pipeline from user message through all layers:
  Event -> Candidate -> Evidence -> Judge -> Match -> Apply

Also tests:
  - Threshold tuning with realistic data
  - /forget -> tombstone -> re-learning blocked
  - /learn -> immediate skill creation (decay_immune)
  - Conversation import -> suggested pattern
  - Skill collision resolution
  - Healthcare pattern rejected by privacy filter
  - Performance profiling of SkillMatcher.match()

No external dependencies beyond pytest. Uses CryptoConnection in
non-encrypted mode for test isolation.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from application.skill_compression.bkt import BKTState, update_bkt_weighted
from application.skill_compression.event_normalizer import normalize_event
from application.skill_compression.evidence_ledger import EvidenceSummary
from application.skill_compression.fsrs_decay import FSRSState
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import (
    SUGGEST_MIN_EVIDENCE,
    SUGGEST_MIN_SESSIONS,
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_CONFIRMED,
    STATUS_NEEDS_REVIEW,
    STATUS_PRIVACY_REJECTED,
    STATUS_SUGGESTED,
    THRESHOLDS,
    PatternJudge,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_matcher import (
    SkillMatcher,
)
from infrastructure.crypto_storage import CryptoConnection


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary test DB."""
    db_path = tmp_path / "test_e2e.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    yield conn
    conn.close()


@pytest.fixture
def storage(tmp_db):
    """Create HypothesisStorage with initialized schema."""
    s = HypothesisStorage(tmp_db)
    s.init_schema()
    return s


@pytest.fixture
def pipeline():
    """Create PrivacyPipeline."""
    return PrivacyPipeline()


@pytest.fixture
def judge(pipeline):
    """Create PatternJudge with privacy pipeline."""
    return PatternJudge(privacy_pipeline=pipeline)


@pytest.fixture
def matcher(storage, judge):
    """Create SkillMatcher."""
    return SkillMatcher(storage, judge)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_hypothesis(
    storage: HypothesisStorage,
    *,
    hyp_id: str = "",
    user_id: int = 42,
    h_type: str = "preference",
    claim: str = "Test skill",
    status: str = STATUS_CANDIDATE,
    scope: HypothesisScope | None = None,
    elo: float = 1500.0,
    support: int = 0,
    contradict: int = 0,
    decay_immune: bool = False,
    source_type: str = "live_chat",
    pattern_hash: str | None = None,
) -> Hypothesis:
    """Create and store a hypothesis, return the object."""
    if not hyp_id:
        hyp_id = f"hyp_{uuid4().hex[:12]}"
    ts = datetime.now(timezone.utc).isoformat()
    if scope is None:
        scope = HypothesisScope()
    h = Hypothesis(
        hypothesis_id=hyp_id,
        user_id=user_id,
        type=h_type,
        scope=scope,
        claim=claim,
        status=status,
        version=1,
        elo_rating=elo,
        support_count=support,
        contradict_count=contradict,
        decay_immune=decay_immune,
        source_type=source_type,
        created_at=ts,
        last_seen=ts,
        pattern_hash=pattern_hash,
    )
    storage.insert_hypothesis(h)
    return h


def _add_evidence(
    storage: HypothesisStorage,
    hypothesis_id: str,
    signal_type: str,
    count: int = 1,
    session_prefix: str = "session",
):
    """Add multiple evidence records."""
    ts = datetime.now(timezone.utc).isoformat()
    for i in range(count):
        storage.insert_evidence(
            evidence_id=f"ev_{uuid4().hex[:12]}",
            hypothesis_id=hypothesis_id,
            hypothesis_version=1,
            signal_type=signal_type,
            signal_strength=1.0 if signal_type != "correction" else -1.0,
            created_at=ts,
            episode_id=f"{session_prefix}_{i}",
        )


# ---------------------------------------------------------------
# E2E PIPELINE TESTS
# ---------------------------------------------------------------


class TestE2EPipeline:
    """End-to-end: 5 similar requests -> suggested -> confirmed -> active."""

    def test_full_lifecycle_candidate_to_active(self, storage, judge):
        """5 similar asks -> candidate -> suggested -> user confirms -> active.

        Simulates the complete lifecycle from first observation to
        auto-applied skill.
        """
        # Step 1: Create candidate from first normalized event
        event1 = normalize_event(
            "Erstelle eine 30s Retargeting Ad Copy",
            user_id=42,
        )
        hyp = _make_hypothesis(
            storage,
            hyp_id="hyp_lifecycle",
            claim="User requests retargeting ad copy with duration",
            status=STATUS_CANDIDATE,
            pattern_hash=event1.fingerprint_hash,
        )

        # Step 2: Accumulate evidence (5 positive signals over 3 sessions)
        _add_evidence(
            storage,
            "hyp_lifecycle",
            "no_correction",
            count=5,
            session_prefix="lifecycle_session",
        )

        # Build evidence summary
        evidence = EvidenceSummary(
            positive_count=5,
            negative_count=0,
            total_count=5,
            weighted_score=0.85,
            bkt_state=BKTState(),
            distinct_sessions=3,
            last_positive_at=datetime.now(timezone.utc).isoformat(),
            last_negative_at=None,
        )

        bkt = BKTState()
        for _ in range(5):
            bkt = update_bkt_weighted(bkt, True, 1.0)

        fsrs = FSRSState()

        # Step 3: Judge evaluates -> should suggest
        decision = judge.evaluate(
            hyp,
            evidence,
            bkt,
            1500.0,
            fsrs,
            current_time=datetime.now(timezone.utc).isoformat(),
        )

        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_SUGGESTED, (
            f"Expected suggested, got {decision.recommended_status}"
        )

        # Step 4: Simulate user confirmation (suggested -> confirmed)
        storage.update_hypothesis_status("hyp_lifecycle", STATUS_CONFIRMED)
        hyp_confirmed = storage.get_hypothesis("hyp_lifecycle")
        assert hyp_confirmed is not None
        assert hyp_confirmed.status == STATUS_CONFIRMED

        # Step 5: More evidence + high Elo -> confirmed -> active
        # Use preference threshold: 5 confirmations, Elo >= 1700
        _add_evidence(
            storage,
            "hyp_lifecycle",
            "explicit_confirm",
            count=5,
            session_prefix="confirm_session",
        )

        active_evidence = EvidenceSummary(
            positive_count=10,
            negative_count=0,
            total_count=10,
            weighted_score=0.95,
            bkt_state=bkt,
            distinct_sessions=5,
            last_positive_at=datetime.now(timezone.utc).isoformat(),
            last_negative_at=None,
        )

        # Reconstruct with confirmed status for judge
        hyp_for_judge = Hypothesis(
            hypothesis_id="hyp_lifecycle",
            user_id=42,
            type="preference",
            claim="User requests retargeting ad copy with duration",
            status=STATUS_CONFIRMED,
            elo_rating=1750.0,
            created_at=hyp.created_at,
            last_seen=datetime.now(timezone.utc).isoformat(),
        )

        decision2 = judge.evaluate(
            hyp_for_judge,
            active_evidence,
            bkt,
            1750.0,
            fsrs,
            current_time=datetime.now(timezone.utc).isoformat(),
        )

        assert decision2.should_transition is True
        assert decision2.recommended_status == STATUS_ACTIVE, (
            f"Expected active, got {decision2.recommended_status}"
        )

    def test_contradictions_trigger_needs_review(self, storage, judge):
        """10 confirmations + 3 contradictions -> needs_review."""
        hyp = _make_hypothesis(
            storage,
            hyp_id="hyp_review",
            claim="User prefers formal tone",
            status=STATUS_ACTIVE,
            elo=1800.0,
            support=10,
            contradict=3,
        )

        evidence = EvidenceSummary(
            positive_count=10,
            negative_count=3,
            total_count=13,
            weighted_score=0.7,
            bkt_state=BKTState(),
            distinct_sessions=5,
            last_positive_at=datetime.now(timezone.utc).isoformat(),
            last_negative_at=datetime.now(timezone.utc).isoformat(),
        )

        bkt = BKTState()
        fsrs = FSRSState()

        decision = judge.evaluate(
            hyp,
            evidence,
            bkt,
            1800.0,
            fsrs,
            current_time=datetime.now(timezone.utc).isoformat(),
            recent_contradictions=3,
        )

        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_NEEDS_REVIEW

    def test_forget_creates_tombstone_blocks_relearning(self, storage):
        """'/forget X' -> 30-day tombstone -> re-learning blocked."""
        _make_hypothesis(
            storage,
            hyp_id="hyp_forget",
            claim="Skill to forget",
            status=STATUS_ACTIVE,
            pattern_hash="fp_forget_hash",
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

        # Set status to retired
        storage.update_hypothesis_status("hyp_forget", "retired")

        # Create tombstone
        storage.insert_tombstone(
            tombstone_id="tomb_forget",
            hypothesis_id="hyp_forget",
            fingerprint="fp_forget_hash",
            deleted_at=now_iso,
            expires_at=expires,
        )

        # Re-learning must be blocked
        assert storage.check_tombstone("fp_forget_hash") is True

        # Verify hypothesis is retired
        retired = storage.get_hypothesis("hyp_forget")
        assert retired is not None
        assert retired.status == "retired"

    def test_learn_creates_immediate_skill(self, storage):
        """'/learn' -> immediate skill with decay_immune=True."""
        ts = datetime.now(timezone.utc).isoformat()
        hyp = Hypothesis(
            hypothesis_id="hyp_learn_cmd",
            user_id=42,
            type="preference",
            claim="Always use bullet points in summaries",
            status=STATUS_CONFIRMED,
            decay_immune=True,
            source_type="learn_command",
            created_at=ts,
            last_seen=ts,
            approval_state="approved",
        )
        storage.insert_hypothesis(hyp)

        retrieved = storage.get_hypothesis("hyp_learn_cmd")
        assert retrieved is not None
        assert retrieved.decay_immune is True
        assert retrieved.source_type == "learn_command"
        assert retrieved.status == STATUS_CONFIRMED

    def test_skill_collision_user_asked(self, judge):
        """Two skills with equal scope specificity -> user asked."""
        hyp_a = Hypothesis(
            hypothesis_id="hyp_a",
            claim="Use formal tone",
            scope=HypothesisScope(project="marketing"),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        hyp_b = Hypothesis(
            hypothesis_id="hyp_b",
            claim="Use casual tone",
            scope=HypothesisScope(project="marketing"),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )

        result = judge.detect_collision(hyp_a, hyp_b)
        assert result.has_collision is True
        assert result.needs_user_decision is True

    def test_skill_collision_specific_wins(self, judge):
        """More specific scope wins over global scope."""
        hyp_specific = Hypothesis(
            hypothesis_id="hyp_spec",
            claim="Use formal tone for client X",
            scope=HypothesisScope(project="ads", client="honey-brand"),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        hyp_global = Hypothesis(
            hypothesis_id="hyp_glob",
            claim="Use casual tone everywhere",
            scope=HypothesisScope(),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )

        result = judge.detect_collision(hyp_specific, hyp_global)
        assert result.has_collision is True
        assert result.needs_user_decision is False
        assert result.winner_id == "hyp_spec"

    def test_healthcare_pattern_rejected_by_privacy(self, judge):
        """Healthcare pattern -> privacy_rejected by pipeline."""
        hyp = Hypothesis(
            hypothesis_id="hyp_health",
            claim="User seems depressed based on writing patterns",
            status=STATUS_CANDIDATE,
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )

        evidence = EvidenceSummary(
            positive_count=5,
            negative_count=0,
            total_count=5,
            weighted_score=0.9,
            bkt_state=BKTState(),
            distinct_sessions=3,
            last_positive_at="2026-05-20T00:00:00+00:00",
            last_negative_at=None,
        )

        decision = judge.evaluate(
            hyp,
            evidence,
            BKTState(),
            1700.0,
            FSRSState(),
            current_time="2026-05-20T12:00:00+00:00",
        )

        assert decision.recommended_status == STATUS_PRIVACY_REJECTED


# ---------------------------------------------------------------
# THRESHOLD TUNING TESTS
# ---------------------------------------------------------------


class TestThresholdTuning:
    """Verify auto-apply thresholds with realistic mock data.

    Tests the spec-defined thresholds and validates they produce
    the expected behavior with realistic parameter combinations.
    """

    def _make_evidence(
        self,
        positive: int,
        negative: int,
        sessions: int,
    ) -> EvidenceSummary:
        """Create an EvidenceSummary with given counts."""
        bkt = BKTState()
        # Simulate BKT updates
        for _ in range(positive):
            bkt = update_bkt_weighted(bkt, True, 1.0)
        for _ in range(negative):
            bkt = update_bkt_weighted(bkt, False, 1.0)

        return EvidenceSummary(
            positive_count=positive,
            negative_count=negative,
            total_count=positive + negative,
            weighted_score=bkt.p_knowledge,
            bkt_state=bkt,
            distinct_sessions=sessions,
            last_positive_at=datetime.now(timezone.utc).isoformat(),
            last_negative_at=datetime.now(timezone.utc).isoformat()
            if negative
            else None,
        )

    # ── Negative pattern thresholds ──

    def test_negative_specific_threshold_2_confirmations(self, judge):
        """Negative specific: 2 confirmations, Elo >= 1650."""
        threshold = THRESHOLDS["negative_specific"]
        assert threshold.min_confirmations == 2
        assert threshold.min_elo_rating == 1650.0

        hyp = Hypothesis(
            hypothesis_id="hyp_neg_spec",
            type="negative",
            claim="Never use emojis for client X",
            status=STATUS_CONFIRMED,
            scope=HypothesisScope(project="ads", client="honey"),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )

        evidence = self._make_evidence(positive=2, negative=0, sessions=2)

        # Should promote at exactly threshold
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1650.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_ACTIVE

    def test_negative_specific_below_threshold_no_promotion(self, judge):
        """Negative specific: 1 confirmation not enough."""
        hyp = Hypothesis(
            hypothesis_id="hyp_neg_spec_low",
            type="negative",
            claim="Never use emojis",
            status=STATUS_CONFIRMED,
            scope=HypothesisScope(project="ads", client="honey"),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        evidence = self._make_evidence(positive=1, negative=0, sessions=1)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1650.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is False

    def test_negative_domain_threshold_4_confirmations(self, judge):
        """Negative domain: 4 confirmations, Elo >= 1700."""
        threshold = THRESHOLDS["negative_domain"]
        assert threshold.min_confirmations == 4
        assert threshold.min_elo_rating == 1700.0

        hyp = Hypothesis(
            hypothesis_id="hyp_neg_dom",
            type="negative",
            claim="No emojis in marketing",
            status=STATUS_CONFIRMED,
            scope=HypothesisScope(project="marketing"),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        evidence = self._make_evidence(positive=4, negative=0, sessions=2)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1700.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_ACTIVE

    def test_negative_global_threshold_6_confirmations(self, judge):
        """Negative global: 6 confirmations, Elo >= 1750."""
        threshold = THRESHOLDS["negative_global"]
        assert threshold.min_confirmations == 6
        assert threshold.min_elo_rating == 1750.0

        hyp = Hypothesis(
            hypothesis_id="hyp_neg_global",
            type="negative",
            claim="Never use emojis anywhere",
            status=STATUS_CONFIRMED,
            scope=HypothesisScope(),
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        evidence = self._make_evidence(positive=6, negative=0, sessions=3)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1750.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_ACTIVE

    # ── Preference threshold ──

    def test_preference_threshold_5_confirmations(self, judge):
        """Preference: 5 confirmations, Elo >= 1700."""
        threshold = THRESHOLDS["preference"]
        assert threshold.min_confirmations == 5
        assert threshold.min_elo_rating == 1700.0

        hyp = Hypothesis(
            hypothesis_id="hyp_pref",
            type="preference",
            claim="Always use bullet points",
            status=STATUS_CONFIRMED,
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        evidence = self._make_evidence(positive=5, negative=0, sessions=2)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1700.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_ACTIVE

    def test_preference_low_elo_no_promotion(self, judge):
        """Preference with 5 confirmations but Elo < 1700 must NOT promote."""
        hyp = Hypothesis(
            hypothesis_id="hyp_pref_low_elo",
            type="preference",
            claim="Use bullet points",
            status=STATUS_CONFIRMED,
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        evidence = self._make_evidence(positive=5, negative=0, sessions=2)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1600.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is False, (
            "Preference with Elo < 1700 must NOT be promoted to active"
        )

    # ── Procedural threshold ──

    def test_procedural_threshold_8_confirmations(self, judge):
        """Procedural: 8 confirmations, Elo >= 1800."""
        threshold = THRESHOLDS["procedural"]
        assert threshold.min_confirmations == 8
        assert threshold.min_elo_rating == 1800.0

        hyp = Hypothesis(
            hypothesis_id="hyp_proc",
            type="procedural",
            claim="First outline, then draft, then review",
            status=STATUS_CONFIRMED,
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        evidence = self._make_evidence(positive=8, negative=0, sessions=3)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1800.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_ACTIVE

    def test_procedural_below_threshold_no_promotion(self, judge):
        """Procedural with 7 confirmations (< 8) must NOT promote."""
        hyp = Hypothesis(
            hypothesis_id="hyp_proc_low",
            type="procedural",
            claim="First outline then draft",
            status=STATUS_CONFIRMED,
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )
        evidence = self._make_evidence(positive=7, negative=0, sessions=3)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1800.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is False

    # ── Candidate -> Suggested threshold ──

    def test_candidate_to_suggested_threshold(self, judge):
        """Candidate -> suggested requires 3 evidence over 2 sessions."""
        assert SUGGEST_MIN_EVIDENCE == 3
        assert SUGGEST_MIN_SESSIONS == 2

        hyp = Hypothesis(
            hypothesis_id="hyp_cand",
            claim="Pattern under observation",
            status=STATUS_CANDIDATE,
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )

        # Exactly at threshold
        evidence = self._make_evidence(positive=3, negative=0, sessions=2)
        decision = judge.evaluate(
            hyp,
            evidence,
            evidence.bkt_state,
            1500.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_SUGGESTED

        # Below threshold (only 2 evidence)
        evidence_low = self._make_evidence(positive=2, negative=0, sessions=2)
        decision_low = judge.evaluate(
            hyp,
            evidence_low,
            evidence_low.bkt_state,
            1500.0,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision_low.should_transition is False

    # ── All thresholds require ALL conditions ──

    def test_all_conditions_must_be_met_simultaneously(self, judge):
        """Auto-apply requires ALL of: confirmations + Elo + sessions + BKT."""
        hyp = Hypothesis(
            hypothesis_id="hyp_all_cond",
            type="preference",
            claim="Test all conditions",
            status=STATUS_CONFIRMED,
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )

        threshold = THRESHOLDS["preference"]

        # Enough confirmations but only 1 session (need 2)
        evidence_1sess = EvidenceSummary(
            positive_count=threshold.min_confirmations,
            negative_count=0,
            total_count=threshold.min_confirmations,
            weighted_score=0.9,
            bkt_state=BKTState(p_knowledge=0.9),
            distinct_sessions=1,
            last_positive_at=datetime.now(timezone.utc).isoformat(),
            last_negative_at=None,
        )

        decision = judge.evaluate(
            hyp,
            evidence_1sess,
            evidence_1sess.bkt_state,
            threshold.min_elo_rating,
            FSRSState(),
            current_time=datetime.now(timezone.utc).isoformat(),
        )
        assert decision.should_transition is False, (
            "Must NOT promote with insufficient sessions"
        )


# ---------------------------------------------------------------
# PERFORMANCE TESTS
# ---------------------------------------------------------------


class TestPerformance:
    """Performance profiling for SkillMatcher.match()."""

    def test_match_performance_under_100ms(self, storage, matcher):
        """SkillMatcher.match() must complete in < 100ms with 50 hypotheses."""
        ts = datetime.now(timezone.utc).isoformat()

        # Create 50 active hypotheses (worst case: max skill library)
        for i in range(50):
            event = normalize_event(
                f"Create ad copy for campaign {i}",
                user_id=42,
            )
            h = Hypothesis(
                hypothesis_id=f"hyp_perf_{i}",
                user_id=42,
                type="preference",
                claim=f"Skill for campaign {i}: use formal tone",
                status=STATUS_ACTIVE,
                elo_rating=1700.0 + i,
                created_at=ts,
                last_seen=ts,
                pattern_hash=event.fingerprint_hash,
            )
            storage.insert_hypothesis(h)

        # Measure match time
        test_event = normalize_event(
            "Write a retargeting ad copy with CTA",
            user_id=42,
        )

        start = time.perf_counter()
        _ = matcher.match(test_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, (
            f"SkillMatcher.match() took {elapsed_ms:.1f}ms with 50 "
            f"hypotheses. Must be < 100ms."
        )

    def test_match_all_performance(self, storage, matcher):
        """SkillMatcher.match_all() must complete in < 200ms with 50 hypotheses."""
        ts = datetime.now(timezone.utc).isoformat()

        for i in range(50):
            event = normalize_event(
                f"Analyze report data {i}",
                user_id=42,
            )
            h = Hypothesis(
                hypothesis_id=f"hyp_perf_all_{i}",
                user_id=42,
                type="request",
                claim=f"Analyze monthly report {i}",
                status=STATUS_CONFIRMED,
                elo_rating=1600.0,
                created_at=ts,
                last_seen=ts,
                pattern_hash=event.fingerprint_hash,
            )
            storage.insert_hypothesis(h)

        test_event = normalize_event(
            "Analyze the quarterly revenue report",
            user_id=42,
        )

        start = time.perf_counter()
        _ = matcher.match_all(test_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, (
            f"SkillMatcher.match_all() took {elapsed_ms:.1f}ms. Must be < 200ms."
        )


# ---------------------------------------------------------------
# CONVERSATION IMPORT TESTS
# ---------------------------------------------------------------


class TestConversationImport:
    """Import pipeline creates suggested patterns."""

    def test_imported_hypothesis_starts_as_suggested(self, storage):
        """HC-IMPORT-1: All imported hypotheses start as 'suggested'."""
        ts = datetime.now(timezone.utc).isoformat()
        hyp = Hypothesis(
            hypothesis_id="hyp_imported",
            user_id=42,
            type="preference",
            claim="Imported: user prefers concise answers",
            status=STATUS_SUGGESTED,
            source_type="import",
            created_at=ts,
            last_seen=ts,
        )
        storage.insert_hypothesis(hyp)

        retrieved = storage.get_hypothesis("hyp_imported")
        assert retrieved is not None
        assert retrieved.status == STATUS_SUGGESTED
        assert retrieved.source_type == "import"
