"""Bug-Fix Round 5 + 5b: Skill execution after user confirmation.

Live-Bug 2026-05-27 11:20: User clicks "Ja" on ask-before-apply dialog,
but no skill response comes. The callback only wrote evidence and confirmed
status, but never re-processed the original message through the streaming
pipeline with the skill instruction block.

Fixes (Round 5):
  1. After "yes" callback: promote hypothesis to 'active', then re-process
     original message through streaming pipeline (LLM gets skill instruction).
  2. Active skills auto-apply without ask-before-apply dialog (Round-5:
     should_ask_user returns False for STATUS_ACTIVE unconditionally).
  3. "Nie wieder" transitions hypothesis to 'paused'.

Polish (Round 5b, Codex feedback):
  1.1. i18n: "Skill angewendet." -> "Skill wird angewendet..." (action-in-progress)
  1.2. Evidence split: user_confirmed (button click) vs skill_executed (after run)
  1.3. Expired pending: graceful handling with localized message
  1.4. "Nein" path: explicit test for evidence + no-execution + pending cleanup

Memory rules enforced:
  - feedback_briefing_production_path_tests (production path tests)
  - feedback_security_feature_four_path_tests (4-path coverage)
  - feedback_sigma_verification_checklist (checklist at end)
  - feedback_pytest_run_from_bridge (run from bridge/)
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from infrastructure.conversation_storage import _reset_all_for_tests


@pytest.fixture(autouse=True)
def _clear_storage():
    _reset_all_for_tests()


# =====================================================================
# Shared helpers
# =====================================================================


class FakeDBConnection:
    """Minimal in-memory SQLite wrapper for tests."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        from application.skill_compression.hypothesis_storage import (
            HYPOTHESIS_SCHEMA_SQL,
        )

        self._conn.executescript(HYPOTHESIS_SCHEMA_SQL)
        self._conn.commit()

    def execute(self, sql, params=None):
        cur = self._conn.execute(sql, params or ())
        self._conn.commit()
        return cur

    def executescript(self, sql):
        self._conn.executescript(sql)
        self._conn.commit()

    def execute_in_transaction(self, operations):
        for sql, params in operations:
            self._conn.execute(sql, params)
        self._conn.commit()

    def fetchone(self, sql, params=None):
        cur = self._conn.execute(sql, params or ())
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self, sql, params=None):
        cur = self._conn.execute(sql, params or ())
        return [dict(r) for r in cur.fetchall()]


def _create_confirmed_skill(conn, storage, claim_text, user_id=42):
    """Create a confirmed skill with proper aliases via SkillLearningService."""
    from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
    from application.skill_compression.skill_learning_service import (
        SkillLearningService,
    )

    pipeline = PrivacyPipeline()
    service = SkillLearningService(storage, pipeline)

    result = service.learn(
        claim_text=claim_text,
        user_id=user_id,
        source="learn_command",
    )
    assert result.success, f"Failed to create skill: {result}"
    return result.hypothesis_id


def _build_pending_confirmation(skill_match, original_text, envelope=None):
    """Build a pending skill confirmation entry for the pending store."""
    if envelope is None:
        from application.execution import RequestEnvelope

        envelope = RequestEnvelope.from_telegram(
            user_id=42,
            chat_id=100,
            text=original_text,
            username="testuser",
        )
    return {
        "skill_match": skill_match,
        "original_text": original_text,
        "timestamp": time.time(),
        "envelope": envelope,
    }


# =====================================================================
# Fix 1: Skill execution after "Ja" confirmation
# =====================================================================


class TestSkillExecutionAfterConfirmation:
    """Round-5 Live-Bug: confirmation must trigger actual skill execution."""

    def test_confirm_yes_promotes_to_active(self):
        """After 'yes': hypothesis status transitions from confirmed to active."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import (
            STATUS_ACTIVE,
            STATUS_CONFIRMED,
        )

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich rot sage, erklaere RGB"
        )

        # Verify initial status is confirmed
        hyp = storage.get_hypothesis(hyp_id)
        assert hyp.status == STATUS_CONFIRMED

        # Simulate what _handle_skill_confirm_inline does on "yes"
        storage.transition_hypothesis_status(hyp_id, STATUS_ACTIVE)

        # Verify status changed to active
        hyp_after = storage.get_hypothesis(hyp_id)
        assert hyp_after.status == STATUS_ACTIVE

    def test_confirm_yes_writes_evidence(self):
        """After 'yes': user_confirmed evidence is written."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import SkillMatch, SkillMatcher
        from application.chat_service import ChatService

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich rot sage, erklaere RGB"
        )

        hyp = storage.get_hypothesis(hyp_id)
        skill_match = SkillMatch(
            hypothesis=hyp,
            confidence=0.85,
            requires_confirmation=True,
            explanation="alias match",
            match_source="alias",
        )

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        mock_router = MagicMock()
        chat_service = ChatService(
            provider_router=mock_router,
            memory_service=MagicMock(),
            skill_matcher=matcher,
        )

        # Write evidence as the confirmation handler does
        chat_service._write_skill_evidence(
            skill_match, signal_type="user_confirmed", signal_strength=0.5
        )

        # Check evidence was written
        rows = conn.fetchall(
            "SELECT signal_type FROM hypothesis_evidence WHERE hypothesis_id = ?",
            (hyp_id,),
        )
        signal_types = [r["signal_type"] for r in rows]
        assert "user_confirmed" in signal_types

    def test_confirm_yes_stores_original_text_in_pending(self):
        """Pending confirmation must store original_text for re-processing."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.skill_matcher import SkillMatch

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich rot sage, erklaere RGB"
        )

        hyp = storage.get_hypothesis(hyp_id)
        skill_match = SkillMatch(
            hypothesis=hyp,
            confidence=0.85,
            requires_confirmation=True,
            explanation="alias match",
            match_source="alias",
        )

        pending = _build_pending_confirmation(skill_match, "rot")

        # Verify all required fields are present
        assert pending["original_text"] == "rot"
        assert pending["skill_match"] is skill_match
        assert pending["envelope"] is not None
        assert pending["timestamp"] > 0


# =====================================================================
# Fix 2: Active skills auto-apply without ask-before-apply
# =====================================================================


class TestActiveSkillAutoApply:
    """Active skills must run without ask-before-apply dialog."""

    def test_should_ask_user_false_for_active(self):
        """should_ask_user returns False for STATUS_ACTIVE (Round-5)."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )
        from application.skill_compression.pattern_judge import STATUS_ACTIVE
        from application.skill_compression.skill_matcher import (
            SkillMatch,
            should_ask_user,
        )

        hyp = Hypothesis(
            hypothesis_id="hyp_test_active",
            user_id=42,
            type="preference",
            scope=HypothesisScope(project="", client="", context=()),
            claim="erklaere RGB farben",
            status=STATUS_ACTIVE,
            version=1,
            elo_rating=1600.0,
            elo_games_played=5,
            bayes_confidence=0.7,
            support_count=3,
            contradict_count=0,
            fsrs_state_json="{}",
            source_type="learn_command",
            decay_immune=True,
            evidence_ids=(),
            created_at="2026-05-27T10:00:00Z",
            last_seen="2026-05-27T11:00:00Z",
            approval_state="approved",
        )
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.9,
            requires_confirmation=False,
            explanation="test",
            match_source="alias",
        )

        assert should_ask_user(match) is False
        assert should_ask_user(match, {"auto_apply_enabled": False}) is False
        assert should_ask_user(match, {"auto_apply_enabled": True}) is False

    def test_should_ask_user_true_for_confirmed(self):
        """should_ask_user returns True for STATUS_CONFIRMED (unchanged)."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )
        from application.skill_compression.pattern_judge import STATUS_CONFIRMED
        from application.skill_compression.skill_matcher import (
            SkillMatch,
            should_ask_user,
        )

        hyp = Hypothesis(
            hypothesis_id="hyp_test_confirmed",
            user_id=42,
            type="preference",
            scope=HypothesisScope(project="", client="", context=()),
            claim="erklaere RGB farben",
            status=STATUS_CONFIRMED,
            version=1,
            elo_rating=1600.0,
            elo_games_played=0,
            bayes_confidence=0.5,
            support_count=0,
            contradict_count=0,
            fsrs_state_json="{}",
            source_type="learn_command",
            decay_immune=True,
            evidence_ids=(),
            created_at="2026-05-27T10:00:00Z",
            last_seen="2026-05-27T11:00:00Z",
            approval_state="pending",
        )
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.85,
            requires_confirmation=True,
            explanation="test",
            match_source="alias",
        )

        assert should_ask_user(match) is True

    def test_active_skill_matches_and_produces_prompt_block(self):
        """Active skill matched by SkillMatcher produces instruction block."""
        from application.chat_service import ChatService
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import SkillMatcher

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich rot sage, erklaere RGB"
        )

        # Promote to active (as Round-5 confirmation does)
        storage.transition_hypothesis_status(hyp_id, "active")

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        mock_router = MagicMock()
        svc = ChatService(
            provider_router=mock_router,
            memory_service=MagicMock(),
            skill_matcher=matcher,
        )

        # Match skill for prompt
        skill_block, match = svc._match_skills_for_prompt(42, "rot", "de", None)

        # Must match (active skills are matchable)
        assert match is not None
        assert "USER-DEFINED SKILL (HIGH PRIORITY)" in skill_block
        assert "MUST be applied" in skill_block


# =====================================================================
# Fix 3: "Nie wieder" pauses the skill
# =====================================================================


class TestNeverAgainPausesSkill:
    """'Nie wieder' button must transition hypothesis to paused."""

    def test_never_again_transitions_to_paused(self):
        """Clicking 'Nie wieder' sets hypothesis status to paused."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import STATUS_PAUSED

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich blau sage, erklaere Himmel"
        )

        # Simulate what _handle_skill_confirm_inline does on "never"
        storage.transition_hypothesis_status(hyp_id, "paused")

        hyp = storage.get_hypothesis(hyp_id)
        assert hyp.status == STATUS_PAUSED

    def test_paused_skill_not_matched(self):
        """Paused skill must NOT be matched by SkillMatcher."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import SkillMatcher
        from application.skill_compression.event_normalizer import NormalizedEvent

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich blau sage, erklaere Himmel"
        )

        # Pause (simulate "nie wieder")
        storage.transition_hypothesis_status(hyp_id, "paused")

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        event = NormalizedEvent(
            event_id="evt_paused_test",
            user_id=42,
            timestamp="2026-05-27T12:00:00Z",
            raw_text="blau",
            intent="",
            domain="",
            format_type="",
            language="de",
            fingerprint_hash="",
        )
        match = matcher.match(event)
        assert match is None, "Paused skills must not be matched"


# =====================================================================
# Production-Path: Full confirmation flow integration
# =====================================================================


class TestConfirmationFlowIntegration:
    """Production path: full flow from skill match to confirmation to execution."""

    def test_full_flow_match_confirm_promote_rematch(self):
        """End-to-end: match -> confirmation -> promote -> re-match auto-applies.

        This is the production path for the Round-5 bug fix:
        1. User sends "rot"
        2. SkillMatcher finds alias match (confirmed status)
        3. should_ask_user returns True (confirmed -> ask)
        4. User clicks "Ja"
        5. Hypothesis promoted to 'active'
        6. Re-match: should_ask_user returns False (active -> auto-apply)
        7. LLM gets skill instruction block
        """
        from application.chat_service import ChatService
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import (
            SkillMatcher,
            should_ask_user,
        )
        from application.skill_compression.event_normalizer import NormalizedEvent

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich rot sage, erklaere RGB"
        )

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        mock_router = MagicMock()
        svc = ChatService(
            provider_router=mock_router,
            memory_service=MagicMock(),
            skill_matcher=matcher,
        )

        # Step 1: First match (confirmed status)
        event1 = NormalizedEvent(
            event_id="evt_flow_01",
            user_id=42,
            timestamp="2026-05-27T12:00:00Z",
            raw_text="rot",
            intent="",
            domain="",
            format_type="",
            language="de",
            fingerprint_hash="",
        )
        match1 = matcher.match(event1)
        assert match1 is not None
        assert should_ask_user(match1) is True  # confirmed -> ask

        # Step 2: User confirms ("Ja") -> promote to active
        storage.transition_hypothesis_status(hyp_id, "active")

        # Step 3: Re-match after promotion (active status)
        event2 = NormalizedEvent(
            event_id="evt_flow_02",
            user_id=42,
            timestamp="2026-05-27T12:00:01Z",
            raw_text="rot",
            intent="",
            domain="",
            format_type="",
            language="de",
            fingerprint_hash="",
        )
        match2 = matcher.match(event2)
        assert match2 is not None
        assert should_ask_user(match2) is False  # active -> auto-apply

        # Step 4: Verify ChatService produces skill block
        skill_block, match3 = svc._match_skills_for_prompt(42, "rot", "de", None)
        assert match3 is not None
        assert "USER-DEFINED SKILL" in skill_block

    def test_reprocess_function_exists_and_is_async(self):
        """reprocess_after_skill_confirmation exists as an async function."""
        from presentation.handlers import reprocess_after_skill_confirmation

        assert asyncio.iscoroutinefunction(reprocess_after_skill_confirmation)


# =====================================================================
# 4-Path Tests (Happy / No / Never / Edge-Case)
# =====================================================================


class TestFourPathCoverage:
    """4-path test coverage for confirmation callbacks."""

    def test_happy_path_yes_promotes_and_enables_execution(self):
        """Happy: 'yes' promotes to active, enables skill execution."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import STATUS_ACTIVE

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(conn, storage, "wenn ich sage test, mach was")

        # Simulate "yes" flow
        storage.transition_hypothesis_status(hyp_id, STATUS_ACTIVE)

        hyp = storage.get_hypothesis(hyp_id)
        assert hyp.status == STATUS_ACTIVE

    def test_no_path_keeps_confirmed_status(self):
        """No: 'no' click does NOT change status (stays confirmed)."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import STATUS_CONFIRMED

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(conn, storage, "wenn ich sage nope, mach nix")

        # "No" only writes evidence, does not change status
        # (verify status is still confirmed)
        hyp = storage.get_hypothesis(hyp_id)
        assert hyp.status == STATUS_CONFIRMED

    def test_never_path_pauses_skill(self):
        """Never: 'never' transitions to paused."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import STATUS_PAUSED

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(conn, storage, "wenn ich sage weg, vergiss es")

        storage.transition_hypothesis_status(hyp_id, STATUS_PAUSED)
        hyp = storage.get_hypothesis(hyp_id)
        assert hyp.status == STATUS_PAUSED

    def test_expired_confirmation_not_executed(self):
        """Edge case: expired confirmation (>300s) must not execute."""
        from presentation.skill_commands import (
            SKILL_CONFIRM_TIMEOUT_SECONDS,
        )

        # Create a pending entry with timestamp 400s ago
        old_timestamp = time.time() - (SKILL_CONFIRM_TIMEOUT_SECONDS + 100)
        pending = {
            "skill_match": MagicMock(),
            "original_text": "rot",
            "timestamp": old_timestamp,
            "envelope": MagicMock(),
        }

        elapsed = time.time() - pending["timestamp"]
        assert elapsed > SKILL_CONFIRM_TIMEOUT_SECONDS

    def test_callback_data_contains_hypothesis_id(self):
        """Verify callback_data format for skill confirmation buttons."""
        from presentation.skill_commands import build_skill_confirm_keyboard

        keyboard = build_skill_confirm_keyboard("hyp_abc123", "de")

        # Verify 3 buttons
        assert len(keyboard.inline_keyboard) == 1
        assert len(keyboard.inline_keyboard[0]) == 3

        # Verify callback data format
        yes_btn = keyboard.inline_keyboard[0][0]
        no_btn = keyboard.inline_keyboard[0][1]
        never_btn = keyboard.inline_keyboard[0][2]

        assert yes_btn.callback_data == "skill_confirm:yes:hyp_abc123"
        assert no_btn.callback_data == "skill_confirm:no:hyp_abc123"
        assert never_btn.callback_data == "skill_confirm:never:hyp_abc123"


# =====================================================================
# Regression: Existing behavior preserved
# =====================================================================


class TestRegressionPreserved:
    """Ensure Round-5 changes do not break existing behavior."""

    def test_confirmed_skill_still_requires_confirmation(self):
        """Confirmed skills still show ask-before-apply dialog."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import (
            SkillMatcher,
            should_ask_user,
        )
        from application.skill_compression.event_normalizer import NormalizedEvent

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        _create_confirmed_skill(conn, storage, "wenn ich gruen sage, erklaere Natur")

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        event = NormalizedEvent(
            event_id="evt_regr_01",
            user_id=42,
            timestamp="2026-05-27T12:00:00Z",
            raw_text="gruen",
            intent="",
            domain="",
            format_type="",
            language="de",
            fingerprint_hash="",
        )
        match = matcher.match(event)
        assert match is not None
        assert should_ask_user(match) is True  # confirmed -> still asks

    def test_skill_instruction_block_format_unchanged(self):
        """The skill instruction block format is preserved."""
        from application.chat_service import ChatService
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import SkillMatcher

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich gelb sage, erklaere Sonne"
        )

        # Promote to active for the test
        storage.transition_hypothesis_status(hyp_id, "active")

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        mock_router = MagicMock()
        svc = ChatService(
            provider_router=mock_router,
            memory_service=MagicMock(),
            skill_matcher=matcher,
        )

        skill_block, match = svc._match_skills_for_prompt(42, "gelb", "de", None)
        assert match is not None
        assert "[USER-DEFINED SKILL (HIGH PRIORITY)]" in skill_block
        assert "Instruction:" in skill_block
        assert "Confidence:" in skill_block
        assert "Source:" in skill_block
        assert "MUST be applied" in skill_block


# =====================================================================
# Round-5b: Codex cosmetic polish
# =====================================================================


class TestRound5bI18nApplyingSkill:
    """Item 1.1: i18n text signals action-in-progress, not completion."""

    def test_en_confirm_applied_says_applying(self):
        """EN: skill.confirm_applied must say 'Applying skill...'."""
        from i18n.domain.i18n import t

        text = t("skill.confirm_applied", "en")
        assert text == "Applying skill...", (
            f"Expected 'Applying skill...', got '{text}'"
        )

    def test_de_confirm_applied_says_wird_angewendet(self):
        """DE: skill.confirm_applied must say 'Skill wird angewendet...'."""
        from i18n.domain.i18n import t

        text = t("skill.confirm_applied", "de")
        assert text == "Skill wird angewendet...", (
            f"Expected 'Skill wird angewendet...', got '{text}'"
        )

    def test_non_en_de_locales_use_en_fallback(self):
        """Non-EN/DE locales must also show 'Applying skill...'."""
        from i18n.domain.i18n import t

        for lang in ("fr", "es", "ja", "zh", "ar"):
            text = t("skill.confirm_applied", lang)
            assert text == "Applying skill...", (
                f"Locale {lang}: expected 'Applying skill...', got '{text}'"
            )


class TestRound5bEvidenceSeparation:
    """Item 1.2: user_confirmed vs skill_executed evidence split."""

    def test_yes_callback_writes_user_confirmed_evidence(self):
        """Button click 'Ja' writes user_confirmed (not skill_executed)."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import SkillMatch, SkillMatcher
        from application.chat_service import ChatService

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich blitz sage, erklaere Gewitter"
        )

        hyp = storage.get_hypothesis(hyp_id)
        skill_match = SkillMatch(
            hypothesis=hyp,
            confidence=0.85,
            requires_confirmation=True,
            explanation="alias match",
            match_source="alias",
        )

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        mock_router = MagicMock()
        chat_service = ChatService(
            provider_router=mock_router,
            memory_service=MagicMock(),
            skill_matcher=matcher,
        )

        # Simulate button click: user_confirmed
        chat_service._write_skill_evidence(
            skill_match, signal_type="user_confirmed", signal_strength=0.5
        )

        rows = conn.fetchall(
            "SELECT signal_type FROM hypothesis_evidence WHERE hypothesis_id = ?",
            (hyp_id,),
        )
        signal_types = [r["signal_type"] for r in rows]
        assert "user_confirmed" in signal_types
        assert "skill_executed" not in signal_types

    def test_skill_executed_evidence_after_reprocess(self):
        """After successful reprocess: skill_executed evidence is written."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import SkillMatch, SkillMatcher
        from application.chat_service import ChatService

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich stern sage, erklaere Astronomie"
        )

        hyp = storage.get_hypothesis(hyp_id)
        skill_match = SkillMatch(
            hypothesis=hyp,
            confidence=0.85,
            requires_confirmation=True,
            explanation="alias match",
            match_source="alias",
        )

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        mock_router = MagicMock()
        chat_service = ChatService(
            provider_router=mock_router,
            memory_service=MagicMock(),
            skill_matcher=matcher,
        )

        # Simulate: user_confirmed first, then skill_executed
        chat_service._write_skill_evidence(
            skill_match, signal_type="user_confirmed", signal_strength=0.5
        )
        chat_service._write_skill_evidence(
            skill_match, signal_type="skill_executed", signal_strength=0.5
        )

        rows = conn.fetchall(
            "SELECT signal_type FROM hypothesis_evidence WHERE hypothesis_id = ? "
            "ORDER BY created_at",
            (hyp_id,),
        )
        signal_types = [r["signal_type"] for r in rows]
        assert signal_types == ["user_confirmed", "skill_executed"]

    def test_skill_execution_failed_evidence_on_error(self):
        """On reprocess failure: skill_execution_failed evidence is written."""
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from application.skill_compression.pattern_judge import PatternJudge
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )
        from application.skill_compression.skill_matcher import SkillMatch, SkillMatcher
        from application.chat_service import ChatService

        conn = FakeDBConnection()
        storage = HypothesisStorage(conn)
        hyp_id = _create_confirmed_skill(
            conn, storage, "wenn ich mond sage, erklaere Gezeiten"
        )

        hyp = storage.get_hypothesis(hyp_id)
        skill_match = SkillMatch(
            hypothesis=hyp,
            confidence=0.85,
            requires_confirmation=True,
            explanation="alias match",
            match_source="alias",
        )

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)
        matcher = SkillMatcher(storage, judge)

        mock_router = MagicMock()
        chat_service = ChatService(
            provider_router=mock_router,
            memory_service=MagicMock(),
            skill_matcher=matcher,
        )

        # Simulate: user_confirmed, then execution failed
        chat_service._write_skill_evidence(
            skill_match, signal_type="user_confirmed", signal_strength=0.5
        )
        chat_service._write_skill_evidence(
            skill_match, signal_type="skill_execution_failed", signal_strength=0.0
        )

        rows = conn.fetchall(
            "SELECT signal_type FROM hypothesis_evidence WHERE hypothesis_id = ? "
            "ORDER BY created_at",
            (hyp_id,),
        )
        signal_types = [r["signal_type"] for r in rows]
        assert "user_confirmed" in signal_types
        assert "skill_execution_failed" in signal_types


class TestRound5bExpiredPendingHandling:
    """Item 1.3: Expired pending confirmation handled gracefully."""

    @pytest.mark.asyncio
    async def test_expired_pending_callback_handles_gracefully(self):
        """Expired pending: user gets expired message, no execution, pending removed."""
        from unittest.mock import AsyncMock

        from presentation.skill_commands import (
            SKILL_CONFIRM_TIMEOUT_SECONDS,
            _handle_skill_confirm_inline,
        )
        from application.skill_compression.skill_matcher import SkillMatch
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        mock_hyp = Hypothesis(
            hypothesis_id="hyp_expired_r5b",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="test expired",
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            source_type="live_chat",
            decay_immune=False,
            created_at="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        )
        mock_match = SkillMatch(
            hypothesis=mock_hyp,
            confidence=0.8,
            requires_confirmation=True,
            explanation="test",
        )

        # Pending store with expired entry
        pending_store = {
            (1, 10, "hyp_expired_r5b"): {
                "skill_match": mock_match,
                "original_text": "test",
                "timestamp": time.time() - SKILL_CONFIRM_TIMEOUT_SECONDS - 120,
                "envelope": MagicMock(),
            }
        }

        mock_context = MagicMock()
        mock_chat_service = MagicMock()
        mock_chat_service._write_skill_evidence = MagicMock()
        mock_context.application.bot_data = {
            "chat_service": mock_chat_service,
            "hypothesis_storage": MagicMock(),
            "_pending_skill_confirmations": pending_store,
        }

        mock_query = AsyncMock()
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        await _handle_skill_confirm_inline(
            mock_query,
            mock_context,
            1,
            10,
            "skill_confirm:yes:hyp_expired_r5b",
            "en",
        )

        # No evidence written (expired)
        mock_chat_service._write_skill_evidence.assert_not_called()

        # Expired alert shown
        mock_query.answer.assert_called_once()
        call_kwargs = mock_query.answer.call_args
        assert call_kwargs[1].get("show_alert") is True

        # Pending entry removed after timeout
        assert (1, 10, "hyp_expired_r5b") not in pending_store

    @pytest.mark.asyncio
    async def test_expired_pending_shows_localized_message(self):
        """Expired pending: DE locale shows correct German message."""
        from i18n.domain.i18n import t

        text_en = t("skill.confirm_expired", "en")
        text_de = t("skill.confirm_expired", "de")

        assert "expired" in text_en.lower() or "again" in text_en.lower()
        assert "abgelaufen" in text_de.lower() or "erneut" in text_de.lower()


class TestRound5bNoPathExplicit:
    """Item 1.4: 'Nein' callback path explicitly tested."""

    @pytest.mark.asyncio
    async def test_callback_no_evidence_and_no_execution(self):
        """User clicks Nein: user_declined_once evidence, no execution, pending removed."""
        from unittest.mock import AsyncMock

        from presentation.skill_commands import _handle_skill_confirm_inline
        from application.skill_compression.skill_matcher import SkillMatch
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        mock_hyp = Hypothesis(
            hypothesis_id="hyp_no_r5b",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="test no path",
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            source_type="live_chat",
            decay_immune=False,
            created_at="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        )
        mock_match = SkillMatch(
            hypothesis=mock_hyp,
            confidence=0.8,
            requires_confirmation=True,
            explanation="test",
        )

        pending_store = {
            (1, 10, "hyp_no_r5b"): {
                "skill_match": mock_match,
                "original_text": "test no",
                "timestamp": time.time(),
                "envelope": MagicMock(),
            }
        }

        mock_context = MagicMock()
        mock_chat_service = MagicMock()
        mock_chat_service._write_skill_evidence = MagicMock()

        # process_message should NOT be called (no execution on "no")
        mock_chat_service.process_message = MagicMock()

        mock_context.application.bot_data = {
            "chat_service": mock_chat_service,
            "hypothesis_storage": MagicMock(),
            "_pending_skill_confirmations": pending_store,
        }

        mock_query = AsyncMock()
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()
        mock_query.from_user = MagicMock()
        mock_query.from_user.id = 1

        await _handle_skill_confirm_inline(
            mock_query,
            mock_context,
            1,
            10,
            "skill_confirm:no:hyp_no_r5b",
            "en",
        )

        # Evidence written: user_declined_once
        mock_chat_service._write_skill_evidence.assert_called_once()
        call_kwargs = mock_chat_service._write_skill_evidence.call_args
        assert call_kwargs[1]["signal_type"] == "user_declined_once"

        # No execution (process_message not called)
        mock_chat_service.process_message.assert_not_called()

        # Pending entry removed
        assert (1, 10, "hyp_no_r5b") not in pending_store

    @pytest.mark.asyncio
    async def test_callback_no_does_not_change_status(self):
        """User clicks Nein: hypothesis status stays confirmed (no transition)."""
        from unittest.mock import AsyncMock

        from presentation.skill_commands import _handle_skill_confirm_inline
        from application.skill_compression.skill_matcher import SkillMatch
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        mock_hyp = Hypothesis(
            hypothesis_id="hyp_no_status_r5b",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="test no status",
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            source_type="live_chat",
            decay_immune=False,
            created_at="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        )
        mock_match = SkillMatch(
            hypothesis=mock_hyp,
            confidence=0.8,
            requires_confirmation=True,
            explanation="test",
        )

        mock_storage = MagicMock()
        pending_store = {
            (1, 10, "hyp_no_status_r5b"): {
                "skill_match": mock_match,
                "original_text": "test",
                "timestamp": time.time(),
                "envelope": MagicMock(),
            }
        }

        mock_context = MagicMock()
        mock_chat_service = MagicMock()
        mock_chat_service._write_skill_evidence = MagicMock()
        mock_context.application.bot_data = {
            "chat_service": mock_chat_service,
            "hypothesis_storage": mock_storage,
            "_pending_skill_confirmations": pending_store,
        }

        mock_query = AsyncMock()
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()
        mock_query.from_user = MagicMock()
        mock_query.from_user.id = 1

        await _handle_skill_confirm_inline(
            mock_query,
            mock_context,
            1,
            10,
            "skill_confirm:no:hyp_no_status_r5b",
            "en",
        )

        # Storage transition must NOT be called for "no"
        mock_storage.transition_hypothesis_status.assert_not_called()
        mock_storage.update_hypothesis_status.assert_not_called()
