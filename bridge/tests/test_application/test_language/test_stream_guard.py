"""Tests for StreamGuard: early streaming abort logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from application.chat_service import ChatService
from application.language.stream_guard import (
    StreamGuard,
    StreamGuardOutcome,
    StreamGuardStats,
    StreamGuardStatsStore,
    _EARLY_ABORT_CONFIDENCE,
    _MAX_CONSECUTIVE_FP,
    _MIN_CHARS_FOR_CHECK,
)
from infrastructure.claude_process_pool import StreamEvent
from infrastructure.conversation_storage import _reset_all_for_tests


class TestStreamGuardBasics:
    """Basic StreamGuard behavior."""

    def test_inactive_guard_always_continues(self) -> None:
        """Disabled guard never aborts."""
        guard = StreamGuard(expected_lang="de", enabled=False)
        assert guard.check_early("This is English " * 50) is True
        assert guard.is_active is False

    def test_short_text_continues(self) -> None:
        """Text shorter than minimum chars always continues."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        assert guard.check_early("Kurz") is True
        assert guard.state.check_performed is False

    def test_check_performed_once(self) -> None:
        """Check is only performed once per session."""
        guard = StreamGuard(expected_lang="de", enabled=True)

        # First call with enough text
        german_text = "Dies ist ein sehr langer deutscher Text " * 10
        result1 = guard.check_early(german_text)
        assert guard.state.check_performed is True

        # Subsequent calls don't re-check
        result2 = guard.check_early(german_text + " noch mehr Text")
        assert result2 == result1  # Same result, no re-check

    def test_german_text_continues_for_german(self) -> None:
        """German text continues when German is expected."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        text = (
            "Dies ist ein sehr langer deutscher Text der genug Zeichen hat "
            "um die Mindestlänge zu überschreiten. Wir brauchen mindestens "
            "zweihundert Zeichen damit der StreamGuard überhaupt prüft. "
            "Dieser zusätzliche Satz stellt sicher dass wir weit über der "
            "Mindestanforderung von zweihundert Zeichen liegen."
        )
        assert len(text) >= _MIN_CHARS_FOR_CHECK
        result = guard.check_early(text)
        assert result is True
        assert guard.state.aborted is False


class TestStreamGuardAbort:
    """Tests for abort signaling."""

    def test_clear_wrong_language_aborts(self) -> None:
        """Clear English text aborts when German expected."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        # Very clearly English text, long enough
        text = (
            "This is a very clearly English text that should be detected "
            "as English with high confidence by the language detection system. "
            "It contains many typical English words and sentence structures "
            "that make it unmistakably English to any language detector."
        )
        assert len(text) >= _MIN_CHARS_FOR_CHECK
        result = guard.check_early(text)
        # Result depends on detection confidence
        if not result:
            assert guard.state.aborted is True
            assert guard.state.detected_lang_at_abort is not None

    def test_after_abort_subsequent_checks_return_false(self) -> None:
        """Once aborted, all subsequent checks return False."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        # Force abort state
        guard._state.check_performed = True
        guard._state.aborted = True
        assert guard.check_early("more text " * 50) is False


class TestStreamGuardStats:
    """Tests for StreamGuardStats and self-calibration."""

    def test_stats_initial_state(self) -> None:
        """Fresh stats are all zeros."""
        stats = StreamGuardStats()
        assert stats.total_checks == 0
        assert stats.total_aborts == 0
        assert stats.false_positives == 0
        assert stats.fp_rate == 0.0
        assert stats.should_disable is False

    def test_fp_rate_calculation(self) -> None:
        """FP rate is correctly calculated."""
        stats = StreamGuardStats(total_aborts=10, false_positives=2)
        assert stats.fp_rate == 0.2

    def test_should_disable_on_consecutive_fp(self) -> None:
        """Auto-disable triggers after consecutive FPs."""
        stats = StreamGuardStats(
            consecutive_fp=_MAX_CONSECUTIVE_FP,
            total_aborts=_MAX_CONSECUTIVE_FP,
            false_positives=_MAX_CONSECUTIVE_FP,
        )
        assert stats.should_disable is True

    def test_should_not_disable_with_low_fp(self) -> None:
        """No disable when FP rate is low."""
        stats = StreamGuardStats(
            total_aborts=20,
            false_positives=0,
            consecutive_fp=0,
        )
        assert stats.should_disable is False


class TestStreamGuardCalibration:
    """Tests for report_final_outcome and auto-disable."""

    def test_correct_abort_resets_consecutive_fp(self) -> None:
        """Confirmed abort resets the consecutive FP counter."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        guard._state.check_performed = True
        guard._state.aborted = True

        stats = StreamGuardStats(consecutive_fp=2)
        guard.report_final_outcome(verification_passed=False, stats=stats)

        assert stats.confirmed_aborts == 1
        assert stats.consecutive_fp == 0

    def test_false_positive_increments_counter(self) -> None:
        """False positive increments consecutive FP counter."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        guard._state.check_performed = True
        guard._state.aborted = True

        stats = StreamGuardStats()
        guard.report_final_outcome(verification_passed=True, stats=stats)

        assert stats.false_positives == 1
        assert stats.consecutive_fp == 1

    def test_auto_disable_on_fp_threshold(self) -> None:
        """Guard auto-disables when FP threshold is exceeded."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        guard._state.check_performed = True
        guard._state.aborted = True

        stats = StreamGuardStats(
            total_aborts=2,
            false_positives=2,
            consecutive_fp=_MAX_CONSECUTIVE_FP - 1,
        )
        guard.report_final_outcome(verification_passed=True, stats=stats)

        assert stats.consecutive_fp == _MAX_CONSECUTIVE_FP
        assert guard.state.disabled is True
        assert guard.state.disable_reason is not None

    def test_no_stats_means_no_tracking(self) -> None:
        """Without stats object, nothing is tracked."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        guard._state.check_performed = True
        guard._state.aborted = True
        # Should not raise
        guard.report_final_outcome(verification_passed=True, stats=None)


class TestStreamGuardAudit:
    """Tests for audit entry generation."""

    def test_audit_entry_basic(self) -> None:
        """Audit entry contains expected fields."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        entry = guard.build_audit_entry()

        assert entry["event_type"] == "stream_guard_check"
        assert entry["expected_lang"] == "de"
        assert entry["check_performed"] is False
        assert entry["aborted"] is False

    def test_audit_entry_after_abort(self) -> None:
        """Audit entry captures abort details."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        guard._state.check_performed = True
        guard._state.aborted = True
        guard._state.detected_lang_at_abort = "en"

        entry = guard.build_audit_entry()
        assert entry["aborted"] is True
        assert entry["detected_at_abort"] == "en"


class TestStreamGuardStatsStore:
    """Issue 1: StreamGuardStatsStore provides per-session stats."""

    def test_get_creates_stats_on_first_access(self) -> None:
        """First access for a (user, chat) pair creates fresh stats."""
        store = StreamGuardStatsStore()
        stats = store.get(user_id=1, chat_id=100)
        assert isinstance(stats, StreamGuardStats)
        assert stats.total_checks == 0
        assert stats.total_aborts == 0

    def test_get_returns_same_stats_for_same_key(self) -> None:
        """Repeated access for same (user, chat) returns the same object."""
        store = StreamGuardStatsStore()
        stats_a = store.get(user_id=1, chat_id=100)
        stats_a.total_checks = 5
        stats_b = store.get(user_id=1, chat_id=100)
        assert stats_b is stats_a
        assert stats_b.total_checks == 5

    def test_get_returns_different_stats_for_different_key(self) -> None:
        """Different (user, chat) pairs get independent stats."""
        store = StreamGuardStatsStore()
        stats_1 = store.get(user_id=1, chat_id=100)
        stats_2 = store.get(user_id=2, chat_id=200)
        stats_1.total_checks = 10
        assert stats_2.total_checks == 0

    def test_all_stats_returns_snapshot(self) -> None:
        """all_stats returns a dict copy of all stored entries."""
        store = StreamGuardStatsStore()
        store.get(user_id=1, chat_id=100)
        store.get(user_id=2, chat_id=200)
        snapshot = store.all_stats()
        assert len(snapshot) == 2
        assert (1, 100) in snapshot
        assert (2, 200) in snapshot

    def test_stats_accumulate_across_sessions(self) -> None:
        """Stats accumulate across multiple StreamGuard sessions."""
        store = StreamGuardStatsStore()
        stats = store.get(user_id=1, chat_id=100)

        # Simulate 3 sessions with false positives
        for _ in range(3):
            guard = StreamGuard(expected_lang="de", enabled=True)
            guard._state.check_performed = True
            guard._state.aborted = True
            guard.report_final_outcome(verification_passed=True, stats=stats)

        assert stats.consecutive_fp == 3
        assert stats.false_positives == 3
        assert stats.should_disable is True


class TestStreamGuardAbortWritesStats:
    """Codex Finding 1: StreamGuard abort path MUST write stats.

    This tests the real abort path in ChatService._stream():
    when StreamGuard detects a violation during streaming,
    cancel_event is set and the stream generator returns.
    Before the fix, save_streaming_result() was never reached
    so report_final_outcome() was never called and the StatsStore
    never saw the abort. The fix calls report_final_outcome()
    directly in the abort path before setting cancel_event.
    """

    async def test_abort_writes_stats_to_store(self) -> None:
        """StatsStore gets an abort entry when StreamGuard triggers."""
        _reset_all_for_tests()

        # Build a mock LanguageEnforcement that enables StreamGuard creation
        mock_enforcement = MagicMock()

        # Build ChatService with language_enforcement so StatsStore is created
        mock_router = MagicMock()
        svc = ChatService(
            provider_router=mock_router,
            language_enforcement=mock_enforcement,
        )

        # Verify StatsStore was created
        assert svc._stream_guard_stats_store is not None

        # Build a mock persistent provider that yields English tokens
        # (enough to trigger StreamGuard check at 200+ chars)
        english_tokens = [
            StreamEvent(event_type="init", was_cold=False, subprocess_pid=999),
        ]
        # Generate English content_delta tokens that will trigger abort
        english_text = (
            "This is clearly English text that should trigger "
            "the StreamGuard abort because it is not German. "
            "The language detection backend will see this as "
            "English with very high confidence because every "
            "single word in this text is in English language."
        )
        # Yield the text in one chunk (over 200 chars)
        english_tokens.append(
            StreamEvent(event_type="content_delta", text=english_text)
        )

        mock_provider = MagicMock()

        async def _fake_streaming(**kwargs):  # noqa: ANN003
            for tok in english_tokens:
                yield tok

        mock_provider.query_streaming = _fake_streaming

        # Mock the language detection backend to return "en" with high confidence
        mock_backend = MagicMock()
        mock_backend.detect_distribution.return_value = {
            "en": _EARLY_ABORT_CONFIDENCE + 0.05,
            "de": 0.05,
        }

        # Mock model profile to enable stream_guard
        mock_profile = MagicMock()
        mock_profile.stream_guard_enabled = True

        # Mock LanguageResolver to return German context
        from application.language.context import LanguageContext

        mock_lang_ctx = LanguageContext(
            code="de",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="test-abort-stats",
        )
        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve = AsyncMock(return_value=mock_lang_ctx)

        cancel_event = asyncio.Event()

        # Patch external dependencies:
        # 1. get_profile to return our mock profile (local import in chat_service)
        # 2. LangdetectBackend to use our mock backend
        # 3. get_history to return empty list
        # 4. LanguageResolver to return German context
        # 5. write_audit_log to suppress I/O
        with (
            patch(
                "application.language.model_profiles.get_profile",
                return_value=mock_profile,
            ),
            patch(
                "application.language.stream_guard.LangdetectBackend",
                return_value=mock_backend,
            ),
            patch(
                "application.chat_service.get_history",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "application.language_resolver.LanguageResolver",
                return_value=mock_resolver_instance,
            ),
            patch(
                "application.chat_service.write_audit_log",
            ),
        ):
            (
                stream_iter,
                mem_count,
                task_meta,
            ) = await svc.process_user_message_streaming(
                text="Wie geht es dir?",
                user_id=42,
                chat_id=100,
                username="testuser",
                system_prompt="Du bist hilfreich.",
                persistent_provider=mock_provider,
                cancel_event=cancel_event,
            )

            # Consume the stream (the abort happens inside _stream())
            events = []
            async for event in stream_iter:
                events.append(event)

        # cancel_event must have been set by the abort path
        assert cancel_event.is_set(), (
            "cancel_event should be set after StreamGuard abort"
        )

        # The stream_guard in task_meta should have aborted
        stream_guard = task_meta.get("_stream_guard")
        assert stream_guard is not None
        assert stream_guard.state.aborted is True

        # THE CRITICAL ASSERTION: StatsStore must have the abort recorded
        stats = svc._stream_guard_stats_store.get(42, 100)
        assert stats.total_checks >= 1, (
            "StatsStore must have at least 1 check after abort"
        )
        assert stats.total_aborts >= 1, (
            "StatsStore must have at least 1 abort entry "
            "(this fails without the fix because report_final_outcome "
            "was never called in the abort path)"
        )
        assert stats.confirmed_aborts >= 1, (
            "Abort with clear wrong-language partial text counts as "
            "CONFIRMED_ABORT via classify_and_report_abort()"
        )


# ---------------------------------------------------------------------------
# FP-Detection Fix: Partial Verification + Outcome Classification
# ---------------------------------------------------------------------------


class _StubBackend:
    """Configurable stub backend for partial-verification tests.

    Returns a fixed distribution regardless of input text.
    """

    def __init__(self, distribution: dict[str, float]) -> None:
        self._distribution = distribution

    def detect_distribution(self, text: str) -> dict[str, float]:
        return dict(self._distribution)


class TestPartialVerificationOutcomes:
    """FP-Detection fix: classify_and_report_abort() partial verification."""

    def test_wrong_language_partial_gives_confirmed_abort(self) -> None:
        """Clear wrong-language partial text: CONFIRMED_ABORT.

        Backend returns high confidence for wrong lang, near-zero
        for target lang. This is the happy path where StreamGuard
        was right to abort.
        """
        backend = _StubBackend({"en": 0.92, "de": 0.03})
        guard = StreamGuard(expected_lang="de", enabled=True, backend=backend)
        guard._state.check_performed = True
        guard._state.aborted = True
        guard._state.detected_lang_at_abort = "en"

        stats = StreamGuardStats()
        outcome = guard.classify_and_report_abort(
            accumulated_text="This is clearly English text " * 10,
            stats=stats,
        )

        assert outcome == StreamGuardOutcome.CONFIRMED_ABORT
        assert guard.state.outcome == StreamGuardOutcome.CONFIRMED_ABORT
        assert stats.confirmed_aborts == 1
        assert stats.false_positives == 0
        assert stats.unknown_aborts == 0
        assert stats.consecutive_fp == 0

    def test_target_language_partial_gives_false_positive_abort(self) -> None:
        """Target language present in partial text: FALSE_POSITIVE_ABORT.

        Backend shows significant target-lang presence (>= 0.30).
        StreamGuard should not have aborted. This is the scenario
        that was previously invisible (always counted as confirmed).
        """
        backend = _StubBackend({"de": 0.45, "en": 0.40})
        guard = StreamGuard(expected_lang="de", enabled=True, backend=backend)
        guard._state.check_performed = True
        guard._state.aborted = True
        guard._state.detected_lang_at_abort = "en"

        stats = StreamGuardStats()
        outcome = guard.classify_and_report_abort(
            accumulated_text="mixed text " * 10,
            stats=stats,
        )

        assert outcome == StreamGuardOutcome.FALSE_POSITIVE_ABORT
        assert stats.false_positives == 1
        assert stats.confirmed_aborts == 0
        assert stats.consecutive_fp == 1

    def test_low_confidence_partial_gives_unknown_abort(self) -> None:
        """Ambiguous partial text: UNKNOWN_ABORT.

        Target lang has some presence (0.10 <= prob < 0.30) but not
        enough to call it a false positive. Detected lang also not
        dominant enough for confirmed. This is the genuinely unclear
        case that previously was lumped in with confirmed aborts.
        """
        backend = _StubBackend({"en": 0.55, "de": 0.20, "nl": 0.15})
        guard = StreamGuard(expected_lang="de", enabled=True, backend=backend)
        guard._state.check_performed = True
        guard._state.aborted = True
        guard._state.detected_lang_at_abort = "en"

        stats = StreamGuardStats()
        outcome = guard.classify_and_report_abort(
            accumulated_text="ambiguous text " * 10,
            stats=stats,
        )

        assert outcome == StreamGuardOutcome.UNKNOWN_ABORT
        assert stats.unknown_aborts == 1
        assert stats.confirmed_aborts == 0
        assert stats.false_positives == 0

    def test_false_positive_abort_increments_consecutive_fp(self) -> None:
        """FALSE_POSITIVE_ABORT must increment consecutive_fp.

        This is the core fix: consecutive_fp now actually gets
        incremented when partial verification detects a false
        positive, enabling the auto-disable mechanism.
        """
        backend = _StubBackend({"de": 0.50, "en": 0.35})
        stats = StreamGuardStats(consecutive_fp=1)

        guard = StreamGuard(expected_lang="de", enabled=True, backend=backend)
        guard._state.check_performed = True
        guard._state.aborted = True
        guard._state.detected_lang_at_abort = "en"

        guard.classify_and_report_abort(
            accumulated_text="text " * 50,
            stats=stats,
        )

        assert stats.consecutive_fp == 2
        assert stats.false_positives == 1

    def test_unknown_aborts_do_not_disable_guard(self) -> None:
        """5x UNKNOWN_ABORT must NOT disable StreamGuard.

        UNKNOWN means "we don't know if the abort was right".
        Disabling the guard based on uncertainty would be wrong.
        Only confirmed false positives (consecutive_fp >= 3) should
        trigger auto-disable.
        """
        backend = _StubBackend({"en": 0.55, "de": 0.20, "nl": 0.15})
        stats = StreamGuardStats()

        for _ in range(5):
            guard = StreamGuard(expected_lang="de", enabled=True, backend=backend)
            guard._state.check_performed = True
            guard._state.aborted = True
            guard._state.detected_lang_at_abort = "en"
            guard.classify_and_report_abort(
                accumulated_text="ambiguous " * 30,
                stats=stats,
            )

        assert stats.unknown_aborts == 5
        assert stats.consecutive_fp == 0
        assert stats.should_disable is False

    def test_audit_entry_contains_outcome_no_user_text(self) -> None:
        """Audit entry must contain outcome but NEVER user text.

        Privacy constraint: the audit log must not leak accumulated
        stream content. It may contain language codes, confidence
        scores, text length, and outcome classification.
        """
        backend = _StubBackend({"en": 0.90, "de": 0.05})
        guard = StreamGuard(expected_lang="de", enabled=True, backend=backend)
        guard._state.check_performed = True
        guard._state.aborted = True
        guard._state.detected_lang_at_abort = "en"
        guard._state.abort_confidence = 0.90
        guard._state.partial_text_length = 248

        guard.classify_and_report_abort(
            accumulated_text="This is secret user text that must not appear " * 5,
        )

        entry = guard.build_audit_entry()

        # Must contain outcome metadata
        assert "outcome" in entry
        assert entry["outcome"] == "confirmed_abort"
        assert "expected_lang" in entry
        assert "detected_at_abort" in entry
        assert "partial_length" in entry
        assert entry["partial_length"] == 248
        assert "partial_verification_confidence" in entry
        assert "abort_confidence" in entry

        # Must NOT contain user text
        all_values = " ".join(str(v) for v in entry.values())
        assert "secret user text" not in all_values
        assert "accumulated" not in entry
        assert "text" not in entry  # no key called "text" or similar
