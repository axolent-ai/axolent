"""Tests for StreamGuard: early streaming abort logic."""

from application.language.stream_guard import (
    StreamGuard,
    StreamGuardStats,
    StreamGuardStatsStore,
    _MAX_CONSECUTIVE_FP,
    _MIN_CHARS_FOR_CHECK,
)


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
