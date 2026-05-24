"""K7: LCP (Language Consistency Protocol) state manipulation tests.

Sticky-language flip-flop, mixed-language boundary,
very short inputs at detection threshold, StreamGuard
manual disable/re-enable, FP counter overflow.
"""

from __future__ import annotations

import pytest

from application.language.stream_guard import (
    StreamGuard,
    StreamGuardStats,
    StreamGuardStatsStore,
    _MIN_CHARS_FOR_CHECK,
)
from application.language.verifier import (
    ResponseLanguageVerifier,
    VerificationStatus,
)
from application.language.context import LanguageContext


@pytest.mark.adversarial
class TestStickyLanguageFlipFlop:
    """Rapid language switching in LanguageContext."""

    def test_language_context_immutability(self) -> None:
        """WHAT: Attempt to modify frozen LanguageContext.
        EXPECTED: AttributeError (dataclass is frozen).
        WHY: Language context must be immutable once created.
        """
        ctx = LanguageContext(
            code="de",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="test-001",
        )
        with pytest.raises(AttributeError):
            ctx.code = "en"  # type: ignore[misc]

    def test_different_contexts_per_language(self) -> None:
        """WHAT: Create multiple LanguageContext for different languages.
        EXPECTED: Each is independent, no shared state.
        WHY: Language flip-flop should create new contexts, not mutate.
        """
        de_ctx = LanguageContext(
            code="de",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="a",
        )
        en_ctx = LanguageContext(
            code="en",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="b",
        )
        assert de_ctx.code == "de"
        assert en_ctx.code == "en"
        assert de_ctx is not en_ctx


@pytest.mark.adversarial
class TestMixedLanguageInputs:
    """Mixed-language text at detection boundaries."""

    def test_verifier_with_exactly_50_50_mixed(self) -> None:
        """WHAT: Text with exactly 50% German, 50% English.
        EXPECTED: Verifier returns a result (PASS/WARN/FAIL), no crash.
        WHY: Mixed-language text is the hardest case for detection.
        """
        verifier = ResponseLanguageVerifier()
        # Mix German and English sentences
        text = (
            "Heute ist ein guter Tag. The weather is nice. "
            "Morgen wird es regnen. Tomorrow will be sunny. "
            "Das Buch ist interessant. The book is fascinating. "
            "Wir gehen zum Markt. We go to the market. "
            "Die Katze sitzt dort. The cat sits there. "
        )
        result = verifier.verify(text, expected_lang="de")
        assert result.status in (
            VerificationStatus.PASS,
            VerificationStatus.WARN,
            VerificationStatus.FAIL,
        )

    def test_verifier_with_code_heavy_text(self) -> None:
        """WHAT: Text that is mostly code (language-neutral).
        EXPECTED: Should pass (code is stripped before detection).
        WHY: Programming code should not affect language detection.
        """
        verifier = ResponseLanguageVerifier()
        text = (
            "Hier ist der Code:\n"
            "```python\n"
            "def hello_world():\n"
            "    print('Hello, World!')\n"
            "    return True\n"
            "```\n"
            "Das war der Code."
        )
        result = verifier.verify(text, expected_lang="de")
        # Short remaining text after code removal may skip verification
        assert (
            result.status in (VerificationStatus.PASS, VerificationStatus.WARN)
            or result.skipped
        )


@pytest.mark.adversarial
class TestShortInputsAtThreshold:
    """Very short inputs at _LCP_AWARE_MIN_CHARS boundary."""

    def test_stream_guard_just_below_min_chars(self) -> None:
        """WHAT: Partial stream at exactly MIN_CHARS - 1.
        EXPECTED: No check performed, returns True (continue).
        WHY: Off-by-one at the minimum character threshold.
        """
        guard = StreamGuard(expected_lang="de", enabled=True)
        text = "x" * (_MIN_CHARS_FOR_CHECK - 1)
        result = guard.check_early(text)
        assert result is True
        assert guard.state.check_performed is False

    def test_stream_guard_exactly_at_min_chars(self) -> None:
        """WHAT: Partial stream at exactly MIN_CHARS.
        EXPECTED: Check is performed.
        WHY: Boundary condition at minimum threshold.
        """
        guard = StreamGuard(expected_lang="de", enabled=True)
        text = "a" * _MIN_CHARS_FOR_CHECK
        _ = guard.check_early(text)
        # Check should have been performed
        assert guard.state.check_performed is True

    def test_verifier_below_min_words(self) -> None:
        """WHAT: Response with fewer than 20 words.
        EXPECTED: Verification skipped, returns PASS.
        WHY: Short text detection is unreliable.
        """
        verifier = ResponseLanguageVerifier()
        result = verifier.verify("Kurze Antwort hier.", expected_lang="de")
        assert result.skipped is True
        assert result.status == VerificationStatus.PASS


@pytest.mark.adversarial
class TestStreamGuardDisableReEnable:
    """StreamGuard manual disable then session reset."""

    def test_disabled_guard_skips_check(self) -> None:
        """WHAT: Guard disabled via state.disabled = True.
        EXPECTED: check_early always returns True.
        WHY: Disabled guard must not abort streams.
        """
        guard = StreamGuard(expected_lang="de", enabled=True)
        guard._state.disabled = True
        text = "This is entirely in English and should normally trigger abort." * 5
        result = guard.check_early(text)
        assert result is True
        assert guard.state.check_performed is False

    def test_reenabled_new_instance_works(self) -> None:
        """WHAT: New StreamGuard after previous was disabled.
        EXPECTED: New instance works independently.
        WHY: State should not leak between guard instances.
        """
        old_guard = StreamGuard(expected_lang="de", enabled=True)
        old_guard._state.disabled = True

        new_guard = StreamGuard(expected_lang="de", enabled=True)
        assert new_guard.is_active is True
        # New guard should be able to perform checks
        text = "a" * _MIN_CHARS_FOR_CHECK
        new_guard.check_early(text)
        assert new_guard.state.check_performed is True


@pytest.mark.adversarial
class TestFPCounterOverflow:
    """False positive counter at extreme values."""

    def test_fp_rate_with_zero_aborts(self) -> None:
        """WHAT: FP rate calculation when total_aborts is 0.
        EXPECTED: Returns 0.0, no ZeroDivisionError.
        WHY: Division by zero in fp_rate property.
        """
        stats = StreamGuardStats()
        assert stats.fp_rate == 0.0

    def test_massive_fp_count(self) -> None:
        """WHAT: 10000 consecutive false positives.
        EXPECTED: should_disable is True, no overflow.
        WHY: Tests counter behavior at extreme values.
        """
        stats = StreamGuardStats()
        stats.total_aborts = 10000
        stats.false_positives = 10000
        stats.consecutive_fp = 10000
        assert stats.should_disable is True
        assert stats.fp_rate == 1.0

    def test_stats_store_eviction_with_full_stats(self) -> None:
        """WHAT: Stats store at capacity with all entries having high FP counts.
        EXPECTED: LRU eviction works correctly, no data leak.
        WHY: Tests that evicted entries don't corrupt remaining entries.
        """
        store = StreamGuardStatsStore(max_entries=5)
        for i in range(10):
            stats = store.get(i, 0)
            stats.total_checks = 100
            stats.total_aborts = 50
            stats.false_positives = 25

        all_stats = store.all_stats()
        assert len(all_stats) <= 5
        # All remaining entries should have consistent state
        for key, stats in all_stats.items():
            assert stats.total_checks == 100
