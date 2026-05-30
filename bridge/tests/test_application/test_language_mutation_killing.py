"""Mutation-killing tests for language/ module.

These tests target specific decision points, thresholds, boundary values,
and branching logic in the Language Control Plane that mutmut mutants
typically survive.

Target: raise language/ mutation score from 38.6% toward 80%.

Covers:
  - verifier.py: threshold boundaries, status determination, cleaning logic
  - stream_guard.py: abort thresholds, state transitions, auto-disable
  - enforcement.py: profile-gated skip, repair pipeline flow
  - model_profiles.py: profile lookup, default fallback
"""

from __future__ import annotations


import pytest

from application.language.context import LanguageContext
from application.language.enforcement import (
    LanguageEnforcement,
)
from application.language.model_profiles import (
    ModelAdherenceProfile,
    get_profile,
)
from application.language.stream_guard import (
    StreamGuard,
    StreamGuardOutcome,
    StreamGuardStats,
    StreamGuardStatsStore,
    _MAX_CONSECUTIVE_FP,
    _MIN_CHARS_FOR_CHECK,
)
from application.language.verifier import (
    ResponseLanguageVerifier,
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------
# Helpers: mock backend for deterministic tests
# ---------------------------------------------------------------


class MockBackend:
    """Deterministic mock for LanguageDetectorBackend."""

    def __init__(self, distribution: dict[str, float] | None = None) -> None:
        self._distribution = distribution or {}

    def detect_distribution(self, text: str) -> dict[str, float]:
        return dict(self._distribution)


# ===============================================================
# VERIFIER: mutation-killing tests
# ===============================================================


class TestVerifierStatusDetermination:
    """Kill mutants in _determine_status threshold logic."""

    def test_low_confidence_always_pass(self) -> None:
        """Mutant: confidence < 0.5 check removed/inverted."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="en",
            expected_lang="de",
            confidence=0.49,
            target_ratio=0.0,
        )
        assert result == VerificationStatus.PASS

    def test_confidence_exactly_0_5_not_low(self) -> None:
        """Mutant: < changed to <= at 0.5 boundary."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="en",
            expected_lang="de",
            confidence=0.5,
            target_ratio=0.1,
        )
        # confidence >= 0.5, detected != expected, target_ratio < 0.6 -> FAIL
        assert result == VerificationStatus.FAIL

    def test_detected_matches_expected_high_ratio_pass(self) -> None:
        """Mutant: target_ratio > 0.8 threshold changed."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="de",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.81,
        )
        assert result == VerificationStatus.PASS

    def test_detected_matches_expected_ratio_exactly_0_8_warn(self) -> None:
        """Mutant: > changed to >= at 0.8 boundary."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="de",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.8,
        )
        # target_ratio == 0.8, not > 0.8, so should be WARN (0.6..0.8)
        assert result == VerificationStatus.WARN

    def test_detected_matches_expected_warn_range(self) -> None:
        """Mutant: >= 0.6 check removed."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="de",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.7,
        )
        assert result == VerificationStatus.WARN

    def test_detected_matches_expected_ratio_exactly_0_6_warn(self) -> None:
        """Mutant: >= changed to > at 0.6 boundary."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="de",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.6,
        )
        assert result == VerificationStatus.WARN

    def test_detected_matches_expected_low_ratio_fail(self) -> None:
        """Mutant: FAIL branch for detected==expected with ratio < 0.6 removed."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="de",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.59,
        )
        assert result == VerificationStatus.FAIL

    def test_mismatch_but_high_target_ratio_pass(self) -> None:
        """Mutant: target_ratio > 0.8 branch missing in mismatch path."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="en",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.81,
        )
        assert result == VerificationStatus.PASS

    def test_mismatch_medium_target_ratio_warn(self) -> None:
        """Mutant: WARN path missing in mismatch branch."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="en",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.65,
        )
        assert result == VerificationStatus.WARN

    def test_mismatch_low_target_ratio_fail(self) -> None:
        """Mutant: FAIL not returned for mismatch + low ratio."""
        result = ResponseLanguageVerifier._determine_status(
            detected_lang="en",
            expected_lang="de",
            confidence=0.9,
            target_ratio=0.3,
        )
        assert result == VerificationStatus.FAIL


class TestVerifierVerifyMethod:
    """Kill mutants in the verify() orchestration logic."""

    def test_short_text_skipped(self) -> None:
        """Mutant: min_words check removed."""
        verifier = ResponseLanguageVerifier(
            backend=MockBackend({"en": 1.0}),
            min_words=20,
        )
        # Text shorter than 20 words
        result = verifier.verify("Short text here", "en")
        assert result.skipped is True
        assert result.status == VerificationStatus.PASS

    def test_short_text_not_skipped_when_threshold_met(self) -> None:
        """Mutant: min_words comparison direction inverted."""
        verifier = ResponseLanguageVerifier(
            backend=MockBackend({"en": 0.95}),
            min_words=3,
        )
        result = verifier.verify("This is a test sentence with enough words", "en")
        assert result.skipped is False

    def test_verify_uses_expected_lang(self) -> None:
        """Mutant: expected_lang not propagated to result."""
        verifier = ResponseLanguageVerifier(
            backend=MockBackend({"de": 0.9}),
            min_words=1,
        )
        result = verifier.verify("Hallo Welt das ist ein Test", "de")
        assert result.expected_lang == "de"

    def test_verify_fail_reason_contains_expected_lang(self) -> None:
        """Mutant: reason string not set on FAIL."""
        verifier = ResponseLanguageVerifier(
            backend=MockBackend({"en": 0.9, "de": 0.1}),
            min_words=1,
        )
        # Use non-whitelist words (e.g. "hello") so cleaning does not strip them
        text = " ".join(["hello"] * 25)
        result = verifier.verify(text, "de")
        # Hard assert: this setup MUST produce FAIL (no vacuous if-gate)
        assert result.skipped is False, "Text must not be skipped by cleaning"
        assert result.status == VerificationStatus.FAIL
        assert result.reason is not None
        assert "de" in result.reason

    def test_top_from_distribution_empty(self) -> None:
        """Mutant: empty dict handling removed."""
        lang, conf = ResponseLanguageVerifier._top_from_distribution({})
        assert lang == ""
        assert conf == 0.0

    def test_top_from_distribution_single(self) -> None:
        """Mutant: max() key function wrong."""
        lang, conf = ResponseLanguageVerifier._top_from_distribution({"en": 0.95})
        assert lang == "en"
        assert conf == 0.95

    def test_top_from_distribution_multiple(self) -> None:
        """Mutant: returns wrong language from distribution."""
        lang, conf = ResponseLanguageVerifier._top_from_distribution(
            {"en": 0.7, "de": 0.3}
        )
        assert lang == "en"
        assert conf == 0.7

    def test_verification_result_passed_property(self) -> None:
        """Mutant: passed property logic wrong."""
        pass_result = VerificationResult(
            expected_lang="de",
            detected_lang="de",
            confidence=0.9,
            foreign_share=0.0,
            target_language_ratio=1.0,
            status=VerificationStatus.PASS,
            reason=None,
        )
        assert pass_result.passed is True

        warn_result = VerificationResult(
            expected_lang="de",
            detected_lang="de",
            confidence=0.9,
            foreign_share=0.2,
            target_language_ratio=0.7,
            status=VerificationStatus.WARN,
            reason=None,
        )
        assert warn_result.passed is True

        fail_result = VerificationResult(
            expected_lang="de",
            detected_lang="en",
            confidence=0.9,
            foreign_share=0.8,
            target_language_ratio=0.2,
            status=VerificationStatus.FAIL,
            reason="wrong language",
        )
        assert fail_result.passed is False


class TestVerifierCleanForDetection:
    """Kill mutants in text cleaning logic."""

    def test_code_blocks_removed(self) -> None:
        """Mutant: code block pattern not applied."""
        verifier = ResponseLanguageVerifier(backend=MockBackend())
        text = "Hello ```def foo(): pass``` world"
        cleaned = verifier._clean_for_detection(text)
        assert "def foo" not in cleaned
        assert "world" in cleaned

    def test_inline_code_removed(self) -> None:
        """Mutant: inline code pattern not applied."""
        verifier = ResponseLanguageVerifier(backend=MockBackend())
        text = "Use the `import os` module"
        cleaned = verifier._clean_for_detection(text)
        assert "`import os`" not in cleaned

    def test_urls_removed(self) -> None:
        """Mutant: URL pattern not applied."""
        verifier = ResponseLanguageVerifier(backend=MockBackend())
        text = "Visit https://example.com/path for details"
        cleaned = verifier._clean_for_detection(text)
        assert "https://example.com" not in cleaned

    def test_markdown_links_keep_text(self) -> None:
        """Mutant: markdown link replacement removes link text too."""
        verifier = ResponseLanguageVerifier(backend=MockBackend())
        text = "See [my link](https://example.com) for info"
        cleaned = verifier._clean_for_detection(text)
        assert "my link" in cleaned
        assert "https://example.com" not in cleaned

    def test_technical_whitelist_terms_removed(self) -> None:
        """Mutant: whitelist filtering not applied."""
        verifier = ResponseLanguageVerifier(backend=MockBackend())
        text = "Use Python and Docker for deployment"
        cleaned = verifier._clean_for_detection(text)
        assert "python" not in cleaned.lower()
        assert "docker" not in cleaned.lower()


class TestVerifierSlidingWindow:
    """Kill mutants in sliding window detection."""

    def test_sliding_window_empty_returns_defaults(self) -> None:
        """Mutant: empty window_results handling removed."""
        verifier = ResponseLanguageVerifier(
            backend=MockBackend({}),
            min_words=200,
        )
        lang, conf, foreign = verifier._sliding_window_detect([], "de")
        assert lang == "de"
        assert conf == 0.0
        assert foreign == 0.0

    def test_sliding_window_counts_foreign(self) -> None:
        """Mutant: foreign_windows increment logic wrong."""

        class AlternatingBackend:
            def __init__(self) -> None:
                self._calls = 0

            def detect_distribution(self, text: str) -> dict[str, float]:
                self._calls += 1
                if self._calls % 2 == 0:
                    return {"en": 0.95}  # foreign when expected is "de"
                return {"de": 0.95}

        verifier = ResponseLanguageVerifier(
            backend=AlternatingBackend(),
            min_words=5,
        )
        words = ["wort"] * 250
        lang, conf, foreign_share = verifier._sliding_window_detect(words, "de")
        # Some windows should be foreign
        assert foreign_share > 0.0


# ===============================================================
# STREAM GUARD: mutation-killing tests
# ===============================================================


class TestStreamGuardCheckEarly:
    """Kill mutants in check_early() decision logic."""

    def test_not_active_returns_true(self) -> None:
        """Mutant: disabled guard returns False."""
        guard = StreamGuard(expected_lang="de", enabled=False)
        assert guard.check_early("x" * 500) is True

    def test_short_text_returns_true(self) -> None:
        """Mutant: _MIN_CHARS_FOR_CHECK comparison removed."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"en": 0.95}),
        )
        # Text shorter than _MIN_CHARS_FOR_CHECK
        result = guard.check_early("x" * (_MIN_CHARS_FOR_CHECK - 1))
        assert result is True
        assert guard.state.check_performed is False

    def test_check_performed_only_once(self) -> None:
        """Mutant: check_performed flag not set."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"de": 0.95}),
        )
        text = "x" * _MIN_CHARS_FOR_CHECK
        guard.check_early(text)
        assert guard.state.check_performed is True
        # Second call should not re-check
        guard.check_early(text + "more")
        assert guard.state.check_performed is True

    def test_abort_on_wrong_language_high_confidence(self) -> None:
        """Mutant: abort condition removed or thresholds changed."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"en": 0.90}),  # > _EARLY_ABORT_CONFIDENCE
        )
        result = guard.check_early("x" * _MIN_CHARS_FOR_CHECK)
        assert result is False
        assert guard.state.aborted is True
        assert guard.state.detected_lang_at_abort == "en"

    def test_no_abort_when_confidence_below_threshold(self) -> None:
        """Mutant: _EARLY_ABORT_CONFIDENCE threshold lowered."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"en": 0.84}),  # < _EARLY_ABORT_CONFIDENCE (0.85)
        )
        result = guard.check_early("x" * _MIN_CHARS_FOR_CHECK)
        assert result is True
        assert guard.state.aborted is False

    def test_no_abort_when_detected_matches_expected(self) -> None:
        """Mutant: detected != expected check removed."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"de": 0.95}),
        )
        result = guard.check_early("x" * _MIN_CHARS_FOR_CHECK)
        assert result is True
        assert guard.state.aborted is False

    def test_abort_sets_partial_text_length(self) -> None:
        """Mutant: partial_text_length not set on abort."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"en": 0.90}),
        )
        text = "x" * 250
        guard.check_early(text)
        assert guard.state.partial_text_length == 250

    def test_abort_sets_abort_confidence(self) -> None:
        """Mutant: abort_confidence not set."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"en": 0.92}),
        )
        guard.check_early("x" * _MIN_CHARS_FOR_CHECK)
        assert guard.state.abort_confidence == 0.92

    def test_empty_distribution_no_abort(self) -> None:
        """Mutant: empty distribution not handled."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({}),
        )
        result = guard.check_early("x" * _MIN_CHARS_FOR_CHECK)
        assert result is True
        assert guard.state.aborted is False

    def test_after_abort_subsequent_calls_return_false(self) -> None:
        """Mutant: aborted state not checked on re-entry."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"en": 0.90}),
        )
        guard.check_early("x" * _MIN_CHARS_FOR_CHECK)
        assert guard.state.aborted is True
        # Subsequent call should still return False (not re-check)
        result = guard.check_early("x" * 500)
        assert result is False


class TestStreamGuardClassifyAbort:
    """Kill mutants in classify_and_report_abort()."""

    def test_no_check_outcome_when_not_aborted(self) -> None:
        """Mutant: outcome not set to NO_CHECK."""
        guard = StreamGuard(expected_lang="de", enabled=True)
        outcome = guard.classify_and_report_abort("text")
        assert outcome == StreamGuardOutcome.NO_CHECK

    def test_false_positive_when_target_present(self) -> None:
        """Mutant: _FP_TARGET_THRESHOLD (0.30) check wrong."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"de": 0.35, "en": 0.65}),
        )
        # Force aborted state
        guard._state.aborted = True
        guard._state.check_performed = True
        guard._state.detected_lang_at_abort = "en"
        outcome = guard.classify_and_report_abort("text")
        assert outcome == StreamGuardOutcome.FALSE_POSITIVE_ABORT

    def test_confirmed_abort_when_target_absent(self) -> None:
        """Mutant: CONFIRMED_ABORT logic wrong."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"de": 0.05, "en": 0.90}),
        )
        guard._state.aborted = True
        guard._state.check_performed = True
        guard._state.detected_lang_at_abort = "en"
        outcome = guard.classify_and_report_abort("text")
        assert outcome == StreamGuardOutcome.CONFIRMED_ABORT

    def test_unknown_abort_for_ambiguous_signal(self) -> None:
        """Mutant: UNKNOWN_ABORT branch removed."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"de": 0.20, "en": 0.50}),
        )
        guard._state.aborted = True
        guard._state.check_performed = True
        guard._state.detected_lang_at_abort = "en"
        outcome = guard.classify_and_report_abort("text")
        assert outcome == StreamGuardOutcome.UNKNOWN_ABORT

    def test_partial_verification_confidence_stored(self) -> None:
        """Mutant: partial_verification_confidence not set."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"de": 0.40}),
        )
        guard._state.aborted = True
        guard._state.check_performed = True
        guard._state.detected_lang_at_abort = "en"
        guard.classify_and_report_abort("text")
        assert guard.state.partial_verification_confidence == 0.40

    def test_outcome_stored_in_state(self) -> None:
        """Mutant: self._state.outcome not set."""
        guard = StreamGuard(
            expected_lang="de",
            backend=MockBackend({"de": 0.05, "en": 0.90}),
        )
        guard._state.aborted = True
        guard._state.check_performed = True
        guard._state.detected_lang_at_abort = "en"
        guard.classify_and_report_abort("text")
        assert guard.state.outcome is not None


class TestStreamGuardStats:
    """Kill mutants in StreamGuardStats properties."""

    def test_fp_rate_zero_when_no_aborts(self) -> None:
        """Mutant: division by zero not handled."""
        stats = StreamGuardStats()
        assert stats.fp_rate == 0.0

    def test_fp_rate_calculation(self) -> None:
        """Mutant: fp_rate formula wrong."""
        stats = StreamGuardStats(total_aborts=10, false_positives=3)
        assert stats.fp_rate == 0.3

    def test_should_disable_consecutive_fp(self) -> None:
        """Mutant: _MAX_CONSECUTIVE_FP comparison wrong."""
        stats = StreamGuardStats(
            consecutive_fp=_MAX_CONSECUTIVE_FP,
            total_aborts=5,
        )
        assert stats.should_disable is True

    def test_should_not_disable_below_threshold(self) -> None:
        """Mutant: should_disable always returns True."""
        stats = StreamGuardStats(consecutive_fp=1, total_aborts=2)
        assert stats.should_disable is False

    def test_should_disable_on_high_fp_rate(self) -> None:
        """Mutant: fp_rate threshold check wrong."""
        stats = StreamGuardStats(
            total_aborts=10,
            false_positives=2,  # fp_rate = 0.2 > 0.05
            consecutive_fp=1,
        )
        assert stats.should_disable is True

    def test_should_not_disable_low_sample_size(self) -> None:
        """Mutant: total_aborts >= 5 check removed."""
        stats = StreamGuardStats(
            total_aborts=4,  # < 5
            false_positives=4,  # fp_rate = 1.0 > threshold, but sample too small
            consecutive_fp=1,
        )
        # consecutive_fp < 3, and total_aborts < 5 -> should NOT disable
        assert stats.should_disable is False


class TestStreamGuardStatsStore:
    """Kill mutants in StreamGuardStatsStore."""

    def test_get_creates_new_stats(self) -> None:
        """Mutant: new stats not created."""
        store = StreamGuardStatsStore()
        stats = store.get(1, 1)
        assert isinstance(stats, StreamGuardStats)
        assert stats.total_checks == 0

    def test_get_returns_same_stats(self) -> None:
        """Mutant: new stats created on every call."""
        store = StreamGuardStatsStore()
        stats1 = store.get(1, 1)
        stats1.total_checks = 5
        stats2 = store.get(1, 1)
        assert stats2.total_checks == 5
        assert stats1 is stats2

    def test_lru_eviction(self) -> None:
        """Mutant: LRU eviction not performed."""
        store = StreamGuardStatsStore(max_entries=3)
        store.get(1, 1)
        store.get(2, 2)
        store.get(3, 3)
        # This should evict (1, 1)
        store.get(4, 4)
        all_stats = store.all_stats()
        assert (1, 1) not in all_stats
        assert (4, 4) in all_stats

    def test_lru_touch_on_access(self) -> None:
        """Mutant: move_to_end not called."""
        store = StreamGuardStatsStore(max_entries=3)
        store.get(1, 1)
        store.get(2, 2)
        store.get(3, 3)
        # Touch (1, 1) to make it recent
        store.get(1, 1)
        # Now adding (4, 4) should evict (2, 2) not (1, 1)
        store.get(4, 4)
        all_stats = store.all_stats()
        assert (1, 1) in all_stats
        assert (2, 2) not in all_stats

    def test_clear_removes_entry(self) -> None:
        """Mutant: clear does not actually delete."""
        store = StreamGuardStatsStore()
        store.get(1, 1)
        assert store.clear(1, 1) is True
        assert (1, 1) not in store.all_stats()

    def test_clear_returns_false_for_missing(self) -> None:
        """Mutant: clear returns True for missing keys."""
        store = StreamGuardStatsStore()
        assert store.clear(99, 99) is False


class TestStreamGuardReportFinalOutcome:
    """Kill mutants in report_final_outcome()."""

    def test_no_stats_is_noop(self) -> None:
        """Mutant: None check on stats removed -> AttributeError."""
        guard = StreamGuard(expected_lang="de", backend=MockBackend({"de": 0.95}))
        guard._state.check_performed = True
        guard._state.aborted = True
        # Calling with stats=None must return without mutating state
        result = guard.report_final_outcome(verification_passed=False, stats=None)
        assert result is None
        # Contrast test: with a real stats object, the function DOES mutate
        real_stats = StreamGuardStats()
        guard.report_final_outcome(verification_passed=False, stats=real_stats)
        assert real_stats.confirmed_aborts == 1, (
            "Guard must increment confirmed_aborts with real stats "
            "(proves None path is truly a no-op, not a broken code path)"
        )

    def test_correct_abort_increments_confirmed(self) -> None:
        """Mutant: confirmed_aborts not incremented."""
        guard = StreamGuard(expected_lang="de", backend=MockBackend({"de": 0.95}))
        guard._state.check_performed = True
        guard._state.aborted = True
        stats = StreamGuardStats()
        guard.report_final_outcome(verification_passed=False, stats=stats)
        assert stats.confirmed_aborts == 1
        assert stats.consecutive_fp == 0

    def test_false_positive_increments_fp(self) -> None:
        """Mutant: false_positives not incremented."""
        guard = StreamGuard(expected_lang="de", backend=MockBackend({"de": 0.95}))
        guard._state.check_performed = True
        guard._state.aborted = True
        stats = StreamGuardStats()
        guard.report_final_outcome(verification_passed=True, stats=stats)
        assert stats.false_positives == 1
        assert stats.consecutive_fp == 1

    def test_auto_disable_triggers(self) -> None:
        """Mutant: auto-disable logic not executed."""
        guard = StreamGuard(expected_lang="de", backend=MockBackend({"de": 0.95}))
        guard._state.check_performed = True
        guard._state.aborted = True
        stats = StreamGuardStats(
            consecutive_fp=_MAX_CONSECUTIVE_FP - 1,
            total_aborts=4,
        )
        guard.report_final_outcome(verification_passed=True, stats=stats)
        assert guard.state.disabled is True
        assert guard.state.disable_reason is not None
        assert stats.disabled_sessions == 1


class TestStreamGuardAuditEntry:
    """Kill mutants in build_audit_entry()."""

    def test_audit_entry_has_event_type(self) -> None:
        """Mutant: event_type key missing."""
        guard = StreamGuard(expected_lang="de")
        entry = guard.build_audit_entry()
        assert entry["event_type"] == "stream_guard_check"

    def test_audit_entry_has_expected_lang(self) -> None:
        """Mutant: expected_lang not in entry."""
        guard = StreamGuard(expected_lang="fr")
        entry = guard.build_audit_entry()
        assert entry["expected_lang"] == "fr"

    def test_audit_entry_includes_abort_data(self) -> None:
        """Mutant: abort data conditionals wrong."""
        guard = StreamGuard(expected_lang="de", backend=MockBackend({"en": 0.90}))
        guard.check_early("x" * _MIN_CHARS_FOR_CHECK)
        entry = guard.build_audit_entry()
        assert entry["aborted"] is True
        assert "detected_at_abort" in entry
        assert "abort_confidence" in entry
        assert "partial_length" in entry


# ===============================================================
# ENFORCEMENT: mutation-killing tests
# ===============================================================


class TestEnforcementProfileGating:
    """Kill mutants in LanguageEnforcement.enforce() profile checks."""

    @pytest.mark.asyncio
    async def test_no_verify_returns_original(self) -> None:
        """Mutant: verify_required check removed."""
        enforcement = LanguageEnforcement()
        ctx = LanguageContext(
            code="de",
            source="override",
            confidence=1.0,
            switched_from=None,
            request_id="test-123",
        )
        result = await enforcement.enforce(
            output="Test output",
            ctx=ctx,
            # claude-opus-4-7 has verify_required=False
            model_id="claude-opus-4-7",
        )
        assert result.final_output == "Test output"
        assert result.was_enforced is False
        assert result.verification is None
        assert result.repair is None

    @pytest.mark.asyncio
    async def test_enforcement_result_has_profile(self) -> None:
        """Mutant: model_profile not set in result."""
        enforcement = LanguageEnforcement()
        ctx = LanguageContext(
            code="de",
            source="override",
            confidence=1.0,
            switched_from=None,
            request_id="test-123",
        )
        result = await enforcement.enforce(
            output="Test output",
            ctx=ctx,
            model_id="claude-opus-4-7",
        )
        assert result.model_profile is not None
        assert isinstance(result.model_profile, ModelAdherenceProfile)


class TestModelProfiles:
    """Kill mutants in model_profiles.py get_profile()."""

    def test_known_model_returns_profile(self) -> None:
        """Mutant: profile lookup always returns default."""
        profile = get_profile("claude-opus-4-7")
        assert profile.model_id == "claude-opus-4-7"
        assert profile.verify_required is False

    def test_unknown_model_returns_default(self) -> None:
        """Mutant: unknown model raises exception."""
        profile = get_profile("unknown-model-xyz")
        assert profile.model_id == "default"

    def test_none_model_returns_default(self) -> None:
        """Mutant: None not handled."""
        profile = get_profile(None)
        assert profile.model_id == "default"

    def test_strict_model_has_verify_required(self) -> None:
        """Mutant: strict profile has verify_required=False."""
        profile = get_profile("mistral-large-latest")
        assert profile.verify_required is True

    def test_default_profile_fields(self) -> None:
        """Mutant: default profile fields wrong."""
        profile = get_profile("totally-unknown-model")
        assert profile.verify_required is True
        assert profile.repair_enabled is True
