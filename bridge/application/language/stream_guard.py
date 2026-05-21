"""StreamGuard: early streaming abort on wrong-language output.

Monitors the token stream during generation and can signal an early
abort if the model is clearly generating in the wrong language.

CRITICAL DESIGN NOTES (from Codex review):
- False positives destroy user experience (stream aborts mid-response)
- Very conservative thresholds: only abort when VERY confident
- Self-calibration: tracks false-positive rate and auto-disables
- Only active for models with stream_guard_enabled=True in their profile
- Not active in the first 100 characters (too little signal)
- Single check point between 200-400 characters
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

from application.language.backends import (
    LanguageDetectorBackend,
    LangdetectBackend,
)

log = logging.getLogger(__name__)


class StreamGuardOutcome(Enum):
    """Classification of a StreamGuard abort decision.

    Instead of a binary ``verification_passed: bool``, this enum
    captures the nuance of what is actually known at abort time:

    NO_CHECK: Guard did not perform a check (text too short, disabled).
    PASSED_NO_ABORT: Check ran, no abort was triggered.
    CONFIRMED_ABORT: Partial verification confirms wrong language.
    FALSE_POSITIVE_ABORT: Partial verification shows target language present.
    UNKNOWN_ABORT: Partial verification inconclusive (low confidence).
    """

    NO_CHECK = "no_check"
    PASSED_NO_ABORT = "passed_no_abort"
    CONFIRMED_ABORT = "confirmed_abort"
    FALSE_POSITIVE_ABORT = "false_positive_abort"
    UNKNOWN_ABORT = "unknown_abort"


# Minimum characters before any detection attempt
_MIN_CHARS_FOR_CHECK = 200

# Maximum characters at which the single check occurs
_MAX_CHARS_FOR_CHECK = 400

# Confidence threshold for early abort (very high to minimize FP)
_EARLY_ABORT_CONFIDENCE = 0.85

# Maximum consecutive false positives before auto-disable
_MAX_CONSECUTIVE_FP = 3

# Session-level false positive rate threshold for disable
_FP_RATE_DISABLE_THRESHOLD = 0.05


@dataclass(slots=True)
class StreamGuardState:
    """Per-session state for StreamGuard.

    Tracks check history and false-positive calibration.

    Attributes:
        check_performed: Whether the early check has been done.
        aborted: Whether an abort was signaled.
        detected_lang_at_abort: Language detected when abort was signaled.
        abort_confidence: Confidence of the detection that triggered abort.
        partial_text_length: Length of accumulated text at abort time.
        outcome: Classification of the abort decision after partial
            verification. Set by classify_and_report_abort().
        partial_verification_confidence: Confidence of the target language
            in the partial verification distribution. Used for audit.
        disabled: Whether guard is disabled due to false positives.
        disable_reason: Why the guard was disabled.
    """

    check_performed: bool = False
    aborted: bool = False
    detected_lang_at_abort: str | None = None
    abort_confidence: float = 0.0
    partial_text_length: int = 0
    outcome: StreamGuardOutcome | None = None
    partial_verification_confidence: float = 0.0
    disabled: bool = False
    disable_reason: str | None = None


@dataclass(slots=True)
class StreamGuardStats:
    """Cumulative statistics for StreamGuard self-calibration.

    Tracks abort decisions and their outcomes over time.

    Attributes:
        total_checks: Total checks performed.
        total_aborts: Total aborts signaled.
        confirmed_aborts: Aborts confirmed correct by partial verification.
        false_positives: Aborts where partial verification found target lang.
        unknown_aborts: Aborts where partial verification was inconclusive.
        consecutive_fp: Current streak of consecutive false positives.
        disabled_sessions: Number of sessions where guard was auto-disabled.
    """

    total_checks: int = 0
    total_aborts: int = 0
    confirmed_aborts: int = 0
    false_positives: int = 0
    unknown_aborts: int = 0
    consecutive_fp: int = 0
    disabled_sessions: int = 0

    @property
    def fp_rate(self) -> float:
        """False positive rate (0.0..1.0)."""
        if self.total_aborts == 0:
            return 0.0
        return self.false_positives / self.total_aborts

    @property
    def should_disable(self) -> bool:
        """Whether the guard should auto-disable based on FP rate."""
        return self.consecutive_fp >= _MAX_CONSECUTIVE_FP or (
            self.total_aborts >= 5 and self.fp_rate > _FP_RATE_DISABLE_THRESHOLD
        )


class StreamGuard:
    """Monitors streaming output for language violations.

    Usage:
        guard = StreamGuard(expected_lang="de")
        # During streaming:
        should_continue = guard.check_early(partial_text)
        if not should_continue:
            # Abort stream, trigger RepairService
            ...
        # After stream completes:
        guard.report_final_outcome(verification_passed=True)
    """

    def __init__(
        self,
        expected_lang: str,
        enabled: bool = True,
        backend: LanguageDetectorBackend | None = None,
    ) -> None:
        """Initialize StreamGuard for a streaming session.

        Args:
            expected_lang: Expected language code for this response.
            enabled: Whether the guard is active (from model profile).
            backend: Detection backend (default: LangdetectBackend).
        """
        self._expected_lang = expected_lang
        self._enabled = enabled
        self._backend = backend or LangdetectBackend()
        self._state = StreamGuardState()

    @property
    def state(self) -> StreamGuardState:
        """Current guard state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the guard is currently active and may abort."""
        return self._enabled and not self._state.disabled

    def check_early(self, partial_stream: str) -> bool:
        """Check partial stream for language violations.

        Returns True if the stream should continue, False if it
        should be aborted (language violation detected with high confidence).

        This method is designed to be called repeatedly during streaming.
        It only performs the actual check once (between 200-400 chars).

        Args:
            partial_stream: Accumulated stream text so far.

        Returns:
            True = continue streaming, False = abort recommended.
        """
        # Guard: not active or already checked
        if not self.is_active:
            return True

        if self._state.check_performed:
            return not self._state.aborted

        # Not enough text yet
        text_len = len(partial_stream)
        if text_len < _MIN_CHARS_FOR_CHECK:
            return True

        # Check window: between 200-400 chars, perform the single check
        if text_len >= _MIN_CHARS_FOR_CHECK:
            self._state.check_performed = True

            # Detect language on the partial text via backend
            distribution = self._backend.detect_distribution(partial_stream)
            if distribution:
                detected = max(distribution, key=distribution.get)  # type: ignore[arg-type]
                confidence = distribution[detected]
            else:
                detected = ""
                confidence = 0.0

            log.debug(
                "StreamGuard check at %d chars: detected=%s, "
                "expected=%s, confidence=%.2f",
                text_len,
                detected,
                self._expected_lang,
                confidence,
            )

            # Only abort if VERY confident it's wrong
            if (
                detected != self._expected_lang
                and confidence >= _EARLY_ABORT_CONFIDENCE
            ):
                self._state.aborted = True
                self._state.detected_lang_at_abort = detected
                self._state.abort_confidence = confidence
                self._state.partial_text_length = text_len
                log.warning(
                    "StreamGuard ABORT: detected '%s' at %d chars "
                    "(expected '%s', confidence=%.2f)",
                    detected,
                    text_len,
                    self._expected_lang,
                    confidence,
                )
                return False

        return True

    def classify_and_report_abort(
        self,
        accumulated_text: str,
        stats: StreamGuardStats | None = None,
    ) -> StreamGuardOutcome:
        """Classify an abort via partial verification and update stats.

        Called in the abort path (chat_service) instead of
        ``report_final_outcome(verification_passed=False)``.
        Runs the detection backend on the accumulated partial text
        to determine whether the abort was justified.

        No new provider call is made. This is a local, synchronous
        detection on text that is already in memory.

        Classification logic:
        - Target language present with >= 0.30 probability in
          distribution: FALSE_POSITIVE_ABORT (abort was wrong).
        - Target language absent or < 0.10, AND detected language
          matches abort language with high confidence: CONFIRMED_ABORT.
        - Everything else (ambiguous signal): UNKNOWN_ABORT.

        The 0.30 threshold for FP detection is deliberately generous
        because at 200-400 chars, detection distributions are noisy.
        We would rather classify uncertain cases as UNKNOWN than
        incorrectly call something a confirmed abort.

        Determinism note: langdetect with seed=0 returns identical
        results for identical text. For texts where target_prob is
        near the 0.30 threshold or the abort_confidence is near 0.85,
        a one-token text difference could flip the classification
        between FP/CONFIRMED/UNKNOWN. Conservative threshold
        calibration (0.30 FP threshold is generous, 0.10 CONFIRMED
        ceiling is strict) keeps this edge case robust in practice.
        If production data shows unstable classifications, consider
        adding hysteresis bands (e.g. 0.25-0.35 = UNKNOWN, not FP).

        Args:
            accumulated_text: The partial stream text at abort time.
            stats: Cumulative stats for self-calibration.

        Returns:
            The classified StreamGuardOutcome.
        """
        if not self._state.aborted:
            outcome = StreamGuardOutcome.NO_CHECK
            self._state.outcome = outcome
            return outcome

        # Run the backend on the partial text for verification.
        # This is the SAME backend instance that check_early() used,
        # so determinism (seed=0) is preserved.
        distribution = self._backend.detect_distribution(accumulated_text)

        target_prob = distribution.get(self._expected_lang, 0.0)
        self._state.partial_verification_confidence = target_prob

        # Classification thresholds for partial text (200-400 chars).
        # These are intentionally conservative:
        # - 0.30 for FP: even moderate target-lang presence means
        #   the abort might have been wrong.
        # - 0.10 ceiling for confirmed: target lang must be nearly
        #   absent to confirm the abort was correct.
        _FP_TARGET_THRESHOLD = 0.30
        _CONFIRMED_TARGET_CEILING = 0.10

        if target_prob >= _FP_TARGET_THRESHOLD:
            outcome = StreamGuardOutcome.FALSE_POSITIVE_ABORT
            log.warning(
                "StreamGuard partial verification: FALSE POSITIVE "
                "(target '%s' at %.2f in partial text, abort was on '%s')",
                self._expected_lang,
                target_prob,
                self._state.detected_lang_at_abort,
            )
        elif (
            target_prob < _CONFIRMED_TARGET_CEILING
            and self._state.detected_lang_at_abort is not None
            and distribution.get(self._state.detected_lang_at_abort, 0.0)
            >= _EARLY_ABORT_CONFIDENCE
        ):
            outcome = StreamGuardOutcome.CONFIRMED_ABORT
            log.info(
                "StreamGuard partial verification: CONFIRMED ABORT "
                "(target '%s' at %.2f, detected '%s' at %.2f)",
                self._expected_lang,
                target_prob,
                self._state.detected_lang_at_abort,
                distribution.get(self._state.detected_lang_at_abort, 0.0),
            )
        else:
            outcome = StreamGuardOutcome.UNKNOWN_ABORT
            log.info(
                "StreamGuard partial verification: UNKNOWN "
                "(target '%s' at %.2f, distribution ambiguous)",
                self._expected_lang,
                target_prob,
            )

        self._state.outcome = outcome
        self._record_check(stats)
        self._apply_outcome_to_stats(outcome, stats)
        return outcome

    def report_final_outcome(
        self,
        verification_passed: bool,
        stats: StreamGuardStats | None = None,
    ) -> None:
        """Report the final verification outcome for calibration.

        Called after the full response is available and verified
        (non-abort path in save_streaming_result). For the abort
        path, use classify_and_report_abort() instead.

        Args:
            verification_passed: Whether final language verification passed.
            stats: Cumulative stats object for self-calibration.
        """
        if stats is None:
            return

        if not self._state.check_performed:
            return

        self._record_check(stats)

        if self._state.aborted:
            stats.total_aborts += 1

            if not verification_passed:
                # Abort was correct (response WAS in wrong language)
                stats.confirmed_aborts += 1
                stats.consecutive_fp = 0
            else:
                # False positive: we aborted but final response was OK
                stats.false_positives += 1
                stats.consecutive_fp += 1
                log.warning(
                    "StreamGuard false positive detected "
                    "(consecutive=%d, total_fp=%d, fp_rate=%.1f%%)",
                    stats.consecutive_fp,
                    stats.false_positives,
                    stats.fp_rate * 100,
                )

        # Check if guard should auto-disable
        if stats.should_disable and not self._state.disabled:
            self._state.disabled = True
            self._state.disable_reason = (
                f"Auto-disabled: {stats.consecutive_fp} consecutive FP "
                f"or FP rate {stats.fp_rate:.1%} > threshold"
            )
            stats.disabled_sessions += 1
            log.warning(
                "StreamGuard AUTO-DISABLED: %s",
                self._state.disable_reason,
            )

    def _record_check(self, stats: StreamGuardStats | None) -> None:
        """Idempotent total_checks increment (once per guard lifecycle).

        Both ``classify_and_report_abort`` and ``report_final_outcome``
        call this, but a single StreamGuard instance only increments
        ``total_checks`` once. If both methods are called on the same
        instance (theoretically), the second call is a no-op.

        Args:
            stats: Cumulative stats (None = no-op).
        """
        if stats is None:
            return
        if getattr(self, "_check_recorded", False):
            return
        self._check_recorded = True
        stats.total_checks += 1

    def _apply_outcome_to_stats(
        self,
        outcome: StreamGuardOutcome,
        stats: StreamGuardStats | None,
    ) -> None:
        """Apply a classified outcome to cumulative stats.

        Called by classify_and_report_abort(). Separated so the
        auto-disable check can run after stats are updated.

        Note: total_checks is handled by _record_check() and is
        NOT incremented here. This prevents double-counting if
        both classify_and_report_abort and report_final_outcome
        are ever called on the same instance.

        Rules:
        - CONFIRMED_ABORT: confirmed_aborts++, consecutive_fp = 0
        - FALSE_POSITIVE_ABORT: false_positives++, consecutive_fp++
        - UNKNOWN_ABORT: unknown_aborts++ (consecutive_fp unchanged)
        - NO_CHECK / PASSED_NO_ABORT: no abort stats changed

        Args:
            outcome: The classified outcome.
            stats: Cumulative stats (None = no-op).
        """
        if stats is None:
            return

        if not self._state.check_performed:
            return

        stats.total_aborts += 1

        if outcome == StreamGuardOutcome.CONFIRMED_ABORT:
            stats.confirmed_aborts += 1
            stats.consecutive_fp = 0
        elif outcome == StreamGuardOutcome.FALSE_POSITIVE_ABORT:
            stats.false_positives += 1
            stats.consecutive_fp += 1
            log.warning(
                "StreamGuard false positive (partial verification): "
                "consecutive=%d, total_fp=%d, fp_rate=%.1f%%",
                stats.consecutive_fp,
                stats.false_positives,
                stats.fp_rate * 100,
            )
        elif outcome == StreamGuardOutcome.UNKNOWN_ABORT:
            stats.unknown_aborts += 1
            # UNKNOWN does NOT touch consecutive_fp.
            # This prevents auto-disable from triggering on
            # genuinely ambiguous cases.

        # Check if guard should auto-disable
        if stats.should_disable and not self._state.disabled:
            self._state.disabled = True
            self._state.disable_reason = (
                f"Auto-disabled: {stats.consecutive_fp} consecutive FP "
                f"or FP rate {stats.fp_rate:.1%} > threshold"
            )
            stats.disabled_sessions += 1
            log.warning(
                "StreamGuard AUTO-DISABLED: %s",
                self._state.disable_reason,
            )

    def build_audit_entry(self) -> dict[str, str | int | float | bool | None]:
        """Build an audit log entry for this stream guard session.

        Privacy: NEVER includes accumulated text, only metadata
        (language codes, confidence scores, lengths, outcome).

        Returns:
            Dict suitable for write_audit_log().
        """
        entry: dict[str, str | int | float | bool | None] = {
            "event_type": "stream_guard_check",
            "expected_lang": self._expected_lang,
            "check_performed": self._state.check_performed,
            "aborted": self._state.aborted,
            "disabled": self._state.disabled,
        }

        if self._state.detected_lang_at_abort:
            entry["detected_at_abort"] = self._state.detected_lang_at_abort

        if self._state.abort_confidence > 0:
            entry["abort_confidence"] = round(self._state.abort_confidence, 3)

        if self._state.partial_text_length > 0:
            entry["partial_length"] = self._state.partial_text_length

        if self._state.outcome is not None:
            entry["outcome"] = self._state.outcome.value

        if self._state.partial_verification_confidence > 0:
            entry["partial_verification_confidence"] = round(
                self._state.partial_verification_confidence, 3
            )

        if self._state.disable_reason:
            entry["disable_reason"] = self._state.disable_reason

        return entry


class StreamGuardStatsStore:
    """Process-wide in-memory store for StreamGuard self-calibration stats.

    Keys are (user_id, chat_id) tuples. Each key maps to a
    StreamGuardStats instance that accumulates across streaming sessions.

    Uses LRU eviction with a configurable max size (default 10000) to
    prevent unbounded memory growth in multi-user deployments.

    Thread-safety note: asyncio is single-threaded, so no lock is needed.
    The store lives for the lifetime of the process; a restart resets stats.
    Persistence (e.g. to DB) can be added later without changing the API.
    """

    _MAX_ENTRIES: int = 10_000  # LRU cap

    def __init__(self, max_entries: int | None = None) -> None:
        self._stats: OrderedDict[tuple[int, int], StreamGuardStats] = OrderedDict()
        if max_entries is not None:
            self._MAX_ENTRIES = max_entries

    def get(self, user_id: int, chat_id: int) -> StreamGuardStats:
        """Get or create stats for a (user_id, chat_id) pair.

        Existing entries are moved to the end (LRU touch).
        When the store exceeds _MAX_ENTRIES, the oldest entry
        is evicted.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.

        Returns:
            The cumulative StreamGuardStats for this session pair.
        """
        key = (user_id, chat_id)
        if key in self._stats:
            self._stats.move_to_end(key)  # LRU touch
            return self._stats[key]
        # Evict oldest if at capacity
        if len(self._stats) >= self._MAX_ENTRIES:
            self._stats.popitem(last=False)
        new_stats = StreamGuardStats()
        self._stats[key] = new_stats
        return new_stats

    def clear(self, user_id: int, chat_id: int) -> bool:
        """Explicitly remove stats for a (user_id, chat_id) pair.

        Useful on /reset to free memory for a terminated session.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.

        Returns:
            True if an entry was removed, False if key was not found.
        """
        key = (user_id, chat_id)
        if key in self._stats:
            del self._stats[key]
            return True
        return False

    def all_stats(self) -> dict[tuple[int, int], StreamGuardStats]:
        """Return all stored stats (read-only snapshot).

        Returns:
            Copy of the internal stats dict.
        """
        return dict(self._stats)
