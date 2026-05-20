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
from dataclasses import dataclass

from application.language.backends import (
    LanguageDetectorBackend,
    LangdetectBackend,
)

log = logging.getLogger(__name__)

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
        disabled: Whether guard is disabled due to false positives.
        disable_reason: Why the guard was disabled.
    """

    check_performed: bool = False
    aborted: bool = False
    detected_lang_at_abort: str | None = None
    disabled: bool = False
    disable_reason: str | None = None


@dataclass(slots=True)
class StreamGuardStats:
    """Cumulative statistics for StreamGuard self-calibration.

    Tracks abort decisions and their outcomes over time.

    Attributes:
        total_checks: Total checks performed.
        total_aborts: Total aborts signaled.
        confirmed_aborts: Aborts confirmed correct by final verifier.
        false_positives: Aborts that turned out wrong (final verify OK).
        consecutive_fp: Current streak of consecutive false positives.
        disabled_sessions: Number of sessions where guard was auto-disabled.
    """

    total_checks: int = 0
    total_aborts: int = 0
    confirmed_aborts: int = 0
    false_positives: int = 0
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

    def report_final_outcome(
        self,
        verification_passed: bool,
        stats: StreamGuardStats | None = None,
    ) -> None:
        """Report the final verification outcome for calibration.

        Called after the full response is available and verified.
        Updates statistics if a stats object is provided.

        Args:
            verification_passed: Whether final language verification passed.
            stats: Cumulative stats object for self-calibration.
        """
        if stats is None:
            return

        if not self._state.check_performed:
            return

        stats.total_checks += 1

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

    def build_audit_entry(self) -> dict[str, str | int | float | bool | None]:
        """Build an audit log entry for this stream guard session.

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

        if self._state.disable_reason:
            entry["disable_reason"] = self._state.disable_reason

        return entry
