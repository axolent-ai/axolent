"""Property-based tests for StreamGuard FP-Detection state machine.

Hypothesis generates random language confidence distributions and
text lengths to find edge cases the example-based tests miss.

Targets:
  1. classify_and_report_abort determinism: same input -> same outcome
  2. consecutive_fp correctness: only FALSE_POSITIVE_ABORT increments
  3. UNKNOWN_ABORT safety: never triggers auto-disable
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from application.language.stream_guard import (
    StreamGuard,
    StreamGuardOutcome,
    StreamGuardStats,
    _MAX_CONSECUTIVE_FP,
)


# ---------------------------------------------------------------------------
# Stub backend returning a fixed distribution (deterministic by design)
# ---------------------------------------------------------------------------


class _HypothesisStubBackend:
    """Backend that returns a caller-specified distribution.

    Used to control exactly what classify_and_report_abort sees,
    so we can test the classification logic in isolation.
    """

    def __init__(self, distribution: dict[str, float]) -> None:
        self._distribution = distribution

    def detect_distribution(self, text: str) -> dict[str, float]:
        return dict(self._distribution)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Classification thresholds from stream_guard.py (private constants,
# replicated here for clarity rather than importing private names).
_FP_TARGET_THRESHOLD = 0.30
_CONFIRMED_TARGET_CEILING = 0.10
_EARLY_ABORT_CONFIDENCE = 0.85


def _make_aborted_guard(
    target_lang: str,
    abort_lang: str,
    backend_distribution: dict[str, float],
) -> StreamGuard:
    """Build a StreamGuard in the 'aborted' state for classify tests."""
    backend = _HypothesisStubBackend(backend_distribution)
    guard = StreamGuard(expected_lang=target_lang, enabled=True, backend=backend)
    guard._state.check_performed = True
    guard._state.aborted = True
    guard._state.detected_lang_at_abort = abort_lang
    guard._state.abort_confidence = 0.90
    return guard


# ---------------------------------------------------------------------------
# Property 1: Classification is deterministic
# ---------------------------------------------------------------------------


@given(
    target_prob=st.floats(min_value=0.0, max_value=1.0),
    abort_prob=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_classification_is_deterministic(
    target_prob: float,
    abort_prob: float,
) -> None:
    """Same distribution must always produce the same outcome.

    We call classify_and_report_abort twice with identical input
    (on fresh guards sharing the same stub backend) and assert
    they return the same StreamGuardOutcome. This catches any
    hidden randomness or mutable state leaking between calls.
    """
    distribution = {"de": target_prob, "en": abort_prob}

    guard_a = _make_aborted_guard("de", "en", distribution)
    guard_b = _make_aborted_guard("de", "en", distribution)

    outcome_a = guard_a.classify_and_report_abort(
        accumulated_text="test text " * 20,
    )
    outcome_b = guard_b.classify_and_report_abort(
        accumulated_text="test text " * 20,
    )

    assert outcome_a == outcome_b, (
        f"Non-deterministic classification: "
        f"target_prob={target_prob}, abort_prob={abort_prob}, "
        f"outcome_a={outcome_a}, outcome_b={outcome_b}"
    )


# ---------------------------------------------------------------------------
# Property 2: consecutive_fp only increments on FALSE_POSITIVE_ABORT
# ---------------------------------------------------------------------------


@given(
    distribution_pairs=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.0, max_value=1.0),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=200)
def test_consecutive_fp_only_counts_real_fps(
    distribution_pairs: list[tuple[float, float]],
) -> None:
    """consecutive_fp must only increment on FALSE_POSITIVE_ABORT,
    not on CONFIRMED_ABORT or UNKNOWN_ABORT.

    We run N classifications with random distributions and verify
    that consecutive_fp equals the length of the trailing streak
    of FALSE_POSITIVE_ABORT outcomes.
    """
    stats = StreamGuardStats()
    outcomes: list[StreamGuardOutcome] = []

    for target_prob, abort_prob in distribution_pairs:
        distribution = {"de": target_prob, "en": abort_prob}
        guard = _make_aborted_guard("de", "en", distribution)
        outcome = guard.classify_and_report_abort(
            accumulated_text="dummy " * 30,
            stats=stats,
        )
        outcomes.append(outcome)

    # Compute expected consecutive_fp: trailing streak of FP outcomes.
    # The code resets consecutive_fp to 0 on CONFIRMED_ABORT,
    # leaves it unchanged on UNKNOWN_ABORT, and increments on FP.
    expected_consecutive = 0
    for outcome in outcomes:
        if outcome == StreamGuardOutcome.FALSE_POSITIVE_ABORT:
            expected_consecutive += 1
        elif outcome == StreamGuardOutcome.CONFIRMED_ABORT:
            expected_consecutive = 0
        # UNKNOWN_ABORT: no change

    assert stats.consecutive_fp == expected_consecutive, (
        f"consecutive_fp mismatch: got {stats.consecutive_fp}, "
        f"expected {expected_consecutive}, "
        f"outcomes={[o.value for o in outcomes]}"
    )


# ---------------------------------------------------------------------------
# Property 3: UNKNOWN_ABORT never triggers auto-disable
# ---------------------------------------------------------------------------


@given(
    abort_count=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=100)
def test_unknown_aborts_never_trigger_auto_disable(
    abort_count: int,
) -> None:
    """UNKNOWN_ABORT outcomes should never cause auto-disable,
    no matter how many consecutive UNKNOWN outcomes accumulate.

    We set up a distribution that always produces UNKNOWN_ABORT
    (target_prob between 0.10 and 0.30) and run N classifications.
    The guard must remain enabled after all of them.
    """
    # Distribution producing UNKNOWN_ABORT:
    # target_prob=0.20 (>= 0.10 but < 0.30) and abort lang not dominant
    distribution = {"de": 0.20, "en": 0.55, "nl": 0.15}
    stats = StreamGuardStats()

    last_guard = None
    for _ in range(abort_count):
        guard = _make_aborted_guard("de", "en", distribution)
        outcome = guard.classify_and_report_abort(
            accumulated_text="ambiguous " * 25,
            stats=stats,
        )
        assert outcome == StreamGuardOutcome.UNKNOWN_ABORT, (
            f"Expected UNKNOWN_ABORT but got {outcome.value} "
            f"(distribution={distribution})"
        )
        last_guard = guard

    # Guard must NOT be disabled
    assert last_guard is not None
    assert not last_guard.state.disabled, (
        f"Guard was disabled after {abort_count} UNKNOWN_ABORT outcomes. "
        f"consecutive_fp={stats.consecutive_fp}, "
        f"should_disable={stats.should_disable}"
    )

    # Verify: consecutive_fp stayed at 0 throughout
    assert stats.consecutive_fp == 0
    assert stats.unknown_aborts == abort_count


# ---------------------------------------------------------------------------
# Property 4: Total FP count matches outcome count
# ---------------------------------------------------------------------------


@given(
    distribution_pairs=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.0, max_value=1.0),
        ),
        min_size=1,
        max_size=15,
    )
)
@settings(max_examples=200)
def test_total_fp_matches_fp_outcomes(
    distribution_pairs: list[tuple[float, float]],
) -> None:
    """stats.false_positives must exactly equal the count of
    FALSE_POSITIVE_ABORT outcomes in the sequence."""
    stats = StreamGuardStats()
    fp_count = 0

    for target_prob, abort_prob in distribution_pairs:
        distribution = {"de": target_prob, "en": abort_prob}
        guard = _make_aborted_guard("de", "en", distribution)
        outcome = guard.classify_and_report_abort(
            accumulated_text="text " * 30,
            stats=stats,
        )
        if outcome == StreamGuardOutcome.FALSE_POSITIVE_ABORT:
            fp_count += 1

    assert stats.false_positives == fp_count


# ---------------------------------------------------------------------------
# Property 5: Auto-disable only triggers on FP streak or FP rate
# ---------------------------------------------------------------------------


@given(
    fp_streak=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=50)
def test_auto_disable_requires_fp_streak(fp_streak: int) -> None:
    """Auto-disable only triggers when consecutive_fp >= MAX_CONSECUTIVE_FP.

    We manually set up a stats object with the given FP streak and
    verify that should_disable matches the threshold condition.
    """
    stats = StreamGuardStats(
        total_aborts=max(fp_streak, 1),
        false_positives=fp_streak,
        consecutive_fp=fp_streak,
    )

    if fp_streak >= _MAX_CONSECUTIVE_FP:
        assert stats.should_disable is True
    else:
        # Only disabled if FP rate exceeds threshold with enough samples
        if stats.total_aborts >= 5 and stats.fp_rate > 0.05:
            assert stats.should_disable is True
        else:
            assert stats.should_disable is False
