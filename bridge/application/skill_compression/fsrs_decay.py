"""FSRS v5-style Decay Engine for Skill-Compression.

Implements individual forgetting curves per hypothesis using the
Free Spaced Repetition Scheduler power-law model.

HC-SC-5 [BLOCKER]: FSRS for individual decay per skill.
  Seasonal patterns are recognized and protected.
HC-SC-6 [BLOCKER]: User-created skills (source_type="manual" or
  "learn_command") are decay_immune. FSRS is never applied to them.

Mathematical model (FSRS-4.5+, power-law forgetting curve):

  Retrievability: R(t, S) = (1 + FACTOR * t / S) ^ DECAY
    where:
      t = elapsed time in days since last review
      S = memory stability (days until R drops to 0.9 by definition)
      FACTOR = 19/81 (derived from: when t=S, R=0.9)
      DECAY = -0.5 (power-law exponent, standard FSRS)

  When t = S: R = (1 + 19/81 * 1)^(-0.5) = (100/81)^(-0.5) = 0.9

  Interval calculation (given desired retention):
    next_interval = (S / FACTOR) * (desired_retention^(1/DECAY) - 1)

Stability update after review:
  For successful recall (rating >= 2):
    S' = S * SInc(D, S, R, G)
  For failed recall (rating = 1, "again"):
    S' = w[11] * D^(-w[12]) * ((S+1)^w[13] - 1) * e^(w[14] * (1-R))

  where SInc (stability increase factor):
    SInc = e^(w[8]) * (11 - D) * S^(-w[9]) * (e^(w[10]*(1-R)) - 1) * hard/easy_bonus

Difficulty update:
  delta_D = -w[6] * (rating - 3)
  D' = w[7] * D_init(rating=4) + (1 - w[7]) * (D + delta_D * (10-D)/9)
  D is clamped to [1.0, 10.0]

Default parameters (21 weights, FSRS v5/v6 defaults from open-spaced-repetition):
  Trained on millions of review logs. Excellent out-of-the-box.

IC-SC-10: Seasonal detection uses interval regularity heuristic.
  If a skill is used at approximately regular intervals (e.g. monthly),
  the decay rate is reduced (stability boost) to prevent archiving
  periodic patterns.

No external dependencies. No sklearn, no torch. Pure Python math.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# FSRS constants
# ---------------------------------------------------------------

# Power-law forgetting curve parameters
DECAY: float = -0.5
FACTOR: float = 19.0 / 81.0  # ~0.2346

# Desired retention (probability threshold for "due")
DESIRED_RETENTION: float = 0.9

# Archive threshold: after 180 days unreviewed, hypothesis is archived
ARCHIVE_DAYS: float = 180.0

# ---------------------------------------------------------------
# Default FSRS weights (21 parameters, v5/v6 standard)
# Source: open-spaced-repetition/py-fsrs defaults
# ---------------------------------------------------------------

DEFAULT_WEIGHTS: tuple[float, ...] = (
    0.212,  # w[0]: initial stability for rating=1 (Again)
    1.2931,  # w[1]: initial stability for rating=2 (Hard)
    2.3065,  # w[2]: initial stability for rating=3 (Good)
    8.2956,  # w[3]: initial stability for rating=4 (Easy)
    6.4133,  # w[4]: initial difficulty for first review
    0.8334,  # w[5]: initial difficulty scaling
    3.0194,  # w[6]: difficulty delta factor
    0.001,  # w[7]: difficulty mean reversion rate
    1.8722,  # w[8]: stability increase base (ln)
    0.1666,  # w[9]: stability power factor
    0.796,  # w[10]: retrievability factor in SInc
    1.4835,  # w[11]: failure new stability base
    0.0614,  # w[12]: failure difficulty exponent
    0.2629,  # w[13]: failure stability exponent
    1.6483,  # w[14]: failure retrievability factor
    0.6014,  # w[15]: hard penalty factor
    1.8729,  # w[16]: easy bonus factor
    0.5425,  # w[17]: short-term stability decay (not used in our model)
    0.0912,  # w[18]: reserved / fuzz factor
    0.0658,  # w[19]: reserved (v6 extension)
    0.1542,  # w[20]: reserved (v6 personalized decay)
)

# Rating constants (1-indexed conceptually, map to array access)
RATING_AGAIN: int = 1
RATING_HARD: int = 2
RATING_GOOD: int = 3
RATING_EASY: int = 4

# Seasonal detection: if interval coefficient of variation < this,
# the pattern is considered regular/seasonal
SEASONAL_CV_THRESHOLD: float = 0.35

# Seasonal stability boost multiplier
SEASONAL_STABILITY_BOOST: float = 1.5

# Minimum reviews to detect seasonality
MIN_REVIEWS_FOR_SEASONAL: int = 3


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FSRSState:
    """FSRS state for a single hypothesis.

    Attributes:
        stability: Memory stability in days (time until R drops to 0.9).
        difficulty: Item difficulty [1.0, 10.0].
        last_reviewed: ISO-8601 timestamp of last review/application.
        reps: Number of successful repetitions.
        lapses: Number of failures (rating=1).
        review_history: JSON-serialized list of review timestamps
            for seasonal detection.
    """

    stability: float = 2.3065  # w[2] default (rating=Good initial)
    difficulty: float = 5.0
    last_reviewed: str = ""
    reps: int = 0
    lapses: int = 0
    review_history: str = "[]"  # JSON list of ISO timestamps

    def to_json(self) -> str:
        """Serialize to JSON string for DB storage."""
        return json.dumps(
            {
                "stability": self.stability,
                "difficulty": self.difficulty,
                "last_reviewed": self.last_reviewed,
                "reps": self.reps,
                "lapses": self.lapses,
                "review_history": self.review_history,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> FSRSState:
        """Deserialize from JSON string.

        Args:
            raw: JSON string or empty string.

        Returns:
            FSRSState instance.
        """
        if not raw or raw == "{}":
            return cls()
        try:
            data = json.loads(raw)
            return cls(
                stability=data.get("stability", 2.3065),
                difficulty=data.get("difficulty", 5.0),
                last_reviewed=data.get("last_reviewed", ""),
                reps=data.get("reps", 0),
                lapses=data.get("lapses", 0),
                review_history=data.get("review_history", "[]"),
            )
        except (json.JSONDecodeError, TypeError):
            return cls()


# ---------------------------------------------------------------
# Core FSRS functions
# ---------------------------------------------------------------


def estimate_recall(state: FSRSState, elapsed_days: float) -> float:
    """Estimate current retrievability (recall probability).

    Uses the FSRS power-law forgetting curve:
      R(t, S) = (1 + FACTOR * t / S) ^ DECAY

    Args:
        state: Current FSRS state.
        elapsed_days: Days since last review.

    Returns:
        Retrievability [0.0, 1.0]. 1.0 = just reviewed, decays over time.
    """
    if elapsed_days <= 0.0:
        return 1.0

    if state.stability <= 0.0:
        return 0.0

    # R = (1 + FACTOR * t / S)^DECAY
    base = 1.0 + FACTOR * elapsed_days / state.stability
    recall = math.pow(base, DECAY)

    return max(0.0, min(1.0, recall))


def _elapsed_days_since(last_reviewed: str, current_time: str) -> float:
    """Compute elapsed days between two ISO-8601 timestamps.

    Args:
        last_reviewed: Previous timestamp.
        current_time: Current timestamp.

    Returns:
        Elapsed days as float. Returns 0.0 if parsing fails.
    """
    if not last_reviewed or not current_time:
        return 0.0
    try:
        t1 = datetime.fromisoformat(last_reviewed)
        t2 = datetime.fromisoformat(current_time)
        delta = t2 - t1
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


def is_due(state: FSRSState, current_time: str) -> bool:
    """Check if a hypothesis is due for review (retrievability < desired).

    Args:
        state: Current FSRS state.
        current_time: Current ISO-8601 timestamp.

    Returns:
        True if the skill's retrievability has dropped below DESIRED_RETENTION.
    """
    elapsed = _elapsed_days_since(state.last_reviewed, current_time)
    recall = estimate_recall(state, elapsed)
    return recall < DESIRED_RETENTION


def is_archive_candidate(state: FSRSState, current_time: str) -> bool:
    """Check if hypothesis should be archived (180+ days without use).

    HC-SC-5: absolute lower bound of 180 days before archiving.

    Args:
        state: Current FSRS state.
        current_time: Current ISO-8601 timestamp.

    Returns:
        True if elapsed > ARCHIVE_DAYS and retrievability is very low.
    """
    elapsed = _elapsed_days_since(state.last_reviewed, current_time)
    if elapsed < ARCHIVE_DAYS:
        return False
    recall = estimate_recall(state, elapsed)
    return recall < 0.2


def _initial_stability(rating: int) -> float:
    """Get initial stability for a first-review rating.

    Uses weights w[0]..w[3] as initial stabilities per rating.

    Args:
        rating: Review rating (1=again, 2=hard, 3=good, 4=easy).

    Returns:
        Initial stability in days.
    """
    idx = max(0, min(3, rating - 1))
    return DEFAULT_WEIGHTS[idx]


def _initial_difficulty(rating: int) -> float:
    """Compute initial difficulty for a first-review rating.

    D_init = w[4] - e^(w[5] * (rating - 1)) + 1
    Clamped to [1.0, 10.0].

    Args:
        rating: Review rating (1=again, 2=hard, 3=good, 4=easy).

    Returns:
        Initial difficulty [1.0, 10.0].
    """
    d = DEFAULT_WEIGHTS[4] - math.exp(DEFAULT_WEIGHTS[5] * (rating - 1)) + 1
    return max(1.0, min(10.0, d))


def _next_difficulty(difficulty: float, rating: int) -> float:
    """Compute updated difficulty after a review.

    delta_D = -w[6] * (rating - 3)
    linear_damping = (10 - D) * delta_D / 9  (prevents difficulty explosion)
    D' = w[7] * D_init(4) + (1 - w[7]) * (D + linear_damping)

    Clamped to [1.0, 10.0].

    Args:
        difficulty: Current difficulty.
        rating: Review rating (1-4).

    Returns:
        Updated difficulty [1.0, 10.0].
    """
    delta_d = -DEFAULT_WEIGHTS[6] * (rating - 3)

    # Linear damping prevents difficulty from escaping [1, 10]
    linear_damping = (10.0 - difficulty) * delta_d / 9.0

    # Mean reversion toward D_init(Easy)
    d_init_easy = _initial_difficulty(RATING_EASY)
    new_d = DEFAULT_WEIGHTS[7] * d_init_easy + (1.0 - DEFAULT_WEIGHTS[7]) * (
        difficulty + linear_damping
    )

    return max(1.0, min(10.0, new_d))


def _stability_after_success(
    stability: float,
    difficulty: float,
    retrievability: float,
    rating: int,
) -> float:
    """Compute new stability after successful recall (rating >= 2).

    SInc = e^(w[8]) * (11 - D) * S^(-w[9]) * (e^(w[10]*(1-R)) - 1) * bonus
    S' = S * SInc

    Hard bonus = w[15], Easy bonus = w[16], Good bonus = 1.0

    Args:
        stability: Current stability.
        difficulty: Current difficulty.
        retrievability: Current retrievability at review time.
        rating: Review rating (2=hard, 3=good, 4=easy).

    Returns:
        New stability (always >= current stability for success).
    """
    # Bonus multiplier based on rating
    if rating == RATING_HARD:
        bonus = DEFAULT_WEIGHTS[15]
    elif rating == RATING_EASY:
        bonus = DEFAULT_WEIGHTS[16]
    else:
        bonus = 1.0

    # SInc formula
    s_inc = (
        math.exp(DEFAULT_WEIGHTS[8])
        * (11.0 - difficulty)
        * math.pow(stability, -DEFAULT_WEIGHTS[9])
        * (math.exp(DEFAULT_WEIGHTS[10] * (1.0 - retrievability)) - 1.0)
        * bonus
    )

    # SInc must be >= 1.0 (stability never decreases on success)
    s_inc = max(1.0, s_inc)

    return stability * s_inc


def _stability_after_failure(
    stability: float,
    difficulty: float,
    retrievability: float,
) -> float:
    """Compute new stability after failed recall (rating = 1, "again").

    S' = w[11] * D^(-w[12]) * ((S+1)^w[13] - 1) * e^(w[14] * (1-R))

    New stability is typically much lower than current (penalty for forgetting).

    Args:
        stability: Current stability.
        difficulty: Current difficulty.
        retrievability: Current retrievability at review time.

    Returns:
        New stability (lower than current, minimum 0.1 days).
    """
    new_s = (
        DEFAULT_WEIGHTS[11]
        * math.pow(difficulty, -DEFAULT_WEIGHTS[12])
        * (math.pow(stability + 1.0, DEFAULT_WEIGHTS[13]) - 1.0)
        * math.exp(DEFAULT_WEIGHTS[14] * (1.0 - retrievability))
    )

    # Floor at 0.1 days (prevent zero/negative stability)
    return max(0.1, min(new_s, stability))


def update_fsrs(state: FSRSState, rating: int, current_time: str = "") -> FSRSState:
    """Update FSRS state after a review/application event.

    Rating semantics for Skill-Compression:
      1 = Again: skill was applied but user rejected/corrected (failure)
      2 = Hard: skill partially applied, minor correction needed
      3 = Good: skill applied successfully, user accepted
      4 = Easy: skill applied perfectly, strong positive signal

    Args:
        state: Current FSRS state.
        rating: Review rating (1=again, 2=hard, 3=good, 4=easy).
        current_time: ISO-8601 timestamp. If empty, uses UTC now.

    Returns:
        New FSRSState with updated stability, difficulty, and review history.
    """
    rating = max(1, min(4, rating))

    if not current_time:
        current_time = datetime.now(timezone.utc).isoformat()

    # Compute elapsed days
    elapsed = _elapsed_days_since(state.last_reviewed, current_time)

    # Current retrievability
    retrievability = estimate_recall(state, elapsed)

    # First review (no prior state)
    if state.reps == 0 and state.lapses == 0 and not state.last_reviewed:
        new_stability = _initial_stability(rating)
        new_difficulty = _initial_difficulty(rating)
        new_reps = 0 if rating == RATING_AGAIN else 1
        new_lapses = 1 if rating == RATING_AGAIN else 0
    else:
        # Update difficulty
        new_difficulty = _next_difficulty(state.difficulty, rating)

        # Update stability
        if rating == RATING_AGAIN:
            new_stability = _stability_after_failure(
                state.stability, state.difficulty, retrievability
            )
            new_reps = 0  # Reset rep counter on failure
            new_lapses = state.lapses + 1
        else:
            new_stability = _stability_after_success(
                state.stability, state.difficulty, retrievability, rating
            )
            new_reps = state.reps + 1
            new_lapses = state.lapses

    # Update review history for seasonal detection
    try:
        history = json.loads(state.review_history)
    except (json.JSONDecodeError, TypeError):
        history = []
    history.append(current_time)
    # Keep last 20 reviews for seasonal analysis
    if len(history) > 20:
        history = history[-20:]

    log.debug(
        "FSRS update: rating=%d elapsed=%.1fd R=%.3f "
        "S=%.2f->%.2f D=%.2f->%.2f reps=%d lapses=%d",
        rating,
        elapsed,
        retrievability,
        state.stability,
        new_stability,
        state.difficulty,
        new_difficulty,
        new_reps,
        new_lapses,
    )

    return FSRSState(
        stability=new_stability,
        difficulty=new_difficulty,
        last_reviewed=current_time,
        reps=new_reps,
        lapses=new_lapses,
        review_history=json.dumps(history, ensure_ascii=False),
    )


# ---------------------------------------------------------------
# Seasonal detection
# ---------------------------------------------------------------


def _compute_intervals_days(timestamps: list[str]) -> list[float]:
    """Compute intervals in days between consecutive timestamps.

    Args:
        timestamps: Chronologically ordered ISO-8601 timestamps.

    Returns:
        List of interval lengths in days.
    """
    if len(timestamps) < 2:
        return []

    intervals: list[float] = []
    for i in range(1, len(timestamps)):
        delta = _elapsed_days_since(timestamps[i - 1], timestamps[i])
        if delta > 0:
            intervals.append(delta)

    return intervals


def _coefficient_of_variation(values: list[float]) -> float:
    """Compute coefficient of variation (std/mean) for a list of values.

    Lower CV means more regular intervals (seasonal pattern).

    Args:
        values: List of positive float values.

    Returns:
        CV value. Returns infinity if mean is zero.
    """
    if not values:
        return float("inf")

    n = len(values)
    mean = sum(values) / n
    if mean <= 0:
        return float("inf")

    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)

    return std / mean


def seasonal_detected(state: FSRSState) -> bool:
    """Detect whether a hypothesis shows seasonal/regular usage.

    A pattern is considered seasonal if:
      1. At least MIN_REVIEWS_FOR_SEASONAL reviews exist
      2. The coefficient of variation of inter-review intervals
         is below SEASONAL_CV_THRESHOLD (intervals are regular)

    HC-SC-5: seasonal patterns are recognized and not auto-archived.

    Args:
        state: Current FSRS state with review history.

    Returns:
        True if the hypothesis shows regular/seasonal usage pattern.
    """
    try:
        history = json.loads(state.review_history)
    except (json.JSONDecodeError, TypeError):
        return False

    if len(history) < MIN_REVIEWS_FOR_SEASONAL:
        return False

    intervals = _compute_intervals_days(history)
    if len(intervals) < MIN_REVIEWS_FOR_SEASONAL - 1:
        return False

    cv = _coefficient_of_variation(intervals)
    is_seasonal = cv < SEASONAL_CV_THRESHOLD

    if is_seasonal:
        log.debug(
            "Seasonal pattern detected: CV=%.3f (threshold=%.3f), intervals=%s",
            cv,
            SEASONAL_CV_THRESHOLD,
            [f"{i:.1f}d" for i in intervals[-5:]],
        )

    return is_seasonal


def apply_seasonal_boost(state: FSRSState) -> FSRSState:
    """Apply stability boost for seasonal patterns.

    If a seasonal pattern is detected, multiply stability by
    SEASONAL_STABILITY_BOOST to prevent premature archiving.

    This is called by the Pattern Judge when evaluating archive candidates.

    Args:
        state: Current FSRS state.

    Returns:
        State with boosted stability if seasonal, unchanged otherwise.
    """
    if not seasonal_detected(state):
        return state

    boosted_stability = state.stability * SEASONAL_STABILITY_BOOST

    log.info(
        "Seasonal boost applied: stability %.2f -> %.2f",
        state.stability,
        boosted_stability,
    )

    return FSRSState(
        stability=boosted_stability,
        difficulty=state.difficulty,
        last_reviewed=state.last_reviewed,
        reps=state.reps,
        lapses=state.lapses,
        review_history=state.review_history,
    )


def next_review_interval(state: FSRSState) -> float:
    """Compute the optimal interval until next review, in days.

    interval = (S / FACTOR) * (desired_retention^(1/DECAY) - 1)

    At desired_retention=0.9 and DECAY=-0.5:
      interval = S (stability = interval when R_target = 0.9)

    Args:
        state: Current FSRS state.

    Returns:
        Optimal interval in days.
    """
    if state.stability <= 0:
        return 1.0

    # desired_retention^(1/DECAY) - 1
    exponent = 1.0 / DECAY  # = -2.0
    target_factor = math.pow(DESIRED_RETENTION, exponent) - 1.0

    interval = (state.stability / FACTOR) * target_factor

    # For our use case, interval essentially equals stability
    # because at R=0.9: (0.9^(-2) - 1) * (S / (19/81)) = S
    return max(1.0, interval)


def create_initial_fsrs_state(
    rating: int = RATING_GOOD,
    current_time: str = "",
) -> FSRSState:
    """Create a fresh FSRS state for a new hypothesis.

    Args:
        rating: Initial rating (determines initial stability).
        current_time: Timestamp. If empty, uses UTC now.

    Returns:
        Initial FSRSState.
    """
    if not current_time:
        current_time = datetime.now(timezone.utc).isoformat()

    return FSRSState(
        stability=_initial_stability(rating),
        difficulty=_initial_difficulty(rating),
        last_reviewed=current_time,
        reps=0,
        lapses=0,
        review_history=json.dumps([current_time], ensure_ascii=False),
    )
