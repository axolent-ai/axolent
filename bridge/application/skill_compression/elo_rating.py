"""Layer 2: Elo Rating System for Pattern Confidence.

Computes pattern confidence using the Elo rating system (chess model).
Pattern application success raises the rating, failure lowers it.
Change magnitude depends on request difficulty.

HC-SC-4 [BLOCKER]: Elo-Rating for pattern confidence + request difficulty.
  Initial rating: 1500.

IC-ELO-1: K-factor = 32 (chess standard). Provides good sensitivity
  for the first ~30 games, then naturally stabilizes as expected
  outcomes become more accurate.

IC-ELO-2: Request difficulty starts at 1500 and learns from observations.
  The pattern_difficulty table (from Step 1) stores per-fingerprint
  difficulty ratings that update with each match outcome.

HC-LAYER2-1: Elo ratings are confidence signals for the Evidence Ledger
  and Pattern Judge. They do NOT directly promote patterns to skills.

Mathematical foundation:
  Expected outcome: E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
  New rating: R'_A = R_A + K * (S_A - E(A))
  where S_A = 1.0 (win), 0.5 (draw), 0.0 (loss)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Default initial rating (HC-SC-4)
INITIAL_RATING: float = 1500.0

# Default K-factor (IC-ELO-1: chess standard)
DEFAULT_K_FACTOR: float = 32.0

# Elo scale factor (standard chess: 400)
SCALE_FACTOR: float = 400.0

# Outcome type
OutcomeType = Literal["pattern_wins", "pattern_loses", "draw"]

# Outcome scores
_OUTCOME_SCORES: dict[str, float] = {
    "pattern_wins": 1.0,
    "draw": 0.5,
    "pattern_loses": 0.0,
}


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EloUpdate:
    """Record of a single Elo rating update.

    Attributes:
        pattern_id: The hypothesis/pattern ID that was rated.
        old_rating: Rating before the update.
        new_rating: Rating after the update.
        request_difficulty: Elo rating of the request/challenge.
        outcome: Match result from the pattern's perspective.
        k_factor: K-factor used for this update.
    """

    pattern_id: str
    old_rating: float
    new_rating: float
    request_difficulty: float
    outcome: OutcomeType
    k_factor: float


# ---------------------------------------------------------------
# Core Elo functions
# ---------------------------------------------------------------


def expected_outcome(
    pattern_rating: float,
    request_difficulty: float,
) -> float:
    """Compute the expected outcome for a pattern vs a request.

    Uses the standard Elo formula:
      E(A) = 1 / (1 + 10^((R_B - R_A) / 400))

    A higher pattern_rating relative to request_difficulty means
    the pattern is expected to succeed (E -> 1.0).

    Args:
        pattern_rating: Current Elo rating of the pattern.
        request_difficulty: Elo rating of the request.

    Returns:
        Expected outcome [0.0, 1.0]. Higher = pattern more likely to win.
    """
    exponent = (request_difficulty - pattern_rating) / SCALE_FACTOR
    # Clamp exponent to avoid overflow in edge cases
    exponent = max(-10.0, min(10.0, exponent))
    return 1.0 / (1.0 + math.pow(10.0, exponent))


def update_elo(
    pattern_rating: float,
    request_difficulty: float,
    outcome: OutcomeType,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
) -> float:
    """Compute the new Elo rating for a pattern after a match.

    The rating change depends on:
      1. The outcome (win/draw/loss from pattern's perspective)
      2. The difference between pattern rating and request difficulty
      3. The K-factor (sensitivity)

    Key insight from Spec: "easy request wrong = big loss,
    hard request wrong = small loss." This emerges naturally from
    the Elo formula: expected_outcome is high for easy requests
    (high pattern rating vs low difficulty), so the surprise
    (actual - expected) is large when the pattern loses.

    Args:
        pattern_rating: Current Elo rating of the pattern.
        request_difficulty: Elo rating of the request.
        outcome: Match result from the pattern's perspective.
        k_factor: Sensitivity factor (default: 32.0).

    Returns:
        New Elo rating for the pattern.
    """
    expected = expected_outcome(pattern_rating, request_difficulty)
    actual = _OUTCOME_SCORES[outcome]
    new_rating = pattern_rating + k_factor * (actual - expected)

    log.debug(
        "Elo update: rating=%.1f difficulty=%.1f outcome=%s "
        "expected=%.3f actual=%.1f delta=%.1f new=%.1f",
        pattern_rating,
        request_difficulty,
        outcome,
        expected,
        actual,
        k_factor * (actual - expected),
        new_rating,
    )

    return new_rating


def compute_elo_update(
    pattern_id: str,
    pattern_rating: float,
    request_difficulty: float,
    outcome: OutcomeType,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
) -> EloUpdate:
    """Compute a full EloUpdate record for a pattern match.

    Convenience function that wraps update_elo and returns a
    structured EloUpdate dataclass.

    Args:
        pattern_id: The hypothesis/pattern ID.
        pattern_rating: Current pattern Elo rating.
        request_difficulty: Current request difficulty rating.
        outcome: Match result.
        k_factor: Sensitivity factor.

    Returns:
        EloUpdate record with old and new ratings.
    """
    new_rating = update_elo(
        pattern_rating,
        request_difficulty,
        outcome,
        k_factor=k_factor,
    )

    return EloUpdate(
        pattern_id=pattern_id,
        old_rating=pattern_rating,
        new_rating=new_rating,
        request_difficulty=request_difficulty,
        outcome=outcome,
        k_factor=k_factor,
    )


def update_difficulty(
    current_difficulty: float,
    pattern_rating: float,
    outcome: OutcomeType,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
) -> float:
    """Update the request difficulty rating after a match.

    The difficulty rating is the "opponent" in the Elo system.
    If the pattern wins, difficulty goes down (request was easier
    than expected). If the pattern loses, difficulty goes up
    (request was harder than expected).

    This is the inverse of the pattern update.

    Args:
        current_difficulty: Current difficulty Elo rating.
        pattern_rating: Elo rating of the pattern that was matched.
        outcome: Match result from the PATTERN's perspective.
        k_factor: Sensitivity factor.

    Returns:
        New difficulty rating.
    """
    # Invert outcome for the difficulty side
    inverse_outcomes: dict[str, OutcomeType] = {
        "pattern_wins": "pattern_loses",
        "pattern_loses": "pattern_wins",
        "draw": "draw",
    }
    difficulty_outcome = inverse_outcomes[outcome]

    return update_elo(
        current_difficulty,
        pattern_rating,
        difficulty_outcome,
        k_factor=k_factor,
    )


def rating_confidence_level(rating: float) -> str:
    """Classify a rating into a human-readable confidence level.

    Used for display and logging, not for algorithmic decisions.

    Args:
        rating: Current Elo rating.

    Returns:
        Confidence level string.
    """
    if rating >= 1800:
        return "high"
    if rating >= 1650:
        return "medium"
    if rating >= 1500:
        return "neutral"
    if rating >= 1350:
        return "low"
    return "very_low"
