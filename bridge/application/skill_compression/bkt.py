"""Bayesian Knowledge Tracing (BKT) for Hypothesis Confidence.

Standard BKT model with 4 parameters, adapted for Skill-Compression.
Each hypothesis carries its own BKT state that updates with every
new evidence observation (positive or negative).

Mathematical foundation (Corbett & Anderson, 1994):

  Given:
    P(L_n) = probability that the hypothesis is "known" (correct)
    P(T)   = transition probability (hypothesis becomes correct)
    P(S)   = slip probability (hypothesis correct but observation negative)
    P(G)   = guess probability (hypothesis wrong but observation positive)

  Update on POSITIVE observation (user confirms, no correction, bookmark):
    P(L_n | positive) = P(L_n) * (1 - P(S))
                        / (P(L_n) * (1 - P(S)) + (1 - P(L_n)) * P(G))

  Update on NEGATIVE observation (correction, rejection):
    P(L_n | negative) = P(L_n) * P(S)
                        / (P(L_n) * P(S) + (1 - P(L_n)) * (1 - P(G)))

  Learning transition (applied AFTER each observation update):
    P(L_{n+1}) = P(L_n | obs) + (1 - P(L_n | obs)) * P(T)

Default parameters (IC-BKT-1):
  P(L0) = 0.5   (uninformative prior, 50/50)
  P(T)  = 0.1   (slow learning, hypothesis gains credibility gradually)
  P(S)  = 0.1   (low slip rate, correct hypothesis rarely contradicted)
  P(G)  = 0.2   (moderate guess rate, false positives somewhat likely)

HC-LAYER2-1: BKT confidence is a signal for the Evidence Ledger
and Pattern Judge. It does NOT directly promote patterns.

No external dependencies. Pure Python math.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Default parameters (IC-BKT-1)
# ---------------------------------------------------------------

DEFAULT_P_INIT: float = 0.5
DEFAULT_P_TRANSITION: float = 0.1
DEFAULT_P_SLIP: float = 0.1
DEFAULT_P_GUESS: float = 0.2


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BKTState:
    """Bayesian Knowledge Tracing state for a single hypothesis.

    Attributes:
        p_knowledge: Current belief that the hypothesis is correct [0, 1].
        p_init: Prior probability (P(L0)).
        p_transition: Learning transition probability (P(T)).
        p_slip: Slip probability (P(S)).
        p_guess: Guess probability (P(G)).
        observations: Total number of observations processed.
    """

    p_knowledge: float = DEFAULT_P_INIT
    p_init: float = DEFAULT_P_INIT
    p_transition: float = DEFAULT_P_TRANSITION
    p_slip: float = DEFAULT_P_SLIP
    p_guess: float = DEFAULT_P_GUESS
    observations: int = 0


# ---------------------------------------------------------------
# Core BKT update
# ---------------------------------------------------------------


def _posterior_given_observation(
    p_know: float,
    p_slip: float,
    p_guess: float,
    *,
    positive: bool,
) -> float:
    """Compute posterior P(L_n | observation) via Bayes' theorem.

    Args:
        p_know: Prior P(L_n).
        p_slip: Slip probability P(S).
        p_guess: Guess probability P(G).
        positive: True for positive observation, False for negative.

    Returns:
        Posterior probability [0, 1].
    """
    if positive:
        # P(L | positive) = P(L) * (1 - P(S))
        #                  / [P(L) * (1 - P(S)) + (1 - P(L)) * P(G)]
        numerator = p_know * (1.0 - p_slip)
        denominator = numerator + (1.0 - p_know) * p_guess
    else:
        # P(L | negative) = P(L) * P(S)
        #                  / [P(L) * P(S) + (1 - P(L)) * (1 - P(G))]
        numerator = p_know * p_slip
        denominator = numerator + (1.0 - p_know) * (1.0 - p_guess)

    if denominator <= 0.0:
        # Edge case: avoid division by zero. Return prior unchanged.
        return p_know

    return numerator / denominator


def _apply_transition(p_posterior: float, p_transition: float) -> float:
    """Apply the learning transition after observation update.

    P(L_{n+1}) = P(L_n | obs) + (1 - P(L_n | obs)) * P(T)

    This models the idea that even if the posterior is low, there is
    a small probability the hypothesis "becomes correct" over time
    (the user's preference may solidify).

    Args:
        p_posterior: Posterior probability after observation.
        p_transition: Transition probability P(T).

    Returns:
        Updated knowledge probability [0, 1].
    """
    return p_posterior + (1.0 - p_posterior) * p_transition


def update_bkt(state: BKTState, positive_observation: bool) -> BKTState:
    """Update BKT state with a new observation.

    Performs the full BKT cycle:
      1. Compute posterior given observation (Bayes update)
      2. Apply learning transition

    Returns a new BKTState (frozen dataclass, no mutation).

    Args:
        state: Current BKT state.
        positive_observation: True if the observation supports the
            hypothesis (no correction, bookmark, confirm, learn_command).
            False if it contradicts (correction, rejection).

    Returns:
        New BKTState with updated p_knowledge and incremented observations.
    """
    # Step 1: Bayesian posterior
    posterior = _posterior_given_observation(
        p_know=state.p_knowledge,
        p_slip=state.p_slip,
        p_guess=state.p_guess,
        positive=positive_observation,
    )

    # Step 2: Learning transition
    new_knowledge = _apply_transition(posterior, state.p_transition)

    # Clamp to [0.001, 0.999] to avoid numerical dead zones
    new_knowledge = max(0.001, min(0.999, new_knowledge))

    log.debug(
        "BKT update: p_know=%.4f -> posterior=%.4f -> new=%.4f (positive=%s, obs=%d)",
        state.p_knowledge,
        posterior,
        new_knowledge,
        positive_observation,
        state.observations + 1,
    )

    return BKTState(
        p_knowledge=new_knowledge,
        p_init=state.p_init,
        p_transition=state.p_transition,
        p_slip=state.p_slip,
        p_guess=state.p_guess,
        observations=state.observations + 1,
    )


def update_bkt_weighted(
    state: BKTState,
    positive_observation: bool,
    weight: float = 1.0,
) -> BKTState:
    """Update BKT state with a weighted observation.

    For strong signals (e.g. learn_command with weight=1.0) the full
    BKT update is applied. For weaker signals (e.g. no_correction
    with weight=0.5) the update is interpolated: the new p_knowledge
    is a weighted blend between old and fully-updated value.

    This allows signal_strength from EvidenceRecord to modulate BKT
    updates without changing the core BKT math.

    Args:
        state: Current BKT state.
        positive_observation: Whether observation is positive.
        weight: Signal weight [0.0, 1.0]. 1.0 = full update.

    Returns:
        New BKTState with weighted update.
    """
    weight = max(0.0, min(1.0, weight))

    if weight <= 0.0:
        # Zero weight: no update, just increment observation counter
        return BKTState(
            p_knowledge=state.p_knowledge,
            p_init=state.p_init,
            p_transition=state.p_transition,
            p_slip=state.p_slip,
            p_guess=state.p_guess,
            observations=state.observations + 1,
        )

    # Full BKT update
    full_update = update_bkt(state, positive_observation)

    if weight >= 1.0:
        return full_update

    # Interpolate between old and fully-updated knowledge
    blended = state.p_knowledge + weight * (full_update.p_knowledge - state.p_knowledge)
    blended = max(0.001, min(0.999, blended))

    return BKTState(
        p_knowledge=blended,
        p_init=state.p_init,
        p_transition=state.p_transition,
        p_slip=state.p_slip,
        p_guess=state.p_guess,
        observations=state.observations + 1,
    )


def batch_update_bkt(
    state: BKTState,
    observations: list[bool],
) -> BKTState:
    """Apply multiple observations sequentially.

    Convenience function for replaying evidence history.

    Args:
        state: Initial BKT state.
        observations: Ordered list of observations (True=positive).

    Returns:
        Final BKTState after all observations.
    """
    current = state
    for obs in observations:
        current = update_bkt(current, obs)
    return current


def create_initial_state(
    *,
    p_init: float = DEFAULT_P_INIT,
    p_transition: float = DEFAULT_P_TRANSITION,
    p_slip: float = DEFAULT_P_SLIP,
    p_guess: float = DEFAULT_P_GUESS,
) -> BKTState:
    """Create an initial BKT state with custom parameters.

    Args:
        p_init: Initial knowledge probability.
        p_transition: Learning transition probability.
        p_slip: Slip probability.
        p_guess: Guess probability.

    Returns:
        Fresh BKTState with zero observations.
    """
    return BKTState(
        p_knowledge=p_init,
        p_init=p_init,
        p_transition=p_transition,
        p_slip=p_slip,
        p_guess=p_guess,
        observations=0,
    )
