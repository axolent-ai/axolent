"""Skill Collision Detection for Skill-Compression (Layer 5).

Resolves conflicts when multiple hypotheses match the same user request.

HC-SC-11 [BLOCKER]: Skill Collision Detection.
  - Specific scope beats global scope automatically.
  - Equal-specificity scope: user decides, NOT auto-decision.

IC-COLLISION-1: Specificity = count of non-null scope_json keys.
  More non-null keys = more specific.

Resolution strategy:
  1. Compute specificity for each candidate
  2. Group by specificity
  3. If one candidate is clearly more specific: auto-resolve (winner)
  4. If multiple candidates share the same (highest) specificity:
     requires_user_decision = True

No external dependencies (except hypothesis_storage for types).
CollisionDetector has NO database access; it operates on in-memory
SkillMatch objects passed by the SkillMatcher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CollisionResult:
    """Result of collision detection between matching hypotheses.

    Attributes:
        winner: The winning hypothesis, or None if user must decide.
        requires_user_decision: True when candidates have equal specificity.
        candidates: All candidates involved in the collision.
        resolution_reason: Human-readable explanation of the resolution.
    """

    winner: Optional[Hypothesis] = None
    requires_user_decision: bool = False
    candidates: tuple[Hypothesis, ...] = ()
    resolution_reason: str = ""


# ---------------------------------------------------------------
# Specificity computation (IC-COLLISION-1)
# ---------------------------------------------------------------


def compute_specificity(scope: HypothesisScope) -> int:
    """Compute the specificity score of a hypothesis scope.

    IC-COLLISION-1: Count non-empty scope fields.
    More non-null/non-empty fields = more specific.

    Scoring (consistent with PatternJudge._scope_specificity):
      +2 for non-empty client
      +1 for non-empty project
      +1 for each context tag (max 3 counted)

    Args:
        scope: The HypothesisScope to evaluate.

    Returns:
        Specificity score (0 = fully global, higher = more specific).
    """
    score = 0
    if scope.client:
        score += 2
    if scope.project:
        score += 1
    score += min(3, len(scope.context))
    return score


# ---------------------------------------------------------------
# CollisionDetector
# ---------------------------------------------------------------


class CollisionDetector:
    """Resolves collisions between multiple matching hypotheses.

    Stateless: no database access, no side effects. Pure logic
    operating on the SkillMatch list from SkillMatcher.

    The detector uses scope specificity as the primary resolution
    criterion:
      - Most specific scope wins automatically
      - Equal specificity: user must decide

    Thread safety: Stateless, inherently thread-safe.

    Usage:
        detector = CollisionDetector()
        result = detector.resolve(matches, current_scope)
        if result.requires_user_decision:
            # Present candidates to user
        else:
            # Apply result.winner
    """

    def resolve(
        self,
        matches: list,
        current_scope: dict,
    ) -> CollisionResult:
        """Resolve collision between multiple matching skill hypotheses.

        HC-SC-11 [BLOCKER]:
          - Specific scope beats global: auto-resolved
          - Equal specificity: user decision required

        Args:
            matches: List of SkillMatch objects (from SkillMatcher).
            current_scope: Current request scope context dict.

        Returns:
            CollisionResult with winner or user-decision flag.
        """
        if not matches:
            return CollisionResult(
                resolution_reason="No candidates to resolve.",
            )

        if len(matches) == 1:
            hyp = matches[0].hypothesis
            return CollisionResult(
                winner=hyp,
                requires_user_decision=False,
                candidates=(hyp,),
                resolution_reason="Single candidate, no collision.",
            )

        # Extract hypotheses and compute specificities
        candidates = [m.hypothesis for m in matches]
        specificities = [(compute_specificity(h.scope), h) for h in candidates]

        # Sort by specificity descending
        specificities.sort(key=lambda x: x[0], reverse=True)

        highest_spec = specificities[0][0]

        # Find all candidates with the highest specificity
        top_candidates = [h for spec, h in specificities if spec == highest_spec]

        if len(top_candidates) == 1:
            # Clear winner: most specific scope
            winner = top_candidates[0]
            log.info(
                "Collision auto-resolved: hyp=%s (specificity=%d) beats %d others",
                winner.hypothesis_id,
                highest_spec,
                len(candidates) - 1,
            )
            return CollisionResult(
                winner=winner,
                requires_user_decision=False,
                candidates=tuple(candidates),
                resolution_reason=(
                    f"hyp={winner.hypothesis_id} has highest scope specificity "
                    f"({highest_spec}), overriding {len(candidates) - 1} "
                    f"less specific candidate(s)."
                ),
            )

        # Tie: multiple candidates with same specificity
        log.info(
            "Collision tie: %d candidates with specificity=%d, user must decide",
            len(top_candidates),
            highest_spec,
        )
        return CollisionResult(
            winner=None,
            requires_user_decision=True,
            candidates=tuple(top_candidates),
            resolution_reason=(
                f"{len(top_candidates)} hypotheses have equal scope "
                f"specificity ({highest_spec}). "
                f"User decision required: "
                + ", ".join(f"'{h.claim}'" for h in top_candidates)
                + "."
            ),
        )

    def resolve_pair(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
    ) -> CollisionResult:
        """Resolve collision between exactly two hypotheses.

        Convenience method for pairwise collision checks.

        Args:
            hypothesis_a: First hypothesis.
            hypothesis_b: Second hypothesis.

        Returns:
            CollisionResult with winner or user-decision flag.
        """
        spec_a = compute_specificity(hypothesis_a.scope)
        spec_b = compute_specificity(hypothesis_b.scope)

        if spec_a == spec_b:
            return CollisionResult(
                winner=None,
                requires_user_decision=True,
                candidates=(hypothesis_a, hypothesis_b),
                resolution_reason=(
                    f"'{hypothesis_a.claim}' and '{hypothesis_b.claim}' "
                    f"have equal scope specificity ({spec_a}). "
                    f"User decision required."
                ),
            )

        if spec_a > spec_b:
            winner = hypothesis_a
            loser = hypothesis_b
        else:
            winner = hypothesis_b
            loser = hypothesis_a

        winner_spec = compute_specificity(winner.scope)
        loser_spec = compute_specificity(loser.scope)

        return CollisionResult(
            winner=winner,
            requires_user_decision=False,
            candidates=(hypothesis_a, hypothesis_b),
            resolution_reason=(
                f"'{winner.claim}' (specificity={winner_spec}) "
                f"overrides '{loser.claim}' (specificity={loser_spec})."
            ),
        )
