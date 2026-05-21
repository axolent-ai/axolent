"""Layer 4: Pattern Judge for Skill-Compression.

Evaluates hypotheses and manages their 7-status lifecycle based on:
  1. Evidence Summary (from EvidenceLedger, Layer 3)
  2. BKT State (Bayesian confidence)
  3. Elo Rating (pattern confidence)
  4. FSRS State (decay/freshness)

Privacy Pipeline (Step 8, Layer 7 integration):
  Before ANY promotion (candidate->suggested, confirmed->active),
  the Privacy Pipeline runs three hard filters:
    - HealthcareFilter (HC-SC-14): blocks clinical phenotyping
    - SecretScanner (HC-SC-13): blocks secrets/PII
    - NudgeFilter (HC-SC-15): blocks nudge policy violations
  If ANY filter rejects, the hypothesis is NOT promoted.

Lifecycle (7 status, HC-SC-1):
  1. candidate   - 1-2 evidence items, internal, user sees nothing
  2. suggested   - 3-5 evidence over 2+ sessions, bot asks user
  3. confirmed   - user confirmed, applied with "Ask Before Applying"
  4. active      - auto-apply threshold reached, applied without asking
  5. needs_review - contradiction detected, goes to user question mode
  6. paused      - user manually paused, stored but not applied
  7. archived    - 180+ days unused (FSRS), user-created: never auto

Plus: retired (user /forget -> tombstone for 30 days)

HC-SC-10 [BLOCKER]: "Ask Before Applying" as default. Auto-Apply
  only after risk-differentiated thresholds are reached.

HC-SC-11 [BLOCKER]: Skill Collision Detection. Specific scope beats
  global scope automatically. Equal-specificity: ask user.

HC-SC-3 [BLOCKER]: Auto-Apply thresholds scope-differentiated:
  Negative specific: 2 confirmations, Elo >= 1650
  Negative domain: 4 confirmations, Elo >= 1700
  Negative global: 6 confirmations, Elo >= 1750
  Preference: 5 confirmations, Elo >= 1700
  Procedural: 8 confirmations, Elo >= 1800

AG: Pattern Judge imports ONLY via EvidenceLedger + Hypothesis.
  Never directly from N-Gram/Markov/Elo modules.
AG: Privacy filters called BEFORE any promotion. No bypass possible.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from application.skill_compression.bkt import BKTState
from application.skill_compression.evidence_ledger import EvidenceSummary
from application.skill_compression.fsrs_decay import (
    FSRSState,
    apply_seasonal_boost,
    is_archive_candidate,
    seasonal_detected,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Lifecycle status constants
# ---------------------------------------------------------------

STATUS_CANDIDATE = "candidate"
STATUS_SUGGESTED = "suggested"
STATUS_CONFIRMED = "confirmed"
STATUS_ACTIVE = "active"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_PAUSED = "paused"
STATUS_ARCHIVED = "archived"
STATUS_RETIRED = "retired"

ALL_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_CANDIDATE,
        STATUS_SUGGESTED,
        STATUS_CONFIRMED,
        STATUS_ACTIVE,
        STATUS_NEEDS_REVIEW,
        STATUS_PAUSED,
        STATUS_ARCHIVED,
        STATUS_RETIRED,
    }
)

# ---------------------------------------------------------------
# Auto-Apply thresholds (HC-SC-3)
# Risk-differentiated by pattern type and scope breadth
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutoApplyThreshold:
    """Threshold configuration for auto-apply promotion.

    A hypothesis must meet ALL conditions to be promoted from
    confirmed -> active (auto-apply without asking).

    Attributes:
        min_confirmations: Minimum positive evidence count.
        min_elo_rating: Minimum Elo rating.
        min_sessions: Minimum distinct sessions.
        min_bkt_confidence: Minimum BKT p_knowledge.
    """

    min_confirmations: int
    min_elo_rating: float
    min_sessions: int = 2
    min_bkt_confidence: float = 0.7


# HC-SC-3: scope-differentiated thresholds
THRESHOLDS: dict[str, AutoApplyThreshold] = {
    # Negative patterns: low bar because avoiding mistakes is high value
    "negative_specific": AutoApplyThreshold(
        min_confirmations=2,
        min_elo_rating=1650.0,
        min_sessions=2,
        min_bkt_confidence=0.65,
    ),
    "negative_domain": AutoApplyThreshold(
        min_confirmations=4,
        min_elo_rating=1700.0,
        min_sessions=2,
        min_bkt_confidence=0.70,
    ),
    "negative_global": AutoApplyThreshold(
        min_confirmations=6,
        min_elo_rating=1750.0,
        min_sessions=3,
        min_bkt_confidence=0.75,
    ),
    # Preference patterns: moderate bar
    "preference": AutoApplyThreshold(
        min_confirmations=5,
        min_elo_rating=1700.0,
        min_sessions=2,
        min_bkt_confidence=0.70,
    ),
    # Procedural patterns: high bar (complex workflows, higher risk)
    "procedural": AutoApplyThreshold(
        min_confirmations=8,
        min_elo_rating=1800.0,
        min_sessions=3,
        min_bkt_confidence=0.80,
    ),
    # Default for request type and unknown types
    "default": AutoApplyThreshold(
        min_confirmations=5,
        min_elo_rating=1700.0,
        min_sessions=2,
        min_bkt_confidence=0.70,
    ),
}

# Needs-review trigger: N contradictions in last M observations
CONTRADICTION_WINDOW: int = 5
CONTRADICTION_THRESHOLD: int = 3

# Candidate -> Suggested: minimum evidence and sessions
SUGGEST_MIN_EVIDENCE: int = 3
SUGGEST_MIN_SESSIONS: int = 2


# ---------------------------------------------------------------
# Scope specificity (for collision detection, HC-SC-11)
# ---------------------------------------------------------------


def _scope_specificity(scope: HypothesisScope) -> int:
    """Compute a specificity score for a hypothesis scope.

    Higher = more specific. Used for collision resolution:
    more specific scope wins automatically.

    Scoring:
      +2 for non-empty client
      +1 for non-empty project
      +1 for each context tag (max 3 counted)

    Args:
        scope: HypothesisScope to evaluate.

    Returns:
        Specificity score (0 = fully global).
    """
    score = 0
    if scope.client:
        score += 2
    if scope.project:
        score += 1
    # Count context tags, cap at 3
    score += min(3, len(scope.context))
    return score


# ---------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------


def _resolve_threshold_key(hypothesis: Hypothesis) -> str:
    """Determine which threshold category applies to a hypothesis.

    Uses hypothesis type and scope to select the appropriate
    risk-differentiated threshold.

    Args:
        hypothesis: The hypothesis to classify.

    Returns:
        Threshold key string (maps to THRESHOLDS dict).
    """
    h_type = hypothesis.type.lower()

    if h_type == "negative":
        # Scope-differentiated negative thresholds
        specificity = _scope_specificity(hypothesis.scope)
        if specificity >= 2:
            return "negative_specific"
        elif specificity >= 1:
            return "negative_domain"
        else:
            return "negative_global"
    elif h_type == "preference":
        return "preference"
    elif h_type == "procedural":
        return "procedural"
    else:
        return "default"


# ---------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    """Result of Pattern Judge evaluation.

    Attributes:
        hypothesis_id: The evaluated hypothesis.
        current_status: Status before evaluation.
        recommended_status: Recommended new status.
        should_transition: Whether a status change is recommended.
        reason: Human-readable explanation for the decision.
        threshold_key: Which threshold category was applied.
    """

    hypothesis_id: str
    current_status: str
    recommended_status: str
    should_transition: bool
    reason: str
    threshold_key: str = ""


@dataclass(frozen=True, slots=True)
class CollisionResult:
    """Result of skill collision detection.

    Attributes:
        has_collision: Whether a collision was detected.
        winner_id: Hypothesis ID that wins (if auto-resolved).
        loser_id: Hypothesis ID that loses (if auto-resolved).
        needs_user_decision: Whether user must decide.
        reason: Explanation of resolution.
    """

    has_collision: bool
    winner_id: Optional[str] = None
    loser_id: Optional[str] = None
    needs_user_decision: bool = False
    reason: str = ""


# ---------------------------------------------------------------
# Pattern Judge
# ---------------------------------------------------------------


STATUS_PRIVACY_REJECTED = "privacy_rejected"


class PatternJudge:
    """Layer 4: Lifecycle manager for hypotheses.

    Evaluates hypothesis state against evidence, BKT, Elo, and FSRS
    to recommend lifecycle transitions.

    Privacy Pipeline (Step 8): Before any promotion, the judge runs
    three hard privacy filters (healthcare, secrets, nudge). If any
    filter rejects, the hypothesis is NOT promoted and receives a
    REJECT decision.

    The judge does NOT mutate hypotheses directly. It returns
    JudgeDecision objects that the orchestrator uses to trigger
    status updates via HypothesisStorage.

    Usage:
        from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline

        judge = PatternJudge(privacy_pipeline=PrivacyPipeline())
        decision = judge.evaluate(hypothesis, evidence_summary, bkt_state, elo, fsrs)
        if decision.should_transition:
            storage.update_hypothesis_status(hyp_id, decision.recommended_status)
    """

    def __init__(self, *, privacy_pipeline=None) -> None:
        """Initialize the Pattern Judge.

        Args:
            privacy_pipeline: Optional PrivacyPipeline instance.
                If None, privacy checks are skipped (for backwards
                compatibility with existing tests).
                In production, MUST be provided.
        """
        self._privacy = privacy_pipeline

    def evaluate(
        self,
        hypothesis: Hypothesis,
        evidence: EvidenceSummary,
        bkt: BKTState,
        elo: float,
        fsrs: FSRSState,
        *,
        current_time: str = "",
        recent_contradictions: int = 0,
    ) -> JudgeDecision:
        """Evaluate a hypothesis and recommend status transition.

        Checks (in priority order):
          1. Should it be archived? (FSRS decay, 180+ days)
          2. Should it go to needs_review? (recent contradictions)
          3. Should it be promoted? (candidate->suggested->confirmed->active)

        Args:
            hypothesis: The hypothesis to evaluate.
            evidence: Aggregated evidence summary.
            bkt: Current BKT state.
            elo: Current Elo rating.
            fsrs: Current FSRS state.
            current_time: ISO-8601 timestamp for FSRS checks.
            recent_contradictions: Count of recent negative signals.

        Returns:
            JudgeDecision with recommendation.
        """
        current_status = hypothesis.status

        # Rule 1: Check for archiving (only non-immune, non-paused)
        if current_status not in (STATUS_PAUSED, STATUS_ARCHIVED, STATUS_RETIRED):
            if not hypothesis.decay_immune and current_time:
                archive_decision = self._check_archive(hypothesis, fsrs, current_time)
                if archive_decision is not None:
                    return archive_decision

        # Rule 2: Check for needs_review (contradiction burst)
        if current_status in (STATUS_CONFIRMED, STATUS_ACTIVE):
            review_decision = self._check_needs_review(
                hypothesis, recent_contradictions
            )
            if review_decision is not None:
                return review_decision

        # Rule 3: Privacy pipeline (HC-SC-13, HC-SC-14, HC-SC-15)
        # Must run BEFORE any promotion to prevent blocked hypotheses
        # from ever reaching suggested/confirmed/active status.
        if self._privacy is not None:
            rejection = self._privacy.check(hypothesis)
            if rejection is not None:
                return JudgeDecision(
                    hypothesis_id=hypothesis.hypothesis_id,
                    current_status=current_status,
                    recommended_status=STATUS_PRIVACY_REJECTED,
                    should_transition=True,
                    reason=f"Privacy pipeline rejection: {rejection.reason}",
                )

        # Rule 4: Check for promotion
        promotion_decision = self._check_promotion(hypothesis, evidence, bkt, elo)
        if promotion_decision is not None:
            return promotion_decision

        # No transition recommended
        return JudgeDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            current_status=current_status,
            recommended_status=current_status,
            should_transition=False,
            reason="No transition criteria met",
        )

    def should_promote(
        self,
        hypothesis: Hypothesis,
        evidence: EvidenceSummary,
        bkt: BKTState,
    ) -> bool:
        """Check if a hypothesis should be promoted to the next stage.

        Simplified boolean check for external use.

        Args:
            hypothesis: The hypothesis to check.
            evidence: Evidence summary.
            bkt: BKT state.

        Returns:
            True if promotion criteria are met.
        """
        decision = self._check_promotion(
            hypothesis, evidence, bkt, hypothesis.elo_rating
        )
        return decision is not None and decision.should_transition

    def should_archive(
        self,
        hypothesis: Hypothesis,
        fsrs: FSRSState,
        current_time: str,
    ) -> bool:
        """Check if a hypothesis should be archived.

        Respects decay_immune flag (HC-SC-6).

        Args:
            hypothesis: The hypothesis to check.
            fsrs: FSRS state.
            current_time: Current ISO-8601 timestamp.

        Returns:
            True if archiving is recommended.
        """
        if hypothesis.decay_immune:
            return False
        decision = self._check_archive(hypothesis, fsrs, current_time)
        return decision is not None

    def should_challenge(
        self,
        hypothesis: Hypothesis,
        recent_contradictions: int,
    ) -> bool:
        """Check if a hypothesis should go to needs_review.

        Triggered when recent contradictions exceed threshold.

        Args:
            hypothesis: The hypothesis to check.
            recent_contradictions: Number of recent negative signals.

        Returns:
            True if challenge is warranted.
        """
        if hypothesis.status not in (STATUS_CONFIRMED, STATUS_ACTIVE):
            return False
        return recent_contradictions >= CONTRADICTION_THRESHOLD

    def detect_collision(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
    ) -> CollisionResult:
        """Detect and resolve collision between two hypotheses.

        HC-SC-11: Specific scope beats global automatically.
        Equal specificity: needs user decision.

        Args:
            hypothesis_a: First conflicting hypothesis.
            hypothesis_b: Second conflicting hypothesis.

        Returns:
            CollisionResult with resolution.
        """
        spec_a = _scope_specificity(hypothesis_a.scope)
        spec_b = _scope_specificity(hypothesis_b.scope)

        if spec_a == spec_b:
            # Equal specificity: user must decide
            return CollisionResult(
                has_collision=True,
                needs_user_decision=True,
                reason=(
                    f"Hypotheses '{hypothesis_a.claim}' and "
                    f"'{hypothesis_b.claim}' have equal scope specificity "
                    f"({spec_a}). User decision required."
                ),
            )

        # More specific scope wins
        if spec_a > spec_b:
            winner = hypothesis_a
            loser = hypothesis_b
        else:
            winner = hypothesis_b
            loser = hypothesis_a

        return CollisionResult(
            has_collision=True,
            winner_id=winner.hypothesis_id,
            loser_id=loser.hypothesis_id,
            needs_user_decision=False,
            reason=(
                f"'{winner.claim}' (specificity "
                f"{_scope_specificity(winner.scope)}) overrides "
                f"'{loser.claim}' (specificity "
                f"{_scope_specificity(loser.scope)})."
            ),
        )

    # ── Private evaluation methods ──────────────────────────────

    def _check_archive(
        self,
        hypothesis: Hypothesis,
        fsrs: FSRSState,
        current_time: str,
    ) -> Optional[JudgeDecision]:
        """Check if hypothesis qualifies for archiving.

        HC-SC-6: decay_immune hypotheses are never auto-archived.
        HC-SC-5: seasonal patterns get stability boost before check.

        Args:
            hypothesis: The hypothesis.
            fsrs: FSRS state.
            current_time: Current timestamp.

        Returns:
            JudgeDecision for archive, or None if not applicable.
        """
        if hypothesis.decay_immune:
            return None

        # Apply seasonal boost before archive check
        effective_fsrs = fsrs
        if seasonal_detected(fsrs):
            effective_fsrs = apply_seasonal_boost(fsrs)

        if is_archive_candidate(effective_fsrs, current_time):
            return JudgeDecision(
                hypothesis_id=hypothesis.hypothesis_id,
                current_status=hypothesis.status,
                recommended_status=STATUS_ARCHIVED,
                should_transition=True,
                reason=(
                    "FSRS decay: hypothesis unused for 180+ days "
                    "with very low retrievability."
                ),
            )

        return None

    def _check_needs_review(
        self,
        hypothesis: Hypothesis,
        recent_contradictions: int,
    ) -> Optional[JudgeDecision]:
        """Check if hypothesis needs review due to contradictions.

        Triggers when recent_contradictions >= CONTRADICTION_THRESHOLD.

        Args:
            hypothesis: The hypothesis.
            recent_contradictions: Count in recent window.

        Returns:
            JudgeDecision for needs_review, or None.
        """
        if recent_contradictions >= CONTRADICTION_THRESHOLD:
            return JudgeDecision(
                hypothesis_id=hypothesis.hypothesis_id,
                current_status=hypothesis.status,
                recommended_status=STATUS_NEEDS_REVIEW,
                should_transition=True,
                reason=(
                    f"{recent_contradictions} contradictions in last "
                    f"{CONTRADICTION_WINDOW} observations. "
                    "Hypothesis needs user review."
                ),
            )
        return None

    def _check_promotion(
        self,
        hypothesis: Hypothesis,
        evidence: EvidenceSummary,
        bkt: BKTState,
        elo: float,
    ) -> Optional[JudgeDecision]:
        """Check if hypothesis should be promoted to next lifecycle stage.

        Promotion paths:
          candidate -> suggested: 3+ evidence, 2+ sessions
          suggested -> confirmed: requires explicit user confirmation
            (not auto-promoted by judge, but signaled)
          confirmed -> active: auto-apply threshold met

        Args:
            hypothesis: The hypothesis.
            evidence: Evidence summary.
            bkt: BKT state.
            elo: Current Elo rating.

        Returns:
            JudgeDecision for promotion, or None.
        """
        current = hypothesis.status

        # candidate -> suggested
        if current == STATUS_CANDIDATE:
            if (
                evidence.total_count >= SUGGEST_MIN_EVIDENCE
                and evidence.distinct_sessions >= SUGGEST_MIN_SESSIONS
            ):
                return JudgeDecision(
                    hypothesis_id=hypothesis.hypothesis_id,
                    current_status=current,
                    recommended_status=STATUS_SUGGESTED,
                    should_transition=True,
                    reason=(
                        f"Evidence threshold met: {evidence.total_count} items "
                        f"over {evidence.distinct_sessions} sessions "
                        f"(need {SUGGEST_MIN_EVIDENCE}/{SUGGEST_MIN_SESSIONS})."
                    ),
                )

        # confirmed -> active (auto-apply threshold)
        elif current == STATUS_CONFIRMED:
            threshold_key = _resolve_threshold_key(hypothesis)
            threshold = THRESHOLDS.get(threshold_key, THRESHOLDS["default"])

            if self._meets_auto_apply_threshold(evidence, bkt, elo, threshold):
                return JudgeDecision(
                    hypothesis_id=hypothesis.hypothesis_id,
                    current_status=current,
                    recommended_status=STATUS_ACTIVE,
                    should_transition=True,
                    reason=(
                        f"Auto-apply threshold met ({threshold_key}): "
                        f"confirmations={evidence.positive_count}/"
                        f"{threshold.min_confirmations}, "
                        f"elo={elo:.0f}/{threshold.min_elo_rating:.0f}, "
                        f"bkt={bkt.p_knowledge:.3f}/"
                        f"{threshold.min_bkt_confidence:.3f}, "
                        f"sessions={evidence.distinct_sessions}/"
                        f"{threshold.min_sessions}."
                    ),
                    threshold_key=threshold_key,
                )

        # suggested -> confirmed is NOT auto-promoted.
        # It requires explicit user confirmation (HC-SC-10).
        # The judge signals "suggested" status; the UI asks the user.

        return None

    def _meets_auto_apply_threshold(
        self,
        evidence: EvidenceSummary,
        bkt: BKTState,
        elo: float,
        threshold: AutoApplyThreshold,
    ) -> bool:
        """Check all auto-apply conditions against threshold.

        ALL conditions must be met simultaneously:
          1. Positive evidence count >= min_confirmations
          2. Elo rating >= min_elo_rating
          3. Distinct sessions >= min_sessions
          4. BKT confidence >= min_bkt_confidence

        Args:
            evidence: Evidence summary.
            bkt: BKT state.
            elo: Elo rating.
            threshold: Threshold configuration.

        Returns:
            True if all conditions met.
        """
        if evidence.positive_count < threshold.min_confirmations:
            return False
        if elo < threshold.min_elo_rating:
            return False
        if evidence.distinct_sessions < threshold.min_sessions:
            return False
        if bkt.p_knowledge < threshold.min_bkt_confidence:
            return False
        return True
