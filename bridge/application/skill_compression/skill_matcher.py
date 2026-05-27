"""Layer 5: SkillMatcher for Skill-Compression.

Matches incoming user requests against the hypothesis pool to find
applicable learned skills. This is the bridge between pattern storage
and skill application.

Match strategy (from Spec, Layer 5):
  1. Direct alias match in hypothesis_aliases (fast, <5ms target)
  2. Fingerprint similarity against active hypotheses (via FingerprintMatcher)
  3. Only hypotheses with status 'confirmed' or 'active' are matched
  4. 'candidate' and 'suggested' are NEVER applied
  5. Score threshold: > 0.7

HC-SC-10 [BLOCKER]: "Ask Before Applying" as default.
  Auto-Apply only after threshold reached AND user opt-in.

IC-MATCH-1: Match score = fingerprint_score * elo_quotient
  where elo_quotient = min(elo_rating / 2000, 1.0).

AG: SkillMatcher does NOT import N-Gram/Markov/Elo directly.
  It accesses hypotheses only through HypothesisStorage and
  uses FingerprintMatcher + PatternJudge for scoring/evaluation.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.fingerprint_matcher import (
    FingerprintMatch,
    compute_similarity,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import (
    STATUS_ACTIVE,
    STATUS_CONFIRMED,
    PatternJudge,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

# Minimum match score to consider a hypothesis (Spec: > 0.7)
MATCH_SCORE_THRESHOLD: float = 0.7

# Elo normalizer for IC-MATCH-1 score computation
ELO_NORMALIZER: float = 2000.0

# Statuses eligible for matching (only confirmed + active)
MATCHABLE_STATUSES: frozenset[str] = frozenset({STATUS_CONFIRMED, STATUS_ACTIVE})

# Default user preferences
DEFAULT_USER_PREFERENCES: dict[str, object] = {
    "auto_apply_enabled": False,
}


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SkillMatch:
    """Result of matching a user request against a hypothesis.

    Attributes:
        hypothesis: The matched hypothesis.
        confidence: Combined match confidence [0.0, 1.0].
        requires_confirmation: True if user must confirm before applying.
        explanation: Human-readable reason for this match.
        match_source: How the match was found ('alias' or 'fingerprint').
    """

    hypothesis: Hypothesis
    confidence: float
    requires_confirmation: bool
    explanation: str
    match_source: str = "fingerprint"


# ---------------------------------------------------------------
# Ask Before Applying (HC-SC-10)
# ---------------------------------------------------------------


def should_ask_user(
    match: SkillMatch,
    user_preferences: dict | None = None,
) -> bool:
    """Determine whether to ask user before applying a matched skill.

    HC-SC-10 [BLOCKER]: "Ask Before Applying" as default.

    Logic:
      - Status 'confirmed': ALWAYS ask (requires user confirmation)
      - Status 'active' + auto_apply_enabled=False: ask
      - Status 'active' + auto_apply_enabled=True: do not ask
      - Default: auto_apply_enabled = False (Ask Before Applying)

    Args:
        match: The SkillMatch to evaluate.
        user_preferences: User settings dict. Expected key:
            'auto_apply_enabled' (bool, default False).

    Returns:
        True if user confirmation is required before applying.
    """
    if user_preferences is None:
        user_preferences = dict(DEFAULT_USER_PREFERENCES)

    status = match.hypothesis.status

    # Confirmed: always ask (user has confirmed the skill exists,
    # but hasn't reached auto-apply threshold yet)
    if status == STATUS_CONFIRMED:
        return True

    # Active: depends on user preference
    if status == STATUS_ACTIVE:
        auto_apply = user_preferences.get("auto_apply_enabled", False)
        return not bool(auto_apply)

    # Any other status should not reach here (not matchable),
    # but defensively: always ask
    return True


# ---------------------------------------------------------------
# SkillMatcher
# ---------------------------------------------------------------


class SkillMatcher:
    """Layer 5: Matches user requests against learned skill hypotheses.

    Uses a two-stage matching strategy:
      1. Fast alias lookup (exact text match in hypothesis_aliases)
      2. Fingerprint similarity for fuzzy matching

    Only hypotheses with status 'confirmed' or 'active' are considered.
    The matcher does NOT apply skills; it returns SkillMatch objects
    that the orchestrator uses for the application decision.

    Thread safety: NOT thread-safe. Designed for single-threaded
    async event loop (Telegram bot context).

    Usage:
        matcher = SkillMatcher(storage, judge)
        result = matcher.match(event)
        if result is not None:
            if should_ask_user(result, user_prefs):
                # Ask user for confirmation
            else:
                # Apply automatically
    """

    def __init__(
        self,
        storage: HypothesisStorage,
        pattern_judge: PatternJudge,
    ) -> None:
        """Initialize the SkillMatcher.

        Args:
            storage: Hypothesis storage for DB access.
            pattern_judge: Pattern Judge for collision detection.
        """
        self._storage = storage
        self._judge = pattern_judge

    @property
    def storage(self) -> HypothesisStorage:
        """Public accessor for hypothesis storage.

        Provides controlled access for callers that need to write
        evidence (e.g. ChatService). Avoids private-attribute access.
        """
        return self._storage

    def match(
        self,
        event: NormalizedEvent,
        scope_hint: dict | None = None,
    ) -> Optional[SkillMatch]:
        """Match a user event against the hypothesis pool.

        Strategy:
          1. Try direct alias match (fastest path)
          2. Fall back to fingerprint similarity
          3. If multiple matches: delegate to CollisionDetector
          4. Score threshold > 0.7

        Args:
            event: Normalized event from the user request.
            scope_hint: Optional scope context for filtering.

        Returns:
            SkillMatch if a match is found, None otherwise.
        """
        user_id = event.user_id

        # Stage 1: Direct alias match
        alias_match = self._try_alias_match(event, user_id)
        if alias_match is not None:
            log.info(
                "Alias match found: hyp=%s confidence=%.3f",
                alias_match.hypothesis.hypothesis_id,
                alias_match.confidence,
            )
            return alias_match

        # Stage 2: Fingerprint similarity match
        fingerprint_matches = self._try_fingerprint_match(event, user_id, scope_hint)

        if not fingerprint_matches:
            log.debug("No matches found for event %s", event.event_id)
            return None

        # Single match: return directly
        if len(fingerprint_matches) == 1:
            return fingerprint_matches[0]

        # Multiple matches: resolve collision
        return self._resolve_collision(fingerprint_matches)

    def match_all(
        self,
        event: NormalizedEvent,
        scope_hint: dict | None = None,
    ) -> list[SkillMatch]:
        """Find all matching hypotheses (for UI display / debugging).

        Unlike match(), does not resolve collisions. Returns all
        candidates above threshold, sorted by confidence.

        Args:
            event: Normalized event from the user request.
            scope_hint: Optional scope context for filtering.

        Returns:
            List of SkillMatch objects, sorted by confidence descending.
        """
        user_id = event.user_id
        all_matches: list[SkillMatch] = []

        # Alias matches
        alias_match = self._try_alias_match(event, user_id)
        if alias_match is not None:
            all_matches.append(alias_match)

        # Fingerprint matches
        fp_matches = self._try_fingerprint_match(event, user_id, scope_hint)
        # Avoid duplicates (same hypothesis_id)
        seen_ids = {m.hypothesis.hypothesis_id for m in all_matches}
        for m in fp_matches:
            if m.hypothesis.hypothesis_id not in seen_ids:
                all_matches.append(m)
                seen_ids.add(m.hypothesis.hypothesis_id)

        # Sort by confidence descending
        all_matches.sort(key=lambda m: m.confidence, reverse=True)
        return all_matches

    # ── Private matching methods ──────────────────────────────

    @staticmethod
    def _normalize_german(text: str) -> str:
        """Normalize text for matching: sharp-s / double-s equivalence + lowercase.

        Round-4 fix: Users may type 'weiss' or the sharp-s variant
        interchangeably. Both must match the same alias regardless of
        stored form.

        Args:
            text: Input text to normalize.

        Returns:
            Lowercased, stripped text with sharp-s replaced by double-s.
        """
        return text.strip().lower().replace("ß", "ss")

    def _try_alias_match(
        self,
        event: NormalizedEvent,
        user_id: int,
    ) -> Optional[SkillMatch]:
        """Try to match via direct alias lookup.

        Checks hypothesis_aliases table for an exact text match
        against the event's raw_text (case-insensitive, sharp-s normalized).

        This is the fast path (target: <5ms).

        Args:
            event: The normalized event.
            user_id: User ID for filtering.

        Returns:
            SkillMatch from alias, or None.
        """
        raw_lower = event.raw_text.strip().lower()
        if not raw_lower:
            return None

        # Round-4: Also try sharp-s / double-s normalized form
        raw_normalized = self._normalize_german(event.raw_text)

        # Query aliases matching this text (exact match first)
        rows = self._storage._conn.fetchall(
            "SELECT ha.hypothesis_id, ha.alias_text, ha.confidence "
            "FROM hypothesis_aliases ha "
            "JOIN hypotheses h ON ha.hypothesis_id = h.hypothesis_id "
            "WHERE LOWER(ha.alias_text) = ? AND h.user_id = ? "
            "AND h.status IN ('confirmed', 'active') "
            "ORDER BY ha.confidence DESC LIMIT 5",
            (raw_lower, user_id),
        )

        # Round-4: If no exact match, try sharp-s normalized match.
        # This handles both directions:
        #   - User sends sharp-s form, alias stored as double-s
        #   - User sends double-s form, alias stored as sharp-s (REPLACE normalizes DB value)
        if not rows:
            rows = self._storage._conn.fetchall(
                "SELECT ha.hypothesis_id, ha.alias_text, ha.confidence "
                "FROM hypothesis_aliases ha "
                "JOIN hypotheses h ON ha.hypothesis_id = h.hypothesis_id "
                "WHERE REPLACE(LOWER(ha.alias_text), 'ß', 'ss') = ? "
                "AND h.user_id = ? "
                "AND h.status IN ('confirmed', 'active') "
                "ORDER BY ha.confidence DESC LIMIT 5",
                (raw_normalized, user_id),
            )

        if not rows:
            return None

        # Take the best alias match
        best_row = rows[0]
        hyp_id = best_row["hypothesis_id"]
        alias_confidence = float(best_row["confidence"])

        hypothesis = self._storage.get_hypothesis(hyp_id)
        if hypothesis is None:
            return None

        # Alias match confidence is boosted (direct match is high signal)
        elo_quotient = min(hypothesis.elo_rating / ELO_NORMALIZER, 1.0)
        combined_confidence = min(1.0, alias_confidence * elo_quotient * 1.2)

        if combined_confidence <= MATCH_SCORE_THRESHOLD:
            return None

        return SkillMatch(
            hypothesis=hypothesis,
            confidence=combined_confidence,
            requires_confirmation=hypothesis.status == STATUS_CONFIRMED,
            explanation=(
                f"Direct alias match: '{best_row['alias_text']}' "
                f"(alias_confidence={alias_confidence:.2f}, "
                f"elo_quotient={elo_quotient:.2f})"
            ),
            match_source="alias",
        )

    def _try_fingerprint_match(
        self,
        event: NormalizedEvent,
        user_id: int,
        scope_hint: dict | None,
    ) -> list[SkillMatch]:
        """Try to match via fingerprint similarity.

        Loads all matchable hypotheses for the user, constructs
        synthetic NormalizedEvents from their stored fields, and
        computes fingerprint similarity.

        IC-MATCH-1: score = fingerprint_score * elo_quotient
        where elo_quotient = min(elo / 2000, 1.0)

        Args:
            event: The normalized event.
            user_id: User ID for filtering.
            scope_hint: Optional scope context.

        Returns:
            List of SkillMatch objects above threshold.
        """
        matches: list[SkillMatch] = []

        # Load matchable hypotheses (confirmed + active only)
        hypotheses = self._load_matchable_hypotheses(user_id)

        if not hypotheses:
            return matches

        for hyp in hypotheses:
            # Scope filtering: if scope_hint provided, check compatibility
            if scope_hint and not self._scope_compatible(hyp.scope, scope_hint):
                continue

            # Build a synthetic event from hypothesis for comparison
            synth_event = self._hypothesis_to_event(hyp, event)

            # Compute fingerprint similarity
            fp_match: FingerprintMatch = compute_similarity(
                event, synth_event, threshold=0.0
            )

            # IC-MATCH-1: combined score = fingerprint_score * elo_quotient
            elo_quotient = min(hyp.elo_rating / ELO_NORMALIZER, 1.0)
            combined_score = fp_match.similarity_score * elo_quotient

            if combined_score > MATCH_SCORE_THRESHOLD:
                matches.append(
                    SkillMatch(
                        hypothesis=hyp,
                        confidence=combined_score,
                        requires_confirmation=hyp.status == STATUS_CONFIRMED,
                        explanation=(
                            f"Fingerprint match: similarity={fp_match.similarity_score:.3f}, "
                            f"elo_quotient={elo_quotient:.3f}, "
                            f"combined={combined_score:.3f}"
                        ),
                        match_source="fingerprint",
                    )
                )

        # Sort by confidence descending
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def _load_matchable_hypotheses(self, user_id: int) -> list[Hypothesis]:
        """Load all hypotheses eligible for matching.

        Only confirmed and active hypotheses are matchable.

        Args:
            user_id: User ID.

        Returns:
            List of matchable Hypothesis objects.
        """
        result: list[Hypothesis] = []
        for status in MATCHABLE_STATUSES:
            result.extend(
                self._storage.get_hypotheses_by_user(user_id, status=status, limit=100)
            )
        return result

    def _resolve_collision(
        self,
        matches: list[SkillMatch],
    ) -> Optional[SkillMatch]:
        """Resolve collision when multiple hypotheses match.

        Delegates to CollisionDetector for scope-based resolution.
        If collision cannot be auto-resolved, returns the match
        that requires user decision (with requires_confirmation=True).

        Args:
            matches: List of matching SkillMatch objects.

        Returns:
            Winning SkillMatch, or None if no resolution.
        """
        if not matches:
            return None

        if len(matches) == 1:
            return matches[0]

        # Import here to avoid circular imports
        from application.skill_compression.collision_detector import (
            CollisionDetector,
        )

        detector = CollisionDetector()
        scope_hint: dict = {}
        result = detector.resolve(matches, scope_hint)

        if result.winner is not None:
            # Find the SkillMatch for the winner
            for m in matches:
                if m.hypothesis.hypothesis_id == result.winner.hypothesis_id:
                    return m

        # Tie: return the first match but mark as requiring confirmation
        if result.requires_user_decision and result.candidates:
            first = matches[0]
            return SkillMatch(
                hypothesis=first.hypothesis,
                confidence=first.confidence,
                requires_confirmation=True,
                explanation=(
                    f"Collision detected: {len(result.candidates)} hypotheses "
                    f"with equal scope specificity. User decision required. "
                    f"{result.resolution_reason}"
                ),
                match_source=first.match_source,
            )

        return matches[0] if matches else None

    @staticmethod
    def _scope_compatible(
        hyp_scope: HypothesisScope,
        scope_hint: dict,
    ) -> bool:
        """Check if a hypothesis scope is compatible with the given context.

        A global scope (empty) is always compatible.
        A specific scope must match the context fields.

        Args:
            hyp_scope: Hypothesis scope.
            scope_hint: Current context scope dict.

        Returns:
            True if compatible.
        """
        # Global scope is always compatible
        if not hyp_scope.project and not hyp_scope.client:
            return True

        # Check project match
        if hyp_scope.project:
            hint_project = scope_hint.get("project", "")
            if hint_project and hyp_scope.project != hint_project:
                return False

        # Check client match
        if hyp_scope.client:
            hint_client = scope_hint.get("client", "")
            if hint_client and hyp_scope.client != hint_client:
                return False

        return True

    @staticmethod
    def _hypothesis_to_event(
        hyp: Hypothesis,
        reference_event: NormalizedEvent,
    ) -> NormalizedEvent:
        """Create a synthetic NormalizedEvent from a hypothesis.

        Uses the hypothesis's scope and pattern_hash to construct
        an event that can be compared via fingerprint similarity.

        The reference_event provides defaults for fields not stored
        in the hypothesis (language, timestamp).

        Args:
            hyp: The hypothesis.
            reference_event: Event providing defaults.

        Returns:
            Synthetic NormalizedEvent.
        """
        scope_dict = {
            "project": hyp.scope.project,
            "client": hyp.scope.client,
        }

        return NormalizedEvent(
            event_id=f"synth_{hyp.hypothesis_id}",
            user_id=hyp.user_id,
            timestamp=reference_event.timestamp,
            raw_text=hyp.claim,
            intent=reference_event.intent,
            domain=reference_event.domain,
            format_type=reference_event.format_type,
            constraints=reference_event.constraints,
            scope=scope_dict,
            language=reference_event.language,
            fingerprint_hash=hyp.pattern_hash or "",
        )
