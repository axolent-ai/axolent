"""Layer 5: SkillMatcher for Skill-Compression.

Matches incoming user requests against the hypothesis pool AND
SkillContracts to find applicable learned skills. This is the bridge
between pattern storage and skill application.

Match strategy (from Spec, Layer 5):
  1. Direct alias match in hypothesis_aliases (fast, <5ms target)
  2. Contract-aware exact-phrase match against ContractStore
  3. Fingerprint similarity against active hypotheses (via FingerprintMatcher)
  4. Only hypotheses with status 'confirmed' or 'active' are matched
  5. 'candidate' and 'suggested' are NEVER applied
  6. Score threshold: > 0.7

Dedup rule (Codex R2): when a Hypothesis has been migrated and exists
as a SkillContract, the Contract takes precedence. The Matcher MUST NOT
return both. Dedup key: hypothesis_id on Contract links back to the
original Hypothesis.

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
from typing import TYPE_CHECKING, Optional

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

if TYPE_CHECKING:
    from application.skill_compression.contract_store import ContractStore
    from application.skill_compression.skill_contract import SkillContract

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
    """Result of matching a user request against a hypothesis or contract.

    Attributes:
        hypothesis: The matched hypothesis.
        confidence: Combined match confidence [0.0, 1.0].
        requires_confirmation: True if user must confirm before applying.
        explanation: Human-readable reason for this match.
        match_source: How the match was found ('alias', 'fingerprint', 'contract').
        contract: The matched SkillContract (None for legacy-only matches).
            When set, PermissionGate enforcement is active in chat_service.
    """

    hypothesis: Hypothesis
    confidence: float
    requires_confirmation: bool
    explanation: str
    match_source: str = "fingerprint"
    contract: Optional["SkillContract"] = None


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
      - Status 'confirmed': ALWAYS ask (requires first-time user confirmation)
      - Status 'active': NEVER ask (user has already confirmed at least once;
        Round-5 fix: after first "yes" confirmation the hypothesis is promoted
        to 'active' and auto-applies without further prompts)
      - Default: ask

    Round-5 change (2026-05-27): Active skills auto-apply unconditionally.
    Previously, active skills still checked auto_apply_enabled preference
    (defaulting to False, effectively always asking). The user expectation
    is clear: after clicking "Ja" once, the skill should apply directly
    on subsequent triggers without asking again.

    Args:
        match: The SkillMatch to evaluate.
        user_preferences: User settings dict (reserved for future use).

    Returns:
        True if user confirmation is required before applying.
    """
    status = match.hypothesis.status

    # Confirmed: always ask (user has confirmed the skill exists,
    # but hasn't explicitly approved application yet)
    if status == STATUS_CONFIRMED:
        return True

    # Active: never ask (user already confirmed at least once)
    # Round-5: active means "user approved", auto-apply unconditionally
    if status == STATUS_ACTIVE:
        return False

    # Any other status should not reach here (not matchable),
    # but defensively: always ask
    return True


# ---------------------------------------------------------------
# SkillMatcher
# ---------------------------------------------------------------


class SkillMatcher:
    """Layer 5: Matches user requests against learned skill hypotheses AND contracts.

    Uses a three-stage matching strategy:
      1. Fast contract match (exact-phrase against ContractStore activation phrases)
      2. Fast alias lookup (exact text match in hypothesis_aliases, deduped)
      3. Fingerprint similarity for fuzzy matching (deduped)

    Dedup rule: when a Hypothesis has been migrated to a Contract, the
    Contract match wins. The legacy Hypothesis is suppressed to prevent
    double-triggering. Dedup key: contract.hypothesis_id == hypothesis.hypothesis_id.

    Only hypotheses with status 'confirmed' or 'active' are considered.
    The matcher does NOT apply skills; it returns SkillMatch objects
    that the orchestrator uses for the application decision.

    Thread safety: NOT thread-safe. Designed for single-threaded
    async event loop (Telegram bot context).

    Usage:
        matcher = SkillMatcher(storage, judge, contract_store=store)
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
        *,
        contract_store: Optional["ContractStore"] = None,
    ) -> None:
        """Initialize the SkillMatcher.

        Args:
            storage: Hypothesis storage for DB access.
            pattern_judge: Pattern Judge for collision detection.
            contract_store: Optional ContractStore for contract-aware matching.
        """
        self._storage = storage
        self._judge = pattern_judge
        self._contract_store = contract_store

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
        """Match a user event against contracts and hypothesis pool.

        Strategy:
          1. Try contract exact-phrase match (highest priority)
          2. Try direct alias match (fast path, deduped against contracts)
          3. Fall back to fingerprint similarity (deduped against contracts)
          4. If multiple matches: delegate to CollisionDetector
          5. Score threshold > 0.7

        Dedup: hypothesis matches whose hypothesis_id is already covered
        by a contract match are suppressed (Contract > Legacy).

        Args:
            event: Normalized event from the user request.
            scope_hint: Optional scope context for filtering.

        Returns:
            SkillMatch if a match is found, None otherwise.
        """
        user_id = event.user_id

        # Stage 0: Collect hypothesis_ids covered by contracts (for dedup)
        contract_covered_hyp_ids: set[str] = set()

        # Stage 1: Contract exact-phrase match (highest priority)
        contract_match = self._try_contract_match(event, user_id)
        if contract_match is not None:
            log.info(
                "Contract match found: contract=%s confidence=%.3f",
                contract_match.contract.id if contract_match.contract else "?",
                contract_match.confidence,
            )
            # Collect the hypothesis_id this contract covers (for dedup)
            if (
                contract_match.contract is not None
                and contract_match.contract.hypothesis_id
            ):
                contract_covered_hyp_ids.add(contract_match.contract.hypothesis_id)
            return contract_match

        # Build dedup set: all hypothesis_ids that have a contract equivalent
        contract_covered_hyp_ids = self._get_contract_covered_hypothesis_ids(user_id)

        # Stage 2: Direct alias match (deduped)
        alias_match = self._try_alias_match(event, user_id)
        if alias_match is not None:
            # Dedup: if this hypothesis is covered by a contract, suppress
            if alias_match.hypothesis.hypothesis_id in contract_covered_hyp_ids:
                log.debug(
                    "Alias match suppressed (contract exists): hyp=%s",
                    alias_match.hypothesis.hypothesis_id,
                )
            else:
                log.info(
                    "Alias match found: hyp=%s confidence=%.3f",
                    alias_match.hypothesis.hypothesis_id,
                    alias_match.confidence,
                )
                return alias_match

        # Stage 3: Fingerprint similarity match (deduped)
        fingerprint_matches = self._try_fingerprint_match(event, user_id, scope_hint)

        # Filter out hypothesis matches covered by contracts
        if contract_covered_hyp_ids:
            fingerprint_matches = [
                m
                for m in fingerprint_matches
                if m.hypothesis.hypothesis_id not in contract_covered_hyp_ids
            ]

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
        """Find all matching contracts and hypotheses (for UI / debugging).

        Unlike match(), does not resolve collisions. Returns all
        candidates above threshold, sorted by confidence.

        Contract-aware: includes contract matches and deduplicates legacy
        matches whose hypothesis_id is covered by a confirmed/active contract.

        Args:
            event: Normalized event from the user request.
            scope_hint: Optional scope context for filtering.

        Returns:
            List of SkillMatch objects, sorted by confidence descending.
        """
        user_id = event.user_id
        all_matches: list[SkillMatch] = []
        seen_ids: set[str] = set()

        # Stage 1: Contract matches (highest priority)
        contract_matches = self._collect_all_contract_matches(event, user_id)
        for cm in contract_matches:
            hyp_id = cm.hypothesis.hypothesis_id
            if hyp_id not in seen_ids:
                all_matches.append(cm)
                seen_ids.add(hyp_id)

        # Build dedup set: hypothesis_ids covered by confirmed/active contracts
        contract_covered_hyp_ids = self._get_contract_covered_hypothesis_ids(user_id)

        # Stage 2: Alias matches (deduped)
        alias_match = self._try_alias_match(event, user_id)
        if alias_match is not None:
            hyp_id = alias_match.hypothesis.hypothesis_id
            if hyp_id not in seen_ids and hyp_id not in contract_covered_hyp_ids:
                all_matches.append(alias_match)
                seen_ids.add(hyp_id)

        # Stage 3: Fingerprint matches (deduped)
        fp_matches = self._try_fingerprint_match(event, user_id, scope_hint)
        for m in fp_matches:
            hyp_id = m.hypothesis.hypothesis_id
            if hyp_id not in seen_ids and hyp_id not in contract_covered_hyp_ids:
                all_matches.append(m)
                seen_ids.add(hyp_id)

        # Sort by confidence descending
        all_matches.sort(key=lambda m: m.confidence, reverse=True)
        return all_matches

    # ── Contract matching methods ────────────────────────────

    def _collect_all_contract_matches(
        self,
        event: NormalizedEvent,
        user_id: int,
    ) -> list[SkillMatch]:
        """Collect all contract matches for an event (for match_all).

        Unlike _try_contract_match which returns only the first hit,
        this collects ALL matching contracts for comprehensive display.

        Args:
            event: The normalized event.
            user_id: User ID for filtering.

        Returns:
            List of SkillMatch objects from contracts.
        """
        if self._contract_store is None:
            return []

        raw_lower = event.raw_text.strip().lower()
        if not raw_lower:
            return []

        raw_normalized = self._normalize_german(raw_lower)
        contracts = self._load_matchable_contracts(user_id)
        if not contracts:
            return []

        matches: list[SkillMatch] = []
        for contract in contracts:
            if contract.activation.mode != "exact_phrase":
                continue
            for phrase in contract.activation.phrases:
                phrase_lower = phrase.strip().lower()
                phrase_normalized = self._normalize_german(phrase_lower)
                if raw_lower == phrase_lower or raw_normalized == phrase_normalized:
                    synth_hyp = self._contract_to_hypothesis(contract, user_id)
                    matches.append(
                        SkillMatch(
                            hypothesis=synth_hyp,
                            confidence=1.0,
                            requires_confirmation=(
                                contract.lifecycle.status == "confirmed"
                            ),
                            explanation=(
                                f"Contract exact-phrase match: '{phrase}' "
                                f"(contract_id={contract.id})"
                            ),
                            match_source="contract",
                            contract=contract,
                        )
                    )
                    break  # One match per contract is enough
        return matches

    def _try_contract_match(
        self,
        event: NormalizedEvent,
        user_id: int,
    ) -> Optional[SkillMatch]:
        """Try to match via contract activation phrases.

        Loads matchable contracts (confirmed + active lifecycle status)
        and checks if the event text matches any activation phrase
        (exact_phrase mode, case-insensitive, German ss/sharp-s equivalence).

        Args:
            event: The normalized event.
            user_id: User ID for filtering.

        Returns:
            SkillMatch from contract, or None.
        """
        if self._contract_store is None:
            return None

        raw_lower = event.raw_text.strip().lower()
        if not raw_lower:
            return None

        raw_normalized = self._normalize_german(raw_lower)

        # Load all matchable contracts for this user
        contracts = self._load_matchable_contracts(user_id)
        if not contracts:
            return None

        for contract in contracts:
            # Only exact_phrase mode supported for now
            if contract.activation.mode != "exact_phrase":
                continue

            for phrase in contract.activation.phrases:
                phrase_lower = phrase.strip().lower()
                phrase_normalized = self._normalize_german(phrase_lower)

                # Check exact match (case-insensitive) or German normalized
                if raw_lower == phrase_lower or raw_normalized == phrase_normalized:
                    # Build a synthetic Hypothesis for backward compatibility
                    # with chat_service which reads match_result.hypothesis.claim
                    synth_hyp = self._contract_to_hypothesis(contract, user_id)

                    return SkillMatch(
                        hypothesis=synth_hyp,
                        confidence=1.0,  # Exact contract match = highest confidence
                        requires_confirmation=(
                            contract.lifecycle.status == "confirmed"
                        ),
                        explanation=(
                            f"Contract exact-phrase match: '{phrase}' "
                            f"(contract_id={contract.id})"
                        ),
                        match_source="contract",
                        contract=contract,
                    )

        return None

    def _load_matchable_contracts(self, user_id: int) -> list:
        """Load all contracts eligible for matching.

        Only confirmed and active lifecycle statuses are matchable.

        Args:
            user_id: User ID.

        Returns:
            List of matchable SkillContract objects.
        """
        if self._contract_store is None:
            return []

        result = []
        for status in ("confirmed", "active"):
            result.extend(self._contract_store.get_by_user(user_id, status=status))
        return result

    def _get_contract_covered_hypothesis_ids(self, user_id: int) -> set[str]:
        """Get hypothesis_ids covered by confirmed/active contracts for this user.

        Used for dedup: if a Hypothesis has been migrated to a confirmed or
        active Contract, the Contract match wins and the legacy Hypothesis
        is suppressed.

        Lifecycle-aware (Option 1): only confirmed and active contracts suppress
        legacy hypotheses. needs_review and flagged contracts do NOT suppress,
        so the old legacy skill continues to trigger until manual review.

        Args:
            user_id: User ID.

        Returns:
            Set of hypothesis_ids that have a confirmed/active contract.
        """
        if self._contract_store is None:
            return set()

        # Only confirmed + active contracts suppress legacy matches
        contracts: list = []
        for status in ("confirmed", "active"):
            contracts.extend(self._contract_store.get_by_user(user_id, status=status))

        return {
            c.hypothesis_id
            for c in contracts
            if c.hypothesis_id is not None and c.hypothesis_id != ""
        }

    @staticmethod
    def _contract_to_hypothesis(contract, user_id: int) -> Hypothesis:
        """Create a synthetic Hypothesis from a SkillContract.

        For backward compatibility: chat_service reads
        match_result.hypothesis.claim for the skill instruction block.
        The synthetic hypothesis carries the contract's instruction as claim
        and the contract's first activation phrase for alias matching.

        Args:
            contract: The SkillContract.
            user_id: User ID.

        Returns:
            A synthetic Hypothesis.
        """
        trigger = ""
        if contract.activation.phrases:
            trigger = contract.activation.phrases[0]
        instruction = contract.execution.instruction

        if trigger:
            claim = f"when I say {trigger}, {instruction}"
        else:
            claim = instruction

        status = contract.lifecycle.status
        # Map contract lifecycle to hypothesis status
        if status not in ("confirmed", "active"):
            status = "confirmed"

        return Hypothesis(
            hypothesis_id=contract.hypothesis_id or contract.id,
            user_id=user_id,
            type="preference",
            scope=HypothesisScope(),
            claim=claim,
            status=status,
            version=contract.contract_version,
            elo_rating=2000.0,  # Contracts get max Elo (trusted)
            elo_games_played=0,
            bayes_confidence=1.0,
            support_count=1,
            contradict_count=0,
            source_type="learn_command",
            decay_immune=True,
            created_at=contract.created_at,
            last_applied=None,
            last_seen=contract.updated_at,
        )

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
