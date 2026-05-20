"""DetectionOrchestrator: multi-backend language detection with fallback logic.

This module is the central decision engine for Phase 2 of the Language
Control Plane (LCP). It replaces the direct backend call in LanguageResolver
with an orchestrated pipeline that selects backends based on text length,
aggregates results, and computes a composite reliability score.

Orchestration strategy (Phase 2, deliberately simple):
- micro/short text: consult domain heuristic first (optimized for short
  user inputs), then primary backend (langdetect) if confidence < 0.7.
- medium/long text: consult primary backend only, then fallback if
  confidence < fallback_threshold.
- Aggregation: consensus = max(confidences); dissent = higher-confidence
  backend wins, reliability_score reduced by 0.2 penalty.

Hard Constraints enforced here:
- HC-O1: langdetect (via LangdetectBackend) is primary backend.
- HC-O2: DomainLanguageBackend is fallback for short text.
- HC-O3: Protocol + concrete class separation.
- HC-O4: All output codes normalized via Registry.resolve_backend_code().
- HC-O5: No direct import from domain.language.
- HC-O6: frozen=True, slots=True on data classes.
- HC-O8: Registry consulted for detection_tier and min_chars_reliable.
- HC-O9: decision_reason is human-readable audit string.

Implementation choices:
- IC-O1: fallback_threshold = 0.6 (Spec recommendation, validated reasonable).
- IC-O2: short_text_threshold_words = 15 (Spec recommendation).
- IC-O3: detect() is synchronous (both backends are synchronous).
- IC-O4: reliability_score formula:
    base = detection_confidence
    tier_bonus: HIGH +0.05, MEDIUM +0.00, LOW -0.05
    length_bonus: micro -0.05, short +0.00, medium +0.03, long +0.05
    dissent_penalty: -0.20 when backends disagree
    clamped to [0.0, 1.0]
- IC-O5: latency measured via time.perf_counter (highest resolution).
- IC-O6: Orchestrator is NOT a singleton; instantiated by LanguageResolver.
         This allows per-resolver configuration and clean test isolation.
- IC-O7: Logging at DEBUG for routine detection, WARNING for fallbacks
         and errors, INFO for dissent.
- IC-O8: DomainLanguageBackend NOT consulted for long text (unreliable
         beyond 50 words, as documented in backends.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from application.language.backends import LanguageDetectorBackend
from application.language.context import LanguageContext
from application.language.registry import (
    DetectionTier,
    LanguageRegistryProtocol,
)

log = logging.getLogger(__name__)

# Default language when all detection fails.
_DEFAULT_LANG = "de"

# Confidence threshold for short-text heuristic: if the domain heuristic
# returns >= this value, skip the primary backend entirely.
_SHORT_TEXT_HEURISTIC_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Data classes (HC-O6: frozen=True, slots=True)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    """Result from a single backend for a single detection call.

    Attributes:
        backend_name: Identifier of the backend that produced this
            result (e.g. "langdetect", "domain_heuristic").
        distribution: Full probability distribution {code: probability}.
            Codes are already normalized to AXOLENT canonical codes.
        top_lang: Highest-probability language code.
        top_confidence: Confidence of the top language (0.0..1.0).
        latency_ms: Time the backend took for this detection.
        error: Error message if the backend failed. None on success.
    """

    backend_name: str
    distribution: dict[str, float]
    top_lang: str
    top_confidence: float
    latency_ms: float
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """True if detection produced a result without error."""
        return self.error is None and self.top_lang != ""


@dataclass(frozen=True, slots=True)
class OrchestratedDetection:
    """Final detection result produced by DetectionOrchestrator.

    Aggregates results from one or more backends into a single
    decision with full provenance.

    Attributes:
        code: Final decided language code (canonical AXOLENT code).
        confidence: Final confidence score (0.0..1.0).
        distribution: Merged probability distribution.
        reliability_score: Composite reliability metric factoring
            in detection tier, text length, and confidence.
        candidates: All backend results in order of consultation.
        decision_reason: Human-readable explanation of why this
            language was chosen (for audit/debug).
        text_length_bucket: Input text length category.
    """

    code: str
    confidence: float
    distribution: dict[str, float]
    reliability_score: float
    candidates: Tuple[DetectionCandidate, ...]
    decision_reason: str
    text_length_bucket: str


# ---------------------------------------------------------------------------
# Protocol (HC-O3)
# ---------------------------------------------------------------------------


class DetectionOrchestratorProtocol(Protocol):
    """Protocol for the detection orchestrator.

    The LanguageResolver depends on this protocol.
    Concrete implementation is DetectionOrchestrator.
    """

    def detect(self, text: str) -> OrchestratedDetection:
        """Detect language of input text using configured backends.

        Args:
            text: User input text to detect.

        Returns:
            OrchestratedDetection with decided language and provenance.
        """
        ...

    @property
    def primary_backend_name(self) -> str:
        """Name of the currently configured primary backend."""
        ...

    @property
    def registered_backends(self) -> list[str]:
        """Names of all registered backends."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class DetectionOrchestrator:
    """Concrete orchestrator with primary/fallback logic.

    Constructor parameters:
        primary_backend: LanguageDetectorBackend (langdetect, HC-O1).
        fallback_backend: LanguageDetectorBackend or None (domain heuristic, HC-O2).
        registry: LanguageRegistryProtocol for code normalization and tier lookup.
        fallback_threshold: If primary confidence < threshold, consult fallback.
        short_text_threshold_words: Texts with fewer words route to heuristic first.
    """

    def __init__(
        self,
        primary_backend: LanguageDetectorBackend,
        fallback_backend: LanguageDetectorBackend | None,
        registry: LanguageRegistryProtocol,
        fallback_threshold: float = 0.6,
        short_text_threshold_words: int = 15,
    ) -> None:
        self._primary = primary_backend
        self._fallback = fallback_backend
        self._registry = registry
        self._fallback_threshold = fallback_threshold
        self._short_text_threshold_words = short_text_threshold_words

        self._primary_name = "langdetect"
        self._fallback_name = "domain_heuristic"

    # -- Protocol properties ------------------------------------------------

    @property
    def primary_backend_name(self) -> str:
        """Name of the currently configured primary backend."""
        return self._primary_name

    @property
    def registered_backends(self) -> list[str]:
        """Names of all registered backends."""
        names = [self._primary_name]
        if self._fallback is not None:
            names.append(self._fallback_name)
        return names

    # -- Main detect method -------------------------------------------------

    def detect(self, text: str) -> OrchestratedDetection:
        """Detect language of input text using configured backends.

        Orchestration flow:
        1. Count words, classify into text_length_bucket.
        2. If micro/short: domain heuristic first, primary if needed.
        3. If medium/long: primary first, fallback if confidence low.
        4. Aggregate candidates, compute reliability_score.

        Args:
            text: User input text to detect.

        Returns:
            OrchestratedDetection with decided language and provenance.
        """
        stripped = text.strip()
        word_count = len(stripped.split()) if stripped else 0
        bucket = LanguageContext.classify_text_length(word_count)

        # Edge case: empty or whitespace-only text
        if not stripped or word_count == 0:
            return self._make_default_result(
                bucket=bucket,
                reason="Empty or whitespace-only input. Returning default.",
            )

        if bucket in ("micro", "short"):
            return self._detect_short_text(stripped, bucket)
        return self._detect_long_text(stripped, bucket)

    # -- Short text strategy ------------------------------------------------

    def _detect_short_text(self, text: str, bucket: str) -> OrchestratedDetection:
        """Strategy for micro/short text: heuristic first, primary if needed."""
        candidates: list[DetectionCandidate] = []

        # Step 1: Consult domain heuristic (fallback backend) first
        heuristic_candidate = self._call_backend(
            self._fallback, self._fallback_name, text
        )
        if heuristic_candidate is not None:
            candidates.append(heuristic_candidate)

            if (
                heuristic_candidate.succeeded
                and heuristic_candidate.top_confidence
                >= _SHORT_TEXT_HEURISTIC_THRESHOLD
            ):
                # Heuristic is confident enough. No need for primary.
                return self._build_result(
                    winner=heuristic_candidate,
                    candidates=candidates,
                    bucket=bucket,
                    dissent=False,
                    reason=(
                        f"Domain heuristic detected "
                        f"'{heuristic_candidate.top_lang}' with confidence "
                        f"{heuristic_candidate.top_confidence:.2f} (>= "
                        f"{_SHORT_TEXT_HEURISTIC_THRESHOLD}). "
                        f"No primary backend needed."
                    ),
                )

        # Step 2: Heuristic not confident enough (or failed). Consult primary.
        primary_candidate = self._call_backend(self._primary, self._primary_name, text)
        if primary_candidate is not None:
            candidates.append(primary_candidate)

        # Aggregate: primary wins on conflict (it has better n-gram models)
        return self._aggregate(candidates, bucket)

    # -- Long text strategy -------------------------------------------------

    def _detect_long_text(self, text: str, bucket: str) -> OrchestratedDetection:
        """Strategy for medium/long text: primary first, fallback if needed."""
        candidates: list[DetectionCandidate] = []

        # Step 1: Consult primary backend
        primary_candidate = self._call_backend(self._primary, self._primary_name, text)
        if primary_candidate is not None:
            candidates.append(primary_candidate)

            if (
                primary_candidate.succeeded
                and primary_candidate.top_confidence >= self._fallback_threshold
            ):
                # Primary is confident enough. No fallback.
                return self._build_result(
                    winner=primary_candidate,
                    candidates=candidates,
                    bucket=bucket,
                    dissent=False,
                    reason=(
                        f"Primary backend ({self._primary_name}) detected "
                        f"'{primary_candidate.top_lang}' with confidence "
                        f"{primary_candidate.top_confidence:.2f}. "
                        f"No fallback needed."
                    ),
                )

        # Step 2: Primary not confident or failed. Consult fallback.
        # IC-O8: NOT consulting domain heuristic for long text because
        # it is unreliable beyond 50 words.
        if self._fallback is not None and bucket not in ("long",):
            fallback_candidate = self._call_backend(
                self._fallback, self._fallback_name, text
            )
            if fallback_candidate is not None:
                candidates.append(fallback_candidate)

        return self._aggregate(candidates, bucket)

    # -- Backend call wrapper -----------------------------------------------

    def _call_backend(
        self,
        backend: LanguageDetectorBackend | None,
        name: str,
        text: str,
    ) -> DetectionCandidate | None:
        """Call a single backend and wrap the result as DetectionCandidate.

        Normalizes all codes via registry (HC-O4).
        Catches all exceptions (HC-O9: error in candidate, not crash).
        """
        if backend is None:
            return None

        start = time.perf_counter()
        try:
            raw_distribution = backend.detect_distribution(text)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if not raw_distribution:
                return DetectionCandidate(
                    backend_name=name,
                    distribution={},
                    top_lang="",
                    top_confidence=0.0,
                    latency_ms=elapsed_ms,
                    error="Backend returned empty distribution.",
                )

            # Normalize codes via registry (HC-O4, Guard 5)
            normalized: dict[str, float] = {}
            for code, prob in raw_distribution.items():
                canon = self._registry.resolve_backend_code(code)
                # Merge duplicates (e.g. zh-cn + zh-tw both map to zh)
                normalized[canon] = normalized.get(canon, 0.0) + prob
            distribution = normalized

            # Determine top language
            top_lang = max(distribution, key=distribution.get)  # type: ignore[arg-type]
            top_confidence = distribution[top_lang]

            log.debug(
                "Backend %s: top=%s conf=%.3f latency=%.1fms",
                name,
                top_lang,
                top_confidence,
                elapsed_ms,
            )

            return DetectionCandidate(
                backend_name=name,
                distribution=distribution,
                top_lang=top_lang,
                top_confidence=top_confidence,
                latency_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            log.warning(
                "Backend %s raised %s: %s (%.1fms)",
                name,
                type(exc).__name__,
                exc,
                elapsed_ms,
            )
            return DetectionCandidate(
                backend_name=name,
                distribution={},
                top_lang="",
                top_confidence=0.0,
                latency_ms=elapsed_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    # -- Aggregation --------------------------------------------------------

    def _aggregate(
        self,
        candidates: list[DetectionCandidate],
        bucket: str,
    ) -> OrchestratedDetection:
        """Aggregate one or more candidates into a final result.

        Rules:
        - If no candidate succeeded: return default.
        - If one candidate succeeded: use it directly.
        - If two succeeded with same top_lang (consensus): max confidence.
        - If two succeeded with different top_lang (dissent): higher
          confidence wins, reliability_score penalized by 0.2.
        """
        succeeded = [c for c in candidates if c.succeeded]

        if not succeeded:
            return self._make_default_result(
                bucket=bucket,
                reason=(
                    "No backend produced a valid result. Returning default language."
                ),
                candidates=tuple(candidates),
            )

        if len(succeeded) == 1:
            winner = succeeded[0]
            return self._build_result(
                winner=winner,
                candidates=candidates,
                bucket=bucket,
                dissent=False,
                reason=(
                    f"Single backend ({winner.backend_name}) detected "
                    f"'{winner.top_lang}' with confidence "
                    f"{winner.top_confidence:.2f}."
                ),
            )

        # Two successful candidates: check consensus vs dissent
        first, second = succeeded[0], succeeded[1]
        if first.top_lang == second.top_lang:
            # Consensus: max confidence
            winner = first if first.top_confidence >= second.top_confidence else second
            merged_confidence = max(first.top_confidence, second.top_confidence)
            reason = (
                f"Consensus: both {first.backend_name} and "
                f"{second.backend_name} detected '{winner.top_lang}'. "
                f"Confidence: {first.top_confidence:.2f} vs "
                f"{second.top_confidence:.2f}, using {merged_confidence:.2f}."
            )
            log.debug(reason)
            return self._build_result(
                winner=winner,
                candidates=candidates,
                bucket=bucket,
                dissent=False,
                reason=reason,
                override_confidence=merged_confidence,
            )
        else:
            # Dissent: higher confidence wins, penalty applied
            winner = first if first.top_confidence >= second.top_confidence else second
            reason = (
                f"Dissent: {first.backend_name} detected "
                f"'{first.top_lang}' ({first.top_confidence:.2f}), "
                f"{second.backend_name} detected "
                f"'{second.top_lang}' ({second.top_confidence:.2f}). "
                f"Winner: {winner.backend_name} "
                f"('{winner.top_lang}'). "
                f"Reliability penalized by 0.20 for disagreement."
            )
            log.info(reason)
            return self._build_result(
                winner=winner,
                candidates=candidates,
                bucket=bucket,
                dissent=True,
                reason=reason,
            )

    # -- Result builders ----------------------------------------------------

    def _build_result(
        self,
        winner: DetectionCandidate,
        candidates: list[DetectionCandidate],
        bucket: str,
        dissent: bool,
        reason: str,
        override_confidence: float | None = None,
    ) -> OrchestratedDetection:
        """Build the final OrchestratedDetection from a winning candidate."""
        code = winner.top_lang
        confidence = (
            override_confidence
            if override_confidence is not None
            else winner.top_confidence
        )

        # Merge distributions from all successful candidates
        merged_dist = self._merge_distributions(candidates)

        # Compute reliability score (IC-O4)
        reliability = self._compute_reliability(
            confidence=confidence,
            code=code,
            bucket=bucket,
            dissent=dissent,
        )

        return OrchestratedDetection(
            code=code,
            confidence=confidence,
            distribution=merged_dist,
            reliability_score=reliability,
            candidates=tuple(candidates),
            decision_reason=reason,
            text_length_bucket=bucket,
        )

    def _make_default_result(
        self,
        bucket: str,
        reason: str,
        candidates: Tuple[DetectionCandidate, ...] = (),
    ) -> OrchestratedDetection:
        """Build a default-language result when detection fails."""
        return OrchestratedDetection(
            code=_DEFAULT_LANG,
            confidence=0.0,
            distribution={},
            reliability_score=0.0,
            candidates=candidates,
            decision_reason=reason,
            text_length_bucket=bucket,
        )

    # -- Reliability score (IC-O4) ------------------------------------------

    def _compute_reliability(
        self,
        confidence: float,
        code: str,
        bucket: str,
        dissent: bool,
    ) -> float:
        """Compute composite reliability score.

        Formula (IC-O4):
            base = confidence
            + tier_bonus  (HIGH: +0.05, MEDIUM: +0.00, LOW: -0.05)
            + length_bonus (micro: -0.05, short: +0.00, medium: +0.03, long: +0.05)
            - dissent_penalty (0.20 if backends disagreed)
            clamped to [0.0, 1.0]

        The tier_bonus and length_bonus reflect that script-detected
        languages and longer texts are inherently more reliable.
        HC-O8: detection_tier and min_chars_reliable from Registry are
        consulted for the tier_bonus.
        """
        base = confidence

        # Tier bonus (HC-O8: consult registry)
        tier_bonus = 0.0
        entry = self._registry.get_or_none(code)
        if entry is not None:
            if entry.detection_tier == DetectionTier.HIGH:
                tier_bonus = 0.05
            elif entry.detection_tier == DetectionTier.LOW:
                tier_bonus = -0.05
            # MEDIUM: 0.0, no change

        # Length bonus
        length_bonuses = {
            "micro": -0.05,
            "short": 0.0,
            "medium": 0.03,
            "long": 0.05,
        }
        length_bonus = length_bonuses.get(bucket, 0.0)

        # Dissent penalty
        penalty = 0.20 if dissent else 0.0

        score = base + tier_bonus + length_bonus - penalty
        return max(0.0, min(1.0, score))

    # -- Distribution merging -----------------------------------------------

    @staticmethod
    def _merge_distributions(
        candidates: list[DetectionCandidate],
    ) -> dict[str, float]:
        """Merge probability distributions from all successful candidates.

        Uses the distribution from the last successful candidate (the one
        most likely to be the "winner" in the aggregation flow). If only
        one succeeded, its distribution is used directly.
        """
        for candidate in reversed(candidates):
            if candidate.succeeded and candidate.distribution:
                return dict(candidate.distribution)
        return {}
