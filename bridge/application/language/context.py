"""LanguageContext: immutable language resolution result.

Canonical location for the frozen dataclass that represents
THE language decision for an entire request lifecycle.

Re-exported from application.language_resolver for backward compatibility.

Phase 2 additions (Step 2/4):
- detection_distribution: full probability distribution from detection
- reliability_score: composite reliability metric (0.0..1.0)
- confidence_history: ordered backend consultation trail
- detection_tier: tier of the detected language from Registry
- text_length_bucket: input text length category for reliability analysis
- backends_consulted: names of backends actually invoked

Implementation choices (IC-C* from Spec):
- IC-C1: reliability_score as own field (not property). Computed by
  DetectionOrchestrator, not derivable from context fields alone.
- IC-C2: text_length_bucket computation as @staticmethod on LanguageContext.
  Bucket boundaries are hardcoded (HC-C5), tightly coupled to context semantics.
- IC-C4: backends_consulted as FrozenSet[str]. Set semantics correct (order
  irrelevant, no duplicates), natively hashable and immutable.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Optional, Tuple

# Valid text length bucket values (HC-C5).
TEXT_LENGTH_BUCKETS: frozenset[str] = frozenset({"micro", "short", "medium", "long"})


@dataclass(frozen=True, slots=True)
class LanguageContext:
    """Immutable language resolution result with detection metadata.

    Once created, this object represents THE language decision
    for an entire request lifecycle. All consumers (ChatService,
    DebateOrchestrator, StatusSession, PromptComposer) must use
    this context rather than resolving language independently.

    Phase 1 fields (unchanged):
        code: ISO-639-1 language code, guaranteed non-empty.
        source: How the language was determined.
        confidence: Detection confidence (0.0..1.0). For sticky/override: 1.0.
        switched_from: Previous sticky value if a smart-switch occurred.
        request_id: Unique ID for audit correlation.

    Phase 2 additions (all with defaults for backward compatibility):
        detection_distribution: Full probability distribution from the
            detection backend. Keys are canonical AXOLENT codes (HC-C3),
            values are probabilities. Empty dict if source is
            override/sticky/default.
        reliability_score: Composite reliability metric (0.0..1.0)
            combining detection confidence, text length signal, and
            detection tier. Used by DetectionOrchestrator for fallback
            decisions. (IC-C1: own field, computed by Orchestrator.)
        confidence_history: Ordered tuple of (backend_name, confidence)
            pairs showing which backends were consulted and what they
            reported. Enables post-hoc analysis of detection quality.
            (HC-C4: Tuple of Tuples for immutability.)
        detection_tier: Tier string of the detected language from Registry
            (e.g. "high", "medium", "low"). None if source is override
            or language not in registry.
        text_length_bucket: Categorization of input text length:
            "micro" (0-10 words), "short" (11-30), "medium" (31-100),
            "long" (101+). None if not computed. (HC-C5: hardcoded.)
        backends_consulted: Names of backends that were actually invoked
            during detection. (IC-C4: FrozenSet for set semantics.)
    """

    # -- Phase 1 fields (unchanged) ----------------------------------------
    code: str
    source: Literal["override", "sticky", "detected", "default"]
    confidence: float
    switched_from: Optional[str]
    request_id: str

    # -- Phase 2 additions (all with defaults, HC-C2) ----------------------
    # Claude Issue 2: detection_distribution wrapped in MappingProxyType
    # for true immutability (frozen dataclass + read-only dict).
    detection_distribution: types.MappingProxyType[str, float] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    reliability_score: float = 0.0
    confidence_history: Tuple[Tuple[str, float], ...] = ()
    detection_tier: Optional[str] = None
    text_length_bucket: Optional[str] = None
    backends_consulted: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        """Ensure detection_distribution is always a MappingProxyType.

        Callers may pass a plain dict for convenience. This converts it
        to a read-only MappingProxyType to enforce full immutability.
        """
        dd = self.detection_distribution
        if isinstance(dd, dict) and not isinstance(dd, types.MappingProxyType):
            object.__setattr__(
                self, "detection_distribution", types.MappingProxyType(dd)
            )

    def effective_lang(self) -> str:
        """Return the effective language code.

        Convenience method for contexts that only need the code.
        """
        return self.code

    @property
    def was_smart_switched(self) -> bool:
        """True if a smart-switch occurred (user implicitly changed language)."""
        return self.switched_from is not None

    @property
    def has_detection_metadata(self) -> bool:
        """True if Phase 2 detection metadata is populated.

        Returns True when detection_distribution contains at least one
        entry, indicating that a detection backend provided probability
        data. Phase-1-style contexts (constructed without distribution)
        return False.
        """
        return len(self.detection_distribution) > 0

    @property
    def top_alternative(self) -> Optional[str]:
        """Second-most-likely language from distribution, if available.

        Useful for monitoring near-miss detections (e.g. DE vs NL).

        Returns:
            The language code with the second-highest probability,
            or None if the distribution has fewer than 2 entries.
            When two languages share the exact same probability,
            the result is deterministic (sorted by code as tiebreaker).
        """
        if len(self.detection_distribution) < 2:
            return None
        # Sort by probability descending, then by code ascending for
        # deterministic tiebreaking when probabilities are equal.
        sorted_langs = sorted(
            self.detection_distribution.items(),
            key=lambda x: (-x[1], x[0]),
        )
        return sorted_langs[1][0]

    def with_request_id(self, new_request_id: str) -> "LanguageContext":
        """Return a copy with a different request_id, preserving all fields.

        Used by the Execution Kernel to synchronize request_ids without
        losing Phase 2 detection metadata (detection_distribution,
        reliability_score, confidence_history, detection_tier,
        text_length_bucket, backends_consulted).

        Args:
            new_request_id: The new request_id to assign.

        Returns:
            A new LanguageContext identical to this one except for request_id.
        """
        return LanguageContext(
            code=self.code,
            source=self.source,
            confidence=self.confidence,
            switched_from=self.switched_from,
            request_id=new_request_id,
            detection_distribution=self.detection_distribution,
            reliability_score=self.reliability_score,
            confidence_history=self.confidence_history,
            detection_tier=self.detection_tier,
            text_length_bucket=self.text_length_bucket,
            backends_consulted=self.backends_consulted,
        )

    @staticmethod
    def classify_text_length(word_count: int) -> str:
        """Classify a word count into a text length bucket (HC-C5).

        Bucket boundaries are hardcoded and not configurable:
        - "micro": 0 to 10 words
        - "short": 11 to 30 words
        - "medium": 31 to 100 words
        - "long": 101+ words

        Args:
            word_count: Number of words in the input text.
                Negative values are treated as 0.

        Returns:
            One of "micro", "short", "medium", "long".
        """
        if word_count <= 10:
            return "micro"
        if word_count <= 30:
            return "short"
        if word_count <= 100:
            return "medium"
        return "long"
