"""Healthcare Filter: blocks behavioral-clinical phenotyping (HC-SC-14).

Hard Red Line: the system must NEVER store pattern inferences from
writing patterns onto health conditions. This filter prevents:

  - Behavioral-clinical phenotyping (cognitive changes from writing)
  - Inference on chronic illness from language changes
  - Depression detection from emotional patterns
  - Mental health inferences of any kind
  - Mood inference from behavioral signals

Multi-layer heuristic check:
  1. Domain Layer: event.domain in blocked health domains
  2. Keyword Layer: 50+ healthcare keywords (DE + EN) in claim text
  3. Behavioral Layer: patterns tracking writing-style changes over time
  4. Mood-Inference Layer: patterns inferring emotional state

HC-SC-14 [BLOCKER]: Healthcare-Filter MUST prevent behavioral-clinical
  phenotyping. Hard-enforced in code.

AG-SC-6 [GUARD]: test_no_phenotyping_inferences verifies this filter
  blocks all health-related pattern materialization.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from application.security.input_normalizer import (
    normalize_aggressive,
    normalize_for_security_check,
)
from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.hypothesis_storage import Hypothesis

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Blocked health domains (Layer 1: Domain check)
# ---------------------------------------------------------------

BLOCKED_HEALTH_DOMAINS: frozenset[str] = frozenset(
    {
        "health",
        "medical",
        "psychology",
        "mental_health",
        "psychiatry",
        "therapy",
        "wellness",
        "fitness_health",
        "nutrition_health",
        "pharmacology",
        "neurology",
        "clinical",
        "diagnosis",
        "pathology",
    }
)

# ---------------------------------------------------------------
# Healthcare keywords (Layer 2: Keyword scan, DE + EN, 50+)
# ---------------------------------------------------------------

# Each entry is lowercased for case-insensitive matching.
# Organized by category for maintainability.

_MENTAL_HEALTH_KEYWORDS: frozenset[str] = frozenset(
    {
        # English
        "depression",
        "depressive",
        "depressed",
        "anxiety",
        "anxious",
        "ptsd",
        "bipolar",
        "schizophrenia",
        "psychosis",
        "psychotic",
        "mania",
        "manic",
        "ocd",
        "adhd",
        "autism",
        "autistic",
        "suicidal",
        "suicide",
        "self-harm",
        "eating disorder",
        "anorexia",
        "bulimia",
        "burnout",
        "trauma",
        "dissociation",
        "panic attack",
        "phobia",
        "insomnia",
        "mental illness",
        "mental health",
        "psychiatric",
        "antidepressant",
        "psychotherapy",
        # German  # noqa: ERA001
        "depressiv",
        "angststörung",
        "panikattacke",
        "zwangsstörung",
        "essstörung",
        "magersucht",
        "bulimie",
        "psychose",
        "schizophrenie",
        "suizid",
        "selbstverletzung",
        "schlafstörung",
        "psychisch",
        "psychiatrisch",
        "antidepressivum",
        "psychotherapie",
        "traumatisiert",
        "dissoziativ",
    }
)

_CLINICAL_KEYWORDS: frozenset[str] = frozenset(
    {
        # English
        "diagnosis",
        "symptom",
        "syndrome",
        "disorder",
        "condition",
        "chronic",
        "acute",
        "pathology",
        "clinical",
        "prognosis",
        "medication",
        "prescription",
        "therapy",
        "treatment",
        "rehabilitation",
        "patient",
        "cognitive decline",
        "cognitive impairment",
        "dementia",
        "alzheimer",
        "parkinson",
        # German  # noqa: ERA001
        "diagnose",
        "krankheit",
        "chronisch",
        "akut",
        "pathologie",
        "klinisch",
        "medikament",
        "rezept",
        "therapie",
        "behandlung",
        "rehabilitation",
        "kognitiver abbau",
        "kognitive beeinträchtigung",
        "demenz",
    }
)

_BEHAVIORAL_PHENOTYPING_KEYWORDS: frozenset[str] = frozenset(
    {
        # Patterns that indicate writing-style-to-health inference
        "writing pattern",
        "typing speed",
        "keystroke",
        "linguistic marker",
        "language change",
        "cognitive marker",
        "behavioral marker",
        "behavioral indicator",
        "behavioral signal",
        "mood indicator",
        "emotional marker",
        "sentiment shift",
        "affective state",
        "arousal level",
        "valence shift",
        "stress indicator",
        "fatigue indicator",
        "cognitive load",
        # German  # noqa: ERA001
        "schreibmuster",
        "tippgeschwindigkeit",
        "sprachveränderung",
        "kognitiver marker",
        "verhaltensmarker",
        "verhaltensindikator",
        "stimmungsindikator",
        "emotionaler marker",
        "stressindikator",
        "ermüdungsindikator",
        "kognitive last",
    }
)

ALL_HEALTHCARE_KEYWORDS: frozenset[str] = (
    _MENTAL_HEALTH_KEYWORDS | _CLINICAL_KEYWORDS | _BEHAVIORAL_PHENOTYPING_KEYWORDS
)

# Pre-compiled regex for efficient scanning.
# Matches any keyword as a whole word (word boundaries).
_HEALTHCARE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:"
    + "|".join(re.escape(kw) for kw in sorted(ALL_HEALTHCARE_KEYWORDS))
    + r")\b",
    re.IGNORECASE,
)

# Aggressive-normalized pattern variants (Phase 1.5 Polish-Polish):
# For Pass 2 (aggressive input), patterns must ALSO be aggressive-normalized.
# Otherwise German keywords like "angststoerung" (aggressive form of
# "angststörung") won't match because the pattern still expects the umlaut.
# Built at module load: each keyword is normalize_aggressive'd, then
# deduplicated and compiled into a single alternation pattern.
_HEALTHCARE_KEYWORDS_AGGRESSIVE: frozenset[str] = frozenset(
    normalize_aggressive(kw) for kw in ALL_HEALTHCARE_KEYWORDS
)

_HEALTHCARE_PATTERN_AGGRESSIVE: re.Pattern[str] = re.compile(
    r"\b(?:"
    + "|".join(re.escape(kw) for kw in sorted(_HEALTHCARE_KEYWORDS_AGGRESSIVE) if kw)
    + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------
# Behavioral-change patterns (Layer 3)
# ---------------------------------------------------------------

# Patterns that indicate tracking writing-style changes over time.
# These are structural patterns in the claim text, not keywords.
_BEHAVIORAL_CHANGE_PATTERNS: list[re.Pattern[str]] = [
    # "User's writing style has changed"
    re.compile(
        r"(?:writing|typing|language|tone|style|schreib|sprach|stil)"
        r".*(?:change|shift|decline|verschlechter|veränder|gewandelt)",
        re.IGNORECASE,
    ),
    # "Over time / over weeks / in recent days"
    re.compile(
        r"(?:over time|over (?:the )?(?:past |last )?(?:weeks?|months?|days?))"
        r"|(?:über (?:die )?(?:letzten )?(?:Wochen|Monate|Tage))"
        r"|(?:im Laufe der Zeit|in letzter Zeit|zunehmend)",
        re.IGNORECASE,
    ),
    # "User shows signs of / exhibits / demonstrates"
    re.compile(
        r"(?:shows? signs|exhibits?|demonstrates?|indicates?|suggests?)"
        r".*(?:of |that )?(?:cognitive|mental|emotional|psychological)",
        re.IGNORECASE,
    ),
    # "Pattern correlates with / is associated with"
    re.compile(
        r"(?:correlat|associat|linked|verbunden|zusammenhang|korreliert)"
        r".*(?:health|mental|cognitive|emotional|illness|disorder|krankheit|störung)",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------
# Mood-inference patterns (Layer 4)
# ---------------------------------------------------------------

_MOOD_INFERENCE_PATTERNS: list[re.Pattern[str]] = [
    # "User seems/appears/is [mood state]"
    re.compile(
        r"(?:user |nutzer )?(?:seems? |appears? |is |looks? |wirkt |scheint |ist )"
        r"(?:to be )?(?:sad|happy|angry|stressed|frustrated|tired|exhausted|"
        r"depressed|anxious|upset|irritable|overwhelmed|"
        r"traurig|glücklich|wütend|gestresst|frustriert|müde|erschöpft|"
        r"deprimiert|ängstlich|verärgert|gereizt|überfordert)",
        re.IGNORECASE,
    ),
    # "Mood / emotional state / Stimmung"
    re.compile(
        r"(?:mood|emotional state|affect|gefuehlslage|stimmung|gemütszustand|befinden)"
        r".*(?:detect|track|monitor|infer|predict|analyz|erken|verfolg|überwach|vorhersag|analysier)",
        re.IGNORECASE,
    ),
    # "Sentiment analysis on user" (not on content the user wants analyzed)
    re.compile(
        r"(?:user|nutzer)(?:'s| ).*(?:sentiment|emotion|feeling|gefühl|empfind)",
        re.IGNORECASE,
    ),
    # "User's [temporal] mood pattern"
    re.compile(
        r"(?:daily|weekly|morning|evening|night|täglich|wöchentlich|morgens|abends)"
        r".*(?:mood|emotion|stimmung|gefühl)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HealthcareFilterResult:
    """Result of healthcare filter check.

    Attributes:
        blocked: Whether the item was blocked.
        layer: Which filter layer triggered the block (1-4).
        reason: Human-readable explanation of the block.
        matched_term: The specific term or pattern that triggered.
    """

    blocked: bool
    layer: int = 0
    reason: str = ""
    matched_term: str = ""


# ---------------------------------------------------------------
# HealthcareFilter
# ---------------------------------------------------------------


class HealthcareFilter:
    """Blocks health-related pattern inferences (HC-SC-14).

    Multi-layer heuristic check that prevents the system from
    storing any pattern that could be used for behavioral-clinical
    phenotyping or mood inference.

    Usage:
        hf = HealthcareFilter()
        if hf.filter_hypothesis(hypothesis):
            # hypothesis is blocked, do not materialize
            reason = hf.get_block_reason(hypothesis)
    """

    def is_health_related_event(self, event: NormalizedEvent) -> bool:
        """Check if an event belongs to a health-related domain.

        Layer 1: Domain check against blocked health domains.

        Args:
            event: Normalized event from Layer 1.

        Returns:
            True if the event domain is health-related.
        """
        return event.domain.lower() in BLOCKED_HEALTH_DOMAINS

    def filter_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Check if a hypothesis should be blocked (True = block).

        Runs all four filter layers in order:
          1. Domain check (from hypothesis scope context)
          2. Keyword scan (claim text)
          3. Behavioral-change pattern detection (claim text)
          4. Mood-inference pattern detection (claim text)

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            True if the hypothesis should be BLOCKED.
        """
        result = self._evaluate(hypothesis)
        if result.blocked:
            log.info(
                "Healthcare filter BLOCKED hypothesis %s (layer=%d): %s",
                hypothesis.hypothesis_id,
                result.layer,
                result.reason,
            )
        return result.blocked

    def get_block_reason(self, hypothesis: Hypothesis) -> Optional[str]:
        """Get the block reason for a hypothesis.

        Returns None if the hypothesis is not blocked.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            Block reason string, or None if not blocked.
        """
        result = self._evaluate(hypothesis)
        if result.blocked:
            return result.reason
        return None

    def _evaluate(self, hypothesis: Hypothesis) -> HealthcareFilterResult:
        """Run all four filter layers on a hypothesis.

        Two-pass pattern matching (Phase 1.5 Polish-Polish architecture):
          Pass 1: normalize_for_security_check (NFKC + Cf-strip).
            Preserves umlauts while stripping Zero-Width and composing
            combining diacritics via NFKC. German keywords match natively.
          Pass 2: normalize_aggressive (NFD + Mn-strip + confusables-fold + NFKC).
            Folds everything to Latin. Uses _HEALTHCARE_PATTERN_AGGRESSIVE
            (keywords also aggressive-normalized) to avoid Split-Brain.

        Args:
            hypothesis: The hypothesis to evaluate.

        Returns:
            HealthcareFilterResult with block decision.
        """
        claim = hypothesis.claim

        # Layer 1: Domain check via scope context tags
        for tag in hypothesis.scope.context:
            if tag.lower() in BLOCKED_HEALTH_DOMAINS:
                return HealthcareFilterResult(
                    blocked=True,
                    layer=1,
                    reason=f"Health-related domain in scope: '{tag}'",
                    matched_term=tag,
                )

        # Two-pass matching (Phase 1.5 Polish-Polish):
        # Pass 1: basic normalization (NFKC + Cf-strip). Preserves umlauts,
        # strips Zero-Width chars, composes combining diacritics to pre-composed.
        # Catches: DE+ZWSP, DE+combining-diaeresis.
        basic_claim = normalize_for_security_check(claim)
        # Pass 2: aggressive normalization. Folds confusables to Latin, strips
        # all combining marks. Catches: Cyrillic-in-DE, mixed-script attacks.
        aggressive_claim = normalize_aggressive(claim)

        # --- Pass 1: basic claim vs raw patterns (umlauts preserved both sides)
        match = _HEALTHCARE_PATTERN.search(basic_claim)
        if match:
            return HealthcareFilterResult(
                blocked=True,
                layer=2,
                reason=f"Healthcare keyword detected: '{match.group()}'",
                matched_term=match.group(),
            )

        for pattern in _BEHAVIORAL_CHANGE_PATTERNS:
            match = pattern.search(basic_claim)
            if match:
                return HealthcareFilterResult(
                    blocked=True,
                    layer=3,
                    reason=(
                        f"Behavioral-clinical phenotyping pattern: '{match.group()}'"
                    ),
                    matched_term=match.group(),
                )

        for pattern in _MOOD_INFERENCE_PATTERNS:
            match = pattern.search(basic_claim)
            if match:
                return HealthcareFilterResult(
                    blocked=True,
                    layer=4,
                    reason=f"Mood-inference pattern: '{match.group()}'",
                    matched_term=match.group(),
                )

        # --- Pass 2: aggressive claim vs aggressive patterns (Latin-only both)
        match = _HEALTHCARE_PATTERN_AGGRESSIVE.search(aggressive_claim)
        if match:
            return HealthcareFilterResult(
                blocked=True,
                layer=2,
                reason=f"Healthcare keyword detected (aggressive): '{match.group()}'",
                matched_term=match.group(),
            )

        for pattern in _BEHAVIORAL_CHANGE_PATTERNS:
            match = pattern.search(aggressive_claim)
            if match:
                return HealthcareFilterResult(
                    blocked=True,
                    layer=3,
                    reason=(
                        f"Behavioral-clinical phenotyping pattern: '{match.group()}'"
                    ),
                    matched_term=match.group(),
                )

        for pattern in _MOOD_INFERENCE_PATTERNS:
            match = pattern.search(aggressive_claim)
            if match:
                return HealthcareFilterResult(
                    blocked=True,
                    layer=4,
                    reason=f"Mood-inference pattern: '{match.group()}'",
                    matched_term=match.group(),
                )

        return HealthcareFilterResult(blocked=False)
