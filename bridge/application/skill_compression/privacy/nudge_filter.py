"""Nudge Filter: enforces the nudge negative list (HC-SC-15).

AXOLENT's self-imposed ethical constraints. Pattern suggestions that
violate the negative list are NOT materialized as hypotheses.

Negative categories (from spec, complete list):
  1. Political / Ideological Manipulation
  2. Emotional Manipulation (FOMO, loss aversion, relationship suggestion)
  3. Dark Patterns (hidden opt-out, roach motel, confirmshaming)
  4. Attention Maximization (streaks, gamification, engagement loops)
  5. Social Manipulation (user comparison, leaderboards)
  6. Behavioral inferences not serving user help
  7. Data flow violations (third parties, default cloud, telemetry)

HC-SC-15 [BLOCKER]: Nudge self-commitment. Negative list fully
  enforced where technically possible.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from application.security.input_normalizer import (
    normalize_aggressive,
    normalize_for_security_check,
)
from application.skill_compression.hypothesis_storage import Hypothesis

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Violation categories
# ---------------------------------------------------------------


class NudgeCategory(str, Enum):
    """Categories of nudge policy violations."""

    POLITICAL_MANIPULATION = "political_manipulation"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    DARK_PATTERNS = "dark_patterns"
    ATTENTION_MAXIMIZATION = "attention_maximization"
    SOCIAL_MANIPULATION = "social_manipulation"
    BEHAVIORAL_INFERENCE = "behavioral_inference"
    DATA_FLOW_VIOLATION = "data_flow_violation"


# ---------------------------------------------------------------
# Category descriptions (for audit log and user feedback)
# ---------------------------------------------------------------

CATEGORY_DESCRIPTIONS: dict[NudgeCategory, str] = {
    NudgeCategory.POLITICAL_MANIPULATION: (
        "Politische/ideologische Manipulation: "
        "keine politische Personalisierung, keine Echo-Chambers, "
        "kein Confirmation-Bias"
    ),
    NudgeCategory.EMOTIONAL_MANIPULATION: (
        "Emotionale Manipulation: "
        "kein FOMO, keine Verlust-Aversion, keine Dringlichkeit, "
        "keine Beziehungs-Suggestionen"
    ),
    NudgeCategory.DARK_PATTERNS: (
        "Dark Patterns: "
        "keine erschwerten Abmeldungen, kein verstecktes Opt-Out, "
        "kein Roach Motel, kein Confirmshaming"
    ),
    NudgeCategory.ATTENTION_MAXIMIZATION: (
        "Aufmerksamkeits-Maximierung: "
        "keine Engagement-Loops, keine Streaks, "
        "keine Gamification, keine Schlafzeit-Benachrichtigungen"
    ),
    NudgeCategory.SOCIAL_MANIPULATION: (
        "Soziale Manipulation: "
        "kein User-Vergleich, keine Leaderboards, "
        "kein sozialer Druck"
    ),
    NudgeCategory.BEHAVIORAL_INFERENCE: (
        "Verhaltensbasierte Inferenz: "
        "keine Stimmungs-Vorhersagen, keine Lebensumstände-Inferenz "
        "ohne explizite User-Aussage"
    ),
    NudgeCategory.DATA_FLOW_VIOLATION: (
        "Datenfluss-Verletzung: "
        "keine Daten an Dritte, keine Default-Cloud, "
        "keine Telemetrie ohne Zustimmung"
    ),
}


# ---------------------------------------------------------------
# Detection patterns per category
# ---------------------------------------------------------------

# Each category has a list of regex patterns that detect violations.
# Patterns are designed to catch the CLAIM text of a hypothesis,
# not user messages. They target system behavior descriptions.

_CATEGORY_PATTERNS: dict[NudgeCategory, list[re.Pattern[str]]] = {
    NudgeCategory.POLITICAL_MANIPULATION: [
        # Political personalization
        re.compile(
            r"(?:political|ideological|partisan|politisch|ideologisch|parteiisch)"
            r".*(?:personali[sz]|target|recommend|bias|vorschlag|empfehl|bevorzug)",
            re.IGNORECASE,
        ),
        # Echo chamber reinforcement
        re.compile(
            r"(?:echo.?chamber|filter.?bubble|confirmation.?bias|bestätigungs.?fehler"
            r"|filterblase|echo.?kammer)",
            re.IGNORECASE,
        ),
        # Bias reinforcement
        re.compile(
            r"(?:reinforce|amplify|strengthen|verstärk|bekräftig)"
            r".*(?:belief|view|opinion|bias|meinung|ansicht|überzeugung)",
            re.IGNORECASE,
        ),
    ],
    NudgeCategory.EMOTIONAL_MANIPULATION: [
        # FOMO
        re.compile(
            r"(?:fomo|fear of missing|verpass(?:t|en)|miss(?:ing|ed) out"
            r"|nicht verpassen)",
            re.IGNORECASE,
        ),
        # Loss aversion tricks
        re.compile(
            r"(?:verlier|lose|losing).*(?:streak|progress|fortschritt|erfolg)",
            re.IGNORECASE,
        ),
        # Artificial urgency
        re.compile(
            r"(?:nur noch|only (?:today|now|left)|limited time|begrenzt"
            r"|last chance|letzte chance|dringend|urgent)"
            r"(?!.*(?:deadline|frist|termin))",  # Allow legit deadlines
            re.IGNORECASE,
        ),
        # Relationship suggestion
        re.compile(
            r"(?:i miss you|ich vermisse dich|i care about|mir liegt an dir"
            r"|i love you|ich liebe dich|we are friends|wir sind freunde"
            r"|i need you|ich brauche dich)",
            re.IGNORECASE,
        ),
    ],
    NudgeCategory.DARK_PATTERNS: [
        # Making deletion/opt-out difficult
        re.compile(
            r"(?:erschwer|hide|versteck|obfuscate|verschleier)"
            r".*(?:delete|opt.?out|cancel|abmeld|kündig|lösch|austrag)",
            re.IGNORECASE,
        ),
        # Roach motel
        re.compile(
            r"(?:roach.?motel|easy.?in.?hard.?out|leicht.?rein.?schwer.?raus)",
            re.IGNORECASE,
        ),
        # Confirmshaming
        re.compile(
            r"(?:confirmshaming|shame|beschäm|guilt|schuld)"
            r".*(?:cancel|decline|opt.?out|ablehnen|abmeld)",
            re.IGNORECASE,
        ),
        # Pre-selected defaults for important decisions
        re.compile(
            r"(?:pre.?select|vorauswahl|default.?(?:on|aktiv|enabled))"
            r".*(?:important|wichtig|critical|consent|einwillig|zustimm)",
            re.IGNORECASE,
        ),
    ],
    NudgeCategory.ATTENTION_MAXIMIZATION: [
        # Engagement loops
        re.compile(
            r"(?:engagement.?loop|infinite.?scroll|endlos.?scroll"
            r"|keep.?(?:them|user).?engaged|user.?(?:bei|am).?ball.?halten)",
            re.IGNORECASE,
        ),
        # Conversation extension when problem is solved
        re.compile(
            r"(?:extend|prolong|verlänger|ausdehnen)"
            r".*(?:conversation|session|chat|gespräch|sitzung)",
            re.IGNORECASE,
        ),
        # Notifications at sleep times
        re.compile(
            r"(?:notif|benachrichtig|push)"
            r".*(?:night|sleep|late|nacht|schlaf|spät)",
            re.IGNORECASE,
        ),
        # Artificial streaks
        re.compile(
            r"(?:streak|daily.?login|tägliche?.?anmeldung"
            r"|consecutive.?day|aufeinanderfolgende.?tage)",
            re.IGNORECASE,
        ),
        # Gamification as attention tool
        re.compile(
            r"(?:gamif|badge|achievement|level.?up|xp|experience.?point"
            r"|punkte.?sammel|level.?aufstieg|errungenschaften)",
            re.IGNORECASE,
        ),
    ],
    NudgeCategory.SOCIAL_MANIPULATION: [
        # User comparison
        re.compile(
            r"(?:compar|vergleich).*(?:other user|andere nutzer|andere user"
            r"|peers|kolleg)",
            re.IGNORECASE,
        ),
        # Leaderboards
        re.compile(
            r"(?:leaderboard|ranking|rangliste|bestenliste|highscore)",
            re.IGNORECASE,
        ),
        # Social pressure
        re.compile(
            r"(?:other(?:s| user| people).*(?:already|schon)|"
            r"andere.*(?:haben schon|nutzen bereits|machen das schon))",
            re.IGNORECASE,
        ),
    ],
    NudgeCategory.BEHAVIORAL_INFERENCE: [
        # Mood/mental state prediction
        re.compile(
            r"(?:predict|infer|detect|erken|vorhersag|ableiten)"
            r".*(?:mood|mental|emotional|stimmung|psychisch|gefühl)",
            re.IGNORECASE,
        ),
        # Life circumstances inference
        re.compile(
            r"(?:infer|detect|erken|ableiten|schließ)"
            r".*(?:life|relationship|financial|leben|beziehung|finanziell)"
            r".*(?:circumstance|situation|status|umstand|lage)",
            re.IGNORECASE,
        ),
    ],
    NudgeCategory.DATA_FLOW_VIOLATION: [
        # Data to third parties
        re.compile(
            r"(?:share|send|transmit|teile|sende|übermittle|weiterleite)"
            r".*(?:third.?part|external|dritte?|extern|partner|vendor|anbieter)",
            re.IGNORECASE,
        ),
        # Default cloud storage
        re.compile(
            r"(?:default|standard).*(?:cloud|remote|server)"
            r".*(?:stor|speicher|upload|sync)",
            re.IGNORECASE,
        ),
        # Telemetry without consent
        re.compile(
            r"(?:telemetr|tracking|analytic|usage.?data|nutzungsdaten)"
            r".*(?:without|ohne|no|kein).*(?:consent|zustimmung|erlaubnis)",
            re.IGNORECASE,
        ),
        # Silent data collection
        re.compile(
            r"(?:silent|heimlich|verdeckt|covert)"
            r".*(?:collect|track|monitor|sammel|erfass|überwach)",
            re.IGNORECASE,
        ),
    ],
}

# Aggressive-normalized pattern variants (Phase 1.5 Polish-Polish):
# For Pass 2, patterns must ALSO be aggressive-normalized so that German
# alternatives with umlauts (e.g. "kündig", "lösch") match the aggressive
# input (e.g. "kundig", "losch"). Built at module load by running
# normalize_aggressive on each pattern's source string. ASCII regex
# metacharacters are unchanged by normalization, only non-ASCII literals
# (umlauts) are folded to Latin equivalents.
_CATEGORY_PATTERNS_AGGRESSIVE: dict[NudgeCategory, list[re.Pattern[str]]] = {}
for _cat, _patterns in _CATEGORY_PATTERNS.items():
    _CATEGORY_PATTERNS_AGGRESSIVE[_cat] = [
        re.compile(normalize_aggressive(p.pattern), p.flags) for p in _patterns
    ]
# Cleanup loop variables from module namespace
del _cat, _patterns


# ---------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NudgeViolation:
    """A detected nudge policy violation.

    Attributes:
        category: The violated category.
        matched_text: The text that triggered the violation.
        description: Category description.
    """

    category: NudgeCategory
    matched_text: str
    description: str


# ---------------------------------------------------------------
# NudgeFilter
# ---------------------------------------------------------------


class NudgeFilter:
    """Enforces the nudge negative list for hypothesis claims (HC-SC-15).

    Checks hypothesis claims against all 7 negative categories.
    Any match = hypothesis is NOT materialized.

    Usage:
        nf = NudgeFilter()
        if nf.violates_nudge_policy(hypothesis):
            category = nf.get_violation_category(hypothesis)
    """

    def violates_nudge_policy(self, hypothesis: Hypothesis) -> bool:
        """Check if a hypothesis violates the nudge negative list.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            True if the hypothesis VIOLATES the policy (= block).
        """
        violation = self._evaluate(hypothesis)
        if violation is not None:
            log.info(
                "Nudge filter BLOCKED hypothesis %s: category=%s",
                hypothesis.hypothesis_id,
                violation.category.value,
            )
            return True
        return False

    def get_violation_category(self, hypothesis: Hypothesis) -> Optional[str]:
        """Get the violation category for a hypothesis.

        Returns None if no violation detected.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            Category value string, or None.
        """
        violation = self._evaluate(hypothesis)
        if violation is not None:
            return violation.category.value
        return None

    def get_violation_detail(self, hypothesis: Hypothesis) -> Optional[NudgeViolation]:
        """Get full violation details for a hypothesis.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            NudgeViolation or None.
        """
        return self._evaluate(hypothesis)

    def _evaluate(self, hypothesis: Hypothesis) -> Optional[NudgeViolation]:
        """Check all negative categories against hypothesis claim.

        Two-pass pattern matching (Phase 1.5 Polish-Polish architecture):
          Pass 1: normalize_for_security_check (NFKC + Cf-strip).
            Preserves umlauts while stripping Zero-Width and composing
            combining diacritics via NFKC. German patterns match natively.
          Pass 2: normalize_aggressive + _CATEGORY_PATTERNS_AGGRESSIVE.
            Folds everything to Latin. Uses aggressive pattern variants
            (German umlaut alternatives also folded) to avoid Split-Brain.

        Args:
            hypothesis: The hypothesis to evaluate.

        Returns:
            NudgeViolation if a violation is found, None otherwise.
        """
        claim = hypothesis.claim

        # Pass 1: basic normalization (NFKC + Cf-strip). Preserves umlauts,
        # strips Zero-Width chars, composes combining diacritics.
        basic_claim = normalize_for_security_check(claim)
        for category, patterns in _CATEGORY_PATTERNS.items():
            for pattern in patterns:
                m = pattern.search(basic_claim)
                if m:
                    return NudgeViolation(
                        category=category,
                        matched_text=m.group()[:50],
                        description=CATEGORY_DESCRIPTIONS[category],
                    )

        # Pass 2: aggressive normalization + aggressive patterns.
        # Folds confusables to Latin, strips combining marks.
        aggressive_claim = normalize_aggressive(claim)
        for category, patterns in _CATEGORY_PATTERNS_AGGRESSIVE.items():
            for pattern in patterns:
                m = pattern.search(aggressive_claim)
                if m:
                    return NudgeViolation(
                        category=category,
                        matched_text=m.group()[:50],
                        description=CATEGORY_DESCRIPTIONS[category],
                    )

        return None
