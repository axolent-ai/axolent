"""Layer 1: Event Normalizer for Skill-Compression.

Reads each bot-user interaction and extracts structured fields:
  intent, domain, format, constraints, scope, language,
  correction_keywords_present, is_re_formulation.

Generates a deterministic SHA-256 fingerprint hash from the structured
fields for later similarity matching.

Implementation: rule-based heuristics (no LLM call, no embedding).
Uses TaskRouter's existing keyword/pattern vocabulary as a foundation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Intent classification keywords (rule-based, no LLM)
# ──────────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "create_code",
        re.compile(
            r"(schreib|write|code|implement|program|script|function|class|bug|fix|debug|refactor)",
            re.IGNORECASE,
        ),
    ),
    (
        "create_text",
        re.compile(
            r"(schreib|write|text|artikel|article|blog|post|essay|brief|letter|mail|email)",
            re.IGNORECASE,
        ),
    ),
    (
        "create_video_concept",
        re.compile(
            r"(dreh|video|reel|tiktok|youtube|concept|konzept|storyboard|skript|script)",
            re.IGNORECASE,
        ),
    ),
    (
        "create_ad_copy",
        re.compile(
            r"(ad\s?copy|anzeige|werbung|headline|hook|cta|retarget|funnel|campaign)",
            re.IGNORECASE,
        ),
    ),
    (
        "analyze",
        re.compile(
            r"(analy|research|recherch|vergleich|compare|evaluat|assess|review|audit)",
            re.IGNORECASE,
        ),
    ),
    (
        "explain",
        re.compile(
            r"(erkl(?:ä|ae)r|explain|what\s+is|was\s+ist|how\s+does|wie\s+funktionier|warum|why)",
            re.IGNORECASE,
        ),
    ),
    (
        "summarize",
        re.compile(
            r"(zusammenfass|fasse?\b|zusammen\b|summarize|summary|tldr|tl;dr|k(?:ü|ue)rz|shorten|condense)",
            re.IGNORECASE,
        ),
    ),
    (
        "translate",
        re.compile(
            r"((?:ü|ue)bersetz|translate|translation|ins?\s+(?:deutsch|englis)|to\s+(?:german|english))",
            re.IGNORECASE,
        ),
    ),
    (
        "plan",
        re.compile(
            r"(plan|strateg|roadmap|timeline|schedule|planung|zeitplan)",
            re.IGNORECASE,
        ),
    ),
    (
        "format",
        re.compile(
            r"(format|struktur|structure|tabelle|table|markdown|json|csv|html)",
            re.IGNORECASE,
        ),
    ),
]

# ──────────────────────────────────────────────────────────────
# Domain classification keywords
# ──────────────────────────────────────────────────────────────

_DOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "marketing",
        re.compile(
            r"(marketing|ad|campaign|funnel|retarget|ctr|roas|conversion|landing\s?page|seo|sem)",
            re.IGNORECASE,
        ),
    ),
    (
        "development",
        re.compile(
            r"(code|python|javascript|typescript|react|api|backend|frontend|database|sql|git|docker)",
            re.IGNORECASE,
        ),
    ),
    (
        "finance",
        re.compile(
            r"(finance|finanz|steuer|tax|buchhalt|accounting|bilanz|balance|cashflow|budget|rechnung|invoice)",
            re.IGNORECASE,
        ),
    ),
    (
        "business",
        re.compile(
            r"(business|gesch(?:ä|ae)ft|revenue|umsatz|gewinn|profit|client|kunde|vertrag)",
            re.IGNORECASE,
        ),
    ),
    (
        "design",
        re.compile(
            r"(design|figma|layout|ui|ux|wireframe|mockup|brand|logo|visual|farbe|color)",
            re.IGNORECASE,
        ),
    ),
    (
        "content",
        re.compile(
            r"(content|blog|artikel|article|social\s?media|post|caption|story|reel|newsletter)",
            re.IGNORECASE,
        ),
    ),
    (
        "data",
        re.compile(
            r"(data|daten|analytics|tracking|report|dashboard|metric|kpi|spreadsheet)",
            re.IGNORECASE,
        ),
    ),
    (
        "legal",
        re.compile(
            r"(legal|recht|gesetz|law|vertrag|contract|gdpr|dsgvo|compliance|datenschutz|privacy)",
            re.IGNORECASE,
        ),
    ),
]

# ──────────────────────────────────────────────────────────────
# Format type classification keywords
# ──────────────────────────────────────────────────────────────

_FORMAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("code", re.compile(r"(code|script|function|class|snippet|```)", re.IGNORECASE)),
    ("table", re.compile(r"(tabelle|table|csv|spreadsheet|matrix)", re.IGNORECASE)),
    ("list", re.compile(r"(liste|list|bullet|aufzählung|punkt)", re.IGNORECASE)),
    ("script", re.compile(r"(skript|script|drehbuch|storyboard)", re.IGNORECASE)),
    ("email", re.compile(r"(mail|email|e-mail|brief|letter|nachricht)", re.IGNORECASE)),
    (
        "report",
        re.compile(r"(report|bericht|analyse|analysis|audit|review)", re.IGNORECASE),
    ),
    ("json", re.compile(r"(json|yaml|xml|config|schema)", re.IGNORECASE)),
    ("html", re.compile(r"(html|webpage|website|landing\s?page)", re.IGNORECASE)),
    ("markdown", re.compile(r"(markdown|md|readme|documentation|doku)", re.IGNORECASE)),
    ("plain_text", re.compile(r"(text|plain|einfach|simple)", re.IGNORECASE)),
]

# ──────────────────────────────────────────────────────────────
# Correction detection keywords
# ──────────────────────────────────────────────────────────────

_CORRECTION_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(nein|no|falsch|wrong|anders|different|nicht\s+so|not\s+like|"
    r"korrigier|correct|änder|change|stattdessen|instead|"
    r"eher|rather|besser|better|doch\s+nicht|actually\s+not)\b",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────
# Constraint extraction patterns
# ──────────────────────────────────────────────────────────────

_DURATION_PATTERN = re.compile(
    r"(\d+)\s*(sekund|second|sec|s\b|minut|min\b|stund|hour|h\b)",
    re.IGNORECASE,
)

_LENGTH_PATTERN = re.compile(
    r"(\d+)\s*(wort|word|zeichen|char|zeile|line|satz|sentence|absatz|paragraph|seite|page|token)",
    re.IGNORECASE,
)

_FUNNEL_PATTERN = re.compile(
    r"(tofu|mofu|bofu|top\s*of\s*funnel|middle\s*of\s*funnel|bottom\s*of\s*funnel|"
    r"awareness|consideration|decision|retarget|conversion)",
    re.IGNORECASE,
)

_AUDIENCE_PATTERN = re.compile(
    r"(zielgruppe|target\s*audience|persona|segment|alter|age\s*group|"
    r"b2b|b2c|enterprise|startup|freelancer|consumer)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────
# Language detection (simple heuristic, not langdetect)
# ──────────────────────────────────────────────────────────────

_GERMAN_INDICATORS = re.compile(
    r"[äöüß]|(\b(und|oder|aber|nicht|ein|eine|der|die|das|ist|sind|hat|haben|"
    r"für|über|nach|mit|bei|von|zu|auf|wie|was|wer|wo|wann|warum|bitte|danke)\b)",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    """Simple language detection heuristic.

    Returns ISO 639-1 code. Defaults to 'en' if unclear.

    Args:
        text: Input text to classify.

    Returns:
        'de' or 'en'.
    """
    german_matches = len(_GERMAN_INDICATORS.findall(text))
    words = len(text.split())
    if words == 0:
        return "en"
    # If more than 15% of word-position matches are German indicators
    if german_matches / max(words, 1) > 0.15:
        return "de"
    return "en"


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """A structured event extracted from a user-bot interaction.

    All fields are extracted via rule-based heuristics (no LLM call).
    The fingerprint_hash provides a deterministic identifier for
    similarity matching in Layer 2.

    Attributes:
        event_id: Unique ID for this event.
        user_id: Telegram user ID.
        timestamp: ISO-8601 UTC timestamp.
        raw_text: Original user message (for debugging, not used in matching).
        intent: Classified intent (e.g. 'create_code', 'analyze', 'general').
        domain: Classified domain (e.g. 'marketing', 'development', 'general').
        format_type: Classified output format (e.g. 'code', 'table', 'plain_text').
        constraints: Extracted constraints (duration, length, funnel, audience).
        scope: Context scope (project, client, session).
        language: ISO 639-1 language code.
        correction_keywords_present: Whether correction keywords were detected.
        is_re_formulation: Whether this looks like a re-formulation of a prior ask.
        fingerprint_hash: SHA-256 over structured fields (deterministic).
    """

    event_id: str = ""
    user_id: int = 0
    timestamp: str = ""
    raw_text: str = ""
    intent: str = "general"
    domain: str = "general"
    format_type: str = "plain_text"
    constraints: dict = field(default_factory=dict)
    scope: dict = field(default_factory=dict)
    language: str = "en"
    correction_keywords_present: bool = False
    is_re_formulation: bool = False
    fingerprint_hash: str = ""

    def to_dict(self) -> dict:
        """Serialize the event to a dict for DB storage."""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "raw_text": self.raw_text,
            "intent": self.intent,
            "domain": self.domain,
            "format_type": self.format_type,
            "constraints": self.constraints,
            "scope": self.scope,
            "language": self.language,
            "correction_keywords_present": self.correction_keywords_present,
            "is_re_formulation": self.is_re_formulation,
            "fingerprint_hash": self.fingerprint_hash,
        }


# ──────────────────────────────────────────────────────────────
# Core extraction functions
# ──────────────────────────────────────────────────────────────


def _classify_by_patterns(
    text: str,
    patterns: list[tuple[str, re.Pattern[str]]],
    default: str = "general",
) -> str:
    """Classify text by matching against a list of (label, regex) pairs.

    Returns the label with the most matches. On tie, the first match wins.

    Args:
        text: Input text.
        patterns: List of (label, compiled_regex) pairs.
        default: Default label if no patterns match.

    Returns:
        Best matching label or default.
    """
    best_label = default
    best_count = 0

    for label, pattern in patterns:
        matches = pattern.findall(text)
        count = len(matches)
        if count > best_count:
            best_count = count
            best_label = label

    return best_label


def _extract_constraints(text: str) -> dict:
    """Extract structured constraints from text.

    Looks for duration, length, funnel stage, and audience indicators.

    Args:
        text: Input text.

    Returns:
        Dict with constraint fields (only non-empty fields included).
    """
    constraints: dict = {}

    duration = _DURATION_PATTERN.search(text)
    if duration:
        value = duration.group(1)
        unit_raw = duration.group(2).lower()
        if unit_raw.startswith(("sekund", "second", "sec", "s")):
            constraints["duration"] = f"{value}s"
        elif unit_raw.startswith(("minut", "min")):
            constraints["duration"] = f"{value}min"
        elif unit_raw.startswith(("stund", "hour", "h")):
            constraints["duration"] = f"{value}h"

    length = _LENGTH_PATTERN.search(text)
    if length:
        value = length.group(1)
        unit_raw = length.group(2).lower()
        # Normalize unit
        unit_map = {
            "wort": "words",
            "word": "words",
            "zeichen": "chars",
            "char": "chars",
            "zeile": "lines",
            "line": "lines",
            "satz": "sentences",
            "sentence": "sentences",
            "absatz": "paragraphs",
            "paragraph": "paragraphs",
            "seite": "pages",
            "page": "pages",
            "token": "tokens",  # nosec B105 (unit-name mapping, not a password)
        }
        for key, normalized in unit_map.items():
            if unit_raw.startswith(key):
                constraints["length"] = f"{value} {normalized}"
                break

    funnel = _FUNNEL_PATTERN.search(text)
    if funnel:
        funnel_raw = funnel.group(0).lower().strip()
        funnel_map = {
            "tofu": "awareness",
            "top of funnel": "awareness",
            "awareness": "awareness",
            "mofu": "consideration",
            "middle of funnel": "consideration",
            "consideration": "consideration",
            "bofu": "decision",
            "bottom of funnel": "decision",
            "decision": "decision",
            "retarget": "retargeting",
            "conversion": "conversion",
        }
        for key, normalized in funnel_map.items():
            if key in funnel_raw:
                constraints["funnel"] = normalized
                break

    audience = _AUDIENCE_PATTERN.search(text)
    if audience:
        constraints["audience"] = audience.group(0).strip().lower()

    return constraints


def compute_fingerprint(
    intent: str,
    domain: str,
    format_type: str,
    constraints: dict,
    scope: dict,
    language: str,
) -> str:
    """Compute a deterministic SHA-256 fingerprint from structured fields.

    The hash is computed over a JSON-serialized canonical representation
    with sorted keys to ensure determinism.

    Args:
        intent: Classified intent.
        domain: Classified domain.
        format_type: Classified format.
        constraints: Extracted constraints.
        scope: Context scope.
        language: ISO 639-1 code.

    Returns:
        64-character hex SHA-256 hash.
    """
    canonical = json.dumps(
        {
            "intent": intent,
            "domain": domain,
            "format_type": format_type,
            "constraints": constraints,
            "scope": scope,
            "language": language,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_event(
    text: str,
    user_id: int,
    *,
    scope: Optional[dict] = None,
    previous_text: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> NormalizedEvent:
    """Normalize a user message into a structured event.

    This is the main entry point for Layer 1. It extracts all structured
    fields via rule-based heuristics and computes the fingerprint hash.

    Args:
        text: User message text.
        user_id: Telegram user ID.
        scope: Optional scope context (project, client, session_id).
        previous_text: Previous user message (for re-formulation detection).
        timestamp: ISO-8601 timestamp. If None, uses current UTC time.

    Returns:
        A fully populated NormalizedEvent.
    """
    if scope is None:
        scope = {}

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    # Classify structured fields
    intent = _classify_by_patterns(text, _INTENT_PATTERNS)
    domain = _classify_by_patterns(text, _DOMAIN_PATTERNS)
    format_type = _classify_by_patterns(text, _FORMAT_PATTERNS)
    constraints = _extract_constraints(text)
    language = _detect_language(text)

    # Correction detection
    correction_present = bool(_CORRECTION_KEYWORDS.search(text))

    # Re-formulation detection (IC-SC-9: within same session context)
    is_reformulation = False
    if previous_text is not None and previous_text.strip():
        # Simple heuristic: if previous and current share >40% of words
        prev_words = set(previous_text.lower().split())
        curr_words = set(text.lower().split())
        if prev_words and curr_words:
            overlap = len(prev_words & curr_words)
            total = len(prev_words | curr_words)
            if total > 0 and overlap / total > 0.4:
                is_reformulation = True

    # Compute fingerprint
    fp_hash = compute_fingerprint(
        intent=intent,
        domain=domain,
        format_type=format_type,
        constraints=constraints,
        scope=scope,
        language=language,
    )

    event = NormalizedEvent(
        event_id=f"evt_{uuid4().hex[:16]}",
        user_id=user_id,
        timestamp=timestamp,
        raw_text=text,
        intent=intent,
        domain=domain,
        format_type=format_type,
        constraints=constraints,
        scope=scope,
        language=language,
        correction_keywords_present=correction_present,
        is_re_formulation=is_reformulation,
        fingerprint_hash=fp_hash,
    )

    log.debug(
        "Event normalized: id=%s intent=%s domain=%s format=%s lang=%s fp=%s",
        event.event_id,
        event.intent,
        event.domain,
        event.format_type,
        event.language,
        event.fingerprint_hash[:12],
    )

    return event
