"""SkillLearningService: unified privacy gate for manual and imported skills.

Single entry point for all skill creation paths:
  - /learn command (manual)
  - Conversation import
  - Future: automatic candidate promotion

All paths run through the full PrivacyPipeline (HC-SC-13, HC-SC-14, HC-SC-15)
before any hypothesis is persisted. This prevents the bypass found in the
Codex review (SC-02): /learn previously only checked SecretScanner, missing
HealthcareFilter and NudgeFilter.

Architecture: Application service. No Telegram imports, no infrastructure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import icontract
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import STATUS_CONFIRMED
from application.skill_compression.privacy.privacy_pipeline import (
    PipelineRejection,
    PrivacyPipeline,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Trigger alias extraction (Bug 2 fix)
# ---------------------------------------------------------------

# Minimum alias length to avoid overly broad triggers.
MIN_ALIAS_LEN: int = 2
MAX_ALIAS_LEN: int = 100

# Stoplist: words that must NEVER become a skill alias because they
# are too common in normal conversation and would fire constantly.
# Stoplist per briefing: ja, nein, ok, yes, no, danke, thanks, help.
# Extended with very common functional words that would fire on every message.
_ALIAS_STOPLIST: frozenset[str] = frozenset(
    {
        "ja",
        "nein",
        "ok",
        "yes",
        "no",
        "danke",
        "thanks",
        "thank",
        "help",
        "hilfe",
        "bitte",
        "please",
        "der",
        "die",
        "das",
        "the",
        "a",
        "an",
        "und",
        "and",
        "or",
        "oder",
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "we",
        "you",
        "it",
    }
)

# Patterns for trigger alias extraction.
# DE: "wenn ich X sage/schreibe/tippe/eingebe/sende" (original direction)
# DE: "wenn ich sage/schreibe/tippe/eingebe/sende X" (reversed direction, Round-4)
_TRIGGER_PATTERNS_DE: list[re.Pattern[str]] = [
    # Original: "wenn ich <TRIGGER> sage/schreibe/..."
    re.compile(
        r"wenn\s+ich\s+(.+?)\s+(?:sage|schreibe|tippe|eingebe|sende)",
        re.IGNORECASE,
    ),
    # Round-4 reversed: "wenn ich schreibe/sage/... <TRIGGER>"
    # Captures the word(s) AFTER the verb up to a comma, period, or end.
    re.compile(
        r"wenn\s+ich\s+(?:sage|schreibe|tippe|eingebe|sende)\s+"
        r"([^,.\n]+?)(?:\s*[,.]|\s*$)",
        re.IGNORECASE,
    ),
]

# EN: "when I say/type/write/send/enter X"
# Strategy: capture the first word after the verb as the trigger.
# If quoted, capture the full quoted content. Otherwise, capture
# the first word(s) before a terminator verb, comma, period, or sentence end.
_TRIGGER_PATTERNS_EN: list[re.Pattern[str]] = [
    # Quoted triggers: "when I say "hello" then do X"
    re.compile(
        r"when\s+I\s+(?:say|type|write|send|enter)\s+"
        r"(['\"])(.+?)\1",
        re.IGNORECASE,
    ),
    # Unquoted triggers: "when I say hello greet me" or "when I say red, explain"
    # Captures the word between the verb and the next separator (comma, period,
    # another word, or end of string). The captured group is group(2).
    re.compile(
        r"when\s+I\s+(?:say|type|write|send|enter)\s+"
        r"()([a-zA-Z0-9_\-]+)"
        r"(?:\s|[,.]|$)",
        re.IGNORECASE,
    ),
]


def _extract_trigger_aliases(claim_text: str) -> list[str]:
    """Extract trigger aliases from a skill claim text.

    Looks for natural-language patterns like:
      DE: "wenn ich rot sage, erkläre mir die RGB-Farben"  -> ["rot"]
      EN: "when I say red, explain me RGB colors"           -> ["red"]
      Multiple: "wenn ich rot oder blau sage, mach X"       -> ["rot", "blau"]

    Applies validation:
      - Min length >= MIN_ALIAS_LEN
      - Max length <= MAX_ALIAS_LEN
      - Not in stoplist
      - Stripped and lowercased

    Args:
        claim_text: The raw skill claim text from the user.

    Returns:
        List of validated alias strings (may be empty).
    """
    raw_aliases: list[str] = []

    # Try DE patterns (use findall for multi-trigger support)
    for pattern in _TRIGGER_PATTERNS_DE:
        for match in pattern.finditer(claim_text):
            captured = match.group(1).strip()
            # Handle "oder"/"or" splitting for multi-trigger
            parts = re.split(r"\s+oder\s+|\s+or\s+", captured, flags=re.IGNORECASE)
            for part in parts:
                cleaned = part.strip().strip("\"'").strip()
                if cleaned:
                    raw_aliases.append(cleaned)

    # Try EN patterns
    for pattern in _TRIGGER_PATTERNS_EN:
        for match in pattern.finditer(claim_text):
            # Group 2 is the actual trigger text (group 1 is optional quote)
            captured = match.group(2).strip()
            parts = re.split(r"\s+oder\s+|\s+or\s+", captured, flags=re.IGNORECASE)
            for part in parts:
                cleaned = part.strip().strip("\"'").strip()
                if cleaned:
                    raw_aliases.append(cleaned)

    # Validate and deduplicate
    seen: set[str] = set()
    valid_aliases: list[str] = []
    for alias in raw_aliases:
        lower = alias.lower()
        if lower in seen:
            continue
        if len(lower) < MIN_ALIAS_LEN or len(lower) > MAX_ALIAS_LEN:
            continue
        if lower in _ALIAS_STOPLIST:
            continue
        # Reject command-like triggers
        if lower.startswith("/"):
            continue
        seen.add(lower)
        valid_aliases.append(lower)

    return valid_aliases


# Allowed source values for learn(). Defined at module level so the
# icontract lambda can reference it without needing `self`.
_ALLOWED_SOURCES: frozenset[str] = frozenset(
    {"learn_command", "import", "auto", "user"}
)


@dataclass(frozen=True, slots=True)
class LearnResult:
    """Result of a learn operation.

    Attributes:
        success: True if the hypothesis was stored.
        hypothesis_id: ID of the stored hypothesis (empty on failure).
        rejection_reason: Human-readable reason if blocked.
        rejection_source: Which filter blocked (empty on success).
    """

    success: bool
    hypothesis_id: str = ""
    rejection_reason: str = ""
    rejection_source: str = ""


class SkillLearningService:
    """Unified service for learning new skills with full privacy gate.

    Ensures that ALL learning paths (manual /learn, import, automatic)
    run through the complete PrivacyPipeline before persisting.

    Usage:
        service = SkillLearningService(storage, privacy_pipeline)
        result = service.learn(
            claim_text="Always respond in German",
            user_id=12345,
            source="learn_command",
        )
        if not result.success:
            print(f"Blocked: {result.rejection_reason}")
    """

    # Exposed as class attribute for test introspection.
    ALLOWED_SOURCES: frozenset[str] = _ALLOWED_SOURCES

    def __init__(
        self,
        storage: HypothesisStorage,
        privacy_pipeline: PrivacyPipeline,
    ) -> None:
        self._storage = storage
        self._privacy = privacy_pipeline

    @icontract.require(
        lambda claim_text: claim_text and claim_text.strip(),
        "claim_text must not be empty or whitespace-only",
    )
    @icontract.require(
        lambda user_id: user_id > 0,
        "user_id must be a positive integer",
    )
    @icontract.require(
        lambda source: source in _ALLOWED_SOURCES,
        "source must be one of the allowed source values",
    )
    @icontract.ensure(
        lambda result: not result.success or result.hypothesis_id,
        "on success, hypothesis_id must be non-empty",
    )
    def learn(
        self,
        claim_text: str,
        user_id: int,
        source: str = "learn_command",
        *,
        status: str = STATUS_CONFIRMED,
        decay_immune: bool = True,
        approval_state: str = "approved",
    ) -> LearnResult:
        """Create a new skill hypothesis with full privacy validation.

        Runs the complete PrivacyPipeline (Healthcare + Secret + Nudge)
        before storing. If any filter rejects, nothing is persisted.

        Contracts:
            Pre: claim_text is not empty.
            Pre: user_id > 0.
            Pre: source in ALLOWED_SOURCES.
            Post: if success=True then hypothesis_id is non-empty.

        Args:
            claim_text: The skill claim text.
            user_id: Telegram user ID.
            source: Origin of the skill (learn_command, import, auto).
            status: Initial status (default: confirmed for /learn).
            decay_immune: Whether the skill is immune to FSRS decay.
            approval_state: Initial approval state.

        Returns:
            LearnResult with success flag and details.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        hyp_id = f"hyp_{uuid4().hex[:16]}"

        # Build a temporary hypothesis for privacy validation
        hypothesis = Hypothesis(
            hypothesis_id=hyp_id,
            user_id=user_id,
            type="preference",
            scope=HypothesisScope(),
            claim=claim_text,
            status=status,
            version=1,
            elo_rating=1500.0,
            elo_games_played=0,
            bayes_confidence=0.5,
            support_count=1,
            contradict_count=0,
            source_type=source,
            decay_immune=decay_immune,
            created_at=now_iso,
            last_applied=None,
            last_seen=now_iso,
            approval_state=approval_state,
        )

        # Full privacy pipeline check (HC-SC-13 + HC-SC-14 + HC-SC-15)
        rejection: Optional[PipelineRejection] = self._privacy.check(hypothesis)
        if rejection is not None:
            log.info(
                "Privacy pipeline blocked learn: hyp=%s source=%s reason=%s",
                hyp_id,
                rejection.source.value,
                rejection.reason,
            )
            return LearnResult(
                success=False,
                hypothesis_id=hyp_id,
                rejection_reason=rejection.reason,
                rejection_source=rejection.source.value,
            )

        # Privacy passed: persist
        self._storage.insert_hypothesis(hypothesis)

        # Bug 2: Extract trigger aliases from the claim text and persist.
        aliases = _extract_trigger_aliases(claim_text)
        for alias_text in aliases:
            alias_id = f"alias_{uuid4().hex[:12]}"
            self._storage.insert_alias(
                alias_id=alias_id,
                hypothesis_id=hyp_id,
                alias_text=alias_text,
                first_seen=now_iso,
                last_seen=now_iso,
                confidence=0.9,
                evidence_count=1,
            )
            # B2-2: Do NOT log alias cleartext. Only log length.
            log.info(
                "Alias created for skill: hyp=%s alias_len=%d",
                hyp_id,
                len(alias_text),
            )

        log.info(
            "Skill learned: hyp=%s user=%d source=%s len=%d aliases=%d",
            hyp_id,
            user_id,
            source,
            len(claim_text),
            len(aliases),
        )

        return LearnResult(
            success=True,
            hypothesis_id=hyp_id,
        )
