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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

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

    def __init__(
        self,
        storage: HypothesisStorage,
        privacy_pipeline: PrivacyPipeline,
    ) -> None:
        self._storage = storage
        self._privacy = privacy_pipeline

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
                "Privacy pipeline blocked learn: hyp=%s source=%s filter=%s",
                hyp_id,
                rejection.source.value,
                rejection.source.value,
            )
            return LearnResult(
                success=False,
                hypothesis_id=hyp_id,
                rejection_reason=rejection.reason,
                rejection_source=rejection.source.value,
            )

        # Privacy passed: persist
        self._storage.insert_hypothesis(hypothesis)

        log.info(
            "Skill learned: hyp=%s user=%d source=%s len=%d",
            hyp_id,
            user_id,
            source,
            len(claim_text),
        )

        return LearnResult(
            success=True,
            hypothesis_id=hyp_id,
        )
