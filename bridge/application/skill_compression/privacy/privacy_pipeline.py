"""Privacy Pipeline: orchestrates all privacy filters before hypothesis creation.

Integrates:
  - HealthcareFilter (HC-SC-14)
  - SecretScanner (HC-SC-13)
  - NudgeFilter (HC-SC-15)

Called by PatternJudge BEFORE any hypothesis promotion from candidate
to suggested. All rejections are logged to an audit trail.

Architecture guard: Privacy filters are hard gates. If any filter
rejects, the hypothesis is NOT promoted. No bypass possible.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from application.skill_compression.hypothesis_storage import Hypothesis
from application.skill_compression.privacy.healthcare_filter import HealthcareFilter
from application.skill_compression.privacy.nudge_filter import NudgeFilter
from application.skill_compression.privacy.secret_scanner import SecretScanner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Rejection reasons
# ---------------------------------------------------------------


class RejectionSource(str, Enum):
    """Source of a privacy pipeline rejection."""

    HEALTHCARE = "healthcare_filter"
    SECRET = "secret_scanner"  # nosec B105 (enum value, not a credential)
    NUDGE = "nudge_filter"


@dataclass(frozen=True, slots=True)
class PipelineRejection:
    """A privacy pipeline rejection record.

    Attributes:
        hypothesis_id: The rejected hypothesis ID.
        source: Which filter rejected it.
        reason: Human-readable reason.
        timestamp: ISO-8601 UTC timestamp of rejection.
    """

    hypothesis_id: str
    source: RejectionSource
    reason: str
    timestamp: str


# ---------------------------------------------------------------
# Audit log (in-memory, rotated per session)
# ---------------------------------------------------------------


@dataclass
class PrivacyAuditLog:
    """In-memory audit log for privacy pipeline rejections.

    Stores the last N rejections for transparency and debugging.
    Not persisted to DB (privacy-by-design: we don't store what
    we rejected, only that we rejected).

    Attributes:
        max_entries: Maximum entries before rotation.
        entries: List of rejection records.
    """

    max_entries: int = 1000
    entries: list[PipelineRejection] = field(default_factory=list)

    def add(self, rejection: PipelineRejection) -> None:
        """Add a rejection to the audit log.

        Rotates oldest entries when max_entries is reached.

        Args:
            rejection: The rejection record to add.
        """
        self.entries.append(rejection)
        if len(self.entries) > self.max_entries:
            # Keep the newest half
            self.entries = self.entries[self.max_entries // 2 :]

    def get_recent(self, count: int = 50) -> list[PipelineRejection]:
        """Get the most recent rejections.

        Args:
            count: Number of recent entries to return.

        Returns:
            List of recent rejections (newest first).
        """
        return list(reversed(self.entries[-count:]))

    @property
    def total_rejections(self) -> int:
        """Total number of rejections in the log."""
        return len(self.entries)


# ---------------------------------------------------------------
# Privacy Pipeline
# ---------------------------------------------------------------


class PrivacyPipeline:
    """Orchestrates all privacy filters for hypothesis validation.

    Runs three filters in order (fail-fast):
      1. HealthcareFilter (HC-SC-14)
      2. SecretScanner (HC-SC-13)
      3. NudgeFilter (HC-SC-15)

    If ANY filter rejects, the hypothesis is blocked.
    All rejections are logged to the audit trail.

    Usage:
        pipeline = PrivacyPipeline()
        rejection = pipeline.check(hypothesis)
        if rejection is not None:
            # Do NOT promote this hypothesis
            pass
    """

    def __init__(self) -> None:
        self._healthcare = HealthcareFilter()
        self._secrets = SecretScanner()
        self._nudge = NudgeFilter()
        self._audit = PrivacyAuditLog()

    @property
    def audit_log(self) -> PrivacyAuditLog:
        """Access the privacy audit log."""
        return self._audit

    @property
    def healthcare_filter(self) -> HealthcareFilter:
        """Access the healthcare filter directly."""
        return self._healthcare

    @property
    def secret_scanner(self) -> SecretScanner:
        """Access the secret scanner directly."""
        return self._secrets

    @property
    def nudge_filter(self) -> NudgeFilter:
        """Access the nudge filter directly."""
        return self._nudge

    def check(self, hypothesis: Hypothesis) -> Optional[PipelineRejection]:
        """Run all privacy filters on a hypothesis.

        Fail-fast: returns on first rejection.

        Args:
            hypothesis: The hypothesis to validate.

        Returns:
            PipelineRejection if blocked, None if clean.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Filter 1: Healthcare (HC-SC-14)
        if self._healthcare.filter_hypothesis(hypothesis):
            reason = (
                self._healthcare.get_block_reason(hypothesis)
                or "Healthcare filter: blocked"
            )
            rejection = PipelineRejection(
                hypothesis_id=hypothesis.hypothesis_id,
                source=RejectionSource.HEALTHCARE,
                reason=reason,
                timestamp=now_iso,
            )
            self._audit.add(rejection)
            return rejection

        # Filter 2: Secrets (HC-SC-13)
        if self._secrets.block_if_secrets(hypothesis):
            reason = (
                self._secrets.get_block_reason(hypothesis) or "Secret scanner: blocked"
            )
            rejection = PipelineRejection(
                hypothesis_id=hypothesis.hypothesis_id,
                source=RejectionSource.SECRET,
                reason=reason,
                timestamp=now_iso,
            )
            self._audit.add(rejection)
            return rejection

        # Filter 3: Nudge (HC-SC-15)
        if self._nudge.violates_nudge_policy(hypothesis):
            category = self._nudge.get_violation_category(hypothesis) or "unknown"
            rejection = PipelineRejection(
                hypothesis_id=hypothesis.hypothesis_id,
                source=RejectionSource.NUDGE,
                reason=f"Nudge policy violation: {category}",
                timestamp=now_iso,
            )
            self._audit.add(rejection)
            return rejection

        return None

    def is_blocked(self, hypothesis: Hypothesis) -> bool:
        """Convenience: check if hypothesis is blocked (True = blocked).

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            True if blocked by any filter.
        """
        return self.check(hypothesis) is not None
