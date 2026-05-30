"""Skill Installer: manual/offline skill installation via /installskill.

Enables users to install skills from local disk (skills/ folder) or
Telegram file upload, bypassing the /learn flow but NOT bypassing safety.

One Safety Gate: every installed skill passes through the SAME safety
pipeline (PrivacyPipeline + ContractValidator) as /learn. There is NO
security bypass via install.

Path-Boundary Security (Codex Addendum K5):
  - _validate_source_path() with Path.resolve() + is_relative_to()
  - Rejects: path traversal (..), symlinks, absolute outside allowed_root
  - Rejects: remote URLs (http/https/file/ftp)
  - allowed_root is a parameter (SKILL_DIR for folder, upload_tmp for uploads)

Dependencies: Python stdlib only (pathlib, json, logging).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from application.skill_compression.contract_store import (
    ContractStore,
    ContractStoreError,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_contract import (
    SkillContract,
    iter_user_text_fields,
    now_iso,
)
from infrastructure.safe_json import safe_json_load

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Remote URL schemes (rejected before path resolution)
# ──────────────────────────────────────────────────────────────

_REMOTE_SCHEMES = ("http://", "https://", "file://", "ftp://")


# ──────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PathValidationResult:
    """Outcome of path boundary validation.

    Attributes:
        valid: True if the path is safe to read.
        reason: Why the path was rejected (empty on valid).
    """

    valid: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class InstallResult:
    """Outcome of a skill installation attempt.

    Attributes:
        success: True if the skill was installed.
        contract_name: Name of the installed contract (on success).
        error: Error description (on failure).
    """

    success: bool
    contract_name: str = ""
    error: str = ""


# ──────────────────────────────────────────────────────────────
# Path-Boundary Validation (Codex Addendum K5)
# ──────────────────────────────────────────────────────────────


def reject_remote_url(raw_input: str) -> Optional[str]:
    """Check if raw user input is a remote URL. Returns rejection reason or None.

    Called BEFORE any path resolution. Rejects http://, https://, file://, ftp://.

    Args:
        raw_input: Raw user input string.

    Returns:
        Rejection reason string if the input is a remote URL, None otherwise.
    """
    lowered = raw_input.strip().lower()
    for scheme in _REMOTE_SCHEMES:
        if lowered.startswith(scheme):
            return (
                "Remote URLs are not accepted. "
                "Please provide a local file in the skills directory."
            )
    return None


def validate_source_path(source: Path, allowed_root: Path) -> PathValidationResult:
    """Validate that a source path is within allowed boundaries.

    Security checks (order matters):
      1. Reject symlinks (BEFORE resolve, prevents symlink-to-outside attacks)
      2. Reject path traversal (..) components
      3. Resolve path and check it's relative to allowed_root

    Args:
        source: The source path to validate.
        allowed_root: The allowed root directory (must be absolute).

    Returns:
        PathValidationResult with valid=True if safe, or reason if rejected.
    """
    # Reject symlinks (check the path itself, not the resolved target)
    if source.is_symlink():
        return PathValidationResult(
            valid=False,
            reason="Symlinks are not allowed for skill installation.",
        )

    # Reject path traversal (..) in any component
    if ".." in source.parts:
        return PathValidationResult(
            valid=False,
            reason="Path traversal (..) is not allowed.",
        )

    # Resolve to absolute and check boundary
    try:
        resolved = source.resolve(strict=False)
    except (OSError, ValueError) as e:
        return PathValidationResult(
            valid=False,
            reason=f"Cannot resolve path: {e}",
        )

    resolved_root = allowed_root.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        return PathValidationResult(
            valid=False,
            reason=f"Path is outside the allowed directory ({resolved_root}).",
        )

    return PathValidationResult(valid=True)


# ──────────────────────────────────────────────────────────────
# Skill Installer Service
# ──────────────────────────────────────────────────────────────


class SkillInstaller:
    """Application service for manual/offline skill installation.

    Accepts a skill contract JSON file from disk or Telegram upload,
    validates it through the full safety pipeline, and persists to ContractStore.

    One Safety Gate: uses the SAME PrivacyPipeline and ContractValidator
    as LearnFlowService. No bypass via install.

    Args:
        contract_store: Persistent contract storage.
        privacy_pipeline: Privacy pipeline for safety checks.
    """

    def __init__(
        self,
        contract_store: ContractStore,
        privacy_pipeline: PrivacyPipeline,
    ) -> None:
        self._contract_store = contract_store
        self._privacy = privacy_pipeline

    def install_from_file(
        self,
        file_path: Path,
        allowed_root: Path,
        user_id: int,
    ) -> InstallResult:
        """Install a skill from a local JSON file.

        Full pipeline: path validation -> read -> deserialize -> safety -> persist.

        Args:
            file_path: Path to the skill contract JSON file.
            allowed_root: The allowed root directory for this source.
            user_id: The installing user's Telegram ID.

        Returns:
            InstallResult with success/error details.
        """
        # Path-Boundary validation
        path_result = validate_source_path(file_path, allowed_root)
        if not path_result.valid:
            log.warning(
                "Install rejected (path boundary): user=%d reason=%s",
                user_id,
                path_result.reason,
            )
            return InstallResult(success=False, error=path_result.reason)

        # Read file
        try:
            raw_json = file_path.read_text(encoding="utf-8")
        except OSError as e:
            return InstallResult(
                success=False,
                error=f"Cannot read file: {e}",
            )

        return self._install_from_json(raw_json, user_id, source_label=str(file_path))

    def install_from_json_string(
        self,
        raw_json: str,
        user_id: int,
    ) -> InstallResult:
        """Install a skill from a raw JSON string (e.g. Telegram upload content).

        Full pipeline: deserialize -> safety -> persist.

        Args:
            raw_json: JSON string of the skill contract.
            user_id: The installing user's Telegram ID.

        Returns:
            InstallResult with success/error details.
        """
        return self._install_from_json(
            raw_json, user_id, source_label="telegram_upload"
        )

    def _install_from_json(
        self,
        raw_json: str,
        user_id: int,
        source_label: str,
    ) -> InstallResult:
        """Internal: deserialize, validate, safety-check, persist.

        One Safety Gate: PrivacyPipeline runs on canonical claim form.
        ContractValidator runs all V1-V17 rules.

        Args:
            raw_json: JSON string of the skill contract.
            user_id: The installing user's Telegram ID.
            source_label: Label for logging (file path or 'telegram_upload').

        Returns:
            InstallResult with success/error details.
        """
        # Deserialize (with size and depth limits)
        try:
            data = safe_json_load(raw_json, max_bytes=10 * 1024 * 1024, max_depth=64)
        except (ValueError,) as e:
            return InstallResult(
                success=False,
                error=f"Invalid JSON: {e}",
            )

        if not isinstance(data, dict):
            return InstallResult(
                success=False,
                error="Skill contract must be a JSON object.",
            )

        # Build SkillContract from dict
        try:
            contract = SkillContract.from_dict(data)
        except Exception as e:
            return InstallResult(
                success=False,
                error=f"Invalid skill contract schema: {e}",
            )

        # Override origin to manual_install (regardless of what the file says)
        from dataclasses import replace

        contract = replace(
            contract,
            origin="manual_install",
            updated_at=now_iso(),
        )

        # Quick structural check: instruction must not be empty for safety pipeline
        if (
            not contract.execution.instruction
            or not contract.execution.instruction.strip()
        ):
            return InstallResult(
                success=False,
                error="Skill contract must have a non-empty execution.instruction.",
            )

        # ONE SAFETY GATE: PrivacyPipeline check (same as /learn)
        safety_rejection = self._check_safety(contract, user_id)
        if safety_rejection is not None:
            log.info(
                "Install rejected (safety): user=%d source=%s",
                user_id,
                source_label,
            )
            return InstallResult(success=False, error=safety_rejection)

        # Persist (ContractStore handles finalize + validate V1-V17 + checksum)
        try:
            saved = self._contract_store.persist(contract, user_id=user_id)
        except ContractStoreError as e:
            log.error(
                "Install persist failed: user=%d source=%s error=%s",
                user_id,
                source_label,
                str(e),
            )
            return InstallResult(success=False, error=str(e))

        log.info(
            "Skill installed: contract=%s name='%s' user=%d source=%s",
            saved.id,
            saved.name,
            user_id,
            source_label,
        )
        return InstallResult(success=True, contract_name=saved.name)

    def _check_safety(
        self,
        contract: SkillContract,
        user_id: int,
    ) -> Optional[str]:
        """Run full PrivacyPipeline on ALL user-controlled contract fields.

        Uses iter_user_text_fields() to ensure every text field is scanned.
        Shared with LearnFlowService._validate_contract_safety() via the
        same iter_user_text_fields() function (One Safety Gate, GAP-14 fix).

        Args:
            contract: The contract to check.
            user_id: The user installing the skill.

        Returns:
            None if safe, rejection reason string if blocked.
        """
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )
        from datetime import datetime, timezone
        from uuid import uuid4

        now_str = datetime.now(timezone.utc).isoformat()

        # Scan ALL user-controlled text fields (GAP-14 fix).
        # iter_user_text_fields returns (label, value) for name, ALL phrases,
        # instruction, tags, intent label/examples.
        all_text_parts = [value for _label, value in iter_user_text_fields(contract)]
        combined_text = " | ".join(all_text_parts) if all_text_parts else ""

        if not combined_text.strip():
            return "Contract has no scannable text content."

        temp_hyp = Hypothesis(
            hypothesis_id=f"hyp_{uuid4().hex[:16]}",
            user_id=user_id,
            type="preference",
            scope=HypothesisScope(),
            claim=combined_text,
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            elo_games_played=0,
            bayes_confidence=0.5,
            support_count=1,
            contradict_count=0,
            source_type="install_command",
            decay_immune=True,
            created_at=now_str,
            last_applied=None,
            last_seen=now_str,
        )

        rejection = self._privacy.check(temp_hyp)
        if rejection is not None:
            return rejection.reason

        return None
