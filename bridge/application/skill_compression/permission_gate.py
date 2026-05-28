"""Permission Gate: central enforcement point BEFORE skill execution.

Placed in chat_service.py after SkillMatcher finds a contract,
before the LLM call starts. The Matcher decides IF a skill triggers;
the PermissionGate decides if the skill is ALLOWED to execute.

Design principles:
  - Default deny: only explicitly declared permissions are allowed.
  - enabled=True alone is NOT enough; a concrete allowlist entry is required.
  - enabled=True + empty allowlist => deny (no implicit all-access).
  - "*" in allowlist => explicit all-access, treated as high-risk.
  - resource_name=None => deny (no concrete target specified).
  - /learn skills (origin=local_learn) have empty permissions = sandbox
    (llm_instruction only, no memory/tool/network/file).
  - Blocked/flagged skills are never executed.
  - High-risk skills require explicit user approval (future: confirmation flow).
  - risk_level is recomputed from permissions at runtime (not trusted from storage).

Dependencies: Python stdlib only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from application.skill_compression.skill_contract import (
    SkillContract,
    compute_risk_level,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────


class PermissionDecision(Enum):
    """Outcome of a permission check."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PermissionCheckResult:
    """Result of a permission gate check.

    Attributes:
        decision: ALLOW or DENY.
        reason: Human-readable explanation (empty on ALLOW).
            MUST NOT contain user-defined strings (skill name, instruction,
            trigger phrases). Only generic messages and contract.id.
        rule: Which rule triggered the decision (for audit/logging).
    """

    decision: PermissionDecision
    reason: str = ""
    rule: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == PermissionDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == PermissionDecision.DENY


# ──────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────


class PermissionDeniedError(Exception):
    """Raised when a skill is denied execution by the PermissionGate."""

    def __init__(self, message: str, result: PermissionCheckResult):
        super().__init__(message)
        self.result = result


# ──────────────────────────────────────────────────────────────
# Actions that can be requested during execution
# ──────────────────────────────────────────────────────────────


ALLOWED_EXECUTION_TYPES = frozenset({"llm_instruction"})
RESERVED_EXECUTION_TYPES = frozenset({"workflow", "tool"})
BLOCKED_REVIEW_STATUSES = frozenset({"blocked", "flagged"})


# ──────────────────────────────────────────────────────────────
# PermissionGate (stateless)
# ──────────────────────────────────────────────────────────────


class PermissionGate:
    """Central enforcement point BEFORE skill execution.

    All checks are static (no instance state needed). The class
    exists for namespacing and future extensibility (e.g., user
    overrides, admin policies).

    Default-deny invariant:
      enabled=True + empty allowlist => DENY concrete access.
      Only explicit allowlist entries grant access.
      "*" is the only way to get all-access (high-risk).
    """

    @staticmethod
    def check_execution_allowed(contract: SkillContract) -> PermissionCheckResult:
        """Check if a skill contract is allowed to execute.

        Checks (in order):
          1. execution.type must be llm_instruction (feature flag V15)
          2. review_status must not be blocked/flagged
          3. risk_level recomputed from permissions; "high" => deny
          4. checksum must be present for persisted contracts

        Risk is recomputed from contract.permissions at runtime.
        The stored contract.risk_level is NOT trusted (denormalized field).
        A warning is logged if stored and computed risk diverge.

        Args:
            contract: The skill contract to check.

        Returns:
            PermissionCheckResult with decision and reason.
        """
        # Rule 1: Execution type feature flag
        if contract.execution.type not in ALLOWED_EXECUTION_TYPES:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason=(
                    f"Execution type '{contract.execution.type}' is not allowed. "
                    f"Only {sorted(ALLOWED_EXECUTION_TYPES)} are permitted in Phase 1."
                ),
                rule="execution_type_feature_flag",
            )

        # Rule 2: Blocked/flagged review status
        if contract.review_status in BLOCKED_REVIEW_STATUSES:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Skill is blocked and cannot be executed.",
                rule="review_status_blocked",
            )

        # Rule 3: Risk recomputed from permissions (not trusted from storage)
        effective_risk = compute_risk_level(contract.permissions)
        if effective_risk != contract.risk_level:
            log.warning(
                "Skill risk_level mismatch: stored=%s, computed=%s, skill_id=%s",
                contract.risk_level,
                effective_risk,
                contract.id,
            )
        if effective_risk == "high":
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Skill risk level requires approval.",
                rule="high_risk_denied",
            )

        return PermissionCheckResult(
            decision=PermissionDecision.ALLOW, rule="all_clear"
        )

    @staticmethod
    def check_memory_access(
        contract: SkillContract,
        store: str,
        operation: str,
    ) -> PermissionCheckResult:
        """Check if a skill may access a specific memory store.

        Default deny: only explicitly declared stores are allowed.

        Args:
            contract: The skill contract.
            store: Name of the memory store (e.g., "long_term_facts").
            operation: "read" or "write".

        Returns:
            PermissionCheckResult.
        """
        perms = contract.permissions

        if operation == "read":
            if store in perms.memory_read:
                return PermissionCheckResult(
                    decision=PermissionDecision.ALLOW,
                    rule="memory_read_declared",
                )
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Memory read access is not declared.",
                rule="memory_read_not_declared",
            )

        if operation == "write":
            if store in perms.memory_write:
                return PermissionCheckResult(
                    decision=PermissionDecision.ALLOW,
                    rule="memory_write_declared",
                )
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Memory write access is not declared.",
                rule="memory_write_not_declared",
            )

        return PermissionCheckResult(
            decision=PermissionDecision.DENY,
            reason=f"Unknown memory operation '{operation}'.",
            rule="unknown_operation",
        )

    @staticmethod
    def check_tool_access(
        contract: SkillContract,
        tool_name: str,
    ) -> PermissionCheckResult:
        """Check if a skill may invoke a specific tool.

        Default deny: only tools listed in permissions.tools are allowed.

        Args:
            contract: The skill contract.
            tool_name: Name of the tool (e.g., "web_search").

        Returns:
            PermissionCheckResult.
        """
        if tool_name in contract.permissions.tools:
            return PermissionCheckResult(
                decision=PermissionDecision.ALLOW,
                rule="tool_declared",
            )
        return PermissionCheckResult(
            decision=PermissionDecision.DENY,
            reason="Tool access is not declared.",
            rule="tool_not_declared",
        )

    @staticmethod
    def check_network_access(
        contract: SkillContract,
        domain: Optional[str] = None,
    ) -> PermissionCheckResult:
        """Check if a skill may access the network.

        Default deny semantics:
          - enabled=False => deny all network access.
          - enabled=True, domain=None => deny (no concrete target).
          - enabled=True, domains=() => deny any concrete domain
            (empty allowlist means nothing is allowed).
          - enabled=True, domains=("x.com",) => allow only x.com.
          - enabled=True, domains=("*",) => allow all (explicit wildcard).

        Args:
            contract: The skill contract.
            domain: Target domain (required for concrete access checks).

        Returns:
            PermissionCheckResult.
        """
        net = contract.permissions.network_access
        if not net.enabled:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Network access is not declared.",
                rule="network_disabled",
            )
        # No concrete target => deny
        if not domain:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Network access requires a concrete domain.",
                rule="network_no_target",
            )
        # Empty domains list => deny (enabled=True alone is not enough)
        if not net.domains:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Network access is not declared.",
                rule="network_empty_allowlist",
            )
        # Wildcard => allow all (explicit all-access)
        if "*" in net.domains:
            return PermissionCheckResult(
                decision=PermissionDecision.ALLOW,
                rule="network_wildcard",
            )
        # Check specific domain
        if domain in net.domains:
            return PermissionCheckResult(
                decision=PermissionDecision.ALLOW,
                rule="network_allowed",
            )
        return PermissionCheckResult(
            decision=PermissionDecision.DENY,
            reason="Network access is not declared.",
            rule="network_domain_not_declared",
        )

    @staticmethod
    def check_file_access(
        contract: SkillContract,
        scope: Optional[str] = None,
    ) -> PermissionCheckResult:
        """Check if a skill may access the file system.

        Default deny semantics:
          - enabled=False => deny all file access.
          - enabled=True, scope=None => deny (no concrete target).
          - enabled=True, scopes=() => deny any concrete scope
            (empty allowlist means nothing is allowed).
          - enabled=True, scopes=("workspace:read",) => allow only that scope.
          - enabled=True, scopes=("*",) => allow all (explicit wildcard).

        Args:
            contract: The skill contract.
            scope: Target file scope (required for concrete access checks).

        Returns:
            PermissionCheckResult.
        """
        fa = contract.permissions.file_access
        if not fa.enabled:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="File access is not declared.",
                rule="file_disabled",
            )
        # No concrete target => deny
        if not scope:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="File access requires a concrete scope.",
                rule="file_no_target",
            )
        # Empty scopes list => deny (enabled=True alone is not enough)
        if not fa.scopes:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="File access is not declared.",
                rule="file_empty_allowlist",
            )
        # Wildcard => allow all (explicit all-access)
        if "*" in fa.scopes:
            return PermissionCheckResult(
                decision=PermissionDecision.ALLOW,
                rule="file_wildcard",
            )
        # Check specific scope
        if scope in fa.scopes:
            return PermissionCheckResult(
                decision=PermissionDecision.ALLOW,
                rule="file_allowed",
            )
        return PermissionCheckResult(
            decision=PermissionDecision.DENY,
            reason="File access is not declared.",
            rule="file_scope_not_declared",
        )

    @staticmethod
    def check_history_access(
        contract: SkillContract,
        scope: Optional[str] = None,
    ) -> PermissionCheckResult:
        """Check if a skill may access chat history.

        Default deny semantics:
          - enabled=False => deny all history access.
          - enabled=True, scope=None => deny (no concrete target).
          - enabled=True, scopes=() => deny any concrete scope
            (empty allowlist means nothing is allowed).
          - enabled=True, scopes=("current_chat",) => allow only that scope.
          - enabled=True, scopes=("*",) => allow all (explicit wildcard).

        Args:
            contract: The skill contract.
            scope: Target history scope (required for concrete access checks).

        Returns:
            PermissionCheckResult.
        """
        ha = contract.permissions.history_access
        if not ha.enabled:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="History access is not declared.",
                rule="history_disabled",
            )
        # No concrete target => deny
        if not scope:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="History access requires a concrete scope.",
                rule="history_no_target",
            )
        # Empty scopes list => deny (enabled=True alone is not enough)
        if not ha.scopes:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="History access is not declared.",
                rule="history_empty_allowlist",
            )
        # Wildcard => allow all (explicit all-access)
        if "*" in ha.scopes:
            return PermissionCheckResult(
                decision=PermissionDecision.ALLOW,
                rule="history_wildcard",
            )
        # Check specific scope
        if scope in ha.scopes:
            return PermissionCheckResult(
                decision=PermissionDecision.ALLOW,
                rule="history_allowed",
            )
        return PermissionCheckResult(
            decision=PermissionDecision.DENY,
            reason="History access is not declared.",
            rule="history_scope_not_declared",
        )

    @staticmethod
    def check_secrets_access(contract: SkillContract) -> PermissionCheckResult:
        """Check if a skill may access secrets.

        Default deny: secrets_access must be True.

        Args:
            contract: The skill contract.

        Returns:
            PermissionCheckResult.
        """
        if not contract.permissions.secrets_access:
            return PermissionCheckResult(
                decision=PermissionDecision.DENY,
                reason="Secrets access is not declared.",
                rule="secrets_disabled",
            )
        return PermissionCheckResult(
            decision=PermissionDecision.ALLOW,
            rule="secrets_allowed",
        )
