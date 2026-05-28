"""Capability Broker: runtime permission enforcement via capability tokens.

Phase 1: Stub implementation with default deny.
Later (KI Store): real token-based enforcement with TTL and crypto.

Architecture model: WebExtensions Capability Model + Microsoft Wassette.
Principle: a skill receives ONLY the capabilities declared in its contract.

The broker creates a CapabilityToken from a SkillContract's permissions,
then checks access requests against that token. The token is a simple
dataclass in Phase 1; later it will be unforgeable with TTL + crypto.

Dependencies: Python stdlib only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from application.skill_compression.skill_contract import (
    SkillContract,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Capability Token
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CapabilityToken:
    """Encapsulates the allowed actions of a skill for one execution.

    Phase 1: plain dataclass (no crypto, no TTL).
    KI Store phase: unforgeable token with expiry and signature.

    Default-deny invariant:
      enabled=True + empty allowlist => DENY concrete access.
      Only explicit allowlist entries grant access.
      "*" in an allowlist => explicit all-access.

    Attributes:
        skill_id: The skill contract's ID.
        allowed_tools: Tuple of tool names the skill may invoke.
        memory_read: Tuple of store names the skill may read.
        memory_write: Tuple of store names the skill may write.
        network_access: Whether the skill declares network access intent.
        allowed_domains: Tuple of allowed network domains (empty = deny all).
        file_access: Whether the skill declares file access intent.
        file_scopes: Tuple of allowed file scopes (empty = deny all).
        history_access: Whether the skill declares history access intent.
        history_scopes: Tuple of allowed history scopes (empty = deny all).
        secrets_access: Whether the skill may access secrets.
    """

    skill_id: str
    allowed_tools: tuple[str, ...] = ()
    memory_read: tuple[str, ...] = ()
    memory_write: tuple[str, ...] = ()
    network_access: bool = False
    allowed_domains: tuple[str, ...] = ()
    file_access: bool = False
    file_scopes: tuple[str, ...] = ()
    history_access: bool = False
    history_scopes: tuple[str, ...] = ()
    secrets_access: bool = False


# ──────────────────────────────────────────────────────────────
# Access check result
# ──────────────────────────────────────────────────────────────


class AccessDecision(Enum):
    """Outcome of a capability check."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class AccessCheckResult:
    """Result of a capability broker access check.

    Attributes:
        decision: ALLOW or DENY.
        reason: Human-readable explanation (empty on ALLOW).
    """

    decision: AccessDecision
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == AccessDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == AccessDecision.DENY


# ──────────────────────────────────────────────────────────────
# Resource categories
# ──────────────────────────────────────────────────────────────

KNOWN_RESOURCE_TYPES = frozenset(
    {
        "tool",
        "memory_read",
        "memory_write",
        "network",
        "file",
        "history",
        "secrets",
    }
)


# ──────────────────────────────────────────────────────────────
# Capability Broker
# ──────────────────────────────────────────────────────────────


class CapabilityBroker:
    """Runtime permission enforcement via capability tokens.

    Phase 1: stub with real enforcement logic but no crypto/TTL.
    Default deny: if a resource is not explicitly declared, access is denied.

    Usage:
        broker = CapabilityBroker()
        token = broker.create_token(contract)
        result = broker.check_access(token, "tool", "web_search")
        if result.denied:
            raise PermissionError(result.reason)
    """

    @staticmethod
    def create_token(contract: SkillContract) -> CapabilityToken:
        """Create a capability token from a skill contract's permissions.

        The token captures the contract's declared permissions at the
        moment of creation. It is used for all subsequent access checks
        during this execution.

        Args:
            contract: The skill contract.

        Returns:
            A CapabilityToken with the contract's declared capabilities.
        """
        perms = contract.permissions
        return CapabilityToken(
            skill_id=contract.id,
            allowed_tools=tuple(perms.tools),
            memory_read=tuple(perms.memory_read),
            memory_write=tuple(perms.memory_write),
            network_access=perms.network_access.enabled,
            allowed_domains=tuple(perms.network_access.domains),
            file_access=perms.file_access.enabled,
            file_scopes=tuple(perms.file_access.scopes),
            history_access=perms.history_access.enabled,
            history_scopes=tuple(perms.history_access.scopes),
            secrets_access=perms.secrets_access,
        )

    @staticmethod
    def check_access(
        token: CapabilityToken,
        resource_type: str,
        resource_name: Optional[str] = None,
    ) -> AccessCheckResult:
        """Check if a capability token allows access to a resource.

        Default: DENY. Only explicitly declared accesses are allowed.

        Args:
            token: The capability token for this execution.
            resource_type: One of the KNOWN_RESOURCE_TYPES.
            resource_name: Specific resource (e.g., tool name, store name, domain).

        Returns:
            AccessCheckResult with decision and reason.
        """
        if resource_type not in KNOWN_RESOURCE_TYPES:
            return AccessCheckResult(
                decision=AccessDecision.DENY,
                reason=f"Unknown resource type '{resource_type}'.",
            )

        if resource_type == "tool":
            if resource_name and resource_name in token.allowed_tools:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            return AccessCheckResult(
                decision=AccessDecision.DENY,
                reason="Tool access is not declared.",
            )

        if resource_type == "memory_read":
            if resource_name and resource_name in token.memory_read:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            return AccessCheckResult(
                decision=AccessDecision.DENY,
                reason="Memory read access is not declared.",
            )

        if resource_type == "memory_write":
            if resource_name and resource_name in token.memory_write:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            return AccessCheckResult(
                decision=AccessDecision.DENY,
                reason="Memory write access is not declared.",
            )

        if resource_type == "network":
            if not token.network_access:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="Network access is not declared.",
                )
            # No concrete target => deny
            if not resource_name:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="Network access requires a concrete domain.",
                )
            # Empty domains list => deny (enabled alone is not enough)
            if not token.allowed_domains:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="Network access is not declared.",
                )
            # Wildcard => allow all
            if "*" in token.allowed_domains:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            # Check specific domain
            if resource_name in token.allowed_domains:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            return AccessCheckResult(
                decision=AccessDecision.DENY,
                reason="Network access is not declared.",
            )

        if resource_type == "file":
            if not token.file_access:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="File access is not declared.",
                )
            # No concrete target => deny
            if not resource_name:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="File access requires a concrete scope.",
                )
            # Empty scopes list => deny (enabled alone is not enough)
            if not token.file_scopes:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="File access is not declared.",
                )
            # Wildcard => allow all
            if "*" in token.file_scopes:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            # Check specific scope
            if resource_name in token.file_scopes:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            return AccessCheckResult(
                decision=AccessDecision.DENY,
                reason="File access is not declared.",
            )

        if resource_type == "history":
            if not token.history_access:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="History access is not declared.",
                )
            # No concrete target => deny
            if not resource_name:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="History access requires a concrete scope.",
                )
            # Empty scopes list => deny (enabled alone is not enough)
            if not token.history_scopes:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="History access is not declared.",
                )
            # Wildcard => allow all
            if "*" in token.history_scopes:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            # Check specific scope
            if resource_name in token.history_scopes:
                return AccessCheckResult(decision=AccessDecision.ALLOW)
            return AccessCheckResult(
                decision=AccessDecision.DENY,
                reason="History access is not declared.",
            )

        if resource_type == "secrets":
            if not token.secrets_access:
                return AccessCheckResult(
                    decision=AccessDecision.DENY,
                    reason="Secrets access is not declared.",
                )
            return AccessCheckResult(decision=AccessDecision.ALLOW)

        # Fallback: deny (should be unreachable given KNOWN_RESOURCE_TYPES check)
        return AccessCheckResult(
            decision=AccessDecision.DENY,
            reason=f"No handler for resource type '{resource_type}'.",
        )
