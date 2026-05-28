"""T5 Tests: CapabilityBroker enforcement (4-path: Happy + Malicious + Rejection + Privacy).

Coverage:
  U15: Capability broker default deny for all
  Token creation from contract permissions (including file_scopes, history_scopes)
  Access checks for all resource types (tool, memory_read, memory_write,
    network, file, history, secrets)

  Default-deny matrix tests for ALL allowlist-based permissions:
    network (allowed_domains), file (file_scopes), history (history_scopes)
  Each matrix tests 6+ cases:
    enabled=False => DENY
    enabled=True, allowlist=() => DENY (CRITICAL)
    enabled=True, allowlist=(X,) => ALLOW X
    enabled=True, allowlist=(X,) => DENY Y
    enabled=True, allowlist=("*",) => ALLOW any
    enabled=True, resource_name=None => DENY
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from application.skill_compression.capability_broker import (
    CapabilityBroker,
    CapabilityToken,
)
from application.skill_compression.skill_contract import (
    FileAccessConfig,
    HistoryAccessConfig,
    NetworkAccessConfig,
    PermissionsConfig,
    SkillContract,
    create_minimal_contract,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def broker() -> CapabilityBroker:
    return CapabilityBroker()


@pytest.fixture
def empty_contract() -> SkillContract:
    """Skill with default (empty) permissions."""
    return create_minimal_contract(
        name="empty-perms",
        phrases=("test",),
        instruction="test instruction",
    )


@pytest.fixture
def full_contract() -> SkillContract:
    """Skill with multiple declared permissions."""
    c = create_minimal_contract(
        name="full-perms",
        phrases=("full",),
        instruction="full test",
    )
    return replace(
        c,
        permissions=PermissionsConfig(
            tools=("web_search", "calculator"),
            memory_read=("long_term_facts", "history_summaries"),
            memory_write=("long_term_facts",),
            network_access=NetworkAccessConfig(
                enabled=True,
                domains=("api.example.com",),
            ),
            file_access=FileAccessConfig(
                enabled=True,
                scopes=("workspace:read",),
            ),
            history_access=HistoryAccessConfig(
                enabled=True,
                scopes=("current_chat",),
            ),
            secrets_access=True,
        ),
    )


# ------------------------------------------------------------------
# Token creation
# ------------------------------------------------------------------


class TestTokenCreation:
    """Token correctly captures contract permissions."""

    def test_empty_permissions_create_empty_token(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        assert token.skill_id == empty_contract.id
        assert token.allowed_tools == ()
        assert token.memory_read == ()
        assert token.memory_write == ()
        assert token.network_access is False
        assert token.allowed_domains == ()
        assert token.file_access is False
        assert token.file_scopes == ()
        assert token.history_access is False
        assert token.history_scopes == ()
        assert token.secrets_access is False

    def test_full_permissions_create_full_token(self, broker, full_contract):
        token = broker.create_token(full_contract)
        assert token.skill_id == full_contract.id
        assert "web_search" in token.allowed_tools
        assert "calculator" in token.allowed_tools
        assert "long_term_facts" in token.memory_read
        assert "history_summaries" in token.memory_read
        assert "long_term_facts" in token.memory_write
        assert token.network_access is True
        assert "api.example.com" in token.allowed_domains
        assert token.file_access is True
        assert "workspace:read" in token.file_scopes
        assert token.history_access is True
        assert "current_chat" in token.history_scopes
        assert token.secrets_access is True


# ------------------------------------------------------------------
# Happy path: declared access allowed
# ------------------------------------------------------------------


class TestCapabilityBrokerHappy:
    """Declared capabilities are allowed through the broker."""

    def test_declared_tool_allowed(self, broker, full_contract):
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "tool", "web_search")
        assert result.allowed

    def test_declared_memory_read_allowed(self, broker, full_contract):
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "memory_read", "long_term_facts")
        assert result.allowed

    def test_declared_memory_write_allowed(self, broker, full_contract):
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "memory_write", "long_term_facts")
        assert result.allowed

    def test_declared_network_allowed(self, broker, full_contract):
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "network", "api.example.com")
        assert result.allowed

    def test_declared_file_scope_allowed(self, broker, full_contract):
        """File access with matching scope is allowed."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "file", "workspace:read")
        assert result.allowed

    def test_declared_history_scope_allowed(self, broker, full_contract):
        """History access with matching scope is allowed."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "history", "current_chat")
        assert result.allowed

    def test_secrets_access_allowed(self, broker, full_contract):
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "secrets")
        assert result.allowed


# ------------------------------------------------------------------
# Default deny: empty token denies everything
# ------------------------------------------------------------------


class TestCapabilityBrokerDefaultDeny:
    """U15: Empty permissions token denies all access (default deny)."""

    def test_tool_denied(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "tool", "web_search")
        assert result.denied

    def test_memory_read_denied(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "memory_read", "long_term_facts")
        assert result.denied

    def test_memory_write_denied(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "memory_write", "long_term_facts")
        assert result.denied

    def test_network_denied(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "network", "api.example.com")
        assert result.denied

    def test_file_denied(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "file", "workspace:read")
        assert result.denied

    def test_history_denied(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "history", "current_chat")
        assert result.denied

    def test_secrets_denied(self, broker, empty_contract):
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "secrets")
        assert result.denied

    def test_unknown_resource_type_denied(self, broker, empty_contract):
        """Unknown resource type is always denied."""
        token = broker.create_token(empty_contract)
        result = broker.check_access(token, "quantum_entanglement", "qubit_1")
        assert result.denied
        assert "Unknown resource type" in result.reason


# ------------------------------------------------------------------
# Malicious path: escalation attempts
# ------------------------------------------------------------------


class TestCapabilityBrokerMalicious:
    """Attempts to access resources beyond declared permissions."""

    def test_undeclared_tool_denied(self, broker, full_contract):
        """Token with tools=[web_search, calculator] denies shell_exec."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "tool", "shell_exec")
        assert result.denied

    def test_undeclared_memory_store_denied(self, broker, full_contract):
        """Token with memory_read=[long_term_facts] denies secrets store."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "memory_read", "secrets")
        assert result.denied

    def test_undeclared_memory_write_store_denied(self, broker, full_contract):
        """Token with memory_write=[long_term_facts] denies other stores."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "memory_write", "history_summaries")
        assert result.denied

    def test_undeclared_network_domain_denied(self, broker, full_contract):
        """Token with domains=[api.example.com] denies evil.com."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "network", "evil.com")
        assert result.denied

    def test_undeclared_file_scope_denied(self, broker, full_contract):
        """Token with file_scopes=[workspace:read] denies home:read."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "file", "home:read")
        assert result.denied

    def test_undeclared_history_scope_denied(self, broker, full_contract):
        """Token with history_scopes=[current_chat] denies all_chats."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "history", "all_chats")
        assert result.denied

    def test_tool_none_resource_name_denied(self, broker, full_contract):
        """Tool check with None resource_name is denied (no tool specified)."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "tool", None)
        assert result.denied

    def test_network_none_resource_name_denied(self, broker, full_contract):
        """Network check with None resource_name is denied."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "network", None)
        assert result.denied

    def test_file_none_resource_name_denied(self, broker, full_contract):
        """File check with None resource_name is denied."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "file", None)
        assert result.denied

    def test_history_none_resource_name_denied(self, broker, full_contract):
        """History check with None resource_name is denied."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "history", None)
        assert result.denied


# ------------------------------------------------------------------
# Privacy path: denial reasons do not leak sensitive info
# ------------------------------------------------------------------


class TestCapabilityBrokerPrivacy:
    """Denial reasons must not leak other declared permissions."""

    def test_tool_denial_does_not_leak_allowed_tools(self, broker, full_contract):
        """Denying shell_exec should not reveal web_search or calculator."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "tool", "shell_exec")
        assert result.denied
        assert "web_search" not in result.reason
        assert "calculator" not in result.reason

    def test_memory_denial_does_not_leak_allowed_stores(self, broker, full_contract):
        """Denying secrets should not reveal long_term_facts."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "memory_read", "secrets")
        assert result.denied
        assert "long_term_facts" not in result.reason

    def test_network_denial_does_not_leak_allowed_domains(self, broker, full_contract):
        """Denying evil.com should not reveal api.example.com."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "network", "evil.com")
        assert result.denied
        assert "api.example.com" not in result.reason

    def test_file_denial_does_not_leak_allowed_scopes(self, broker, full_contract):
        """Denying home:read should not reveal workspace:read."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "file", "home:read")
        assert result.denied
        assert "workspace:read" not in result.reason

    def test_history_denial_does_not_leak_allowed_scopes(self, broker, full_contract):
        """Denying all_chats should not reveal current_chat."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "history", "all_chats")
        assert result.denied
        assert "current_chat" not in result.reason

    def test_denial_does_not_leak_skill_id(self, broker, full_contract):
        """Denial reasons should not contain the skill_id."""
        token = broker.create_token(full_contract)
        result = broker.check_access(token, "tool", "shell_exec")
        assert result.denied
        assert token.skill_id not in result.reason


# ------------------------------------------------------------------
# NETWORK MATRIX: CapabilityBroker default-deny semantics
# ------------------------------------------------------------------


class TestBrokerNetworkMatrix:
    """Permission matrix for network via CapabilityBroker."""

    def test_disabled_denies(self, broker):
        """network_access=False => DENY."""
        token = CapabilityToken(skill_id="t1", network_access=False)
        assert broker.check_access(token, "network", "api.example.com").denied

    def test_enabled_empty_domains_denies(self, broker):
        """CRITICAL: network_access=True, allowed_domains=() => DENY."""
        token = CapabilityToken(skill_id="t2", network_access=True, allowed_domains=())
        assert broker.check_access(token, "network", "api.example.com").denied

    def test_enabled_specific_domain_allows(self, broker):
        """network_access=True, domains=(api.example.com,) => ALLOW."""
        token = CapabilityToken(
            skill_id="t3",
            network_access=True,
            allowed_domains=("api.example.com",),
        )
        assert broker.check_access(token, "network", "api.example.com").allowed

    def test_enabled_specific_domain_denies_other(self, broker):
        """network_access=True, domains=(api.example.com,) => DENY evil.com."""
        token = CapabilityToken(
            skill_id="t4",
            network_access=True,
            allowed_domains=("api.example.com",),
        )
        assert broker.check_access(token, "network", "evil.com").denied

    def test_enabled_wildcard_allows_any(self, broker):
        """network_access=True, domains=("*",) => ALLOW any."""
        token = CapabilityToken(
            skill_id="t5",
            network_access=True,
            allowed_domains=("*",),
        )
        assert broker.check_access(token, "network", "evil.com").allowed

    def test_enabled_none_resource_denies(self, broker):
        """network_access=True, resource_name=None => DENY."""
        token = CapabilityToken(
            skill_id="t6",
            network_access=True,
            allowed_domains=("api.example.com",),
        )
        assert broker.check_access(token, "network", None).denied


# ------------------------------------------------------------------
# FILE MATRIX: CapabilityBroker default-deny semantics
# ------------------------------------------------------------------


class TestBrokerFileMatrix:
    """Permission matrix for file via CapabilityBroker."""

    def test_disabled_denies(self, broker):
        """file_access=False => DENY."""
        token = CapabilityToken(skill_id="f1", file_access=False)
        assert broker.check_access(token, "file", "workspace:read").denied

    def test_enabled_empty_scopes_denies(self, broker):
        """CRITICAL: file_access=True, file_scopes=() => DENY."""
        token = CapabilityToken(skill_id="f2", file_access=True, file_scopes=())
        assert broker.check_access(token, "file", "workspace:read").denied

    def test_enabled_specific_scope_allows(self, broker):
        """file_access=True, file_scopes=(workspace:read,) => ALLOW."""
        token = CapabilityToken(
            skill_id="f3",
            file_access=True,
            file_scopes=("workspace:read",),
        )
        assert broker.check_access(token, "file", "workspace:read").allowed

    def test_enabled_specific_scope_denies_other(self, broker):
        """file_access=True, file_scopes=(workspace:read,) => DENY home:read."""
        token = CapabilityToken(
            skill_id="f4",
            file_access=True,
            file_scopes=("workspace:read",),
        )
        assert broker.check_access(token, "file", "home:read").denied

    def test_enabled_wildcard_allows_any(self, broker):
        """file_access=True, file_scopes=("*",) => ALLOW any."""
        token = CapabilityToken(
            skill_id="f5",
            file_access=True,
            file_scopes=("*",),
        )
        assert broker.check_access(token, "file", "home:read").allowed

    def test_enabled_none_resource_denies(self, broker):
        """file_access=True, resource_name=None => DENY."""
        token = CapabilityToken(
            skill_id="f6",
            file_access=True,
            file_scopes=("workspace:read",),
        )
        assert broker.check_access(token, "file", None).denied


# ------------------------------------------------------------------
# HISTORY MATRIX: CapabilityBroker default-deny semantics
# ------------------------------------------------------------------


class TestBrokerHistoryMatrix:
    """Permission matrix for history via CapabilityBroker."""

    def test_disabled_denies(self, broker):
        """history_access=False => DENY."""
        token = CapabilityToken(skill_id="h1", history_access=False)
        assert broker.check_access(token, "history", "current_chat").denied

    def test_enabled_empty_scopes_denies(self, broker):
        """CRITICAL: history_access=True, history_scopes=() => DENY."""
        token = CapabilityToken(skill_id="h2", history_access=True, history_scopes=())
        assert broker.check_access(token, "history", "current_chat").denied

    def test_enabled_specific_scope_allows(self, broker):
        """history_access=True, history_scopes=(current_chat,) => ALLOW."""
        token = CapabilityToken(
            skill_id="h3",
            history_access=True,
            history_scopes=("current_chat",),
        )
        assert broker.check_access(token, "history", "current_chat").allowed

    def test_enabled_specific_scope_denies_other(self, broker):
        """history_access=True, history_scopes=(current_chat,) => DENY all_chats."""
        token = CapabilityToken(
            skill_id="h4",
            history_access=True,
            history_scopes=("current_chat",),
        )
        assert broker.check_access(token, "history", "all_chats").denied

    def test_enabled_wildcard_allows_any(self, broker):
        """history_access=True, history_scopes=("*",) => ALLOW any."""
        token = CapabilityToken(
            skill_id="h5",
            history_access=True,
            history_scopes=("*",),
        )
        assert broker.check_access(token, "history", "all_chats").allowed

    def test_enabled_none_resource_denies(self, broker):
        """history_access=True, resource_name=None => DENY."""
        token = CapabilityToken(
            skill_id="h6",
            history_access=True,
            history_scopes=("current_chat",),
        )
        assert broker.check_access(token, "history", None).denied


# ------------------------------------------------------------------
# SECURITY INVARIANT: enabled + empty allowlist = deny (generic)
# ------------------------------------------------------------------


class TestBrokerSecurityInvariantEmptyAllowlistDenies:
    """Generic invariant: enabled=True + empty allowlist never grants access."""

    def test_network_enabled_empty_domains_denies(self, broker):
        token = CapabilityToken(
            skill_id="inv1", network_access=True, allowed_domains=()
        )
        assert broker.check_access(token, "network", "any.domain.com").denied

    def test_file_enabled_empty_scopes_denies(self, broker):
        token = CapabilityToken(skill_id="inv2", file_access=True, file_scopes=())
        assert broker.check_access(token, "file", "workspace:read").denied

    def test_history_enabled_empty_scopes_denies(self, broker):
        token = CapabilityToken(skill_id="inv3", history_access=True, history_scopes=())
        assert broker.check_access(token, "history", "current_chat").denied
