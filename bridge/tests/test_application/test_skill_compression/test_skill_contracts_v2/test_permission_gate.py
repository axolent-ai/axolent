"""T4 Tests: PermissionGate enforcement (4-path: Happy + Malicious + Rejection + Privacy).

Coverage:
  U13: Permission gate blocks unauthorized memory access
  U14: Permission gate allows authorized tool access
  S_PG1 (Happy): Allowed skill executes
  S_PG2 (Malicious): Workflow/tool execution type denied
  S_PG3 (Rejection): Blocked/flagged review status denied
  S_PG4 (Privacy): Permission check logs no sensitive values

  Default-deny matrix tests for ALL allowlist-based permissions:
    network (domains), file (scopes), history (scopes)
  Each matrix tests 6+ cases:
    enabled=False => DENY
    enabled=True, allowlist=() => DENY (CRITICAL)
    enabled=True, allowlist=(X,) => ALLOW X
    enabled=True, allowlist=(X,) => DENY Y
    enabled=True, allowlist=("*",) => ALLOW any
    enabled=True, resource=None => DENY

  Risk recomputation from permissions (denormalized field test).
  caplog privacy tests (real log output verification).

  Local-learn skills (origin=local_learn) with empty permissions:
    - can only run llm_instruction
    - cannot access memory, tools, network, file, secrets
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from application.skill_compression.permission_gate import (
    PermissionCheckResult,
    PermissionDecision,
    PermissionDeniedError,
    PermissionGate,
)
from application.skill_compression.skill_contract import (
    ExecutionConfig,
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
def sandbox_contract() -> SkillContract:
    """A minimal local_learn skill with empty permissions (sandbox)."""
    return create_minimal_contract(
        name="test-sandbox",
        phrases=("hello",),
        instruction="say hi",
        origin="local_learn",
    )


@pytest.fixture
def tool_contract() -> SkillContract:
    """A skill that declares tool access to web_search."""
    c = create_minimal_contract(
        name="test-tool",
        phrases=("search",),
        instruction="search the web",
    )
    return replace(
        c,
        permissions=PermissionsConfig(tools=("web_search",)),
    )


@pytest.fixture
def memory_contract() -> SkillContract:
    """A skill that declares memory read/write access."""
    c = create_minimal_contract(
        name="test-memory",
        phrases=("remember",),
        instruction="remember this",
    )
    return replace(
        c,
        permissions=PermissionsConfig(
            memory_read=("long_term_facts",),
            memory_write=("long_term_facts",),
        ),
    )


@pytest.fixture
def network_contract() -> SkillContract:
    """A skill that declares network access to specific domains."""
    c = create_minimal_contract(
        name="test-network",
        phrases=("fetch",),
        instruction="fetch data",
    )
    return replace(
        c,
        permissions=PermissionsConfig(
            network_access=NetworkAccessConfig(
                enabled=True,
                domains=("api.example.com", "data.example.com"),
            ),
        ),
    )


# ------------------------------------------------------------------
# Happy path: allowed execution
# ------------------------------------------------------------------


class TestPermissionGateHappy:
    """S_PG1: Skills that should be allowed to execute."""

    def test_sandbox_skill_allowed_to_execute(self, sandbox_contract):
        """Sandbox (local_learn, empty perms) is allowed for llm_instruction."""
        result = PermissionGate.check_execution_allowed(sandbox_contract)
        assert result.allowed
        assert result.decision == PermissionDecision.ALLOW

    def test_tool_contract_allowed_to_execute(self, tool_contract):
        """Skill with tool permissions is allowed (execution check only)."""
        result = PermissionGate.check_execution_allowed(tool_contract)
        assert result.allowed

    def test_declared_tool_access_allowed(self, tool_contract):
        """U14: Skill with tools=["web_search"] can access web_search."""
        result = PermissionGate.check_tool_access(tool_contract, "web_search")
        assert result.allowed

    def test_declared_memory_read_allowed(self, memory_contract):
        """Skill with memory_read declared can read that store."""
        result = PermissionGate.check_memory_access(
            memory_contract, "long_term_facts", "read"
        )
        assert result.allowed

    def test_declared_memory_write_allowed(self, memory_contract):
        """Skill with memory_write declared can write to that store."""
        result = PermissionGate.check_memory_access(
            memory_contract, "long_term_facts", "write"
        )
        assert result.allowed

    def test_declared_network_access_allowed(self, network_contract):
        """Skill with network_access.enabled can access declared domains."""
        result = PermissionGate.check_network_access(
            network_contract, "api.example.com"
        )
        assert result.allowed

    def test_file_access_allowed_when_declared_with_scope(self):
        """Skill with file_access.enabled=True and matching scope is allowed."""
        c = create_minimal_contract(
            name="file-skill", phrases=("file",), instruction="read file"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",)),
            ),
        )
        result = PermissionGate.check_file_access(c, "workspace:read")
        assert result.allowed

    def test_history_access_allowed_when_declared_with_scope(self):
        """Skill with history_access.enabled=True and matching scope is allowed."""
        c = create_minimal_contract(
            name="history-skill", phrases=("hist",), instruction="read history"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
            ),
        )
        result = PermissionGate.check_history_access(c, "current_chat")
        assert result.allowed

    def test_secrets_access_allowed_when_declared(self):
        """Skill with secrets_access=True can access secrets."""
        c = create_minimal_contract(
            name="secrets-skill", phrases=("secret",), instruction="get secret"
        )
        c = replace(c, permissions=PermissionsConfig(secrets_access=True))
        result = PermissionGate.check_secrets_access(c)
        assert result.allowed


# ------------------------------------------------------------------
# Malicious path: attacks that must be denied
# ------------------------------------------------------------------


class TestPermissionGateMalicious:
    """S_PG2: Malicious execution types and capability escalation."""

    def test_workflow_execution_type_denied(self, sandbox_contract):
        """Workflow execution type is feature-flagged and must be denied."""
        c = replace(
            sandbox_contract,
            execution=ExecutionConfig(type="workflow", instruction="run workflow"),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert "workflow" in result.reason

    def test_tool_execution_type_denied(self, sandbox_contract):
        """Tool execution type is feature-flagged and must be denied."""
        c = replace(
            sandbox_contract,
            execution=ExecutionConfig(type="tool", instruction="run tool"),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied

    def test_unknown_execution_type_denied(self, sandbox_contract):
        """Unknown execution type is denied."""
        c = replace(
            sandbox_contract,
            execution=ExecutionConfig(type="code_eval", instruction="eval this"),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied

    def test_sandbox_skill_cannot_access_memory(self, sandbox_contract):
        """U13: Local_learn skill with empty perms cannot read memory."""
        result = PermissionGate.check_memory_access(
            sandbox_contract, "long_term_facts", "read"
        )
        assert result.denied

    def test_sandbox_skill_cannot_write_memory(self, sandbox_contract):
        """Local_learn skill with empty perms cannot write memory."""
        result = PermissionGate.check_memory_access(
            sandbox_contract, "long_term_facts", "write"
        )
        assert result.denied

    def test_sandbox_skill_cannot_access_tools(self, sandbox_contract):
        """Local_learn skill with empty perms cannot use tools."""
        result = PermissionGate.check_tool_access(sandbox_contract, "web_search")
        assert result.denied

    def test_sandbox_skill_cannot_access_network(self, sandbox_contract):
        """Local_learn skill with empty perms cannot access network."""
        result = PermissionGate.check_network_access(sandbox_contract, "evil.com")
        assert result.denied

    def test_sandbox_skill_cannot_access_files(self, sandbox_contract):
        """Local_learn skill with empty perms cannot access files."""
        result = PermissionGate.check_file_access(sandbox_contract, "workspace:read")
        assert result.denied

    def test_sandbox_skill_cannot_access_secrets(self, sandbox_contract):
        """Local_learn skill with empty perms cannot access secrets."""
        result = PermissionGate.check_secrets_access(sandbox_contract)
        assert result.denied

    def test_undeclared_tool_denied(self, tool_contract):
        """Tool not in permissions.tools is denied even if other tools declared."""
        result = PermissionGate.check_tool_access(tool_contract, "shell_exec")
        assert result.denied

    def test_undeclared_memory_store_denied(self, memory_contract):
        """Memory store not in permissions is denied."""
        result = PermissionGate.check_memory_access(memory_contract, "secrets", "read")
        assert result.denied

    def test_undeclared_network_domain_denied(self, network_contract):
        """Domain not in allowed_domains is denied."""
        result = PermissionGate.check_network_access(network_contract, "evil.com")
        assert result.denied


# ------------------------------------------------------------------
# Rejection path: review status and risk level
# ------------------------------------------------------------------


class TestPermissionGateRejection:
    """S_PG3: Skills that should be rejected based on status/risk."""

    def test_blocked_skill_denied(self, sandbox_contract):
        """Skill with review_status=blocked is denied."""
        c = replace(sandbox_contract, review_status="blocked")
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "review_status_blocked"

    def test_flagged_skill_denied(self, sandbox_contract):
        """Skill with review_status=flagged is denied."""
        c = replace(sandbox_contract, review_status="flagged")
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "review_status_blocked"

    def test_high_risk_skill_denied(self, sandbox_contract):
        """Skill with risk_level=high is denied (future: needs explicit approval)."""
        c = replace(
            sandbox_contract,
            risk_level="high",
            permissions=PermissionsConfig(secrets_access=True),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "high_risk_denied"

    def test_medium_risk_skill_allowed(self, sandbox_contract):
        """Skill with risk_level=medium is allowed (only high is blocked)."""
        c = replace(sandbox_contract, risk_level="medium")
        result = PermissionGate.check_execution_allowed(c)
        assert result.allowed

    def test_unknown_memory_operation_denied(self, memory_contract):
        """Unknown memory operation is denied."""
        result = PermissionGate.check_memory_access(
            memory_contract, "long_term_facts", "delete"
        )
        assert result.denied
        assert result.rule == "unknown_operation"

    def test_permission_denied_error_carries_result(self, sandbox_contract):
        """PermissionDeniedError stores the check result for inspection."""
        result = PermissionCheckResult(
            decision=PermissionDecision.DENY,
            reason="test reason",
            rule="test_rule",
        )
        error = PermissionDeniedError("blocked", result)
        assert error.result.denied
        assert error.result.reason == "test reason"


# ------------------------------------------------------------------
# Privacy path: no sensitive data in check results
# ------------------------------------------------------------------


class TestPermissionGatePrivacy:
    """S_PG4: Permission checks must not leak sensitive values."""

    def test_denial_reason_does_not_contain_instruction(self, sandbox_contract):
        """Denial reason must not contain the skill instruction."""
        c = replace(
            sandbox_contract,
            execution=ExecutionConfig(
                type="workflow",
                instruction="my-secret-api-key-abc123",
            ),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert "my-secret-api-key-abc123" not in result.reason

    def test_denial_reason_does_not_contain_trigger_phrases(self, sandbox_contract):
        """Denial reason must not contain trigger phrases."""
        c = replace(sandbox_contract, review_status="blocked")
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        for phrase in c.activation.phrases:
            assert phrase not in result.reason

    def test_denial_reason_does_not_contain_skill_name(self):
        """Denial reason must not contain the user-defined skill name."""
        c = create_minimal_contract(
            name="Kunde Mueller Steuerdaten",
            phrases=("steuern",),
            instruction="fetch tax data",
        )
        c = replace(c, review_status="blocked")
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert "Kunde Mueller" not in result.reason
        assert "Steuerdaten" not in result.reason

    def test_tool_denial_does_not_leak_other_tools(self, tool_contract):
        """When denying a tool, the reason should not list all declared tools."""
        result = PermissionGate.check_tool_access(tool_contract, "shell_exec")
        assert result.denied
        assert "web_search" not in result.reason

    def test_memory_denial_does_not_leak_other_stores(self, memory_contract):
        """When denying a memory store, reason should not list declared stores."""
        result = PermissionGate.check_memory_access(memory_contract, "secrets", "read")
        assert result.denied
        assert "long_term_facts" not in result.reason

    def test_network_denial_does_not_leak_skill_name(self, network_contract):
        """Network denial should not contain the skill name."""
        result = PermissionGate.check_network_access(network_contract, "evil.com")
        assert result.denied
        assert "test-network" not in result.reason

    def test_file_denial_does_not_leak_skill_name(self):
        """File denial should not contain the skill name."""
        c = create_minimal_contract(
            name="Therapie-Notizen Anna",
            phrases=("therapy",),
            instruction="read notes",
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",)),
            ),
        )
        result = PermissionGate.check_file_access(c, "home:read")
        assert result.denied
        assert "Therapie-Notizen" not in result.reason
        assert "Anna" not in result.reason

    def test_history_denial_does_not_leak_skill_name(self):
        """History denial should not contain the skill name."""
        c = create_minimal_contract(
            name="Patient Data Viewer",
            phrases=("patient",),
            instruction="show patient history",
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
            ),
        )
        result = PermissionGate.check_history_access(c, "all_chats")
        assert result.denied
        assert "Patient Data Viewer" not in result.reason


# ------------------------------------------------------------------
# caplog privacy tests: real log output verification
# ------------------------------------------------------------------


class TestPermissionGateCaplogPrivacy:
    """caplog-based tests: verify NO sensitive data reaches actual log output."""

    def test_risk_mismatch_log_does_not_leak_skill_name(self, caplog):
        """Risk mismatch warning must log skill_id, NOT skill name."""
        c = create_minimal_contract(
            name="Kunde Mueller Steuerdaten",
            phrases=("tax",),
            instruction="fetch tax data",
        )
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(
                    enabled=True, domains=("api.example.com",)
                ),
            ),
        )
        with caplog.at_level(logging.WARNING):
            result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert "Kunde Mueller" not in caplog.text
        assert "Steuerdaten" not in caplog.text
        assert c.id in caplog.text

    def test_blocked_skill_deny_does_not_log_name(self, caplog):
        """Blocked skill denial reason in logs must not contain skill name."""
        c = create_minimal_contract(
            name="Private Health Records",
            phrases=("health",),
            instruction="get records",
        )
        c = replace(c, review_status="blocked")
        with caplog.at_level(logging.DEBUG):
            result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert "Private Health Records" not in result.reason


# ------------------------------------------------------------------
# NETWORK MATRIX: default-deny semantics
# ------------------------------------------------------------------


class TestNetworkAccessMatrix:
    """Permission matrix for network_access (domains allowlist)."""

    def test_enabled_false_empty_domains_denies(self):
        """enabled=False, domains=() => DENY api.example.com."""
        c = create_minimal_contract(name="n1", phrases=("n",), instruction="n")
        c = replace(
            c,
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(enabled=False, domains=()),
            ),
        )
        result = PermissionGate.check_network_access(c, "api.example.com")
        assert result.denied
        assert result.rule == "network_disabled"

    def test_enabled_true_empty_domains_denies(self):
        """CRITICAL: enabled=True, domains=() => DENY api.example.com."""
        c = create_minimal_contract(name="n2", phrases=("n",), instruction="n")
        c = replace(
            c,
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(enabled=True, domains=()),
            ),
        )
        result = PermissionGate.check_network_access(c, "api.example.com")
        assert result.denied
        assert result.rule == "network_empty_allowlist"

    def test_enabled_true_specific_domain_allows(self):
        """enabled=True, domains=(api.example.com,) => ALLOW api.example.com."""
        c = create_minimal_contract(name="n3", phrases=("n",), instruction="n")
        c = replace(
            c,
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(
                    enabled=True, domains=("api.example.com",)
                ),
            ),
        )
        result = PermissionGate.check_network_access(c, "api.example.com")
        assert result.allowed

    def test_enabled_true_specific_domain_denies_other(self):
        """enabled=True, domains=(api.example.com,) => DENY evil.com."""
        c = create_minimal_contract(name="n4", phrases=("n",), instruction="n")
        c = replace(
            c,
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(
                    enabled=True, domains=("api.example.com",)
                ),
            ),
        )
        result = PermissionGate.check_network_access(c, "evil.com")
        assert result.denied

    def test_enabled_true_wildcard_allows_any(self):
        """enabled=True, domains=("*",) => ALLOW evil.com (explicit wildcard)."""
        c = create_minimal_contract(name="n5", phrases=("n",), instruction="n")
        c = replace(
            c,
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(enabled=True, domains=("*",)),
            ),
        )
        result = PermissionGate.check_network_access(c, "evil.com")
        assert result.allowed
        assert result.rule == "network_wildcard"

    def test_enabled_true_domain_none_denies(self):
        """enabled=True, domain=None => DENY (no concrete target)."""
        c = create_minimal_contract(name="n6", phrases=("n",), instruction="n")
        c = replace(
            c,
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(
                    enabled=True, domains=("api.example.com",)
                ),
            ),
        )
        result = PermissionGate.check_network_access(c, None)
        assert result.denied
        assert result.rule == "network_no_target"


# ------------------------------------------------------------------
# FILE MATRIX: default-deny semantics
# ------------------------------------------------------------------


class TestFileAccessMatrix:
    """Permission matrix for file_access (scopes allowlist)."""

    def test_enabled_false_empty_scopes_denies(self):
        """enabled=False, scopes=() => DENY workspace:read."""
        c = create_minimal_contract(name="f1", phrases=("f",), instruction="f")
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=False, scopes=()),
            ),
        )
        result = PermissionGate.check_file_access(c, "workspace:read")
        assert result.denied
        assert result.rule == "file_disabled"

    def test_enabled_true_empty_scopes_denies(self):
        """CRITICAL: enabled=True, scopes=() => DENY workspace:read."""
        c = create_minimal_contract(name="f2", phrases=("f",), instruction="f")
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=()),
            ),
        )
        result = PermissionGate.check_file_access(c, "workspace:read")
        assert result.denied
        assert result.rule == "file_empty_allowlist"

    def test_enabled_true_specific_scope_allows(self):
        """enabled=True, scopes=(workspace:read,) => ALLOW workspace:read."""
        c = create_minimal_contract(name="f3", phrases=("f",), instruction="f")
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",)),
            ),
        )
        result = PermissionGate.check_file_access(c, "workspace:read")
        assert result.allowed

    def test_enabled_true_specific_scope_denies_other(self):
        """enabled=True, scopes=(workspace:read,) => DENY home:read."""
        c = create_minimal_contract(name="f4", phrases=("f",), instruction="f")
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",)),
            ),
        )
        result = PermissionGate.check_file_access(c, "home:read")
        assert result.denied

    def test_enabled_true_wildcard_allows_any(self):
        """enabled=True, scopes=("*",) => ALLOW home:read (explicit wildcard)."""
        c = create_minimal_contract(name="f5", phrases=("f",), instruction="f")
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=("*",)),
            ),
        )
        result = PermissionGate.check_file_access(c, "home:read")
        assert result.allowed
        assert result.rule == "file_wildcard"

    def test_enabled_true_scope_none_denies(self):
        """enabled=True, scope=None => DENY (no concrete target)."""
        c = create_minimal_contract(name="f6", phrases=("f",), instruction="f")
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",)),
            ),
        )
        result = PermissionGate.check_file_access(c, None)
        assert result.denied
        assert result.rule == "file_no_target"


# ------------------------------------------------------------------
# HISTORY MATRIX: default-deny semantics
# ------------------------------------------------------------------


class TestHistoryAccessMatrix:
    """Permission matrix for history_access (scopes allowlist)."""

    def test_enabled_false_empty_scopes_denies(self):
        """enabled=False, scopes=() => DENY current_chat."""
        c = create_minimal_contract(name="h1", phrases=("h",), instruction="h")
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=False, scopes=()),
            ),
        )
        result = PermissionGate.check_history_access(c, "current_chat")
        assert result.denied
        assert result.rule == "history_disabled"

    def test_enabled_true_empty_scopes_denies(self):
        """CRITICAL: enabled=True, scopes=() => DENY current_chat."""
        c = create_minimal_contract(name="h2", phrases=("h",), instruction="h")
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=()),
            ),
        )
        result = PermissionGate.check_history_access(c, "current_chat")
        assert result.denied
        assert result.rule == "history_empty_allowlist"

    def test_enabled_true_specific_scope_allows(self):
        """enabled=True, scopes=(current_chat,) => ALLOW current_chat."""
        c = create_minimal_contract(name="h3", phrases=("h",), instruction="h")
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
            ),
        )
        result = PermissionGate.check_history_access(c, "current_chat")
        assert result.allowed

    def test_enabled_true_specific_scope_denies_other(self):
        """enabled=True, scopes=(current_chat,) => DENY all_chats."""
        c = create_minimal_contract(name="h4", phrases=("h",), instruction="h")
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
            ),
        )
        result = PermissionGate.check_history_access(c, "all_chats")
        assert result.denied

    def test_enabled_true_wildcard_allows_any(self):
        """enabled=True, scopes=("*",) => ALLOW all_chats (explicit wildcard)."""
        c = create_minimal_contract(name="h5", phrases=("h",), instruction="h")
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=("*",)),
            ),
        )
        result = PermissionGate.check_history_access(c, "all_chats")
        assert result.allowed
        assert result.rule == "history_wildcard"

    def test_enabled_true_scope_none_denies(self):
        """enabled=True, scope=None => DENY (no concrete target)."""
        c = create_minimal_contract(name="h6", phrases=("h",), instruction="h")
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
            ),
        )
        result = PermissionGate.check_history_access(c, None)
        assert result.denied
        assert result.rule == "history_no_target"


# ------------------------------------------------------------------
# DENORMALIZED FIELD TEST: risk recomputed from permissions
# ------------------------------------------------------------------


class TestPermissionGateRiskRecomputation:
    """Risk level must be recomputed from permissions, not trusted from storage."""

    def test_stale_low_risk_with_network_permissions_denied(self):
        """Stored risk_level=low + network_access=True => recomputed high => DENY."""
        c = create_minimal_contract(
            name="stale-risk", phrases=("stale",), instruction="test"
        )
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(
                    enabled=True, domains=("api.example.com",)
                ),
            ),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "high_risk_denied"

    def test_stale_low_risk_with_file_permissions_denied(self):
        """Stored risk_level=low + file_access=True => recomputed high => DENY."""
        c = create_minimal_contract(
            name="stale-risk-file", phrases=("stale",), instruction="test"
        )
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",)),
            ),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "high_risk_denied"

    def test_stale_low_risk_with_secrets_denied(self):
        """Stored risk_level=low + secrets_access=True => recomputed high => DENY."""
        c = create_minimal_contract(
            name="stale-risk-secrets", phrases=("stale",), instruction="test"
        )
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(secrets_access=True),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "high_risk_denied"

    def test_correct_low_risk_with_no_permissions_allowed(self):
        """Stored risk_level=low + no permissions => recomputed low => ALLOW."""
        c = create_minimal_contract(
            name="correct-low", phrases=("safe",), instruction="test"
        )
        c = replace(c, risk_level="low")
        result = PermissionGate.check_execution_allowed(c)
        assert result.allowed

    def test_risk_mismatch_logs_warning(self, caplog):
        """When stored risk != computed risk, a warning is logged."""
        c = create_minimal_contract(
            name="mismatch-test", phrases=("test",), instruction="test"
        )
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(
                    enabled=True, domains=("api.example.com",)
                ),
            ),
        )
        with caplog.at_level(logging.WARNING):
            PermissionGate.check_execution_allowed(c)
        assert "risk_level mismatch" in caplog.text
        assert "stored=low" in caplog.text
        assert "computed=high" in caplog.text


# ------------------------------------------------------------------
# SECURITY INVARIANT: enabled + empty allowlist = deny (generic)
# ------------------------------------------------------------------


class TestSecurityInvariantEmptyAllowlistDenies:
    """Generic invariant: enabled=True + empty allowlist never grants access."""

    def test_network_enabled_empty_domains_denies(self):
        c = create_minimal_contract(name="inv", phrases=("i",), instruction="i")
        c = replace(
            c,
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(enabled=True, domains=()),
            ),
        )
        assert PermissionGate.check_network_access(c, "any.domain.com").denied

    def test_file_enabled_empty_scopes_denies(self):
        c = create_minimal_contract(name="inv", phrases=("i",), instruction="i")
        c = replace(
            c,
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=()),
            ),
        )
        assert PermissionGate.check_file_access(c, "workspace:read").denied

    def test_history_enabled_empty_scopes_denies(self):
        c = create_minimal_contract(name="inv", phrases=("i",), instruction="i")
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=()),
            ),
        )
        assert PermissionGate.check_history_access(c, "current_chat").denied
