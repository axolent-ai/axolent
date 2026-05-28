"""Security Round 2 Tests: Access/Risk/Package Consistency Matrix.

Covers Codex review findings from Round 1 re-review:
  Fix 1 (BLOCKER): history_access wildcard must be high-risk
  Fix 2 (PROAKTIV): ALL permissions checked for Access/Risk/Package consistency
  Fix 3 (MEDIUM): active_count lock discipline (async count_active)
  Fix 4 (TEST-GAP): chat_service caplog test for deny-log privacy

Codex GO-List (8 tests):
  1. test_history_wildcard_is_high_risk
  2. test_history_all_chats_is_high_risk
  3. test_history_current_chat_is_low_or_medium
  4. test_history_access_not_local_skill_package_type
  5. test_history_wildcard_denied_by_execution_gate_when_stored_low
  6. test_history_wildcard_full_security_path (store persist + gate)
  7. test_power_permissions_are_not_low_or_local (parametrized)
  8. test_chat_service_permission_deny_log_does_not_leak_skill_name (caplog)

Additional consistency matrix tests for all permission types.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from application.skill_compression.contract_store import (
    ContractStore,
    _finalize_security_metadata,
)
from application.skill_compression.permission_gate import (
    PermissionGate,
)
from application.skill_compression.skill_contract import (
    FileAccessConfig,
    HistoryAccessConfig,
    NetworkAccessConfig,
    PermissionsConfig,
    compute_package_type,
    compute_risk_level,
    create_minimal_contract,
)


# ------------------------------------------------------------------
# DB fixtures (same pattern as test_contract_store_security.py)
# ------------------------------------------------------------------


class _TestConnection:
    """Minimal SQLite wrapper matching DBConnection protocol."""

    def __init__(self, raw_conn: sqlite3.Connection):
        self._conn = raw_conn

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)

    def executescript(self, sql):
        self._conn.executescript(sql)

    def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations):
        self._conn.execute("BEGIN")
        try:
            for sql, params in operations:
                self._conn.execute(sql, params)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return _TestConnection(conn)


@pytest.fixture
def store(db_conn) -> ContractStore:
    s = ContractStore(db=db_conn)
    s.init_schema()
    return s


USER_ID = 99999


# ══════════════════════════════════════════════════════════════
# GO-LIST 1: test_history_wildcard_is_high_risk
# ══════════════════════════════════════════════════════════════


class TestHistoryRiskLevel:
    """History access risk level based on scopes."""

    def test_history_wildcard_is_high_risk(self):
        """GO-1: scopes=('*',) => high risk."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
        )
        assert compute_risk_level(p) == "high"

    def test_history_all_chats_is_high_risk(self):
        """GO-2: scopes=('all_chats',) => high risk."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",))
        )
        assert compute_risk_level(p) == "high"

    def test_history_all_is_high_risk(self):
        """scopes=('all',) => high risk (power scope)."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("all",))
        )
        assert compute_risk_level(p) == "high"

    def test_history_current_chat_is_low(self):
        """GO-3: scopes=('current_chat',) => low risk (not a power scope)."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",))
        )
        assert compute_risk_level(p) == "low"

    def test_history_disabled_is_low(self):
        """Disabled history access => low risk."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=False, scopes=())
        )
        assert compute_risk_level(p) == "low"


# ══════════════════════════════════════════════════════════════
# GO-LIST 4: test_history_access_not_local_skill_package_type
# ══════════════════════════════════════════════════════════════


class TestHistoryPackageType:
    """History access package type classification."""

    def test_history_access_not_local_skill_package_type(self):
        """GO-4: ANY history_access.enabled=True => NOT local_skill."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",))
        )
        assert compute_package_type(p) != "local_skill"
        assert compute_package_type(p) == "declarative_skill"

    def test_history_wildcard_not_local_skill(self):
        """History wildcard also not local_skill."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
        )
        assert compute_package_type(p) != "local_skill"

    def test_history_disabled_can_be_local_skill(self):
        """Disabled history => can still be local_skill (if no other perms)."""
        p = PermissionsConfig()
        assert compute_package_type(p) == "local_skill"


# ══════════════════════════════════════════════════════════════
# GO-LIST 5: test_history_wildcard_denied_by_execution_gate_when_stored_low
# ══════════════════════════════════════════════════════════════


class TestHistoryExecutionGate:
    """Execution gate blocks history wildcard even if stored as low."""

    def test_history_wildcard_denied_by_execution_gate_when_stored_low(self):
        """GO-5: stale low + history wildcard => execution denied."""
        c = create_minimal_contract(name="probe-stale", phrases=("p",), instruction="i")
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
            ),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "high_risk_denied"

    def test_history_all_chats_denied_by_execution_gate_when_stored_low(self):
        """stale low + all_chats => execution denied."""
        c = create_minimal_contract(
            name="probe-stale-allchats", phrases=("p",), instruction="i"
        )
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",))
            ),
        )
        result = PermissionGate.check_execution_allowed(c)
        assert result.denied
        assert result.rule == "high_risk_denied"


# ══════════════════════════════════════════════════════════════
# GO-LIST 6: test_history_wildcard_full_security_path
# ══════════════════════════════════════════════════════════════


class TestHistoryFullSecurityPath:
    """Full pipeline: persist => risk high, package != local, gate denied."""

    def test_history_wildcard_full_security_path(self, store):
        """GO-6: Contract with history wildcard gets correct classification
        through the full persist + gate pipeline."""
        c = create_minimal_contract(
            name="full-path-history",
            phrases=("fp",),
            instruction="test full path",
        )
        c = replace(
            c,
            risk_level="low",  # deliberately wrong
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
            ),
        )
        # Persist: finalize must correct risk and package_type
        persisted = store.persist(c, USER_ID)
        assert persisted.risk_level == "high"
        assert persisted.store_meta.package_type != "local_skill"
        assert persisted.store_meta.package_type == "declarative_skill"
        # Gate: high-risk must be denied
        result = PermissionGate.check_execution_allowed(persisted)
        assert result.denied
        assert result.rule == "high_risk_denied"

    def test_history_all_chats_full_security_path(self, store):
        """all_chats through full pipeline."""
        c = create_minimal_contract(
            name="full-path-allchats",
            phrases=("fp2",),
            instruction="test full path allchats",
        )
        c = replace(
            c,
            risk_level="low",
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",))
            ),
        )
        persisted = store.persist(c, USER_ID)
        assert persisted.risk_level == "high"
        result = PermissionGate.check_execution_allowed(persisted)
        assert result.denied


# ══════════════════════════════════════════════════════════════
# GO-LIST 7: test_power_permissions_are_not_low_or_local (parametrized)
# ══════════════════════════════════════════════════════════════


_POWER_PERMISSION_CONFIGS = [
    pytest.param(
        PermissionsConfig(
            network_access=NetworkAccessConfig(enabled=True, domains=("*",))
        ),
        id="network_wildcard",
    ),
    pytest.param(
        PermissionsConfig(
            network_access=NetworkAccessConfig(enabled=True, domains=("api.x.com",))
        ),
        id="network_specific",
    ),
    pytest.param(
        PermissionsConfig(file_access=FileAccessConfig(enabled=True, scopes=("*",))),
        id="file_wildcard",
    ),
    pytest.param(
        PermissionsConfig(
            file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",))
        ),
        id="file_specific",
    ),
    pytest.param(
        PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
        ),
        id="history_wildcard",
    ),
    pytest.param(
        PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",))
        ),
        id="history_all_chats",
    ),
    pytest.param(
        PermissionsConfig(secrets_access=True),
        id="secrets_access",
    ),
    pytest.param(
        PermissionsConfig(tools=("*",)),
        id="tools_wildcard",
    ),
]


class TestPowerPermissionsConsistency:
    """GO-7: Power permissions must never be low-risk or local_skill."""

    @pytest.mark.parametrize("permissions", _POWER_PERMISSION_CONFIGS)
    def test_power_permissions_are_not_low_or_local(self, permissions):
        """No power permission may be classified as low-risk AND local_skill."""
        risk = compute_risk_level(permissions)
        pkg = compute_package_type(permissions)
        assert risk != "low" or pkg != "local_skill", (
            f"Power permission classified as risk={risk}, package={pkg}"
        )

    @pytest.mark.parametrize(
        "permissions",
        [
            pytest.param(
                PermissionsConfig(
                    network_access=NetworkAccessConfig(enabled=True, domains=("*",))
                ),
                id="network_wildcard_strict",
            ),
            pytest.param(
                PermissionsConfig(
                    file_access=FileAccessConfig(enabled=True, scopes=("*",))
                ),
                id="file_wildcard_strict",
            ),
            pytest.param(
                PermissionsConfig(
                    history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
                ),
                id="history_wildcard_strict",
            ),
            pytest.param(
                PermissionsConfig(
                    history_access=HistoryAccessConfig(
                        enabled=True, scopes=("all_chats",)
                    )
                ),
                id="history_all_chats_strict",
            ),
            pytest.param(
                PermissionsConfig(secrets_access=True),
                id="secrets_strict",
            ),
            pytest.param(
                PermissionsConfig(tools=("*",)),
                id="tools_wildcard_strict",
            ),
        ],
    )
    def test_wildcard_power_permissions_are_high_risk(self, permissions):
        """Wildcards and secrets MUST be high-risk (not just non-low)."""
        risk = compute_risk_level(permissions)
        assert risk == "high", f"Expected high, got {risk}"

    @pytest.mark.parametrize(
        "permissions",
        [
            pytest.param(
                PermissionsConfig(
                    network_access=NetworkAccessConfig(enabled=True, domains=("*",))
                ),
                id="network_wildcard_not_local",
            ),
            pytest.param(
                PermissionsConfig(
                    file_access=FileAccessConfig(enabled=True, scopes=("*",))
                ),
                id="file_wildcard_not_local",
            ),
            pytest.param(
                PermissionsConfig(
                    history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
                ),
                id="history_wildcard_not_local",
            ),
            pytest.param(
                PermissionsConfig(secrets_access=True),
                id="secrets_not_local",
            ),
            pytest.param(
                PermissionsConfig(tools=("*",)),
                id="tools_wildcard_not_local",
            ),
        ],
    )
    def test_wildcard_power_permissions_not_local_skill(self, permissions):
        """Wildcards and secrets MUST NOT be local_skill."""
        pkg = compute_package_type(permissions)
        assert pkg != "local_skill", f"Expected non-local, got {pkg}"


# ══════════════════════════════════════════════════════════════
# GO-LIST 8: test_chat_service_permission_deny_log_does_not_leak_skill_name
# ══════════════════════════════════════════════════════════════


class TestChatServiceDenyLogPrivacy:
    """Verify chat_service deny-log does not leak sensitive skill names.

    Tests the real wiring path: ChatService._match_skills_for_prompt()
    triggers PermissionGate and logs the denial. The log MUST contain
    skill_id and rule but MUST NOT contain skill name, instruction,
    or trigger phrases.
    """

    def test_chat_service_permission_deny_log_does_not_leak_skill_name(self, caplog):
        """GO-8: Real wiring path: deny log must not leak skill name."""
        from application.chat_service import ChatService
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.skill_matcher import SkillMatch

        # Build a contract with a sensitive name + high-risk permissions
        contract = create_minimal_contract(
            name="Kunde Mueller Steuerdaten",
            phrases=("steuern",),
            instruction="fetch tax data for client",
        )
        contract = replace(
            contract,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
            ),
        )
        # Finalize so risk_level is computed
        contract = _finalize_security_metadata(contract)

        # Build a fake SkillMatch with the contract attached
        hyp = Hypothesis(
            hypothesis_id="hyp_test_deny",
            user_id=42,
            claim="Kunde Mueller Steuerdaten",
            status="active",
            created_at="2026-05-28T10:00:00+00:00",
            last_seen="2026-05-28T10:00:00+00:00",
        )
        match = SkillMatch(
            hypothesis=hyp,
            confidence=0.99,
            requires_confirmation=False,
            explanation="exact phrase match",
        )
        # Attach contract to match (as the contract-aware matcher would)
        match_with_contract = MagicMock(wraps=match)
        match_with_contract.contract = contract
        match_with_contract.hypothesis = hyp
        match_with_contract.confidence = 0.99

        # Build ChatService with a mocked skill_matcher
        mock_matcher = MagicMock()
        mock_matcher.match.return_value = match_with_contract

        # Minimal ChatService construction (only need skill_matcher wiring)
        service = ChatService(
            provider_router=MagicMock(),
            skill_matcher=mock_matcher,
        )

        # Call the method under test
        with caplog.at_level(logging.INFO):
            block, result = service._match_skills_for_prompt(
                user_id=42,
                text="steuern",
                lang="de",
                task_slot_name=None,
            )

        # Skill must be denied (high risk)
        assert block == ""
        assert result is None

        # Log must contain skill_id and rule
        assert contract.id in caplog.text
        assert "high_risk_denied" in caplog.text

        # Log must NOT contain sensitive data
        assert "Kunde Mueller" not in caplog.text
        assert "Steuerdaten" not in caplog.text
        assert "fetch tax data" not in caplog.text
        assert "steuern" not in caplog.text


# ══════════════════════════════════════════════════════════════
# FIX 2 PROAKTIV: Full Access/Risk/Package Consistency Matrix
# ══════════════════════════════════════════════════════════════


class TestConsistencyMatrix:
    """Verify Access/Risk/Package consistency for ALL permission types.

    For each permission type, three questions:
      1. Is access enforcement present? (tested in test_permission_gate.py)
      2. What is the computed risk_level?
      3. What is the computed package_type?

    Policy decisions documented here:
      network (any)     => high risk, code_plugin
      file (any)        => high risk, code_plugin
      history current   => low risk, declarative_skill
      history all/wild  => HIGH risk, declarative_skill
      memory_read spec  => low risk, declarative_skill
      memory_read *     => medium risk, declarative_skill (policy: broad read)
      memory_write      => medium risk, declarative_skill
      tools specific    => medium risk, tool_workflow
      tools *           => HIGH risk, code_plugin (policy: wildcard escalation)
      secrets           => high risk, privileged_plugin
      no permissions    => low risk, local_skill
    """

    def test_network_specific_domain(self):
        """network domains=(specific) => high, code_plugin."""
        p = PermissionsConfig(
            network_access=NetworkAccessConfig(
                enabled=True, domains=("api.example.com",)
            )
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_network_wildcard_domain(self):
        """network domains=('*',) => high, code_plugin."""
        p = PermissionsConfig(
            network_access=NetworkAccessConfig(enabled=True, domains=("*",))
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_file_specific_scope(self):
        """file scopes=(specific) => high, code_plugin."""
        p = PermissionsConfig(
            file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",))
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_file_wildcard_scope(self):
        """file scopes=('*',) => high, code_plugin."""
        p = PermissionsConfig(file_access=FileAccessConfig(enabled=True, scopes=("*",)))
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_history_current_chat(self):
        """history scopes=(current_chat,) => low, declarative_skill."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",))
        )
        assert compute_risk_level(p) == "low"
        assert compute_package_type(p) == "declarative_skill"

    def test_history_all_chats(self):
        """history scopes=(all_chats,) => HIGH, declarative_skill."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",))
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "declarative_skill"

    def test_history_wildcard(self):
        """history scopes=('*',) => HIGH, declarative_skill."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "declarative_skill"

    def test_memory_read_specific(self):
        """memory_read=(specific,) => low, declarative_skill."""
        p = PermissionsConfig(memory_read=("long_term_facts",))
        assert compute_risk_level(p) == "low"
        assert compute_package_type(p) == "declarative_skill"

    def test_memory_read_wildcard(self):
        """memory_read=('*',) => medium, declarative_skill.
        Policy: wildcard read across all stores is elevated."""
        p = PermissionsConfig(memory_read=("*",))
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "declarative_skill"

    def test_memory_write(self):
        """memory_write=(specific,) => medium, declarative_skill."""
        p = PermissionsConfig(memory_write=("long_term_facts",))
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "declarative_skill"

    def test_tools_specific(self):
        """tools=(specific,) => medium, tool_workflow."""
        p = PermissionsConfig(tools=("web_search",))
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "tool_workflow"

    def test_tools_wildcard(self):
        """tools=('*',) => HIGH, code_plugin.
        Policy: wildcard tool access is power permission."""
        p = PermissionsConfig(tools=("*",))
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_secrets_access(self):
        """secrets_access=True => high, privileged_plugin."""
        p = PermissionsConfig(secrets_access=True)
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "privileged_plugin"

    def test_no_permissions(self):
        """No permissions => low, local_skill."""
        p = PermissionsConfig()
        assert compute_risk_level(p) == "low"
        assert compute_package_type(p) == "local_skill"


# ══════════════════════════════════════════════════════════════
# FIX 3: active_count lock discipline
# ══════════════════════════════════════════════════════════════


class TestDraftStoreActiveCountLock:
    """Verify active_count property is non-mutating, count_active is async."""

    @pytest.mark.asyncio
    async def test_active_count_property_is_raw(self):
        """active_count returns raw count without cleanup."""
        from application.skill_compression.draft_store import DraftStore

        store = DraftStore(ttl_seconds=1)
        c = create_minimal_contract(
            name="lock-test", phrases=("lt",), instruction="test"
        )
        await store.create(42, 100, c)

        # Before expiry: raw count matches
        assert store.active_count == 1

    @pytest.mark.asyncio
    async def test_count_active_is_async_and_cleans_expired(self):
        """count_active() is async and cleans expired drafts."""
        import asyncio

        from application.skill_compression.draft_store import DraftStore

        store = DraftStore(ttl_seconds=1)
        c = create_minimal_contract(
            name="lock-test-async", phrases=("lta",), instruction="test"
        )
        await store.create(42, 100, c)
        await asyncio.sleep(1.5)

        # Raw count still includes expired draft
        assert store.active_count == 1

        # Async count_active cleans up
        count = await store.count_active()
        assert count == 0

        # After cleanup, raw count is also 0
        assert store.active_count == 0

    @pytest.mark.asyncio
    async def test_count_active_matches_after_cleanup(self):
        """count_active returns same as active_count after cleanup."""
        from application.skill_compression.draft_store import DraftStore

        store = DraftStore(ttl_seconds=3600)
        c1 = create_minimal_contract(
            name="lock-test-1", phrases=("lt1",), instruction="test1"
        )
        c2 = create_minimal_contract(
            name="lock-test-2", phrases=("lt2",), instruction="test2"
        )
        await store.create(42, 100, c1)
        await store.create(42, 200, c2)

        count = await store.count_active()
        assert count == 2
        assert store.active_count == 2


# ══════════════════════════════════════════════════════════════
# Consistency: _finalize_security_metadata pipeline
# ══════════════════════════════════════════════════════════════


class TestFinalizeSecurityMetadata:
    """Verify _finalize_security_metadata correctly classifies all permission types."""

    def test_finalize_history_wildcard(self):
        """History wildcard gets finalized to high risk, declarative_skill."""
        c = create_minimal_contract(
            name="finalize-hw", phrases=("fhw",), instruction="test"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=("*",))
            ),
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "high"
        assert finalized.store_meta.package_type == "declarative_skill"

    def test_finalize_tools_wildcard(self):
        """Tools wildcard gets finalized to high risk, code_plugin."""
        c = create_minimal_contract(
            name="finalize-tw", phrases=("ftw",), instruction="test"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(tools=("*",)),
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "high"
        assert finalized.store_meta.package_type == "code_plugin"

    def test_finalize_memory_read_wildcard(self):
        """memory_read wildcard gets finalized to medium risk, declarative_skill."""
        c = create_minimal_contract(
            name="finalize-mrw", phrases=("fmrw",), instruction="test"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(memory_read=("*",)),
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "medium"
        assert finalized.store_meta.package_type == "declarative_skill"

    def test_finalize_no_permissions(self):
        """No permissions => low risk, local_skill."""
        c = create_minimal_contract(
            name="finalize-none", phrases=("fn",), instruction="test"
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "low"
        assert finalized.store_meta.package_type == "local_skill"
