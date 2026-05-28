"""Security Round 3 Tests: Monotonicity + Pairwise Combination Matrix.

Covers Codex review findings from Round 2 re-review:
  Fix 1 (BLOCKER): compute_risk_level() must be monotonic (High-first)
  Fix 2 (HIGH):    compute_package_type() must be hierarchical (no masking)
  Fix 3 (PROAKTIV): All classification functions audited for early-return masking

Codex GO-List Round 3 (8 checks):
  1. history current_chat + tools=("*") => risk high, package code_plugin
  2. history current_chat + tools specific => risk medium, package tool_workflow
  3. history current_chat + memory_write => risk medium
  4. history current_chat + memory_read wildcard => risk medium
  5. Adding a permission never downgrades risk rank (monotonicity invariant)
  6. Adding a permission never downgrades package rank (hierarchy invariant)
  7. Store-finalizer combination: history + tools wildcard
  8. Store-finalizer combination: history + memory_write

Additional: Pairwise permission matrix (all relevant 2-permission combinations).
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from application.skill_compression.contract_store import (
    ContractStore,
    _finalize_security_metadata,
)
from application.skill_compression.skill_contract import (
    FileAccessConfig,
    HistoryAccessConfig,
    NetworkAccessConfig,
    PermissionsConfig,
    SafetyConfig,
    compute_package_type,
    compute_risk_level,
    create_minimal_contract,
)


# ------------------------------------------------------------------
# DB fixtures (same pattern as test_security_round2.py)
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


# ------------------------------------------------------------------
# Helper: merge two PermissionsConfig (OR for bools, union for tuples)
# ------------------------------------------------------------------


def _merge_permissions(
    base: PermissionsConfig, added: PermissionsConfig
) -> PermissionsConfig:
    """Merge two PermissionsConfig: OR for bools, union for tuples.

    This is the correct semantic for "adding permissions": the result
    must be at least as permissive as either input.
    """

    # Union helper for tuple fields (deduplicated, order-stable)
    def _union_tuples(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for item in (*a, *b):
            if item not in seen:
                seen.add(item)
                result.append(item)
        return tuple(result)

    return PermissionsConfig(
        tools=_union_tuples(base.tools, added.tools),
        memory_read=_union_tuples(base.memory_read, added.memory_read),
        memory_write=_union_tuples(base.memory_write, added.memory_write),
        network_access=NetworkAccessConfig(
            enabled=base.network_access.enabled or added.network_access.enabled,
            domains=_union_tuples(
                base.network_access.domains, added.network_access.domains
            ),
        ),
        file_access=FileAccessConfig(
            enabled=base.file_access.enabled or added.file_access.enabled,
            scopes=_union_tuples(base.file_access.scopes, added.file_access.scopes),
        ),
        history_access=HistoryAccessConfig(
            enabled=base.history_access.enabled or added.history_access.enabled,
            scopes=_union_tuples(
                base.history_access.scopes, added.history_access.scopes
            ),
        ),
        secrets_access=base.secrets_access or added.secrets_access,
    )


# ------------------------------------------------------------------
# Rank maps for monotonicity invariants
# ------------------------------------------------------------------

_RISK_RANK = {"low": 1, "medium": 2, "high": 3}

_PACKAGE_RANK = {
    "local_skill": 1,
    "declarative_skill": 2,
    "tool_workflow": 3,
    "code_plugin": 4,
    "privileged_plugin": 5,
}


# ==================================================================
# GO-LIST 1-4: Specific combination tests (Codex-required)
# ==================================================================


class TestCombinationRiskLevel:
    """Codex GO-List: history current_chat combined with other permissions."""

    def test_history_current_plus_tools_wildcard_is_high(self):
        """GO-1: history current_chat + tools=('*',) => HIGH risk."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            tools=("*",),
        )
        assert compute_risk_level(p) == "high"

    def test_history_current_plus_tools_wildcard_is_code_plugin(self):
        """GO-1b: history current_chat + tools=('*',) => code_plugin."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            tools=("*",),
        )
        assert compute_package_type(p) == "code_plugin"

    def test_history_current_plus_tools_specific_is_medium(self):
        """GO-2: history current_chat + tools=('web_search',) => MEDIUM risk."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            tools=("web_search",),
        )
        assert compute_risk_level(p) == "medium"

    def test_history_current_plus_tools_specific_is_tool_workflow(self):
        """GO-2b: history current_chat + tools=('web_search',) => tool_workflow."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            tools=("web_search",),
        )
        assert compute_package_type(p) == "tool_workflow"

    def test_history_current_plus_memory_write_is_medium(self):
        """GO-3: history current_chat + memory_write => MEDIUM risk."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            memory_write=("long_term_facts",),
        )
        assert compute_risk_level(p) == "medium"

    def test_history_current_plus_memory_read_wildcard_is_medium(self):
        """GO-4: history current_chat + memory_read=('*',) => MEDIUM risk."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            memory_read=("*",),
        )
        assert compute_risk_level(p) == "medium"


# ==================================================================
# GO-LIST 5: Risk monotonicity invariant (parametrized)
# ==================================================================


_BASE_PERMISSIONS = [
    pytest.param(PermissionsConfig(), id="empty_base"),
    pytest.param(
        PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",))
        ),
        id="history_current_base",
    ),
    pytest.param(
        PermissionsConfig(memory_read=("long_term_facts",)),
        id="memory_read_specific_base",
    ),
]

_ADDED_PERMISSIONS_WITH_MIN_RISK = [
    pytest.param(PermissionsConfig(tools=("*",)), "high", id="add_tools_wildcard"),
    pytest.param(
        PermissionsConfig(tools=("web_search",)), "medium", id="add_tools_specific"
    ),
    pytest.param(
        PermissionsConfig(memory_write=("long_term_facts",)),
        "medium",
        id="add_memory_write",
    ),
    pytest.param(
        PermissionsConfig(memory_read=("*",)),
        "medium",
        id="add_memory_read_wildcard",
    ),
    pytest.param(
        PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",))
        ),
        "high",
        id="add_history_all_chats",
    ),
    pytest.param(PermissionsConfig(secrets_access=True), "high", id="add_secrets"),
    pytest.param(
        PermissionsConfig(
            network_access=NetworkAccessConfig(
                enabled=True, domains=("api.example.com",)
            )
        ),
        "high",
        id="add_network",
    ),
    pytest.param(
        PermissionsConfig(
            file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",))
        ),
        "high",
        id="add_file_access",
    ),
]


class TestRiskMonotonicity:
    """GO-5: Adding a permission must never downgrade risk level."""

    @pytest.mark.parametrize("base", _BASE_PERMISSIONS)
    @pytest.mark.parametrize("added, min_expected", _ADDED_PERMISSIONS_WITH_MIN_RISK)
    def test_adding_permission_never_downgrades_risk(self, base, added, min_expected):
        """Merged risk >= base risk AND merged risk >= min_expected."""
        merged = _merge_permissions(base, added)
        base_risk = compute_risk_level(base)
        merged_risk = compute_risk_level(merged)
        assert _RISK_RANK[merged_risk] >= _RISK_RANK[min_expected], (
            f"Expected >= {min_expected}, got {merged_risk} "
            f"(base={base_risk}, merged perms={merged})"
        )
        assert _RISK_RANK[merged_risk] >= _RISK_RANK[base_risk], (
            f"Risk downgraded from {base_risk} to {merged_risk} "
            f"when adding permissions (merged perms={merged})"
        )


# ==================================================================
# GO-LIST 6: Package type hierarchy monotonicity (parametrized)
# ==================================================================


class TestPackageMonotonicity:
    """GO-6: Adding a permission must never downgrade package type."""

    @pytest.mark.parametrize("base", _BASE_PERMISSIONS)
    @pytest.mark.parametrize("added, _min_risk", _ADDED_PERMISSIONS_WITH_MIN_RISK)
    def test_adding_permission_never_downgrades_package(self, base, added, _min_risk):
        """Merged package_type rank >= base package_type rank."""
        merged = _merge_permissions(base, added)
        base_pkg = compute_package_type(base)
        merged_pkg = compute_package_type(merged)
        assert _PACKAGE_RANK[merged_pkg] >= _PACKAGE_RANK[base_pkg], (
            f"Package downgraded from {base_pkg} to {merged_pkg} "
            f"when adding permissions (merged perms={merged})"
        )


# ==================================================================
# Pairwise Permission Matrix
# ==================================================================


class TestPairwisePermissionMatrix:
    """Full pairwise combination matrix for 2-permission scenarios.

    Policy decisions:
      history current + tools specific       => medium, tool_workflow
      history current + tools *              => high, code_plugin
      history current + memory_write         => medium, declarative_skill
      history current + memory_read *        => medium, declarative_skill
      memory_read specific + tools specific  => medium, tool_workflow
      memory_read * + tools specific         => medium, tool_workflow
      memory_write + tools *                 => high, code_plugin
    """

    def test_history_current_plus_tools_specific(self):
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            tools=("web_search",),
        )
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "tool_workflow"

    def test_history_current_plus_tools_wildcard(self):
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            tools=("*",),
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_history_current_plus_memory_write(self):
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            memory_write=("long_term_facts",),
        )
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "declarative_skill"

    def test_history_current_plus_memory_read_wildcard(self):
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
            memory_read=("*",),
        )
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "declarative_skill"

    def test_memory_read_specific_plus_tools_specific(self):
        p = PermissionsConfig(
            memory_read=("long_term_facts",),
            tools=("web_search",),
        )
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "tool_workflow"

    def test_memory_read_wildcard_plus_tools_specific(self):
        p = PermissionsConfig(
            memory_read=("*",),
            tools=("web_search",),
        )
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "tool_workflow"

    def test_memory_write_plus_tools_wildcard(self):
        p = PermissionsConfig(
            memory_write=("long_term_facts",),
            tools=("*",),
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_memory_write_plus_memory_read_wildcard(self):
        """memory_write + memory_read wildcard => medium, declarative_skill."""
        p = PermissionsConfig(
            memory_write=("long_term_facts",),
            memory_read=("*",),
        )
        assert compute_risk_level(p) == "medium"
        assert compute_package_type(p) == "declarative_skill"

    def test_history_power_plus_tools_specific(self):
        """history all_chats + tools specific => high, tool_workflow."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",)),
            tools=("web_search",),
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "tool_workflow"

    def test_history_power_plus_tools_wildcard(self):
        """history all_chats + tools * => high, code_plugin."""
        p = PermissionsConfig(
            history_access=HistoryAccessConfig(enabled=True, scopes=("all_chats",)),
            tools=("*",),
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_tools_specific_plus_file_access(self):
        """tools specific + file_access => high, code_plugin."""
        p = PermissionsConfig(
            tools=("web_search",),
            file_access=FileAccessConfig(enabled=True, scopes=("workspace:read",)),
        )
        assert compute_risk_level(p) == "high"
        assert compute_package_type(p) == "code_plugin"

    def test_memory_read_specific_plus_history_current(self):
        """memory_read specific + history current => low, declarative_skill."""
        p = PermissionsConfig(
            memory_read=("long_term_facts",),
            history_access=HistoryAccessConfig(enabled=True, scopes=("current_chat",)),
        )
        assert compute_risk_level(p) == "low"
        assert compute_package_type(p) == "declarative_skill"


# ==================================================================
# GO-LIST 7-8: Store-Finalizer Combination Tests
# ==================================================================


class TestStoreFinalizerCombinations:
    """Store-finalizer must correctly classify permission combinations."""

    def test_store_finalizer_history_plus_tools_wildcard(self, store):
        """GO-7: persist history+tools_wildcard => risk high, package code_plugin."""
        c = create_minimal_contract(
            name="finalizer-hist-tools-wild",
            phrases=("fhtw",),
            instruction="test finalizer combo",
        )
        c = replace(
            c,
            safety=SafetyConfig(allow_tools=True),
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
                tools=("*",),
            ),
        )
        persisted = store.persist(c, USER_ID)
        assert persisted.risk_level == "high"
        assert persisted.store_meta.package_type == "code_plugin"

    def test_store_finalizer_history_plus_memory_write(self, store):
        """GO-8: persist history+memory_write => risk medium."""
        c = create_minimal_contract(
            name="finalizer-hist-memwrite",
            phrases=("fhmw",),
            instruction="test finalizer combo 2",
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
                memory_write=("long_term_facts",),
            ),
        )
        persisted = store.persist(c, USER_ID)
        assert persisted.risk_level == "medium"
        assert persisted.store_meta.package_type == "declarative_skill"

    def test_store_finalizer_history_plus_tools_specific(self, store):
        """persist history+tools_specific => risk medium, package tool_workflow."""
        c = create_minimal_contract(
            name="finalizer-hist-tools-spec",
            phrases=("fhts",),
            instruction="test finalizer combo 3",
        )
        c = replace(
            c,
            safety=SafetyConfig(allow_tools=True),
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
                tools=("web_search",),
            ),
        )
        persisted = store.persist(c, USER_ID)
        assert persisted.risk_level == "medium"
        assert persisted.store_meta.package_type == "tool_workflow"

    def test_store_finalizer_memory_write_plus_tools_wildcard(self, store):
        """persist memory_write+tools_wildcard => risk high, package code_plugin."""
        c = create_minimal_contract(
            name="finalizer-memw-tools-wild",
            phrases=("fmtw",),
            instruction="test finalizer combo 4",
        )
        c = replace(
            c,
            safety=SafetyConfig(allow_tools=True),
            permissions=PermissionsConfig(
                memory_write=("long_term_facts",),
                tools=("*",),
            ),
        )
        persisted = store.persist(c, USER_ID)
        assert persisted.risk_level == "high"
        assert persisted.store_meta.package_type == "code_plugin"


# ==================================================================
# Finalize-only tests (without full store, via _finalize_security_metadata)
# ==================================================================


class TestFinalizeSecurityMetadataCombinations:
    """_finalize_security_metadata correctly handles permission combos."""

    def test_finalize_history_current_plus_tools_wildcard(self):
        c = create_minimal_contract(
            name="fin-combo-1", phrases=("fc1",), instruction="test"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
                tools=("*",),
            ),
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "high"
        assert finalized.store_meta.package_type == "code_plugin"

    def test_finalize_history_current_plus_memory_write(self):
        c = create_minimal_contract(
            name="fin-combo-2", phrases=("fc2",), instruction="test"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
                memory_write=("long_term_facts",),
            ),
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "medium"
        assert finalized.store_meta.package_type == "declarative_skill"

    def test_finalize_history_current_plus_tools_specific(self):
        c = create_minimal_contract(
            name="fin-combo-3", phrases=("fc3",), instruction="test"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
                tools=("web_search",),
            ),
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "medium"
        assert finalized.store_meta.package_type == "tool_workflow"

    def test_finalize_history_current_plus_memory_read_wildcard(self):
        c = create_minimal_contract(
            name="fin-combo-4", phrases=("fc4",), instruction="test"
        )
        c = replace(
            c,
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(
                    enabled=True, scopes=("current_chat",)
                ),
                memory_read=("*",),
            ),
        )
        finalized = _finalize_security_metadata(c)
        assert finalized.risk_level == "medium"
        assert finalized.store_meta.package_type == "declarative_skill"
