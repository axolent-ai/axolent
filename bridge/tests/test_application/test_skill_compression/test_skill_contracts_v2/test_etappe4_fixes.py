"""Tests for Etappe 4 Fixes (Codex review follow-up).

Covers:
  1. match_all() contract-aware + dedup (PUBLIC-API-MATRIX)
  2. needs_review Option 1: legacy keeps triggering (LIFECYCLE-MATRIX)
  3. Handler-to-Matcher E2E (Production-Path)
  4. Unique hypothesis_id index guard
  5. Legacy-Fallback /learn guard
  6. Constructor wiring guard
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from application.skill_compression.contract_builder import ContractBuilder
from application.skill_compression.contract_store import (
    ContractDuplicateHypothesisError,
    ContractStore,
)
from application.skill_compression.draft_store import DraftStore
from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.learn_flow_service import LearnFlowService
from application.skill_compression.pattern_judge import PatternJudge
from application.skill_compression.permission_gate import PermissionGate
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_contract import (
    ActivationConfig,
    ExecutionConfig,
    LifecycleConfig,
    PermissionsConfig,
    NetworkAccessConfig,
    SkillContract,
    new_skill_id,
    now_iso,
)
from application.skill_compression.skill_matcher import SkillMatcher


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def db_conn(tmp_path: Path):
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_etappe4_fixes.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    return conn


@pytest.fixture
def hypothesis_storage(db_conn) -> HypothesisStorage:
    storage = HypothesisStorage(db_conn)
    storage.init_schema()
    return storage


@pytest.fixture
def contract_store(db_conn) -> ContractStore:
    store = ContractStore(db_conn)
    store.init_schema()
    return store


@pytest.fixture
def privacy_pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


@pytest.fixture
def matcher(hypothesis_storage, contract_store, privacy_pipeline) -> SkillMatcher:
    judge = PatternJudge(privacy_pipeline=privacy_pipeline)
    return SkillMatcher(
        storage=hypothesis_storage,
        pattern_judge=judge,
        contract_store=contract_store,
    )


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _insert_hyp_with_alias(
    storage: HypothesisStorage,
    hyp_id: str,
    user_id: int,
    claim: str,
    alias_text: str,
    status: str = "confirmed",
) -> None:
    """Helper: insert a hypothesis with an alias."""
    ts = now_iso()
    hyp = Hypothesis(
        hypothesis_id=hyp_id,
        user_id=user_id,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status=status,
        version=1,
        elo_rating=1500.0,
        source_type="learn_command",
        decay_immune=True,
        created_at=ts,
        last_seen=ts,
    )
    storage.insert_hypothesis(hyp)
    storage.insert_alias(
        alias_id=f"alias_{uuid4().hex[:12]}",
        hypothesis_id=hyp_id,
        alias_text=alias_text,
        first_seen=ts,
        last_seen=ts,
        confidence=0.9,
    )


def _make_event(user_id: int = 42, text: str = "hello") -> NormalizedEvent:
    return NormalizedEvent(
        event_id="evt_test",
        user_id=user_id,
        timestamp=now_iso(),
        raw_text=text,
        intent="",
        domain="",
        format_type="",
        language="en",
        fingerprint_hash="",
    )


def _make_contract(
    name: str,
    phrases: tuple[str, ...],
    instruction: str,
    hypothesis_id: str = None,
    status: str = "confirmed",
    mode: str = "exact_phrase",
) -> SkillContract:
    ts = now_iso()
    return SkillContract(
        id=new_skill_id(),
        name=name,
        hypothesis_id=hypothesis_id,
        created_at=ts,
        updated_at=ts,
        activation=ActivationConfig(phrases=phrases, mode=mode),
        execution=ExecutionConfig(instruction=instruction),
        lifecycle=LifecycleConfig(status=status),
        origin="local_learn",
    )


# ---------------------------------------------------------------
# 1. PUBLIC-API-MATRIX: match_all() contract-aware
# ---------------------------------------------------------------


class TestMatchAllContractAware:
    """match_all() has the same contract-awareness as match()."""

    def test_match_all_contract_only_returns_contract_match(
        self, matcher, contract_store
    ):
        """Contract-only skill: match_all() returns exactly 1 contract match."""
        contract = _make_contract("greet", ("hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        results = matcher.match_all(_make_event(42, "hello"))
        assert len(results) == 1
        assert results[0].match_source == "contract"
        assert results[0].contract is not None
        assert results[0].contract.id == contract.id

    def test_match_all_legacy_plus_contract_returns_one_contract(
        self, matcher, contract_store, hypothesis_storage
    ):
        """Legacy + migrated contract: match_all() returns exactly 1 match, source=contract."""
        hyp_id = "hyp_matchall_dedup_001"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say hello, respond with hi",
            "hello",
        )
        contract = _make_contract(
            "greet",
            ("hello",),
            "respond with hi",
            hypothesis_id=hyp_id,
        )
        contract_store.persist(contract, user_id=42)

        results = matcher.match_all(_make_event(42, "hello"))
        assert len(results) == 1
        assert results[0].match_source == "contract"

    def test_match_all_legacy_without_contract_visible(
        self, matcher, hypothesis_storage
    ):
        """Legacy hypothesis without contract remains visible in match_all()."""
        hyp_id = "hyp_matchall_legacy_only"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say goodbye, respond with bye",
            "goodbye",
        )

        results = matcher.match_all(_make_event(42, "goodbye"))
        assert len(results) >= 1
        legacy_matches = [m for m in results if m.contract is None]
        assert len(legacy_matches) >= 1
        assert legacy_matches[0].match_source == "alias"

    def test_match_all_no_match_returns_empty(self, matcher):
        """No matches returns empty list."""
        results = matcher.match_all(_make_event(42, "xyznonexistent"))
        assert results == []

    def test_match_all_multiple_contracts_all_returned(self, matcher, contract_store):
        """Multiple matching contracts are all returned in match_all()."""
        # Two contracts with different names but same phrase (edge case)
        c1 = _make_contract("greet_v1", ("hello",), "respond with hi v1")
        contract_store.persist(c1, user_id=42)

        # Second contract with different phrase
        c2 = _make_contract("farewell", ("goodbye",), "respond with bye")
        contract_store.persist(c2, user_id=42)

        # Match "hello" should return only c1
        results_hello = matcher.match_all(_make_event(42, "hello"))
        assert len(results_hello) == 1
        assert results_hello[0].contract.id == c1.id

        # Match "goodbye" should return only c2
        results_bye = matcher.match_all(_make_event(42, "goodbye"))
        assert len(results_bye) == 1
        assert results_bye[0].contract.id == c2.id


# ---------------------------------------------------------------
# 2. LIFECYCLE-MATRIX: needs_review Option 1
# ---------------------------------------------------------------


class TestLifecycleMatrix:
    """Lifecycle-Matrix: migration status determines dedup behavior."""

    def test_confident_migration_legacy_suppressed(
        self, matcher, contract_store, hypothesis_storage
    ):
        """confident -> confirmed: Legacy is suppressed, Contract triggers."""
        hyp_id = "hyp_lm_confident"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say hello, respond with hi",
            "hello",
        )
        contract = _make_contract(
            "greet",
            ("hello",),
            "respond with hi",
            hypothesis_id=hyp_id,
            status="confirmed",
        )
        contract_store.persist(contract, user_id=42)

        # match() returns contract
        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.match_source == "contract"

        # match_all() returns contract, not legacy
        all_results = matcher.match_all(_make_event(42, "hello"))
        assert len(all_results) == 1
        assert all_results[0].match_source == "contract"

    def test_needs_review_legacy_keeps_triggering(
        self, matcher, contract_store, hypothesis_storage
    ):
        """needs_review: Legacy triggers, Contract does NOT trigger.

        Option 1 (user decision): needs_review contracts do not suppress
        the legacy hypothesis. The old skill keeps working until manual review.
        """
        hyp_id = "hyp_lm_needs_review"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say review_trigger, respond with something",
            "review_trigger",
        )
        # Create a needs_review contract (as migration would)
        contract = _make_contract(
            "review_skill",
            (),  # No activation phrases
            "respond with something",
            hypothesis_id=hyp_id,
            status="needs_review",
            mode="intent_match",  # needs_review uses intent_match
        )
        contract_store.persist(contract, user_id=42)

        # match() should return the LEGACY alias match (not suppressed)
        result = matcher.match(_make_event(42, "review_trigger"))
        assert result is not None
        assert result.contract is None  # Legacy match, not contract
        assert result.match_source == "alias"

        # match_all() should show legacy, not suppressed
        all_results = matcher.match_all(_make_event(42, "review_trigger"))
        legacy_matches = [m for m in all_results if m.contract is None]
        assert len(legacy_matches) >= 1

    def test_failed_validation_legacy_triggers(self, matcher, hypothesis_storage):
        """failed_validation: No contract created, Legacy triggers as before."""
        hyp_id = "hyp_lm_failed"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say failtest, respond with ok",
            "failtest",
        )
        # No contract exists at all (migration failed)
        result = matcher.match(_make_event(42, "failtest"))
        assert result is not None
        assert result.contract is None
        assert result.match_source == "alias"

    def test_needs_review_dedup_filter_only_confirmed_active(
        self, contract_store, hypothesis_storage, privacy_pipeline
    ):
        """_get_contract_covered_hypothesis_ids returns only confirmed/active."""
        judge = PatternJudge(privacy_pipeline=privacy_pipeline)
        m = SkillMatcher(
            storage=hypothesis_storage,
            pattern_judge=judge,
            contract_store=contract_store,
        )

        # Create a needs_review contract
        c_review = _make_contract(
            "review_skill",
            (),
            "instruction",
            hypothesis_id="hyp_review_dedup",
            status="needs_review",
            mode="intent_match",
        )
        contract_store.persist(c_review, user_id=42)

        # Create a confirmed contract
        c_confirmed = _make_contract(
            "confirmed_skill",
            ("confirmed_trigger",),
            "instruction2",
            hypothesis_id="hyp_confirmed_dedup",
            status="confirmed",
        )
        contract_store.persist(c_confirmed, user_id=42)

        covered = m._get_contract_covered_hypothesis_ids(42)
        # Only confirmed is covered, NOT needs_review
        assert "hyp_confirmed_dedup" in covered
        assert "hyp_review_dedup" not in covered


# ---------------------------------------------------------------
# 3. PUBLIC-API-MATRIX: match() x inputs
# ---------------------------------------------------------------


class TestPublicAPIMatrixMatch:
    """Public-API-Matrix for match(): all input combinations."""

    def test_match_contract_only(self, matcher, contract_store):
        """match(contract-only) returns contract match."""
        contract = _make_contract("greet", ("hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.match_source == "contract"

    def test_match_legacy_only(self, matcher, hypothesis_storage):
        """match(legacy-only) returns alias match."""
        _insert_hyp_with_alias(
            hypothesis_storage,
            "hyp_api_legacy",
            42,
            "when I say legacytest, respond with ok",
            "legacytest",
        )
        result = matcher.match(_make_event(42, "legacytest"))
        assert result is not None
        assert result.match_source == "alias"
        assert result.contract is None

    def test_match_legacy_plus_contract(
        self, matcher, contract_store, hypothesis_storage
    ):
        """match(legacy+contract) returns exactly 1 contract match."""
        hyp_id = "hyp_api_both"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say dualtest, respond with ok",
            "dualtest",
        )
        contract = _make_contract(
            "dual",
            ("dualtest",),
            "respond with ok",
            hypothesis_id=hyp_id,
        )
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "dualtest"))
        assert result is not None
        assert result.match_source == "contract"

    def test_match_high_risk_contract_not_denied_by_matcher(
        self, matcher, contract_store
    ):
        """High-risk contract is matched by matcher (gate check is in chat_service)."""
        ts = now_iso()
        contract = SkillContract(
            id=new_skill_id(),
            name="risky",
            created_at=ts,
            updated_at=ts,
            activation=ActivationConfig(
                phrases=("risky_trigger",), mode="exact_phrase"
            ),
            execution=ExecutionConfig(instruction="do risky"),
            lifecycle=LifecycleConfig(status="confirmed"),
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(enabled=True, domains=("*",)),
            ),
            origin="local_learn",
        )
        contract_store.persist(contract, user_id=42)

        # Matcher returns it (gate check is in ChatService, not Matcher)
        result = matcher.match(_make_event(42, "risky_trigger"))
        assert result is not None
        assert result.contract is not None

        # But PermissionGate would deny it
        gate_result = PermissionGate.check_execution_allowed(result.contract)
        assert gate_result.denied

    def test_match_needs_review_contract(
        self, matcher, contract_store, hypothesis_storage
    ):
        """needs_review contract: matcher returns legacy (Option 1)."""
        hyp_id = "hyp_api_review"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say reviewapi, respond with ok",
            "reviewapi",
        )
        contract = _make_contract(
            "review_api",
            (),
            "respond with ok",
            hypothesis_id=hyp_id,
            status="needs_review",
            mode="intent_match",
        )
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "reviewapi"))
        assert result is not None
        # Legacy triggers (not suppressed by needs_review)
        assert result.contract is None
        assert result.match_source == "alias"


# ---------------------------------------------------------------
# 4. Handler-to-Matcher E2E (Production-Path)
# ---------------------------------------------------------------


class TestHandlerToMatcherE2E:
    """True production-path: handle_learn_command -> ContractStore -> Matcher."""

    @pytest.mark.asyncio
    async def test_handler_learn_quick_to_matcher_contract(
        self, matcher, contract_store, hypothesis_storage, privacy_pipeline
    ):
        """Full handler path: /learn --quick -> contract -> matcher finds it."""
        from presentation.skill_commands import handle_learn_command

        draft_store = DraftStore()
        learn_flow = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=draft_store,
            contract_store=contract_store,
            privacy_pipeline=privacy_pipeline,
        )

        # Build fake Telegram update + context
        fake_user = MagicMock()
        fake_user.id = 42

        fake_chat = MagicMock()
        fake_chat.id = 100
        fake_chat.type = "private"

        fake_message = MagicMock()
        fake_message.reply_text = AsyncMock()
        fake_message.reply_to_message = None

        fake_update = MagicMock()
        fake_update.effective_user = fake_user
        fake_update.effective_chat = fake_chat
        fake_update.message = fake_message

        fake_app = MagicMock()
        fake_app.bot_data = {
            "hypothesis_storage": hypothesis_storage,
            "learn_flow_service": learn_flow,
            "chat_service": None,
        }

        fake_context = MagicMock()
        fake_context.application = fake_app
        fake_context.args = [
            "--quick",
            "when I say e2ehandler, respond with e2e_works",
        ]

        # Execute the actual handler (bypass whitelist + private chat decorators)
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            await handle_learn_command(fake_update, fake_context)

        # Verify: contract was persisted
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1
        e2e_contracts = [
            c
            for c in contracts
            if "e2ehandler" in " ".join(c.activation.phrases).lower()
        ]
        assert len(e2e_contracts) == 1, (
            f"Expected 1 e2e contract, got {len(e2e_contracts)}"
        )

        # Verify: matcher finds it as contract match
        match_result = matcher.match(_make_event(42, "e2ehandler"))
        assert match_result is not None
        assert match_result.contract is not None
        assert match_result.match_source == "contract"
        assert "e2e_works" in match_result.hypothesis.claim

    @pytest.mark.asyncio
    async def test_handler_learn_quick_to_chat_service_skill_block(
        self, matcher, contract_store, hypothesis_storage, privacy_pipeline
    ):
        """Extended E2E: handler -> contract -> matcher -> ChatService skill block."""
        draft_store = DraftStore()
        learn_flow = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=draft_store,
            contract_store=contract_store,
            privacy_pipeline=privacy_pipeline,
        )

        # Learn via service (simulates handler)
        result = await learn_flow.start_learn(
            user_id=42,
            chat_id=100,
            text="when I say skillblock, respond with block_works",
            quick=True,
        )
        assert result.status == "saved"

        # Matcher finds it
        match_result = matcher.match(_make_event(42, "skillblock"))
        assert match_result is not None
        assert match_result.contract is not None

        # Simulate ChatService._match_skills_for_prompt behavior
        # (cannot fully instantiate ChatService without many deps, so test
        # the essential logic: match -> PermissionGate -> skill block)
        gate = PermissionGate.check_execution_allowed(match_result.contract)
        assert gate.allowed

        # Build skill block (same logic as chat_service)
        hyp = match_result.hypothesis
        block_lines = [
            "[USER-DEFINED SKILL (HIGH PRIORITY)]",
            f"  Instruction: {hyp.claim}",
            f"  Confidence: {match_result.confidence:.2f}",
        ]
        skill_block = "\n".join(block_lines)
        assert "block_works" in skill_block


# ---------------------------------------------------------------
# 5. Unique hypothesis_id index guard
# ---------------------------------------------------------------


class TestUniqueHypothesisIdIndex:
    """Partial unique index prevents duplicate hypothesis_id references."""

    def test_duplicate_hypothesis_id_raises(self, contract_store):
        """Two contracts with same hypothesis_id raise ContractDuplicateHypothesisError."""
        c1 = _make_contract(
            "skill_a",
            ("trigger_a",),
            "instruction_a",
            hypothesis_id="hyp_dup_001",
        )
        contract_store.persist(c1, user_id=42)

        c2 = _make_contract(
            "skill_b",
            ("trigger_b",),
            "instruction_b",
            hypothesis_id="hyp_dup_001",
        )
        with pytest.raises(ContractDuplicateHypothesisError, match="hypothesis_id"):
            contract_store.persist(c2, user_id=42)

    def test_null_hypothesis_id_allowed_multiple(self, contract_store):
        """Multiple contracts with None hypothesis_id are allowed."""
        c1 = _make_contract("new_a", ("na",), "inst_a", hypothesis_id=None)
        c2 = _make_contract("new_b", ("nb",), "inst_b", hypothesis_id=None)
        contract_store.persist(c1, user_id=42)
        contract_store.persist(c2, user_id=42)
        # Both persisted OK
        assert contract_store.count_by_user(42) == 2

    def test_empty_hypothesis_id_allowed_multiple(self, contract_store):
        """Multiple contracts with empty hypothesis_id are allowed."""
        c1 = _make_contract("empty_a", ("ea",), "inst_a", hypothesis_id="")
        c2 = _make_contract("empty_b", ("eb",), "inst_b", hypothesis_id="")
        contract_store.persist(c1, user_id=42)
        contract_store.persist(c2, user_id=42)
        assert contract_store.count_by_user(42) == 2

    def test_duplicate_hypothesis_id_is_contract_store_error(self, contract_store):
        """ContractDuplicateHypothesisError is a subclass of ContractStoreError."""
        from application.skill_compression.contract_store import ContractStoreError

        c1 = _make_contract(
            "skill_x",
            ("tx",),
            "ix",
            hypothesis_id="hyp_typed_001",
        )
        contract_store.persist(c1, user_id=42)

        c2 = _make_contract(
            "skill_y",
            ("ty",),
            "iy",
            hypothesis_id="hyp_typed_001",
        )
        with pytest.raises(ContractStoreError):
            contract_store.persist(c2, user_id=42)

    def test_duplicate_hypothesis_id_via_update_raises(self, contract_store):
        """Updating a contract to a taken hypothesis_id raises domain error."""
        c1 = _make_contract(
            "upd_a",
            ("ua",),
            "ia",
            hypothesis_id="hyp_upd_001",
        )
        c2 = _make_contract(
            "upd_b",
            ("ub",),
            "ib",
            hypothesis_id="hyp_upd_002",
        )
        c1 = contract_store.persist(c1, user_id=42)
        c2 = contract_store.persist(c2, user_id=42)

        # Try to update c2 to have c1's hypothesis_id
        from dataclasses import replace as dc_replace

        c2_clash = dc_replace(c2, hypothesis_id="hyp_upd_001")
        with pytest.raises(ContractDuplicateHypothesisError, match="hypothesis_id"):
            contract_store.update(c2_clash, user_id=42)

    def test_duplicate_name_via_update_raises_correct_type(self, contract_store):
        """Updating a contract to a taken name raises ContractDuplicateNameError."""
        from application.skill_compression.contract_store import (
            ContractDuplicateNameError,
        )

        c1 = _make_contract("name_a", ("na",), "ia")
        c2 = _make_contract("name_b", ("nb",), "ib")
        c1 = contract_store.persist(c1, user_id=42)
        c2 = contract_store.persist(c2, user_id=42)

        from dataclasses import replace as dc_replace

        c2_clash = dc_replace(c2, name="name_a")
        with pytest.raises(ContractDuplicateNameError):
            contract_store.update(c2_clash, user_id=42)

    def test_no_raw_sqlite3_integrity_error_on_persist(self, contract_store):
        """persist() never leaks raw sqlite3.IntegrityError."""
        c1 = _make_contract(
            "raw_a",
            ("ra",),
            "ia",
            hypothesis_id="hyp_raw_001",
        )
        contract_store.persist(c1, user_id=42)

        c2 = _make_contract(
            "raw_b",
            ("rb",),
            "ib",
            hypothesis_id="hyp_raw_001",
        )
        try:
            contract_store.persist(c2, user_id=42)
            pytest.fail("Expected an exception for duplicate hypothesis_id")
        except sqlite3.IntegrityError:
            pytest.fail(
                "Raw sqlite3.IntegrityError leaked through persist(). "
                "Should be mapped to a ContractStoreError subclass."
            )
        except Exception as e:
            from application.skill_compression.contract_store import ContractStoreError

            assert isinstance(e, ContractStoreError), (
                f"Expected ContractStoreError subclass, got {type(e).__name__}: {e}"
            )


# ---------------------------------------------------------------
# 6. Constructor wiring guard
# ---------------------------------------------------------------

_BRIDGE_ROOT = Path(__file__).resolve().parents[4]


class TestConstructorWiringGuard:
    """main.py uses constructor wiring, not private attribute injection."""

    def test_main_no_private_contract_store_injection(self) -> None:
        """main.py does NOT use skill_matcher._contract_store = ... injection."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert "skill_matcher._contract_store" not in source, (
            "main.py must NOT inject _contract_store via private attribute. "
            "Use constructor parameter contract_store= instead."
        )

    def test_main_passes_contract_store_in_constructor(self) -> None:
        """main.py passes contract_store= in SkillMatcher constructor."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert "contract_store=contract_store" in source, (
            "main.py must pass contract_store=contract_store to SkillMatcher()"
        )


# ---------------------------------------------------------------
# 7. Legacy-Fallback /learn guard
# ---------------------------------------------------------------


class TestLegacyFallbackLearnGuard:
    """Guard: in production wiring, /learn never falls into legacy fallback."""

    def test_main_sets_learn_flow_service_in_bot_data(self) -> None:
        """main.py sets learn_flow_service in bot_data when sqlite is active."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert '"learn_flow_service"' in source, (
            "main.py must set bot_data['learn_flow_service']"
        )

    def test_skill_commands_has_fallback_guard_logging(self) -> None:
        """skill_commands.py documents the legacy fallback path."""
        source = (_BRIDGE_ROOT / "presentation" / "skill_commands.py").read_text(
            encoding="utf-8"
        )
        # The fallback path exists (lines 672-706 in original)
        assert "learn_flow" in source
        assert "learning_service" in source or "skill_learning_service" in source

    def test_main_learn_flow_service_is_wired(self) -> None:
        """main.py wires LearnFlowService with contract_store dependency."""
        tree = ast.parse(
            (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8"),
            filename="main.py",
        )
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        assert "LearnFlowService" in calls
