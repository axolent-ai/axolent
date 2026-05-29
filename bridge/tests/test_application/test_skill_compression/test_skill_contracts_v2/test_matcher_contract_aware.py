"""Tests for contract-aware SkillMatcher (Etappe 4, T10).

Tests:
  1. Contract is matched via exact-phrase activation
  2. Contract match carries the contract object (for PermissionGate)
  3. Dedup: migrated hypothesis + contract = 1 match, source=contract
  4. Legacy hypothesis without contract still matches
  5. German ss/sharp-s equivalence on contract phrases
  6. PermissionGate enforcement (4-path: happy, denied, legacy passthrough)
  7. Production-path end-to-end: /learn -> contract -> matcher -> trigger
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.skill_compression.contract_builder import ContractBuilder
from application.skill_compression.contract_store import ContractStore
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

    db_path = tmp_path / "test_matcher_contract.db"
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


def _insert_hyp_with_alias(
    storage: HypothesisStorage,
    hyp_id: str,
    user_id: int,
    claim: str,
    alias_text: str,
    status: str = "confirmed",
) -> None:
    """Helper: insert a hypothesis with an alias."""
    from uuid import uuid4

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
) -> SkillContract:
    ts = now_iso()
    return SkillContract(
        id=new_skill_id(),
        name=name,
        hypothesis_id=hypothesis_id,
        created_at=ts,
        updated_at=ts,
        activation=ActivationConfig(phrases=phrases, mode="exact_phrase"),
        execution=ExecutionConfig(instruction=instruction),
        lifecycle=LifecycleConfig(status=status),
        origin="local_learn",
    )


# ---------------------------------------------------------------
# T10: Contract matching
# ---------------------------------------------------------------


class TestContractMatching:
    """Contract-aware matcher finds contracts via activation phrases."""

    def test_contract_exact_phrase_match(self, matcher, contract_store):
        """Contract with matching activation phrase is found."""
        contract = _make_contract("greet", ("hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.contract is not None
        assert result.contract.id == contract.id
        assert result.match_source == "contract"
        assert result.confidence == 1.0

    def test_contract_case_insensitive(self, matcher, contract_store):
        """Contract matching is case-insensitive."""
        contract = _make_contract("greet", ("Hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.contract is not None

    def test_contract_german_ss_equivalence(self, matcher, contract_store):
        """German sharp-s and double-s are equivalent in contract matching."""
        contract = _make_contract("weiss", ("weiß",), "respond with white")
        contract_store.persist(contract, user_id=42)

        # User types "weiss" (double-s), contract has sharp-s
        result = matcher.match(_make_event(42, "weiss"))
        assert result is not None
        assert result.contract is not None

    def test_contract_no_match_wrong_user(self, matcher, contract_store):
        """Contract for user 42 does not match user 99."""
        contract = _make_contract("greet", ("hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(99, "hello"))
        assert result is None

    def test_contract_no_match_wrong_phrase(self, matcher, contract_store):
        """Contract does not match if phrase does not match."""
        contract = _make_contract("greet", ("hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "goodbye"))
        assert result is None

    def test_contract_active_status_matches(self, matcher, contract_store):
        """Active contracts are matchable."""
        contract = _make_contract(
            "greet", ("hello",), "respond with hi", status="active"
        )
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.requires_confirmation is False  # active = auto-apply

    def test_contract_confirmed_requires_confirmation(self, matcher, contract_store):
        """Confirmed contracts require user confirmation."""
        contract = _make_contract(
            "greet", ("hello",), "respond with hi", status="confirmed"
        )
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.requires_confirmation is True

    def test_contract_carries_hypothesis_for_backward_compat(
        self, matcher, contract_store
    ):
        """Contract match provides a synthetic hypothesis for chat_service."""
        contract = _make_contract("greet", ("hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.hypothesis is not None
        assert "respond with hi" in result.hypothesis.claim


# ---------------------------------------------------------------
# T10: Dedup (Contract > Legacy)
# ---------------------------------------------------------------


class TestMatcherDedup:
    """Dedup: migrated hypothesis + contract = 1 match from contract."""

    def test_dedup_migrated_hypothesis_suppressed(
        self, matcher, contract_store, hypothesis_storage
    ):
        """When hypothesis is migrated to contract, only contract match returns."""
        # Create legacy hypothesis with alias
        hyp_id = "hyp_test_dedup_001"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say hello, respond with hi",
            "hello",
        )

        # Create contract linking to same hypothesis
        contract = _make_contract(
            "greet",
            ("hello",),
            "respond with hi",
            hypothesis_id=hyp_id,
        )
        contract_store.persist(contract, user_id=42)

        # Match should return contract, not legacy
        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.contract is not None
        assert result.match_source == "contract"

    def test_legacy_without_contract_still_matches(self, matcher, hypothesis_storage):
        """Legacy hypothesis without a contract still matches via alias."""
        hyp_id = "hyp_test_legacy_only"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say goodbye, respond with bye",
            "goodbye",
        )

        result = matcher.match(_make_event(42, "goodbye"))
        assert result is not None
        assert result.contract is None  # Legacy match, no contract
        assert result.match_source == "alias"


# ---------------------------------------------------------------
# T10: PermissionGate enforcement (4-path)
# ---------------------------------------------------------------


class TestPermissionGateEnforcement:
    """PermissionGate checks are active for contract matches."""

    def test_local_skill_contract_allowed(self, matcher, contract_store):
        """Local skill (no permissions) passes PermissionGate."""
        contract = _make_contract("greet", ("hello",), "respond with hi")
        contract_store.persist(contract, user_id=42)

        result = matcher.match(_make_event(42, "hello"))
        assert result is not None
        assert result.contract is not None

        # Check via PermissionGate
        gate_result = PermissionGate.check_execution_allowed(result.contract)
        assert gate_result.allowed

    def test_high_risk_contract_denied_by_gate(self, contract_store):
        """High-risk contract (network_access) is denied by PermissionGate."""
        ts = now_iso()
        contract = SkillContract(
            id=new_skill_id(),
            name="risky_skill",
            created_at=ts,
            updated_at=ts,
            activation=ActivationConfig(phrases=("risky",), mode="exact_phrase"),
            execution=ExecutionConfig(instruction="do risky thing"),
            lifecycle=LifecycleConfig(status="confirmed"),
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(enabled=True, domains=("*",)),
            ),
            origin="local_learn",
        )
        contract_store.persist(contract, user_id=42)

        gate_result = PermissionGate.check_execution_allowed(contract)
        assert gate_result.denied

    def test_legacy_match_no_gate_check(self, matcher, hypothesis_storage):
        """Legacy hypothesis match has no contract, gate is not checked."""
        hyp_id = "hyp_test_no_gate"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say test, respond with ok",
            "test",
        )

        result = matcher.match(_make_event(42, "test"))
        assert result is not None
        assert result.contract is None  # No gate check for legacy


# ---------------------------------------------------------------
# Production-path end-to-end: /learn -> contract -> matcher -> trigger
# ---------------------------------------------------------------


class TestProductionPathEndToEnd:
    """/learn creates contract, matcher finds it, skill triggers."""

    @pytest.mark.asyncio
    async def test_learn_creates_contract_matcher_finds_it(
        self, matcher, contract_store, privacy_pipeline
    ):
        """End-to-end: /learn --quick -> contract -> matcher match."""
        draft_store = DraftStore()
        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=draft_store,
            contract_store=contract_store,
            privacy_pipeline=privacy_pipeline,
        )

        # /learn --quick creates a contract
        result = await service.start_learn(
            user_id=42,
            chat_id=100,
            text="wenn ich greetme sage, antworte mit willkommen",
            quick=True,
        )
        assert result.status == "saved"

        # Matcher finds the contract
        match = matcher.match(_make_event(42, "greetme"))
        assert match is not None
        assert match.contract is not None
        assert match.match_source == "contract"
        assert "willkommen" in match.hypothesis.claim

    @pytest.mark.asyncio
    async def test_learn_preview_save_matcher_finds_it(
        self, matcher, contract_store, privacy_pipeline
    ):
        """End-to-end: /learn -> preview -> save -> matcher match."""
        draft_store = DraftStore()
        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=draft_store,
            contract_store=contract_store,
            privacy_pipeline=privacy_pipeline,
        )

        # /learn creates a preview
        flow = await service.start_learn(
            user_id=42,
            chat_id=100,
            text="wenn ich farewell sage, antworte mit tschuess",
        )
        assert flow.status == "preview"

        # Save the draft
        save = await service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
        )
        assert save.success

        # Matcher finds the contract
        match = matcher.match(_make_event(42, "farewell"))
        assert match is not None
        assert match.contract is not None
        assert match.match_source == "contract"

    def test_matcher_without_contract_store_falls_back_to_legacy(
        self, hypothesis_storage, privacy_pipeline
    ):
        """Matcher without contract_store still works with legacy hypotheses."""
        judge = PatternJudge(privacy_pipeline=privacy_pipeline)
        matcher_no_contracts = SkillMatcher(
            storage=hypothesis_storage,
            pattern_judge=judge,
            # No contract_store
        )

        hyp_id = "hyp_test_no_store"
        _insert_hyp_with_alias(
            hypothesis_storage,
            hyp_id,
            42,
            "when I say fallback, respond with yes",
            "fallback",
        )

        result = matcher_no_contracts.match(_make_event(42, "fallback"))
        assert result is not None
        assert result.contract is None
        assert result.match_source == "alias"
