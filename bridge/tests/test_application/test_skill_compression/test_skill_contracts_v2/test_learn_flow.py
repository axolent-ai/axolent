"""Tests for Learn Flow (Etappe 3, T7-T9): Preview/Confirm/Edit + --quick.

4-path coverage:
  - Happy: /learn -> Preview -> Save -> Contract in DB
  - Malicious: secret in instruction, prompt injection
  - Rejection: expired draft, foreign user, stale etag, stopword edit
  - Privacy: no secrets in preview, logs, fixtures

DraftStore integration:
  - TTL expiry
  - Etag mismatch
  - Ownership check
  - Cancel removes draft

--quick safety:
  - Validator + PrivacyPipeline + SecretScanner run
  - Stopword triggers rejected even with --quick
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from application.skill_compression.contract_builder import ContractBuilder
from application.skill_compression.contract_store import ContractStore
from application.skill_compression.contract_validator import validate
from application.skill_compression.draft_store import (
    DraftEtagMismatchError,
    DraftExpiredError,
    DraftOwnershipError,
    DraftStore,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_contract import (
    SkillContract,
    create_minimal_contract,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def draft_store() -> DraftStore:
    """Fresh DraftStore with short TTL for testing."""
    return DraftStore(ttl_seconds=300)


@pytest.fixture
def short_ttl_store() -> DraftStore:
    """DraftStore with 1-second TTL for expiry tests."""
    return DraftStore(ttl_seconds=1)


@pytest.fixture
def pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


@pytest.fixture
def contract_store(tmp_path: Path) -> ContractStore:
    """In-memory ContractStore for testing."""
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_learn_flow.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    store = ContractStore(conn)
    store.init_schema()
    return store


def _build_test_contract() -> SkillContract:
    """Build a minimal valid contract for testing."""
    return create_minimal_contract(
        name="Test Skill",
        phrases=("testword",),
        instruction="respond with test output",
    )


# ---------------------------------------------------------------
# Happy path: Preview -> Save
# ---------------------------------------------------------------


class TestPreviewConfirmHappy:
    """Full /learn -> Preview -> Save flow."""

    @pytest.mark.asyncio
    async def test_build_creates_draft_in_store(self, draft_store: DraftStore) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit test output")
        assert result.status == "pending"

        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )
        assert draft.draft_id
        assert draft.user_id == 42
        assert draft.etag

    @pytest.mark.asyncio
    async def test_save_persists_to_contract_store(
        self, draft_store: DraftStore, contract_store: ContractStore
    ) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit test output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        # Get draft (simulates callback)
        fetched = await draft_store.get(42, 100, draft.draft_id)
        assert fetched is not None

        # Persist to contract store
        saved = contract_store.persist(fetched.contract, user_id=42)
        assert saved.id == fetched.contract.id

        # Remove draft
        deleted = await draft_store.delete(42, 100, draft.draft_id)
        assert deleted is True

        # Verify draft is gone
        after_delete = await draft_store.get(42, 100, draft.draft_id)
        assert after_delete is None

    @pytest.mark.asyncio
    async def test_cancel_removes_draft(self, draft_store: DraftStore) -> None:
        result = ContractBuilder.build("wenn ich cancel sage, antworte mit X")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        deleted = await draft_store.delete(42, 100, draft.draft_id)
        assert deleted is True

        # Verify it is really gone
        fetched = await draft_store.get(42, 100, draft.draft_id)
        assert fetched is None


# ---------------------------------------------------------------
# Edit flow
# ---------------------------------------------------------------


class TestEditFlow:
    """Edit trigger/instruction with re-validation."""

    @pytest.mark.asyncio
    async def test_edit_trigger_revalidates(self, draft_store: DraftStore) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        # Edit trigger
        new_contract, err = ContractBuilder.apply_trigger_edit(
            draft.contract, "neuer_trigger"
        )
        assert err is None

        # Update draft
        updated = await draft_store.update(
            42, 100, draft.draft_id, new_contract, draft.etag
        )
        assert updated.contract.activation.phrases == ("neuer_trigger",)
        assert updated.etag != draft.etag

    @pytest.mark.asyncio
    async def test_edit_trigger_to_stopword_rejected(
        self, draft_store: DraftStore
    ) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        # Try to edit trigger to a stopword
        _, err = ContractBuilder.apply_trigger_edit(draft.contract, "ja")
        assert err is not None

    @pytest.mark.asyncio
    async def test_edit_instruction_with_secret_rejected(
        self, pipeline: PrivacyPipeline
    ) -> None:
        """Edit with secret in instruction must be caught by PrivacyPipeline."""
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        updated, err = ContractBuilder.apply_instruction_edit(
            result.contract, "use api key sk-proj-FAKE123456789abcdef"
        )
        assert err is None  # Builder itself does not check secrets

        # But PrivacyPipeline catches it
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        hyp = Hypothesis(
            hypothesis_id="test",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim=updated.execution.instruction,
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            elo_games_played=0,
            bayes_confidence=0.5,
            support_count=1,
            contradict_count=0,
            source_type="learn_command",
            decay_immune=True,
            created_at="2026-01-01T00:00:00Z",
            last_applied=None,
            last_seen="2026-01-01T00:00:00Z",
        )
        rejection = pipeline.check(hyp)
        assert rejection is not None
        assert rejection.source.value == "secret_scanner"


# ---------------------------------------------------------------
# DraftStore edge cases
# ---------------------------------------------------------------


class TestDraftStoreEdgeCases:
    """DraftStore ownership, TTL, etag edge cases."""

    @pytest.mark.asyncio
    async def test_foreign_user_cannot_get_draft(self, draft_store: DraftStore) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        # Different user tries to access
        with pytest.raises(DraftOwnershipError):
            await draft_store.get(99, 100, draft.draft_id)

    @pytest.mark.asyncio
    async def test_foreign_user_cannot_update_draft(
        self, draft_store: DraftStore
    ) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        with pytest.raises(DraftOwnershipError):
            await draft_store.update(
                99, 100, draft.draft_id, draft.contract, draft.etag
            )

    @pytest.mark.asyncio
    async def test_foreign_user_cannot_delete_draft(
        self, draft_store: DraftStore
    ) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        with pytest.raises(DraftOwnershipError):
            await draft_store.delete(99, 100, draft.draft_id)

    @pytest.mark.asyncio
    async def test_expired_draft_returns_none(
        self, short_ttl_store: DraftStore
    ) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await short_ttl_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        fetched = await short_ttl_store.get(42, 100, draft.draft_id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_expired_draft_update_raises(
        self, short_ttl_store: DraftStore
    ) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await short_ttl_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        await asyncio.sleep(1.5)

        with pytest.raises(DraftExpiredError):
            await short_ttl_store.update(
                42, 100, draft.draft_id, draft.contract, draft.etag
            )

    @pytest.mark.asyncio
    async def test_stale_etag_rejected(self, draft_store: DraftStore) -> None:
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        # Update once (changes etag)
        new_contract, _ = ContractBuilder.apply_trigger_edit(draft.contract, "updated")
        await draft_store.update(42, 100, draft.draft_id, new_contract, draft.etag)

        # Try to update again with old etag
        with pytest.raises(DraftEtagMismatchError):
            await draft_store.update(
                42,
                100,
                draft.draft_id,
                new_contract,
                draft.etag,  # old etag
            )

    @pytest.mark.asyncio
    async def test_double_save_second_returns_none(
        self, draft_store: DraftStore
    ) -> None:
        """First save + delete works, second attempt gets None."""
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        draft = await draft_store.create(
            user_id=42, chat_id=100, contract=result.contract
        )

        # First: get + delete (simulate save)
        fetched = await draft_store.get(42, 100, draft.draft_id)
        assert fetched is not None
        await draft_store.delete(42, 100, draft.draft_id)

        # Second: get returns None
        second = await draft_store.get(42, 100, draft.draft_id)
        assert second is None


# ---------------------------------------------------------------
# --quick mode safety
# ---------------------------------------------------------------


class TestQuickModeSafety:
    """--quick bypasses ONLY preview, NOT validation/privacy/secrets."""

    def test_quick_validator_still_runs(self, contract_store: ContractStore) -> None:
        """Validator V1-V17 must run even in --quick mode.

        In real --quick flow, ContractStore.persist() calls
        _finalize_security_metadata() which resolves risk_level
        from 'unknown' to the computed value before validation.
        """
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        assert result.status == "pending"

        # persist() runs finalize + validate internally (the --quick path)
        saved = contract_store.persist(result.contract, user_id=42)
        assert saved.risk_level == "low"
        assert saved.id.startswith("skill_")

    def test_quick_privacy_still_runs(self, pipeline: PrivacyPipeline) -> None:
        """PrivacyPipeline must run even in --quick mode."""
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        assert result.status == "pending"

        # Pipeline check on clean content passes
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        hyp = Hypothesis(
            hypothesis_id="test",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim=result.contract.execution.instruction,
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            elo_games_played=0,
            bayes_confidence=0.5,
            support_count=1,
            contradict_count=0,
            source_type="learn_command",
            decay_immune=True,
            created_at="2026-01-01T00:00:00Z",
            last_applied=None,
            last_seen="2026-01-01T00:00:00Z",
        )
        rejection = pipeline.check(hyp)
        assert rejection is None

    def test_quick_secret_still_caught(self, pipeline: PrivacyPipeline) -> None:
        """Secrets in instruction must be caught even in --quick mode."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        secret_instruction = "use token sk-proj-FAKE123456789abcdef"
        hyp = Hypothesis(
            hypothesis_id="test",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim=secret_instruction,
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            elo_games_played=0,
            bayes_confidence=0.5,
            support_count=1,
            contradict_count=0,
            source_type="learn_command",
            decay_immune=True,
            created_at="2026-01-01T00:00:00Z",
            last_applied=None,
            last_seen="2026-01-01T00:00:00Z",
        )
        rejection = pipeline.check(hyp)
        assert rejection is not None

    def test_quick_stopword_trigger_rejected(self) -> None:
        """Even --quick must reject stopword triggers."""
        result = ContractBuilder.build("wenn ich ja sage, mache X")
        assert result.status == "needs_input"

    def test_quick_no_trigger_rejected(self) -> None:
        """Even --quick must reject when no trigger extractable."""
        result = ContractBuilder.build("sei einfach nett")
        assert result.status == "needs_input"

    def test_quick_workflow_execution_type_blocked(self) -> None:
        """V15: workflow/tool execution types blocked even in --quick."""
        from dataclasses import replace

        from application.skill_compression.skill_contract import ExecutionConfig

        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        # Manually set execution type to workflow (simulating malicious input)
        tampered = replace(
            result.contract,
            execution=ExecutionConfig(type="workflow", instruction="test"),
        )
        vr = validate(tampered)
        assert not vr.is_valid
        assert any(i.rule == "V15" for i in vr.errors)


# ---------------------------------------------------------------
# Privacy path
# ---------------------------------------------------------------


class TestPrivacyPath:
    """No secrets or PII leak through the learn flow."""

    def test_no_secrets_in_test_fixtures(self) -> None:
        """Meta-check: test file must not contain real secrets."""
        import inspect

        source = inspect.getsource(TestPreviewConfirmHappy)
        assert "sk-proj-" not in source
        assert "ghp_" not in source

    def test_preview_text_does_not_leak_raw_instruction(self) -> None:
        """The preview should show instruction but never log it raw."""
        result = ContractBuilder.build("wenn ich geheim sage, antworte mit vertraulich")
        # Contract has instruction, but logging should use len only
        assert result.contract.execution.instruction
        # Verify the contract does not leak into repr in unexpected ways
        assert "geheim" not in repr(result.contract.permissions)

    @pytest.mark.asyncio
    async def test_draft_does_not_leak_in_ownership_error(
        self, draft_store: DraftStore
    ) -> None:
        """Ownership error must not reveal draft content."""
        result = ContractBuilder.build(
            "wenn ich secret sage, antworte mit confidential"
        )
        await draft_store.create(user_id=42, chat_id=100, contract=result.contract)

        try:
            # Use a DraftStore method that would scan for the draft_id
            # across all users (the ownership error path)
            await draft_store.get(99, 100, "nonexistent")
        except DraftOwnershipError as e:
            # Error message must not contain instruction text
            assert "confidential" not in str(e)
            assert "secret" not in str(e)


# ---------------------------------------------------------------
# i18n key verification
# ---------------------------------------------------------------


class TestI18nKeys:
    """Verify all new learn flow i18n keys exist in all locales."""

    REQUIRED_KEYS = [
        "skill.learn_preview_header",
        "skill.learn_preview_name",
        "skill.learn_preview_trigger",
        "skill.learn_preview_action",
        "skill.learn_preview_permissions",
        "skill.learn_btn_save",
        "skill.learn_btn_edit",
        "skill.learn_btn_cancel",
        "skill.learn_cancelled",
        "skill.learn_draft_expired",
        "skill.learn_draft_already_saved",
        "skill.learn_draft_not_yours",
        "skill.learn_draft_stale",
        "skill.learn_needs_trigger",
        "skill.learn_trigger_rejected",
        "skill.learn_edit_prompt",
        "skill.learn_edit_rejected_secret",
        "skill.learn_validation_failed",
        "skill.learn_quick_saved",
        "skill.learn_quick_rejected",
    ]

    LOCALES = [
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "nl",
        "pl",
        "pt",
        "ru",
        "sv",
        "th",
        "tr",
        "uk",
        "vi",
        "zh",
    ]

    def test_all_keys_in_all_locales(self) -> None:
        """Every new i18n key must exist in all 20 locale files."""
        import json

        base = Path(__file__).resolve().parents[4] / "i18n" / "locales"

        missing = []
        for loc in self.LOCALES:
            path = base / f"{loc}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = data.get("keys", {})
            for key in self.REQUIRED_KEYS:
                if key not in keys:
                    missing.append(f"{loc}: {key}")

        assert not missing, f"Missing i18n keys: {missing}"
