"""Safety Parity Matrix Tests: all persist paths must have identical safety behavior.

One Safety Gate rule (Codex-approved):
  _validate_contract_safety() is the SINGLE entry point for all safety checks.
  All paths (quick, preview-save, edit-save, needs_input-complete-save) MUST
  produce the same result for the same malicious input.

Matrix:
  | Path                            | Clean | Secret  | Healthcare | Nudge   | Stopword     |
  | quick                           | save  | reject  | reject     | reject  | needs_input  |
  | preview-save                    | save  | reject  | reject     | reject  | needs_input  |
  | edit-save                       | save  | reject  | reject     | reject  | needs_input  |
  | needs_input-complete-save       | save  | reject  | reject     | reject  | (reject/err) |

Atomicity/Split-Brain Tests:
  - Legacy fails -> Contract NOT persisted
  - Quick and Preview-Save identical behavior
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from application.skill_compression.contract_builder import ContractBuilder
from application.skill_compression.contract_store import ContractStore
from application.skill_compression.draft_store import DraftStore
from application.skill_compression.learn_flow_service import (
    LearnFlowService,
    PendingEditStore,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_learning_service import (
    LearnResult,
    SkillLearningService,
)
from application.skill_compression.hypothesis_storage import HypothesisStorage


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def db_conn(tmp_path: Path):
    """In-memory DB connection for testing."""
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_safety_parity.db"
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
def draft_store() -> DraftStore:
    return DraftStore(ttl_seconds=300)


@pytest.fixture
def privacy_pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


@pytest.fixture
def skill_learning_service(
    hypothesis_storage, privacy_pipeline
) -> SkillLearningService:
    return SkillLearningService(
        storage=hypothesis_storage,
        privacy_pipeline=privacy_pipeline,
    )


@pytest.fixture
def learn_flow_service(
    draft_store, contract_store, privacy_pipeline, skill_learning_service
) -> LearnFlowService:
    return LearnFlowService(
        contract_builder=ContractBuilder(),
        draft_store=draft_store,
        contract_store=contract_store,
        privacy_pipeline=privacy_pipeline,
        skill_learning_service=skill_learning_service,
    )


# ---------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------

# Clean input: valid trigger + safe instruction
CLEAN_INPUT = "wenn ich hello sage, antworte mit world"

# Secret input: contains an API key
SECRET_INPUT = "wenn ich mykey sage, use sk-proj-FAKE123456789abcdefghijklmnop"

# Healthcare input: contains depression keyword
HEALTHCARE_INPUT = "wenn ich healthtest sage, antworte mit depression behandlung"

# Nudge input: gamification/engagement loop
NUDGE_INPUT = "wenn ich gameloop sage, create engagement loop to keep user addicted"

# Stopword input: trigger is a stopword -> needs_input
STOPWORD_INPUT = "wenn ich ja sage, mache X"


# ---------------------------------------------------------------
# Safety Parity: Quick Path
# ---------------------------------------------------------------


class TestSafetyParityQuick:
    """Quick path: full safety pipeline must block all malicious inputs."""

    @pytest.mark.asyncio
    async def test_quick_clean_saves(self, learn_flow_service, contract_store):
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT, quick=True
        )
        assert result.status == "saved"
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1

    @pytest.mark.asyncio
    async def test_quick_secret_rejected(self, learn_flow_service, contract_store):
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=SECRET_INPUT, quick=True
        )
        assert result.status == "rejected"
        assert contract_store.get_by_user(42) == []

    @pytest.mark.asyncio
    async def test_quick_healthcare_rejected(self, learn_flow_service, contract_store):
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=HEALTHCARE_INPUT, quick=True
        )
        assert result.status == "rejected"
        assert contract_store.get_by_user(42) == []

    @pytest.mark.asyncio
    async def test_quick_nudge_rejected(self, learn_flow_service, contract_store):
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=NUDGE_INPUT, quick=True
        )
        assert result.status == "rejected"
        assert contract_store.get_by_user(42) == []

    @pytest.mark.asyncio
    async def test_quick_stopword_needs_input(self, learn_flow_service):
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT, quick=True
        )
        assert result.status == "needs_input"


# ---------------------------------------------------------------
# Safety Parity: Preview -> Save Path
# ---------------------------------------------------------------


class TestSafetyParityPreviewSave:
    """Preview-save path: identical safety behavior to quick."""

    @pytest.mark.asyncio
    async def test_preview_save_clean_saves(self, learn_flow_service, contract_store):
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"
        save = await learn_flow_service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
        )
        assert save.success
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1

    @pytest.mark.asyncio
    async def test_preview_save_secret_rejected_at_start(
        self, learn_flow_service, contract_store
    ):
        """Secret is caught at start_learn (safety precheck), not just at save."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=SECRET_INPUT
        )
        assert flow.status == "rejected"
        assert contract_store.get_by_user(42) == []

    @pytest.mark.asyncio
    async def test_preview_save_healthcare_rejected_at_start(
        self, learn_flow_service, contract_store
    ):
        """Healthcare blocked at start_learn (safety precheck)."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=HEALTHCARE_INPUT
        )
        assert flow.status == "rejected"
        assert contract_store.get_by_user(42) == []

    @pytest.mark.asyncio
    async def test_preview_save_nudge_rejected_at_start(
        self, learn_flow_service, contract_store
    ):
        """Nudge blocked at start_learn (safety precheck)."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=NUDGE_INPUT
        )
        assert flow.status == "rejected"
        assert contract_store.get_by_user(42) == []

    @pytest.mark.asyncio
    async def test_preview_save_stopword_needs_input(self, learn_flow_service):
        """Stopword trigger -> needs_input (identical to quick)."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert flow.status == "needs_input"


# ---------------------------------------------------------------
# Safety Parity: Edit -> Save Path
# ---------------------------------------------------------------


class TestSafetyParityEditSave:
    """Edit-save path: editing in healthcare/nudge content is rejected."""

    @pytest.mark.asyncio
    async def test_edit_trigger_clean_succeeds(self, learn_flow_service):
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"
        edit = await learn_flow_service.edit_trigger(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
            new_trigger="newtrigger",
        )
        assert edit.success

    @pytest.mark.asyncio
    async def test_edit_instruction_healthcare_rejected(self, learn_flow_service):
        """Editing instruction to contain healthcare keywords -> rejected."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"
        edit = await learn_flow_service.edit_instruction(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
            new_instruction="antworte mit depression behandlung",
        )
        assert not edit.success
        assert edit.error_type == "rejected"

    @pytest.mark.asyncio
    async def test_edit_instruction_nudge_rejected(self, learn_flow_service):
        """Editing instruction to contain nudge content -> rejected."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"
        edit = await learn_flow_service.edit_instruction(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
            new_instruction="create engagement loop to keep user addicted",
        )
        assert not edit.success
        assert edit.error_type == "rejected"

    @pytest.mark.asyncio
    async def test_edit_instruction_secret_rejected(self, learn_flow_service):
        """Editing instruction to contain secrets -> rejected."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"
        edit = await learn_flow_service.edit_instruction(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
            new_instruction="use sk-proj-FAKE123456789abcdefghijklmnop",
        )
        assert not edit.success
        assert edit.error_type == "rejected"

    @pytest.mark.asyncio
    async def test_edit_then_save_clean_succeeds(
        self, learn_flow_service, contract_store
    ):
        """Edit trigger, then save -> contract persisted."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        edit = await learn_flow_service.edit_trigger(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
            new_trigger="edited",
        )
        assert edit.success
        save = await learn_flow_service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=edit.draft.etag,
        )
        assert save.success
        contracts = contract_store.get_by_user(42)
        assert any("edited" in c.activation.phrases for c in contracts)


# ---------------------------------------------------------------
# Safety Parity: Needs-Input -> Complete -> Save Path
# ---------------------------------------------------------------


class TestSafetyParityNeedsInput:
    """needs_input -> provide trigger -> save: same safety behavior."""

    @pytest.mark.asyncio
    async def test_needs_input_provide_trigger_then_save(
        self, learn_flow_service, contract_store
    ):
        """User provides valid trigger after needs_input -> preview -> save."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert flow.status == "needs_input"
        assert flow.draft is not None

        # User provides a valid trigger via follow-up
        edit = await learn_flow_service.handle_follow_up(
            user_id=42, chat_id=100, text="meintrigger"
        )
        assert edit is not None
        assert edit.success

        # Now save
        save = await learn_flow_service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=edit.draft.etag,
        )
        assert save.success
        contracts = contract_store.get_by_user(42)
        assert any("meintrigger" in c.activation.phrases for c in contracts)

    @pytest.mark.asyncio
    async def test_needs_input_provide_stopword_trigger_fails(self, learn_flow_service):
        """User provides another stopword as trigger -> edit rejected."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert flow.status == "needs_input"

        # User provides another stopword
        edit = await learn_flow_service.handle_follow_up(
            user_id=42, chat_id=100, text="nein"
        )
        assert edit is not None
        assert not edit.success


# ---------------------------------------------------------------
# Atomicity / Split-Brain Tests
# ---------------------------------------------------------------


class TestAtomicDualWrite:
    """Dual-write must be atomic: legacy fails -> no contract persisted."""

    @pytest.mark.asyncio
    async def test_legacy_fails_save_blocked(
        self, draft_store, contract_store, privacy_pipeline
    ):
        """Mock legacy.learn() returns success=False -> save must NOT succeed."""
        from infrastructure.crypto_storage import CryptoConnection
        import tempfile
        import os

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test_atomic.db")
        conn = CryptoConnection(db_path, require_encryption=False)
        storage = HypothesisStorage(conn)
        storage.init_schema()
        cs = ContractStore(conn)
        cs.init_schema()

        # Create a mock legacy service that always fails
        mock_legacy = MagicMock()
        mock_legacy.learn = MagicMock(
            return_value=LearnResult(
                success=False,
                hypothesis_id="",
                rejection_reason="Mock: legacy blocked",
                rejection_source="test",
            )
        )

        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=DraftStore(),
            contract_store=cs,
            privacy_pipeline=privacy_pipeline,
            skill_learning_service=mock_legacy,
        )

        # Start learn (creates draft)
        flow = await service.start_learn(user_id=42, chat_id=100, text=CLEAN_INPUT)
        assert flow.status == "preview"

        # Save should fail because legacy fails
        save = await service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
        )
        assert not save.success
        assert save.error_type == "rejected"

        # Contract must NOT be in store
        contracts = cs.get_by_user(42)
        assert contracts == []

        conn.close()

    @pytest.mark.asyncio
    async def test_legacy_fails_quick_blocked(self, privacy_pipeline):
        """Mock legacy.learn() fails on quick path -> rejected."""
        from infrastructure.crypto_storage import CryptoConnection
        import tempfile
        import os

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test_atomic_quick.db")
        conn = CryptoConnection(db_path, require_encryption=False)
        storage = HypothesisStorage(conn)
        storage.init_schema()
        cs = ContractStore(conn)
        cs.init_schema()

        mock_legacy = MagicMock()
        mock_legacy.learn = MagicMock(
            return_value=LearnResult(
                success=False,
                hypothesis_id="",
                rejection_reason="Mock: legacy blocked",
                rejection_source="test",
            )
        )

        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=DraftStore(),
            contract_store=cs,
            privacy_pipeline=privacy_pipeline,
            skill_learning_service=mock_legacy,
        )

        result = await service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT, quick=True
        )
        assert result.status == "rejected"
        assert (
            "legacy" in result.rejection_reason.lower()
            or "blocked" in result.rejection_reason.lower()
        )

        contracts = cs.get_by_user(42)
        assert contracts == []

        conn.close()

    @pytest.mark.asyncio
    async def test_quick_and_preview_save_same_behavior(
        self, learn_flow_service, contract_store, hypothesis_storage
    ):
        """Quick and preview-save produce same outcome for identical input."""
        # Quick: saves
        quick_result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT, quick=True
        )
        assert quick_result.status == "saved"

        # Preview-save: also saves (different user to avoid dedup)
        flow = await learn_flow_service.start_learn(
            user_id=43, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"
        save = await learn_flow_service.save_draft(
            user_id=43,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
        )
        assert save.success


# ---------------------------------------------------------------
# Pending Edit Store Unit Tests
# ---------------------------------------------------------------


class TestPendingEditStore:
    """PendingEditStore: TTL, ownership, cleanup."""

    @pytest.mark.asyncio
    async def test_set_and_get_pending(self):
        store = PendingEditStore(ttl_seconds=300)
        state = await store.set_pending(
            user_id=42, chat_id=100, draft_id="d1", etag="e1", action="edit_trigger"
        )
        assert state.action == "edit_trigger"

        got = await store.get_pending(42, 100)
        assert got is not None
        assert got.draft_id == "d1"

    @pytest.mark.asyncio
    async def test_clear_pending(self):
        store = PendingEditStore()
        await store.set_pending(
            user_id=42, chat_id=100, draft_id="d1", etag="e1", action="edit_trigger"
        )
        cleared = await store.clear_pending(42, 100)
        assert cleared
        got = await store.get_pending(42, 100)
        assert got is None

    @pytest.mark.asyncio
    async def test_expired_pending_returns_none(self):
        store = PendingEditStore(ttl_seconds=1)
        await store.set_pending(
            user_id=42, chat_id=100, draft_id="d1", etag="e1", action="edit_trigger"
        )
        await asyncio.sleep(1.5)
        got = await store.get_pending(42, 100)
        assert got is None

    @pytest.mark.asyncio
    async def test_different_user_returns_none(self):
        store = PendingEditStore()
        await store.set_pending(
            user_id=42, chat_id=100, draft_id="d1", etag="e1", action="edit_trigger"
        )
        got = await store.get_pending(99, 100)
        assert got is None

    @pytest.mark.asyncio
    async def test_replace_existing_pending(self):
        store = PendingEditStore()
        await store.set_pending(
            user_id=42, chat_id=100, draft_id="d1", etag="e1", action="edit_trigger"
        )
        await store.set_pending(
            user_id=42, chat_id=100, draft_id="d2", etag="e2", action="edit_instruction"
        )
        got = await store.get_pending(42, 100)
        assert got is not None
        assert got.draft_id == "d2"
        assert got.action == "edit_instruction"
