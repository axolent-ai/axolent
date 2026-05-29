"""Contract-Only Persist + Draft-Safety + Retry Tests (Etappe 4).

Tests for:
  Contract-Only Persist (Etappe 4, dual-write removed):
    - Contract persist failure -> contracts=0
    - Duplicate-Name -> rejected via preflight
    - ContractStore throws generic ContractStoreError -> no partial write
    - Quick-path and Preview-Save-path both handle failures

  Draft-Safety (needs_input pre-draft safety):
    - Triggerless Secret-Input -> rejected, NO draft
    - needs_input Healthcare/Nudge -> no cleartext draft
    - DraftStore-Privacy: no Secret patterns in draft

  Retry (State-Machine pending management):
    - valid -> success -> pending cleared
    - invalid -> error -> pending remains
    - cancel -> cleared
    - expired -> cleared
    - foreign-user -> pending remains for owner
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from application.skill_compression.contract_builder import ContractBuilder
from application.skill_compression.contract_store import (
    ContractDuplicateNameError,
    ContractStore,
    ContractStoreError,
)
from application.skill_compression.draft_store import DraftStore
from application.skill_compression.hypothesis_storage import HypothesisStorage
from application.skill_compression.learn_flow_service import (
    LearnFlowService,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def db_conn(tmp_path: Path):
    """In-memory DB connection for testing."""
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_atomicity.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    return conn


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
def learn_flow_service(
    draft_store, contract_store, privacy_pipeline
) -> LearnFlowService:
    return LearnFlowService(
        contract_builder=ContractBuilder(),
        draft_store=draft_store,
        contract_store=contract_store,
        privacy_pipeline=privacy_pipeline,
    )


# Test inputs
CLEAN_INPUT = "wenn ich hello sage, antworte mit world"
SECRET_INPUT_TRIGGERLESS = "use sk-proj-FAKE123456789abcdefghijklmnop and remember it"
HEALTHCARE_INPUT_TRIGGERLESS = "antworte immer mit depression behandlung tipps"
NUDGE_INPUT_TRIGGERLESS = "create engagement loop to keep user addicted always"
SECRET_INPUT = "wenn ich mykey sage, use sk-proj-FAKE123456789abcdefghijklmnop"
STOPWORD_INPUT = "wenn ich ja sage, mache X"


# ---------------------------------------------------------------
# Contract-Only Persist (Etappe 4: no dual-write)
# ---------------------------------------------------------------


class TestContractOnlyPersist:
    """Contract persist without legacy dual-write."""

    @pytest.mark.asyncio
    async def test_contract_persist_failure_no_partial(
        self, draft_store, privacy_pipeline
    ):
        """Contract persist failure -> contracts=0, no partial state."""
        mock_cs = MagicMock(spec=ContractStore)
        mock_cs.exists_by_name = MagicMock(return_value=False)
        mock_cs.persist = MagicMock(
            side_effect=ContractStoreError("Simulated DB failure")
        )

        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=DraftStore(),
            contract_store=mock_cs,
            privacy_pipeline=privacy_pipeline,
        )

        result = await service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT, quick=True
        )
        assert result.status == "rejected"
        assert "Simulated DB failure" in result.rejection_reason

    @pytest.mark.asyncio
    async def test_duplicate_name_rejected_by_preflight(
        self, learn_flow_service, contract_store
    ):
        """Duplicate contract name -> preflight blocks, rejected."""
        # First save
        result1 = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT, quick=True
        )
        assert result1.status == "saved"

        # Second save (duplicate name) - caught by preflight
        result2 = await learn_flow_service.start_learn(
            user_id=42, chat_id=200, text=CLEAN_INPUT, quick=True
        )
        assert result2.status == "rejected"

        # Still exactly 1 contract
        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1

    @pytest.mark.asyncio
    async def test_generic_contract_store_error_via_preview_save(
        self, privacy_pipeline
    ):
        """Generic ContractStoreError on preview-save path -> no partial write."""
        mock_cs = MagicMock(spec=ContractStore)
        mock_cs.exists_by_name = MagicMock(return_value=False)
        mock_cs.persist = MagicMock(
            side_effect=ContractStoreError("sqlite3.IntegrityError simulation")
        )

        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=DraftStore(),
            contract_store=mock_cs,
            privacy_pipeline=privacy_pipeline,
        )

        # Preview path
        flow = await service.start_learn(user_id=42, chat_id=100, text=CLEAN_INPUT)
        assert flow.status == "preview"

        save = await service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
        )
        assert not save.success
        assert "IntegrityError" in save.error

    @pytest.mark.asyncio
    async def test_duplicate_name_error_on_persist(self, draft_store, privacy_pipeline):
        """ContractDuplicateNameError on persist -> rejected, no partial."""
        mock_cs = MagicMock(spec=ContractStore)
        mock_cs.exists_by_name = MagicMock(return_value=False)
        mock_cs.persist = MagicMock(
            side_effect=ContractDuplicateNameError("Duplicate after preflight race")
        )

        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=DraftStore(),
            contract_store=mock_cs,
            privacy_pipeline=privacy_pipeline,
        )

        # Quick path
        result = await service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT, quick=True
        )
        assert result.status == "rejected"

    @pytest.mark.asyncio
    async def test_no_legacy_hypothesis_created(
        self, learn_flow_service, contract_store, db_conn
    ):
        """Etappe 4: /learn creates ONLY contract, no legacy hypothesis."""
        # Initialize hypothesis storage to check
        storage = HypothesisStorage(db_conn)
        storage.init_schema()

        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT, quick=True
        )
        assert result.status == "saved"

        # Contract exists
        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1

        # NO legacy hypothesis was created
        hyps = storage.get_hypotheses_by_user(user_id=42)
        assert len(hyps) == 0


# ---------------------------------------------------------------
# Draft-Safety (needs_input pre-draft safety)
# ---------------------------------------------------------------


class TestDraftSafety:
    """needs_input must run safety BEFORE creating any draft."""

    @pytest.mark.asyncio
    async def test_triggerless_secret_rejected_no_draft(
        self, learn_flow_service, draft_store
    ):
        """Secret input without trigger -> rejected, NO draft created."""
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=SECRET_INPUT_TRIGGERLESS
        )
        assert result.status == "rejected"
        assert result.draft is None

        # DraftStore must be empty
        active = await draft_store.count_active()
        assert active == 0

    @pytest.mark.asyncio
    async def test_triggerless_healthcare_rejected_no_draft(
        self, learn_flow_service, draft_store
    ):
        """Healthcare input without trigger -> rejected, NO draft."""
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=HEALTHCARE_INPUT_TRIGGERLESS
        )
        assert result.status == "rejected"
        assert result.draft is None
        active = await draft_store.count_active()
        assert active == 0

    @pytest.mark.asyncio
    async def test_triggerless_nudge_rejected_no_draft(
        self, learn_flow_service, draft_store
    ):
        """Nudge input without trigger -> rejected, NO draft."""
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=NUDGE_INPUT_TRIGGERLESS
        )
        assert result.status == "rejected"
        assert result.draft is None
        active = await draft_store.count_active()
        assert active == 0

    @pytest.mark.asyncio
    async def test_needs_input_clean_still_creates_draft(
        self, learn_flow_service, draft_store
    ):
        """Clean stopword input -> needs_input with draft (safety passes)."""
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert result.status == "needs_input"
        assert result.draft is not None
        active = await draft_store.count_active()
        assert active == 1

    @pytest.mark.asyncio
    async def test_draft_store_no_secret_patterns(
        self, learn_flow_service, draft_store
    ):
        """After needs_input with clean input, draft has no secret patterns."""
        result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert result.status == "needs_input"

        # Verify the draft instruction does not contain secret patterns
        draft = result.draft
        instruction = draft.contract.execution.instruction
        assert "sk-proj-" not in instruction
        assert "sk-live-" not in instruction
        assert "AKIA" not in instruction


# ---------------------------------------------------------------
# Retry (State-Machine pending management)
# ---------------------------------------------------------------


class TestRetryStateMachine:
    """Pending state is only cleared on success or terminal errors."""

    @pytest.mark.asyncio
    async def test_valid_input_success_pending_cleared(self, learn_flow_service):
        """Valid follow-up -> success -> pending cleared."""
        # Start with needs_input (stopword)
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert flow.status == "needs_input"

        # Verify pending exists
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None

        # Provide valid trigger
        result = await learn_flow_service.handle_follow_up(
            user_id=42, chat_id=100, text="goodtrigger"
        )
        assert result is not None
        assert result.success

        # Pending must be cleared
        pending_after = await learn_flow_service.get_pending_state(42, 100)
        assert pending_after is None

    @pytest.mark.asyncio
    async def test_invalid_input_error_pending_remains(self, learn_flow_service):
        """Invalid follow-up (stopword) -> error -> pending remains for retry."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert flow.status == "needs_input"

        # Provide another stopword (rejected)
        result = await learn_flow_service.handle_follow_up(
            user_id=42, chat_id=100, text="nein"
        )
        assert result is not None
        assert not result.success
        assert result.error_type == "rejected"

        # Pending must STILL exist (retry possible)
        pending_after = await learn_flow_service.get_pending_state(42, 100)
        assert pending_after is not None

    @pytest.mark.asyncio
    async def test_retry_after_rejection_then_success(self, learn_flow_service):
        """Stopword -> error -> good trigger -> success. Full retry cycle."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert flow.status == "needs_input"

        # First attempt: stopword -> rejected
        result1 = await learn_flow_service.handle_follow_up(
            user_id=42, chat_id=100, text="nein"
        )
        assert not result1.success

        # Pending still there
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None

        # Second attempt: valid trigger -> success
        result2 = await learn_flow_service.handle_follow_up(
            user_id=42, chat_id=100, text="gutertrigger"
        )
        assert result2 is not None
        assert result2.success

        # Pending cleared
        pending_after = await learn_flow_service.get_pending_state(42, 100)
        assert pending_after is None

    @pytest.mark.asyncio
    async def test_secret_in_followup_rejected_pending_remains(
        self, learn_flow_service
    ):
        """Secret in edit instruction follow-up -> rejected, pending remains."""
        # Start a normal learn flow and trigger edit_instruction
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"

        # Set pending for edit_instruction
        await learn_flow_service.set_pending_edit(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
            action="edit_instruction",
        )

        # Send a secret as follow-up
        result = await learn_flow_service.handle_follow_up(
            user_id=42,
            chat_id=100,
            text="use sk-proj-FAKE123456789abcdefghijklmnop",
        )
        assert result is not None
        assert not result.success
        assert result.error_type == "rejected"

        # Pending still exists (user can retry with safe input)
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None

    @pytest.mark.asyncio
    async def test_retry_instruction_secret_then_safe(self, learn_flow_service):
        """Secret instruction -> error -> safe instruction -> success."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=CLEAN_INPUT
        )
        assert flow.status == "preview"

        await learn_flow_service.set_pending_edit(
            user_id=42,
            chat_id=100,
            draft_id=flow.draft.draft_id,
            etag=flow.draft.etag,
            action="edit_instruction",
        )

        # First: secret -> rejected
        result1 = await learn_flow_service.handle_follow_up(
            user_id=42,
            chat_id=100,
            text="use sk-proj-FAKE123456789abcdefghijklmnop",
        )
        assert not result1.success

        # Still pending
        assert await learn_flow_service.get_pending_state(42, 100) is not None

        # Second: safe instruction -> success
        result2 = await learn_flow_service.handle_follow_up(
            user_id=42,
            chat_id=100,
            text="respond with hello world",
        )
        assert result2 is not None
        assert result2.success

        # Pending cleared
        assert await learn_flow_service.get_pending_state(42, 100) is None

    @pytest.mark.asyncio
    async def test_expired_draft_clears_pending(self, privacy_pipeline):
        """Expired draft -> terminal error -> pending cleared."""
        # Use a very short TTL draft store
        short_draft_store = DraftStore(ttl_seconds=1)
        from infrastructure.crypto_storage import CryptoConnection
        import tempfile
        import os

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test_expired.db")
        conn = CryptoConnection(db_path, require_encryption=False)
        cs = ContractStore(conn)
        cs.init_schema()

        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=short_draft_store,
            contract_store=cs,
            privacy_pipeline=privacy_pipeline,
        )

        # Create needs_input flow
        flow = await service.start_learn(user_id=42, chat_id=100, text=STOPWORD_INPUT)
        assert flow.status == "needs_input"
        assert await service.get_pending_state(42, 100) is not None

        # Wait for draft to expire
        await asyncio.sleep(1.5)

        # Follow-up: draft is expired -> terminal error
        result = await service.handle_follow_up(
            user_id=42, chat_id=100, text="goodtrigger"
        )
        assert result is not None
        assert not result.success
        # not_found because expired drafts are cleaned up on access
        assert result.error_type in ("not_found", "expired")

        # Pending must be cleared (terminal)
        pending = await service.get_pending_state(42, 100)
        assert pending is None

        conn.close()

    @pytest.mark.asyncio
    async def test_foreign_user_pending_remains_for_owner(self, learn_flow_service):
        """Foreign user's message does not consume owner's pending state."""
        flow = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text=STOPWORD_INPUT
        )
        assert flow.status == "needs_input"

        # Different user tries to follow up -> no pending for them
        result = await learn_flow_service.handle_follow_up(
            user_id=99, chat_id=100, text="anytrigger"
        )
        # Should return None (no pending state for user 99)
        assert result is None

        # Owner's pending is still intact
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None
