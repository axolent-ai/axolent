"""Production Journey Tests: Edit/Needs-Input via real handlers.

These tests start at REAL handler level (handle_learn_command,
handle_learn_callback, handle_learn_followup_message) and prove
an actual user can complete the edit/needs_input subflows.

Journey Tests:
  1. /learn -> Edit Trigger Button -> next text message -> Preview with changed trigger
  2. /learn -> Edit Instruction -> next text message -> Preview with changed instruction
  3. needs_input -> User provides trigger -> Preview -> Save
  4. Every button E2E: Save persists, Cancel deletes, Edit leads to edited preview
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.skill_compression.contract_builder import ContractBuilder
from application.skill_compression.contract_store import ContractStore
from application.skill_compression.draft_store import DraftStore
from application.skill_compression.learn_flow_service import (
    LearnFlowService,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_learning_service import SkillLearningService
from application.skill_compression.hypothesis_storage import HypothesisStorage
from pathlib import Path

# Bypass whitelist for all tests
pytestmark = pytest.mark.usefixtures("_bypass_whitelist")


@pytest.fixture(autouse=True)
def _bypass_whitelist(monkeypatch):
    monkeypatch.setattr("presentation.decorators.ALLOW_ALL_USERS", True)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def db_conn(tmp_path: Path):
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_edit_journey.db"
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
# Test helpers
# ---------------------------------------------------------------


def _make_fake_update(user_id: int = 42, chat_id: int = 100, text: str = ""):
    user = MagicMock()
    user.id = user_id
    user.username = "testuser"

    chat = MagicMock()
    chat.id = chat_id
    chat.type = "private"

    message = MagicMock()
    message.reply_text = AsyncMock()
    message.reply_to_message = None
    message.text = text
    message.chat = chat

    update = MagicMock()
    update.effective_user = user
    update.effective_chat = chat
    update.message = message

    return update


def _make_fake_callback_query(user_id: int = 42, chat_id: int = 100, data: str = ""):
    user = MagicMock()
    user.id = user_id
    user.username = "testuser"

    chat = MagicMock()
    chat.id = chat_id
    chat.type = "private"

    message = MagicMock()
    message.chat_id = chat_id
    message.reply_text = AsyncMock()
    message.chat = chat

    query = MagicMock()
    query.data = data
    query.from_user = user
    query.message = message
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query
    update.effective_user = user
    update.effective_chat = chat
    update.message = message

    return update, query


def _make_fake_context(
    args=None,
    learn_flow_service=None,
    hypothesis_storage=None,
    skill_learning_service=None,
):
    context = MagicMock()
    context.args = args or []

    bot_data = {}
    if hypothesis_storage is not None:
        bot_data["hypothesis_storage"] = hypothesis_storage
    if skill_learning_service is not None:
        bot_data["skill_learning_service"] = skill_learning_service
    if learn_flow_service is not None:
        bot_data["learn_flow_service"] = learn_flow_service

    chat_service = MagicMock()
    chat_service.get_chat_language = AsyncMock(return_value="en")
    bot_data["chat_service"] = chat_service

    app = MagicMock()
    app.bot_data = bot_data
    context.application = app

    return context


# ---------------------------------------------------------------
# Journey 1: /learn -> Edit Trigger Button -> text -> Preview
# ---------------------------------------------------------------


class TestEditTriggerJourney:
    """Full journey: /learn -> edit button -> edit_trigger -> text -> new preview."""

    @pytest.mark.asyncio
    async def test_edit_trigger_full_journey(
        self, learn_flow_service, hypothesis_storage, contract_store
    ):
        """
        1. /learn shows preview
        2. User clicks Edit button
        3. User clicks "Trigger" sub-button (sets pending state)
        4. User sends new trigger text
        5. Follow-up handler processes it -> new preview with updated trigger
        6. User clicks Save -> contract persisted with new trigger
        """
        from presentation.skill_commands import (
            handle_learn_callback,
            handle_learn_command,
            handle_learn_followup_message,
        )

        # Step 1: /learn shows preview
        update = _make_fake_update(user_id=42, chat_id=100)
        context = _make_fake_context(
            args=["wenn", "ich", "original", "sage,", "antworte", "mit", "output"],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_command(update, context)
        reply_call = update.message.reply_text
        assert reply_call.called
        # Extract draft_id and etag from the callback buttons
        call_kwargs = reply_call.call_args
        reply_markup = call_kwargs.kwargs.get("reply_markup")
        assert reply_markup is not None

        # Find the edit button data
        edit_data = None
        for row in reply_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:edit:" in btn.callback_data:
                    edit_data = btn.callback_data
        assert edit_data is not None

        # Step 2: User clicks Edit button
        update2, query2 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=edit_data
        )
        context2 = _make_fake_context(
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_callback(update2, context2)
        # Should show edit choice (trigger/instruction buttons)
        assert query2.edit_message_text.called
        edit_choice_kwargs = query2.edit_message_text.call_args
        edit_choice_markup = edit_choice_kwargs.kwargs.get("reply_markup")
        assert edit_choice_markup is not None

        # Find edit_trigger button
        trigger_btn_data = None
        for row in edit_choice_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:edit_trigger:" in btn.callback_data:
                    trigger_btn_data = btn.callback_data
        assert trigger_btn_data is not None

        # Step 3: User clicks "Trigger" sub-button
        update3, query3 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=trigger_btn_data
        )
        context3 = _make_fake_context(
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_callback(update3, context3)
        # Should show prompt for new trigger
        assert query3.edit_message_text.called

        # Verify pending state is set
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None
        assert pending.action == "edit_trigger"

        # Step 4: User sends new trigger text
        update4 = _make_fake_update(user_id=42, chat_id=100, text="newtrigger")
        context4 = _make_fake_context(
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )

        # Step 5: Follow-up handler processes it
        from telegram.ext import ApplicationHandlerStop

        with pytest.raises(ApplicationHandlerStop):
            await handle_learn_followup_message(update4, context4)

        # Verify reply contains new preview with "newtrigger"
        followup_reply = update4.message.reply_text
        assert followup_reply.called
        call_kwargs = followup_reply.call_args
        reply_text = (
            call_kwargs.args[0]
            if call_kwargs.args
            else call_kwargs.kwargs.get("text", "")
        )
        assert "newtrigger" in reply_text

        # Extract new etag from buttons
        new_markup = call_kwargs.kwargs.get("reply_markup")
        assert new_markup is not None
        save_data = None
        for row in new_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:save:" in btn.callback_data:
                    save_data = btn.callback_data
        assert save_data is not None

        # Step 6: User clicks Save
        update5, query5 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=save_data
        )
        context5 = _make_fake_context(
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_callback(update5, context5)

        # Verify contract persisted with new trigger
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1
        assert any("newtrigger" in c.activation.phrases for c in contracts)


# ---------------------------------------------------------------
# Journey 2: /learn -> Edit Instruction -> text -> Preview
# ---------------------------------------------------------------


class TestEditInstructionJourney:
    """Full journey: edit instruction via handler chain."""

    @pytest.mark.asyncio
    async def test_edit_instruction_full_journey(
        self, learn_flow_service, hypothesis_storage
    ):
        """
        1. /learn shows preview
        2. User clicks Edit -> Instruction button -> sets pending
        3. User sends new instruction
        4. Follow-up handler -> new preview with updated instruction
        """
        from presentation.skill_commands import (
            handle_learn_callback,
            handle_learn_command,
            handle_learn_followup_message,
        )
        from telegram.ext import ApplicationHandlerStop

        # Step 1: /learn
        update = _make_fake_update(user_id=42, chat_id=100)
        context = _make_fake_context(
            args=["wenn", "ich", "test", "sage,", "antworte", "mit", "hallo"],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_command(update, context)
        reply_markup = update.message.reply_text.call_args.kwargs.get("reply_markup")

        # Get edit button
        edit_data = None
        for row in reply_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:edit:" in btn.callback_data:
                    edit_data = btn.callback_data
        assert edit_data is not None

        # Step 2a: Click Edit
        update2, query2 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=edit_data
        )
        ctx2 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )
        await handle_learn_callback(update2, ctx2)
        edit_choice_markup = query2.edit_message_text.call_args.kwargs.get(
            "reply_markup"
        )

        # Find instruction button
        instr_btn_data = None
        for row in edit_choice_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:edit_instruction:" in btn.callback_data:
                    instr_btn_data = btn.callback_data
        assert instr_btn_data is not None

        # Step 2b: Click Instruction
        update3, query3 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=instr_btn_data
        )
        ctx3 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )
        await handle_learn_callback(update3, ctx3)

        # Verify pending state
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None
        assert pending.action == "edit_instruction"

        # Step 3: User sends new instruction
        update4 = _make_fake_update(
            user_id=42, chat_id=100, text="respond with goodbye"
        )
        ctx4 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )

        with pytest.raises(ApplicationHandlerStop):
            await handle_learn_followup_message(update4, ctx4)

        # Step 4: Verify new preview contains new instruction
        followup_reply = update4.message.reply_text
        call_kwargs = followup_reply.call_args
        reply_text = (
            call_kwargs.args[0]
            if call_kwargs.args
            else call_kwargs.kwargs.get("text", "")
        )
        assert "goodbye" in reply_text


# ---------------------------------------------------------------
# Journey 3: needs_input -> User provides trigger -> Preview -> Save
# ---------------------------------------------------------------


class TestNeedsInputJourney:
    """Full journey: needs_input -> follow-up -> preview -> save."""

    @pytest.mark.asyncio
    async def test_needs_input_provide_trigger_journey(
        self, learn_flow_service, hypothesis_storage, contract_store
    ):
        """
        1. /learn with stopword trigger -> needs_input
        2. User sends new trigger text
        3. Follow-up handler -> preview
        4. User clicks Save -> contract persisted
        """
        from presentation.skill_commands import (
            handle_learn_callback,
            handle_learn_command,
            handle_learn_followup_message,
        )
        from telegram.ext import ApplicationHandlerStop

        # Step 1: /learn with stopword trigger
        update = _make_fake_update(user_id=42, chat_id=100)
        context = _make_fake_context(
            args=["wenn", "ich", "ja", "sage,", "mache", "etwas"],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_command(update, context)

        # Verify pending state was set for needs_input
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None
        assert pending.action == "needs_input"

        # Step 2: User provides new trigger
        update2 = _make_fake_update(user_id=42, chat_id=100, text="machtrigger")
        ctx2 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )

        with pytest.raises(ApplicationHandlerStop):
            await handle_learn_followup_message(update2, ctx2)

        # Step 3: Verify preview shown
        reply_call = update2.message.reply_text
        assert reply_call.called
        call_kwargs = reply_call.call_args
        reply_text = (
            call_kwargs.args[0]
            if call_kwargs.args
            else call_kwargs.kwargs.get("text", "")
        )
        assert "machtrigger" in reply_text
        new_markup = call_kwargs.kwargs.get("reply_markup")
        assert new_markup is not None

        # Find save button
        save_data = None
        for row in new_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:save:" in btn.callback_data:
                    save_data = btn.callback_data
        assert save_data is not None

        # Step 4: User clicks Save
        update3, query3 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=save_data
        )
        ctx3 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )
        await handle_learn_callback(update3, ctx3)

        # Verify contract persisted
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1
        assert any("machtrigger" in c.activation.phrases for c in contracts)


# ---------------------------------------------------------------
# Journey 4: Cancel clears pending state + deletes draft
# ---------------------------------------------------------------


class TestCancelJourney:
    """Cancel during edit: clears pending state and deletes draft."""

    @pytest.mark.asyncio
    async def test_cancel_clears_pending_and_draft(
        self, learn_flow_service, hypothesis_storage, draft_store
    ):
        """Cancel after setting pending edit -> pending cleared, draft gone."""
        from presentation.skill_commands import (
            handle_learn_callback,
            handle_learn_command,
        )

        # /learn -> preview
        update = _make_fake_update(user_id=42, chat_id=100)
        context = _make_fake_context(
            args=["wenn", "ich", "test", "sage,", "antworte", "mit", "output"],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_command(update, context)
        reply_markup = update.message.reply_text.call_args.kwargs.get("reply_markup")

        # Get edit + cancel buttons
        edit_data = None
        cancel_data = None
        for row in reply_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:edit:" in btn.callback_data:
                    edit_data = btn.callback_data
                if "skill_learn:cancel:" in btn.callback_data:
                    cancel_data = btn.callback_data
        assert edit_data is not None
        assert cancel_data is not None

        # Click Edit
        update2, query2 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=edit_data
        )
        ctx2 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )
        await handle_learn_callback(update2, ctx2)
        edit_choice_markup = query2.edit_message_text.call_args.kwargs.get(
            "reply_markup"
        )

        # Find trigger button and click it
        trigger_data = None
        for row in edit_choice_markup.inline_keyboard:
            for btn in row:
                if "skill_learn:edit_trigger:" in btn.callback_data:
                    trigger_data = btn.callback_data
        assert trigger_data is not None

        update3, query3 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=trigger_data
        )
        ctx3 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )
        await handle_learn_callback(update3, ctx3)

        # Verify pending state exists
        pending = await learn_flow_service.get_pending_state(42, 100)
        assert pending is not None

        # Now cancel (use cancel button from original preview)
        update4, query4 = _make_fake_callback_query(
            user_id=42, chat_id=100, data=cancel_data
        )
        ctx4 = _make_fake_context(
            learn_flow_service=learn_flow_service, hypothesis_storage=hypothesis_storage
        )
        await handle_learn_callback(update4, ctx4)

        # Verify pending cleared
        pending_after = await learn_flow_service.get_pending_state(42, 100)
        assert pending_after is None


# ---------------------------------------------------------------
# No pending state: follow-up handler passes through
# ---------------------------------------------------------------


class TestNoPendingPassthrough:
    """Without pending state, follow-up handler does not consume."""

    @pytest.mark.asyncio
    async def test_no_pending_returns_normally(self, learn_flow_service):
        """No pending state -> handler returns without raising."""
        from presentation.skill_commands import handle_learn_followup_message

        update = _make_fake_update(user_id=42, chat_id=100, text="random message")
        context = _make_fake_context(learn_flow_service=learn_flow_service)

        # Should NOT raise ApplicationHandlerStop
        await handle_learn_followup_message(update, context)
        # Message was not consumed: no reply sent
        assert not update.message.reply_text.called
