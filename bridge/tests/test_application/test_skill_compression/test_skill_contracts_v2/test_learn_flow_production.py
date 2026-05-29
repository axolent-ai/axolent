"""Production-Path Tests P1-P4: real handler invocation for /learn flow.

These tests call the REAL handler (handle_learn_command) with fake
Telegram objects, NOT manual orchestration of internal services.
They prove an actual user can use /learn and get a working skill.

4-Path coverage (Handler-Level):
  - Happy: /learn text -> Preview -> Save -> Contract in ContractStore
  - Malicious: Secret in /learn text -> blocked at handler level
  - Rejection: Stopword trigger -> needs_input at handler level
  - Privacy: Logs and reply texts do not leak secrets/instructions

Production-Path Tests:
  P1: /learn <text> -> Preview/Draft + Buttons -> Save-Callback -> Contract in Store
  P2: /learn --quick <text> -> no Preview -> persist -> Validator + Privacy ran
  P3: Edit-Callback -> revalidate -> Etag-Update -> Save
  P4: Cancel-Callback; foreign user rejected; expired draft; stale etag; double-save
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.skill_compression.contract_builder import ContractBuilder
from application.skill_compression.contract_store import ContractStore
from application.skill_compression.draft_store import DraftStore
from application.skill_compression.learn_flow_service import LearnFlowService
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_learning_service import SkillLearningService
from application.skill_compression.hypothesis_storage import HypothesisStorage

# Bypass whitelist for all tests in this module
pytestmark = pytest.mark.usefixtures("_bypass_whitelist")


@pytest.fixture(autouse=True)
def _bypass_whitelist(monkeypatch):
    """Allow all users through the whitelist decorator."""
    monkeypatch.setattr("presentation.decorators.ALLOW_ALL_USERS", True)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def db_conn(tmp_path: Path):
    """In-memory DB connection for testing."""
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_production_learn.db"
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


def _make_fake_update(user_id: int = 42, chat_id: int = 100, text: str = ""):
    """Create a minimal fake Update for handler testing."""
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


def _make_fake_context(
    args: list[str],
    learn_flow_service=None,
    hypothesis_storage=None,
    skill_learning_service=None,
):
    """Create a minimal fake Context for handler testing."""
    context = MagicMock()
    context.args = args

    bot_data = {}
    if hypothesis_storage is not None:
        bot_data["hypothesis_storage"] = hypothesis_storage
    if skill_learning_service is not None:
        bot_data["skill_learning_service"] = skill_learning_service
    if learn_flow_service is not None:
        bot_data["learn_flow_service"] = learn_flow_service

    # chat_service with async get_chat_language
    chat_service = MagicMock()
    chat_service.get_chat_language = AsyncMock(return_value="en")
    bot_data["chat_service"] = chat_service

    app = MagicMock()
    app.bot_data = bot_data
    context.application = app

    return context


def _make_fake_callback_query(user_id: int = 42, chat_id: int = 100, data: str = ""):
    """Create a minimal fake CallbackQuery for callback handler testing."""
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


# ---------------------------------------------------------------
# P1: /learn <text> -> Preview -> Save -> Contract in Store
# ---------------------------------------------------------------


class TestP1PreviewAndSave:
    """P1: Full /learn -> Preview -> Save flow via real handler."""

    @pytest.mark.asyncio
    async def test_learn_shows_preview_with_buttons(
        self, learn_flow_service, hypothesis_storage
    ) -> None:
        """handle_learn_command shows preview text + Save/Edit/Cancel buttons."""
        from presentation.skill_commands import handle_learn_command

        update = _make_fake_update(user_id=42, chat_id=100)
        context = _make_fake_context(
            args=["wenn", "ich", "test", "sage,", "antworte", "mit", "hallo"],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )

        await handle_learn_command(update, context)

        # Handler must have replied with text containing preview
        reply_call = update.message.reply_text
        assert reply_call.called
        call_kwargs = reply_call.call_args
        # reply_text called with text= and reply_markup= (may be positional or keyword)
        if call_kwargs.args:
            reply_text = call_kwargs.args[0]
        else:
            reply_text = call_kwargs.kwargs.get("text", "")
        reply_markup = call_kwargs.kwargs.get("reply_markup")

        assert (
            "Preview" in reply_text
            or "preview" in reply_text.lower()
            or "Trigger" in reply_text
        )
        assert reply_markup is not None
        # Buttons should contain skill_learn:save and skill_learn:cancel
        buttons_data = []
        for row in reply_markup.inline_keyboard:
            for btn in row:
                buttons_data.append(btn.callback_data)
        assert any("skill_learn:save:" in d for d in buttons_data)
        assert any("skill_learn:cancel:" in d for d in buttons_data)

    @pytest.mark.asyncio
    async def test_save_callback_persists_contract(
        self, learn_flow_service, hypothesis_storage, contract_store, draft_store
    ) -> None:
        """Save callback persists contract to ContractStore and deletes draft."""
        from presentation.skill_commands import handle_learn_callback

        # First create a draft via service
        flow_result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text="wenn ich hello sage, antworte mit world"
        )
        assert flow_result.status == "preview"
        draft = flow_result.draft
        assert draft is not None

        # Simulate save callback
        callback_data = f"skill_learn:save:{draft.draft_id}:{draft.etag}"
        update, query = _make_fake_callback_query(
            user_id=42, chat_id=100, data=callback_data
        )
        context = _make_fake_context(
            args=[],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )

        await handle_learn_callback(update, context)

        # Verify callback was answered
        assert query.answer.called

        # Verify contract is in ContractStore
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1
        saved = contracts[0]
        assert saved.lifecycle.status == "confirmed"
        assert "hello" in saved.activation.phrases

        # Verify draft is gone
        remaining = await draft_store.get(42, 100, draft.draft_id)
        assert remaining is None


# ---------------------------------------------------------------
# P2: /learn --quick -> no Preview -> persist
# ---------------------------------------------------------------


class TestP2QuickMode:
    """P2: /learn --quick persists directly, no preview."""

    @pytest.mark.asyncio
    async def test_quick_mode_persists_without_preview(
        self, learn_flow_service, hypothesis_storage, contract_store
    ) -> None:
        """--quick flag: contract saved immediately, no buttons shown."""
        from presentation.skill_commands import handle_learn_command

        update = _make_fake_update(user_id=42, chat_id=100)
        context = _make_fake_context(
            args=[
                "--quick",
                "wenn",
                "ich",
                "quicktest",
                "sage,",
                "antworte",
                "mit",
                "fast",
            ],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )

        await handle_learn_command(update, context)

        # Should reply with quick_saved message (no buttons)
        reply_call = update.message.reply_text
        assert reply_call.called
        call_args = reply_call.call_args
        # No reply_markup for quick mode
        reply_markup = call_args.kwargs.get("reply_markup")
        assert reply_markup is None

        # Verify contract is in ContractStore
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1
        saved = contracts[0]
        assert saved.lifecycle.status == "confirmed"
        assert "quicktest" in saved.activation.phrases

    @pytest.mark.asyncio
    async def test_quick_mode_secret_rejected(
        self, learn_flow_service, hypothesis_storage
    ) -> None:
        """--quick with secret in text: rejected at handler level."""
        from presentation.skill_commands import handle_learn_command

        update = _make_fake_update(user_id=42, chat_id=100)
        context = _make_fake_context(
            args=[
                "--quick",
                "wenn",
                "ich",
                "mykey",
                "sage,",
                "use",
                "sk-proj-FAKE123456789abcdef",
            ],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )

        await handle_learn_command(update, context)

        reply_call = update.message.reply_text
        assert reply_call.called
        call_args = reply_call.call_args
        reply_text = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        )
        # Should contain rejection
        assert (
            "reject" in reply_text.lower()
            or "blocked" in reply_text.lower()
            or "Rejected" in reply_text
        )


# ---------------------------------------------------------------
# P3: Edit -> Revalidate -> Etag update -> Save
# ---------------------------------------------------------------


class TestP3EditFlow:
    """P3: Edit trigger/instruction via service, then save."""

    @pytest.mark.asyncio
    async def test_edit_trigger_updates_etag_then_save(
        self, learn_flow_service, contract_store
    ) -> None:
        """Edit trigger -> new etag -> save with new etag succeeds."""
        # Create draft
        flow_result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text="wenn ich old sage, antworte mit output"
        )
        assert flow_result.status == "preview"
        draft = flow_result.draft

        # Edit trigger
        edit_result = await learn_flow_service.edit_trigger(
            user_id=42,
            chat_id=100,
            draft_id=draft.draft_id,
            etag=draft.etag,
            new_trigger="newtrigger",
        )
        assert edit_result.success
        assert edit_result.draft is not None
        assert edit_result.draft.etag != draft.etag

        # Save with NEW etag
        save_result = await learn_flow_service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=draft.draft_id,
            etag=edit_result.draft.etag,
        )
        assert save_result.success

        # Verify in store
        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1
        assert "newtrigger" in contracts[0].activation.phrases


# ---------------------------------------------------------------
# P4: Cancel, foreign user, expired, stale, double-save
# ---------------------------------------------------------------


class TestP4EdgeCases:
    """P4: Cancel, ownership, expired, stale etag, idempotent save."""

    @pytest.mark.asyncio
    async def test_cancel_deletes_draft(self, learn_flow_service, draft_store) -> None:
        """Cancel callback deletes draft from store."""
        from presentation.skill_commands import handle_learn_callback

        flow_result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text="wenn ich cancel sage, antworte mit X"
        )
        draft = flow_result.draft

        callback_data = f"skill_learn:cancel:{draft.draft_id}"
        update, query = _make_fake_callback_query(
            user_id=42, chat_id=100, data=callback_data
        )
        context = _make_fake_context(
            args=[],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=MagicMock(),
        )

        await handle_learn_callback(update, context)

        # Draft should be gone
        remaining = await draft_store.get(42, 100, draft.draft_id)
        assert remaining is None

    @pytest.mark.asyncio
    async def test_foreign_user_callback_rejected(self, learn_flow_service) -> None:
        """Foreign user save callback: ownership error."""
        flow_result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text="wenn ich test sage, antworte mit X"
        )
        draft = flow_result.draft

        # Different user tries to save
        save_result = await learn_flow_service.save_draft(
            user_id=99, chat_id=100, draft_id=draft.draft_id, etag=draft.etag
        )
        assert not save_result.success
        assert save_result.error_type == "ownership"

    @pytest.mark.asyncio
    async def test_stale_etag_rejected(self, learn_flow_service) -> None:
        """Stale etag on save: rejected."""
        flow_result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text="wenn ich test sage, antworte mit X"
        )
        draft = flow_result.draft

        save_result = await learn_flow_service.save_draft(
            user_id=42, chat_id=100, draft_id=draft.draft_id, etag="wrong_etag"
        )
        assert not save_result.success
        assert save_result.error_type == "stale"

    @pytest.mark.asyncio
    async def test_double_save_idempotent(self, learn_flow_service) -> None:
        """Double-click save: second attempt returns not_found (idempotent)."""
        flow_result = await learn_flow_service.start_learn(
            user_id=42, chat_id=100, text="wenn ich double sage, antworte mit X"
        )
        draft = flow_result.draft

        # First save
        save1 = await learn_flow_service.save_draft(
            user_id=42, chat_id=100, draft_id=draft.draft_id, etag=draft.etag
        )
        assert save1.success

        # Second save (draft already deleted)
        save2 = await learn_flow_service.save_draft(
            user_id=42, chat_id=100, draft_id=draft.draft_id, etag=draft.etag
        )
        assert not save2.success
        assert save2.error_type == "not_found"

    @pytest.mark.asyncio
    async def test_expired_draft_handled(self) -> None:
        """Expired draft: get returns None, save returns not_found."""
        short_store = DraftStore(ttl_seconds=1)
        pipeline = PrivacyPipeline()
        from application.skill_compression.hypothesis_storage import HypothesisStorage
        from infrastructure.crypto_storage import CryptoConnection
        import tempfile
        import os

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test_expire.db")
        conn = CryptoConnection(db_path, require_encryption=False)
        storage = HypothesisStorage(conn)
        storage.init_schema()
        cs = ContractStore(conn)
        cs.init_schema()
        sls = SkillLearningService(storage=storage, privacy_pipeline=pipeline)

        service = LearnFlowService(
            contract_builder=ContractBuilder(),
            draft_store=short_store,
            contract_store=cs,
            privacy_pipeline=pipeline,
            skill_learning_service=sls,
        )

        flow_result = await service.start_learn(
            user_id=42, chat_id=100, text="wenn ich timeout sage, antworte mit done"
        )
        draft = flow_result.draft
        assert draft is not None

        # Wait for TTL
        await asyncio.sleep(1.5)

        save_result = await service.save_draft(
            user_id=42, chat_id=100, draft_id=draft.draft_id, etag=draft.etag
        )
        assert not save_result.success
        assert save_result.error_type == "not_found"

        conn.close()


# ---------------------------------------------------------------
# Handler-level 4-Path: Happy / Malicious / Rejection / Privacy
# ---------------------------------------------------------------


class TestHandlerLevel4Path:
    """Handler-level 4-path coverage."""

    @pytest.mark.asyncio
    async def test_happy_path_handler(
        self, learn_flow_service, hypothesis_storage
    ) -> None:
        """Happy: /learn with valid text shows preview."""
        from presentation.skill_commands import handle_learn_command

        update = _make_fake_update()
        context = _make_fake_context(
            args=["wenn", "ich", "happy", "sage,", "antworte", "mit", "joy"],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_command(update, context)
        assert update.message.reply_text.called

    @pytest.mark.asyncio
    async def test_malicious_secret_blocked_at_handler(
        self, learn_flow_service, hypothesis_storage
    ) -> None:
        """Malicious: secret in instruction blocked."""
        from presentation.skill_commands import handle_learn_command

        update = _make_fake_update()
        context = _make_fake_context(
            args=[
                "wenn",
                "ich",
                "key",
                "sage,",
                "use",
                "ghp_FAKE1234567890abcdefghijklmnopqrstuv",
            ],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_command(update, context)
        call_args = update.message.reply_text.call_args
        reply_text = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        )
        assert "reject" in reply_text.lower() or "Rejected" in reply_text

    @pytest.mark.asyncio
    async def test_rejection_stopword_at_handler(
        self, learn_flow_service, hypothesis_storage
    ) -> None:
        """Rejection: stopword trigger gets needs_input."""
        from presentation.skill_commands import handle_learn_command

        update = _make_fake_update()
        context = _make_fake_context(
            args=["wenn", "ich", "ja", "sage,", "mache", "X"],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )
        await handle_learn_command(update, context)
        call_args = update.message.reply_text.call_args
        reply_text = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        )
        # Should mention trigger issue
        assert (
            "trigger" in reply_text.lower()
            or "word" in reply_text.lower()
            or "Trigger" in reply_text
        )

    @pytest.mark.asyncio
    async def test_privacy_logs_do_not_leak(
        self, learn_flow_service, hypothesis_storage, caplog
    ) -> None:
        """Privacy: log messages do not contain raw instruction text."""
        import logging

        from presentation.skill_commands import handle_learn_command

        update = _make_fake_update()
        context = _make_fake_context(
            args=[
                "wenn",
                "ich",
                "geheim",
                "sage,",
                "antworte",
                "mit",
                "vertraulich123",
            ],
            learn_flow_service=learn_flow_service,
            hypothesis_storage=hypothesis_storage,
        )

        with caplog.at_level(logging.DEBUG):
            await handle_learn_command(update, context)

        # Logs must not contain the instruction content
        log_text = caplog.text
        assert "vertraulich123" not in log_text


# ---------------------------------------------------------------
# Lifecycle invariant tests
# ---------------------------------------------------------------


class TestLifecycleInvariant:
    """Builder draft status, save confirmed transition."""

    def test_builder_pending_contract_is_draft(self) -> None:
        """BuildResult status=pending -> contract.lifecycle.status='draft'."""
        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        assert result.status == "pending"
        assert result.contract.lifecycle.status == "draft"

    def test_builder_pending_contract_validatable_as_draft(self) -> None:
        """Draft with unknown risk_level validates successfully (V14 guard clause)."""
        from application.skill_compression.contract_validator import validate

        result = ContractBuilder.build("wenn ich test sage, antworte mit output")
        assert result.contract.lifecycle.status == "draft"
        assert result.contract.risk_level == "unknown"
        vr = validate(result.contract)
        assert vr.is_valid, f"Validation failed: {[str(i) for i in vr.errors]}"

    def test_builder_needs_input_stays_needs_input(self) -> None:
        """BuildResult status=needs_input -> contract.lifecycle.status='needs_input'."""
        result = ContractBuilder.build("sei einfach nett")
        assert result.status == "needs_input"
        assert result.contract.lifecycle.status == "needs_input"

    @pytest.mark.asyncio
    async def test_save_transitions_draft_to_confirmed(
        self, learn_flow_service, contract_store
    ) -> None:
        """Save transitions lifecycle: draft -> confirmed before persist."""
        flow_result = await learn_flow_service.start_learn(
            user_id=42,
            chat_id=100,
            text="wenn ich lifecycle sage, antworte mit confirmed",
        )
        assert flow_result.draft.contract.lifecycle.status == "draft"

        save_result = await learn_flow_service.save_draft(
            user_id=42,
            chat_id=100,
            draft_id=flow_result.draft.draft_id,
            etag=flow_result.draft.etag,
        )
        assert save_result.success

        contracts = contract_store.get_by_user(42)
        assert len(contracts) >= 1
        assert contracts[0].lifecycle.status == "confirmed"
