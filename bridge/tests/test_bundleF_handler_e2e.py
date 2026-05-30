"""Bundle F: /installskill Reply-Document Handler E2E test.

Tests the FULL handler path: Telegram Update with reply-to-document
-> handle_installskill_command -> SkillInstaller -> ContractStore -> SkillMatcher.

Item 1 of Bundle F (Codex mandatory recommendation).
Item 2 doku-lock: secret_scanner must not claim 'homoglyph' without qualifier.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.skill_compression.contract_store import ContractStore
from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.skill_contract import create_minimal_contract
from application.skill_compression.skill_installer import SkillInstaller
from application.skill_compression.skill_matcher import SkillMatcher
from infrastructure.crypto_storage import CryptoConnection


# =========================================================================
# Helpers
# =========================================================================


def _make_installskill_update(
    *,
    user_id: int = 42,
    chat_id: int = 42,
    file_name: str = "skill.json",
    file_size: int = 500,
    download_bytes: bytes | None = None,
) -> MagicMock:
    """Build a mock Telegram Update with reply-to-document for /installskill."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_user.language_code = "en"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message = MagicMock()
    update.message.text = "/installskill"
    update.message.message_id = 1
    update.message.reply_text = AsyncMock()
    update.callback_query = None

    # Build reply_to_message with document
    reply_msg = MagicMock()
    doc = MagicMock()
    doc.file_name = file_name
    doc.file_size = file_size

    tg_file = MagicMock()
    tg_file.download_as_bytearray = AsyncMock(
        return_value=bytearray(download_bytes or b"")
    )
    doc.get_file = AsyncMock(return_value=tg_file)

    reply_msg.document = doc
    update.message.reply_to_message = reply_msg

    return update


def _make_installskill_context(
    *,
    skill_installer: SkillInstaller | None = None,
    args: list[str] | None = None,
) -> MagicMock:
    """Build a mock Telegram context for /installskill."""
    context = MagicMock()
    context.args = args or []
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()
    context.application = MagicMock()

    chat_svc = MagicMock()
    chat_svc.get_chat_language = AsyncMock(return_value="en")

    context.application.bot_data = {
        "chat_service": chat_svc,
        "system_prompt": "test",
        "memory_service": MagicMock(),
        "persistent_provider": None,
        "process_pool": MagicMock(),
        "rate_limiter": MagicMock(),
        "bookmark_service": MagicMock(),
        "context_kernel": MagicMock(),
        "model_service": MagicMock(),
        "task_router": MagicMock(),
        "onboarding_storage": None,
        "hypothesis_storage": None,
        "skill_explainer": None,
        "import_orchestrator": None,
        "skill_learning_service": None,
        "language_enforcement": None,
        "skill_installer": skill_installer,
    }
    return context


def _setup_real_installer(tmp_path: Path):
    """Create a real ContractStore + SkillInstaller backed by a temp DB."""
    db_path = tmp_path / "test_handler_e2e.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    store = ContractStore(conn)
    store.init_schema()
    pipeline = MagicMock()
    pipeline.check.return_value = None
    installer = SkillInstaller(contract_store=store, privacy_pipeline=pipeline)
    return installer, store


# =========================================================================
# Item 1: /installskill Reply-Document Handler E2E
# =========================================================================


class TestInstallskillHandlerE2E:
    """Full E2E path: Telegram reply-to-document -> handler -> store -> matcher."""

    @pytest.fixture(autouse=True)
    def _allow_all(self):
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield

    async def test_reply_document_full_path(self, tmp_path: Path) -> None:
        """Happy path: valid JSON skill via reply-to-document is installed and matchable.

        Codex 7 steps:
        1. Real ContractStore + SkillInstaller in bot_data
        2. Fake Update with reply-to-document
        3. await handle_installskill_command(update, context)
        4. Assert: success-reply sent
        5. Assert: contract in store
        6. Assert: SkillMatcher.match(NormalizedEvent(raw_text=trigger)) finds it
        """
        from presentation.skill_commands import handle_installskill_command

        installer, store = _setup_real_installer(tmp_path)

        # Create a valid contract JSON
        contract = create_minimal_contract(
            name="Handler E2E Test Skill",
            phrases=("handler e2e test trigger",),
            instruction="Reply with: handler matched!",
        )
        contract_json_bytes = contract.to_json().encode("utf-8")

        # Build update with reply-to-document
        update = _make_installskill_update(
            file_name="skill.json",
            file_size=len(contract_json_bytes),
            download_bytes=contract_json_bytes,
        )
        context = _make_installskill_context(skill_installer=installer)

        # Call the real handler
        await handle_installskill_command(update, context)

        # Step 4: Assert success-reply was sent
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        # The success reply should contain the skill name
        assert "Handler E2E Test Skill" in reply_text

        # Step 5: Assert contract is in the store
        contracts = store.get_by_user(user_id=42)
        assert len(contracts) >= 1
        stored = [c for c in contracts if c.name == "Handler E2E Test Skill"]
        assert len(stored) == 1

        # Step 6: SkillMatcher.match() finds the installed skill
        matcher = SkillMatcher(
            storage=MagicMock(),
            pattern_judge=MagicMock(),
            contract_store=store,
        )
        event = NormalizedEvent(
            event_id="handler_e2e_1",
            user_id=42,
            raw_text="handler e2e test trigger",
        )
        match_result = matcher.match(event)
        assert match_result is not None, (
            "Matcher should find the handler-installed skill"
        )
        assert match_result.contract is not None
        assert match_result.contract.name == "Handler E2E Test Skill"
        assert match_result.confidence == 1.0

    async def test_reply_document_deep_json_rejected(self, tmp_path: Path) -> None:
        """Negativtest: deeply nested JSON document is rejected, no persist.

        Codex step 7: Deep-JSON -> fail-reply, no persist.
        """
        from presentation.skill_commands import handle_installskill_command

        installer, store = _setup_real_installer(tmp_path)

        # Create a deeply nested JSON payload (not a valid contract, also too deep)
        deep_json = "[" * 200 + '{"name": "evil"}' + "]" * 200
        deep_json_bytes = deep_json.encode("utf-8")

        update = _make_installskill_update(
            file_name="skill.json",
            file_size=len(deep_json_bytes),
            download_bytes=deep_json_bytes,
        )
        context = _make_installskill_context(skill_installer=installer)

        await handle_installskill_command(update, context)

        # Fail-reply should have been sent
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        # The reply should indicate failure (not success)
        assert "Handler E2E Test Skill" not in reply_text  # no success name

        # No contract should be persisted
        contracts = store.get_by_user(user_id=42)
        assert len(contracts) == 0

    async def test_reply_document_non_json_file_rejected(self, tmp_path: Path) -> None:
        """Non-.json file extension is rejected early."""
        from presentation.skill_commands import handle_installskill_command

        installer, store = _setup_real_installer(tmp_path)

        update = _make_installskill_update(
            file_name="skill.txt",
            file_size=100,
            download_bytes=b'{"name": "test"}',
        )
        context = _make_installskill_context(skill_installer=installer)

        await handle_installskill_command(update, context)

        # Reply sent (json-only error)
        update.message.reply_text.assert_called_once()
        # No contract persisted
        contracts = store.get_by_user(user_id=42)
        assert len(contracts) == 0

    async def test_reply_document_oversized_rejected(self, tmp_path: Path) -> None:
        """File over 100KB is rejected."""
        from presentation.skill_commands import handle_installskill_command

        installer, store = _setup_real_installer(tmp_path)

        update = _make_installskill_update(
            file_name="skill.json",
            file_size=200_000,  # over 100KB limit
            download_bytes=b"{}",
        )
        context = _make_installskill_context(skill_installer=installer)

        await handle_installskill_command(update, context)

        update.message.reply_text.assert_called_once()
        contracts = store.get_by_user(user_id=42)
        assert len(contracts) == 0


# =========================================================================
# Item 2 (optional): Doku-lock test for secret_scanner 'homoglyph' claim
# =========================================================================


class TestSecretScannerDokuLock:
    """secret_scanner must not claim 'homoglyph' without Phase 1.5 qualifier."""

    def test_no_unqualified_homoglyph_claim(self) -> None:
        """The word 'homoglyph' must not appear in secret_scanner without
        a negative qualifier (Phase 1.5, 'remain', 'not', etc.)."""
        src = Path(__file__).resolve().parents[1] / (
            "application/security/secret_scanner.py"
        )
        content = src.read_text(encoding="utf-8")
        # The word 'homoglyph' should either not appear at all,
        # or only in a comment that qualifies the limitation
        import re

        hits = list(re.finditer(r"homoglyph", content, re.IGNORECASE))
        for hit in hits:
            # Get the full line containing the hit
            line_start = content.rfind("\n", 0, hit.start()) + 1
            line_end = content.find("\n", hit.end())
            line = content[line_start:line_end]
            # Must be qualified with limitation language
            assert any(
                qualifier in line.lower()
                for qualifier in ("phase 1.5", "remain", "not", "limitation", "uts-39")
            ), f"Unqualified 'homoglyph' claim in secret_scanner.py: {line.strip()}"
