"""Tests: Text Guard integration in /debate output.

Verifies that debate responses pass through TextGuardService
before being sent to the user, fixing ae/oe/ue to proper umlauts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.debate_orchestrator import DebateResult


def _make_update(user_id: int = 1, chat_id: int = 10, text: str = "") -> MagicMock:
    """Create a mocked Telegram update."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.reply_to_message = None
    return update


def _make_context(args: list[str] | None = None) -> MagicMock:
    """Create a mocked Telegram context with bot_data."""
    mock_router = MagicMock()
    mock_router.route = AsyncMock()

    from application.chat_service import ChatService

    svc = ChatService(provider_router=mock_router, memory_service=None)

    # Mock ContextKernel that returns a minimal ExecutionContext
    from application.execution import ContextKernel, ExecutionContext, ExecutionPlanner
    from application.language_resolver import LanguageContext

    mock_kernel = MagicMock(spec=ContextKernel)
    mock_exec_ctx = ExecutionContext(
        request_id="test-debate-001",
        user_id=1,
        chat_id=10,
        channel="telegram",
        language=LanguageContext(
            code="de",
            source="detection",
            confidence=0.9,
            switched_from=None,
            request_id="test-debate-001",
        ),
    )
    mock_kernel.build = AsyncMock(return_value=mock_exec_ctx)

    context = MagicMock()
    context.args = args or []
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()
    context.application = MagicMock()
    context.application.bot_data = {
        "chat_service": svc,
        "system_prompt": "Test prompt.",
        "memory_service": None,
        "persistent_provider": None,
        "process_pool": None,
        "rate_limiter": None,
        "bookmark_service": None,
        "context_kernel": mock_kernel,
        "execution_planner": ExecutionPlanner(),
    }
    return context


class TestDebateTextGuard:
    """Text Guard must fix diacritics in /debate output."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @patch("presentation.handlers.write_raw_audit")
    async def test_debate_output_has_correct_umlauts(
        self, mock_audit: MagicMock
    ) -> None:
        """Debate output with 'fuer/ueber/muessen' is fixed to proper umlauts."""
        from presentation.handlers import handle_debate_command

        # Build a DebateResult with broken umlauts in responses
        mock_result = DebateResult(
            question="Was muessen wir ueber KI wissen?",
            responses={
                "claude_persistent": (
                    "KI ist fuer viele Bereiche relevant. "
                    "Man muss ueber die Risiken nachdenken "
                    "und muessen ethische Grundsaetze beachten."
                ),
            },
            errors={},
            consensus_analysis="Alle Provider sind sich einig ueber die Relevanz.",
            duration_seconds=3.0,
            providers_queried=["claude_persistent"],
        )

        update = _make_update(user_id=1, chat_id=10, text="/debate Test")
        # First reply_text call returns status message, rest return None
        status_msg = MagicMock()
        status_msg.delete = AsyncMock()
        update.message.reply_text = AsyncMock(
            side_effect=[status_msg, None, None, None]
        )

        context = _make_context(
            args=["Was", "muessen", "wir", "ueber", "KI", "wissen?"]
        )

        with (
            patch(
                "application.debate_orchestrator.DebateOrchestrator.debate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "application.chat_service.ChatService.get_chat_language",
                new_callable=AsyncMock,
                return_value="de",
            ),
        ):
            await handle_debate_command(update, context)

        # Collect all text sent to user (skip the status message which is call 0)
        all_sent_text = ""
        for call in update.message.reply_text.call_args_list[1:]:
            if call[0]:
                all_sent_text += call[0][0]

        # Verify umlauts are corrected
        assert "für" in all_sent_text, f"Expected 'für' in output, got: {all_sent_text}"
        assert "über" in all_sent_text, (
            f"Expected 'über' in output, got: {all_sent_text}"
        )
        assert "müssen" in all_sent_text, (
            f"Expected 'müssen' in output, got: {all_sent_text}"
        )

        # Verify broken forms are gone
        assert "fuer" not in all_sent_text, "Broken 'fuer' still in output"
        assert "ueber" not in all_sent_text, "Broken 'ueber' still in output"
        assert "muessen" not in all_sent_text, "Broken 'muessen' still in output"
