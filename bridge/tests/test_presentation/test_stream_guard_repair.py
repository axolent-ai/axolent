"""Tests for StreamGuard abort repair mechanism (HIGH-2 fix).

Verifies:
1. Guard abort triggers a non-streaming repair call (not hard discard).
2. User /stop still triggers hard discard (no repair).
3. The repair result is finalized as the streaming message.
4. task_meta._guard_abort flag differentiates the two abort types.

Production-path: tests go through _handle_message_streaming's actual
control flow (mocked providers, real branching logic).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.streaming_handler import StreamingSession


# ---------------------------------------------------------------------------
# Unit tests: StreamingSession guard_abort vs cancel
# ---------------------------------------------------------------------------


class TestStreamingSessionAbortTypes:
    """StreamingSession must distinguish guard_abort from user cancel."""

    def test_cancel_sets_cancelled_without_guard_flag(self) -> None:
        """cancel() sets is_cancelled=True but is_guard_abort=False."""
        msg = MagicMock()
        session = StreamingSession(message=msg)
        session.cancel()
        assert session.is_cancelled is True
        assert session.is_guard_abort is False

    def test_guard_abort_sets_both_flags(self) -> None:
        """guard_abort() sets is_cancelled=True AND is_guard_abort=True."""
        msg = MagicMock()
        session = StreamingSession(message=msg)
        session.guard_abort()
        assert session.is_cancelled is True
        assert session.is_guard_abort is True

    def test_default_state_is_not_cancelled(self) -> None:
        """Fresh session is neither cancelled nor guard-aborted."""
        msg = MagicMock()
        session = StreamingSession(message=msg)
        assert session.is_cancelled is False
        assert session.is_guard_abort is False


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------


def _make_envelope(user_id: int = 42, chat_id: int = 123) -> Any:
    """Create a RequestEnvelope for testing."""
    from application.execution import RequestEnvelope

    return RequestEnvelope(
        user_id=user_id,
        chat_id=chat_id,
        raw_text="Hello",
        timestamp_utc=datetime.now(timezone.utc),
        channel="telegram",
        request_id="test-req-guard",
    )


def _make_update(chat_id: int = 123) -> MagicMock:
    """Create a minimal Telegram Update mock."""
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.send_action = AsyncMock()
    return update


def _make_context(chat_service: Any) -> MagicMock:
    """Create a minimal bot context mock."""
    context = MagicMock()
    context.bot_data = {
        "chat_service": chat_service,
        "language_enforcement": MagicMock(),
    }
    return context


def _exec_ctx_mock() -> MagicMock:
    """Create a mock ExecutionContext."""
    ctx = MagicMock()
    ctx.language = MagicMock(code="de", source="user", confidence=1.0)
    return ctx


def _exec_plan_mock() -> MagicMock:
    """Create a mock ExecutionPlan."""
    plan = MagicMock()
    plan.task_type = "chat"
    plan.provider_chain = ["claude_persistent"]
    plan.memory_used = set()
    plan.verifier_profile = "default"
    plan.audit_required = False
    return plan


# ---------------------------------------------------------------------------
# Integration test: task_meta._guard_abort controls repair vs discard
# ---------------------------------------------------------------------------


class TestGuardAbortTriggersRepair:
    """Verify the handler dispatches repair on guard abort."""

    @pytest.fixture
    def mock_chat_service(self) -> MagicMock:
        """ChatService mock with process_user_message returning a repair."""
        svc = MagicMock()
        repair_result = MagicMock()
        repair_result.response = "Repaired answer in correct language"
        svc.process_user_message = AsyncMock(return_value=repair_result)
        return svc

    async def test_guard_abort_task_meta_flag_triggers_repair_path(
        self,
        mock_chat_service: MagicMock,
    ) -> None:
        """When task_meta has _guard_abort=True, repair is called."""
        from presentation.handlers import _handle_message_streaming

        async def _fake_streaming(*args: Any, **kwargs: Any) -> tuple:
            """Simulate a stream that immediately gets guard-aborted."""
            cancel_event = kwargs.get("cancel_event")
            task_meta: dict[str, Any] = {
                "_guard_abort": True,
                "_guard_abort_text": "Wrong language partial",
                "_language_code": "de",
                "_provider_name": "claude_persistent",
            }

            async def _empty_stream():
                if cancel_event is not None:
                    cancel_event.set()
                return
                yield  # noqa: F841

            return _empty_stream(), 0, task_meta

        mock_chat_service.process_user_message_streaming = AsyncMock(
            side_effect=_fake_streaming
        )

        with (
            patch(
                "presentation.handlers.create_streaming_message",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "presentation.handlers.finalize_streaming",
                new=AsyncMock(),
            ) as mock_finalize,
            patch(
                "presentation.handlers.abort_streaming",
                new=AsyncMock(),
            ),
            patch(
                "presentation.handlers._get_system_prompt",
                return_value="system prompt",
            ),
            patch(
                "presentation.handlers._get_context_kernel",
                return_value=MagicMock(build=AsyncMock(return_value=_exec_ctx_mock())),
            ),
            patch(
                "presentation.handlers._get_execution_planner",
                return_value=MagicMock(
                    plan_chat=MagicMock(return_value=_exec_plan_mock())
                ),
            ),
            patch("presentation.handlers.write_raw_audit"),
            patch(
                "presentation.handlers._typing_keepalive",
                return_value=asyncio.sleep(0),
            ),
            patch(
                "application.text_guard_service.TextGuardService",
                return_value=MagicMock(
                    get_streaming_guard=MagicMock(return_value=None),
                    get_guard=MagicMock(return_value=None),
                ),
            ),
            patch("application.status_manager.SHOW_STATUS_UPDATES", False),
        ):
            await _handle_message_streaming(
                update=_make_update(),
                context=_make_context(mock_chat_service),
                chat_service=mock_chat_service,
                persistent_provider=MagicMock(),
                user_id=42,
                chat_id=123,
                username="testuser",
                text="Hello",
                reply_to_text=None,
                envelope=_make_envelope(),
            )

            # Verify: process_user_message was called (the repair path)
            mock_chat_service.process_user_message.assert_called_once()
            # Verify: finalize_streaming was called with the repaired text
            mock_finalize.assert_called_once()
            call_args = mock_finalize.call_args
            assert call_args[0][1] == "Repaired answer in correct language"

    async def test_user_stop_does_not_trigger_repair(self) -> None:
        """When cancel is from /stop (no _guard_abort), no repair call."""
        from presentation.handlers import _handle_message_streaming

        mock_chat_service = MagicMock()
        mock_chat_service.process_user_message = AsyncMock()

        async def _fake_streaming(*args: Any, **kwargs: Any) -> tuple:
            """Simulate a stream that gets user-cancelled."""
            cancel_event = kwargs.get("cancel_event")
            task_meta: dict[str, Any] = {
                "_language_code": "de",
                "_provider_name": "claude_persistent",
            }

            async def _empty_stream():
                if cancel_event is not None:
                    cancel_event.set()
                return
                yield  # noqa: F841

            return _empty_stream(), 0, task_meta

        mock_chat_service.process_user_message_streaming = AsyncMock(
            side_effect=_fake_streaming
        )

        with (
            patch(
                "presentation.handlers.create_streaming_message",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "presentation.handlers.finalize_streaming",
                new=AsyncMock(),
            ),
            patch(
                "presentation.handlers.abort_streaming",
                new=AsyncMock(),
            ),
            patch(
                "presentation.handlers._get_system_prompt",
                return_value="system prompt",
            ),
            patch(
                "presentation.handlers._get_context_kernel",
                return_value=MagicMock(build=AsyncMock(return_value=_exec_ctx_mock())),
            ),
            patch(
                "presentation.handlers._get_execution_planner",
                return_value=MagicMock(
                    plan_chat=MagicMock(return_value=_exec_plan_mock())
                ),
            ),
            patch("presentation.handlers.write_raw_audit"),
            patch(
                "presentation.handlers._typing_keepalive",
                return_value=asyncio.sleep(0),
            ),
            patch(
                "application.text_guard_service.TextGuardService",
                return_value=MagicMock(
                    get_streaming_guard=MagicMock(return_value=None),
                    get_guard=MagicMock(return_value=None),
                ),
            ),
            patch("application.status_manager.SHOW_STATUS_UPDATES", False),
        ):
            await _handle_message_streaming(
                update=_make_update(),
                context=_make_context(mock_chat_service),
                chat_service=mock_chat_service,
                persistent_provider=MagicMock(),
                user_id=42,
                chat_id=123,
                username="testuser",
                text="Hello",
                reply_to_text=None,
                envelope=_make_envelope(),
            )

            # Verify: process_user_message was NOT called (hard discard)
            mock_chat_service.process_user_message.assert_not_called()
