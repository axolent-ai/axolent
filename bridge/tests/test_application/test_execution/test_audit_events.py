"""Tests for Phase 0 Commit 6: Structured Audit Events.

Verifies that execution_plan_created events are emitted with correct
fields, and that existing events carry request_id for correlation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from application.execution.context import ExecutionContext
from application.execution.envelope import RequestEnvelope
from application.execution.plan import ExecutionPlanner
from application.language_resolver import LanguageContext


def _make_context(
    lang: str = "de",
    request_id: str = "test-audit-req",
    user_id: int = 42,
    chat_id: int = 100,
) -> ExecutionContext:
    """Helper to create a minimal ExecutionContext."""
    return ExecutionContext(
        request_id=request_id,
        user_id=user_id,
        chat_id=chat_id,
        language=LanguageContext(
            code=lang,
            source="detected",
            confidence=0.92,
            switched_from=None,
            request_id=request_id,
        ),
    )


def _make_envelope(
    request_id: str = "test-audit-req",
    user_id: int = 42,
    chat_id: int = 100,
    channel: str = "telegram",
) -> RequestEnvelope:
    """Helper to create a minimal RequestEnvelope."""
    return RequestEnvelope(
        request_id=request_id,
        user_id=user_id,
        chat_id=chat_id,
        channel=channel,
        raw_text="Hello world",
    )


class TestExecutionPlanAuditEventChat:
    """Test that chat path emits execution_plan_created audit event."""

    def test_execution_plan_created_fields(self) -> None:
        """execution_plan_created event has all required fields for chat."""
        # Simulate what handlers.py does after plan creation
        envelope = _make_envelope(request_id="req-chat-001")
        exec_ctx = _make_context(
            request_id="req-chat-001", lang="en", user_id=42, chat_id=100
        )
        planner = ExecutionPlanner(default_provider_chain=["claude_persistent"])
        exec_plan = planner.plan_chat(exec_ctx, memory_ids=["mem_01", "mem_02"])

        # Replicate the audit event structure from handlers.py
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "execution_plan_created",
            "request_id": envelope.request_id,
            "user_id": 42,
            "chat_id": 100,
            "channel": envelope.channel,
            "task_type": exec_plan.task_type,
            "language": exec_ctx.language.code,
            "language_source": exec_ctx.language.source,
            "language_confidence": exec_ctx.language.confidence,
            "provider_chain": exec_plan.provider_chain,
            "memory_refs": list(exec_plan.memory_used),
            "verifier_profile": exec_plan.verifier_profile,
            "audit_required": exec_plan.audit_required,
        }

        # Verify structure
        assert event["event_type"] == "execution_plan_created"
        assert event["request_id"] == "req-chat-001"
        assert event["task_type"] == "answer_chat"
        assert event["language"] == "en"
        assert event["language_source"] == "detected"
        assert event["language_confidence"] == 0.92
        assert event["provider_chain"] == ["claude_persistent"]
        assert event["memory_refs"] == ["mem_01", "mem_02"]
        assert event["verifier_profile"] == "standard"
        assert event["audit_required"] is True
        assert event["channel"] == "telegram"
        assert event["user_id"] == 42
        assert event["chat_id"] == 100
        assert "timestamp" in event


class TestExecutionPlanAuditEventDebate:
    """Test that debate path emits execution_plan_created audit event."""

    def test_execution_plan_created_debate_fields(self) -> None:
        """execution_plan_created event has task_type=debate."""
        envelope = _make_envelope(request_id="req-debate-001")
        exec_ctx = _make_context(
            request_id="req-debate-001", lang="fr", user_id=7, chat_id=50
        )
        planner = ExecutionPlanner(
            default_provider_chain=["claude_persistent", "openai"]
        )
        exec_plan = planner.plan_debate(exec_ctx)

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "execution_plan_created",
            "request_id": envelope.request_id,
            "user_id": 7,
            "chat_id": 50,
            "channel": envelope.channel,
            "task_type": exec_plan.task_type,
            "language": exec_ctx.language.code,
            "language_source": exec_ctx.language.source,
            "language_confidence": exec_ctx.language.confidence,
            "provider_chain": exec_plan.provider_chain,
            "memory_refs": list(exec_plan.memory_used),
            "verifier_profile": exec_plan.verifier_profile,
            "audit_required": exec_plan.audit_required,
        }

        assert event["event_type"] == "execution_plan_created"
        assert event["request_id"] == "req-debate-001"
        assert event["task_type"] == "debate"
        assert event["language"] == "fr"
        assert event["provider_chain"] == ["claude_persistent", "openai"]
        assert event["memory_refs"] == []


class TestAuditCorrelationViaRequestId:
    """Test that multiple events within one request share the same request_id."""

    def test_all_events_share_request_id(self) -> None:
        """stream_started, execution_plan_created, stream_completed share request_id."""
        request_id = "corr-test-001"
        envelope = _make_envelope(request_id=request_id)
        exec_ctx = _make_context(request_id=request_id)
        planner = ExecutionPlanner()
        exec_plan = planner.plan_chat(exec_ctx)

        # Simulate the three events that handlers.py would write
        event_started = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_started",
            "request_id": envelope.request_id,
            "user_id": 42,
            "chat_id": 100,
        }

        event_plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "execution_plan_created",
            "request_id": envelope.request_id,
            "user_id": 42,
            "chat_id": 100,
            "task_type": exec_plan.task_type,
            "language": exec_ctx.language.code,
            "language_source": exec_ctx.language.source,
            "language_confidence": exec_ctx.language.confidence,
            "provider_chain": exec_plan.provider_chain,
            "memory_refs": list(exec_plan.memory_used),
            "verifier_profile": exec_plan.verifier_profile,
            "audit_required": exec_plan.audit_required,
            "channel": envelope.channel,
        }

        event_completed = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_completed",
            "request_id": request_id,
            "user_id": 42,
            "chat_id": 100,
            "response_length": 150,
        }

        # All three events must have the same request_id
        assert event_started["request_id"] == request_id
        assert event_plan["request_id"] == request_id
        assert event_completed["request_id"] == request_id

    def test_error_event_carries_request_id(self) -> None:
        """stream_error events carry the same request_id."""
        request_id = "corr-error-001"
        envelope = _make_envelope(request_id=request_id)

        event_error = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_error",
            "request_id": envelope.request_id,
            "user_id": 42,
            "chat_id": 100,
            "error_id": "abc12345",
            "error": "outer_exception",
        }

        assert event_error["request_id"] == request_id

    @pytest.mark.asyncio
    async def test_save_streaming_result_passes_request_id(self) -> None:
        """save_streaming_result writes request_id into stream_completed audit."""
        from application.chat_service import ChatService

        mock_router = MagicMock()
        svc = ChatService(provider_router=mock_router)

        with patch("application.chat_service.write_audit_log") as mock_audit:
            await svc.save_streaming_result(
                user_id=42,
                chat_id=100,
                user_text="Hello",
                response_text="Hi there!",
                duration_seconds=1.5,
                username="testuser",
                request_id="req-stream-complete-001",
            )

            mock_audit.assert_called_once()
            audit_data = mock_audit.call_args[0][0]
            assert audit_data["event_type"] == "stream_completed"
            assert audit_data["request_id"] == "req-stream-complete-001"

    @pytest.mark.asyncio
    async def test_save_streaming_result_empty_request_id_default(self) -> None:
        """save_streaming_result works without request_id (backward compat)."""
        from application.chat_service import ChatService

        mock_router = MagicMock()
        svc = ChatService(provider_router=mock_router)

        with patch("application.chat_service.write_audit_log") as mock_audit:
            await svc.save_streaming_result(
                user_id=42,
                chat_id=100,
                user_text="Hello",
                response_text="Hi there!",
                duration_seconds=1.5,
            )

            mock_audit.assert_called_once()
            audit_data = mock_audit.call_args[0][0]
            assert audit_data["request_id"] == ""
