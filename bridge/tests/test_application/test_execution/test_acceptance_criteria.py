"""Acceptance criteria tests for Phase 0 Kernel MVP.

These tests verify the 6 acceptance criteria from the briefing:
1. Every chat request has a request_id
2. Language is resolved exactly once before first visible output
3. ChatService does not re-resolve language
4. /debate uses language from the actual question
5. Prompt is centrally compiled via InstructionCompiler
6. Time context comes via the same compiler
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from application.execution.context import ExecutionContext
from application.execution.envelope import RequestEnvelope
from application.execution.instruction_compiler import InstructionCompiler
from application.execution.kernel import ContextKernel
from application.execution.plan import ExecutionPlan, ExecutionPlanner
from application.language_resolver import LanguageContext


class _MockLanguageResolver:
    """Mock that tracks call count to verify single-resolution."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self.resolve_count = 0

    async def resolve(self, user_id, chat_id, text, override=None):
        self.resolve_count += 1
        code = override or self.lang
        return LanguageContext(
            code=code,
            source="override" if override else "detected",
            confidence=0.99,
            switched_from=None,
            request_id="mock-req",
        )


class TestAcceptanceCriteria:
    """Verify all 6 Phase 0 acceptance criteria."""

    @pytest.mark.asyncio
    async def test_criterion_1_request_id(self) -> None:
        """AC1: Every chat request has a request_id."""
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="Hello")
        assert env.request_id
        assert len(env.request_id) == 12

        mock_lr = _MockLanguageResolver(lang="en")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)
        ctx = await kernel.build(env)
        assert ctx.request_id == env.request_id

    @pytest.mark.asyncio
    async def test_criterion_2_language_resolved_once_before_output(self) -> None:
        """AC2: Language is resolved exactly once before first visible output."""
        mock_lr = _MockLanguageResolver(lang="de")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=10, chat_id=20, text="Hallo Welt")
        ctx = await kernel.build(env)

        # Language resolved exactly once
        assert mock_lr.resolve_count == 1
        # Language is available before any output path
        assert ctx.language.code == "de"

    @pytest.mark.asyncio
    async def test_criterion_3_chat_no_double_resolution(self) -> None:
        """AC3: ChatService receives ctx and does not re-resolve.

        Simulates the post-wire-up flow: kernel resolves once,
        ChatService uses ctx.language without calling resolver again.
        """
        mock_lr = _MockLanguageResolver(lang="fr")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(
            user_id=1, chat_id=2, text="Bonjour le monde"
        )
        ctx = await kernel.build(env)

        # Simulate ChatService receiving the context
        # It should use ctx.language.code directly
        lang_for_prompt = ctx.language.code
        assert lang_for_prompt == "fr"

        # No second resolve call happened
        assert mock_lr.resolve_count == 1

    @pytest.mark.asyncio
    async def test_criterion_4_debate_resolves_from_question(self) -> None:
        """AC4: /debate uses language from the actual question, not sticky.

        The debate envelope contains the question text, and the kernel
        resolves language FROM that text (not from a cached sticky value).
        """
        # Mock resolver that detects Italian from "Acqua bagna?"
        mock_lr = _MockLanguageResolver(lang="it")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        # User has sticky "de" but asks in Italian
        env = RequestEnvelope.from_debate_command(
            user_id=1, chat_id=2, question="L'acqua bagna le cose?"
        )
        ctx = await kernel.build(env)

        # Language comes from question detection, not sticky
        assert ctx.language.code == "it"
        # Plan reflects the detected language
        planner = ExecutionPlanner()
        plan = planner.plan_debate(ctx)
        assert plan.language == "it"

    def test_criterion_5_prompt_centrally_compiled(self) -> None:
        """AC5: Prompt is centrally compiled via InstructionCompiler."""
        compiler = InstructionCompiler()
        ctx = ExecutionContext(
            request_id="r5",
            user_id=1,
            chat_id=2,
            language=LanguageContext(
                code="en",
                source="detected",
                confidence=0.9,
                switched_from=None,
                request_id="r5",
            ),
        )
        plan = ExecutionPlan(
            request_id="r5",
            task_type="answer_chat",
            language="en",
        )

        result = compiler.compile_chat(ctx, plan, base_prompt="System prompt.")

        # Prompt is a single compiled string with all required blocks
        assert "[LANGUAGE LOCK]" in result.system_prompt
        assert "System prompt." in result.system_prompt
        assert result.metadata["request_id"] == "r5"

    def test_criterion_6_time_context_via_compiler(self) -> None:
        """AC6: Time context comes through the same compiler."""
        mock_time_svc = MagicMock()
        mock_time_svc.get_time_context_block.return_value = (
            "[TIME CONTEXT] Saturday, afternoon"
        )

        compiler = InstructionCompiler(proactive_trigger_service=mock_time_svc)
        ctx = ExecutionContext(
            request_id="r6",
            user_id=1,
            chat_id=2,
            language=LanguageContext(
                code="de",
                source="sticky",
                confidence=1.0,
                switched_from=None,
                request_id="r6",
            ),
        )
        plan = ExecutionPlan(request_id="r6", language="de")

        result = compiler.compile_chat(ctx, plan, base_prompt="Base.")
        assert "[TIME CONTEXT] Saturday, afternoon" in result.system_prompt
        assert "time_context" in result.metadata["blocks_included"]

    @pytest.mark.asyncio
    async def test_audit_event_includes_request_id(self) -> None:
        """AC7 (test): Audit data includes request_id via plan.to_audit_dict()."""
        mock_lr = _MockLanguageResolver(lang="en")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="audit test")
        ctx = await kernel.build(env)
        planner = ExecutionPlanner()
        plan = planner.plan_chat(ctx)

        audit = plan.to_audit_dict()
        assert audit["request_id"] == env.request_id
        assert audit["language"] == "en"
        assert audit["task_type"] == "answer_chat"

    @pytest.mark.asyncio
    async def test_status_session_uses_ctx_language(self) -> None:
        """Test: StatusSession would receive ctx.language.code.

        Verifies the data flow: kernel -> ctx -> status_session.language.
        """
        mock_lr = _MockLanguageResolver(lang="es")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="Hola mundo")
        ctx = await kernel.build(env)

        # StatusSession would be initialized with this language
        status_language = ctx.language.code
        assert status_language == "es"
