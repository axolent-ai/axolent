"""Tests for InstructionCompiler: prompt compilation from context."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from application.execution.context import ExecutionContext
from application.execution.instruction_compiler import (
    CompiledPrompt,
    InstructionCompiler,
)
from application.execution.plan import ExecutionPlan
from application.language_resolver import LanguageContext


def _make_context(lang: str = "de", request_id: str = "test-req") -> ExecutionContext:
    """Helper to create a minimal ExecutionContext."""
    return ExecutionContext(
        request_id=request_id,
        user_id=42,
        chat_id=99,
        language=LanguageContext(
            code=lang,
            source="detected",
            confidence=0.95,
            switched_from=None,
            request_id=request_id,
        ),
    )


def _make_plan(lang: str = "de", task_type: str = "answer_chat") -> ExecutionPlan:
    """Helper to create a minimal ExecutionPlan."""
    return ExecutionPlan(
        request_id="test-req",
        task_type=task_type,
        language=lang,
        provider_chain=("claude_persistent",),
    )


class TestInstructionCompilerBlockOrder:
    """Test that blocks are compiled in correct priority order."""

    def test_language_lock_always_present(self) -> None:
        """Language lock is always in the compiled prompt."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="fr")
        plan = _make_plan(lang="fr")

        result = compiler.compile_chat(ctx, plan, base_prompt="Base prompt.")

        assert "[LANGUAGE LOCK]" in result.system_prompt
        assert "'fr'" in result.system_prompt

    def test_language_lock_for_german(self) -> None:
        """Language lock is present even for German (default)."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="de")
        plan = _make_plan(lang="de")

        result = compiler.compile_chat(ctx, plan, base_prompt="Basis.")

        assert "[LANGUAGE LOCK]" in result.system_prompt
        assert "'de'" in result.system_prompt

    def test_block_order_security_before_language(self) -> None:
        """Security block (future) would come before language lock.

        Phase 0: verified by checking language lock position.
        The base prompt (task) comes first, language lock appended.
        """
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(ctx, plan, base_prompt="TASK_START")
        # Task is at the beginning
        assert result.system_prompt.startswith("TASK_START")
        # Language lock follows
        lang_pos = result.system_prompt.find("[LANGUAGE LOCK]")
        assert lang_pos > 0

    def test_memory_block_included(self) -> None:
        """Memory block is included when provided."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="de")
        plan = _make_plan(lang="de")

        result = compiler.compile_chat(
            ctx,
            plan,
            base_prompt="Base.",
            memory_block="[STORED NOTES]\nUser likes cats.",
        )

        assert "[STORED NOTES]" in result.system_prompt
        assert "User likes cats." in result.system_prompt

    def test_memory_after_language_lock(self) -> None:
        """Memory block comes after language lock (correct order)."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(
            ctx,
            plan,
            base_prompt="Base.",
            memory_block="[STORED NOTES]\nSome memory.",
        )

        lang_pos = result.system_prompt.find("[LANGUAGE LOCK]")
        mem_pos = result.system_prompt.find("[STORED NOTES]")
        assert lang_pos < mem_pos

    def test_anti_repetition_present(self) -> None:
        """Anti-repetition style rule is included."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(ctx, plan, base_prompt="Base.")
        assert "[STYLE RULE]" in result.system_prompt

    def test_time_service_integrated(self) -> None:
        """Time service block is included when available (EK-04: uses ctx.time)."""
        mock_time_service = MagicMock()
        mock_time_service.get_time_context_block.return_value = "[TIME] Monday, morning"

        compiler = InstructionCompiler(proactive_trigger_service=mock_time_service)
        ctx = _make_context(lang="en")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(ctx, plan, base_prompt="Base.")
        assert "[TIME] Monday, morning" in result.system_prompt
        # EK-04: now= must be passed from ctx.time.now_local
        mock_time_service.get_time_context_block.assert_called_once_with(
            42, now=ctx.time.now_local, lang="en"
        )

    def test_style_service_integrated(self) -> None:
        """Style adaption block is included when available."""
        mock_style = MagicMock()
        mock_style.get_prompt_block.return_value = "[STYLE] Casual tone."

        compiler = InstructionCompiler(style_adaption_service=mock_style)
        ctx = _make_context(lang="de")
        plan = _make_plan(lang="de")

        result = compiler.compile_chat(ctx, plan, base_prompt="Base.")
        assert "[STYLE] Casual tone." in result.system_prompt

    def test_metadata_includes_blocks(self) -> None:
        """Metadata tracks which blocks were included."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(
            ctx,
            plan,
            base_prompt="Base.",
            memory_block="[STORED NOTES]\nTest.",
        )

        assert "language_lock" in result.metadata["blocks_included"]
        assert "task_objective" in result.metadata["blocks_included"]
        assert "memory" in result.metadata["blocks_included"]
        assert "anti_repetition" in result.metadata["blocks_included"]

    def test_metadata_request_id(self) -> None:
        """Metadata includes request_id for audit correlation."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en", request_id="audit-123")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(ctx, plan, base_prompt="x")
        assert result.metadata["request_id"] == "audit-123"


class TestInstructionCompilerDebate:
    """Test debate compilation."""

    def test_debate_provider_prompt(self) -> None:
        """Debate provider gets concise instruction with language lock."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="it")
        plan = _make_plan(lang="it", task_type="debate")

        result = compiler.compile_debate(ctx, plan, role="provider")

        assert "[LANGUAGE LOCK]" in result.system_prompt
        assert "'it'" in result.system_prompt
        assert "concisely" in result.system_prompt

    def test_debate_judge_prompt(self) -> None:
        """Debate judge gets evaluation instruction."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en")
        plan = _make_plan(lang="en", task_type="debate")

        result = compiler.compile_debate(ctx, plan, role="judge")

        assert "neutral arbiter" in result.system_prompt
        assert "JSON" in result.system_prompt
        assert "[LANGUAGE LOCK]" in result.system_prompt

    def test_debate_metadata(self) -> None:
        """Debate metadata includes role info."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="de")
        plan = _make_plan(lang="de", task_type="debate")

        result = compiler.compile_debate(ctx, plan, role="provider")
        assert result.metadata["debate_role"] == "provider"
        assert result.metadata["task_type"] == "debate"


class TestCompiledPrompt:
    """Test CompiledPrompt dataclass."""

    def test_default_empty(self) -> None:
        """Default CompiledPrompt has empty strings."""
        cp = CompiledPrompt()
        assert cp.system_prompt == ""
        assert cp.user_prompt == ""
        assert cp.metadata == {}

    def test_frozen(self) -> None:
        """CompiledPrompt is immutable."""
        cp = CompiledPrompt(system_prompt="test")
        with pytest.raises(Exception):
            cp.system_prompt = "changed"  # type: ignore[misc]
