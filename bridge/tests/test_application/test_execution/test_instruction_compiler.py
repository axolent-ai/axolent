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

        assert "IMPORTANT: Respond only in the language" in result.system_prompt
        assert "'fr'" in result.system_prompt

    def test_language_lock_for_german(self) -> None:
        """Language lock is present even for German (default)."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="de")
        plan = _make_plan(lang="de")

        result = compiler.compile_chat(ctx, plan, base_prompt="Basis.")

        assert "IMPORTANT: Respond only in the language" in result.system_prompt
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
        lang_pos = result.system_prompt.find("IMPORTANT: Respond only in the language")
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

        lang_pos = result.system_prompt.find("IMPORTANT: Respond only in the language")
        mem_pos = result.system_prompt.find("[STORED NOTES]")
        assert lang_pos < mem_pos

    def test_anti_repetition_present(self) -> None:
        """Anti-repetition style rule is included."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(ctx, plan, base_prompt="Base.")
        assert "anti_repetition" in result.get_metadata("blocks_included")

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

        blocks = result.get_metadata("blocks_included")
        assert "language_lock" in blocks
        assert "task_objective" in blocks
        assert "memory" in blocks
        assert "anti_repetition" in blocks

    def test_metadata_request_id(self) -> None:
        """Metadata includes request_id for audit correlation."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="en", request_id="audit-123")
        plan = _make_plan(lang="en")

        result = compiler.compile_chat(ctx, plan, base_prompt="x")
        assert result.get_metadata("request_id") == "audit-123"


class TestInstructionCompilerDebate:
    """Test debate compilation."""

    def test_debate_provider_prompt(self) -> None:
        """Debate provider gets concise instruction with language lock."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="it")
        plan = _make_plan(lang="it", task_type="debate")

        result = compiler.compile_debate(ctx, plan, role="provider")

        assert "IMPORTANT: Respond only in the language" in result.system_prompt
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
        assert "IMPORTANT: Respond only in the language" in result.system_prompt

    def test_debate_metadata(self) -> None:
        """Debate metadata includes role info."""
        compiler = InstructionCompiler()
        ctx = _make_context(lang="de")
        plan = _make_plan(lang="de", task_type="debate")

        result = compiler.compile_debate(ctx, plan, role="provider")
        assert result.get_metadata("debate_role") == "provider"
        assert result.get_metadata("task_type") == "debate"


class TestCompiledPrompt:
    """Test CompiledPrompt dataclass."""

    def test_default_empty(self) -> None:
        """Default CompiledPrompt has empty strings and empty metadata tuple."""
        cp = CompiledPrompt()
        assert cp.system_prompt == ""
        assert cp.user_prompt == ""
        assert cp.metadata == ()

    def test_frozen(self) -> None:
        """CompiledPrompt is immutable."""
        cp = CompiledPrompt(system_prompt="test")
        with pytest.raises(Exception):
            cp.system_prompt = "changed"  # type: ignore[misc]


class TestCompiledPromptMetadataImmutability:
    """EK-06: CompiledPrompt.metadata must be immutable tuple-of-tuples."""

    def test_metadata_is_tuple(self) -> None:
        """metadata field is a tuple, not a mutable dict."""
        cp = CompiledPrompt(
            system_prompt="s",
            metadata=(("key", "val"), ("n", 42)),
        )
        assert isinstance(cp.metadata, tuple)

    def test_metadata_immutable_no_assignment(self) -> None:
        """Cannot reassign metadata on a frozen CompiledPrompt."""
        cp = CompiledPrompt(
            system_prompt="s",
            metadata=(("a", 1),),
        )
        with pytest.raises(Exception):
            cp.metadata = (("b", 2),)  # type: ignore[misc]

    def test_metadata_item_assignment_fails(self) -> None:
        """Tuple does not support item assignment."""
        cp = CompiledPrompt(
            system_prompt="s",
            metadata=(("x", "y"),),
        )
        with pytest.raises(TypeError):
            cp.metadata[0] = ("z", "w")  # type: ignore[index]

    def test_get_metadata_helper(self) -> None:
        """get_metadata helper looks up keys correctly."""
        cp = CompiledPrompt(
            metadata=(("request_id", "r1"), ("language", "de")),
        )
        assert cp.get_metadata("request_id") == "r1"
        assert cp.get_metadata("language") == "de"
        assert cp.get_metadata("missing") is None
        assert cp.get_metadata("missing", "fb") == "fb"

    def test_as_metadata_dict_helper(self) -> None:
        """as_metadata_dict converts to plain dict."""
        cp = CompiledPrompt(
            metadata=(("a", 1), ("b", "two")),
        )
        d = cp.as_metadata_dict()
        assert d == {"a": 1, "b": "two"}
        # Mutation of returned dict does not affect original
        d["c"] = "injected"
        assert cp.get_metadata("c") is None

    def test_blocks_included_immutable(self) -> None:
        """blocks_included inside metadata is a tuple, not a mutable list."""
        compiler = InstructionCompiler()
        ctx = ExecutionContext(
            request_id="bi-1",
            user_id=1,
            chat_id=2,
            language=LanguageContext(
                code="de",
                source="detected",
                confidence=0.9,
                switched_from=None,
                request_id="bi-1",
            ),
        )
        plan = ExecutionPlan(request_id="bi-1", language="de")

        result = compiler.compile_chat(ctx, plan, base_prompt="Base.")
        blocks = result.get_metadata("blocks_included")
        assert isinstance(blocks, tuple)
        # Cannot append to a tuple
        with pytest.raises(AttributeError):
            blocks.append("injected")  # type: ignore[attr-defined]
