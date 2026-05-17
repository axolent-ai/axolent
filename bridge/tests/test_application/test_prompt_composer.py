"""Tests for application.prompt_composer: PromptComposer.

Tests various block combinations for chat, debate, and minimal paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock


from application.language_resolver import LanguageContext, LanguageResolver
from application.prompt_composer import PromptComposer


def _make_ctx(lang: str = "de", source: str = "sticky") -> LanguageContext:
    """Helper: create a LanguageContext for testing."""
    return LanguageResolver.from_code(lang, source)


class TestPromptComposer:
    """Tests for PromptComposer.compose()."""

    def test_compose_language_block_included(self) -> None:
        """Language block should inject [LANGUAGE LOCK] into prompt."""
        composer = PromptComposer()
        ctx = _make_ctx("en")

        result = composer.compose(
            base_prompt="You are helpful.",
            ctx=ctx,
            purpose="chat",
            blocks=["language"],
        )

        assert "[LANGUAGE LOCK]" in result
        assert "'en'" in result
        assert "You are helpful." in result

    def test_compose_language_block_for_german(self) -> None:
        """German also gets a language lock (fix for T33)."""
        composer = PromptComposer()
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Du bist hilfreich.",
            ctx=ctx,
            purpose="chat",
            blocks=["language"],
        )

        assert "[LANGUAGE LOCK]" in result
        assert "'de'" in result

    def test_compose_anti_repetition_block(self) -> None:
        """Anti-repetition block should be appended."""
        composer = PromptComposer()
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["anti_repetition"],
        )

        assert "[STYLE RULE]" in result
        assert "Gerne" in result

    def test_compose_anti_repetition_english(self) -> None:
        """English anti-repetition rule uses English fillers."""
        composer = PromptComposer()
        ctx = _make_ctx("en")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["anti_repetition"],
        )

        assert "Sure" in result

    def test_compose_memory_block(self) -> None:
        """Memory block is appended when provided."""
        composer = PromptComposer()
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["memory"],
            memory_block="[STORED NOTES]\nUser likes dolphins.",
        )

        assert "[STORED NOTES]" in result
        assert "dolphins" in result

    def test_compose_no_memory_block_when_empty(self) -> None:
        """Empty memory block is not appended."""
        composer = PromptComposer()
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["memory"],
            memory_block="",
        )

        assert result == "Base."

    def test_compose_time_block_with_service(self) -> None:
        """Time block is included when proactive_trigger_service provides one."""
        mock_time_svc = MagicMock()
        mock_time_svc.get_time_context_block.return_value = (
            "[TIME CONTEXT]\nCurrent: 14:30"
        )

        composer = PromptComposer(proactive_trigger_service=mock_time_svc)
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["time"],
            user_id=42,
        )

        assert "[TIME CONTEXT]" in result
        mock_time_svc.get_time_context_block.assert_called_once_with(42, lang="de")

    def test_compose_time_block_without_service(self) -> None:
        """No time block when service is None."""
        composer = PromptComposer(proactive_trigger_service=None)
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["time"],
        )

        assert result == "Base."

    def test_compose_style_block(self) -> None:
        """Style block is included when service provides one."""
        mock_style_svc = MagicMock()
        mock_style_svc.get_prompt_block.return_value = (
            "[USER STYLE PROFILE]\nTerse user."
        )

        composer = PromptComposer(style_adaption_service=mock_style_svc)
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["style"],
            user_id=42,
        )

        assert "[USER STYLE PROFILE]" in result
        mock_style_svc.get_prompt_block.assert_called_once_with(42, "de")

    def test_compose_self_awareness_block(self) -> None:
        """Self-awareness block is included when service provides one."""
        mock_awareness = MagicMock()
        mock_awareness.build.return_value = "[SELF-AWARENESS]\nModel: Opus 4.7"

        composer = PromptComposer(self_awareness_service=mock_awareness)
        ctx = _make_ctx("de")

        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["self_awareness"],
            user_id=42,
            user_model="claude-opus-4-7",
            task_slot_name="chat",
        )

        assert "[SELF-AWARENESS]" in result
        mock_awareness.build.assert_called_once_with(
            user_id=42,
            user_model="claude-opus-4-7",
            task_slot_name="chat",
            lang="de",
        )

    def test_compose_for_chat_all_blocks(self) -> None:
        """compose_for_chat includes all standard blocks."""
        mock_time = MagicMock()
        mock_time.get_time_context_block.return_value = "[TIME]"
        mock_style = MagicMock()
        mock_style.get_prompt_block.return_value = "[STYLE]"
        mock_awareness = MagicMock()
        mock_awareness.build.return_value = "[AWARENESS]"

        composer = PromptComposer(
            proactive_trigger_service=mock_time,
            style_adaption_service=mock_style,
            self_awareness_service=mock_awareness,
        )
        ctx = _make_ctx("en")

        result = composer.compose_for_chat(
            base_prompt="System prompt.",
            ctx=ctx,
            user_id=1,
            memory_block="[MEMORY]",
        )

        assert "[LANGUAGE LOCK]" in result
        assert "[TIME]" in result
        assert "[STYLE]" in result
        assert "[AWARENESS]" in result
        assert "[MEMORY]" in result
        assert "[STYLE RULE]" in result  # anti_repetition

    def test_compose_for_debate_provider(self) -> None:
        """compose_for_debate_provider uses minimal blocks."""
        composer = PromptComposer()
        ctx = _make_ctx("it")

        result = composer.compose_for_debate_provider(ctx)

        assert "[LANGUAGE LOCK]" in result
        assert "'it'" in result
        assert "concisely" in result

    def test_compose_for_debate_judge(self) -> None:
        """compose_for_debate_judge uses judge-specific base."""
        composer = PromptComposer()
        ctx = _make_ctx("fr")

        result = composer.compose_for_debate_judge(ctx)

        assert "[LANGUAGE LOCK]" in result
        assert "'fr'" in result
        assert "neutral arbiter" in result

    def test_block_order_is_fixed(self) -> None:
        """Blocks are always in fixed order regardless of list order."""
        mock_time = MagicMock()
        mock_time.get_time_context_block.return_value = "TIME_MARKER"
        mock_style = MagicMock()
        mock_style.get_prompt_block.return_value = "STYLE_MARKER"

        composer = PromptComposer(
            proactive_trigger_service=mock_time,
            style_adaption_service=mock_style,
        )
        ctx = _make_ctx("de")

        # Pass blocks in reverse order
        result = composer.compose(
            base_prompt="Base.",
            ctx=ctx,
            purpose="chat",
            blocks=["style", "time", "language"],
            user_id=1,
        )

        # Time should come before style (internal fixed order)
        time_pos = result.index("TIME_MARKER")
        style_pos = result.index("STYLE_MARKER")
        assert time_pos < style_pos
