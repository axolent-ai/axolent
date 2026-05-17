"""PromptComposer: legacy prompt construction facade.

.. deprecated:: Phase 0 Commit 5
    Use :class:`application.execution.instruction_compiler.InstructionCompiler`
    instead. InstructionCompiler accepts ``ExecutionContext + ExecutionPlan``
    and is the canonical prompt path. PromptComposer remains only for
    backward compatibility with the legacy ChatService/DebateOrchestrator
    code paths (when no ExecutionContext is available).

Blocks (in order of injection):
    1. Language Lock (always, including for "de")
    2. Time Context (if ProactiveTriggerService available)
    3. Style Adaption (if user has mature profile)
    4. Self-Awareness (model identity)
    5. Memory Context (relevant stored entries)
    6. Anti-Repetition Rule (style quality)

Design: Facade over existing functions. Does NOT rewrite the internals
of build_effective_prompt or the time service, but orchestrates them
into a single composable call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional

from application.language_resolver import LanguageContext
from domain.personality import build_effective_prompt
from i18n import t

if TYPE_CHECKING:
    from application.proactive_trigger_service import ProactiveTriggerService
    from application.self_awareness_service import SelfAwarenessService
    from application.style_adaption_service import StyleAdaptionService

log = logging.getLogger(__name__)

# Block types that can be requested
BlockType = Literal[
    "language",
    "time",
    "style",
    "self_awareness",
    "memory",
    "anti_repetition",
]

# Anti-repetition rule prefix (the translated text is appended after this)
_ANTI_REPETITION_PREFIX = "\n\n[STYLE RULE] "


class PromptComposer:
    """Legacy prompt construction facade.

    .. deprecated:: Phase 0 Commit 5
        Use ``InstructionCompiler`` from
        ``application.execution.instruction_compiler`` instead.

    Retained for backward compatibility with code paths that do
    not yet have an ExecutionContext available. New code should
    always go through InstructionCompiler.
    """

    def __init__(
        self,
        proactive_trigger_service: Optional["ProactiveTriggerService"] = None,
        style_adaption_service: Optional["StyleAdaptionService"] = None,
        self_awareness_service: Optional["SelfAwarenessService"] = None,
    ) -> None:
        """Initialize with optional services.

        Args:
            proactive_trigger_service: For time context blocks.
            style_adaption_service: For user style profile blocks.
            self_awareness_service: For model identity blocks.
        """
        self._time_service = proactive_trigger_service
        self._style_service = style_adaption_service
        self._awareness_service = self_awareness_service

    def compose(
        self,
        base_prompt: str,
        ctx: LanguageContext,
        purpose: Literal["chat", "debate_provider", "debate_judge", "status"],
        blocks: list[BlockType],
        *,
        user_id: int = 0,
        user_model: Optional[str] = None,
        task_slot_name: Optional[str] = None,
        memory_block: str = "",
    ) -> str:
        """Compose the full system prompt from base + requested blocks.

        Args:
            base_prompt: The combined base prompt (personality + constitution).
            ctx: Resolved LanguageContext (from LanguageResolver).
            purpose: What this prompt is for (affects block selection).
            blocks: Which blocks to include (order is fixed internally).
            user_id: Telegram user ID (for style/awareness lookups).
            user_model: Resolved model ID (for self-awareness).
            task_slot_name: Task slot name (for self-awareness).
            memory_block: Pre-formatted memory context string.

        Returns:
            Complete system prompt with all requested blocks.
        """
        lang = ctx.code

        # Step 1: Language lock (always via build_effective_prompt)
        if "language" in blocks:
            result = build_effective_prompt(base_prompt, lang)
        else:
            result = base_prompt

        # Step 2: Time context
        if "time" in blocks and self._time_service is not None:
            time_block = self._time_service.get_time_context_block(user_id, lang=lang)
            if time_block:
                result = f"{result}\n\n{time_block}"

        # Step 3: Style adaption
        if "style" in blocks and self._style_service is not None:
            style_block = self._style_service.get_prompt_block(user_id, lang)
            if style_block:
                result = f"{result}\n\n{style_block}"

        # Step 4: Self-awareness
        if "self_awareness" in blocks and self._awareness_service is not None:
            awareness_block = self._awareness_service.build(
                user_id=user_id,
                user_model=user_model,
                task_slot_name=task_slot_name,
                lang=lang,
            )
            if awareness_block:
                result = f"{result}\n\n{awareness_block}"

        # Step 5: Memory context
        if "memory" in blocks and memory_block:
            result = f"{result}\n\n{memory_block}"

        # Step 6: Anti-repetition rule (i18n via t())
        if "anti_repetition" in blocks:
            rule_text = t("style.no_repetition_rule", lang)
            result = f"{result}{_ANTI_REPETITION_PREFIX}{rule_text}"

        return result

    def compose_for_chat(
        self,
        base_prompt: str,
        ctx: LanguageContext,
        *,
        user_id: int = 0,
        user_model: Optional[str] = None,
        task_slot_name: Optional[str] = None,
        memory_block: str = "",
    ) -> str:
        """Convenience: compose for the main chat path.

        Includes all blocks: language, time, style, self_awareness,
        memory, anti_repetition.
        """
        return self.compose(
            base_prompt=base_prompt,
            ctx=ctx,
            purpose="chat",
            blocks=[
                "language",
                "time",
                "style",
                "self_awareness",
                "memory",
                "anti_repetition",
            ],
            user_id=user_id,
            user_model=user_model,
            task_slot_name=task_slot_name,
            memory_block=memory_block,
        )

    def compose_for_debate_provider(
        self,
        ctx: LanguageContext,
    ) -> str:
        """Convenience: compose for debate provider calls.

        Minimal prompt with language lock only.
        """
        base = (
            "Answer concisely and informatively. Keep it to 2-4 sentences if possible."
        )
        return self.compose(
            base_prompt=base,
            ctx=ctx,
            purpose="debate_provider",
            blocks=["language", "anti_repetition"],
        )

    def compose_for_debate_judge(
        self,
        ctx: LanguageContext,
    ) -> str:
        """Convenience: compose for the debate judge call.

        Judge-specific base prompt with language lock.
        """
        base = (
            "You are a neutral arbiter evaluating AI answers. "
            "You do not know the provider names and evaluate purely on quality: "
            "correctness, completeness, clarity, and relevance. "
            "ALWAYS respond with valid JSON, never with prose."
        )
        return self.compose(
            base_prompt=base,
            ctx=ctx,
            purpose="debate_judge",
            blocks=["language"],
        )
