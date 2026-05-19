"""InstructionCompiler: builds the final model instructions from context and plan.

Replaces the scattered prompt-building logic with a single, ordered,
auditable compilation step. The compiler does not decide anything
itself; it translates the ExecutionContext + ExecutionPlan into
a prompt structure.

Block priority (fixed order, never violated):
    1. Security / Non-disclosure
    2. Privacy / Tool restrictions
    3. User language lock
    4. Task objective (base prompt)
    5. Time / location / channel context
    6. Memory with provenance
    7. Style / personality
    8. Output format contract

Phase 0: blocks 3-7 implemented. Blocks 1, 2, 8 are stubs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from application.execution.context import ExecutionContext
from application.execution.plan import ExecutionPlan
from domain.personality import build_effective_prompt
from i18n import t

if TYPE_CHECKING:
    from application.proactive_trigger_service import ProactiveTriggerService
    from application.self_awareness_service import SelfAwarenessService
    from application.style_adaption_service import StyleAdaptionService

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    """Result of prompt compilation.

    Contains the fully assembled system and user prompts
    plus metadata for audit.

    Attributes:
        system_prompt: Complete system prompt with all blocks.
        user_prompt: The user-facing prompt (context + message).
        metadata: Compilation metadata (immutable tuple-of-tuples).
    """

    system_prompt: str = ""
    user_prompt: str = ""
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Look up a metadata value by key.

        Args:
            key: The metadata key to look up.
            default: Value to return if key is not found.

        Returns:
            The metadata value, or default.
        """
        for k, v in self.metadata:
            if k == key:
                return v
        return default

    def as_metadata_dict(self) -> dict[str, Any]:
        """Convert metadata to a plain dict for serialization.

        Returns:
            Dict representation of metadata.
        """
        return dict(self.metadata)


class InstructionCompiler:
    """Compiles ExecutionContext + Plan into model instructions.

    Uses the existing build_effective_prompt() for language lock
    and integrates with existing services (time, style, awareness).
    The key difference from PromptComposer: strict block ordering
    and context-driven (no loose parameters).
    """

    def __init__(
        self,
        proactive_trigger_service: Optional["ProactiveTriggerService"] = None,
        style_adaption_service: Optional["StyleAdaptionService"] = None,
        self_awareness_service: Optional["SelfAwarenessService"] = None,
    ) -> None:
        """Initialize with optional services for enrichment blocks.

        Args:
            proactive_trigger_service: For time context blocks.
            style_adaption_service: For user style adaptation.
            self_awareness_service: For model identity blocks.
        """
        self._time_service = proactive_trigger_service
        self._style_service = style_adaption_service
        self._awareness_service = self_awareness_service

    def compile_chat(
        self,
        ctx: ExecutionContext,
        plan: ExecutionPlan,
        base_prompt: str,
        memory_block: str = "",
        user_prompt: str = "",
        *,
        user_model: Optional[str] = None,
        task_slot_name: Optional[str] = None,
    ) -> CompiledPrompt:
        """Compile a complete prompt for the chat path.

        Assembles blocks in strict priority order:
            1. Security (Phase 0: empty)
            2. Privacy (Phase 0: empty)
            3. Language Lock (always, from ctx.language)
            4. Task (base_prompt)
            5. Time context
            6. Memory block
            7. Style + Self-awareness + Anti-repetition
            8. Format contract (Phase 0: empty)

        Args:
            ctx: Resolved execution context.
            plan: Execution plan for this request.
            base_prompt: Base system prompt (personality + constitution).
            memory_block: Pre-formatted memory context string.
            user_prompt: The user message / context block.
            user_model: Resolved model ID (for self-awareness).
            task_slot_name: Task slot name (for self-awareness).

        Returns:
            CompiledPrompt with system_prompt and user_prompt.
        """
        lang = ctx.language.code
        blocks_included: list[str] = []

        # Block 1: Security (Phase 0: no-op)
        # TODO Phase 1: inject security/non-disclosure rules

        # Block 2: Privacy (Phase 0: no-op)
        # TODO Phase 1: inject privacy restrictions

        # Block 3 + 4: Language Lock + Task (via build_effective_prompt)
        result = build_effective_prompt(base_prompt, lang)
        blocks_included.append("language_lock")
        blocks_included.append("task_objective")

        # Block 5: Time context (EK-04: use ctx.time.now_local to ensure
        # audit time and prompt time are from the same snapshot)
        if self._time_service is not None:
            time_block = self._time_service.get_time_context_block(
                ctx.user_id, now=ctx.time.now_local, lang=lang
            )
            if time_block:
                result = f"{result}\n\n{time_block}"
                blocks_included.append("time_context")

        # Block 6: Memory
        if memory_block:
            result = f"{result}\n\n{memory_block}"
            blocks_included.append("memory")

        # Block 7a: Style adaption
        if self._style_service is not None:
            style_block = self._style_service.get_prompt_block(ctx.user_id, lang)
            if style_block:
                result = f"{result}\n\n{style_block}"
                blocks_included.append("style")

        # Block 7b: Self-awareness
        if self._awareness_service is not None:
            awareness_block = self._awareness_service.build(
                user_id=ctx.user_id,
                user_model=user_model,
                task_slot_name=task_slot_name,
                lang=lang,
            )
            if awareness_block:
                result = f"{result}\n\n{awareness_block}"
                blocks_included.append("self_awareness")

        # Block 7c: Anti-repetition
        rule_text = t("style.no_repetition_rule", lang)
        if rule_text:
            result = f"{result}\n\n{rule_text}"
            blocks_included.append("anti_repetition")

        # Block 8: Format contract (Phase 0: no-op)
        # TODO Phase 2: output format enforcement

        metadata = (
            ("request_id", ctx.request_id),
            ("language", lang),
            ("task_type", plan.task_type),
            ("blocks_included", tuple(blocks_included)),
        )

        return CompiledPrompt(
            system_prompt=result,
            user_prompt=user_prompt,
            metadata=metadata,
        )

    def compile_debate(
        self,
        ctx: ExecutionContext,
        plan: ExecutionPlan,
        role: str = "provider",
    ) -> CompiledPrompt:
        """Compile a prompt for debate provider or judge calls.

        Args:
            ctx: Resolved execution context.
            plan: Execution plan for this debate.
            role: "provider" or "judge".

        Returns:
            CompiledPrompt with debate-specific system_prompt.
        """
        lang = ctx.language.code

        if role == "judge":
            base = (
                "You are a neutral arbiter evaluating AI answers. "
                "You do not know the provider names and evaluate purely on quality: "
                "correctness, completeness, clarity, and relevance. "
                "ALWAYS respond with valid JSON, never with prose."
            )
        else:
            base = (
                "Answer concisely and informatively. "
                "Keep it to 2-4 sentences if possible."
            )

        result = build_effective_prompt(base, lang)

        # Anti-repetition for provider (not judge)
        if role == "provider":
            rule_text = t("style.no_repetition_rule", lang)
            if rule_text:
                result = f"{result}\n\n{rule_text}"

        metadata = (
            ("request_id", ctx.request_id),
            ("language", lang),
            ("task_type", "debate"),
            ("debate_role", role),
            ("blocks_included", ("language_lock", "task_objective")),
        )

        return CompiledPrompt(
            system_prompt=result,
            user_prompt="",
            metadata=metadata,
        )
