"""Chat service: use case for LLM calls.

Coordinates: conversation history -> language detection -> prompt building -> provider router -> audit log.
No Telegram code here, only business orchestration.

Since Phase 1: uses ProviderRouter instead of direct claude_cli.
Default provider: Claude (Mode B, CLI subprocess).

Since Phase 1 (auto-memory): automatically loads relevant memory entries
and injects them into the system prompt before the LLM call.

Since R04: streaming-capable via ClaudePersistentProvider.
process_user_message_streaming() yields an AsyncIterator of StreamEvents
for real-time Telegram edits.

Since Phase 0 Commit 3: accepts ExecutionContext + ExecutionPlan as
first-class parameters. When provided, language is NOT re-resolved.
InstructionCompiler is the single path for prompt assembly.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from application.execution.context import ExecutionContext
from application.execution.instruction_compiler import InstructionCompiler
from application.memory_conflict_detector import (
    MemoryConflictDetector,
    is_conflict_relevant_to_intent,
)
from application.execution.plan import ExecutionPlan
from application.leakage_filter import check_for_system_prompt_leakage
from application.prompt_composer import PromptComposer
from application.security.prompt_delimiters import escape_prompt_delimited_text
from domain.conversation import ConversationTurn, build_context_block
from domain.language import DEFAULT_LANGUAGE
from application.audit_service import filter_task_meta
from infrastructure.audit_log import write_audit_log
from infrastructure.claude_process_pool import StreamEvent
from infrastructure.conversation_storage import (
    get_history,
    get_language,
    save_turn,
    set_language,
)
from infrastructure.conversation_storage import (
    reset_conversation as _infra_reset_conversation,
)
from infrastructure.providers.base import ProviderError
from typeguard import typechecked

if TYPE_CHECKING:
    from application.fallback_resolver import FallbackResolver
    from application.language.enforcement import LanguageEnforcement
    from application.memory_service import MemoryService
    from application.model_service import ModelService
    from application.proactive_trigger_service import ProactiveTriggerService
    from application.provider_router import ProviderRouter
    from application.self_awareness_service import SelfAwarenessService
    from application.skill_compression.skill_matcher import SkillMatch, SkillMatcher
    from application.style_adaption_service import StyleAdaptionService
    from application.task_router import TaskRouter
    from infrastructure.providers.claude_persistent import ClaudePersistentProvider

log = logging.getLogger(__name__)

# Memory token budget: prevents prompt explosion with long /remember entries
MAX_MEMORY_CHARS_PER_ENTRY = 400
MAX_MEMORY_TOTAL_CHARS = 4000  # Default fallback if provider reports no capability


def _truncate(text: str, n: int) -> str:
    """Truncate text to at most n characters with ellipsis."""
    return text if len(text) <= n else text[: n - 3] + "..."


_STOP_WORDS_DE = frozenset(
    {
        "diese",
        "diesen",
        "dieser",
        "diesem",
        "dieses",
        "ihre",
        "ihren",
        "ihrer",
        "ihrem",
        "ihres",
        "meine",
        "meinen",
        "meiner",
        "meinem",
        "meines",
        "deine",
        "deinen",
        "seiner",
        "seinem",
        "sehr",
        "schon",
        "noch",
        "auch",
        "etwas",
        "sollte",
        "müsste",
        "könnte",
        "wollte",
        "welche",
        "welchen",
        "welcher",
        "welchem",
        "manche",
        "alles",
        "viele",
        "wenig",
        "haben",
        "machen",
        "geben",
        "nehmen",
    }
)
_STOP_WORDS_EN = frozenset(
    {
        "this",
        "that",
        "these",
        "those",
        "their",
        "there",
        "where",
        "which",
        "would",
        "could",
        "should",
        "very",
        "much",
        "more",
        "most",
        "want",
        "need",
        "have",
        "been",
        "from",
        "some",
        "many",
        "few",
        "what",
        "when",
        "while",
    }
)
_STOP_WORDS = _STOP_WORDS_DE | _STOP_WORDS_EN


def _extract_keywords(text: str) -> list[str]:
    """Extract search terms from a user message.

    Strategy: strip punctuation, filter stop words,
    then all words with > 3 chars, lowercase, deduplicated.

    Args:
        text: User message.

    Returns:
        List of keywords (longest first).
    """
    stripped = [w.strip("?,.:;!'\"-—–…()[]{}") for w in text.split()]
    candidates = {w.lower() for w in stripped if len(w) > 3}
    keywords = list(candidates - _STOP_WORDS)
    keywords.sort(key=len, reverse=True)
    return keywords


class ChatResult:
    """Result of a chat call.

    Attributes:
        success: True if the provider delivered a valid response.
        response: Provider response (empty on error).
        error_message: User-friendly error message (empty on success).
        error_id: Error ID for debugging (empty on success).
        duration: Duration of the provider call in seconds.
        detected_language: Detected language of the user message.
        provider_name: Name of the provider used.
    """

    __slots__ = (
        "success",
        "response",
        "error_message",
        "error_id",
        "duration",
        "detected_language",
        "provider_name",
    )

    def __init__(
        self,
        success: bool = False,
        response: str = "",
        error_message: str = "",
        error_id: str = "",
        duration: float = 0.0,
        detected_language: str = "de",
        provider_name: str = "claude",
    ) -> None:
        self.success = success
        self.response = response
        self.error_message = error_message
        self.error_id = error_id
        self.duration = duration
        self.detected_language = detected_language
        self.provider_name = provider_name


class ChatService:
    """Main logic for user requests.

    Coordinates provider call, memory loading, conversation history,
    and language detection. Replaces the old module globals + setters
    with clean constructor injection.
    """

    @typechecked
    def __init__(
        self,
        provider_router: "ProviderRouter",
        memory_service: "MemoryService | None" = None,
        model_service: "ModelService | None" = None,
        task_router: "TaskRouter | None" = None,
        self_awareness_service: "SelfAwarenessService | None" = None,
        style_adaption_service: "StyleAdaptionService | None" = None,
        proactive_trigger_service: "ProactiveTriggerService | None" = None,
        fallback_resolver: "FallbackResolver | None" = None,
        language_enforcement: "LanguageEnforcement | None" = None,
        skill_matcher: "SkillMatcher | None" = None,
    ) -> None:
        self.provider_router = provider_router
        self.memory_service = memory_service
        self.model_service = model_service
        self.task_router = task_router
        self.skill_matcher: "SkillMatcher | None" = skill_matcher
        self.self_awareness_service = self_awareness_service
        self.style_adaption_service = style_adaption_service
        self.proactive_trigger_service = proactive_trigger_service
        self.fallback_resolver = fallback_resolver
        self._language_enforcement = language_enforcement

        # LCP v1: StreamGuard stats store for self-calibration (Issue 1).
        # Process-wide, keyed by (user_id, chat_id).
        self._stream_guard_stats_store: Any = None
        if language_enforcement is not None:
            from application.language.stream_guard import StreamGuardStatsStore

            self._stream_guard_stats_store = StreamGuardStatsStore()

        # Central prompt composer (Phase 2): single source for system prompt construction
        self._composer = PromptComposer(
            proactive_trigger_service=proactive_trigger_service,
            style_adaption_service=style_adaption_service,
            self_awareness_service=self_awareness_service,
        )

        # Phase 0 Commit 3: InstructionCompiler (the canonical prompt path)
        self._instruction_compiler = InstructionCompiler(
            proactive_trigger_service=proactive_trigger_service,
            style_adaption_service=style_adaption_service,
            self_awareness_service=self_awareness_service,
        )

    def _get_memory_budget(self, provider_name: str | None = None) -> int:
        """Read the memory budget from ProviderCapabilities.

        If the provider defines its own max_memory_chars capability,
        that is used. Otherwise falls back to MAX_MEMORY_TOTAL_CHARS.

        Args:
            provider_name: Name of the provider (None = default from router).

        Returns:
            Max chars for memory block in system prompt.
        """
        if self.provider_router is None:
            return MAX_MEMORY_TOTAL_CHARS
        try:
            name = provider_name or self.provider_router.default
            providers_dict = self.provider_router.providers
            if not isinstance(providers_dict, dict):
                return MAX_MEMORY_TOTAL_CHARS
            provider = providers_dict.get(name)
            if provider:
                caps = provider.get_capabilities()
                value = getattr(caps, "max_memory_chars", MAX_MEMORY_TOTAL_CHARS)
                if isinstance(value, int):
                    return value
        except (AttributeError, TypeError):
            pass
        return MAX_MEMORY_TOTAL_CHARS

    def _build_memory_context(
        self,
        user_id: int,
        query: str,
        provider_name: str | None = None,
        skill_trigger: str | None = None,
    ) -> tuple[str, int]:
        """Load relevant memory entries for the user based on query.

        Search strategy: keyword extraction -> substring match across all layers.
        Prioritizes longest keywords as primary search.

        Args:
            user_id: Telegram user ID.
            query: Current user message.
            provider_name: Optional provider name for memory budget lookup.
            skill_trigger: If a skill matched, the trigger text. When set,
                memory conflicts irrelevant to the skill are suppressed from
                the prompt (Round 3 Skill > Memory priority rule).

        Returns:
            Tuple of (memory context string, number of loaded entries).
            Empty string + 0 if no hits or no MemoryService.
        """
        if self.memory_service is None:
            return "", 0

        keywords = _extract_keywords(query)
        if not keywords:
            # No keywords extracted (short message). Load recent entries as context.
            episodic = self.memory_service.list_recent(
                user_id, layer="episodic", limit=5
            )
            semantic = self.memory_service.list_recent(
                user_id, layer="semantic", limit=3
            )
            if not (episodic or semantic):
                return "", 0
            # Jump directly to context building
            return self._format_memory_context(
                episodic,
                semantic,
                [],
                provider_name,
                skill_trigger=skill_trigger,
                user_input=query,
            )

        # Primary search term: longest keyword (most specific)
        primary_query = keywords[0]

        episodic = self.memory_service.recall(
            user_id, primary_query, layer="episodic", limit=3
        )
        semantic = self.memory_service.recall(
            user_id, primary_query, layer="semantic", limit=3
        )
        procedural = self.memory_service.recall(
            user_id, primary_query, layer="procedural", limit=2
        )

        # If primary yields no hits: try secondary (if available)
        if not (episodic or semantic or procedural) and len(keywords) > 1:
            secondary_query = keywords[1]
            episodic = self.memory_service.recall(
                user_id, secondary_query, layer="episodic", limit=3
            )
            semantic = self.memory_service.recall(
                user_id, secondary_query, layer="semantic", limit=3
            )
            procedural = self.memory_service.recall(
                user_id, secondary_query, layer="procedural", limit=2
            )

        # NEU-02 fix: Do NOT fall back to random recent entries when keywords
        # were extracted but yielded no matches. This prevents "phantom knowledge"
        # where the bot appears to know things unrelated to the question.
        # Only use recent fallback when NO keywords could be extracted at all
        # (already handled above in the `if not keywords:` branch).
        if not (episodic or semantic or procedural):
            return "", 0

        return self._format_memory_context(
            episodic,
            semantic,
            procedural,
            provider_name,
            skill_trigger=skill_trigger,
            user_input=query,
        )

    def _format_memory_context(
        self,
        episodic: list[dict],
        semantic: list[dict],
        procedural: list[dict],
        provider_name: str | None = None,
        skill_trigger: str | None = None,
        user_input: str = "",
    ) -> tuple[str, int]:
        """Format memory entries into a context block for the system prompt.

        Args:
            episodic: Episodic memory entries.
            semantic: Semantic memory entries.
            procedural: Procedural memory entries.
            provider_name: Optional provider name for budget lookup.
            skill_trigger: If a skill matched, the trigger text (for conflict
                relevance filtering). None = no skill matched, show all conflicts.
            user_input: Original user input (for conflict relevance check).

        Returns:
            Tuple of (formatted memory block, total entry count).
        """
        total_entries = len(episodic) + len(semantic) + len(procedural)

        sections: list[str] = ["[STORED NOTES]", ""]
        sections.append(
            "These entries were stored by the user. Honor them this way:\n"
            "\n"
            "  - Reference only what is explicitly stored. Never invent "
            "reasons, motivations, or context that the user did not provide.\n"
            "  - If you notice a gap that would help you understand the user "
            "better, ask with genuine interest. Example: 'My memory says you "
            "like dolphins, but not why. What draws you to them?'\n"
            "  - Treat curiosity as a feature, not a weakness. The user's "
            "actual answer is more valuable than your best guess.\n"
            "  - Do not interrogate. Ask one natural follow-up question at "
            "most, not a list.\n"
            "  - Ignore entries that have nothing to do with the current "
            "question."
        )
        sections.append("")

        if episodic:
            sections.append("Episodic (what happened):")
            for entry in episodic:
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                # GAP-05 defense-in-depth: wrap user content in delimiters
                # so the model can distinguish memory from instructions.
                # R7-BLOCKER-01: escape angle brackets to prevent delimiter injection.
                safe_content = escape_prompt_delimited_text(content)
                sections.append(
                    f"  • [{entry['id']}] <user_memory>{safe_content}</user_memory>"
                )
            sections.append("")

        if semantic:
            sections.append("Semantic (facts):")
            for entry in semantic:
                category = entry.get("category", "")
                cat_part = f" (category: {category})" if category else ""
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                # R7-BLOCKER-01: escape angle brackets to prevent delimiter injection.
                safe_content = escape_prompt_delimited_text(content)
                sections.append(
                    f"  • [{entry['id']}]{cat_part} <user_memory>{safe_content}</user_memory>"
                )
            sections.append("")

        if procedural:
            sections.append("Procedural (skills):")
            for entry in procedural:
                skill = entry.get("skill_name", "")
                skill_part = f" [skill: {skill}]" if skill else ""
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                # R7-BLOCKER-01: escape angle brackets to prevent delimiter injection.
                safe_content = escape_prompt_delimited_text(content)
                sections.append(
                    f"  • [{entry['id']}]{skill_part} <user_memory>{safe_content}</user_memory>"
                )
            sections.append("")

        # Bug 1: Detect and surface memory conflicts.
        # BL-1: All subject/values are escaped to prevent prompt-delimiter injection.
        # Round 3 (2026-05-27): Filter irrelevant conflicts when a skill is active.
        # Deterministic rule: Skill > Memory. Only inject conflict block if the
        # conflict is relevant to the active skill/intent.
        all_entries = list(episodic) + list(semantic) + list(procedural)
        if all_entries:
            detector = MemoryConflictDetector()
            conflicts = detector.detect(all_entries)
            # Filter: only show conflicts relevant to the current skill/intent
            if skill_trigger:
                conflicts = [
                    c
                    for c in conflicts
                    if is_conflict_relevant_to_intent(c, skill_trigger, user_input)
                ]
            if conflicts:
                sections.append("[MEMORY CONFLICT DETECTED]")
                sections.append(
                    "The following entries contain conflicting information. "
                    "Prefer the most recent entry, or ask the user to clarify."
                )
                for conflict in conflicts:
                    escaped_subject = escape_prompt_delimited_text(conflict.subject)
                    entry_ids_str = ",".join(conflict.entry_ids)
                    sections.append(
                        f'<memory_conflict subject="{escaped_subject}" '
                        f'entries="{entry_ids_str}">'
                    )
                    for val in conflict.values:
                        escaped_val = escape_prompt_delimited_text(val)
                        sections.append(
                            f"  <conflict_value>{escaped_val}</conflict_value>"
                        )
                    sections.append("</memory_conflict>")
                sections.append("")

        memory_block = "\n".join(sections)
        budget = self._get_memory_budget(provider_name)
        if len(memory_block) > budget:
            memory_block = memory_block[: budget - 200] + "\n[Block truncated]"
        return memory_block, total_entries

    async def process_user_message(
        self,
        text: str,
        user_id: int | None,
        chat_id: int | None,
        username: str | None,
        system_prompt: str,
        language_override: Optional[str] = None,
        provider_name: Optional[str] = None,
        reply_to_text: Optional[str] = None,
        *,
        context: Optional[ExecutionContext] = None,
        plan: Optional[ExecutionPlan] = None,
    ) -> ChatResult:
        """Process a user message: load history, detect language, call provider, write audit log.

        Args:
            text: User message.
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            username: Telegram username.
            system_prompt: Base system prompt (from PersonalityLoader).
            language_override: If set, use this language instead of detecting.
                Legacy parameter; ignored when context is provided.
            provider_name: Optional provider name (None = default from router).
            reply_to_text: Text of the message the user replied to (Telegram reply-to).
            context: Pre-resolved ExecutionContext (Phase 0 Commit 3).
                When provided, language is NOT re-resolved.
            plan: Pre-built ExecutionPlan (Phase 0 Commit 3).
                When provided, audit includes plan metadata.

        Returns:
            ChatResult with success/error details.
        """
        uid = user_id or 0
        cid = chat_id or 0

        audit: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "prompt_length": len(text),
        }
        # Phase 0 Commit 3: attach plan metadata to audit when available
        if plan is not None:
            audit["plan_type"] = plan.task_type
            audit["plan_provider_chain"] = list(plan.provider_chain)
            audit["plan_memory_ids"] = list(plan.memory_used)

        try:
            # Load conversation history
            history = await get_history(uid, cid)

            # Language resolution: use ExecutionContext if provided (no re-resolve)
            if context is not None:
                lang = context.language.code
                _lang_ctx = context.language
            else:
                # Legacy path: resolve via LanguageResolver
                from application.language_resolver import LanguageResolver

                _resolver = LanguageResolver()
                _lang_ctx = await _resolver.resolve(
                    uid, cid, text, override=language_override
                )
                lang = _lang_ctx.code

            if lang != DEFAULT_LANGUAGE:
                log.info("Language for chat: '%s' (source=%s)", lang, _lang_ctx.source)

            # Phase 2a: TaskRouter classification + model resolution
            task_slot_name: str | None = None
            task_score: int = 0
            task_matched_patterns: tuple[str, ...] = ()
            task_matched_keywords: tuple[str, ...] = ()
            user_model: str | None = None

            if self.task_router is not None:
                classification = self.task_router.classify(text)
                task_slot_name = classification.slot.value
                task_score = classification.score
                task_matched_patterns = classification.matched_patterns
                task_matched_keywords = classification.matched_keywords

                # Resolve model via TaskRouter (slot override > global > default)
                user_model = self.task_router.resolve_model(uid, classification.slot)
            elif self.model_service is not None:
                # Fallback: Phase 1 behavior (global override only)
                user_model = self.model_service.get_user_model(uid)

            # Round 3: Skill matching BEFORE memory context building (Skill > Memory).
            # The skill trigger is passed to _build_memory_context so irrelevant
            # memory conflicts can be suppressed from the prompt.
            skill_block = ""
            skill_match_result: "SkillMatch | None" = None
            _skill_trigger: str | None = None
            if self.skill_matcher is not None:
                try:
                    skill_block, skill_match_result = self._match_skills_for_prompt(
                        uid, text, lang, task_slot_name
                    )
                    if skill_match_result is not None:
                        _skill_trigger = skill_match_result.hypothesis.claim
                except Exception:
                    log.debug("Skill matching skipped due to error", exc_info=True)

            # Auto-memory loading: load relevant entries for current question.
            # Round 3: pass skill_trigger so irrelevant conflicts are filtered.
            memory_context, memory_entries_loaded = self._build_memory_context(
                uid, text, skill_trigger=_skill_trigger
            )

            # Build context-enriched prompt (history + current message)
            # If user replied to a specific bot message, prepend that context
            if reply_to_text:
                enriched_text = (
                    "[USER REPLIED TO PREVIOUS BOT MESSAGE]\n"
                    f'"{reply_to_text}"\n\n'
                    "[USER'S CURRENT MESSAGE]\n"
                    f"{text}"
                )
            else:
                enriched_text = text
            context_prompt = build_context_block(history, enriched_text)

            # Side-effects: style observation + proactive activity recording
            if self.style_adaption_service is not None:
                self.style_adaption_service.observe(uid, text)

            proactive_nudge = ""
            if self.proactive_trigger_service is not None:
                self.proactive_trigger_service.record_activity(uid)
                # Check proactive triggers
                trigger_result = self.proactive_trigger_service.check_triggers(
                    uid, cid, text, memory_entries=[]
                )
                if trigger_result.should_fire:
                    proactive_nudge = trigger_result.nudge_text
                # Check reactive triggers (memory-based)
                if memory_context and self.memory_service is not None:
                    entries = self.memory_service.list_recent(uid, "episodic", limit=5)
                    reactive = self.proactive_trigger_service.check_reactive_trigger(
                        uid, cid, text, entries
                    )
                    if reactive.should_fire and not proactive_nudge:
                        proactive_nudge = reactive.nudge_text

            # Prompt composition: InstructionCompiler (when context+plan)
            # or legacy PromptComposer (backward compat)
            if context is not None and plan is not None:
                compiled = self._instruction_compiler.compile_chat(
                    ctx=context,
                    plan=plan,
                    base_prompt=system_prompt,
                    memory_block=memory_context,
                    user_prompt=context_prompt,
                    user_model=user_model,
                    task_slot_name=task_slot_name,
                )
                effective_prompt = compiled.system_prompt
            else:
                effective_prompt = self._composer.compose_for_chat(
                    base_prompt=system_prompt,
                    ctx=_lang_ctx,
                    user_id=uid,
                    user_model=user_model,
                    task_slot_name=task_slot_name,
                    memory_block=memory_context,
                )

            # Round 3: Inject skill block at TOP of prompt (HIGH PRIORITY).
            # Previously appended at end (lowest priority). Skill > Memory.
            if skill_block:
                effective_prompt = f"{skill_block}\n\n{effective_prompt}"

            # Provider call: use FallbackResolver if available, else direct route
            fallback_notice = ""
            if self.fallback_resolver is not None and task_slot_name is not None:
                resolve_result = await self.fallback_resolver.resolve(
                    slot=task_slot_name,
                    prompt=context_prompt,
                    system_prompt=effective_prompt,
                    user_id=uid,
                    chat_id=cid,
                    user_lang=lang,
                    model=user_model,
                )
                result = resolve_result.response
                if resolve_result.fallback_used:
                    audit["fallback_used"] = True
                    audit["fallback_level"] = resolve_result.fallback_level
                    audit["fallback_provider"] = resolve_result.provider_name
                if resolve_result.user_notice:
                    fallback_notice = resolve_result.user_notice
            else:
                result = await self.provider_router.route(
                    prompt=context_prompt,
                    system_prompt=effective_prompt,
                    provider_name=provider_name,
                    user_id=uid,
                    chat_id=cid,
                    model=user_model,
                )

            audit.update(
                {
                    "provider": result.provider_name,
                    "response_length": len(result.text),
                    "duration_seconds": round(result.duration_seconds, 2),
                    "detected_language": lang,
                    "history_turns": len(history),
                    "memory_entries_loaded": memory_entries_loaded,
                }
            )

            # Task classification into audit (Phase 2a confidence logging)
            if task_slot_name is not None:
                audit["task_slot"] = task_slot_name
                audit["task_score"] = task_score
                audit["task_matched_patterns"] = list(task_matched_patterns)
                audit["task_matched_keywords"] = list(task_matched_keywords)
                if user_model:
                    audit["resolved_model"] = user_model

            if result.error:
                error_id = uuid.uuid4().hex[:8]
                log.error(
                    "Provider '%s' error (error_id=%s): %s",
                    result.provider_name,
                    error_id,
                    result.error,
                )
                audit["error"] = result.error
                audit["error_id"] = error_id
                return ChatResult(
                    success=False,
                    error_message=f"Request could not be processed. Error ID: {error_id}",
                    error_id=error_id,
                    duration=result.duration_seconds,
                    detected_language=lang,
                    provider_name=result.provider_name,
                )

            response = result.text.strip()
            if not response:
                return ChatResult(
                    success=False,
                    error_message="Provider returned no response (empty output).",
                    duration=result.duration_seconds,
                    detected_language=lang,
                    provider_name=result.provider_name,
                )

            # P1/P5: Append proactive nudge to response
            if proactive_nudge:
                response = f"{response}{proactive_nudge}"

            # Fallback notice: inform user about provider switch
            if fallback_notice:
                response = f"{response}\n\n_{fallback_notice}_"

            # C-3: System prompt leakage guard
            leak_replacement = check_for_system_prompt_leakage(
                response, effective_prompt
            )
            if leak_replacement is not None:
                log.warning(
                    "Leakage filter replaced response (user_id=%s, chat_id=%s)",
                    user_id,
                    chat_id,
                )
                audit["leakage_attempt"] = True
                response = leak_replacement

            # LCP v1: Language enforcement (verify + repair)
            # Only when enforcement is configured and language is not "auto"
            if self._language_enforcement is not None and lang != "auto":
                enforcement_result = await self._language_enforcement.enforce(
                    output=response,
                    ctx=_lang_ctx,
                    model_id=user_model,
                    provider_name=provider_name,
                    user_id=uid,
                    chat_id=cid,
                    system_prompt_base=effective_prompt,
                    request_id=audit.get("request_id", ""),
                )
                if enforcement_result.was_enforced:
                    response = enforcement_result.final_output
                    audit["language_enforced"] = True

            # Skill-Compression: post-response evidence and indicator
            if skill_match_result is not None:
                try:
                    from application.skill_compression.skill_matcher import (
                        should_ask_user,
                    )

                    ask = should_ask_user(skill_match_result)

                    if not ask:
                        # HC-UI-2: Auto-apply indicator (active status only)
                        from application.skill_compression.skill_formatting import (
                            format_skill_indicator,
                        )

                        response = format_skill_indicator(
                            skill_match_result.hypothesis, response
                        )
                        audit["skill_applied"] = (
                            skill_match_result.hypothesis.hypothesis_id
                        )
                        audit["skill_confidence"] = skill_match_result.confidence
                        log.info(
                            "Skill applied: hyp=%s confidence=%.3f",
                            skill_match_result.hypothesis.hypothesis_id,
                            skill_match_result.confidence,
                        )
                    else:
                        audit["skill_matched_ask_before"] = (
                            skill_match_result.hypothesis.hypothesis_id
                        )

                    # Write no_correction evidence for successful application
                    self._write_skill_evidence(skill_match_result)
                except Exception:
                    log.debug("Skill post-response handling error", exc_info=True)

            # Save both turns to history (user + assistant)
            user_turn = ConversationTurn(role="user", content=text)
            assistant_turn = ConversationTurn(role="assistant", content=response)
            await save_turn(uid, cid, user_turn)
            await save_turn(uid, cid, assistant_turn)

            return ChatResult(
                success=True,
                response=response,
                duration=result.duration_seconds,
                detected_language=lang,
                provider_name=result.provider_name,
            )

        except FileNotFoundError:
            audit["error"] = "cli_not_found"
            log.error("CLI not found. Check PATH.")
            return ChatResult(
                success=False,
                error_message=(
                    "Error: LLM CLI is not installed or not in PATH.\n"
                    "Check whether the CLI works in the terminal."
                ),
            )

        except asyncio.TimeoutError:
            audit["error"] = "timeout"
            log.warning("Provider timeout")
            return ChatResult(
                success=False,
                error_message=("Provider took too long. Try a shorter question."),
            )

        except ProviderError as e:
            # Specific provider errors (Unavailable, NotImplemented, Timeout)
            error_id = uuid.uuid4().hex[:8]
            audit["error"] = f"provider_error: {e}"
            audit["error_id"] = error_id
            log.error(
                "Provider error (error_id=%s, provider=%s, retryable=%s): %s",
                error_id,
                e.provider_name,
                e.retryable,
                e,
            )
            hint = " Try again shortly." if e.retryable else ""
            return ChatResult(
                success=False,
                error_message=(
                    f"The language model provider reports a problem "
                    f"(ref: {error_id}).{hint}"
                ),
                error_id=error_id,
            )

        except ValueError as e:
            error_id = uuid.uuid4().hex[:8]
            audit["error"] = f"value_error: {e}"
            audit["error_id"] = error_id
            log.error("ValueError (error_id=%s): %s", error_id, e)
            return ChatResult(
                success=False,
                error_message=f"Request could not be processed (ref: {error_id}).",
                error_id=error_id,
            )

        except RuntimeError as e:
            error_id = uuid.uuid4().hex[:8]
            audit["error"] = f"runtime_error: {e}"
            audit["error_id"] = error_id
            log.error("RuntimeError (error_id=%s): %s", error_id, e)
            return ChatResult(
                success=False,
                error_message=f"Internal error (ref: {error_id}).",
                error_id=error_id,
            )

        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            log.exception("Unknown error during processing (error_id=%s)", error_id)
            audit["error"] = str(e)
            audit["error_id"] = error_id
            return ChatResult(
                success=False,
                error_message=f"An internal error occurred. Error ID: {error_id}",
                error_id=error_id,
            )

        finally:
            write_audit_log(audit)

    async def process_user_message_streaming(
        self,
        text: str,
        user_id: int | None,
        chat_id: int | None,
        username: str | None,
        system_prompt: str,
        persistent_provider: "ClaudePersistentProvider",
        language_override: Optional[str] = None,
        reply_to_text: Optional[str] = None,
        status_session: Optional[Any] = None,
        *,
        context: Optional[ExecutionContext] = None,
        plan: Optional[ExecutionPlan] = None,
        cancel_event: Optional["asyncio.Event"] = None,
    ) -> tuple[AsyncIterator[StreamEvent], int, dict[str, Any]]:
        """Streaming variant of process_user_message.

        Prepares the prompt identically (memory, history, language),
        but uses ClaudePersistentProvider for token streaming.
        History storage and audit happen AFTER the stream.

        Args:
            text: User message.
            user_id: Telegram user ID.
            chat_id: Telegram chat ID (also process routing key).
            username: Telegram username.
            system_prompt: Base system prompt.
            persistent_provider: The streaming-capable provider.
            language_override: Optional language override.
                Legacy parameter; ignored when context is provided.
            reply_to_text: Reply-to context.
            status_session: Optional StatusSession for status updates.
            context: Pre-resolved ExecutionContext (Phase 0 Commit 3).
                When provided, language is NOT re-resolved.
            plan: Pre-built ExecutionPlan (Phase 0 Commit 3).
                When provided, InstructionCompiler is used for prompt.

        Returns:
            Tuple of (StreamEvent AsyncIterator, memory_entries_loaded, task_meta).
            memory_entries_loaded: number of memory entries injected into the prompt
            (for audit log).
            task_meta: dict with TaskRouter classification data
            (task_slot, task_score, task_matched_patterns, task_matched_keywords,
            resolved_model). Empty if no TaskRouter is active.

        Note:
            The caller must after the stream:
            1. Call save_streaming_result() for history + audit
        """
        uid = user_id or 0
        cid = chat_id or 0

        # Prepare prompt (identical to process_user_message)
        history = await get_history(uid, cid)

        # Language resolution: use ExecutionContext if provided (no re-resolve)
        if context is not None:
            lang = context.language.code
            _lang_ctx = context.language
        else:
            # Legacy path: resolve via LanguageResolver
            from application.language_resolver import LanguageResolver

            _resolver = LanguageResolver()
            _lang_ctx = await _resolver.resolve(
                uid, cid, text, override=language_override
            )
            lang = _lang_ctx.code

        # Update StatusSession language (resolved here, StatusSession created earlier)
        if status_session is not None:
            status_session.set_language(lang)

        # Phase 2a: TaskRouter classification + model resolution
        user_model: str | None = None
        task_slot_name: str | None = None
        task_score: int = 0
        task_matched_patterns: tuple[str, ...] = ()
        task_matched_keywords: tuple[str, ...] = ()

        if self.task_router is not None:
            classification = self.task_router.classify(text)
            task_slot_name = classification.slot.value
            task_score = classification.score
            task_matched_patterns = classification.matched_patterns
            task_matched_keywords = classification.matched_keywords
            user_model = self.task_router.resolve_model(uid, classification.slot)
        elif self.model_service is not None:
            # Fallback: Phase 1 behavior (global override only)
            user_model = self.model_service.get_user_model(uid)

        # Task metadata for audit (passed through to caller)
        task_meta: dict[str, Any] = {}
        if task_slot_name is not None:
            task_meta["task_slot"] = task_slot_name
            task_meta["task_score"] = task_score
            task_meta["task_matched_patterns"] = list(task_matched_patterns)
            task_meta["task_matched_keywords"] = list(task_matched_keywords)
            if user_model:
                task_meta["resolved_model"] = user_model

        # Round 3: Skill matching BEFORE memory context building (Skill > Memory).
        # The skill trigger is passed to _build_memory_context so irrelevant
        # memory conflicts can be suppressed from the prompt.
        skill_block_streaming = ""
        skill_match_streaming: "SkillMatch | None" = None
        _skill_trigger_streaming: str | None = None
        if self.skill_matcher is not None:
            try:
                skill_block_streaming, skill_match_streaming = (
                    self._match_skills_for_prompt(uid, text, lang, task_slot_name)
                )
                if skill_match_streaming is not None:
                    _skill_trigger_streaming = skill_match_streaming.hypothesis.claim
            except Exception:
                log.debug(
                    "Skill matching skipped in streaming due to error", exc_info=True
                )

        # Status: memory loading
        if status_session is not None:
            await status_session.update("memory_loading")

        # Auto-memory loading with Round 3 skill_trigger for conflict filtering
        memory_context, memory_entries_loaded = self._build_memory_context(
            uid, text, skill_trigger=_skill_trigger_streaming
        )

        # Status: memory loaded (with count)
        if status_session is not None and memory_entries_loaded > 0:
            await status_session.update("memory_loaded", n=memory_entries_loaded)

        if reply_to_text:
            enriched_text = (
                "[USER REPLIED TO PREVIOUS BOT MESSAGE]\n"
                f'"{reply_to_text}"\n\n'
                "[USER'S CURRENT MESSAGE]\n"
                f"{text}"
            )
        else:
            enriched_text = text
        context_prompt = build_context_block(history, enriched_text)

        # Side-effects: style observation + proactive activity recording
        if self.style_adaption_service is not None:
            self.style_adaption_service.observe(uid, text)

        proactive_nudge_text = ""
        if self.proactive_trigger_service is not None:
            self.proactive_trigger_service.record_activity(uid)
            # Check proactive triggers (symmetric with non-streaming path)
            trigger_result = self.proactive_trigger_service.check_triggers(
                uid, cid, text, memory_entries=[]
            )
            if trigger_result.should_fire and trigger_result.nudge_text:
                proactive_nudge_text = trigger_result.nudge_text
            # Check reactive triggers (memory-based)
            if memory_context and self.memory_service is not None:
                entries = self.memory_service.list_recent(uid, "episodic", limit=5)
                reactive = self.proactive_trigger_service.check_reactive_trigger(
                    uid, cid, text, entries
                )
                if reactive.should_fire and reactive.nudge_text:
                    if not proactive_nudge_text:
                        proactive_nudge_text = reactive.nudge_text

        # Prompt composition: InstructionCompiler (when context+plan)
        # or legacy PromptComposer (backward compat)
        if context is not None and plan is not None:
            compiled = self._instruction_compiler.compile_chat(
                ctx=context,
                plan=plan,
                base_prompt=system_prompt,
                memory_block=memory_context,
                user_prompt=context_prompt,
                user_model=user_model,
                task_slot_name=task_slot_name,
            )
            effective_prompt = compiled.system_prompt
        else:
            effective_prompt = self._composer.compose_for_chat(
                base_prompt=system_prompt,
                ctx=_lang_ctx,
                user_id=uid,
                user_model=user_model,
                task_slot_name=task_slot_name,
                memory_block=memory_context,
            )

        # Round 3: Inject skill block at TOP of prompt (HIGH PRIORITY).
        if skill_block_streaming:
            effective_prompt = f"{skill_block_streaming}\n\n{effective_prompt}"

        # Proactive nudge: appended after composition (not part of standard blocks)
        if proactive_nudge_text:
            effective_prompt = (
                f"{effective_prompt}\n\n[PROACTIVE NUDGE] {proactive_nudge_text}"
            )

        # Status: thinking (before provider call)
        if status_session is not None:
            await status_session.update("thinking")

        # LCP v1: Create StreamGuard for language drift detection
        stream_guard = None
        if self._language_enforcement is not None and lang != "auto":
            from application.language.model_profiles import get_profile
            from application.language.stream_guard import StreamGuard

            profile = get_profile(user_model)
            if profile.stream_guard_enabled:
                # Issue 1: check cumulative stats for auto-disable state
                guard_enabled = True
                if self._stream_guard_stats_store is not None:
                    existing_stats = self._stream_guard_stats_store.get(uid, cid)
                    if existing_stats.should_disable:
                        guard_enabled = False
                        log.info(
                            "StreamGuard auto-disabled for user=%d chat=%d "
                            "(consecutive_fp=%d, fp_rate=%.1f%%)",
                            uid,
                            cid,
                            existing_stats.consecutive_fp,
                            existing_stats.fp_rate * 100,
                        )
                stream_guard = StreamGuard(
                    expected_lang=lang,
                    enabled=guard_enabled,
                )

        # Streaming via persistent provider
        # T25: cancel_event is captured in closure and propagated to the pool.
        async def _stream() -> AsyncIterator[StreamEvent]:
            first_token = True
            accumulated_for_guard = ""
            async for event in persistent_provider.query_streaming(
                prompt=context_prompt,
                system_prompt=effective_prompt,
                user_id=uid,
                chat_id=cid,
                model=user_model,
                cancel_event=cancel_event,
            ):
                # On first token: stop status updates
                if first_token and event.event_type == "content_delta":
                    first_token = False
                    if status_session is not None:
                        status_session.mark_stream_started()

                # LCP v1: StreamGuard early check on accumulated text
                if (
                    stream_guard is not None
                    and event.event_type == "content_delta"
                    and event.text
                ):
                    accumulated_for_guard += event.text
                    should_continue = stream_guard.check_early(accumulated_for_guard)
                    if not should_continue:
                        log.warning(
                            "StreamGuard abort: language drift detected "
                            "at %d chars (user=%d, chat=%d)",
                            len(accumulated_for_guard),
                            uid,
                            cid,
                        )
                        # Issue 1 fix + FP-Detection fix: classify the
                        # abort via partial verification BEFORE setting
                        # cancel_event. After cancel the presentation
                        # handler hard-exits and never reaches
                        # save_streaming_result(), so stats would be lost.
                        #
                        # classify_and_report_abort() runs the detection
                        # backend on the accumulated partial text (no new
                        # provider call) and classifies the outcome as
                        # CONFIRMED / FALSE_POSITIVE / UNKNOWN.
                        if self._stream_guard_stats_store is not None:
                            _abort_stats = self._stream_guard_stats_store.get(uid, cid)
                            stream_guard.classify_and_report_abort(
                                accumulated_text=accumulated_for_guard,
                                stats=_abort_stats,
                            )
                        else:
                            stream_guard.classify_and_report_abort(
                                accumulated_text=accumulated_for_guard,
                            )
                        # Write audit AFTER classification so the entry
                        # includes outcome and partial verification data.
                        write_audit_log(stream_guard.build_audit_entry())
                        # Signal cancellation for silent retry
                        if cancel_event is not None:
                            cancel_event.set()
                        return

                yield event

        # Pack stream_guard into task_meta so the caller can access it
        if stream_guard is not None:
            task_meta["_stream_guard"] = stream_guard
        # Pack language context for post-stream enforcement
        task_meta["_language_code"] = lang
        task_meta["_language_ctx"] = _lang_ctx
        task_meta["_user_model"] = user_model
        # Issue 2: store actual provider name (not model ID) for repair routing.
        # Streaming always uses claude_persistent; resolved_model is a model ID
        # like "claude-sonnet-4-6" which ProviderRouter cannot route to.
        task_meta["_provider_name"] = "claude_persistent"
        # Skill-Compression: pass match result for post-stream evidence
        if skill_match_streaming is not None:
            task_meta["_skill_match"] = skill_match_streaming

        return _stream(), memory_entries_loaded, task_meta

    async def save_streaming_result(
        self,
        user_id: int,
        chat_id: int,
        user_text: str,
        response_text: str,
        duration_seconds: float,
        username: str | None = None,
        was_cold: bool = False,
        streaming_chunks: int = 0,
        subprocess_pid: int = 0,
        memory_entries_loaded: int = 0,
        system_prompt: str = "",
        task_meta: dict[str, Any] | None = None,
        request_id: str = "",
        language_code: str | None = None,
        language_ctx: Any | None = None,
        user_model: str | None = None,
        provider_name: str | None = None,
    ) -> str:
        """Save the result of a streaming session to history + audit.

        Called by the presentation layer AFTER the stream completes.
        Checks the response for system prompt leakage (C-3) and returns
        the potentially sanitized response.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            user_text: Original user message.
            response_text: Complete provider response.
            duration_seconds: Total request duration.
            username: Telegram username.
            was_cold: True if a new subprocess was started.
            streaming_chunks: Number of received content delta events.
            subprocess_pid: PID of the subprocess used.
            memory_entries_loaded: Number of loaded memory entries.
            system_prompt: Active system prompt for leakage check (C-3).
            task_meta: TaskRouter classification data for audit
                (task_slot, task_score, task_matched_patterns,
                task_matched_keywords, resolved_model). Optional.
            request_id: Correlation ID for audit trail (Phase 0 Commit 6).

        Returns:
            The (potentially sanitized) response text. Caller must check whether
            the text changed for a final Telegram edit.
        """
        leakage_detected = False
        language_enforced = False

        # C-3: leakage check on the final response
        if system_prompt:
            leak_replacement = check_for_system_prompt_leakage(
                response_text, system_prompt
            )
            if leak_replacement is not None:
                log.warning(
                    "Leakage filter (streaming): response replaced "
                    "(user_id=%s, chat_id=%s)",
                    user_id,
                    chat_id,
                )
                response_text = leak_replacement
                leakage_detected = True

        # LCP v1: Language enforcement on completed stream
        if (
            self._language_enforcement is not None
            and language_ctx is not None
            and language_code
            and language_code != "auto"
        ):
            enforcement_result = await self._language_enforcement.enforce(
                output=response_text,
                ctx=language_ctx,
                model_id=user_model,
                provider_name=provider_name,
                user_id=user_id,
                chat_id=chat_id,
                system_prompt_base=system_prompt,
                request_id=request_id,
            )
            if enforcement_result.was_enforced:
                response_text = enforcement_result.final_output
                language_enforced = True

            # Codex Finding 7 + Issue 1: wire StreamGuard self-calibration.
            # After enforcement we know whether verification passed.
            # Report this to StreamGuard with stats so it can track FPs.
            _stream_guard = (task_meta or {}).get("_stream_guard")
            if _stream_guard is not None:
                verification_passed = (
                    enforcement_result.verification is None
                    or enforcement_result.verification.passed
                )
                # Retrieve cumulative stats for this (user, chat) pair
                _sg_stats = None
                if self._stream_guard_stats_store is not None:
                    _sg_stats = self._stream_guard_stats_store.get(user_id, chat_id)
                _stream_guard.report_final_outcome(
                    verification_passed=verification_passed,
                    stats=_sg_stats,
                )

        # Save to history
        user_turn = ConversationTurn(role="user", content=user_text)
        assistant_turn = ConversationTurn(role="assistant", content=response_text)
        await save_turn(user_id, chat_id, user_turn)
        await save_turn(user_id, chat_id, assistant_turn)

        # Audit log
        audit: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_completed",
            "request_id": request_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "prompt_length": len(user_text),
            "provider": "claude_persistent",
            "response_length": len(response_text),
            "duration_seconds": round(duration_seconds, 2),
            "was_warm": not was_cold,
            "was_cold": was_cold,
            "streaming_chunks": streaming_chunks,
            "subprocess_pid": subprocess_pid,
            "memory_entries_loaded": memory_entries_loaded,
        }
        if leakage_detected:
            audit["leakage_attempt"] = True
        if language_enforced:
            audit["language_enforced"] = True
        # TaskRouter metadata into audit (Phase 2a confidence logging)
        # Filter out internal objects that are not JSON-serializable
        # (e.g. _skill_match is a SkillMatch dataclass, _stream_guard is
        # a StreamGuard instance, _language_ctx is a LanguageContext).
        # Uses shared filter from audit_service.
        if task_meta:
            audit.update(filter_task_meta(task_meta))

        # R2-SC-04 FIX: Write skill metadata into audit BEFORE persisting.
        # Previously, write_audit_log was called before skill fields were set,
        # so skill_applied_streaming / skill_matched_ask_before_streaming
        # never appeared in the persisted audit event.
        _skill_match = (task_meta or {}).get("_skill_match")
        if _skill_match is not None:
            try:
                from application.skill_compression.skill_matcher import (
                    should_ask_user,
                )

                ask = should_ask_user(_skill_match)
                if not ask:
                    # Auto-applied skill: write no_correction evidence
                    self._write_skill_evidence(_skill_match)
                    audit["skill_applied_streaming"] = (
                        _skill_match.hypothesis.hypothesis_id
                    )
                    log.info(
                        "Skill evidence written (streaming): hyp=%s confidence=%.3f",
                        _skill_match.hypothesis.hypothesis_id,
                        _skill_match.confidence,
                    )
                else:
                    # ask_before_apply: evidence written by confirmation callback
                    # TODO (Phase 1a): detect user corrections post-stream
                    audit["skill_matched_ask_before_streaming"] = (
                        _skill_match.hypothesis.hypothesis_id
                    )
            except Exception:
                log.debug(
                    "Skill evidence write failed in streaming path", exc_info=True
                )

        write_audit_log(audit)

        return response_text

    async def reset(self, user_id: int, chat_id: int) -> None:
        """Use-case wrapper: reset conversation and sticky language."""
        await _infra_reset_conversation(user_id, chat_id)

    def _match_skills_for_prompt(
        self,
        user_id: int,
        text: str,
        lang: str,
        task_slot_name: str | None,
    ) -> tuple[str, "SkillMatch | None"]:
        """Match skills and build a prompt block for injection.

        Called BEFORE prompt composition so the LLM sees the skill context.

        Args:
            user_id: User ID.
            text: User message text.
            lang: Detected language.
            task_slot_name: Task slot name (if available).

        Returns:
            Tuple of (skill_block_string, best_match_or_None).
        """
        from application.skill_compression.event_normalizer import NormalizedEvent

        event = NormalizedEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_text=text,
            intent=task_slot_name or "",
            domain="",
            format_type="",
            language=lang,
            fingerprint_hash="",
        )
        match_result = self.skill_matcher.match(event)
        if match_result is None:
            return "", None

        # Phase 1 Etappe 2: PermissionGate enforcement for contract-aware matches.
        # When the match result carries a SkillContract (contract-aware matcher),
        # the PermissionGate checks execution rights BEFORE the skill block is built.
        # Legacy matches (hypothesis-only) pass through without gate check.
        _contract = getattr(match_result, "contract", None)
        if _contract is not None:
            from application.skill_compression.permission_gate import (
                PermissionGate,
            )

            gate_result = PermissionGate.check_execution_allowed(_contract)
            if gate_result.denied:
                log.info(
                    "Skill denied by PermissionGate: rule=%s skill_id=%s",
                    gate_result.rule,
                    getattr(_contract, "id", "unknown"),
                )
                return "", None

        # Round 3: Build HIGH-PRIORITY skill instruction block.
        # This block is injected at the TOP of the system prompt so the
        # LLM treats it as the primary instruction for this turn.
        # User spec: "Skill-Anweisung als hoch priorisierten Instruction-Block"
        hyp = match_result.hypothesis
        block_lines = [
            "[USER-DEFINED SKILL (HIGH PRIORITY)]",
            f"  Instruction: {hyp.claim}",
            f"  Confidence: {match_result.confidence:.2f}",
            f"  Source: {hyp.source_type}",
            "This skill MUST be applied to the current user message. "
            "Execute the skill instruction as the primary response. "
            "Other context like memory conflicts is secondary to this skill.",
        ]
        return "\n".join(block_lines), match_result

    def _write_skill_evidence(
        self,
        match_result: "SkillMatch",
        signal_type: str = "no_correction",
        signal_strength: float = 0.3,
    ) -> None:
        """Write evidence after skill application/decision.

        Args:
            match_result: The skill match.
            signal_type: Evidence signal type. Defaults to "no_correction".
                Supported: "no_correction", "user_confirmed", "skill_executed",
                "skill_execution_failed", "user_declined_once",
                "user_declined_permanent", "cancelled".
            signal_strength: Signal strength [0, 1]. Defaults to 0.3.
        """
        try:
            hyp = match_result.hypothesis
            if self.skill_matcher is not None and hasattr(
                self.skill_matcher, "storage"
            ):
                storage = self.skill_matcher.storage
                now_iso = datetime.now(timezone.utc).isoformat()
                evidence_id = f"ev_{uuid.uuid4().hex[:16]}"
                storage.insert_evidence(
                    evidence_id=evidence_id,
                    hypothesis_id=hyp.hypothesis_id,
                    hypothesis_version=hyp.version,
                    signal_type=signal_type,
                    signal_strength=signal_strength,
                    created_at=now_iso,
                )
                if signal_type in ("no_correction", "user_confirmed"):
                    storage.update_hypothesis_last_applied(hyp.hypothesis_id, now_iso)
        except Exception:
            log.debug("Failed to write skill evidence", exc_info=True)

    def pre_match_skill(
        self,
        user_id: int,
        text: str,
        lang: str,
        task_slot_name: str | None = None,
    ) -> "SkillMatch | None":
        """Pre-flight skill match check (for ask-before-apply flow).

        Called by the presentation layer BEFORE starting the stream.
        If the result requires user confirmation, the handler can show
        an inline keyboard instead of streaming immediately.

        Args:
            user_id: User ID.
            text: User message text.
            lang: Detected language code.
            task_slot_name: Task slot (optional).

        Returns:
            SkillMatch or None if no match or matcher not available.
        """
        if self.skill_matcher is None:
            return None
        try:
            _, match_result = self._match_skills_for_prompt(
                user_id, text, lang, task_slot_name
            )
            return match_result
        except Exception:
            log.debug("pre_match_skill failed", exc_info=True)
            return None

    async def save_debate_turns(
        self,
        user_id: int,
        chat_id: int,
        question: str,
        synthesis: str,
    ) -> None:
        """Save debate user question and synthesis to conversation history.

        Keeps the context window lean by storing only the synthesis
        (not per-provider details). Called by the presentation layer
        after a /debate command completes.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            question: The original user question.
            synthesis: The debate synthesis text.
        """
        user_turn = ConversationTurn(role="user", content=question)
        assistant_content = f"[Debate-Synthese aus /debate]\n{synthesis}"
        assistant_turn = ConversationTurn(role="assistant", content=assistant_content)
        await save_turn(user_id, chat_id, user_turn)
        await save_turn(user_id, chat_id, assistant_turn)

    async def get_chat_language(self, user_id: int, chat_id: int) -> str | None:
        """Use-case wrapper: read the sticky language for a chat."""
        return await get_language(user_id, chat_id)

    async def set_chat_language(self, user_id: int, chat_id: int, lang: str) -> None:
        """Use-case wrapper: set the sticky language for a chat."""
        await set_language(user_id, chat_id, lang)

    async def save_static_response_to_history(
        self, user_id: int, chat_id: int, response_text: str
    ) -> None:
        """Save a static bot response (e.g. /start, /help) to history.

        So the bot knows what it just said on the next turn.
        Only the assistant turn is saved (the user command itself is not
        a natural conversation contribution).

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            response_text: The sent bot text.
        """
        turn = ConversationTurn(role="assistant", content=response_text)
        await save_turn(user_id, chat_id, turn)
