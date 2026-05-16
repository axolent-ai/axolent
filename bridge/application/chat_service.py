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
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from application.leakage_filter import check_for_system_prompt_leakage
from domain.conversation import ConversationTurn, build_context_block
from domain.language import detect_language_with_confidence
from domain.personality import build_effective_prompt
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

if TYPE_CHECKING:
    from application.memory_service import MemoryService
    from application.model_service import ModelService
    from application.provider_router import ProviderRouter
    from application.self_awareness_service import SelfAwarenessService
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

    def __init__(
        self,
        provider_router: "ProviderRouter",
        memory_service: "MemoryService | None" = None,
        model_service: "ModelService | None" = None,
        task_router: "TaskRouter | None" = None,
        self_awareness_service: "SelfAwarenessService | None" = None,
    ) -> None:
        self.provider_router = provider_router
        self.memory_service = memory_service
        self.model_service = model_service
        self.task_router = task_router
        self.self_awareness_service = self_awareness_service

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
        self, user_id: int, query: str, provider_name: str | None = None
    ) -> tuple[str, int]:
        """Load relevant memory entries for the user based on query.

        Search strategy: keyword extraction -> substring match across all layers.
        Prioritizes longest keywords as primary search.

        Args:
            user_id: Telegram user ID.
            query: Current user message.
            provider_name: Optional provider name for memory budget lookup.

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
            return self._format_memory_context(episodic, semantic, [], provider_name)

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

        # Fallback: if keyword search yields nothing, load recent episodic entries
        # so the LLM can use them as context (covers cases where user asks about
        # stored facts without using matching keywords, e.g. "welche Tiere mag ich"
        # when stored entry is "Ich mag Delfine").
        if not (episodic or semantic or procedural):
            episodic = self.memory_service.list_recent(
                user_id, layer="episodic", limit=5
            )
            semantic = self.memory_service.list_recent(
                user_id, layer="semantic", limit=3
            )
            if not (episodic or semantic):
                return "", 0

        return self._format_memory_context(
            episodic, semantic, procedural, provider_name
        )

    def _format_memory_context(
        self,
        episodic: list[dict],
        semantic: list[dict],
        procedural: list[dict],
        provider_name: str | None = None,
    ) -> tuple[str, int]:
        """Format memory entries into a context block for the system prompt.

        Args:
            episodic: Episodic memory entries.
            semantic: Semantic memory entries.
            procedural: Procedural memory entries.
            provider_name: Optional provider name for budget lookup.

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
                sections.append(f"  • [{entry['id']}] {content}")
            sections.append("")

        if semantic:
            sections.append("Semantic (facts):")
            for entry in semantic:
                category = entry.get("category", "")
                cat_part = f" (category: {category})" if category else ""
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                sections.append(f"  • [{entry['id']}]{cat_part} {content}")
            sections.append("")

        if procedural:
            sections.append("Procedural (skills):")
            for entry in procedural:
                skill = entry.get("skill_name", "")
                skill_part = f" [skill: {skill}]" if skill else ""
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                sections.append(f"  • [{entry['id']}]{skill_part} {content}")
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
    ) -> ChatResult:
        """Process a user message: load history, detect language, call provider, write audit log.

        Args:
            text: User message.
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            username: Telegram username.
            system_prompt: Base system prompt (from PersonalityLoader).
            language_override: If set, use this language instead of detecting.
            provider_name: Optional provider name (None = default from router).
            reply_to_text: Text of the message the user replied to (Telegram reply-to).

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

        try:
            # Load conversation history
            history = await get_history(uid, cid)

            # Auto-memory loading: load relevant entries for current question
            memory_context, memory_entries_loaded = self._build_memory_context(
                uid, text
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

            # Language: smart detection on every message.
            # Sticky is overwritten when detection is clear enough (Variant 1).
            if language_override:
                lang = language_override
            else:
                sticky_lang = await get_language(uid, cid)
                detected_lang, confidence = detect_language_with_confidence(text)

                if not sticky_lang:
                    # First turn: detect and set sticky
                    lang = detected_lang if confidence > 0 else "de"
                    await set_language(uid, cid, lang)
                elif confidence > 0.7 and detected_lang != sticky_lang:
                    # User implicitly switched language
                    lang = detected_lang
                    await set_language(uid, cid, lang)
                    log.info(
                        "Smart language switch: %s -> %s (confidence=%.2f)",
                        sticky_lang,
                        lang,
                        confidence,
                    )
                else:
                    lang = sticky_lang

            if lang != "de":
                log.info("Language for chat: '%s' (sticky)", lang)

            # Build effective prompt with language override
            effective_prompt = build_effective_prompt(system_prompt, lang)

            # Inject memory context into system prompt (before LLM call)
            if memory_context:
                effective_prompt = f"{effective_prompt}\n\n{memory_context}"

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

            # Self-awareness block: inject model info into system prompt
            if self.self_awareness_service is not None:
                self_awareness = self.self_awareness_service.build(
                    user_id=uid,
                    user_model=user_model,
                    task_slot_name=task_slot_name,
                    lang=lang,
                )
                if self_awareness:
                    effective_prompt = f"{effective_prompt}\n\n{self_awareness}"

            # Provider router call (replaces direct claude_cli call)
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
            reply_to_text: Reply-to context.
            status_session: Optional StatusSession for status updates.

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

        # Status: memory loading
        if status_session is not None:
            await status_session.update("memory_loading")

        # Prepare prompt (identical to process_user_message)
        history = await get_history(uid, cid)
        memory_context, memory_entries_loaded = self._build_memory_context(uid, text)

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

        # Language: smart detection on every message (Variant 1).
        if language_override:
            lang = language_override
        else:
            sticky_lang = await get_language(uid, cid)
            detected_lang, confidence = detect_language_with_confidence(text)

            if not sticky_lang:
                lang = detected_lang if confidence > 0 else "de"
                await set_language(uid, cid, lang)
            elif confidence > 0.7 and detected_lang != sticky_lang:
                # User implicitly switched language
                lang = detected_lang
                await set_language(uid, cid, lang)
                log.info(
                    "Smart language switch (streaming): %s -> %s (confidence=%.2f)",
                    sticky_lang,
                    lang,
                    confidence,
                )
            else:
                lang = sticky_lang

        # Update StatusSession language (bug fix: sticky language
        # is only determined here, but StatusSession was created earlier)
        if status_session is not None:
            status_session.set_language(lang)

        effective_prompt = build_effective_prompt(system_prompt, lang)
        if memory_context:
            effective_prompt = f"{effective_prompt}\n\n{memory_context}"

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

        # Self-awareness block: inject model info into system prompt (i18n)
        if self.self_awareness_service is not None:
            self_awareness = self.self_awareness_service.build(
                user_id=uid,
                user_model=user_model,
                task_slot_name=task_slot_name,
                lang=lang,
            )
            if self_awareness:
                effective_prompt = f"{effective_prompt}\n\n{self_awareness}"

        # Status: thinking (before provider call)
        if status_session is not None:
            await status_session.update("thinking")

        # Streaming via persistent provider
        async def _stream() -> AsyncIterator[StreamEvent]:
            first_token = True
            async for event in persistent_provider.query_streaming(
                prompt=context_prompt,
                system_prompt=effective_prompt,
                user_id=uid,
                chat_id=cid,
                model=user_model,
            ):
                # On first token: stop status updates
                if first_token and event.event_type == "content_delta":
                    first_token = False
                    if status_session is not None:
                        status_session.mark_stream_started()
                yield event

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

        Returns:
            The (potentially sanitized) response text. Caller must check whether
            the text changed for a final Telegram edit.
        """
        leakage_detected = False

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

        # Save to history
        user_turn = ConversationTurn(role="user", content=user_text)
        assistant_turn = ConversationTurn(role="assistant", content=response_text)
        await save_turn(user_id, chat_id, user_turn)
        await save_turn(user_id, chat_id, assistant_turn)

        # Audit log
        audit: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_completed",
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
        # TaskRouter metadata into audit (Phase 2a confidence logging)
        if task_meta:
            audit.update(task_meta)
        write_audit_log(audit)

        return response_text

    async def reset(self, user_id: int, chat_id: int) -> None:
        """Use-case wrapper: reset conversation and sticky language."""
        await _infra_reset_conversation(user_id, chat_id)

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
