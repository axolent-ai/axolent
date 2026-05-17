"""LanguageResolver: single-entry-point resolution for user language.

Solves the core architectural problem identified in the root-cause reviews:
language was determined at multiple points (Handler, ChatService, Debate)
with inconsistent fallbacks. This resolver provides ONE call that returns
a frozen LanguageContext, which all consumers must accept.

Usage:
    resolver = LanguageResolver(conv_storage)
    ctx = await resolver.resolve(user_id, chat_id, text)
    # ctx.code is always a valid ISO-639-1 code, never empty

Design principles:
    - Frozen dataclass: once resolved, language cannot drift
    - Confidence-based smart-switch (threshold 0.7)
    - Explicit source tracking for debugging
    - request_id for audit correlation
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from domain.language import DEFAULT_LANGUAGE, detect_language_with_confidence

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Smart-switch threshold: only switch sticky language when detection
# confidence exceeds this value AND the detected language differs.
_SMART_SWITCH_THRESHOLD: float = 0.7

# Minimum confidence to use detection result for first-time users
_FIRST_TIME_THRESHOLD: float = 0.0


@dataclass(frozen=True, slots=True)
class LanguageContext:
    """Immutable language resolution result.

    Once created, this object represents THE language decision
    for an entire request lifecycle. All consumers (ChatService,
    DebateOrchestrator, StatusSession, PromptComposer) must use
    this context rather than resolving language independently.

    Attributes:
        code: ISO-639-1 language code, guaranteed non-empty.
        source: How the language was determined.
        confidence: Detection confidence (0.0..1.0). For sticky/override: 1.0.
        switched_from: Previous sticky value if a smart-switch occurred.
        request_id: Unique ID for audit correlation.
    """

    code: str
    source: Literal["override", "sticky", "detected", "default"]
    confidence: float
    switched_from: Optional[str]
    request_id: str

    def effective_lang(self) -> str:
        """Return the effective language code.

        Convenience method for contexts that only need the code.
        """
        return self.code

    @property
    def was_smart_switched(self) -> bool:
        """True if a smart-switch occurred (user implicitly changed language)."""
        return self.switched_from is not None


class LanguageResolver:
    """Single-entry-point language resolution for all request paths.

    Replaces the duplicated resolution logic in ChatService (both
    streaming and non-streaming paths) and DebateOrchestrator.

    Resolution priority:
        1. Explicit override (from /lang command or language_override param)
        2. Sticky language with smart-switch detection
        3. Detection from message text
        4. Default language (from config, typically "de")
    """

    def __init__(self, default_lang: str = DEFAULT_LANGUAGE) -> None:
        """Initialize the resolver.

        Args:
            default_lang: Fallback language when no other signal exists.
        """
        self._default = default_lang

    async def resolve(
        self,
        user_id: int,
        chat_id: int,
        text: str,
        override: Optional[str] = None,
    ) -> LanguageContext:
        """Resolve the language for a user request.

        This is THE single method all paths must call. After this returns,
        the language decision is final for the entire request.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            text: User message text (for detection).
            override: Explicit language override (e.g. from /lang).

        Returns:
            Frozen LanguageContext with the resolved language.
        """
        # Lazy import to avoid circular dependencies at module level
        from infrastructure.conversation_storage import get_language, set_language

        request_id = uuid.uuid4().hex[:12]

        # Priority 1: Explicit override
        if override:
            log.debug(
                "Language resolved via override: %s (request_id=%s)",
                override,
                request_id,
            )
            return LanguageContext(
                code=override,
                source="override",
                confidence=1.0,
                switched_from=None,
                request_id=request_id,
            )

        # Read sticky language
        sticky = await get_language(user_id, chat_id)

        # Detect from text
        detected, confidence = detect_language_with_confidence(text)

        # Priority 2/3: No sticky yet (first interaction)
        if not sticky:
            if confidence > _FIRST_TIME_THRESHOLD:
                code = detected
                source: Literal["override", "sticky", "detected", "default"] = (
                    "detected"
                )
            else:
                code = self._default
                source = "default"
            await set_language(user_id, chat_id, code)
            log.debug(
                "Language resolved (first time): %s source=%s conf=%.2f (request_id=%s)",
                code,
                source,
                confidence,
                request_id,
            )
            return LanguageContext(
                code=code,
                source=source,
                confidence=confidence,
                switched_from=None,
                request_id=request_id,
            )

        # Priority 2: Sticky exists, check for smart-switch
        if confidence > _SMART_SWITCH_THRESHOLD and detected != sticky:
            # Smart-switch: user implicitly changed language
            await set_language(user_id, chat_id, detected)
            log.info(
                "Smart language switch: %s -> %s (confidence=%.2f, request_id=%s)",
                sticky,
                detected,
                confidence,
                request_id,
            )
            return LanguageContext(
                code=detected,
                source="detected",
                confidence=confidence,
                switched_from=sticky,
                request_id=request_id,
            )

        # Priority 2: Use sticky (no switch needed)
        return LanguageContext(
            code=sticky,
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id=request_id,
        )

    @staticmethod
    def from_code(code: str, source: str = "override") -> LanguageContext:
        """Create a LanguageContext directly from a language code.

        Used for backward compatibility when callers pass a plain string.
        This wraps it into a proper LanguageContext.

        Args:
            code: ISO-639-1 language code.
            source: Source label (default: "override").

        Returns:
            LanguageContext wrapping the given code.
        """
        return LanguageContext(
            code=code,
            source=source,  # type: ignore[arg-type]
            confidence=1.0,
            switched_from=None,
            request_id=uuid.uuid4().hex[:12],
        )
