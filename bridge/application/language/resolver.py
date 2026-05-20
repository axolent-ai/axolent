"""LanguageResolver: single-entry-point resolution for user language.

Canonical location within the Language Control Plane subsystem.
Solves the core architectural problem: language was determined at
multiple points with inconsistent fallbacks. This resolver provides
ONE call that returns a frozen LanguageContext.

Usage:
    resolver = LanguageResolver(conv_storage)
    ctx = await resolver.resolve(user_id, chat_id, text)
    # ctx.code is always a valid ISO-639-1 code, never empty

Design principles:
    - Frozen dataclass: once resolved, language cannot drift
    - Confidence-based smart-switch (threshold 0.7)
    - Explicit source tracking for debugging
    - request_id for audit correlation

Phase 2 migration (Step 4/4):
    - HC-O7: No longer imports domain.language directly.
      Uses DetectionOrchestrator for all language detection.
    - LanguageContext now populated with Phase 2 fields
      (detection_distribution, reliability_score, confidence_history,
      detection_tier, text_length_bucket, backends_consulted).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from application.language.context import LanguageContext
from application.language.orchestrator import (
    DetectionOrchestrator,
    OrchestratedDetection,
)
from application.language.registry import InMemoryLanguageRegistry

log = logging.getLogger(__name__)

# Smart-switch threshold: only switch sticky language when detection
# confidence exceeds this value AND the detected language differs.
_SMART_SWITCH_THRESHOLD: float = 0.7

# Minimum confidence to use detection result for first-time users
_FIRST_TIME_THRESHOLD: float = 0.0

# Default language fallback (matches domain.language.DEFAULT_LANGUAGE).
# Defined here to avoid importing domain.language (HC-O7).
_DEFAULT_LANGUAGE: str = "de"


def _build_default_orchestrator() -> DetectionOrchestrator:
    """Build the default DetectionOrchestrator with production backends.

    HC-O1: LangdetectBackend as primary.
    HC-O2: DomainLanguageBackend as fallback for short text.

    Note: This is called lazily (on first detect() call) to avoid
    importing langdetect at module-load time. The LangdetectBackend
    constructor imports langdetect, and we need to preserve the
    existing lazy-import behavior for environments where langdetect
    is optional.

    Graceful degradation: if langdetect is not installed, falls back
    to DomainLanguageBackend as primary (same behavior as pre-Phase-2
    resolver which used domain.language directly). This preserves
    backward compatibility for test environments without langdetect.
    """
    from application.language.backends import DomainLanguageBackend

    try:
        from application.language.backends import LangdetectBackend

        primary = LangdetectBackend()
    except (ImportError, ModuleNotFoundError):
        log.warning(
            "langdetect not installed. Falling back to DomainLanguageBackend "
            "as primary. Install langdetect for production use."
        )
        primary = DomainLanguageBackend()

    return DetectionOrchestrator(
        primary_backend=primary,
        fallback_backend=DomainLanguageBackend(),
        registry=InMemoryLanguageRegistry(),
    )


def _effective_confidence(detection: OrchestratedDetection) -> float:
    """Compute effective confidence for resolver decisions.

    For smart-switch and first-time detection decisions, the raw backend
    confidence can be misleading for very short texts. langdetect returns
    high confidence for 1-2 word inputs even though detection is unreliable.

    This function checks the DomainLanguageBackend candidate (the heuristic
    backend) if it was consulted. If the heuristic returned confidence <= 0,
    that means it could not reliably identify the language (text too short
    or no marker words found). In that case we clamp to 0.0 to avoid
    trusting langdetect's spurious high-confidence results on very short input.

    This preserves backward compatibility with the pre-Phase-2 behavior
    where domain.language.detect_language_with_confidence() returned
    confidence=0.0 for texts too short to classify reliably.
    """
    if detection.confidence == 0.0:
        return 0.0

    # Check if the domain heuristic was consulted and could not identify
    # the language. The heuristic returns confidence=0.0 for texts that
    # are too short or have no recognizable marker words.
    for candidate in detection.candidates:
        if candidate.backend_name == "domain_heuristic" and candidate.succeeded:
            if candidate.top_confidence <= 0.0:
                # Heuristic says "too short to tell". Don't trust the
                # primary backend's result for decision-making.
                return 0.0
            break

    return detection.confidence


def _detection_to_context(
    detection: OrchestratedDetection,
    request_id: str,
    source: str,
    switched_from: Optional[str] = None,
) -> LanguageContext:
    """Map an OrchestratedDetection to a fully-populated LanguageContext.

    Populates all Phase 2 fields from the orchestrator result.
    """
    return LanguageContext(
        code=detection.code,
        source=source,  # type: ignore[arg-type]
        confidence=detection.confidence,
        switched_from=switched_from,
        request_id=request_id,
        detection_distribution=detection.distribution,
        reliability_score=detection.reliability_score,
        confidence_history=tuple(
            (c.backend_name, c.top_confidence) for c in detection.candidates
        ),
        detection_tier=detection.text_length_bucket
        and _get_detection_tier(detection.code),
        text_length_bucket=detection.text_length_bucket,
        backends_consulted=frozenset(c.backend_name for c in detection.candidates),
    )


def _get_detection_tier(code: str) -> Optional[str]:
    """Look up detection tier for a language code from the registry.

    Returns None if the code is not in the registry.
    """
    registry = InMemoryLanguageRegistry()
    entry = registry.get_or_none(code)
    if entry is not None:
        return entry.detection_tier.value
    return None


class LanguageResolver:
    """Single-entry-point language resolution for all request paths.

    Replaces the duplicated resolution logic in ChatService (both
    streaming and non-streaming paths) and DebateOrchestrator.

    Resolution priority:
        1. Explicit override (from /lang command or language_override param)
        2. Sticky language with smart-switch detection
        3. Detection from message text
        4. Default language (from config, typically "de")

    Phase 2: Detection is delegated to DetectionOrchestrator (HC-O7).
    """

    def __init__(
        self,
        default_lang: str = _DEFAULT_LANGUAGE,
        orchestrator: DetectionOrchestrator | None = None,
    ) -> None:
        """Initialize the resolver.

        Args:
            default_lang: Fallback language when no other signal exists.
            orchestrator: Optional custom DetectionOrchestrator. If None,
                the default production orchestrator is built lazily on
                first detection to avoid importing langdetect at
                module-load time.
        """
        self._default = default_lang
        self._orchestrator: DetectionOrchestrator | None = orchestrator

    def _get_orchestrator(self) -> DetectionOrchestrator:
        """Return the orchestrator, building the default lazily if needed."""
        if self._orchestrator is None:
            self._orchestrator = _build_default_orchestrator()
        return self._orchestrator

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

        # Detect from text via DetectionOrchestrator (HC-O7)
        detection = self._get_orchestrator().detect(text)
        detected = detection.code
        confidence = _effective_confidence(detection)

        # Priority 2/3: No sticky yet (first interaction)
        if not sticky:
            if confidence > _FIRST_TIME_THRESHOLD:
                code = detected
                source = "detected"
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
            return _detection_to_context(
                detection=detection,
                request_id=request_id,
                source=source,
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
            return _detection_to_context(
                detection=detection,
                request_id=request_id,
                source="detected",
                switched_from=sticky,
            )

        # Priority 2: Use sticky (no switch needed)
        return LanguageContext(
            code=sticky,
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id=request_id,
        )

    async def resolve_readonly(
        self,
        user_id: int,
        chat_id: int,
        text: str,
        override: Optional[str] = None,
    ) -> LanguageContext:
        """Read-only language resolution. Safe for preflight/reject UI.

        Same logic as resolve() but NEVER writes to storage. Use this
        before rate-limit or policy decisions where rejected requests
        must not mutate user state.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            text: User message text (for detection).
            override: Explicit language override (e.g. from /lang).

        Returns:
            Frozen LanguageContext (no side-effects).
        """
        from infrastructure.conversation_storage import get_language

        request_id = uuid.uuid4().hex[:12]

        # Priority 1: Explicit override
        if override:
            return LanguageContext(
                code=override,
                source="override",
                confidence=1.0,
                switched_from=None,
                request_id=request_id,
            )

        # Read sticky language (read-only, no set_language call)
        sticky = await get_language(user_id, chat_id)

        # Detect from text via DetectionOrchestrator (HC-O7)
        detection = self._get_orchestrator().detect(text)
        detected = detection.code
        confidence = _effective_confidence(detection)

        # No sticky: use detection or default (but do NOT persist)
        if not sticky:
            if confidence > _FIRST_TIME_THRESHOLD:
                return _detection_to_context(
                    detection=detection,
                    request_id=request_id,
                    source="detected",
                )
            return LanguageContext(
                code=self._default,
                source="default",
                confidence=confidence,
                switched_from=None,
                request_id=request_id,
            )

        # Sticky exists: check if smart-switch WOULD trigger (but don't persist)
        if confidence > _SMART_SWITCH_THRESHOLD and detected != sticky:
            return _detection_to_context(
                detection=detection,
                request_id=request_id,
                source="detected",
                switched_from=sticky,
            )

        # Use sticky (no switch)
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
