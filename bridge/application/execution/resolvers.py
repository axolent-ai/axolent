"""Resolver pipeline: small, composable resolvers for ExecutionContext.

Each resolver fills exactly one part of the PartialExecutionContext.
Resolvers are stateless (receive dependencies via constructor)
and independently testable.

Phase 0 resolvers:
    - LanguageResolverAdapter: wraps existing LanguageResolver
    - TimeResolver: fills TimeContext
    - ChannelResolver: fills ChannelCapabilities
"""

from __future__ import annotations

import abc
import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from application.execution.context import (
    ChannelCapabilities,
    PartialExecutionContext,
    TimeContext,
)
from application.language_resolver import LanguageContext, LanguageResolver

log = logging.getLogger(__name__)


class BaseResolver(abc.ABC):
    """Abstract base for all context resolvers.

    Each resolver takes a PartialExecutionContext, enriches it,
    and returns it. Resolvers must not perform I/O that modifies
    external state (read-only from storage is acceptable).
    """

    @abc.abstractmethod
    async def resolve(
        self, partial: PartialExecutionContext
    ) -> PartialExecutionContext:
        """Enrich the partial context with this resolver's data.

        Args:
            partial: Current partial context (mutable).

        Returns:
            The same partial context instance (mutated in place).
        """
        ...


class LanguageResolverAdapter(BaseResolver):
    """Wraps the existing LanguageResolver into the resolver pipeline.

    Delegates to LanguageResolver.resolve() and stores the result
    in partial.language. If language is already set (e.g. from override),
    this resolver is a no-op.
    """

    def __init__(self, language_resolver: Optional[LanguageResolver] = None) -> None:
        """Initialize with an optional LanguageResolver instance.

        Args:
            language_resolver: Existing LanguageResolver. If None, creates default.
        """
        self._resolver = language_resolver or LanguageResolver()

    async def resolve(
        self, partial: PartialExecutionContext
    ) -> PartialExecutionContext:
        """Resolve language from text, sticky, or override.

        Uses the existing LanguageResolver which handles:
        - Priority 1: explicit override
        - Priority 2: sticky with smart-switch
        - Priority 3: detection from text
        - Priority 4: default language

        Args:
            partial: Partial context with user_id, chat_id, raw_text.

        Returns:
            Partial context with language field populated.
        """
        # Skip if already resolved (e.g. by a higher-priority path)
        if partial.language is not None:
            return partial

        lang_ctx = await self._resolver.resolve(
            user_id=partial.user_id,
            chat_id=partial.chat_id,
            text=partial.raw_text,
            override=partial.language_override,
        )

        # Adopt the LanguageResolver's request_id or use ours
        if lang_ctx.request_id != partial.request_id:
            # Re-wrap with our request_id for consistent correlation
            lang_ctx = LanguageContext(
                code=lang_ctx.code,
                source=lang_ctx.source,
                confidence=lang_ctx.confidence,
                switched_from=lang_ctx.switched_from,
                request_id=partial.request_id,
            )

        partial.language = lang_ctx
        return partial


def _classify_time_of_day(
    hour: int,
) -> Literal["morning", "afternoon", "evening", "night"]:
    """Classify the hour into a time-of-day bucket.

    Args:
        hour: Hour in 24h format (0-23).

    Returns:
        Time of day classification.
    """
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


_WEEKDAY_NAMES_EN = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_WEEKDAY_NAMES_DE = [
    "Montag",
    "Dienstag",
    "Mittwoch",
    "Donnerstag",
    "Freitag",
    "Samstag",
    "Sonntag",
]


def _get_weekday_name(weekday: int, lang: str) -> str:
    """Get localized weekday name.

    Args:
        weekday: 0=Monday, 6=Sunday.
        lang: Language code.

    Returns:
        Weekday name string.
    """
    if lang == "de":
        return _WEEKDAY_NAMES_DE[weekday]
    return _WEEKDAY_NAMES_EN[weekday]


class TimeResolver(BaseResolver):
    """Resolves time context for the current request.

    Phase 0: uses UTC (no user timezone detection yet).
    The time context enables time-aware prompts and scheduling.
    """

    async def resolve(
        self, partial: PartialExecutionContext
    ) -> PartialExecutionContext:
        """Fill time context with current UTC time.

        Args:
            partial: Partial context (language should be resolved first).

        Returns:
            Partial context with time field populated.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()

        # Use resolved language for weekday name, fallback to "en"
        lang = "de"
        if partial.language is not None:
            lang = partial.language.code

        partial.time = TimeContext(
            now_utc=now,
            now_local=now,  # Phase 0: no timezone conversion yet
            weekday=weekday,
            weekday_name=_get_weekday_name(weekday, lang),
            time_of_day=_classify_time_of_day(now.hour),
            timezone_name="UTC",
        )
        return partial


class ChannelResolver(BaseResolver):
    """Resolves channel capabilities based on the channel identifier.

    Phase 0: only Telegram is supported. Future phases will add
    Desktop, Mini-App, CLI with different capability profiles.
    """

    # Channel capability profiles
    _PROFILES: dict[str, ChannelCapabilities] = {
        "telegram": ChannelCapabilities(
            streaming_supported=True,
            max_message_length=4096,
            markdown_supported=True,
            inline_buttons_supported=True,
        ),
        "desktop": ChannelCapabilities(
            streaming_supported=True,
            max_message_length=100_000,
            markdown_supported=True,
            inline_buttons_supported=False,
        ),
        "cli": ChannelCapabilities(
            streaming_supported=True,
            max_message_length=1_000_000,
            markdown_supported=False,
            inline_buttons_supported=False,
        ),
    }

    async def resolve(
        self, partial: PartialExecutionContext
    ) -> PartialExecutionContext:
        """Set channel capabilities based on channel type.

        Args:
            partial: Partial context with channel field set.

        Returns:
            Partial context with channel_capabilities populated.
        """
        partial.channel_capabilities = self._PROFILES.get(
            partial.channel,
            ChannelCapabilities(),
        )
        return partial
