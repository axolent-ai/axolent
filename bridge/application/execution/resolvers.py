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
import os
from datetime import datetime, timezone
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from application.execution.context import (
    ChannelCapabilities,
    PartialExecutionContext,
    TimeContext,
)
from application.language_resolver import LanguageResolver

log = logging.getLogger(__name__)


def _resolve_timezone() -> tuple[ZoneInfo, str]:
    """Resolve timezone from env var AXOLENT_TIMEZONE or system default.

    Falls back to Europe/Berlin if both fail (bot is currently
    single-tenant central europe).
    """
    tz_name = os.environ.get("AXOLENT_TIMEZONE")
    if tz_name:
        try:
            return ZoneInfo(tz_name), tz_name
        except Exception:  # nosec B110 - intentional silent fallback to system tz
            pass

    # Fallback: try to detect system local timezone
    try:
        local_now = datetime.now().astimezone()
        sys_tz = local_now.tzinfo
        if sys_tz is not None:
            sys_name = str(sys_tz)
            # Try as IANA name; if fails, fall through
            try:
                return ZoneInfo(sys_name), sys_name
            except Exception:  # nosec B110 - silent fallback to Europe/Berlin
                pass
    except Exception:  # nosec B110 - silent fallback to Europe/Berlin
        pass

    # Final fallback
    return ZoneInfo("Europe/Berlin"), "Europe/Berlin"


class BaseResolver(abc.ABC):
    """Abstract base for all context resolvers.

    Each resolver takes a PartialExecutionContext, enriches it,
    and returns it.

    ARCHITECTURE RULE: Resolvers MUST NOT perform state-mutating I/O.
    Reading from storage is acceptable, but writing (e.g. persisting
    sticky language) violates resolver purity. If a resolver needs to
    persist state, that persistence must happen AFTER the request has
    been allowed through preflight policies (rate-limit, auth, etc.).

    The LanguageResolverAdapter delegates to LanguageResolver.resolve()
    which DOES persist. This is acceptable because the resolver pipeline
    only runs AFTER the rate-limit check. For pre-rate-limit UI, use
    LanguageResolver.resolve_readonly() instead.
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

        # Adopt the LanguageResolver's request_id or use ours.
        # Uses with_request_id() to preserve ALL fields including Phase 2
        # metadata (detection_distribution, reliability_score, etc.).
        if lang_ctx.request_id != partial.request_id:
            lang_ctx = lang_ctx.with_request_id(partial.request_id)

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

    Uses AXOLENT_TIMEZONE env var, system local timezone, or
    Europe/Berlin fallback. The time context enables time-aware
    prompts and scheduling.
    """

    async def resolve(
        self, partial: PartialExecutionContext
    ) -> PartialExecutionContext:
        """Fill time context with current local and UTC time.

        Args:
            partial: Partial context (language should be resolved first).

        Returns:
            Partial context with time field populated.
        """
        now_utc = datetime.now(timezone.utc)
        tz_info, tz_name = _resolve_timezone()
        now_local = now_utc.astimezone(tz_info)

        lang = "de"
        if partial.language is not None:
            lang = partial.language.code

        partial.time = TimeContext(
            now_utc=now_utc,
            now_local=now_local,
            weekday=now_local.weekday(),
            weekday_name=_get_weekday_name(now_local.weekday(), lang),
            time_of_day=_classify_time_of_day(now_local.hour),
            timezone_name=tz_name,
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
