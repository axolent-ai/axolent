"""ExecutionContext: the single source of truth per request.

Contains all resolved facts needed by downstream components
(InstructionCompiler, ProviderMesh, StatusSession, TextGuard).
No component may resolve context independently once this exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Optional

from application.language_resolver import LanguageContext

if TYPE_CHECKING:
    from application.execution.envelope import RequestEnvelope


@dataclass(frozen=True, slots=True)
class TimeContext:
    """Resolved time context for the current request.

    Provides localized time info without downstream components
    needing to call datetime functions themselves.

    Attributes:
        now_utc: Current UTC timestamp.
        now_local: Local timestamp (user timezone, defaults to UTC).
        weekday: Weekday as integer (0=Monday, 6=Sunday).
        weekday_name: Localized weekday name.
        time_of_day: Classification of current time.
        timezone_name: IANA timezone name or "UTC".
    """

    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    now_local: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    weekday: int = 0
    weekday_name: str = "Monday"
    time_of_day: Literal["morning", "afternoon", "evening", "night"] = "morning"
    timezone_name: str = "UTC"


@dataclass(frozen=True, slots=True)
class ChannelCapabilities:
    """Capabilities of the requesting channel.

    Used by the InstructionCompiler and ResponseRenderer
    to adapt output format.

    Attributes:
        streaming_supported: Whether the channel supports streaming responses.
        max_message_length: Maximum characters per message (Telegram: 4096).
        markdown_supported: Whether Markdown formatting is supported.
        inline_buttons_supported: Whether inline buttons are available.
    """

    streaming_supported: bool = True
    max_message_length: int = 4096
    markdown_supported: bool = True
    inline_buttons_supported: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """The central truth for a single request lifecycle.

    Once created by ContextKernel.build(), this object is immutable.
    All downstream components receive this rather than resolving
    their own context.

    Attributes:
        request_id: Unique request identifier (from envelope).
        user_id: Numeric user ID.
        chat_id: Numeric chat/conversation ID.
        channel: Origin channel.
        language: Resolved language context (from LanguageResolver).
        time: Resolved time context.
        channel_capabilities: Channel feature flags.
        audit_tags: Extensible metadata for future phases.
    """

    request_id: str = ""
    user_id: int = 0
    chat_id: int = 0
    channel: str = "telegram"
    language: LanguageContext = field(
        default_factory=lambda: LanguageContext(
            code="de",
            source="default",
            confidence=1.0,
            switched_from=None,
            request_id="",
        )
    )
    time: TimeContext = field(default_factory=TimeContext)
    channel_capabilities: ChannelCapabilities = field(
        default_factory=ChannelCapabilities
    )
    audit_tags: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def get_audit_tag(self, key: str, default: Any = None) -> Any:
        """Look up a single audit tag by key.

        Args:
            key: The tag key to look up.
            default: Value to return if key is not found.

        Returns:
            The tag value, or default.
        """
        for k, v in self.audit_tags:
            if k == key:
                return v
        return default

    def as_audit_dict(self) -> dict[str, Any]:
        """Convert audit_tags to a plain dict for serialization.

        Returns:
            Dict representation of audit tags.
        """
        return dict(self.audit_tags)


@dataclass(slots=True)
class PartialExecutionContext:
    """Mutable builder for ExecutionContext.

    Used by the resolver pipeline: each resolver fills its part,
    then freeze() creates the immutable ExecutionContext.
    """

    request_id: str = ""
    user_id: int = 0
    chat_id: int = 0
    channel: str = "telegram"
    language: Optional[LanguageContext] = None
    time: Optional[TimeContext] = None
    channel_capabilities: Optional[ChannelCapabilities] = None
    audit_tags: dict[str, Any] = field(default_factory=dict)
    # Raw text preserved for resolvers that need it
    raw_text: str = ""
    # Override hint (e.g. from /lang command)
    language_override: Optional[str] = None

    @classmethod
    def from_envelope(
        cls,
        envelope: "RequestEnvelope",
        language_override: Optional[str] = None,
    ) -> "PartialExecutionContext":
        """Create a partial context from a RequestEnvelope.

        Args:
            envelope: The incoming request envelope.
            language_override: Optional explicit language override.

        Returns:
            Mutable PartialExecutionContext ready for resolver pipeline.
        """

        return cls(
            request_id=envelope.request_id,
            user_id=envelope.user_id,
            chat_id=envelope.chat_id,
            channel=envelope.channel,
            raw_text=envelope.raw_text,
            language_override=language_override,
        )

    def freeze(self) -> ExecutionContext:
        """Convert to an immutable ExecutionContext.

        Missing sub-contexts get sensible defaults.

        Returns:
            Frozen ExecutionContext instance.
        """
        lang = self.language or LanguageContext(
            code="de",
            source="default",
            confidence=1.0,
            switched_from=None,
            request_id=self.request_id,
        )
        return ExecutionContext(
            request_id=self.request_id,
            user_id=self.user_id,
            chat_id=self.chat_id,
            channel=self.channel,
            language=lang,
            time=self.time or TimeContext(),
            channel_capabilities=self.channel_capabilities or ChannelCapabilities(),
            audit_tags=tuple(self.audit_tags.items()),
        )
