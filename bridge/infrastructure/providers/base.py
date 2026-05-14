"""Provider interface for LLM integrations.

Every provider must implement query(), is_available(), and get_capabilities().
Mode B requirement: Claude provider uses subprocess.
Other providers may use API keys, but never hijack OAuth tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass(frozen=True)
class ProviderCapabilities:
    """Describes what a provider can do.

    Used by the router to route requests appropriately.

    Attributes:
        supports_streaming: Can the provider stream responses?
        supports_tool_use: Can the provider do function calling / tool use?
        supports_vision: Can the provider process images?
        max_context_tokens: Maximum context length in tokens.
        max_memory_chars: Max memory chars in system prompt (Phase 1+, per provider).
        cost_class: Cost class ("free", "subscription", "pay_per_use").
        privacy_class: Where is data processed? ("cloud", "local").
        available_models: List of available model IDs.
    """

    supports_streaming: bool = False
    supports_tool_use: bool = False
    supports_vision: bool = False
    max_context_tokens: int = 32_000
    max_memory_chars: int = 4000
    cost_class: str = "free"  # "free", "subscription", "pay_per_use"
    privacy_class: str = "cloud"  # "cloud", "local"
    available_models: list[str] = field(default_factory=list)


@dataclass
class ProviderResponse:
    """Standardized response from an LLM provider.

    Attributes:
        text: The generated response (empty on error).
        duration_seconds: Duration of the call in seconds.
        provider_name: Name of the provider that responded.
        model: Optional model identifier (e.g. "claude-sonnet-4-6").
        error: Error description if the call failed.
    """

    text: str
    duration_seconds: float
    provider_name: str
    model: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """True if no error occurred and text is present."""
        return self.error is None and bool(self.text)


# ---------------------------------------------------------------------------
# Provider error hierarchy
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class for all provider errors.

    Attributes:
        provider_name: Name of the provider that raised the error.
        retryable: True if a retry makes sense (e.g. timeout).
    """

    def __init__(self, provider_name: str, retryable: bool, message: str) -> None:
        self.provider_name = provider_name
        self.retryable = retryable
        super().__init__(message)


class ProviderTimeout(ProviderError):
    """Provider took too long. Retry makes sense."""

    def __init__(self, provider_name: str, timeout_seconds: int) -> None:
        super().__init__(
            provider_name,
            retryable=True,
            message=f"Timeout after {timeout_seconds}s",
        )
        self.timeout_seconds = timeout_seconds


class ProviderUnavailable(ProviderError):
    """Provider is not available (CLI missing, no API key, etc.)."""

    def __init__(self, provider_name: str, reason: str) -> None:
        super().__init__(
            provider_name,
            retryable=False,
            message=f"{provider_name} unavailable: {reason}",
        )


class ProviderNotImplemented(ProviderError):
    """Stub provider, query() is not yet implemented."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(
            provider_name,
            retryable=False,
            message=f"{provider_name} is not yet implemented (Phase 1+)",
        )


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Each concrete provider registers with a unique `name`
    and implements query(), is_available(), and get_capabilities().
    """

    name: str  # "claude", "openai", "gemini", "mistral", "ollama_local"

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider."""
        ...

    @abstractmethod
    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
        model: str | None = None,
    ) -> ProviderResponse:
        """Send a request to the provider and return a response.

        Args:
            prompt: The user message / prompt.
            system_prompt: Optional system prompt.
            timeout_seconds: Maximum wait time before timeout.
            model: Optional model identifier (None = provider default).

        Returns:
            ProviderResponse with text or error.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available on this system.

        Checks e.g. whether the CLI binary is in PATH or an API key is set.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class StreamingProvider(ABC):
    """Mixin for providers with streaming support.

    Providers that support token streaming should implement this mixin
    in addition to LLMProvider. Enables type-safe isinstance() checks
    instead of fragile hasattr() checks.
    """

    @abstractmethod
    async def query_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        chat_id: int | None = None,
        user_id: int | None = None,
    ) -> "AsyncIterator":
        """Send a request and stream the response token by token.

        Args:
            prompt: User message.
            system_prompt: Optional system prompt.
            chat_id: Chat ID for process routing.
            user_id: User ID for process routing.

        Yields:
            StreamEvent objects (content_delta, result, error).
        """
        ...
