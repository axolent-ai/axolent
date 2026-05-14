"""Claude Persistent Provider: uses the process pool for warm subprocesses.

Mode B compliant: no OAuth token reading, no custom HTTP client.
Uses persistent stdin pipe to `claude --print --input-format stream-json`.
User has their own Pro/Max subscription.

Advantages over claude_cli.py (legacy provider):
    * 74% faster responses (no cold start with warm pipes)
    * Streaming support (token by token)
    * Process reuse saves OS overhead

The legacy provider (claude_cli.py) remains as fallback for:
    * Crash recovery when persistent pipe fails
    * Situations where the process pool is not running
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from infrastructure.claude_process_pool import (
    ClaudeProcessPool,
    StreamEvent,
)
from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderResponse,
    StreamingProvider,
)

log = logging.getLogger(__name__)

_CAPABILITIES = ProviderCapabilities(
    supports_streaming=True,
    supports_tool_use=True,
    supports_vision=True,
    max_context_tokens=200_000,
    cost_class="subscription",
    privacy_class="cloud",
    available_models=[
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5-20251001",
    ],
)


class ClaudePersistentProvider(LLMProvider, StreamingProvider):
    """Claude Code CLI as LLM provider with persistent subprocess (Mode B).

    Uses ClaudeProcessPool for warm subprocesses per user.
    Streaming capable: query_streaming() delivers StreamEvents.
    Fallback: query() collects all events and returns ProviderResponse.
    """

    name = "claude_persistent"

    def __init__(self, process_pool: ClaudeProcessPool) -> None:
        self._pool = process_pool

    def get_capabilities(self) -> ProviderCapabilities:
        """Claude capabilities: cloud, subscription, 200k context, streaming."""
        return _CAPABILITIES

    def is_available(self) -> bool:
        """Check if `claude` CLI is in PATH."""
        return ClaudeProcessPool.is_cli_available()

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
        model: str | None = None,
        user_id: int | None = None,
        chat_id: int | None = None,
    ) -> ProviderResponse:
        """Send a request via persistent subprocess and wait for complete response.

        Non-streaming variant: collects all events until result.
        For streaming: use query_streaming().

        Args:
            prompt: User message.
            system_prompt: Optional system prompt.
            timeout_seconds: Timeout (handled internally by the process pool).
            model: Optional model identifier (passed through to the process pool).
            user_id: Telegram user ID (required).
            chat_id: Telegram chat ID (required).

        Returns:
            ProviderResponse with Claude response or error.

        Raises:
            ValueError: If user_id or chat_id is not provided.
        """
        if user_id is None:
            raise ValueError("user_id is required for ClaudePersistentProvider")
        if chat_id is None:
            raise ValueError("chat_id is required for ClaudePersistentProvider")

        start = time.monotonic()

        try:
            full_text = ""
            async for event in self._pool.send_message(
                user_id=user_id,
                chat_id=chat_id,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
            ):
                if event.event_type == "result":
                    full_text = event.full_text
                    break
                elif event.event_type == "error":
                    duration = time.monotonic() - start
                    return ProviderResponse(
                        text="",
                        duration_seconds=duration,
                        provider_name=self.name,
                        error=f"subprocess_error: {event.text}",
                    )

            duration = time.monotonic() - start

            if not full_text:
                return ProviderResponse(
                    text="",
                    duration_seconds=duration,
                    provider_name=self.name,
                    error="empty_response",
                )

            return ProviderResponse(
                text=full_text.strip(),
                duration_seconds=duration,
                provider_name=self.name,
            )

        except RuntimeError as e:
            duration = time.monotonic() - start
            log.error("ClaudePersistentProvider error: %s", e)
            return ProviderResponse(
                text="",
                duration_seconds=duration,
                provider_name=self.name,
                error=f"runtime_error: {e}",
            )

    async def query_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        user_id: int | None = None,
        chat_id: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send a request and stream the response token by token.

        Args:
            prompt: User message.
            system_prompt: Optional system prompt.
            user_id: Telegram user ID (required).
            chat_id: Telegram chat ID (required).
            model: Optional model ID (None = pool default).

        Yields:
            StreamEvent objects (content_delta, result, error).

        Raises:
            ValueError: If user_id or chat_id is not provided.
        """
        if user_id is None:
            raise ValueError("user_id is required for ClaudePersistentProvider")
        if chat_id is None:
            raise ValueError("chat_id is required for ClaudePersistentProvider")

        async for event in self._pool.send_message(
            user_id=user_id,
            chat_id=chat_id,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
        ):
            yield event
