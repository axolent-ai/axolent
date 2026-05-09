"""Claude Persistent Provider: nutzt den Process-Pool für warme Subprocesses.

Modus-B-konform: Kein OAuth-Token-Lesen, kein eigener HTTP-Client.
Nutzt persistent stdin-Pipe zu `claude --print --input-format stream-json`.
User hat eigene Pro/Max-Subscription.

Vorteile gegenüber claude_cli.py (Legacy-Provider):
    - 74% schnellere Antworten (kein Cold-Start bei warmen Pipes)
    - Streaming-Support (Token für Token)
    - Process-Reuse spart OS-Overhead

Der Legacy-Provider (claude_cli.py) bleibt als Fallback für:
    - Crash-Recovery wenn persistent Pipe fehlschlägt
    - Situationen in denen der Process-Pool nicht läuft
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
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-3-5-20241022",
    ],
)


class ClaudePersistentProvider(LLMProvider, StreamingProvider):
    """Claude Code CLI als LLM-Provider mit persistentem Subprocess (Modus B).

    Nutzt ClaudeProcessPool für warme Subprocesses pro User.
    Streaming-fähig: query_streaming() liefert StreamEvents.
    Fallback: query() sammelt alle Events und gibt ProviderResponse zurück.
    """

    name = "claude_persistent"

    def __init__(self, process_pool: ClaudeProcessPool) -> None:
        self._pool = process_pool

    def get_capabilities(self) -> ProviderCapabilities:
        """Claude-Capabilities: Cloud, Subscription, 200k Context, Streaming."""
        return _CAPABILITIES

    def is_available(self) -> bool:
        """Prüft ob `claude` CLI im PATH ist."""
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
        """Sendet eine Anfrage via persistent Subprocess und wartet auf vollständige Antwort.

        Non-Streaming-Variante: sammelt alle Events bis zum Result.
        Für Streaming: query_streaming() verwenden.

        Args:
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt.
            timeout_seconds: Timeout (wird vom Process-Pool intern gehandled).
            model: Optionaler Modell-Identifier (aktuell ignoriert).
            user_id: Telegram-User-ID (Pflicht).
            chat_id: Telegram-Chat-ID (Pflicht).

        Returns:
            ProviderResponse mit Claude-Antwort oder Fehler.

        Raises:
            ValueError: Wenn user_id oder chat_id nicht angegeben.
        """
        if user_id is None:
            raise ValueError("user_id ist Pflicht für ClaudePersistentProvider")
        if chat_id is None:
            raise ValueError("chat_id ist Pflicht für ClaudePersistentProvider")

        start = time.monotonic()

        try:
            full_text = ""
            async for event in self._pool.send_message(
                user_id=user_id,
                chat_id=chat_id,
                prompt=prompt,
                system_prompt=system_prompt,
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
            log.error("ClaudePersistentProvider Fehler: %s", e)
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
    ) -> AsyncIterator[StreamEvent]:
        """Sendet eine Anfrage und streamt die Antwort Token für Token.

        Args:
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt.
            user_id: Telegram-User-ID (Pflicht).
            chat_id: Telegram-Chat-ID (Pflicht).

        Yields:
            StreamEvent-Objekte (content_delta, result, error).

        Raises:
            ValueError: Wenn user_id oder chat_id nicht angegeben.
        """
        if user_id is None:
            raise ValueError("user_id ist Pflicht für ClaudePersistentProvider")
        if chat_id is None:
            raise ValueError("chat_id ist Pflicht für ClaudePersistentProvider")

        async for event in self._pool.send_message(
            user_id=user_id,
            chat_id=chat_id,
            prompt=prompt,
            system_prompt=system_prompt,
        ):
            yield event
