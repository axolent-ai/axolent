"""Claude Persistent Provider: nutzt den Process-Pool fuer warme Subprocesses.

Modus-B-konform: Kein OAuth-Token-Lesen, kein eigener HTTP-Client.
Nutzt persistent stdin-Pipe zu `claude --print --input-format stream-json`.
User hat eigene Pro/Max-Subscription.

Vorteile gegenueber claude_cli.py (Legacy-Provider):
    - 74% schnellere Antworten (kein Cold-Start bei warmen Pipes)
    - Streaming-Support (Token fuer Token)
    - Process-Reuse spart OS-Overhead

Der Legacy-Provider (claude_cli.py) bleibt als Fallback fuer:
    - Crash-Recovery wenn persistent Pipe fehlschlaegt
    - Situationen in denen der Process-Pool nicht laeuft
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


class ClaudePersistentProvider(LLMProvider):
    """Claude Code CLI als LLM-Provider mit persistentem Subprocess (Modus B).

    Nutzt ClaudeProcessPool fuer warme Subprocesses pro User.
    Streaming-faehig: query_streaming() liefert StreamEvents.
    Fallback: query() sammelt alle Events und gibt ProviderResponse zurueck.
    """

    name = "claude_persistent"

    def __init__(self, process_pool: ClaudeProcessPool) -> None:
        self._pool = process_pool

    def get_capabilities(self) -> ProviderCapabilities:
        """Claude-Capabilities: Cloud, Subscription, 200k Context, Streaming."""
        return _CAPABILITIES

    def is_available(self) -> bool:
        """Prueft ob `claude` CLI im PATH ist."""
        return ClaudeProcessPool.is_cli_available()

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
        model: str | None = None,
        chat_id: int = 0,
    ) -> ProviderResponse:
        """Sendet eine Anfrage via persistent Subprocess und wartet auf vollstaendige Antwort.

        Non-Streaming-Variante: sammelt alle Events bis zum Result.
        Fuer Streaming: query_streaming() verwenden.

        Args:
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt.
            timeout_seconds: Timeout (wird vom Process-Pool intern gehandled).
            model: Optionaler Modell-Identifier (aktuell ignoriert).
            chat_id: Telegram-Chat-ID als Routing-Key fuer Process-Isolation.

        Returns:
            ProviderResponse mit Claude-Antwort oder Fehler.
        """
        start = time.monotonic()

        try:
            full_text = ""
            async for event in self._pool.send_message(
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
        chat_id: int = 0,
    ) -> AsyncIterator[StreamEvent]:
        """Sendet eine Anfrage und streamt die Antwort Token fuer Token.

        Args:
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt.
            chat_id: Telegram-Chat-ID als Routing-Key.

        Yields:
            StreamEvent-Objekte (content_delta, result, error).
        """
        async for event in self._pool.send_message(
            chat_id=chat_id,
            prompt=prompt,
            system_prompt=system_prompt,
        ):
            yield event
