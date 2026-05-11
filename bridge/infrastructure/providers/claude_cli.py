"""Claude Provider via Claude Code CLI Subprozess (Modus B).

WICHTIG: Modus-B-konform. Kein OAuth-Token-Lesen, kein eigener HTTP-Client.
Nur subprocess-Aufruf von `claude -p`. User hat eigene Pro/Max-Subscription.

Dieses Modul ersetzt den alten infrastructure/claude_cli.py und implementiert
das LLMProvider-Interface.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time

from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderResponse,
)

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: int = 120

_CAPABILITIES = ProviderCapabilities(
    supports_streaming=False,
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


class ClaudeProvider(LLMProvider):
    """Claude Code CLI als LLM-Provider (Modus B).

    Ruft `claude -p` als async Subprozess auf.
    Keine direkte API-Kommunikation, kein Token-Handling.
    """

    name = "claude"

    def get_capabilities(self) -> ProviderCapabilities:
        """Claude-Capabilities: Cloud, Subscription, 200k Context."""
        return _CAPABILITIES

    def is_available(self) -> bool:
        """Prüft ob `claude` CLI im PATH ist."""
        return shutil.which("claude") is not None

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        model: str | None = None,
        user_id: int | None = None,
        chat_id: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        """Ruft Claude Code CLI auf und liefert die Antwort.

        Args:
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt (via stdin, nicht argv).
            timeout_seconds: Timeout für den Subprozess.
            model: Optionaler Modell-Identifier (aktuell ignoriert, CLI nutzt Default).
            user_id: Telegram User-ID (akzeptiert fuer Interface-Kompatibilitaet, ignoriert).
            chat_id: Telegram Chat-ID (akzeptiert fuer Interface-Kompatibilitaet, ignoriert).
            **kwargs: Safety-Net fuer zukuenftige Provider-Interface-Erweiterungen.

        Returns:
            ProviderResponse mit Claude-Antwort oder Fehler.
        """
        start = time.monotonic()
        cmd: list[str] = ["claude", "-p"]

        # Privacy: kompletten Prompt via stdin senden, nicht als argv.
        # --append-system-prompt würde Memory-Inhalte in Prozesslisten zeigen.
        if system_prompt:
            combined = f"{system_prompt}\n\n---\n\nUser: {prompt}"
        else:
            combined = prompt

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=combined.encode("utf-8")),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            duration = time.monotonic() - start
            log.warning("Claude CLI Timeout nach %.1fs", duration)
            return ProviderResponse(
                text="",
                duration_seconds=duration,
                provider_name=self.name,
                error=f"timeout_after_{timeout_seconds}s",
            )

        duration = time.monotonic() - start
        stdout = stdout_bytes.decode("utf-8", "replace")
        stderr = stderr_bytes.decode("utf-8", "replace")

        if proc.returncode != 0:
            log.error(
                "Claude CLI exit code %d. stderr: %s",
                proc.returncode,
                stderr[:500],
            )
            return ProviderResponse(
                text="",
                duration_seconds=duration,
                provider_name=self.name,
                error=f"exit_code_{proc.returncode}: {stderr[:200]}",
            )

        return ProviderResponse(
            text=stdout.strip(),
            duration_seconds=duration,
            provider_name=self.name,
        )
