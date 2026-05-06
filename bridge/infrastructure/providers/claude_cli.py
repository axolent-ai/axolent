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

from infrastructure.providers.base import LLMProvider, ProviderResponse

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: int = 120


class ClaudeProvider(LLMProvider):
    """Claude Code CLI als LLM-Provider (Modus B).

    Ruft `claude -p` als async Subprozess auf.
    Keine direkte API-Kommunikation, kein Token-Handling.
    """

    name = "claude"

    def is_available(self) -> bool:
        """Prueft ob `claude` CLI im PATH ist."""
        return shutil.which("claude") is not None

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> ProviderResponse:
        """Ruft Claude Code CLI auf und liefert die Antwort.

        Args:
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt (via --append-system-prompt).
            timeout_seconds: Timeout fuer den Subprozess.

        Returns:
            ProviderResponse mit Claude-Antwort oder Fehler.
        """
        start = time.monotonic()
        cmd: list[str] = ["claude", "-p"]

        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
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
