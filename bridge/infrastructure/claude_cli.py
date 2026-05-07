"""Claude CLI Adapter: LEGACY-Kompatibilitäts-Wrapper.

VERALTET: Neuer Code soll infrastructure.providers.claude_cli.ClaudeProvider nutzen.
Dieses Modul bleibt für Backward-Compatibility (Tests, alte Imports).

Kapselt den Aufruf von `claude -p` als async Subprozess.
Kein Telegram-Code, keine Presentation-Logik.
"""

from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger(__name__)

CLAUDE_TIMEOUT_SECONDS: int = 120


async def call_claude_async(
    prompt: str,
    system_prompt: str = "",
) -> tuple[int, str, str, float]:
    """Ruft Claude Code CLI nativ async auf, blockiert keinen Thread.

    LEGACY: Wird von neuem Code nicht mehr direkt aufgerufen.
    ChatService nutzt jetzt ProviderRouter -> ClaudeProvider.

    Args:
        prompt: User-Nachricht die an Claude weitergegeben wird.
        system_prompt: Effektiver System-Prompt (bereits mit Language-Override).

    Returns:
        Tuple aus (exit_code, stdout, stderr, duration_seconds).

    Raises:
        FileNotFoundError: Wenn `claude` nicht im PATH gefunden wird.
        asyncio.TimeoutError: Wenn Claude länger als CLAUDE_TIMEOUT_SECONDS braucht.
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
            timeout=CLAUDE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    duration = time.monotonic() - start
    return (
        proc.returncode or 0,
        stdout_bytes.decode("utf-8", "replace"),
        stderr_bytes.decode("utf-8", "replace"),
        duration,
    )
