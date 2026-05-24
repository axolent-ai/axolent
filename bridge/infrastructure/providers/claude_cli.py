"""Claude provider via Claude Code CLI subprocess (Mode B).

IMPORTANT: Mode B compliant. No OAuth token reading, no custom HTTP client.
Only subprocess call to `claude -p`. User has their own Pro/Max subscription.

This module replaces the old infrastructure/claude_cli.py and implements
the LLMProvider interface.

GAP-11 FIX: Subprocess env is scrubbed via allowlist.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time

from infrastructure.security.env_scrubber import build_scrubbed_env
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
    """Claude Code CLI as LLM provider (Mode B).

    Calls `claude -p` as async subprocess.
    No direct API communication, no token handling.
    """

    name = "claude"

    def get_capabilities(self) -> ProviderCapabilities:
        """Claude capabilities: cloud, subscription, 200k context."""
        return _CAPABILITIES

    def is_available(self) -> bool:
        """Check if `claude` CLI is in PATH."""
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
        """Call Claude Code CLI and return the response.

        Args:
            prompt: User message.
            system_prompt: Optional system prompt (via stdin, not argv).
            timeout_seconds: Timeout for the subprocess.
            model: Optional model identifier (passed as --model to CLI).
            user_id: Telegram user ID (accepted for interface compatibility, ignored).
            chat_id: Telegram chat ID (accepted for interface compatibility, ignored).
            **kwargs: Safety net for future provider interface extensions.

        Returns:
            ProviderResponse with Claude response or error.
        """
        start = time.monotonic()
        cmd: list[str] = ["claude", "-p"]
        if model is not None:
            cmd.extend(["--model", model])

        # Privacy: send complete prompt via stdin, not as argv.
        # --append-system-prompt would expose memory contents in process lists.
        if system_prompt:
            combined = f"{system_prompt}\n\n---\n\nUser: {prompt}"
        else:
            combined = prompt

        # GAP-11 FIX: Use scrubbed env (allowlist only) to prevent
        # TELEGRAM_BOT_TOKEN, SENTRY_DSN, and other secrets from leaking.
        scrubbed_env = build_scrubbed_env()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=scrubbed_env,
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
            log.warning("Claude CLI timeout after %.1fs", duration)
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
