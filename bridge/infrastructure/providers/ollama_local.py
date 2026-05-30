"""Ollama local provider.

Uses Ollama HTTP API (localhost:11434) for local LLM inference.
Mode B compliant: Ollama runs locally, no cloud API, no token hijacking.

Default model configurable via OLLAMA_MODEL env var (default: llama3.2:3b).
Host configurable via OLLAMA_HOST env var (default: http://localhost:11434).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request

import httpx

from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderResponse,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL: str = "llama3.2:3b"
DEFAULT_HOST: str = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS: int = 60

_CAPABILITIES = ProviderCapabilities(
    supports_streaming=False,
    supports_tool_use=False,
    supports_vision=False,
    max_context_tokens=8192,
    max_memory_chars=4000,
    cost_class="free",
    privacy_class="local",
    available_models=["llama3.2:3b", "mistral:7b", "qwen2.5:7b", "phi3.5"],
)


def _get_host() -> str:
    """Return the configured Ollama host.

    Validates the URL scheme: only http:// and https:// are allowed.
    Prevents file:// or other potentially dangerous schemes.
    """
    host = os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    if not host.startswith(("http://", "https://")):
        log.warning(
            "OLLAMA_HOST '%s' has invalid scheme, using default: %s",
            host,
            DEFAULT_HOST,
        )
        return DEFAULT_HOST
    return host


def _get_default_model() -> str:
    """Return the configured default model."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


class OllamaProvider(LLMProvider):
    """Ollama as local LLM provider.

    Communicates with the Ollama HTTP API on localhost.
    Privacy class: local (no data leaves the machine).
    """

    name = "ollama_local"

    def get_capabilities(self) -> ProviderCapabilities:
        """Ollama capabilities: local, free, 8k context (model-dependent)."""
        return _CAPABILITIES

    async def is_available(self) -> bool:
        """Check if Ollama is reachable on the configured host.

        Makes an async HTTP GET to /api/tags with short timeout (2s).
        True on 200 OK, False on connection error or timeout.
        Non-blocking: uses httpx.AsyncClient to avoid event-loop stalls.
        """
        try:
            host = _get_host()
            url = f"{host}/api/tags"
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except (httpx.HTTPError, OSError, TimeoutError, ConnectionError):
            return False

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
        """Query Ollama via HTTP API (non-streaming).

        Args:
            prompt: User message / prompt.
            system_prompt: Optional system prompt.
            timeout_seconds: Timeout for the request.
            model: Model identifier (None = OLLAMA_MODEL env or llama3.2:3b).
            user_id: Accepted for interface compatibility, ignored.
            chat_id: Accepted for interface compatibility, ignored.
            **kwargs: Safety net for future provider interface extensions.

        Returns:
            ProviderResponse with Ollama response or error.
        """
        start = time.monotonic()
        actual_model = model or _get_default_model()
        host = _get_host()

        payload: dict = {
            "model": actual_model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response_text = await asyncio.wait_for(
                asyncio.to_thread(self._sync_query, host, payload, timeout_seconds),
                timeout=timeout_seconds + 5,
            )
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            log.warning("Ollama timeout after %.1fs", duration)
            return ProviderResponse(
                text="",
                duration_seconds=duration,
                provider_name=self.name,
                model=actual_model,
                error=f"timeout_after_{timeout_seconds}s",
            )
        except Exception as exc:
            duration = time.monotonic() - start
            log.error("Ollama query error: %s", exc)
            return ProviderResponse(
                text="",
                duration_seconds=duration,
                provider_name=self.name,
                model=actual_model,
                error=str(exc),
            )

        duration = time.monotonic() - start
        return ProviderResponse(
            text=response_text,
            duration_seconds=duration,
            provider_name=self.name,
            model=actual_model,
        )

    @staticmethod
    def _sync_query(host: str, payload: dict, timeout_seconds: int) -> str:
        """Synchronous HTTP POST to Ollama /api/generate.

        Called via asyncio.to_thread to avoid blocking the event loop.

        Returns:
            The generated text from the response field.

        Raises:
            RuntimeError: On HTTP errors or unexpected response format.
        """
        url = f"{host}/api/generate"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # nosec B310
                if resp.status != 200:
                    body = resp.read().decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"Ollama HTTP {resp.status}: {body}")
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama unreachable: {exc.reason}") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama invalid JSON response: {body[:200]}") from exc

        if "response" not in result:
            raise RuntimeError(
                f"Ollama response field missing. Keys: {list(result.keys())}"
            )

        return result["response"].strip()
