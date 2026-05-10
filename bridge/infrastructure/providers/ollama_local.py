"""Ollama Local Provider.

Nutzt Ollama HTTP-API (localhost:11434) fuer lokale LLM-Inference.
Modus-B-konform: Ollama laeuft lokal, keine Cloud-API, kein Token-Hijacking.

Default-Modell konfigurierbar via OLLAMA_MODEL env-var (Default: llama3.2:3b).
Host konfigurierbar via OLLAMA_HOST env-var (Default: http://localhost:11434).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request

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
    """Gibt den konfigurierten Ollama-Host zurueck.

    Validiert das URL-Scheme: nur http:// und https:// sind erlaubt.
    Verhindert file:// oder andere potentiell gefaehrliche Schemes.
    """
    host = os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    if not host.startswith(("http://", "https://")):
        log.warning(
            "OLLAMA_HOST '%s' hat ungültiges Scheme, verwende Default: %s",
            host,
            DEFAULT_HOST,
        )
        return DEFAULT_HOST
    return host


def _get_default_model() -> str:
    """Gibt das konfigurierte Default-Modell zurueck."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


class OllamaProvider(LLMProvider):
    """Ollama als lokaler LLM-Provider.

    Kommuniziert mit der Ollama HTTP-API auf localhost.
    Privacy-Class: local (keine Daten verlassen den Rechner).
    """

    name = "ollama_local"

    def get_capabilities(self) -> ProviderCapabilities:
        """Ollama-Capabilities: Local, Free, 8k Context (modellabhaengig)."""
        return _CAPABILITIES

    def is_available(self) -> bool:
        """Prueft ob Ollama auf dem konfigurierten Host erreichbar ist.

        Macht einen HTTP-GET auf /api/tags mit kurzem Timeout (2s).
        True wenn 200 OK, False bei Connection-Error oder Timeout.
        """
        try:
            host = _get_host()
            url = f"{host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urllib.request.urlopen(req, timeout=2) as response:  # nosec B310
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
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
        """Fragt Ollama via HTTP-API (non-streaming).

        Args:
            prompt: User-Nachricht / Prompt.
            system_prompt: Optionaler System-Prompt.
            timeout_seconds: Timeout fuer den Request.
            model: Modell-Identifier (None = OLLAMA_MODEL env oder llama3.2:3b).
            user_id: Akzeptiert fuer Interface-Kompatibilitaet, ignoriert.
            chat_id: Akzeptiert fuer Interface-Kompatibilitaet, ignoriert.
            **kwargs: Safety-Net fuer zukuenftige Provider-Interface-Erweiterungen.

        Returns:
            ProviderResponse mit Ollama-Antwort oder Fehler.
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
            log.warning("Ollama Timeout nach %.1fs", duration)
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
        """Synchroner HTTP-POST an Ollama /api/generate.

        Wird via asyncio.to_thread aufgerufen um den Event-Loop nicht zu blockieren.

        Returns:
            Der generierte Text aus dem response-Feld.

        Raises:
            RuntimeError: Bei HTTP-Fehlern oder unerwartetem Response-Format.
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
            raise RuntimeError(f"Ollama nicht erreichbar: {exc.reason}") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama ungueltige JSON-Antwort: {body[:200]}") from exc

        if "response" not in result:
            raise RuntimeError(
                f"Ollama response-Feld fehlt. Keys: {list(result.keys())}"
            )

        return result["response"].strip()
