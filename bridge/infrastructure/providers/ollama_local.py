"""Ollama Local Provider (Stub).

Geplant für Phase 1+. Nutzt `ollama` CLI oder HTTP-API (localhost:11434).
Aktuell: nur Skelett, query() raised ProviderNotImplemented.
"""

from __future__ import annotations

from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderNotImplemented,
    ProviderResponse,
)

_CAPABILITIES = ProviderCapabilities(
    supports_streaming=True,
    supports_tool_use=False,
    supports_vision=False,
    max_context_tokens=128_000,
    cost_class="free",
    privacy_class="local",
    available_models=["llama-3.3", "qwen-3"],
)


class OllamaProvider(LLMProvider):
    """Ollama als lokaler LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User Ollama installiert hat
    2. Mindestens ein Modell gepullt ist (ollama list)
    """

    name = "ollama"

    def get_capabilities(self) -> ProviderCapabilities:
        """Ollama-Capabilities: Local, Free, 128k Context."""
        return _CAPABILITIES

    def is_available(self) -> bool:
        """Stub: immer False bis zur echten Implementierung. Prüfe Verfügbarkeit."""
        return False

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
        model: str | None = None,
    ) -> ProviderResponse:
        """Noch nicht implementiert. Stub für Phase 1+."""
        raise ProviderNotImplemented(self.name)
