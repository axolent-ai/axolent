"""Mistral Vibe CLI Provider (Stub).

Geplant für Phase 1+. Nutzt das `mistral` CLI-Binary als Subprozess.
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
    supports_tool_use=True,
    supports_vision=False,
    max_context_tokens=128_000,
    cost_class="free",
    privacy_class="cloud",
    available_models=["mistral-large-3"],
)


class MistralVibeProvider(LLMProvider):
    """Mistral via Vibe CLI als LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User mistral/vibe CLI installiert hat
    2. User mit eigenem Mistral-Account eingeloggt ist
    """

    name = "mistral"

    def get_capabilities(self) -> ProviderCapabilities:
        """Mistral-Capabilities: Cloud, Free, 128k Context."""
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
