"""OpenAI Codex CLI Provider (Stub).

Geplant für Phase 1+. Nutzt das `codex` CLI-Binary als Subprozess.
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
    supports_vision=True,
    max_context_tokens=128_000,
    cost_class="subscription",
    privacy_class="cloud",
    available_models=["gpt-5", "gpt-5-mini"],
)


class OpenAICodexProvider(LLMProvider):
    """OpenAI via Codex CLI als LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User codex CLI installiert hat
    2. User mit eigenem OpenAI-Account eingeloggt ist
    """

    name = "openai"

    def get_capabilities(self) -> ProviderCapabilities:
        """OpenAI-Capabilities: Cloud, Subscription, 128k Context."""
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
