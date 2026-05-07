"""Google Gemini CLI Provider (Stub).

Geplant für Phase 1+. Nutzt das `gemini` CLI-Binary als Subprozess.
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
    max_context_tokens=1_000_000,
    cost_class="free",
    privacy_class="cloud",
    available_models=["gemini-2.5-flash", "gemini-3.1-pro"],
)


class GeminiProvider(LLMProvider):
    """Google Gemini via CLI als LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User gemini CLI installiert hat
    2. User mit eigenem Google-Account eingeloggt ist
    """

    name = "gemini"

    def get_capabilities(self) -> ProviderCapabilities:
        """Gemini-Capabilities: Cloud, Free, 1M Context."""
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
