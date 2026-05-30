"""Mistral Vibe CLI provider (stub).

Planned for Phase 1+. Uses the `mistral` CLI binary as subprocess.
Currently: skeleton only, query() raises ProviderNotImplemented.
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
    """Mistral via Vibe CLI as LLM provider (stub).

    Activated once:
    1. User has mistral/vibe CLI installed
    2. User is logged in with their own Mistral account
    """

    name = "mistral"

    def get_capabilities(self) -> ProviderCapabilities:
        """Mistral capabilities: cloud, free, 128k context."""
        return _CAPABILITIES

    async def is_available(self) -> bool:
        """Stub: always False until real implementation."""
        return False

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
        model: str | None = None,
    ) -> ProviderResponse:
        """Not yet implemented. Stub for Phase 1+."""
        raise ProviderNotImplemented(self.name)
