"""Google Gemini CLI provider (stub).

Planned for Phase 1+. Uses the `gemini` CLI binary as subprocess.
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
    supports_vision=True,
    max_context_tokens=1_000_000,
    cost_class="free",
    privacy_class="cloud",
    available_models=["gemini-2.5-flash", "gemini-3.1-pro"],
)


class GeminiProvider(LLMProvider):
    """Google Gemini via CLI as LLM provider (stub).

    Activated once:
    1. User has gemini CLI installed
    2. User is logged in with their own Google account
    """

    name = "gemini"

    def get_capabilities(self) -> ProviderCapabilities:
        """Gemini capabilities: cloud, free, 1M context."""
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
