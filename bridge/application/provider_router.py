"""Provider router: selects the appropriate LLM provider for each request.

Strategies:
  Current: Claude only (default).
  Planned Phase 1+: per-request user choice, auto-routing by task type,
  fallback chains, cost-aware routing.

Model compatibility (T32 fix):
  When a model ID is passed, the router checks whether it belongs to the
  target provider's family via ModelRegistry. Incompatible models (e.g.
  ``claude-opus-4-7`` sent to Ollama) are silently replaced with ``None``
  so the provider uses its own default. This prevents HTTP 404 errors from
  providers that do not know foreign model namespaces.
"""

from __future__ import annotations

import logging
from typing import Optional

from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderResponse,
    ProviderUnavailable,
)

log = logging.getLogger(__name__)

# Provider-name to ModelRegistry provider-key mapping.
# Keys: provider names as registered in the router (e.g. "claude", "claude_persistent").
# Values: provider keys as used in models.yaml (e.g. "anthropic").
# Providers not listed here have no model compatibility check (model passed as-is).
_PROVIDER_FAMILY: dict[str, str] = {
    "claude": "anthropic",
    "claude_persistent": "anthropic",
    "ollama_local": "ollama",
    "openai": "openai",
    "gemini": "google",
    "mistral": "mistral",
}


def _model_compatible_with_provider(model: str, provider_name: str) -> bool:
    """Check if a model ID belongs to the same family as the provider.

    Uses ModelRegistry to look up the model's provider key and compares
    it to the provider's expected family.

    Args:
        model: Model ID (e.g. "claude-opus-4-7").
        provider_name: Router provider name (e.g. "ollama_local").

    Returns:
        True if compatible or if either side is unknown (permissive fallback).
    """
    expected_family = _PROVIDER_FAMILY.get(provider_name)
    if expected_family is None:
        # Unknown provider: no filtering, pass model through
        return True

    try:
        from application.model_registry import get_registry

        registry = get_registry()
        metadata = registry.get(model)
        if metadata is None:
            # Unknown model: pass through (provider will handle the error)
            return True
        return metadata.provider == expected_family
    except Exception:
        # Registry unavailable: permissive fallback
        return True


class ProviderRouter:
    """Routes requests to the appropriate LLM provider.

    Maintains a registry of all registered providers and selects
    the right one based on provider_name or the default.

    Attributes:
        providers: Dict of provider name to provider instance.
        default: Name of the default provider.
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        default: str = "claude",
    ) -> None:
        self.providers = providers
        self.default = default

        if default not in providers:
            available = list(providers.keys())
            raise ValueError(
                f"Default provider '{default}' not in registered providers: {available}"
            )

        log.info(
            "ProviderRouter initialized. Default: '%s', registered: %s",
            default,
            list(providers.keys()),
        )

    async def route(
        self,
        prompt: str,
        system_prompt: str = "",
        provider_name: Optional[str] = None,
        timeout_seconds: int = 120,
        user_id: int | None = None,
        chat_id: int | None = None,
        model: str | None = None,
    ) -> ProviderResponse:
        """Send a request to the desired provider (or default).

        Args:
            prompt: User message / prompt.
            system_prompt: Optional system prompt.
            provider_name: Explicit provider name (None = default).
            timeout_seconds: Timeout for the provider call.
            user_id: Optional Telegram user ID (required by claude_persistent).
            chat_id: Optional Telegram chat ID (required by claude_persistent).
            model: Optional model ID (None = provider default).

        Returns:
            ProviderResponse with answer or error.

        Raises:
            ValueError: If the provider is not registered.
            ProviderUnavailable: If the provider is not available.
            ProviderError: For other provider errors (base class).
        """
        target = provider_name or self.default

        if target not in self.providers:
            raise ValueError(
                f"Provider '{target}' not registered. "
                f"Available: {list(self.providers.keys())}"
            )

        provider = self.providers[target]

        if not provider.is_available():
            raise ProviderUnavailable(
                target,
                reason="CLI not installed or no API key set",
            )

        # T32 fix: filter model by provider compatibility.
        # If the model belongs to a different provider family (e.g. user set
        # /setmodel opus but this call targets Ollama), drop the model so the
        # provider falls back to its own default.
        effective_model = model
        if model is not None and not _model_compatible_with_provider(model, target):
            log.info(
                "Model '%s' is not compatible with provider '%s', "
                "using provider default instead",
                model,
                target,
            )
            effective_model = None

        log.info(
            "Routing to provider '%s'%s",
            target,
            f" (model={effective_model})" if effective_model else "",
        )

        # Build provider-specific kwargs.
        # claude_persistent needs user_id/chat_id, other providers ignore them.
        kwargs: dict = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "timeout_seconds": timeout_seconds,
        }
        if user_id is not None:
            kwargs["user_id"] = user_id
        if chat_id is not None:
            kwargs["chat_id"] = chat_id
        if effective_model is not None:
            kwargs["model"] = effective_model

        return await provider.query(**kwargs)

    def list_available(self) -> list[str]:
        """Return a list of all available providers."""
        return [
            name for name, provider in self.providers.items() if provider.is_available()
        ]

    def list_registered(self) -> list[str]:
        """Return a list of all registered providers."""
        return list(self.providers.keys())

    def get_capabilities(self, provider_name: str) -> ProviderCapabilities:
        """Return the capabilities of a registered provider.

        Args:
            provider_name: Name of the provider.

        Returns:
            ProviderCapabilities instance.

        Raises:
            ValueError: If the provider is not registered.
        """
        if provider_name not in self.providers:
            raise ValueError(
                f"Provider '{provider_name}' not registered. "
                f"Available: {list(self.providers.keys())}"
            )
        return self.providers[provider_name].get_capabilities()
