"""Tests für den ProviderRouter.

Verifiziert:
    - Default-Routing funktioniert
    - Explizites Provider-Routing funktioniert
    - Unbekannter Provider raised ValueError
    - Nicht verfuegbarer Provider raised ProviderUnavailable
    - list_available() und list_registered() korrekt
    - get_capabilities() liefert ProviderCapabilities
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.provider_router import ProviderRouter
from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderResponse,
    ProviderUnavailable,
)


def _make_mock_provider(
    name: str, available: bool = True, *, with_async_query: bool = False
) -> LLMProvider:
    """Erstellt einen gemockten Provider mit Capabilities.

    Args:
        name: Provider-Name.
        available: Ob is_available() True liefert.
        with_async_query: Wenn True, wird query als AsyncMock gesetzt
            (nötig für Tests die route() awaiten). Sonst bleibt query
            ein normaler MagicMock (vermeidet RuntimeWarning bei
            nie-awaiteten Coroutines).
    """
    provider = MagicMock(spec=LLMProvider)
    provider.name = name
    provider.is_available = MagicMock(return_value=available)
    if with_async_query:
        provider.query = AsyncMock(
            return_value=ProviderResponse(
                text=f"Response from {name}",
                duration_seconds=0.5,
                provider_name=name,
            )
        )
    provider.get_capabilities = MagicMock(
        return_value=ProviderCapabilities(
            max_context_tokens=100_000,
            cost_class="subscription",
            privacy_class="cloud",
            available_models=[f"{name}-default"],
        )
    )
    return provider


class TestProviderRouterInit:
    """Tests für Router-Initialisierung."""

    def test_init_with_valid_default(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        router = ProviderRouter(providers=providers, default="claude")
        assert router.default == "claude"

    def test_init_with_invalid_default_raises(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        with pytest.raises(ValueError, match="Default-Provider"):
            ProviderRouter(providers=providers, default="nonexistent")


class TestProviderRouterRouting:
    """Tests für route()."""

    @pytest.mark.asyncio
    async def test_default_routing(self) -> None:
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        result = await router.route("Hallo")

        assert result.provider_name == "claude"
        assert result.text == "Response from claude"
        mock_claude.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_provider_routing(self) -> None:
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        mock_gemini = _make_mock_provider("gemini", with_async_query=True)
        providers = {"claude": mock_claude, "gemini": mock_gemini}
        router = ProviderRouter(providers=providers, default="claude")

        result = await router.route("Hallo", provider_name="gemini")

        assert result.provider_name == "gemini"
        mock_gemini.query.assert_called_once()
        mock_claude.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_value_error(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        router = ProviderRouter(providers=providers, default="claude")

        with pytest.raises(ValueError, match="nicht registriert"):
            await router.route("Hallo", provider_name="unknown")

    @pytest.mark.asyncio
    async def test_unavailable_provider_raises_provider_unavailable(self) -> None:
        mock_claude = _make_mock_provider("claude", available=False)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        with pytest.raises(ProviderUnavailable) as exc_info:
            await router.route("Hallo")
        assert "verf" in str(exc_info.value).lower()
        assert exc_info.value.retryable is False


class TestProviderRouterListing:
    """Tests für list_available() und list_registered()."""

    def test_list_registered(self) -> None:
        providers = {
            "claude": _make_mock_provider("claude"),
            "gemini": _make_mock_provider("gemini"),
        }
        router = ProviderRouter(providers=providers, default="claude")
        assert sorted(router.list_registered()) == ["claude", "gemini"]

    def test_list_available_filters_unavailable(self) -> None:
        providers = {
            "claude": _make_mock_provider("claude", available=True),
            "gemini": _make_mock_provider("gemini", available=False),
            "ollama": _make_mock_provider("ollama", available=True),
        }
        router = ProviderRouter(providers=providers, default="claude")
        available = router.list_available()
        assert "claude" in available
        assert "ollama" in available
        assert "gemini" not in available


class TestProviderRouterCapabilities:
    """Tests für get_capabilities()."""

    def test_get_capabilities_returns_correct_data(self) -> None:
        mock_claude = _make_mock_provider("claude")
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        caps = router.get_capabilities("claude")
        assert isinstance(caps, ProviderCapabilities)
        assert caps.max_context_tokens == 100_000
        assert caps.cost_class == "subscription"
        assert caps.available_models == ["claude-default"]

    def test_get_capabilities_unknown_provider_raises(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        router = ProviderRouter(providers=providers, default="claude")

        with pytest.raises(ValueError, match="nicht registriert"):
            router.get_capabilities("unknown")
