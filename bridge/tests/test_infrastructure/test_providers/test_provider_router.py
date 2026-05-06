"""Tests fuer den ProviderRouter.

Verifiziert:
    - Default-Routing funktioniert
    - Explizites Provider-Routing funktioniert
    - Unbekannter Provider raised ValueError
    - Nicht verfuegbarer Provider raised RuntimeError
    - list_available() und list_registered() korrekt
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.provider_router import ProviderRouter
from infrastructure.providers.base import LLMProvider, ProviderResponse


def _make_mock_provider(name: str, available: bool = True) -> LLMProvider:
    """Erstellt einen gemockten Provider."""
    provider = MagicMock(spec=LLMProvider)
    provider.name = name
    provider.is_available = MagicMock(return_value=available)
    provider.query = AsyncMock(
        return_value=ProviderResponse(
            text=f"Response from {name}",
            duration_seconds=0.5,
            provider_name=name,
        )
    )
    return provider


class TestProviderRouterInit:
    """Tests fuer Router-Initialisierung."""

    def test_init_with_valid_default(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        router = ProviderRouter(providers=providers, default="claude")
        assert router.default == "claude"

    def test_init_with_invalid_default_raises(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        with pytest.raises(ValueError, match="Default-Provider"):
            ProviderRouter(providers=providers, default="nonexistent")


class TestProviderRouterRouting:
    """Tests fuer route()."""

    @pytest.mark.asyncio
    async def test_default_routing(self) -> None:
        mock_claude = _make_mock_provider("claude")
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        result = await router.route("Hallo")

        assert result.provider_name == "claude"
        assert result.text == "Response from claude"
        mock_claude.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_provider_routing(self) -> None:
        mock_claude = _make_mock_provider("claude")
        mock_gemini = _make_mock_provider("gemini")
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
    async def test_unavailable_provider_raises_runtime_error(self) -> None:
        mock_claude = _make_mock_provider("claude", available=False)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        with pytest.raises(RuntimeError, match="nicht verfuegbar"):
            await router.route("Hallo")


class TestProviderRouterListing:
    """Tests fuer list_available() und list_registered()."""

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
