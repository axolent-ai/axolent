"""Tests for the ProviderRouter.

Verifies:
    * Default routing works
    * Explicit provider routing works
    * Unknown provider raises ValueError
    * Unavailable provider raises ProviderUnavailable
    * list_available() and list_registered() correct
    * get_capabilities() delivers ProviderCapabilities
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
    """Create a mocked provider with capabilities.

    Args:
        name: Provider name.
        available: Whether is_available() returns True.
        with_async_query: If True, query is set as AsyncMock
            (needed for tests that await route()). Otherwise query
            stays a normal MagicMock (avoids RuntimeWarning for
            never-awaited coroutines).
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
    """Tests for router initialization."""

    def test_init_with_valid_default(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        router = ProviderRouter(providers=providers, default="claude")
        assert router.default == "claude"

    def test_init_with_invalid_default_raises(self) -> None:
        providers = {"claude": _make_mock_provider("claude")}
        with pytest.raises(ValueError, match="Default provider"):
            ProviderRouter(providers=providers, default="nonexistent")


class TestProviderRouterRouting:
    """Tests for route()."""

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

        with pytest.raises(ValueError, match="not registered"):
            await router.route("Hallo", provider_name="unknown")

    @pytest.mark.asyncio
    async def test_unavailable_provider_raises_provider_unavailable(self) -> None:
        mock_claude = _make_mock_provider("claude", available=False)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        with pytest.raises(ProviderUnavailable) as exc_info:
            await router.route("Hallo")
        assert "unavailable" in str(exc_info.value).lower()
        assert exc_info.value.retryable is False


class TestProviderRouterListing:
    """Tests for list_available() and list_registered()."""

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
            "ollama_local": _make_mock_provider("ollama_local", available=True),
        }
        router = ProviderRouter(providers=providers, default="claude")
        available = router.list_available()
        assert "claude" in available
        assert "ollama_local" in available
        assert "gemini" not in available


class TestProviderRouterCapabilities:
    """Tests for get_capabilities()."""

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

        with pytest.raises(ValueError, match="not registered"):
            router.get_capabilities("unknown")


class TestProviderRouterUserChatId:
    """Tests for user_id/chat_id pass-through (Task 4: non-streaming + claude_persistent)."""

    @pytest.mark.asyncio
    async def test_route_passes_user_id_chat_id_to_provider(self) -> None:
        """user_id and chat_id are passed through to provider.query()."""
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        await router.route("Hallo", user_id=42, chat_id=100)

        call_kwargs = mock_claude.query.call_args[1]
        assert call_kwargs["user_id"] == 42
        assert call_kwargs["chat_id"] == 100

    @pytest.mark.asyncio
    async def test_route_without_user_id_omits_from_kwargs(self) -> None:
        """Without user_id/chat_id they are not passed to the provider."""
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        await router.route("Hallo")

        call_kwargs = mock_claude.query.call_args[1]
        assert "user_id" not in call_kwargs
        assert "chat_id" not in call_kwargs

    @pytest.mark.asyncio
    async def test_route_with_only_user_id_passes_only_user_id(self) -> None:
        """Only user_id without chat_id: only user_id is passed."""
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        await router.route("Hallo", user_id=42)

        call_kwargs = mock_claude.query.call_args[1]
        assert call_kwargs["user_id"] == 42
        assert "chat_id" not in call_kwargs

    @pytest.mark.asyncio
    async def test_route_passes_model_to_provider(self) -> None:
        """model parameter is passed through to provider.query()."""
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        await router.route("Hallo", model="claude-opus-4-7")

        call_kwargs = mock_claude.query.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-7"


class TestProviderRouterModelCompatibility:
    """T32: model-by-provider compatibility filter.

    When /setmodel sets a Claude model and a debate queries Ollama,
    the Claude model ID must NOT be forwarded to Ollama (HTTP 404).
    Instead, model=None so the provider uses its own default.
    """

    @pytest.mark.asyncio
    async def test_compatible_model_passed_through(self) -> None:
        """Claude model is passed to Claude provider (compatible)."""
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        providers = {"claude": mock_claude}
        router = ProviderRouter(providers=providers, default="claude")

        await router.route("Hallo", model="claude-opus-4-7")

        call_kwargs = mock_claude.query.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-7"

    @pytest.mark.asyncio
    async def test_incompatible_model_dropped_for_ollama(self) -> None:
        """Claude model sent to Ollama provider is dropped (None = default)."""
        mock_ollama = _make_mock_provider("ollama_local", with_async_query=True)
        mock_claude = _make_mock_provider("claude", with_async_query=True)
        providers = {"claude": mock_claude, "ollama_local": mock_ollama}
        router = ProviderRouter(providers=providers, default="claude")

        await router.route(
            "Hallo", provider_name="ollama_local", model="claude-opus-4-7"
        )

        call_kwargs = mock_ollama.query.call_args[1]
        assert "model" not in call_kwargs, (
            "Incompatible model should be dropped, not passed to provider"
        )

    @pytest.mark.asyncio
    async def test_debate_user_model_only_to_compatible_providers(self) -> None:
        """/setmodel opus -> Debate -> Claude gets Opus, Ollama gets None."""
        mock_claude = _make_mock_provider("claude_persistent", with_async_query=True)
        mock_ollama = _make_mock_provider("ollama_local", with_async_query=True)
        providers = {
            "claude_persistent": mock_claude,
            "ollama_local": mock_ollama,
        }
        router = ProviderRouter(providers=providers, default="claude_persistent")

        # Route to Claude with model
        await router.route(
            "Test",
            provider_name="claude_persistent",
            model="claude-opus-4-7",
            user_id=1,
            chat_id=10,
        )
        claude_kwargs = mock_claude.query.call_args[1]
        assert claude_kwargs["model"] == "claude-opus-4-7"

        # Route to Ollama with same model
        await router.route(
            "Test",
            provider_name="ollama_local",
            model="claude-opus-4-7",
            user_id=1,
            chat_id=10,
        )
        ollama_kwargs = mock_ollama.query.call_args[1]
        assert "model" not in ollama_kwargs

    @pytest.mark.asyncio
    async def test_none_model_stays_none(self) -> None:
        """model=None is never modified (no compatibility check needed)."""
        mock_ollama = _make_mock_provider("ollama_local", with_async_query=True)
        providers = {
            "claude": _make_mock_provider("claude"),
            "ollama_local": mock_ollama,
        }
        router = ProviderRouter(providers=providers, default="claude")

        await router.route("Hallo", provider_name="ollama_local", model=None)

        call_kwargs = mock_ollama.query.call_args[1]
        assert "model" not in call_kwargs
