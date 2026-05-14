"""Tests for the provider interface (abstract base class).

Verifies:
    - LLMProvider cannot be instantiated directly
    - Concrete subclasses must implement query(), is_available(), and get_capabilities()
    - ProviderResponse has correct success property
    - ProviderCapabilities is frozen
    - Error hierarchy: ProviderError, ProviderTimeout, ProviderUnavailable, ProviderNotImplemented
"""

from __future__ import annotations

import pytest

from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderNotImplemented,
    ProviderResponse,
    ProviderTimeout,
    ProviderUnavailable,
    StreamingProvider,
)


class TestProviderResponse:
    """Tests für ProviderResponse Dataclass."""

    def test_success_when_text_present_no_error(self) -> None:
        resp = ProviderResponse(
            text="Hello",
            duration_seconds=1.0,
            provider_name="test",
        )
        assert resp.success is True

    def test_not_success_when_error_set(self) -> None:
        resp = ProviderResponse(
            text="Hello",
            duration_seconds=1.0,
            provider_name="test",
            error="something went wrong",
        )
        assert resp.success is False

    def test_not_success_when_text_empty(self) -> None:
        resp = ProviderResponse(
            text="",
            duration_seconds=1.0,
            provider_name="test",
        )
        assert resp.success is False

    def test_model_field_optional(self) -> None:
        resp = ProviderResponse(
            text="Hi",
            duration_seconds=0.5,
            provider_name="test",
            model="gpt-4o",
        )
        assert resp.model == "gpt-4o"


class TestProviderCapabilities:
    """Tests für ProviderCapabilities Dataclass."""

    def test_defaults(self) -> None:
        caps = ProviderCapabilities()
        assert caps.supports_streaming is False
        assert caps.supports_tool_use is False
        assert caps.supports_vision is False
        assert caps.max_context_tokens == 32_000
        assert caps.cost_class == "free"
        assert caps.privacy_class == "cloud"
        assert caps.available_models == []

    def test_custom_values(self) -> None:
        caps = ProviderCapabilities(
            supports_streaming=True,
            max_context_tokens=200_000,
            cost_class="subscription",
            privacy_class="local",
            available_models=["model-a", "model-b"],
        )
        assert caps.supports_streaming is True
        assert caps.max_context_tokens == 200_000
        assert caps.available_models == ["model-a", "model-b"]

    def test_frozen(self) -> None:
        caps = ProviderCapabilities()
        with pytest.raises(AttributeError):
            caps.max_context_tokens = 999  # type: ignore


class TestProviderErrorHierarchy:
    """Tests für die Provider-Error-Hierarchie."""

    def test_provider_error_is_exception(self) -> None:
        err = ProviderError("test", retryable=False, message="boom")
        assert isinstance(err, Exception)
        assert err.provider_name == "test"
        assert err.retryable is False
        assert "boom" in str(err)

    def test_provider_timeout(self) -> None:
        err = ProviderTimeout("claude", timeout_seconds=120)
        assert isinstance(err, ProviderError)
        assert err.retryable is True
        assert err.timeout_seconds == 120
        assert "120" in str(err)

    def test_provider_unavailable(self) -> None:
        err = ProviderUnavailable("gemini", reason="CLI fehlt")
        assert isinstance(err, ProviderError)
        assert err.retryable is False
        assert "gemini" in str(err)
        assert "CLI fehlt" in str(err)

    def test_provider_not_implemented(self) -> None:
        err = ProviderNotImplemented("ollama")
        assert isinstance(err, ProviderError)
        assert err.retryable is False
        assert "ollama" in str(err)
        assert "not yet implemented" in str(err)

    def test_all_errors_catchable_as_provider_error(self) -> None:
        """Alle spezifischen Fehler sind via ProviderError fangbar."""
        errors = [
            ProviderTimeout("x", 60),
            ProviderUnavailable("x", "reason"),
            ProviderNotImplemented("x"),
        ]
        for err in errors:
            try:
                raise err
            except ProviderError as caught:
                assert caught is err


class TestLLMProviderInterface:
    """Tests dass LLMProvider als ABC korrekt erzwingt."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore

    def test_incomplete_subclass_raises(self) -> None:
        """Eine Subklasse ohne query/is_available/get_capabilities kann nicht instanziiert werden."""

        class IncompleteProvider(LLMProvider):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore

    def test_complete_subclass_works(self) -> None:
        """Eine vollständige Subklasse kann instanziiert werden."""

        class DummyProvider(LLMProvider):
            name = "dummy"

            def get_capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(max_context_tokens=8_000)

            def is_available(self) -> bool:
                return True

            async def query(
                self, prompt, system_prompt="", timeout_seconds=120, model=None
            ):
                return ProviderResponse(
                    text="dummy response",
                    duration_seconds=0.1,
                    provider_name=self.name,
                )

        provider = DummyProvider()
        assert provider.name == "dummy"
        assert provider.is_available() is True
        caps = provider.get_capabilities()
        assert caps.max_context_tokens == 8_000


class TestStreamingProviderMixin:
    """Tests für das StreamingProvider-Mixin."""

    def test_cannot_instantiate_abstract(self) -> None:
        """StreamingProvider kann nicht direkt instanziiert werden."""
        with pytest.raises(TypeError):
            StreamingProvider()  # type: ignore

    def test_incomplete_streaming_provider_raises(self) -> None:
        """Subklasse ohne query_streaming() kann nicht instanziiert werden."""

        class IncompleteStreamingProvider(StreamingProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteStreamingProvider()  # type: ignore

    def test_complete_streaming_provider_works(self) -> None:
        """Vollständige StreamingProvider-Subklasse ist instanziierbar."""
        from typing import AsyncIterator

        class DummyStreamingProvider(LLMProvider, StreamingProvider):
            name = "streaming_dummy"

            def get_capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(supports_streaming=True)

            def is_available(self) -> bool:
                return True

            async def query(
                self, prompt, system_prompt="", timeout_seconds=120, model=None
            ):
                return ProviderResponse(
                    text="response",
                    duration_seconds=0.1,
                    provider_name=self.name,
                )

            async def query_streaming(
                self, prompt, system_prompt="", chat_id=None, user_id=None
            ) -> AsyncIterator:
                yield "token"  # type: ignore

        provider = DummyStreamingProvider()
        assert isinstance(provider, StreamingProvider)
        assert isinstance(provider, LLMProvider)
        assert provider.name == "streaming_dummy"

    def test_isinstance_check_for_streaming(self) -> None:
        """isinstance(provider, StreamingProvider) ist der korrekte Type-Check."""
        from infrastructure.providers.claude_persistent import ClaudePersistentProvider

        # ClaudePersistentProvider erweitert StreamingProvider
        assert issubclass(ClaudePersistentProvider, StreamingProvider)
