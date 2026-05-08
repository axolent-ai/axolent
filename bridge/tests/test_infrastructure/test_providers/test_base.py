"""Tests für das Provider-Interface (Abstract Base Class).

Verifiziert:
    - LLMProvider kann nicht direkt instanziiert werden
    - Konkrete Subklassen müssen query(), is_available() und get_capabilities() implementieren
    - ProviderResponse hat korrektes success-Property
    - ProviderCapabilities ist frozen
    - Error-Hierarchie: ProviderError, ProviderTimeout, ProviderUnavailable, ProviderNotImplemented
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
        assert "nicht implementiert" in str(err)

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
        """Eine vollstaendige Subklasse kann instanziiert werden."""

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
