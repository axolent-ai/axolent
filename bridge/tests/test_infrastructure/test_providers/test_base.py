"""Tests fuer das Provider-Interface (Abstract Base Class).

Verifiziert:
    - LLMProvider kann nicht direkt instanziiert werden
    - Konkrete Subklassen muessen query() und is_available() implementieren
    - ProviderResponse hat korrektes success-Property
"""

from __future__ import annotations

import pytest

from infrastructure.providers.base import LLMProvider, ProviderResponse


class TestProviderResponse:
    """Tests fuer ProviderResponse Dataclass."""

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


class TestLLMProviderInterface:
    """Tests dass LLMProvider als ABC korrekt erzwingt."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore

    def test_incomplete_subclass_raises(self) -> None:
        """Eine Subklasse ohne query/is_available kann nicht instanziiert werden."""

        class IncompleteProvider(LLMProvider):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore

    def test_complete_subclass_works(self) -> None:
        """Eine vollstaendige Subklasse kann instanziiert werden."""

        class DummyProvider(LLMProvider):
            name = "dummy"

            def is_available(self) -> bool:
                return True

            async def query(self, prompt, system_prompt="", timeout_seconds=120):
                return ProviderResponse(
                    text="dummy response",
                    duration_seconds=0.1,
                    provider_name=self.name,
                )

        provider = DummyProvider()
        assert provider.name == "dummy"
        assert provider.is_available() is True
