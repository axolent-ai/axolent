"""Provider-Interface für LLM-Anbindungen.

Jeder Provider muss query(), is_available() und get_capabilities() implementieren.
Modus-B-Pflicht: Claude-Provider nutzt Subprozess.
Andere Provider können API-Keys nutzen, aber niemals OAuth-Tokens hijacken.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass(frozen=True)
class ProviderCapabilities:
    """Beschreibt was ein Provider kann.

    Wird vom Router genutzt um Anfragen passend zu routen.

    Attributes:
        supports_streaming: Kann der Provider Streaming-Responses?
        supports_tool_use: Kann der Provider Function-Calling / Tool-Use?
        supports_vision: Kann der Provider Bilder verarbeiten?
        max_context_tokens: Maximale Context-Länge in Tokens.
        max_memory_chars: Max Memory-Zeichen im System-Prompt (Phase 1+, per-Provider).
        cost_class: Kostenklasse ("free", "subscription", "pay_per_use").
        privacy_class: Wo werden Daten verarbeitet? ("cloud", "local").
        available_models: Liste der verfügbaren Modell-IDs.
    """

    supports_streaming: bool = False
    supports_tool_use: bool = False
    supports_vision: bool = False
    max_context_tokens: int = 32_000
    max_memory_chars: int = 4000
    cost_class: str = "free"  # "free", "subscription", "pay_per_use"
    privacy_class: str = "cloud"  # "cloud", "local"
    available_models: list[str] = field(default_factory=list)


@dataclass
class ProviderResponse:
    """Standardisierte Antwort eines LLM-Providers.

    Attributes:
        text: Die generierte Antwort (leer bei Fehler).
        duration_seconds: Dauer des Aufrufs in Sekunden.
        provider_name: Name des Providers der geantwortet hat.
        model: Optionaler Modell-Identifier (z.B. "claude-sonnet-4-20250514").
        error: Fehlerbeschreibung wenn der Aufruf fehlgeschlagen ist.
    """

    text: str
    duration_seconds: float
    provider_name: str
    model: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """True wenn kein Fehler aufgetreten ist und Text vorhanden."""
        return self.error is None and bool(self.text)


# ---------------------------------------------------------------------------
# Provider-Error-Hierarchie
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Basis-Klasse für alle Provider-Fehler.

    Attributes:
        provider_name: Name des Providers der den Fehler ausgelöst hat.
        retryable: True wenn ein Retry sinnvoll ist (z.B. Timeout).
    """

    def __init__(self, provider_name: str, retryable: bool, message: str) -> None:
        self.provider_name = provider_name
        self.retryable = retryable
        super().__init__(message)


class ProviderTimeout(ProviderError):
    """Provider hat zu lange gebraucht. Retry sinnvoll."""

    def __init__(self, provider_name: str, timeout_seconds: int) -> None:
        super().__init__(
            provider_name,
            retryable=True,
            message=f"Timeout nach {timeout_seconds}s",
        )
        self.timeout_seconds = timeout_seconds


class ProviderUnavailable(ProviderError):
    """Provider ist nicht verfügbar (CLI fehlt, kein API-Key, etc.)."""

    def __init__(self, provider_name: str, reason: str) -> None:
        super().__init__(
            provider_name,
            retryable=False,
            message=f"{provider_name} nicht verfügbar: {reason}",
        )


class ProviderNotImplemented(ProviderError):
    """Stub-Provider, query() ist noch nicht implementiert."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(
            provider_name,
            retryable=False,
            message=f"{provider_name} ist noch nicht implementiert (Phase 1+)",
        )


class LLMProvider(ABC):
    """Abstrakte Basisklasse für alle LLM-Provider.

    Jeder konkrete Provider registriert sich mit einem eindeutigen `name`
    und implementiert query(), is_available() und get_capabilities().
    """

    name: str  # "claude", "openai", "gemini", "mistral", "ollama_local"

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Gibt die Fähigkeiten dieses Providers zurück."""
        ...

    @abstractmethod
    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
        model: str | None = None,
    ) -> ProviderResponse:
        """Sendet eine Anfrage an den Provider und liefert eine Antwort.

        Args:
            prompt: Die User-Nachricht / der Prompt.
            system_prompt: Optionaler System-Prompt.
            timeout_seconds: Maximale Wartezeit bevor Timeout.
            model: Optionaler Modell-Identifier (None = Provider-Default).

        Returns:
            ProviderResponse mit Text oder Fehler.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Prüft ob der Provider auf diesem System verfügbar ist.

        Prüft z.B. ob das CLI-Binary im PATH liegt oder ein API-Key gesetzt ist.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class StreamingProvider(ABC):
    """Mixin für Provider mit Streaming-Support.

    Provider die Token-Streaming unterstützen sollten dieses Mixin
    zusätzlich zu LLMProvider implementieren. Ermöglicht Type-Safe
    isinstance()-Checks statt fragiler hasattr()-Prüfungen.
    """

    @abstractmethod
    async def query_streaming(
        self,
        prompt: str,
        system_prompt: str = "",
        chat_id: int | None = None,
        user_id: int | None = None,
    ) -> "AsyncIterator":
        """Sendet eine Anfrage und streamt die Antwort Token für Token.

        Args:
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt.
            chat_id: Chat-ID für Process-Routing.
            user_id: User-ID für Process-Routing.

        Yields:
            StreamEvent-Objekte (content_delta, result, error).
        """
        ...
