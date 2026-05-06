"""Provider-Interface fuer LLM-Anbindungen.

Jeder Provider muss query() und is_available() implementieren.
Modus-B-Pflicht: Claude-Provider nutzt Subprozess.
Andere Provider koennen API-Keys nutzen, aber niemals OAuth-Tokens hijacken.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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


class LLMProvider(ABC):
    """Abstract base class fuer alle LLM-Provider.

    Jeder konkrete Provider registriert sich mit einem eindeutigen `name`
    und implementiert query() + is_available().
    """

    name: str  # "claude", "openai", "gemini", "mistral", "ollama"

    @abstractmethod
    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
    ) -> ProviderResponse:
        """Sendet eine Anfrage an den Provider und liefert eine Antwort.

        Args:
            prompt: Die User-Nachricht / der Prompt.
            system_prompt: Optionaler System-Prompt.
            timeout_seconds: Maximale Wartezeit bevor Timeout.

        Returns:
            ProviderResponse mit Text oder Fehler.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Prueft ob der Provider auf diesem System verfuegbar ist.

        Prueft z.B. ob das CLI-Binary im PATH liegt oder ein API-Key gesetzt ist.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
