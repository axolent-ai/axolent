"""Provider-Router: waehlt fuer jede Anfrage den passenden LLM-Provider.

Strategien:
  Aktuell: nur Claude (default).
  Geplant Phase 1+: User-Wahl pro Anfrage, Auto-Routing nach Aufgabentyp,
  Fallback-Chains, Cost-Aware-Routing.
"""

from __future__ import annotations

import logging
from typing import Optional

from infrastructure.providers.base import LLMProvider, ProviderResponse

log = logging.getLogger(__name__)


class ProviderRouter:
    """Routet Anfragen an den richtigen LLM-Provider.

    Haelt eine Registry aller registrierten Provider und waehlt
    anhand des provider_name oder des Defaults den richtigen aus.

    Attributes:
        providers: Dict von Provider-Name zu Provider-Instanz.
        default: Name des Default-Providers.
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
                f"Default-Provider '{default}' nicht in registrierten "
                f"Providern: {available}"
            )

        log.info(
            "ProviderRouter initialisiert. Default: '%s', registriert: %s",
            default,
            list(providers.keys()),
        )

    async def route(
        self,
        prompt: str,
        system_prompt: str = "",
        provider_name: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> ProviderResponse:
        """Sendet eine Anfrage an den gewuenschten Provider (oder Default).

        Args:
            prompt: User-Nachricht / Prompt.
            system_prompt: Optionaler System-Prompt.
            provider_name: Expliziter Provider-Name (None = Default).
            timeout_seconds: Timeout fuer den Provider-Aufruf.

        Returns:
            ProviderResponse mit Antwort oder Fehler.

        Raises:
            ValueError: Wenn der Provider nicht registriert ist.
            RuntimeError: Wenn der Provider nicht verfuegbar ist.
        """
        target = provider_name or self.default

        if target not in self.providers:
            raise ValueError(
                f"Provider '{target}' nicht registriert. "
                f"Verfuegbar: {list(self.providers.keys())}"
            )

        provider = self.providers[target]

        if not provider.is_available():
            raise RuntimeError(
                f"Provider '{target}' ist nicht verfuegbar. "
                f"Pruefe ob das CLI installiert ist."
            )

        log.info("Routing an Provider '%s'", target)
        return await provider.query(
            prompt=prompt,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
        )

    def list_available(self) -> list[str]:
        """Gibt eine Liste aller verfuegbaren Provider zurueck."""
        return [
            name for name, provider in self.providers.items() if provider.is_available()
        ]

    def list_registered(self) -> list[str]:
        """Gibt eine Liste aller registrierten Provider zurueck."""
        return list(self.providers.keys())
