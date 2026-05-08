"""Provider-Router: wählt für jede Anfrage den passenden LLM-Provider.

Strategien:
  Aktuell: nur Claude (default).
  Geplant Phase 1+: User-Wahl pro Anfrage, Auto-Routing nach Aufgabentyp,
  Fallback-Chains, Cost-Aware-Routing.
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


class ProviderRouter:
    """Routet Anfragen an den richtigen LLM-Provider.

    Hält eine Registry aller registrierten Provider und wählt
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
        user_id: int | None = None,
        chat_id: int | None = None,
    ) -> ProviderResponse:
        """Sendet eine Anfrage an den gewünschten Provider (oder Default).

        Args:
            prompt: User-Nachricht / Prompt.
            system_prompt: Optionaler System-Prompt.
            provider_name: Expliziter Provider-Name (None = Default).
            timeout_seconds: Timeout für den Provider-Aufruf.
            user_id: Optionale Telegram-User-ID (benötigt von claude_persistent).
            chat_id: Optionale Telegram-Chat-ID (benötigt von claude_persistent).

        Returns:
            ProviderResponse mit Antwort oder Fehler.

        Raises:
            ValueError: Wenn der Provider nicht registriert ist.
            ProviderUnavailable: Wenn der Provider nicht verfügbar ist.
            ProviderError: Bei sonstigen Provider-Fehlern (Basis-Klasse).
        """
        target = provider_name or self.default

        if target not in self.providers:
            raise ValueError(
                f"Provider '{target}' nicht registriert. "
                f"Verfügbar: {list(self.providers.keys())}"
            )

        provider = self.providers[target]

        if not provider.is_available():
            raise ProviderUnavailable(
                target,
                reason="CLI nicht installiert oder kein API-Key gesetzt",
            )

        log.info("Routing an Provider '%s'", target)

        # Provider-spezifische kwargs zusammenbauen.
        # claude_persistent braucht user_id/chat_id, andere Provider ignorieren sie.
        kwargs: dict = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "timeout_seconds": timeout_seconds,
        }
        if user_id is not None:
            kwargs["user_id"] = user_id
        if chat_id is not None:
            kwargs["chat_id"] = chat_id

        return await provider.query(**kwargs)

    def list_available(self) -> list[str]:
        """Gibt eine Liste aller verfügbaren Provider zurück."""
        return [
            name for name, provider in self.providers.items() if provider.is_available()
        ]

    def list_registered(self) -> list[str]:
        """Gibt eine Liste aller registrierten Provider zurück."""
        return list(self.providers.keys())

    def get_capabilities(self, provider_name: str) -> ProviderCapabilities:
        """Gibt die Capabilities eines registrierten Providers zurück.

        Args:
            provider_name: Name des Providers.

        Returns:
            ProviderCapabilities-Instanz.

        Raises:
            ValueError: Wenn der Provider nicht registriert ist.
        """
        if provider_name not in self.providers:
            raise ValueError(
                f"Provider '{provider_name}' nicht registriert. "
                f"Verfügbar: {list(self.providers.keys())}"
            )
        return self.providers[provider_name].get_capabilities()
