"""Mistral Vibe CLI Provider (Stub).

Geplant fuer Phase 1+. Nutzt das `mistral` CLI-Binary als Subprozess.
Aktuell: nur Skelett, query() raised NotImplementedError.
"""

from __future__ import annotations

import shutil

from infrastructure.providers.base import LLMProvider, ProviderResponse


class MistralVibeProvider(LLMProvider):
    """Mistral via Vibe CLI als LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User mistral/vibe CLI installiert hat
    2. User mit eigenem Mistral-Account eingeloggt ist
    """

    name = "mistral"

    def is_available(self) -> bool:
        """Prueft ob `mistral` CLI im PATH ist."""
        return shutil.which("mistral") is not None

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
    ) -> ProviderResponse:
        """Noch nicht implementiert. Stub fuer Phase 1+."""
        raise NotImplementedError(
            "Mistral Vibe CLI Provider ist noch nicht implementiert. Kommt in Phase 1+."
        )
