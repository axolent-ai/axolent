"""Ollama Local Provider (Stub).

Geplant fuer Phase 1+. Nutzt `ollama` CLI oder HTTP-API (localhost:11434).
Aktuell: nur Skelett, query() raised NotImplementedError.
"""

from __future__ import annotations

import shutil

from infrastructure.providers.base import LLMProvider, ProviderResponse


class OllamaProvider(LLMProvider):
    """Ollama als lokaler LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User Ollama installiert hat
    2. Mindestens ein Modell gepullt ist (ollama list)
    """

    name = "ollama"

    def is_available(self) -> bool:
        """Prueft ob `ollama` CLI im PATH ist."""
        return shutil.which("ollama") is not None

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
    ) -> ProviderResponse:
        """Noch nicht implementiert. Stub fuer Phase 1+."""
        raise NotImplementedError(
            "Ollama Local Provider ist noch nicht implementiert. Kommt in Phase 1+."
        )
