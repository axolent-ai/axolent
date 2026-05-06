"""Google Gemini CLI Provider (Stub).

Geplant fuer Phase 1+. Nutzt das `gemini` CLI-Binary als Subprozess.
Aktuell: nur Skelett, query() raised NotImplementedError.
"""

from __future__ import annotations

import shutil

from infrastructure.providers.base import LLMProvider, ProviderResponse


class GeminiProvider(LLMProvider):
    """Google Gemini via CLI als LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User gemini CLI installiert hat
    2. User mit eigenem Google-Account eingeloggt ist
    """

    name = "gemini"

    def is_available(self) -> bool:
        """Prueft ob `gemini` CLI im PATH ist."""
        return shutil.which("gemini") is not None

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
    ) -> ProviderResponse:
        """Noch nicht implementiert. Stub fuer Phase 1+."""
        raise NotImplementedError(
            "Gemini CLI Provider ist noch nicht implementiert. Kommt in Phase 1+."
        )
