"""OpenAI Codex CLI Provider (Stub).

Geplant fuer Phase 1+. Nutzt das `codex` CLI-Binary als Subprozess.
Aktuell: nur Skelett, query() raised NotImplementedError.
"""

from __future__ import annotations

import shutil

from infrastructure.providers.base import LLMProvider, ProviderResponse


class OpenAICodexProvider(LLMProvider):
    """OpenAI via Codex CLI als LLM-Provider (Stub).

    Wird aktiviert sobald:
    1. User codex CLI installiert hat
    2. User mit eigenem OpenAI-Account eingeloggt ist
    """

    name = "openai"

    def is_available(self) -> bool:
        """Prueft ob `codex` CLI im PATH ist."""
        return shutil.which("codex") is not None

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
    ) -> ProviderResponse:
        """Noch nicht implementiert. Stub fuer Phase 1+."""
        raise NotImplementedError(
            "OpenAI Codex CLI Provider ist noch nicht implementiert. Kommt in Phase 1+."
        )
