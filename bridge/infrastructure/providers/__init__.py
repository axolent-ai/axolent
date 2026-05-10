"""LLM Provider-Infrastruktur: Multi-Provider-Abstraktion.

Exportiert das Provider-Interface und alle konkreten Provider-Implementierungen.
Aktuell aktiv:
    - ClaudePersistentProvider (R04, persistent stdin-Pipe, Streaming)
    - ClaudeProvider (Legacy-Fallback, einzelne Subprozesse)
    - OllamaProvider (lokale Inference via Ollama HTTP-API)
Stubs: OpenAI Codex, Gemini, Mistral Vibe.
"""

from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderNotImplemented,
    ProviderResponse,
    ProviderTimeout,
    ProviderUnavailable,
)
from infrastructure.providers.claude_cli import ClaudeProvider
from infrastructure.providers.claude_persistent import ClaudePersistentProvider
from infrastructure.providers.openai_codex_cli import OpenAICodexProvider
from infrastructure.providers.gemini_cli import GeminiProvider
from infrastructure.providers.mistral_vibe_cli import MistralVibeProvider
from infrastructure.providers.ollama_local import OllamaProvider

__all__ = [
    "LLMProvider",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderNotImplemented",
    "ProviderResponse",
    "ProviderTimeout",
    "ProviderUnavailable",
    "ClaudeProvider",
    "ClaudePersistentProvider",
    "OpenAICodexProvider",
    "GeminiProvider",
    "MistralVibeProvider",
    "OllamaProvider",
]
