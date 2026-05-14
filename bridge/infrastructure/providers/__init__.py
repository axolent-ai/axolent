"""LLM provider infrastructure: multi-provider abstraction.

Exports the provider interface and all concrete provider implementations.
Currently active:
    * ClaudePersistentProvider (R04, persistent stdin pipe, streaming)
    * ClaudeProvider (legacy fallback, individual subprocesses)
    * OllamaProvider (local inference via Ollama HTTP API)
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
