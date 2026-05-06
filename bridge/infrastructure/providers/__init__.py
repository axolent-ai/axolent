"""LLM Provider-Infrastruktur: Multi-Provider-Abstraktion.

Exportiert das Provider-Interface und alle konkreten Provider-Implementierungen.
Aktuell aktiv: ClaudeProvider (Modus B, CLI-Subprozess).
Stubs: OpenAI Codex, Gemini, Mistral Vibe, Ollama.
"""

from infrastructure.providers.base import LLMProvider, ProviderResponse
from infrastructure.providers.claude_cli import ClaudeProvider
from infrastructure.providers.openai_codex_cli import OpenAICodexProvider
from infrastructure.providers.gemini_cli import GeminiProvider
from infrastructure.providers.mistral_vibe_cli import MistralVibeProvider
from infrastructure.providers.ollama_local import OllamaProvider

__all__ = [
    "LLMProvider",
    "ProviderResponse",
    "ClaudeProvider",
    "OpenAICodexProvider",
    "GeminiProvider",
    "MistralVibeProvider",
    "OllamaProvider",
]
