"""Tests für application.debate_orchestrator (R10: Multi-AI-Debate).

Testet:
- 2 Provider parallel, beide antworten erfolgreich
- 1 Provider crasht, anderer antwortet (errors-Dict enthält ihn)
- Alle Provider crashen (DebateResult mit leeren responses)
- Timeout wird respektiert
- Konsens-Heuristik gibt sinnvollen Output
- Kein Provider verfügbar
- Provider-Deduplizierung (claude_persistent + claude = eine Gruppe)
"""

from __future__ import annotations

from unittest.mock import patch

from application.debate_orchestrator import (
    DebateOrchestrator,
    DebateResult,
    deduplicate_providers,
)
from application.provider_router import ProviderRouter
from infrastructure.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderResponse,
)


class _MockProvider(LLMProvider):
    """Mock-Provider fuer Tests."""

    def __init__(
        self,
        name: str,
        available: bool = True,
        response_text: str = "Mock response",
        should_raise: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self._available = available
        self._response_text = response_text
        self._should_raise = should_raise
        self._delay = delay

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def is_available(self) -> bool:
        return self._available

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout_seconds: int = 120,
        model: str | None = None,
        **kwargs,
    ) -> ProviderResponse:
        import asyncio

        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_raise is not None:
            raise self._should_raise
        return ProviderResponse(
            text=self._response_text,
            duration_seconds=0.1,
            provider_name=self.name,
        )


def _make_router(providers: dict[str, _MockProvider]) -> ProviderRouter:
    """Erstellt einen ProviderRouter mit Mock-Providern."""
    # Sichergestellt dass ein Default existiert
    default = next(iter(providers.keys()))
    return ProviderRouter(providers=providers, default=default)


class TestDebateOrchestratorBasic:
    """Grundlegende Debate-Tests."""

    async def test_two_providers_both_succeed(self) -> None:
        """Zwei Provider antworten erfolgreich."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="Antwort A"),
            "beta": _MockProvider("beta", response_text="Antwort B"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(
            question="Was ist Bitcoin?", user_id=1, chat_id=10
        )

        assert isinstance(result, DebateResult)
        assert result.question == "Was ist Bitcoin?"
        assert "alpha" in result.responses
        assert "beta" in result.responses
        assert result.responses["alpha"] == "Antwort A"
        assert result.responses["beta"] == "Antwort B"
        assert len(result.errors) == 0
        assert result.duration_seconds >= 0.0
        assert result.consensus_analysis is not None
        assert set(result.providers_queried) == {"alpha", "beta"}

    async def test_one_provider_crashes_other_succeeds(self) -> None:
        """Ein Provider crasht, der andere antwortet normal."""
        providers = {
            "good": _MockProvider("good", response_text="Gute Antwort"),
            "bad": _MockProvider("bad", should_raise=RuntimeError("Provider kaputt")),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert "good" in result.responses
        assert result.responses["good"] == "Gute Antwort"
        assert "bad" in result.errors
        assert "Provider kaputt" in result.errors["bad"]

    async def test_all_providers_crash(self) -> None:
        """Alle Provider crashen: leere responses, gefuellte errors."""
        providers = {
            "crash1": _MockProvider("crash1", should_raise=RuntimeError("Crash 1")),
            "crash2": _MockProvider("crash2", should_raise=RuntimeError("Crash 2")),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert len(result.responses) == 0
        assert "crash1" in result.errors
        assert "crash2" in result.errors

    async def test_timeout_respected(self) -> None:
        """Provider mit langem Delay wird per Timeout abgebrochen."""
        providers = {
            "fast": _MockProvider("fast", response_text="Schnell"),
            "slow": _MockProvider("slow", response_text="Nie", delay=100.0),
        }
        router = _make_router(providers)
        # Sehr kurzer Timeout
        orchestrator = DebateOrchestrator(provider_router=router, timeout_seconds=1)

        result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert "fast" in result.responses
        assert "slow" in result.errors
        assert (
            "Timeout" in result.errors["slow"]
            or "timeout" in result.errors["slow"].lower()
        )

    async def test_no_providers_available(self) -> None:
        """Keine Provider verfuegbar: system-Fehler."""
        providers = {
            "offline": _MockProvider("offline", available=False),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert len(result.responses) == 0
        assert "system" in result.errors
        assert result.providers_queried == []

    async def test_single_provider_consensus_note(self) -> None:
        """Nur ein Provider verfuegbar: Konsens-Analyse vermerkt das."""
        providers = {
            "solo": _MockProvider("solo", response_text="Einzelantwort"),
            "off": _MockProvider("off", available=False),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert "solo" in result.responses
        assert result.consensus_analysis is not None
        assert "Nur ein Provider" in result.consensus_analysis


class TestConsensusHeuristic:
    """Tests für die Konsens-Heuristik."""

    async def test_high_overlap_detected(self) -> None:
        """Ähnliche Antworten -> hohe Übereinstimmung."""
        providers = {
            "a": _MockProvider(
                "a",
                response_text=(
                    "Bitcoin ist eine dezentrale digitale Währung "
                    "die auf Blockchain-Technologie basiert"
                ),
            ),
            "b": _MockProvider(
                "b",
                response_text=(
                    "Bitcoin ist eine digitale dezentrale Kryptowährung "
                    "basierend auf der Blockchain-Technologie"
                ),
            ),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(
            question="Was ist Bitcoin?", user_id=1, chat_id=10
        )

        assert result.consensus_analysis is not None
        # Bei hohem Overlap sollte "überein" oder "Übereinstimmung" stehen
        assert (
            "überein" in result.consensus_analysis.lower()
            or "übereinstimmung" in result.consensus_analysis.lower()
        )

    async def test_low_overlap_detected(self) -> None:
        """Sehr unterschiedliche Antworten -> Dissens erkannt."""
        providers = {
            "a": _MockProvider(
                "a",
                response_text="Die Sonne scheint hell am Himmel ueber dem Meer",
            ),
            "b": _MockProvider(
                "b",
                response_text=(
                    "Quantencomputer nutzen Superposition fuer "
                    "parallele Berechnungen in Millisekunden"
                ),
            ),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(question="Irgendwas", user_id=1, chat_id=10)

        assert result.consensus_analysis is not None
        assert "unterschiedlich" in result.consensus_analysis.lower()


class TestDebateProviderConfig:
    """Tests fuer DEBATE_PROVIDERS-Konfiguration."""

    async def test_configured_providers_filter(self) -> None:
        """DEBATE_PROVIDERS env-var filtert Provider."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
            "beta": _MockProvider("beta", response_text="B"),
            "gamma": _MockProvider("gamma", response_text="C"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        with patch(
            "application.debate_orchestrator._get_configured_providers",
            return_value=["alpha", "gamma"],
        ):
            result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert "alpha" in result.responses
        assert "gamma" in result.responses
        assert "beta" not in result.responses

    async def test_configured_unavailable_provider_skipped(self) -> None:
        """Konfigurierter aber nicht-verfuegbarer Provider wird uebersprungen."""
        providers = {
            "online": _MockProvider("online", response_text="OK"),
            "offline": _MockProvider("offline", available=False),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        with patch(
            "application.debate_orchestrator._get_configured_providers",
            return_value=["online", "offline"],
        ):
            result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert "online" in result.responses
        assert "offline" not in result.responses
        assert "offline" not in result.errors


class TestProviderDeduplication:
    """Tests für Provider-Deduplizierung (R10-Fix).

    claude_persistent und claude nutzen beide die Claude CLI.
    Im Debate soll nur einer der beiden genutzt werden, um verzerrte
    Konsens-Analysen und Token-Verschwendung zu vermeiden.
    """

    def test_dedup_both_claude_available(self) -> None:
        """Beide Claude-Provider verfügbar: nur claude_persistent bleibt."""
        result = deduplicate_providers(["claude_persistent", "claude", "ollama_local"])
        assert result == ["claude_persistent", "ollama_local"]

    def test_dedup_only_legacy_claude(self) -> None:
        """Nur Legacy-Claude verfügbar: wird behalten."""
        result = deduplicate_providers(["claude", "ollama_local"])
        assert result == ["claude", "ollama_local"]

    def test_dedup_only_persistent(self) -> None:
        """Nur claude_persistent verfügbar: wird behalten."""
        result = deduplicate_providers(["claude_persistent"])
        assert result == ["claude_persistent"]

    def test_dedup_standalone_providers_untouched(self) -> None:
        """Standalone-Provider (nicht in einer Gruppe) werden nie entfernt."""
        result = deduplicate_providers(["ollama_local", "openai", "gemini"])
        assert result == ["ollama_local", "openai", "gemini"]

    def test_dedup_empty_list(self) -> None:
        """Leere Liste bleibt leer."""
        result = deduplicate_providers([])
        assert result == []

    def test_dedup_order_matters(self) -> None:
        """Erster Provider der Gruppe gewinnt (basierend auf Input-Reihenfolge)."""
        # Wenn claude VOR claude_persistent steht, gewinnt claude
        result = deduplicate_providers(["claude", "claude_persistent", "ollama_local"])
        assert result == ["claude", "ollama_local"]

    async def test_debate_deduplicates_claude_providers(self) -> None:
        """Im Debate-Flow: nur claude_persistent (nicht auch Legacy-Claude)."""
        providers = {
            "claude_persistent": _MockProvider(
                "claude_persistent", response_text="Persistent sagt: BTC ist P2P-Geld"
            ),
            "claude": _MockProvider(
                "claude", response_text="Legacy sagt: BTC ist digital cash"
            ),
            "ollama_local": _MockProvider(
                "ollama_local", response_text="Llama sagt: BTC ist Krypto"
            ),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(
            question="Was ist Bitcoin?", user_id=42, chat_id=100
        )

        # claude_persistent vertritt die Gruppe, claude wird übersprungen
        assert "claude_persistent" in result.responses
        assert "claude" not in result.responses
        assert "ollama_local" in result.responses
        assert len(result.errors) == 0
        assert result.consensus_analysis is not None
        # Nur 2 Provider wurden tatsächlich gefragt
        assert set(result.providers_queried) == {
            "claude_persistent",
            "ollama_local",
        }

    async def test_debate_falls_back_to_legacy_claude(self) -> None:
        """Wenn claude_persistent nicht verfügbar: Legacy-Claude wird genutzt."""
        providers = {
            "claude_persistent": _MockProvider("claude_persistent", available=False),
            "claude": _MockProvider("claude", response_text="Legacy antwortet"),
            "ollama_local": _MockProvider(
                "ollama_local", response_text="Llama antwortet"
            ),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(question="Test?", user_id=42, chat_id=100)

        # claude_persistent ist offline, claude vertritt die Gruppe
        assert "claude" in result.responses
        assert "claude_persistent" not in result.responses
        assert "ollama_local" in result.responses
