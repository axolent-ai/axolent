"""Tests für application.debate_orchestrator (R10: Multi-AI-Debate).

Testet:
- 2 Provider parallel, beide antworten erfolgreich
- 1 Provider crasht, anderer antwortet (errors-Dict enthält ihn)
- Alle Provider crashen (DebateResult mit leeren responses)
- Timeout wird respektiert
- Konsens-Heuristik gibt sinnvollen Output
- Kein Provider verfügbar
- Provider-Deduplizierung (claude_persistent + claude = eine Gruppe)
- Final Review: JSON-Parse-Erfolg
- Final Review: JSON-Parse-Fehler -> graceful Fallback
- Final Review: Provider-Namen anonymisiert im Judge-Prompt (Bias-Mitigation)
- Final Review: Integration in Debate-Flow
"""

from __future__ import annotations

import json
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
    """Mock-Provider für Tests."""

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
        """Keine Provider verfügbar: system-Fehler."""
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
        """Nur ein Provider verfügbar: Konsens-Analyse vermerkt das."""
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
                response_text="Die Sonne scheint hell am Himmel über dem Meer",
            ),
            "b": _MockProvider(
                "b",
                response_text=(
                    "Quantencomputer nutzen Superposition für "
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
    """Tests für DEBATE_PROVIDERS-Konfiguration."""

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
        """Konfigurierter aber nicht-verfügbarer Provider wird übersprungen."""
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


class TestFinalReviewParsing:
    """Tests für die JSON-Parsing-Logik des Final Review."""

    def test_parse_valid_json(self) -> None:
        """Valides JSON wird korrekt zu FinalVerdict geparst."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
            "beta": _MockProvider("beta", response_text="B"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        raw_json = json.dumps(
            {
                "winner": "A",
                "synthesis": "Alpha und Beta ergaenzen sich: A ist praezise, B ausfuehrlich.",
                "recommendation": "Antwort A ist präziser.",
                "evaluations": [
                    {"label": "A", "pros": ["Korrekt", "Präzise"], "cons": ["Kurz"]},
                    {"label": "B", "pros": ["Ausführlich"], "cons": ["Vage"]},
                ],
                "reasoning": "A liefert die genauere Antwort.",
            }
        )

        label_to_provider = {"A": "alpha", "B": "beta"}
        result = orchestrator._parse_judge_response(raw_json, label_to_provider)

        assert result is not None
        assert result.winner == "alpha"
        assert result.recommendation == "Antwort A ist präziser."
        assert (
            result.synthesis
            == "Alpha und Beta ergaenzen sich: A ist praezise, B ausfuehrlich."
        )
        assert result.reasoning == "A liefert die genauere Antwort."
        assert len(result.evaluations) == 2
        assert result.evaluations[0].provider == "alpha"
        assert "Korrekt" in result.evaluations[0].pros
        assert "Kurz" in result.evaluations[0].cons
        assert result.evaluations[1].provider == "beta"

    def test_parse_tie_winner(self) -> None:
        """Winner 'tie' wird korrekt erkannt."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        raw_json = json.dumps(
            {
                "winner": "tie",
                "synthesis": "Beide Antworten sind gleichwertig und decken denselben Inhalt ab.",
                "recommendation": "Beide gleichwertig.",
                "evaluations": [],
                "reasoning": "Keine signifikanten Unterschiede.",
            }
        )

        result = orchestrator._parse_judge_response(raw_json, {"A": "alpha"})
        assert result is not None
        assert result.winner == "tie"
        assert "gleichwertig" in result.synthesis

    def test_parse_json_in_markdown_codeblock(self) -> None:
        """JSON in Markdown-Codeblock wird korrekt extrahiert."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        raw_text = (
            "```json\n"
            '{"winner": "A", "synthesis": "A ist die beste Wahl.", '
            '"recommendation": "A gewinnt.", '
            '"evaluations": [], "reasoning": "Besser."}\n'
            "```"
        )

        result = orchestrator._parse_judge_response(raw_text, {"A": "alpha"})
        assert result is not None
        assert result.winner == "alpha"
        assert result.recommendation == "A gewinnt."
        assert result.synthesis == "A ist die beste Wahl."

    def test_parse_invalid_json_returns_none(self) -> None:
        """Ungültiges JSON gibt None zurück (graceful fallback)."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = orchestrator._parse_judge_response(
            "Das ist kein JSON, sorry!", {"A": "alpha"}
        )
        assert result is None

    def test_parse_non_dict_json_returns_none(self) -> None:
        """JSON das kein Dict ist gibt None zurück."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = orchestrator._parse_judge_response(
            '["not", "a", "dict"]', {"A": "alpha"}
        )
        assert result is None


class TestFinalReviewBiasMitigation:
    """Tests für Bias-Mitigation: Provider-Namen sind im Judge-Prompt anonymisiert."""

    def test_prompt_contains_no_provider_names(self) -> None:
        """Der Judge-Prompt enthält keine echten Provider-Namen."""
        providers = {
            "claude_persistent": _MockProvider(
                "claude_persistent", response_text="Claude sagt X"
            ),
            "ollama_local": _MockProvider("ollama_local", response_text="Llama sagt Y"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        responses = {
            "claude_persistent": "Claude sagt X",
            "ollama_local": "Llama sagt Y",
        }

        prompt, label_to_provider = orchestrator._build_judge_prompt(
            "Was ist Bitcoin?", responses
        )

        # Provider-Namen dürfen nicht im Prompt vorkommen
        assert "claude_persistent" not in prompt
        assert "ollama_local" not in prompt
        # Aber die anonymen Labels schon
        assert "Antwort A" in prompt
        assert "Antwort B" in prompt
        # Mapping muss korrekt sein
        assert label_to_provider["A"] == "claude_persistent"
        assert label_to_provider["B"] == "ollama_local"

    def test_prompt_contains_answer_content(self) -> None:
        """Der Judge-Prompt enthält den Antworttext."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        responses = {
            "alpha": "Bitcoin ist digitales Geld",
            "beta": "Bitcoin ist eine Kryptowährung",
        }

        prompt, _ = orchestrator._build_judge_prompt("Was ist Bitcoin?", responses)

        assert "Bitcoin ist digitales Geld" in prompt
        assert "Bitcoin ist eine Kryptowährung" in prompt
        assert "Was ist Bitcoin?" in prompt


class TestFinalReviewIntegration:
    """Integration-Tests: Final Review im Debate-Flow."""

    async def test_debate_includes_final_verdict(self) -> None:
        """Debate mit 2 Providern liefert ein FinalVerdict mit Synthese."""
        judge_json = json.dumps(
            {
                "winner": "A",
                "synthesis": "Bitcoin ist eine dezentrale digitale Währung auf Blockchain-Basis.",
                "recommendation": "Antwort A ist besser.",
                "evaluations": [
                    {"label": "A", "pros": ["Klar"], "cons": []},
                    {"label": "B", "pros": [], "cons": ["Unklar"]},
                ],
                "reasoning": "A ist präziser.",
            }
        )

        # Spezieller Mock-Provider der als Judge fungiert
        providers = {
            "claude_persistent": _MockProvider(
                "claude_persistent", response_text="Claude Antwort"
            ),
            "ollama_local": _MockProvider(
                "ollama_local", response_text="Llama Antwort"
            ),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        # Mock: Judge-Call liefert vorbereitetes JSON
        async def _mock_final_review(question, responses, user_id, chat_id):
            """Simuliert einen erfolgreichen Final Review."""
            prompt, label_to_provider = orchestrator._build_judge_prompt(
                question, responses
            )
            return orchestrator._parse_judge_response(judge_json, label_to_provider)

        # Patch final_review um den echten Provider-Call zu umgehen
        with patch.object(orchestrator, "final_review", side_effect=_mock_final_review):
            result = await orchestrator.debate(
                question="Was ist Bitcoin?", user_id=1, chat_id=10
            )

        assert result.final_verdict is not None
        assert result.final_verdict.winner == "claude_persistent"
        assert result.final_verdict.recommendation == "Antwort A ist besser."
        assert result.final_verdict.synthesis == (
            "Bitcoin ist eine dezentrale digitale Währung auf Blockchain-Basis."
        )
        assert len(result.final_verdict.evaluations) == 2

    async def test_debate_graceful_when_judge_fails(self) -> None:
        """Wenn der Judge fehlschlägt, hat result trotzdem consensus_analysis."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="Antwort Alpha"),
            "beta": _MockProvider("beta", response_text="Antwort Beta"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        # Mock: Judge-Call liefert None (Fehler)
        with patch.object(orchestrator, "final_review", return_value=None):
            result = await orchestrator.debate(
                question="Was ist Bitcoin?", user_id=1, chat_id=10
            )

        # Kein Verdict, aber Konsens-Analyse existiert
        assert result.final_verdict is None
        assert result.consensus_analysis is not None
        # Antworten sind trotzdem da
        assert "alpha" in result.responses
        assert "beta" in result.responses

    async def test_final_review_uses_claude_persistent_as_judge(self) -> None:
        """Final Review bevorzugt claude_persistent als Judge."""
        judge_json = json.dumps(
            {
                "winner": "A",
                "synthesis": "A liefert die klarere Antwort mit korrekten Fakten.",
                "recommendation": "A gewinnt.",
                "evaluations": [
                    {"label": "A", "pros": ["Gut"], "cons": []},
                    {"label": "B", "pros": [], "cons": ["Schlecht"]},
                ],
                "reasoning": "A ist besser.",
            }
        )

        providers = {
            "claude_persistent": _MockProvider(
                "claude_persistent", response_text=judge_json
            ),
            "ollama_local": _MockProvider(
                "ollama_local", response_text="Sollte nicht als Judge genutzt werden"
            ),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        responses = {
            "claude_persistent": "Claude sagt was",
            "ollama_local": "Llama sagt was",
        }

        verdict = await orchestrator.final_review(
            question="Test?",
            responses=responses,
            user_id=1,
            chat_id=10,
        )

        assert verdict is not None
        assert verdict.judge_provider == "claude_persistent"
        assert verdict.judge_quality_warning is None

    async def test_final_review_fallback_to_ollama_with_warning(self) -> None:
        """Wenn nur ollama_local verfügbar: wird als Judge mit Warnung genutzt."""
        judge_json = json.dumps(
            {
                "winner": "A",
                "synthesis": "A bietet die bessere Zusammenfassung.",
                "recommendation": "A gewinnt.",
                "evaluations": [],
                "reasoning": "A ist besser.",
            }
        )

        providers = {
            "claude_persistent": _MockProvider("claude_persistent", available=False),
            "ollama_local": _MockProvider("ollama_local", response_text=judge_json),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        responses = {"alpha": "Antwort A", "beta": "Antwort B"}

        verdict = await orchestrator.final_review(
            question="Test?",
            responses=responses,
            user_id=1,
            chat_id=10,
        )

        assert verdict is not None
        assert verdict.judge_provider == "ollama_local"
        assert verdict.judge_quality_warning is not None
        assert "Lokaler Judge" in verdict.judge_quality_warning

    async def test_final_review_skipped_with_single_response(self) -> None:
        """Final Review wird übersprungen wenn weniger als 2 Antworten."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        verdict = await orchestrator.final_review(
            question="Test?",
            responses={"alpha": "Nur eine Antwort"},
            user_id=1,
            chat_id=10,
        )

        assert verdict is None

    async def test_final_review_returns_none_on_invalid_judge_json(self) -> None:
        """Wenn der Judge ungültiges JSON liefert: None statt Crash."""
        providers = {
            "claude_persistent": _MockProvider(
                "claude_persistent",
                response_text="Sorry, ich kann das nicht als JSON.",
            ),
            "ollama_local": _MockProvider("ollama_local", response_text="X"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        responses = {"alpha": "Antwort A", "beta": "Antwort B"}

        verdict = await orchestrator.final_review(
            question="Test?",
            responses=responses,
            user_id=1,
            chat_id=10,
        )

        assert verdict is None


class TestSynthesisFeature:
    """Tests für Phase-1 Synthesis: Judge generiert eine inhaltliche Synthese."""

    def test_parse_synthesis_field_extracted(self) -> None:
        """Synthesis-Feld wird korrekt aus JSON extrahiert."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        raw_json = json.dumps(
            {
                "winner": "A",
                "synthesis": (
                    "Bitcoin ist eine dezentrale digitale Währung "
                    "die auf Blockchain-Technologie basiert und "
                    "Peer-to-Peer-Transaktionen ohne Mittelsmann ermöglicht."
                ),
                "recommendation": "A ist praeziser.",
                "evaluations": [
                    {"label": "A", "pros": ["Korrekt"], "cons": []},
                ],
                "reasoning": "A deckt alle Kernpunkte ab.",
            }
        )

        label_to_provider = {"A": "alpha"}
        result = orchestrator._parse_judge_response(raw_json, label_to_provider)

        assert result is not None
        assert "dezentrale digitale Währung" in result.synthesis
        assert "Peer-to-Peer" in result.synthesis

    def test_parse_missing_synthesis_defaults_empty(self) -> None:
        """Wenn Judge kein synthesis-Feld liefert: Default ist leerer String."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        # JSON ohne synthesis-Feld (Backward-Compat)
        raw_json = json.dumps(
            {
                "winner": "A",
                "recommendation": "A gewinnt.",
                "evaluations": [],
                "reasoning": "A ist besser.",
            }
        )

        label_to_provider = {"A": "alpha"}
        result = orchestrator._parse_judge_response(raw_json, label_to_provider)

        assert result is not None
        assert result.synthesis == ""

    def test_judge_prompt_requests_synthesis(self) -> None:
        """Der Judge-Prompt fordert explizit eine Synthese an."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
            "beta": _MockProvider("beta", response_text="B"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        responses = {"alpha": "Antwort A", "beta": "Antwort B"}
        prompt, _ = orchestrator._build_judge_prompt("Was ist Bitcoin?", responses)

        # Prompt muss Synthesis-Anweisung enthalten
        assert "SYNTHESE" in prompt
        assert "synthesis" in prompt
        # Prompt muss das neue Schema-Feld beinhalten
        assert "Vollständige synthetisierte Antwort" in prompt

    def test_synthesis_preserved_through_final_review_reconstruction(self) -> None:
        """Synthesis bleibt erhalten wenn FinalVerdict in final_review() rekonstruiert wird."""
        from application.debate_orchestrator import FinalVerdict, ProviderEvaluation

        # Simuliere was _parse_judge_response zurückgibt
        parsed = FinalVerdict(
            winner="alpha",
            recommendation="Alpha ist besser.",
            synthesis="Die Synthese vereint beide Perspektiven zu einer klaren Antwort.",
            evaluations=[ProviderEvaluation(provider="alpha", pros=["Gut"], cons=[])],
            reasoning="Alpha ist praeziser.",
        )

        # Simuliere was final_review() daraus macht (neues frozen Objekt mit Metadaten)
        reconstructed = FinalVerdict(
            winner=parsed.winner,
            recommendation=parsed.recommendation,
            synthesis=parsed.synthesis,
            evaluations=parsed.evaluations,
            reasoning=parsed.reasoning,
            judge_provider="claude_persistent",
            judge_quality_warning=None,
        )

        assert reconstructed.synthesis == parsed.synthesis
        assert reconstructed.judge_provider == "claude_persistent"


class TestRobustJsonExtraction:
    """Tests für robuste JSON-Extraktion aus Judge-Responses.

    Bug-Kontext: Im Live-Test lieferte der Judge-Call Text mit umgebendem Prosa
    oder Markdown-Wrapping, was den alten Parser zum Scheitern brachte und
    den Konsens-Fallback triggerte statt der Synthese.
    """

    def _make_orchestrator(self) -> DebateOrchestrator:
        providers = {"alpha": _MockProvider("alpha", response_text="A")}
        router = _make_router(providers)
        return DebateOrchestrator(provider_router=router)

    def test_extract_json_with_prose_before(self) -> None:
        """JSON mit erklaertendem Text davor wird korrekt extrahiert."""
        orchestrator = self._make_orchestrator()

        raw_text = (
            "Here is my evaluation:\n\n"
            '{"winner": "A", "synthesis": "A bietet die klarere Antwort.", '
            '"recommendation": "A gewinnt.", '
            '"evaluations": [{"label": "A", "pros": ["Klar"], "cons": []}], '
            '"reasoning": "A ist praeziser."}'
        )

        result = orchestrator._parse_judge_response(raw_text, {"A": "alpha"})
        assert result is not None
        assert result.winner == "alpha"
        assert result.synthesis == "A bietet die klarere Antwort."

    def test_extract_json_with_prose_before_and_after(self) -> None:
        """JSON mit Text davor UND danach wird korrekt extrahiert."""
        orchestrator = self._make_orchestrator()

        raw_text = (
            "Meine Bewertung der Antworten:\n\n"
            '{"winner": "B", "synthesis": "B liefert die vollständigere Antwort.", '
            '"recommendation": "B gewinnt.", '
            '"evaluations": [{"label": "B", "pros": ["Detailliert"], "cons": []}], '
            '"reasoning": "B ist ausfuehrlicher."}\n\n'
            "Ich hoffe das hilft!"
        )

        label_map = {"A": "alpha", "B": "beta"}
        result = orchestrator._parse_judge_response(raw_text, label_map)
        assert result is not None
        assert result.winner == "beta"
        assert "vollständigere" in result.synthesis

    def test_extract_json_in_codeblock_with_prose_prefix(self) -> None:
        """JSON in Markdown-Codeblock, NACH erklaertendem Text."""
        orchestrator = self._make_orchestrator()

        raw_text = (
            "Here's my analysis in JSON format:\n\n"
            "```json\n"
            '{"winner": "A", "synthesis": "Synthese-Text hier.", '
            '"recommendation": "Empfehlung.", '
            '"evaluations": [], "reasoning": "Weil A."}\n'
            "```\n"
        )

        result = orchestrator._parse_judge_response(raw_text, {"A": "alpha"})
        assert result is not None
        assert result.winner == "alpha"
        assert result.synthesis == "Synthese-Text hier."

    def test_extract_json_with_nested_braces_in_strings(self) -> None:
        """JSON das geschweifte Klammern in String-Werten enthaelt."""
        orchestrator = self._make_orchestrator()

        raw_text = json.dumps(
            {
                "winner": "A",
                "synthesis": "Die Antwort nutzt {Platzhalter} korrekt.",
                "recommendation": "A gewinnt.",
                "evaluations": [],
                "reasoning": "A ist besser.",
            }
        )

        result = orchestrator._parse_judge_response(raw_text, {"A": "alpha"})
        assert result is not None
        assert "{Platzhalter}" in result.synthesis

    def test_extract_json_multiline_pretty_printed(self) -> None:
        """Pretty-printed JSON (mehrzeilig mit Einrueckung)."""
        orchestrator = self._make_orchestrator()

        raw_text = json.dumps(
            {
                "winner": "A",
                "synthesis": "Multi-line synthesis test.",
                "recommendation": "A gewinnt.",
                "evaluations": [
                    {"label": "A", "pros": ["Pro1", "Pro2"], "cons": ["Con1"]},
                    {"label": "B", "pros": ["Pro3"], "cons": ["Con2", "Con3"]},
                ],
                "reasoning": "A deckt mehr ab.",
            },
            indent=2,
        )

        label_map = {"A": "alpha", "B": "beta"}
        result = orchestrator._parse_judge_response(raw_text, label_map)
        assert result is not None
        assert result.winner == "alpha"
        assert len(result.evaluations) == 2
        assert "Pro1" in result.evaluations[0].pros

    def test_extract_no_json_at_all(self) -> None:
        """Kein JSON im Text: gibt None zurück."""
        orchestrator = self._make_orchestrator()
        result = orchestrator._parse_judge_response(
            "Ich kann das leider nicht bewerten. Bitte versuche es erneut.",
            {"A": "alpha"},
        )
        assert result is None

    def test_extract_json_array_not_object(self) -> None:
        """JSON-Array mit nur einem Element: Brace-Matcher extrahiert inneres Dict."""
        orchestrator = self._make_orchestrator()
        # Der Parser findet das erste { innerhalb des Arrays und extrahiert das Dict.
        # Das ist akzeptables Verhalten: besser ein Teilresultat als gar keins.
        result = orchestrator._parse_judge_response('[{"winner": "A"}]', {"A": "alpha"})
        # Extrahiert das innere Dict, winner wird gemapped
        assert result is not None
        assert result.winner == "alpha"

    def test_pure_array_no_dict_gives_none(self) -> None:
        """Reines JSON-Array ohne brauchbares Dict: gibt None zurück."""
        orchestrator = self._make_orchestrator()
        result = orchestrator._parse_judge_response(
            '["not", "a", "dict"]', {"A": "alpha"}
        )
        assert result is None

    def test_static_extract_json_object_method(self) -> None:
        """Direkter Test der statischen _extract_json_object Methode."""
        # Reines JSON
        assert DebateOrchestrator._extract_json_object('{"a": 1}') == '{"a": 1}'

        # Mit Prefix
        result = DebateOrchestrator._extract_json_object('Hello\n{"a": 1}')
        assert result is not None
        assert json.loads(result) == {"a": 1}

        # Kein JSON
        assert DebateOrchestrator._extract_json_object("no json here") is None

        # Leerer String
        assert DebateOrchestrator._extract_json_object("") is None


class TestMultiQuestionCoverage:
    """Regressions-Tests: Multi-Fragen müssen alle Aspekte in der Kernaussage abdecken.

    Bug-Kontext: Bei Fragen wie 'Was ist Bitcoin UND sollte ich einsteigen?'
    deckte die Kernaussage bisher nur den Handlungs-Aspekt ab. Der Definitions-Teil
    fiel hinten runter. Der geschaerfte Judge-Prompt verlangt jetzt explizit dass
    alle Teilaspekte adressiert werden.
    """

    def _make_orchestrator(self) -> DebateOrchestrator:
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
            "beta": _MockProvider("beta", response_text="B"),
        }
        router = _make_router(providers)
        return DebateOrchestrator(provider_router=router)

    def test_judge_prompt_contains_multi_question_instruction(self) -> None:
        """Der Judge-Prompt enthaelt die Multi-Fragen-Anweisung."""
        orchestrator = self._make_orchestrator()

        responses = {
            "alpha": "Bitcoin ist eine digitale Währung.",
            "beta": "Du solltest vorsichtig investieren.",
        }
        prompt, _ = orchestrator._build_judge_prompt(
            "Was ist Bitcoin und sollte ich einsteigen?", responses
        )

        assert "ALLE Aspekte der Frage abdecken" in prompt
        assert "Multi-Fragen" in prompt
        assert "Teilaspekt" in prompt

    def test_judge_prompt_requests_2_4_sentences(self) -> None:
        """Das JSON-Schema verlangt 2-4 Sätze für die Kernaussage."""
        orchestrator = self._make_orchestrator()

        responses = {"alpha": "A", "beta": "B"}
        prompt, _ = orchestrator._build_judge_prompt("Test?", responses)

        assert "2-4 Sätze" in prompt

    def test_multi_question_verdict_covers_both_aspects(self) -> None:
        """Mock-Judge-Response zu Multi-Frage deckt beide Aspekte ab."""
        orchestrator = self._make_orchestrator()

        # Simuliere eine gute Judge-Response die beide Aspekte abdeckt
        raw_json = json.dumps(
            {
                "winner": "A",
                "synthesis": (
                    "Bitcoin ist eine dezentrale digitale Währung auf Blockchain-Basis. "
                    "Ein Einstieg ist mit Risiken verbunden, kleine Positionen als Start "
                    "sind empfehlenswert."
                ),
                "recommendation": (
                    "Bitcoin ist eine dezentrale Kryptowährung die auf "
                    "Blockchain-Technologie basiert. Ein Investment ist möglich, "
                    "aber nur mit Geld das man bereit ist zu verlieren."
                ),
                "evaluations": [
                    {
                        "label": "A",
                        "pros": ["Gute Definition"],
                        "cons": ["Kein Investment-Rat"],
                    },
                    {
                        "label": "B",
                        "pros": ["Guter Rat"],
                        "cons": ["Keine Definition"],
                    },
                ],
                "reasoning": "A liefert die bessere Basis-Erklärung.",
            }
        )

        label_to_provider = {"A": "alpha", "B": "beta"}
        verdict = orchestrator._parse_judge_response(raw_json, label_to_provider)

        assert verdict is not None
        # Kernaussage deckt Definition ab (Bitcoin/Kryptowährung/Blockchain)
        assert "Bitcoin" in verdict.recommendation
        assert (
            "Blockchain" in verdict.recommendation or "Krypto" in verdict.recommendation
        )
        # Kernaussage deckt Investment-Aspekt ab
        assert (
            "Investment" in verdict.recommendation
            or "verlieren" in verdict.recommendation
        )

    async def test_multi_question_debate_integration(self) -> None:
        """Integration: Multi-Frage Debate mit Mock-Judge liefert vollständige Kernaussage."""
        judge_json = json.dumps(
            {
                "winner": "A",
                "synthesis": (
                    "Bitcoin ist dezentrales digitales Geld. "
                    "Ein Einstieg sollte mit kleinen Betraegen beginnen."
                ),
                "recommendation": (
                    "Bitcoin ist eine dezentrale Kryptowährung auf Blockchain-Basis. "
                    "Ein Einstieg kann sinnvoll sein, aber nur mit Risikokapital."
                ),
                "evaluations": [
                    {"label": "A", "pros": ["Praezise"], "cons": []},
                    {"label": "B", "pros": ["Praktisch"], "cons": []},
                ],
                "reasoning": "A ist praeziser.",
            }
        )

        providers = {
            "claude_persistent": _MockProvider(
                "claude_persistent",
                response_text=(
                    "Bitcoin ist eine dezentrale digitale Währung "
                    "die auf Blockchain-Technologie basiert."
                ),
            ),
            "ollama_local": _MockProvider(
                "ollama_local",
                response_text=(
                    "Du solltest vorsichtig einsteigen, nur mit Geld "
                    "das du bereit bist zu verlieren."
                ),
            ),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        async def _mock_final_review(question, responses, user_id, chat_id):
            prompt, label_to_provider = orchestrator._build_judge_prompt(
                question, responses
            )
            return orchestrator._parse_judge_response(judge_json, label_to_provider)

        with patch.object(orchestrator, "final_review", side_effect=_mock_final_review):
            result = await orchestrator.debate(
                question="Was ist Bitcoin und sollte ich einsteigen?",
                user_id=1,
                chat_id=10,
            )

        assert result.final_verdict is not None
        key_takeaway = result.final_verdict.recommendation
        # Beide Aspekte müssen in der Kernaussage auftauchen
        assert "Bitcoin" in key_takeaway
        assert "Blockchain" in key_takeaway or "Krypto" in key_takeaway
        assert "Einstieg" in key_takeaway or "Risikokapital" in key_takeaway
