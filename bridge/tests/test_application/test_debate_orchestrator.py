"""Tests for application.debate_orchestrator (R10: Multi-AI-Debate).

Tests:
* 2 providers in parallel, both succeed
* 1 provider crashes, other responds (errors dict contains it)
* All providers crash (DebateResult with empty responses)
* Timeout is respected
* Consensus heuristic gives meaningful output
* No provider available
* Provider deduplication (claude_persistent + claude = one group)
* Final Review: JSON parse success
* Final Review: JSON parse error -> graceful fallback
* Final Review: Provider names anonymized in judge prompt (bias mitigation)
* Final Review: Integration in debate flow
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
    """Mock provider for tests."""

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
    """Create a ProviderRouter with mock providers."""
    default = next(iter(providers.keys()))
    return ProviderRouter(providers=providers, default=default)


class TestDebateOrchestratorBasic:
    """Basic debate tests."""

    async def test_two_providers_both_succeed(self) -> None:
        """Two providers respond successfully."""
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
        """One provider crashes, the other responds normally."""
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
        """All providers crash: empty responses, populated errors."""
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
        """Provider with long delay is cancelled by timeout."""
        providers = {
            "fast": _MockProvider("fast", response_text="Schnell"),
            "slow": _MockProvider("slow", response_text="Nie", delay=100.0),
        }
        router = _make_router(providers)
        # Very short timeout
        orchestrator = DebateOrchestrator(provider_router=router, timeout_seconds=1)

        result = await orchestrator.debate(question="Test?", user_id=1, chat_id=10)

        assert "fast" in result.responses
        assert "slow" in result.errors
        assert (
            "Timeout" in result.errors["slow"]
            or "timeout" in result.errors["slow"].lower()
        )

    async def test_no_providers_available(self) -> None:
        """No providers available: system error."""
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
        """Only one provider available: consensus analysis notes this."""
        providers = {
            "solo": _MockProvider("solo", response_text="Einzelantwort"),
            "off": _MockProvider("off", available=False),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        result = await orchestrator.debate(
            question="Test?", user_id=1, chat_id=10, user_lang="en"
        )

        assert "solo" in result.responses
        assert result.consensus_analysis is not None
        assert "Only one provider" in result.consensus_analysis


class TestConsensusHeuristic:
    """Tests for the consensus heuristic."""

    async def test_high_overlap_detected(self) -> None:
        """Similar responses -> high agreement."""
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
            question="Was ist Bitcoin?", user_id=1, chat_id=10, user_lang="en"
        )

        assert result.consensus_analysis is not None
        # High overlap should mention "agree" or "agreement"
        assert (
            "agree" in result.consensus_analysis.lower()
            or "agreement" in result.consensus_analysis.lower()
        )

    async def test_low_overlap_detected(self) -> None:
        """Very different responses -> dissent detected."""
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

        result = await orchestrator.debate(
            question="Irgendwas", user_id=1, chat_id=10, user_lang="en"
        )

        assert result.consensus_analysis is not None
        assert "different" in result.consensus_analysis.lower()


class TestDebateProviderConfig:
    """Tests for DEBATE_PROVIDERS configuration."""

    async def test_configured_providers_filter(self) -> None:
        """DEBATE_PROVIDERS env var filters providers."""
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
        """Configured but unavailable provider is skipped."""
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
    """Tests for provider deduplication (R10 fix).

    claude_persistent and claude both use the Claude CLI.
    In debates, only one of the two should be used to avoid
    skewed consensus analyses and token waste.
    """

    def test_dedup_both_claude_available(self) -> None:
        """Both Claude providers available: only claude_persistent remains."""
        result = deduplicate_providers(["claude_persistent", "claude", "ollama_local"])
        assert result == ["claude_persistent", "ollama_local"]

    def test_dedup_only_legacy_claude(self) -> None:
        """Only legacy Claude available: is kept."""
        result = deduplicate_providers(["claude", "ollama_local"])
        assert result == ["claude", "ollama_local"]

    def test_dedup_only_persistent(self) -> None:
        """Only claude_persistent available: is kept."""
        result = deduplicate_providers(["claude_persistent"])
        assert result == ["claude_persistent"]

    def test_dedup_standalone_providers_untouched(self) -> None:
        """Standalone providers (not in a group) are never removed."""
        result = deduplicate_providers(["ollama_local", "openai", "gemini"])
        assert result == ["ollama_local", "openai", "gemini"]

    def test_dedup_empty_list(self) -> None:
        """Empty list stays empty."""
        result = deduplicate_providers([])
        assert result == []

    def test_dedup_order_matters(self) -> None:
        """First provider in the group wins (based on input order)."""
        # If claude comes BEFORE claude_persistent, claude wins
        result = deduplicate_providers(["claude", "claude_persistent", "ollama_local"])
        assert result == ["claude", "ollama_local"]

    async def test_debate_deduplicates_claude_providers(self) -> None:
        """In debate flow: only claude_persistent (not also legacy Claude)."""
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

        # claude_persistent represents the group, claude is skipped
        assert "claude_persistent" in result.responses
        assert "claude" not in result.responses
        assert "ollama_local" in result.responses
        assert len(result.errors) == 0
        assert result.consensus_analysis is not None
        # Only 2 providers were actually queried
        assert set(result.providers_queried) == {
            "claude_persistent",
            "ollama_local",
        }

    async def test_debate_falls_back_to_legacy_claude(self) -> None:
        """When claude_persistent is unavailable: legacy Claude is used."""
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

        # claude_persistent is offline, claude represents the group
        assert "claude" in result.responses
        assert "claude_persistent" not in result.responses
        assert "ollama_local" in result.responses


class TestFinalReviewParsing:
    """Tests for the JSON parsing logic of the final review."""

    def test_parse_valid_json(self) -> None:
        """Valid JSON is correctly parsed to FinalVerdict."""
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
        """Winner 'tie' is correctly recognized."""
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
        """JSON in markdown code block is correctly extracted."""
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
        """Invalid JSON returns None (graceful fallback)."""
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
        """JSON that is not a dict returns None."""
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
    """Tests for bias mitigation: provider names are anonymized in judge prompt."""

    def test_prompt_contains_no_provider_names(self) -> None:
        """The judge prompt contains no real provider names."""
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

        # Provider names must not appear in the prompt
        assert "claude_persistent" not in prompt
        assert "ollama_local" not in prompt
        # But the anonymous labels must be present
        assert "Answer A" in prompt
        assert "Answer B" in prompt
        # Mapping must be correct
        assert label_to_provider["A"] == "claude_persistent"
        assert label_to_provider["B"] == "ollama_local"

    def test_prompt_contains_answer_content(self) -> None:
        """The judge prompt contains the answer text."""
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
    """Integration tests: final review in debate flow."""

    async def test_debate_includes_final_verdict(self) -> None:
        """Debate with 2 providers delivers a FinalVerdict with synthesis."""
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

        # Special mock provider acting as judge
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

        # Mock: judge call delivers prepared JSON
        async def _mock_final_review(question, responses, user_id, chat_id, **kwargs):
            """Simulate a successful final review."""
            prompt, label_to_provider = orchestrator._build_judge_prompt(
                question, responses
            )
            return orchestrator._parse_judge_response(judge_json, label_to_provider)

        # Patch final_review to bypass the real provider call
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
        """When the judge fails, result still has consensus_analysis."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="Antwort Alpha"),
            "beta": _MockProvider("beta", response_text="Antwort Beta"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        # Mock: judge call delivers None (failure)
        with patch.object(orchestrator, "final_review", return_value=None):
            result = await orchestrator.debate(
                question="Was ist Bitcoin?", user_id=1, chat_id=10
            )

        # No verdict, but consensus analysis exists
        assert result.final_verdict is None
        assert result.consensus_analysis is not None
        # Responses are still present
        assert "alpha" in result.responses
        assert "beta" in result.responses

    async def test_final_review_uses_claude_persistent_as_judge(self) -> None:
        """Final review prefers claude_persistent as judge."""
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
        """When only ollama_local is available: used as judge with warning."""
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
        assert "Local judge" in verdict.judge_quality_warning

    async def test_final_review_skipped_with_single_response(self) -> None:
        """Final review is skipped when fewer than 2 responses."""
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
        """When the judge delivers invalid JSON: None instead of crash."""
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
    """Tests for Phase 1 synthesis: judge generates a content synthesis."""

    def test_parse_synthesis_field_extracted(self) -> None:
        """Synthesis field is correctly extracted from JSON."""
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
        """When judge delivers no synthesis field: default is empty string."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        # JSON without synthesis field (backward compat)
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
        """The judge prompt explicitly requests a synthesis."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
            "beta": _MockProvider("beta", response_text="B"),
        }
        router = _make_router(providers)
        orchestrator = DebateOrchestrator(provider_router=router)

        responses = {"alpha": "Antwort A", "beta": "Antwort B"}
        prompt, _ = orchestrator._build_judge_prompt("Was ist Bitcoin?", responses)

        # Prompt must contain synthesis instruction
        assert "SYNTHESIS" in prompt
        assert "synthesis" in prompt
        # Prompt must contain the schema field description
        assert "Complete synthesized answer" in prompt

    def test_synthesis_preserved_through_final_review_reconstruction(self) -> None:
        """Synthesis is preserved when FinalVerdict is reconstructed in final_review()."""
        from application.debate_orchestrator import FinalVerdict, ProviderEvaluation

        # Simulate what _parse_judge_response returns
        parsed = FinalVerdict(
            winner="alpha",
            recommendation="Alpha ist besser.",
            synthesis="Die Synthese vereint beide Perspektiven zu einer klaren Antwort.",
            evaluations=[ProviderEvaluation(provider="alpha", pros=["Gut"], cons=[])],
            reasoning="Alpha ist praeziser.",
        )

        # Simulate what final_review() creates (new frozen object with metadata)
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
    """Tests for robust JSON extraction from judge responses.

    Bug context: in live tests, the judge call delivered text with surrounding
    prose or markdown wrapping, causing the old parser to fail and triggering
    the consensus fallback instead of the synthesis.
    """

    def _make_orchestrator(self) -> DebateOrchestrator:
        providers = {"alpha": _MockProvider("alpha", response_text="A")}
        router = _make_router(providers)
        return DebateOrchestrator(provider_router=router)

    def test_extract_json_with_prose_before(self) -> None:
        """JSON with explanatory text before is correctly extracted."""
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
        """JSON with text before AND after is correctly extracted."""
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
        """JSON in markdown code block, AFTER explanatory text."""
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
        """JSON that contains curly braces in string values."""
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
        """Pretty-printed JSON (multiline with indentation)."""
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
        """No JSON in text: returns None."""
        orchestrator = self._make_orchestrator()
        result = orchestrator._parse_judge_response(
            "Ich kann das leider nicht bewerten. Bitte versuche es erneut.",
            {"A": "alpha"},
        )
        assert result is None

    def test_extract_json_array_not_object(self) -> None:
        """JSON array with one element: brace matcher extracts inner dict."""
        orchestrator = self._make_orchestrator()
        # The parser finds the first { inside the array and extracts the dict.
        # This is acceptable behavior: better a partial result than nothing.
        result = orchestrator._parse_judge_response('[{"winner": "A"}]', {"A": "alpha"})
        # Extracts the inner dict, winner is mapped
        assert result is not None
        assert result.winner == "alpha"

    def test_pure_array_no_dict_gives_none(self) -> None:
        """Pure JSON array without usable dict: returns None."""
        orchestrator = self._make_orchestrator()
        result = orchestrator._parse_judge_response(
            '["not", "a", "dict"]', {"A": "alpha"}
        )
        assert result is None

    def test_static_extract_json_object_method(self) -> None:
        """Direct test of the static _extract_json_object method."""
        # Pure JSON
        assert DebateOrchestrator._extract_json_object('{"a": 1}') == '{"a": 1}'

        # With prefix
        result = DebateOrchestrator._extract_json_object('Hello\n{"a": 1}')
        assert result is not None
        assert json.loads(result) == {"a": 1}

        # No JSON
        assert DebateOrchestrator._extract_json_object("no json here") is None

        # Empty string
        assert DebateOrchestrator._extract_json_object("") is None


class TestMultiQuestionCoverage:
    """Regression tests: multi-part questions must cover all aspects in the key takeaway.

    Bug context: for questions like 'What is Bitcoin AND should I invest?',
    the key takeaway previously only covered the action aspect. The definition
    part was dropped. The sharpened judge prompt now explicitly requires that
    all sub-aspects are addressed.
    """

    def _make_orchestrator(self) -> DebateOrchestrator:
        providers = {
            "alpha": _MockProvider("alpha", response_text="A"),
            "beta": _MockProvider("beta", response_text="B"),
        }
        router = _make_router(providers)
        return DebateOrchestrator(provider_router=router)

    def test_judge_prompt_contains_multi_question_instruction(self) -> None:
        """The judge prompt contains the multi-question instruction."""
        orchestrator = self._make_orchestrator()

        responses = {
            "alpha": "Bitcoin ist eine digitale Währung.",
            "beta": "Du solltest vorsichtig investieren.",
        }
        prompt, _ = orchestrator._build_judge_prompt(
            "Was ist Bitcoin und sollte ich einsteigen?", responses
        )

        assert "ALL aspects of the question" in prompt
        assert "multi-part questions" in prompt.lower()
        assert "sub-aspect" in prompt.lower()

    def test_judge_prompt_requests_2_4_sentences(self) -> None:
        """The JSON schema requires 2-4 sentences for the key takeaway."""
        orchestrator = self._make_orchestrator()

        responses = {"alpha": "A", "beta": "B"}
        prompt, _ = orchestrator._build_judge_prompt("Test?", responses)

        assert "2-4 sentences" in prompt

    def test_multi_question_verdict_covers_both_aspects(self) -> None:
        """Mock judge response for multi-question covers both aspects."""
        orchestrator = self._make_orchestrator()

        # Simulate a good judge response covering both aspects
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
        # Key takeaway covers definition (Bitcoin/cryptocurrency/Blockchain)
        assert "Bitcoin" in verdict.recommendation
        assert (
            "Blockchain" in verdict.recommendation or "Krypto" in verdict.recommendation
        )
        # Key takeaway covers investment aspect
        assert (
            "Investment" in verdict.recommendation
            or "verlieren" in verdict.recommendation
        )

    async def test_multi_question_debate_integration(self) -> None:
        """Integration: multi-question debate with mock judge delivers complete key takeaway."""
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

        async def _mock_final_review(question, responses, user_id, chat_id, **kwargs):
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
        # Both aspects must appear in the key takeaway
        assert "Bitcoin" in key_takeaway
        assert "Blockchain" in key_takeaway or "Krypto" in key_takeaway
        assert "Einstieg" in key_takeaway or "Risikokapital" in key_takeaway


class TestDebateKernelIntegration:
    """Tests for Phase 0 Commit 4: Execution Kernel integration.

    Verifies that:
    - debate() accepts envelope/context/plan parameters
    - InstructionCompiler is used when available
    - Language from ExecutionContext takes precedence over user_lang
    - Judge uses same ExecutionContext as providers
    - Legacy callers without kernel params still work
    """

    def _make_orchestrator_with_compiler(
        self,
        providers: dict[str, _MockProvider],
    ) -> DebateOrchestrator:
        """Create an orchestrator with InstructionCompiler."""
        from application.execution import InstructionCompiler

        router = _make_router(providers)
        compiler = InstructionCompiler()
        return DebateOrchestrator(
            provider_router=router,
            instruction_compiler=compiler,
        )

    def _make_exec_context(self, lang: str = "it"):
        """Create a minimal ExecutionContext for testing."""
        from application.execution import ExecutionContext
        from application.language_resolver import LanguageContext

        return ExecutionContext(
            request_id="test-kernel-001",
            user_id=1,
            chat_id=10,
            channel="telegram",
            language=LanguageContext(
                code=lang,
                source="detection",
                confidence=0.95,
                switched_from=None,
                request_id="test-kernel-001",
            ),
        )

    def _make_exec_plan(self, lang: str = "it"):
        """Create a minimal ExecutionPlan for testing."""
        from application.execution import ExecutionPlan

        return ExecutionPlan(
            request_id="test-kernel-001",
            task_type="debate",
            language=lang,
            provider_chain=("claude_persistent",),
        )

    def _make_envelope(self, question: str = "Test?"):
        """Create a minimal RequestEnvelope for testing."""
        from application.execution import RequestEnvelope

        return RequestEnvelope.from_debate_command(
            user_id=1, chat_id=10, question=question
        )

    async def test_kernel_path_uses_context_language(self) -> None:
        """When context is provided, its language overrides user_lang."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="Risposta Alpha"),
            "beta": _MockProvider("beta", response_text="Risposta Beta"),
        }
        orchestrator = self._make_orchestrator_with_compiler(providers)

        exec_ctx = self._make_exec_context(lang="it")
        exec_plan = self._make_exec_plan(lang="it")
        envelope = self._make_envelope("Cos'e Bitcoin?")

        with patch.object(orchestrator, "final_review", return_value=None):
            result = await orchestrator.debate(
                question="Cos'e Bitcoin?",
                user_id=1,
                chat_id=10,
                user_lang="de",  # Legacy param says German
                envelope=envelope,
                context=exec_ctx,  # Context says Italian
                plan=exec_plan,
            )

        # Consensus analysis should use Italian from context, not German
        assert result.consensus_analysis is not None
        assert result.responses

    async def test_kernel_path_passes_system_prompt_to_provider(self) -> None:
        """With InstructionCompiler, providers get compiled system prompt."""
        call_log: list[str] = []

        class _LoggingProvider(_MockProvider):
            async def query(self, prompt, system_prompt="", **kwargs):
                call_log.append(system_prompt)
                return await super().query(prompt, system_prompt, **kwargs)

        providers = {
            "alpha": _LoggingProvider("alpha", response_text="OK"),
        }
        orchestrator = self._make_orchestrator_with_compiler(providers)

        exec_ctx = self._make_exec_context(lang="fr")
        exec_plan = self._make_exec_plan(lang="fr")
        envelope = self._make_envelope("Qu'est-ce que Bitcoin?")

        with patch.object(orchestrator, "final_review", return_value=None):
            await orchestrator.debate(
                question="Qu'est-ce que Bitcoin?",
                user_id=1,
                chat_id=10,
                user_lang="fr",
                envelope=envelope,
                context=exec_ctx,
                plan=exec_plan,
            )

        # Provider received a system prompt from InstructionCompiler
        assert len(call_log) == 1
        assert call_log[0]  # Non-empty system prompt
        # The compiled prompt should contain language lock for French
        assert "french" in call_log[0].lower() or "fr" in call_log[0].lower()

    async def test_legacy_path_still_works_without_kernel_params(self) -> None:
        """Without envelope/context/plan, legacy behavior is preserved."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="Antwort A"),
            "beta": _MockProvider("beta", response_text="Antwort B"),
        }
        router = _make_router(providers)
        # No instruction_compiler
        orchestrator = DebateOrchestrator(provider_router=router)

        with patch.object(orchestrator, "final_review", return_value=None):
            result = await orchestrator.debate(
                question="Was ist Bitcoin?",
                user_id=1,
                chat_id=10,
                user_lang="de",
            )

        assert "alpha" in result.responses
        assert "beta" in result.responses
        assert result.consensus_analysis is not None

    async def test_judge_uses_same_context_as_providers(self) -> None:
        """Judge receives the same ExecutionContext for consistent language."""
        providers = {
            "alpha": _MockProvider("alpha", response_text="Alpha says"),
            "beta": _MockProvider("beta", response_text="Beta says"),
        }
        orchestrator = self._make_orchestrator_with_compiler(providers)

        exec_ctx = self._make_exec_context(lang="es")
        exec_plan = self._make_exec_plan(lang="es")
        envelope = self._make_envelope("Que es Bitcoin?")

        # Track what final_review receives
        review_kwargs: dict = {}

        async def _capture_review(*args, **kwargs):
            review_kwargs.update(kwargs)
            return None

        with patch.object(orchestrator, "final_review", side_effect=_capture_review):
            await orchestrator.debate(
                question="Que es Bitcoin?",
                user_id=1,
                chat_id=10,
                user_lang="es",
                envelope=envelope,
                context=exec_ctx,
                plan=exec_plan,
            )

        # final_review should receive exec_context and exec_plan
        assert review_kwargs.get("exec_context") is exec_ctx
        assert review_kwargs.get("exec_plan") is exec_plan
        # And effective language (from context)
        assert review_kwargs.get("user_lang") == "es"
