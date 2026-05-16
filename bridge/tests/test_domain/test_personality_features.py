"""Tests for Personality Features P2, P3, P4, P6 in system prompts.

Verifies that the system prompt contains the required personality
clauses for natural communication, response calibration, and
weakness admission.
"""

from pathlib import Path

import pytest

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_SYSTEM_PROMPT_EXAMPLE = _CONFIG_DIR / "system_prompt.example.md"


@pytest.fixture
def system_prompt_example() -> str:
    """Load the example system prompt."""
    return _SYSTEM_PROMPT_EXAMPLE.read_text(encoding="utf-8")


class TestP2ReducedAITalk:
    """P2: System prompt prohibits AI self-referencing."""

    def test_no_ai_talk_section_exists(self, system_prompt_example: str) -> None:
        """P2 section exists in system prompt."""
        assert (
            "No AI-Talk" in system_prompt_example or "AI-Talk" in system_prompt_example
        )

    def test_prohibits_as_an_ai(self, system_prompt_example: str) -> None:
        """Explicitly prohibits 'As an AI' phrasing."""
        assert "As an AI" in system_prompt_example

    def test_prohibits_language_model_reference(
        self, system_prompt_example: str
    ) -> None:
        """Explicitly prohibits 'As a language model' phrasing."""
        assert "language model" in system_prompt_example

    def test_prohibits_happy_to_help(self, system_prompt_example: str) -> None:
        """Explicitly prohibits 'I'm happy to help' filler."""
        assert "happy to help" in system_prompt_example

    def test_allows_medical_legal_exceptions(self, system_prompt_example: str) -> None:
        """Medical/legal transparency exception is documented."""
        assert (
            "medical" in system_prompt_example.lower()
            or "legal" in system_prompt_example.lower()
        )


class TestP3ContextualSilence:
    """P3: Response length and style adaptation rules."""

    def test_response_length_section_exists(self, system_prompt_example: str) -> None:
        """P3 section about response length exists."""
        assert (
            "Response Length" in system_prompt_example
            or "response length" in system_prompt_example.lower()
        )

    def test_yes_no_brevity_rule(self, system_prompt_example: str) -> None:
        """Yes/No questions should get 1-2 sentences."""
        assert (
            "1 to 2 sentences" in system_prompt_example
            or "1-2" in system_prompt_example
        )

    def test_rule_of_thumb_present(self, system_prompt_example: str) -> None:
        """The 'if another sentence does not make it clearer, stop' rule."""
        assert "clearer" in system_prompt_example.lower()

    def test_style_adaptation_section(self, system_prompt_example: str) -> None:
        """Style adaptation (mirror emoji, formality, tonality) is defined."""
        assert "emoji" in system_prompt_example.lower()
        assert (
            "formality" in system_prompt_example.lower()
            or "formal" in system_prompt_example.lower()
        )

    def test_no_length_mirroring(self, system_prompt_example: str) -> None:
        """Explicitly states NOT to mirror input length."""
        assert "NOT adapt" in system_prompt_example or "do NOT" in system_prompt_example

    def test_device_awareness_present(self, system_prompt_example: str) -> None:
        """Device awareness (mobile patterns) is mentioned."""
        assert "mobile" in system_prompt_example.lower()


class TestP4ConfidenceCalibration:
    """P4: Verbal confidence calibration rules."""

    def test_confidence_section_exists(self, system_prompt_example: str) -> None:
        """P4 confidence calibration section exists."""
        assert (
            "Confidence" in system_prompt_example
            or "confidence" in system_prompt_example
        )

    def test_verbal_uncertainty_expression(self, system_prompt_example: str) -> None:
        """Instructs to express uncertainty verbally."""
        assert (
            "uncertain" in system_prompt_example.lower()
            or "unsicher" in system_prompt_example.lower()
        )

    def test_percentage_calibration(self, system_prompt_example: str) -> None:
        """Instructs percentage-based confidence expression."""
        assert "70%" in system_prompt_example

    def test_confident_no_hedging(self, system_prompt_example: str) -> None:
        """When confident, no unnecessary hedging."""
        assert (
            "hedging" in system_prompt_example.lower()
            or "Just state it" in system_prompt_example
        )


class TestP6WeaknessShowing:
    """P6: Genuine weakness and curiosity rules."""

    def test_weakness_section_exists(self, system_prompt_example: str) -> None:
        """P6 section about weakness showing exists."""
        assert (
            "Weakness" in system_prompt_example or "weakness" in system_prompt_example
        )

    def test_i_dont_know_is_valid(self, system_prompt_example: str) -> None:
        """'I do not know' is explicitly marked as excellent answer."""
        assert (
            "do not know" in system_prompt_example
            or "not know" in system_prompt_example
        )

    def test_ask_instead_of_guess(self, system_prompt_example: str) -> None:
        """Instructs to ask when ambiguous, not guess."""
        assert (
            "guess" in system_prompt_example.lower()
            or "ask" in system_prompt_example.lower()
        )

    def test_no_hallucination_filling(self, system_prompt_example: str) -> None:
        """Prohibits filling gaps with hallucinated content."""
        assert "hallucin" in system_prompt_example.lower()

    def test_natural_framing(self, system_prompt_example: str) -> None:
        """Good/bad examples of framing are provided."""
        assert "Good:" in system_prompt_example or "Bad:" in system_prompt_example


class TestPromptIntegrity:
    """Cross-cutting concerns: prompt structure remains valid."""

    def test_system_prompt_example_exists(self) -> None:
        """The example system prompt file exists."""
        assert _SYSTEM_PROMPT_EXAMPLE.exists()

    def test_system_prompt_not_empty(self, system_prompt_example: str) -> None:
        """System prompt is not empty after additions."""
        assert len(system_prompt_example) > 500

    def test_original_sections_preserved(self, system_prompt_example: str) -> None:
        """Original sections (Role, Memory, Confidentiality) still present."""
        assert "Your Role" in system_prompt_example
        assert "Memory Behavior" in system_prompt_example
        assert "Instruction Confidentiality" in system_prompt_example

    def test_no_duplicate_style_requirements(self, system_prompt_example: str) -> None:
        """Style Requirements section appears exactly once."""
        count = system_prompt_example.count("## Style Requirements")
        assert count == 1
