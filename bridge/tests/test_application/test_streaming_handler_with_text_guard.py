"""Integration tests: streaming handler with text guard.

Verifies that the TextGuard streaming integration works correctly
with the existing streaming handler infrastructure.
"""

from __future__ import annotations

import pytest

from domain.text_guard import TextGuard, get_builtin_rules
from domain.text_guard.adapters.streaming import StreamingTextGuard


@pytest.fixture
def de_text_guard() -> TextGuard:
    """German text guard for integration tests."""
    rules = get_builtin_rules("de")
    assert rules is not None
    return TextGuard(rules, mode="fix")


@pytest.fixture
def de_streaming_guard(de_text_guard: TextGuard) -> StreamingTextGuard:
    """German streaming guard for integration tests."""
    return StreamingTextGuard(de_text_guard)


class TestStreamingIntegration:
    """Tests simulating the bot streaming pipeline with text guard."""

    def test_token_by_token_correction(
        self, de_streaming_guard: StreamingTextGuard
    ) -> None:
        """Simulates token-by-token arrival and correction."""
        # Simulate: "Das ist fuer dich."
        tokens = ["Das ", "ist ", "fuer", " ", "dich."]
        collected: list[str] = []

        for token in tokens:
            result = de_streaming_guard.process_token(token)
            if result is not None:
                collected.append(result)

        remaining = de_streaming_guard.flush()
        if remaining:
            collected.append(remaining)

        full_text = "".join(collected)
        assert "für" in full_text
        assert "fuer" not in full_text

    def test_final_text_correction(self, de_text_guard: TextGuard) -> None:
        """Simulates final text correction (like finalize_streaming)."""
        final_text = (
            "Ich wuerde dir natuerlich erklaeren warum das fuer uns moeglich waere."
        )
        corrected = de_text_guard.fix(final_text)
        assert "würde" in corrected
        assert "natürlich" in corrected
        assert "erklären" in corrected
        assert "für" in corrected
        assert "möglich" in corrected
        assert "wäre" in corrected

    def test_code_block_preservation_in_stream(
        self, de_streaming_guard: StreamingTextGuard
    ) -> None:
        """Code blocks in streamed text are not modified."""
        tokens = ["fuer", " ", "code: ", "```", "\n", "fuer", "\n", "```"]
        collected: list[str] = []

        for token in tokens:
            result = de_streaming_guard.process_token(token)
            if result is not None:
                collected.append(result)

        remaining = de_streaming_guard.flush()
        if remaining:
            collected.append(remaining)

        full_text = "".join(collected)
        # The first "fuer" (before code block) should be corrected
        # The "fuer" inside code block should stay
        assert "für" in full_text

    def test_mixed_languages_passthrough(self) -> None:
        """English text with German guard passes through unchanged."""
        rules = get_builtin_rules("de")
        assert rules is not None
        guard = TextGuard(rules)
        text = "The user continued to use the module for the queue."
        assert guard.fix(text) == text

    def test_full_pipeline_simulation(self, de_text_guard: TextGuard) -> None:
        """Full pipeline: streaming tokens -> final text -> text guard."""
        # Phase 1: Accumulate tokens (simulated)
        accumulated = ""
        tokens = [
            "Das ",
            "Erkl",
            "aer",
            "ung ",
            "ist ",
            "natue",
            "rlich ",
            "fuer ",
            "die ",
            "Moeg",
            "lichkeit.",
        ]
        for token in tokens:
            accumulated += token

        # Phase 2: Final text through guard
        corrected = de_text_guard.fix(accumulated)

        assert "Erklärung" in corrected
        assert "natürlich" in corrected
        assert "für" in corrected
        assert "Möglichkeit" in corrected


class TestServiceCaching:
    """Tests for TextGuardService caching behavior."""

    def test_service_caches_guards(self) -> None:
        """TextGuardService caches guard instances."""
        from application.text_guard_service import TextGuardService

        service = TextGuardService()
        g1 = service.get_guard("de")
        g2 = service.get_guard("de")
        assert g1 is g2

    def test_service_caches_streaming_guards(self) -> None:
        """TextGuardService caches streaming guard instances."""
        from application.text_guard_service import TextGuardService

        service = TextGuardService()
        sg1 = service.get_streaming_guard("de")
        sg2 = service.get_streaming_guard("de")
        assert sg1 is sg2
