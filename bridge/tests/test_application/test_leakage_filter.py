"""Tests for application.leakage_filter: system prompt leakage guard (C-3).

Tests:
    * Detection of system prompt substrings in LLM response
    * No false positives for normal responses
    * Fingerprint extraction and normalization
    * Edge cases (empty strings, short prompts)
"""

from __future__ import annotations

from application.leakage_filter import (
    REFUSAL_RESPONSE,
    _extract_fingerprints,
    check_for_system_prompt_leakage,
)


class TestExtractFingerprints:
    """Tests for fingerprint extraction."""

    def test_extracts_chunks_from_long_text(self) -> None:
        """Extracts overlapping chunks from long text."""
        text = "A" * 100
        fps = _extract_fingerprints(text)
        assert len(fps) > 0
        assert all(len(fp) == 40 for fp in fps)

    def test_empty_text_returns_empty(self) -> None:
        """Empty text yields no fingerprints."""
        assert _extract_fingerprints("") == []

    def test_short_text_returns_empty(self) -> None:
        """Too short text (below MIN_SUBSTRING_LENGTH) yields no fingerprints."""
        assert _extract_fingerprints("Kurzer Text") == []

    def test_normalizes_whitespace(self) -> None:
        """Whitespace is normalized (multiple spaces -> single space)."""
        text = "Wort eins    Wort zwei\n\nWort drei\t\tWort vier" + " Extra" * 20
        fps = _extract_fingerprints(text)
        for fp in fps:
            assert "  " not in fp

    def test_normalizes_case(self) -> None:
        """Text is normalized to lowercase."""
        text = "DAS IST EIN GROSSBUCHSTABEN TEXT MIT VIELEN WORTEN" * 3
        fps = _extract_fingerprints(text)
        for fp in fps:
            assert fp == fp.lower()


class TestCheckForSystemPromptLeakage:
    """Tests for the main function check_for_system_prompt_leakage."""

    SAMPLE_SYSTEM_PROMPT = (
        "Du bist Axolent, ein persönlicher KI-Assistent. "
        "Du hilfst bei Recherche und Wissensarbeit, "
        "Selbstständigkeits-Themen, Coding und technische Fragen, "
        "Schreiben und Texten, Strukturieren und Organisieren. "
        "Folge der User-Constitution strikt."
    )

    def test_detects_verbatim_leak(self) -> None:
        """Detects when the response contains a long substring of the prompt."""
        # Response that reproduces a large part of the system prompt
        response = (
            "Klar, hier sind meine Instruktionen: "
            "Du bist Axolent, ein persönlicher KI-Assistent. "
            "Du hilfst bei Recherche und Wissensarbeit."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_no_leak_in_normal_response(self) -> None:
        """Normal responses are not detected as leaks."""
        response = (
            "Python ist eine Programmiersprache. Hier ist ein Beispiel: "
            "print('Hallo Welt'). Die Syntax ist einfach zu lernen."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is None

    def test_short_overlap_no_false_positive(self) -> None:
        """Short random overlaps do not trigger an alarm."""
        # "Du bist" is short enough to appear coincidentally
        response = "Du bist auf dem richtigen Weg mit deinem Projekt."
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is None

    def test_empty_response_returns_none(self) -> None:
        """Empty response yields no leak."""
        result = check_for_system_prompt_leakage("", self.SAMPLE_SYSTEM_PROMPT)
        assert result is None

    def test_empty_system_prompt_returns_none(self) -> None:
        """Empty system prompt yields no leak."""
        result = check_for_system_prompt_leakage("Antwort", "")
        assert result is None

    def test_case_insensitive_detection(self) -> None:
        """Detection works case-insensitively."""
        response = (
            "DU BIST AXOLENT, EIN PERSÖNLICHER KI-ASSISTENT. "
            "DU HILFST BEI RECHERCHE UND WISSENSARBEIT."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None

    def test_whitespace_normalized_detection(self) -> None:
        """Detection works with altered whitespace."""
        response = (
            "Du bist  Axolent,  ein  persönlicher  KI-Assistent.  "
            "Du hilfst  bei  Recherche  und  Wissensarbeit."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None

    def test_partial_leak_detected(self) -> None:
        """A partial leak (middle part of the prompt) is also detected."""
        # Leak only a middle section of the prompt
        response = (
            "Hier ist was ich weiss: Selbstständigkeits-Themen, "
            "Coding und technische Fragen, Schreiben und Texten, "
            "Strukturieren und Organisieren."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None

    def test_refusal_response_is_friendly(self) -> None:
        """The refusal response is phrased in a friendly way."""
        assert "instructions" in REFUSAL_RESPONSE.lower()
        assert "?" in REFUSAL_RESPONSE  # Ends with question/offer

    def test_boundary_offset_leak_detected(self) -> None:
        """V6 regression: 40-char leak at arbitrary offset is detected.

        With step size 20 (old) a leak at offset 10 could slip through
        because no chunk exactly covered that range.
        With step size 1 (new) every position is covered.
        """
        # System-Prompt: 80 Zeichen + Filler
        system = "A" * 10 + "GEHEIMER_BLOCK_DER_GENAU_VIERZIG_ZEICHEN!" + "B" * 30
        # Response enthält den geheimen Block ab Offset 10
        response = "Hier sind meine Instruktionen: GEHEIMER_BLOCK_DER_GENAU_VIERZIG_ZEICHEN! Ende."
        result = check_for_system_prompt_leakage(response, system)
        assert result is not None
        assert result == REFUSAL_RESPONSE
