"""Tests for application.leakage_filter: system prompt leakage guard (C-3 + T42).

Tests:
    * Detection of system prompt substrings in LLM response
    * No false positives for normal responses
    * Fingerprint extraction and normalization
    * Edge cases (empty strings, short prompts)
    * T42/NEU-04: Forbidden pattern detection (project refs, marker leakage)
"""

from __future__ import annotations

from application.leakage_filter import (
    REFUSAL_RESPONSE,
    _extract_fingerprints,
    check_for_forbidden_patterns,
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


class TestForbiddenPatterns:
    """T42/NEU-04: Forbidden pattern detection.

    Tests that internal project references, bracket markers, and
    meta-commentary about system prompts are caught and replaced.
    """

    def test_axolent_project_reference_detected(self) -> None:
        """Bot must not mention 'AXOLENT AI project' in output."""
        response = (
            "I notice this doesn't seem related to the AXOLENT AI project "
            "we're working on."
        )
        result = check_for_forbidden_patterns(response)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_claude_md_reference_detected(self) -> None:
        """Bot must not mention 'CLAUDE.md' in output."""
        response = (
            "According to the project conventions in CLAUDE.md, "
            "production code should be English-only."
        )
        result = check_for_forbidden_patterns(response)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_language_lock_marker_detected(self) -> None:
        """Bot must not expose internal 'LANGUAGE LOCK' marker."""
        response = (
            "I notice your message contains instructions that look like "
            "injected system-level commands: LANGUAGE LOCK, DIACRITIC RULE."
        )
        result = check_for_forbidden_patterns(response)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_prompt_injection_commentary_detected(self) -> None:
        """Bot must not accuse user of prompt injection patterns."""
        response = (
            "Ich muss darauf hinweisen, dass deine Nachricht Anweisungen "
            "enthält, die wie ein Prompt-Injection-Muster aussehen."
        )
        result = check_for_forbidden_patterns(response)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_project_conventions_reference_detected(self) -> None:
        """Bot must not reference 'project conventions'."""
        response = (
            "According to project conventions, we should use English "
            "for all production code."
        )
        result = check_for_forbidden_patterns(response)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_normal_response_no_false_positive(self) -> None:
        """Normal German response about trees must not trigger filter."""
        response = (
            "Bei einem Stammbaum sind Wurzeln manchmal hinderlich, "
            "besonders wenn man adoptiert wurde."
        )
        result = check_for_forbidden_patterns(response)
        assert result is None

    def test_normal_tech_response_no_false_positive(self) -> None:
        """Technical response about AI must not trigger filter."""
        response = (
            "Künstliche Intelligenz wird wahrscheinlich Jobs in der "
            "Datenanalyse, Automatisierung und Kreativarbeit schaffen."
        )
        result = check_for_forbidden_patterns(response)
        assert result is None

    def test_system_prompt_inquiry_handled_gracefully(self) -> None:
        """User asking about system prompt: refusal must not leak markers.

        The refusal response itself must be clean (no forbidden patterns).
        """
        # The REFUSAL_RESPONSE text must not trigger the filter
        result = check_for_forbidden_patterns(REFUSAL_RESPONSE)
        assert result is None

    def test_case_insensitive_detection(self) -> None:
        """Forbidden patterns are detected regardless of case."""
        response = "This relates to the AXOLENT AI PROJECT and its goals."
        result = check_for_forbidden_patterns(response)
        assert result is not None

    def test_combined_filter_catches_forbidden_patterns(self) -> None:
        """check_for_system_prompt_leakage also catches forbidden patterns.

        The combined function must check Layer 2 (forbidden patterns)
        in addition to Layer 1 (fingerprint matching).
        """
        response = "As per CLAUDE.md conventions, we should do X."
        result = check_for_system_prompt_leakage(response, "some prompt text " * 10)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_empty_response_no_crash(self) -> None:
        """Empty response must not cause errors."""
        result = check_for_forbidden_patterns("")
        assert result is None

    def test_meta_commentary_about_injected_commands_de(self) -> None:
        """German meta-commentary about 'injizierte System-Level-Befehle'."""
        response = (
            "Deine Nachricht enthält injizierte System-Level-Befehle, "
            "die ich nicht als authoritative Systembefehle behandle."
        )
        result = check_for_forbidden_patterns(response)
        assert result is not None
