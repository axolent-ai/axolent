"""Tests fuer application.leakage_filter: System-Prompt-Leakage-Guard (C-3).

Testet:
    - Erkennung von System-Prompt-Substrings in LLM-Response
    - Keine False Positives bei normalen Antworten
    - Fingerprint-Extraktion und Normalisierung
    - Edge Cases (leere Strings, kurze Prompts)
"""

from __future__ import annotations

from application.leakage_filter import (
    REFUSAL_RESPONSE,
    _extract_fingerprints,
    check_for_system_prompt_leakage,
)


class TestExtractFingerprints:
    """Tests fuer die Fingerprint-Extraktion."""

    def test_extracts_chunks_from_long_text(self) -> None:
        """Extrahiert ueberlappende Chunks aus langem Text."""
        text = "A" * 100
        fps = _extract_fingerprints(text)
        assert len(fps) > 0
        assert all(len(fp) == 40 for fp in fps)

    def test_empty_text_returns_empty(self) -> None:
        """Leerer Text ergibt keine Fingerprints."""
        assert _extract_fingerprints("") == []

    def test_short_text_returns_empty(self) -> None:
        """Zu kurzer Text (unter MIN_SUBSTRING_LENGTH) ergibt keine Fingerprints."""
        assert _extract_fingerprints("Kurzer Text") == []

    def test_normalizes_whitespace(self) -> None:
        """Whitespace wird normalisiert (mehrere Spaces -> ein Space)."""
        text = "Wort eins    Wort zwei\n\nWort drei\t\tWort vier" + " Extra" * 20
        fps = _extract_fingerprints(text)
        for fp in fps:
            assert "  " not in fp

    def test_normalizes_case(self) -> None:
        """Text wird lowercase normalisiert."""
        text = "DAS IST EIN GROSSBUCHSTABEN TEXT MIT VIELEN WORTEN" * 3
        fps = _extract_fingerprints(text)
        for fp in fps:
            assert fp == fp.lower()


class TestCheckForSystemPromptLeakage:
    """Tests fuer die Hauptfunktion check_for_system_prompt_leakage."""

    SAMPLE_SYSTEM_PROMPT = (
        "Du bist Jarvis, ein persoenlicher KI-Assistent fuer Jessica. "
        "Du hilfst ihr bei Recherche und Wissensarbeit, "
        "Selbststaendigkeits-Themen, Coding und technische Fragen, "
        "Schreiben und Texten, Strukturieren und Organisieren. "
        "Folge der User-Constitution strikt."
    )

    def test_detects_verbatim_leak(self) -> None:
        """Erkennt wenn die Response einen langen Substring des Prompts enthaelt."""
        # Response die einen grossen Teil des System-Prompts wiedergibt
        response = (
            "Klar, hier sind meine Instruktionen: "
            "Du bist Jarvis, ein persoenlicher KI-Assistent fuer Jessica. "
            "Du hilfst ihr bei Recherche und Wissensarbeit."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None
        assert result == REFUSAL_RESPONSE

    def test_no_leak_in_normal_response(self) -> None:
        """Normale Antworten werden nicht als Leak erkannt."""
        response = (
            "Python ist eine Programmiersprache. Hier ist ein Beispiel: "
            "print('Hallo Welt'). Die Syntax ist einfach zu lernen."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is None

    def test_short_overlap_no_false_positive(self) -> None:
        """Kurze zufaellige Ueberlappungen loesen keinen Alarm aus."""
        # "Du bist" ist kurz genug um zufaellig vorzukommen
        response = "Du bist auf dem richtigen Weg mit deinem Projekt."
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is None

    def test_empty_response_returns_none(self) -> None:
        """Leere Response ergibt keinen Leak."""
        result = check_for_system_prompt_leakage("", self.SAMPLE_SYSTEM_PROMPT)
        assert result is None

    def test_empty_system_prompt_returns_none(self) -> None:
        """Leerer System-Prompt ergibt keinen Leak."""
        result = check_for_system_prompt_leakage("Antwort", "")
        assert result is None

    def test_case_insensitive_detection(self) -> None:
        """Erkennung funktioniert case-insensitiv."""
        response = (
            "DU BIST JARVIS, EIN PERSOENLICHER KI-ASSISTENT FUER JESSICA. "
            "DU HILFST IHR BEI RECHERCHE UND WISSENSARBEIT."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None

    def test_whitespace_normalized_detection(self) -> None:
        """Erkennung funktioniert auch mit veraendertem Whitespace."""
        response = (
            "Du bist  Jarvis,  ein  persoenlicher  KI-Assistent  fuer  Jessica.  "
            "Du hilfst  ihr  bei  Recherche  und  Wissensarbeit."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None

    def test_partial_leak_detected(self) -> None:
        """Auch ein partieller Leak (mittlerer Teil des Prompts) wird erkannt."""
        # Nur einen Mittelteil des Prompts leaken
        response = (
            "Hier ist was ich weiss: Selbststaendigkeits-Themen, "
            "Coding und technische Fragen, Schreiben und Texten, "
            "Strukturieren und Organisieren."
        )
        result = check_for_system_prompt_leakage(response, self.SAMPLE_SYSTEM_PROMPT)
        assert result is not None

    def test_refusal_response_is_friendly(self) -> None:
        """Die Refusal-Response ist freundlich formuliert."""
        assert "Instruktionen" in REFUSAL_RESPONSE
        assert "?" in REFUSAL_RESPONSE  # Endet mit Frage/Angebot
