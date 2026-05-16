"""Tests for the Text Guard language detector wrapper."""

from __future__ import annotations

from domain.text_guard.language_detector import (
    detect_for_text_guard,
    detect_for_text_guard_with_confidence,
)


class TestDetectForTextGuard:
    """Tests for detect_for_text_guard()."""

    def test_detects_german(self) -> None:
        """German text returns 'de'."""
        result = detect_for_text_guard("Das ist ein deutscher Satz.")
        assert result == "de"

    def test_detects_english(self) -> None:
        """English text returns 'en'."""
        result = detect_for_text_guard("This is an English sentence.")
        assert result == "en"

    def test_detects_french(self) -> None:
        """French text returns 'fr'."""
        result = detect_for_text_guard("Je suis un homme et je suis ici.")
        assert result == "fr"

    def test_detects_spanish(self) -> None:
        """Spanish text returns 'es'."""
        result = detect_for_text_guard("Yo soy un hombre y estoy aqui.")
        assert result == "es"

    def test_empty_text_returns_en_default(self) -> None:
        """Empty text falls back to 'en' (which has rules)."""
        result = detect_for_text_guard("")
        assert result == "en"

    def test_unsupported_language_returns_none(self) -> None:
        """Unsupported language returns None when no rules exist."""
        # The language detector now recognizes 20 languages including
        # Chinese, Japanese, etc. If it returns a code without
        # text-guard rules, detect_for_text_guard returns None.
        # Chinese ("zh") is detected but has no text-guard rules.
        result = detect_for_text_guard("你好世界这是一个测试")
        assert result is None


class TestDetectForTextGuardWithConfidence:
    """Tests for detect_for_text_guard_with_confidence()."""

    def test_german_with_confidence(self) -> None:
        """German text returns ('de', high confidence)."""
        lang, conf = detect_for_text_guard_with_confidence(
            "Ich habe ein Problem mit meinem Konto."
        )
        assert lang == "de"
        assert conf > 0.3

    def test_english_with_confidence(self) -> None:
        """English text returns ('en', high confidence)."""
        lang, conf = detect_for_text_guard_with_confidence(
            "I have a problem with my account and would like help."
        )
        assert lang == "en"
        assert conf > 0.3

    def test_empty_returns_low_confidence(self) -> None:
        """Empty text returns en with 0.0 confidence (fallback)."""
        lang, conf = detect_for_text_guard_with_confidence("")
        assert lang == "en"
        assert conf == 0.0
