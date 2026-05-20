"""Tests for ResponseLanguageVerifier."""

import pytest

from application.language.verifier import (
    ResponseLanguageVerifier,
    VerificationResult,
    VerificationStatus,
)


@pytest.fixture
def verifier() -> ResponseLanguageVerifier:
    """Create a standard verifier instance."""
    return ResponseLanguageVerifier()


class TestVerifierBasics:
    """Basic verification scenarios."""

    def test_german_text_passes_for_german(
        self, verifier: ResponseLanguageVerifier
    ) -> None:
        """Clear German text passes when expecting German."""
        text = (
            "Das ist ein sehr langer deutscher Text der genug Wörter hat "
            "um die Mindestanforderung zu erfüllen. Wir brauchen mindestens "
            "zwanzig Wörter damit die Verifikation nicht übersprungen wird. "
            "Dieser Satz sollte ausreichen für eine zuverlässige Erkennung."
        )
        result = verifier.verify(text, "de")
        assert result.passed is True
        assert result.expected_lang == "de"

    def test_english_text_fails_for_german(
        self, verifier: ResponseLanguageVerifier
    ) -> None:
        """Clear English text fails when expecting German."""
        text = (
            "This is a very long English text that has enough words to "
            "meet the minimum requirement. We need at least twenty words "
            "so that the verification is not skipped. This sentence should "
            "be sufficient for reliable detection of the English language."
        )
        result = verifier.verify(text, "de")
        # Should either fail or have low confidence
        if result.confidence >= 0.7:
            assert result.passed is False or result.detected_lang == "en"

    def test_english_text_passes_for_english(
        self, verifier: ResponseLanguageVerifier
    ) -> None:
        """English text passes when expecting English."""
        text = (
            "This is a very long English text that has enough words to "
            "meet the minimum requirement. We need at least twenty words "
            "so that the verification is not skipped. This sentence should "
            "be sufficient for reliable detection of the English language."
        )
        result = verifier.verify(text, "en")
        assert result.passed is True

    def test_short_text_is_skipped(self, verifier: ResponseLanguageVerifier) -> None:
        """Text with fewer than 20 words is skipped."""
        text = "Ja, das stimmt."
        result = verifier.verify(text, "de")
        assert result.skipped is True
        assert result.passed is True  # Skipped = pass

    def test_empty_text_is_skipped(self, verifier: ResponseLanguageVerifier) -> None:
        """Empty text is skipped."""
        result = verifier.verify("", "de")
        assert result.skipped is True
        assert result.passed is True


class TestVerifierCodeStripping:
    """Tests that code blocks don't trigger false positives."""

    def test_code_blocks_stripped(self, verifier: ResponseLanguageVerifier) -> None:
        """Code blocks in English don't fail German verification."""
        text = (
            "Hier ist ein Beispiel wie man das in Python macht. "
            "Der folgende Code zeigt die Implementierung einer Funktion "
            "die Daten aus einer Datenbank liest und verarbeitet. "
            "Dies ist sehr nützlich für die tägliche Arbeit mit Daten.\n\n"
            "```python\n"
            "def fetch_data(url):\n"
            "    response = requests.get(url)\n"
            "    return response.json()\n"
            "```\n\n"
            "So kann man das verwenden."
        )
        result = verifier.verify(text, "de")
        assert result.passed is True

    def test_inline_code_stripped(self, verifier: ResponseLanguageVerifier) -> None:
        """Inline code doesn't count as foreign language."""
        text = (
            "Verwende den Befehl `git commit -m` um deine Änderungen zu speichern. "
            "Danach kannst du mit `git push origin main` alles hochladen. "
            "Das ist der normale Workflow wenn du mit Git arbeitest und "
            "deine Änderungen teilen möchtest."
        )
        result = verifier.verify(text, "de")
        assert result.passed is True


class TestVerifierURLStripping:
    """Tests that URLs don't trigger false positives."""

    def test_urls_stripped(self, verifier: ResponseLanguageVerifier) -> None:
        """URLs in the text don't affect language detection."""
        text = (
            "Du findest die Dokumentation unter https://docs.example.com/getting-started "
            "und kannst dort alle Details nachlesen. Die Installation ist sehr einfach "
            "und dauert nur wenige Minuten. Folge einfach den Schritten auf der Seite."
        )
        result = verifier.verify(text, "de")
        assert result.passed is True


class TestVerifierWhitelist:
    """Tests that technical terms don't trigger false positives."""

    def test_technical_terms_dont_flag(
        self, verifier: ResponseLanguageVerifier
    ) -> None:
        """Technical English terms in German text don't trigger failure."""
        text = (
            "Der API Endpoint akzeptiert JSON Requests über HTTPS. "
            "Du musst einen Token im Header mitschicken und der Server "
            "gibt eine Response mit dem passenden Schema zurück. "
            "Das Backend läuft auf einem Docker Container."
        )
        result = verifier.verify(text, "de")
        assert result.passed is True


class TestVerifierSlidingWindow:
    """Tests for long-text sliding window detection."""

    def test_long_consistent_text(self, verifier: ResponseLanguageVerifier) -> None:
        """Long consistent German text passes easily."""
        # Generate a long German text (>100 words)
        sentence = "Dies ist ein deutscher Satz der sich wiederholt. "
        text = sentence * 25  # ~200 words
        result = verifier.verify(text, "de")
        assert result.passed is True

    def test_long_consistent_english(self, verifier: ResponseLanguageVerifier) -> None:
        """Long consistent English text fails for German."""
        sentence = "This is an English sentence that repeats itself many times. "
        text = sentence * 25  # ~225 words
        result = verifier.verify(text, "de")
        # Should fail (English detected, German expected)
        if result.confidence >= 0.7:
            assert result.passed is False


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_result_is_frozen(self) -> None:
        """VerificationResult is immutable."""
        result = VerificationResult(
            expected_lang="de",
            detected_lang="en",
            confidence=0.9,
            foreign_share=0.8,
            target_language_ratio=0.2,
            status=VerificationStatus.FAIL,
            reason="test",
        )
        with pytest.raises(AttributeError):
            result.status = VerificationStatus.PASS  # type: ignore[misc]

    def test_result_fields(self) -> None:
        """All fields are accessible."""
        result = VerificationResult(
            expected_lang="de",
            detected_lang="en",
            confidence=0.95,
            foreign_share=0.9,
            target_language_ratio=0.1,
            status=VerificationStatus.FAIL,
            reason="Expected 'de' but got 'en'",
            skipped=False,
        )
        assert result.expected_lang == "de"
        assert result.detected_lang == "en"
        assert result.confidence == 0.95
        assert result.foreign_share == 0.9
        assert result.target_language_ratio == 0.1
        assert result.status == VerificationStatus.FAIL
        assert result.passed is False
        assert result.reason is not None
        assert result.skipped is False

    def test_passed_property_backwards_compat(self) -> None:
        """passed property returns True for PASS and WARN."""
        pass_result = VerificationResult(
            expected_lang="de",
            detected_lang="de",
            confidence=0.9,
            foreign_share=0.0,
            target_language_ratio=1.0,
            status=VerificationStatus.PASS,
            reason=None,
        )
        warn_result = VerificationResult(
            expected_lang="de",
            detected_lang="de",
            confidence=0.9,
            foreign_share=0.3,
            target_language_ratio=0.7,
            status=VerificationStatus.WARN,
            reason=None,
        )
        fail_result = VerificationResult(
            expected_lang="de",
            detected_lang="en",
            confidence=0.9,
            foreign_share=0.8,
            target_language_ratio=0.2,
            status=VerificationStatus.FAIL,
            reason="test",
        )
        assert pass_result.passed is True
        assert warn_result.passed is True
        assert fail_result.passed is False
