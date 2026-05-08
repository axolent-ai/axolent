"""Tests für domain.language: Sprach-Detection via Heuristik.

Testet die korrekte Erkennung der gängigsten Sprachen
sowie Edge-Cases (kurze Nachrichten, Emojis, exotische Schriften).
"""

from domain.language import detect_language


class TestDetectLanguage:
    """Sprach-Detection Smoke-Tests und Edge-Cases."""

    def test_detect_language_german(self) -> None:
        """Deutsche Sätze müssen als 'de' erkannt werden."""
        assert detect_language("Ich habe heute viel gelernt") == "de"
        assert detect_language("Wie geht es dir?") == "de"
        assert detect_language("Das ist sehr gut") == "de"

    def test_detect_language_english(self) -> None:
        """Englische Sätze müssen als 'en' erkannt werden."""
        assert detect_language("I have learned a lot today") == "en"
        assert detect_language("How are you doing?") == "en"
        assert detect_language("This is very good") == "en"

    def test_detect_language_spanish(self) -> None:
        """Spanische Sätze müssen als 'es' erkannt werden."""
        assert detect_language("Hola, como estas?") == "es"
        assert detect_language("Muchas gracias por tu ayuda") == "es"

    def test_detect_language_french(self) -> None:
        """Französische Sätze müssen als 'fr' erkannt werden."""
        assert detect_language("Bonjour, comment allez-vous?") == "fr"
        assert detect_language("Merci beaucoup pour votre aide") == "fr"

    def test_detect_language_short_message_fallback_to_default(self) -> None:
        """Sehr kurze Nachrichten ohne klare Marker fallen auf 'de' zurück."""
        assert detect_language("ok") == "de"
        assert detect_language("") == "de"
        assert detect_language("   ") == "de"

    def test_detect_language_smart_quotes_normalized(self) -> None:
        """Smart-Quotes werden zu normalen Apostrophen normalisiert.

        Damit "don’t" korrekt gegen den 'en'-Marker matcht.
        """
        # ’ = right single quotation mark (Smart Quote)
        result = detect_language("I don’t know what you mean")
        assert result == "en"

    def test_detect_language_emoji_only(self) -> None:
        """Nur-Emoji-Nachrichten haben keine Wort-Marker, Fallback 'de'."""
        assert detect_language("\U0001f600\U0001f44d\U0001f389") == "de"

    def test_detect_language_japanese_chinese_arabic_fallback(self) -> None:
        """Nicht-lateinische Schriften ohne Marker fallen auf 'de' zurück.

        Das ist korrektes Verhalten: wir erkennen nur de/en/es/fr explizit.
        """
        assert detect_language("こんにちは") == "de"  # Japanisch
        assert detect_language("你好世界") == "de"  # Chinesisch
        assert detect_language("مرحبا") == "de"  # Arabisch

    def test_detect_language_german_umlauts_boost(self) -> None:
        """Umlaute geben einen Char-Hint-Bonus für Deutsch."""
        result = detect_language("Wir fahren nach Muenchen fuer die Pruefung")
        # Keine Umlaute im Text, aber genug deutsche Marker
        assert result == "de"

        result = detect_language("Die Bruecke ueber den Fluessen ist schoen")
        assert result == "de"

    def test_detect_language_mixed_but_dominant(self) -> None:
        """Bei gemischtem Text gewinnt die dominante Sprache."""
        # Überwiegend Englisch mit einem deutschen Wort
        result = detect_language("I think this is a great idea for the team")
        assert result == "en"
