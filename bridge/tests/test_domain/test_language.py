"""Tests for domain.language: language detection via heuristics.

Tests correct detection of all 20 supported languages
as well as edge cases (short messages, emojis, mixed scripts).
Since R02-B: also detect_language_with_confidence().
"""

import pytest

from domain.language import detect_language, detect_language_with_confidence


@pytest.mark.unit
@pytest.mark.i18n
class TestDetectLanguage:
    """Detection smoke tests for all 20 languages."""

    def test_detect_language_german(self) -> None:
        """Deutsche Saetze muessen als 'de' erkannt werden."""
        assert detect_language("Ich habe heute viel gelernt") == "de"
        assert detect_language("Wie geht es dir?") == "de"
        assert detect_language("Das ist sehr gut") == "de"

    def test_detect_language_english(self) -> None:
        """English sentences must be detected as 'en'."""
        assert detect_language("I have learned a lot today") == "en"
        assert detect_language("How are you doing?") == "en"
        assert detect_language("This is very good") == "en"

    def test_detect_language_spanish(self) -> None:
        """Spanish sentences must be detected as 'es'."""
        assert detect_language("Hola, como estas?") == "es"
        assert detect_language("Muchas gracias por tu ayuda") == "es"

    def test_detect_language_french(self) -> None:
        """French sentences must be detected as 'fr'."""
        assert detect_language("Bonjour, comment allez-vous?") == "fr"
        assert detect_language("Merci beaucoup pour votre aide") == "fr"

    def test_detect_language_dutch(self) -> None:
        """Dutch sentences must be detected as 'nl'."""
        assert detect_language("Dit is een heel goed voorbeeld voor het team") == "nl"
        assert detect_language("We hebben niet veel tijd meer") == "nl"

    def test_detect_language_italian(self) -> None:
        """Italian sentences must be detected as 'it'."""
        assert detect_language("Questo non sono cose che posso fare") == "it"
        assert detect_language("Grazie molto per il tuo aiuto") == "it"

    def test_detect_language_portuguese(self) -> None:
        """Portuguese sentences must be detected as 'pt'."""
        assert detect_language("Eu nao sei como fazer isso para voce") == "pt"
        assert detect_language("Muito obrigado pela sua ajuda") == "pt"

    def test_detect_language_polish(self) -> None:
        """Polish sentences must be detected as 'pl'."""
        assert detect_language("To jest bardzo dobrze dla nas") == "pl"
        assert detect_language("Nie wiem jak to zrobic teraz") == "pl"

    def test_detect_language_swedish(self) -> None:
        """Swedish sentences must be detected as 'sv'."""
        assert detect_language("Det har inte varit mycket att gora sedan dess") == "sv"
        assert detect_language("Jag kan inte hitta det utan din hjalp") == "sv"

    def test_detect_language_turkish(self) -> None:
        """Turkish sentences must be detected as 'tr'."""
        assert detect_language("Bu bir cok iyi sonra gelecek") == "tr"
        assert detect_language("Burada nasil bir sey var ancak bilmiyorum") == "tr"

    def test_detect_language_indonesian(self) -> None:
        """Indonesian sentences must be detected as 'id'."""
        assert detect_language("Saya tidak bisa melakukan ini untuk kami") == "id"
        assert detect_language("Yang ini adalah sangat baik untuk kita") == "id"

    def test_detect_language_vietnamese(self) -> None:
        """Vietnamese sentences must be detected as 'vi'."""
        assert detect_language("Tôi không biết làm thế nào") == "vi"
        assert detect_language("Day la mot cai rat tot") == "vi"

    def test_detect_language_russian(self) -> None:
        """Russian (Cyrillic) sentences must be detected as 'ru'."""
        assert detect_language("Это очень хороший пример") == "ru"
        assert detect_language("Я не знаю как это сделать") == "ru"

    def test_detect_language_ukrainian(self) -> None:
        """Ukrainian sentences must be detected as 'uk'."""
        assert detect_language("Це є дуже гарний приклад") == "uk"
        assert detect_language("Я не знаю як це зробити") == "uk"

    def test_detect_language_arabic(self) -> None:
        """Arabic sentences must be detected as 'ar'."""
        assert detect_language("هذا اختبار جيد جدا") == "ar"
        assert detect_language("كيف حالك اليوم") == "ar"

    def test_detect_language_hindi(self) -> None:
        """Hindi (Devanagari) sentences must be detected as 'hi'."""
        assert detect_language("यह एक बहुत अच्छा उदाहरण है") == "hi"
        assert detect_language("मुझे नहीं पता कैसे करना है") == "hi"

    def test_detect_language_chinese(self) -> None:
        """Chinese sentences must be detected as 'zh'."""
        assert detect_language("你好世界这是一个测试") == "zh"
        assert detect_language("我不知道怎么做这件事") == "zh"

    def test_detect_language_japanese(self) -> None:
        """Japanese (Hiragana/Katakana) sentences must be detected as 'ja'."""
        assert detect_language("これはテストです") == "ja"
        assert detect_language("こんにちは元気ですか") == "ja"

    def test_detect_language_korean(self) -> None:
        """Korean (Hangul) sentences must be detected as 'ko'."""
        assert detect_language("이것은 테스트입니다") == "ko"
        assert detect_language("안녕하세요 잘 지내세요") == "ko"

    def test_detect_language_thai(self) -> None:
        """Thai sentences must be detected as 'th'."""
        assert detect_language("นี่คือการทดสอบ") == "th"
        assert detect_language("สวัสดีครับ สบายดีไหม") == "th"


class TestDetectLanguageEdgeCases:
    """Edge cases and fallback behaviour."""

    def test_empty_text_fallback_to_en(self) -> None:
        """Empty or whitespace-only text falls back to 'en'."""
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"

    def test_emoji_only_fallback(self) -> None:
        """Emoji-only messages fall back to 'en'."""
        assert detect_language("\U0001f600\U0001f44d\U0001f389") == "en"

    def test_smart_quotes_normalized(self) -> None:
        """Smart quotes are normalized so contractions match."""
        result = detect_language("I don’t know what you mean")
        assert result == "en"

    def test_german_umlauts_boost(self) -> None:
        """Umlauts give a char-hint bonus for German."""
        result = detect_language("Die Bruecke uber den Fluss ist schon")
        assert result == "de"

    def test_mixed_but_dominant(self) -> None:
        """Dominant language wins in mixed text."""
        result = detect_language("I think this is a great idea for the team")
        assert result == "en"

    def test_polish_diacritics(self) -> None:
        """Polish diacritics trigger 'pl' detection."""
        result = detect_language("Dziękuję bardzo za pomoc")
        assert result == "pl"

    def test_vietnamese_diacritics(self) -> None:
        """Vietnamese diacritics trigger 'vi' detection."""
        result = detect_language("Cảm ơn bạn rất nhiều")
        assert result == "vi"

    def test_turkish_diacritics(self) -> None:
        """Turkish special chars trigger 'tr' detection."""
        result = detect_language("Teşekkür ederim çok güzel")
        assert result == "tr"

    def test_cyrillic_with_ukrainian_chars(self) -> None:
        """Cyrillic text with 'ї' or 'є' is Ukrainian, not Russian."""
        assert detect_language("Він їде додому") == "uk"
        assert detect_language("Це є правда") == "uk"

    def test_cyrillic_without_ukrainian_chars_is_russian(self) -> None:
        """Cyrillic text without Ukrainian markers defaults to Russian."""
        assert detect_language("Он идет домой") == "ru"


class TestDetectLanguageWithConfidence:
    """Tests for detect_language_with_confidence() (Smart-Language-Detection)."""

    def test_returns_tuple(self) -> None:
        """Always returns tuple (lang, confidence)."""
        result = detect_language_with_confidence("Hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_clear_german_high_confidence(self) -> None:
        """Clear German text has high confidence (> 0.7)."""
        lang, conf = detect_language_with_confidence(
            "Was ist die Hauptstadt von Frankreich?"
        )
        assert lang == "de"
        assert conf > 0.7

    def test_clear_english_high_confidence(self) -> None:
        """Clear English text has high confidence (> 0.7)."""
        lang, conf = detect_language_with_confidence("What is the capital of France?")
        assert lang == "en"
        assert conf > 0.7

    def test_script_detection_full_confidence(self) -> None:
        """Non-Latin script detection returns confidence 1.0."""
        lang, conf = detect_language_with_confidence("これはテストです")
        assert lang == "ja"
        assert conf == 1.0

    def test_short_ambiguous_low_confidence(self) -> None:
        """Very short ambiguous text has low confidence."""
        _, conf = detect_language_with_confidence("ok")
        assert conf < 0.7

    def test_empty_text_zero_confidence(self) -> None:
        """Empty text returns confidence 0.0."""
        lang, conf = detect_language_with_confidence("")
        assert lang == "en"
        assert conf == 0.0

    def test_confidence_range(self) -> None:
        """Confidence is always between 0.0 and 1.0."""
        texts = [
            "Hello",
            "Hallo wie geht es dir?",
            "I don't know what you mean",
            "",
            "ok",
            "Bonjour comment allez-vous?",
            "これはテストです",
            "이것은 테스트입니다",
        ]
        for text in texts:
            _, conf = detect_language_with_confidence(text)
            assert 0.0 <= conf <= 1.0, f"Confidence out of range for: {text}"

    def test_backwards_compatible_with_detect_language(self) -> None:
        """detect_language() returns same result as lang part."""
        texts = [
            "Ich habe heute viel gelernt",
            "I have learned a lot today",
            "Hola, como estas?",
            "",
            "これはテストです",
            "Це є тест",
        ]
        for text in texts:
            simple = detect_language(text)
            with_conf, _ = detect_language_with_confidence(text)
            assert simple == with_conf, f"Mismatch for: {text}"


@pytest.mark.unit
@pytest.mark.i18n
class TestCharHintDisambiguation:
    """Regression tests for shared diacritics disambiguation (2026-05-18).

    Verifies that languages sharing ä/ö (DE, SV, FI) or å/æ/ø (SV, DA, NB)
    are correctly disambiguated via marker-word priority over shared char hints.
    """

    def test_sv_with_shared_umlauts(self) -> None:
        """Swedish text with ä/ö must detect as 'sv', not 'de'."""
        text = "Berätta hela historien om NVIDIA fram till idag."
        result = detect_language(text)
        assert result == "sv", f"Expected 'sv', got '{result}' for: {text}"

    def test_sv_with_short_marker_phrase(self) -> None:
        """Short Swedish sentence with ö must detect as 'sv'."""
        text = "Vilka är fördelarna med öppen källkod?"
        result = detect_language(text)
        assert result == "sv", f"Expected 'sv', got '{result}' for: {text}"

    def test_de_clear_marker(self) -> None:
        """Clear German text without shared-char ambiguity."""
        text = "Was sind die Vorteile von Open-Source-Software?"
        result = detect_language(text)
        assert result == "de", f"Expected 'de', got '{result}' for: {text}"

    def test_da_with_shared_chars(self) -> None:
        """Danish text with shared markers must detect as 'da'."""
        text = "Hvad er fordelene ved open source-software?"
        result = detect_language(text)
        assert result == "da", f"Expected 'da', got '{result}' for: {text}"

    def test_nl_with_shared_chars(self) -> None:
        """Dutch text must detect as 'nl'."""
        text = "Wat zijn de voordelen van open source software?"
        result = detect_language(text)
        assert result == "nl", f"Expected 'nl', got '{result}' for: {text}"

    def test_no_with_shared_chars(self) -> None:
        """Norwegian text with å must detect as 'nb'."""
        text = "Hva er fordelene med åpen kildekode-programvare?"
        result = detect_language(text)
        assert result in ("nb", "no"), f"Expected 'nb'/'no', got '{result}' for: {text}"

    def test_fi_with_shared_chars(self) -> None:
        """Finnish text with ä/ö must detect as 'fi'."""
        text = "Mitkä ovat avoimen lähdekoodin ohjelmistojen edut?"
        result = detect_language(text)
        assert result == "fi", f"Expected 'fi', got '{result}' for: {text}"

    def test_en_no_diacritics(self) -> None:
        """English text without diacritics must detect as 'en'."""
        text = "What are the advantages of open-source software?"
        result = detect_language(text)
        assert result == "en", f"Expected 'en', got '{result}' for: {text}"

    def test_de_long_text_no_diacritics(self) -> None:
        """Longer German text without ä/ö/ü must still detect as 'de'."""
        text = (
            "Ich bin der Meinung dass wir hier nicht nur mit den "
            "Vorteilen sondern auch mit den Risiken von Open Source "
            "Software rechnen sollten wenn wir eine Entscheidung treffen"
        )
        result = detect_language(text)
        assert result == "de", f"Expected 'de', got '{result}' for: {text}"

    def test_short_text_de_confidence(self) -> None:
        """Short German text has medium confidence (marker-based)."""
        lang, conf = detect_language_with_confidence("ist das richtig")
        assert lang == "de", f"Expected 'de', got '{lang}'"
        assert conf > 0.0, "Confidence should be above zero"

    def test_short_text_sv_confidence(self) -> None:
        """Short Swedish text has medium confidence (marker-based)."""
        lang, conf = detect_language_with_confidence("är detta rätt")
        assert lang == "sv", f"Expected 'sv', got '{lang}'"
        assert conf > 0.0, "Confidence should be above zero"

    def test_mixed_de_sv_majority_wins(self) -> None:
        """Mixed text with majority Swedish markers detects as 'sv'."""
        text = (
            "Det var inte mycket att göra sedan dess men jag har "
            "försökt att hitta en lösning som alla kan acceptera"
        )
        result = detect_language(text)
        assert result == "sv", f"Expected 'sv', got '{result}' for: {text}"

    def test_de_with_exclusive_eszett(self) -> None:
        """German text with ß (DE-exclusive) must detect as 'de'."""
        text = "Die Straße ist sehr groß und schön"
        result = detect_language(text)
        assert result == "de", f"Expected 'de', got '{result}' for: {text}"

    def test_sv_long_nvidia_prompt(self) -> None:
        """The original bug: long SV prompt must NOT detect as 'de'."""
        text = (
            "Berätta hela historien om NVIDIA fram till idag. "
            "Var så utförlig som möjligt minst 4 000 ord."
        )
        result = detect_language(text)
        assert result == "sv", f"Expected 'sv', got '{result}' for: {text}"

    def test_da_with_oe_ligature(self) -> None:
        """Danish text with ø (DA/NB-exclusive) must detect as 'da'."""
        text = "Vi bør gøre det bedre næste gang"
        result = detect_language(text)
        assert result == "da", f"Expected 'da', got '{result}' for: {text}"

    def test_nb_with_oe_and_markers(self) -> None:
        """Norwegian text with ø and NB markers must detect as 'nb'."""
        text = "Vi bør gjøre det bedre neste gang fordi det ikke var bra"
        result = detect_language(text)
        assert result == "nb", f"Expected 'nb', got '{result}' for: {text}"
