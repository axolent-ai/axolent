"""Tests für domain.onboarding: Wizard-Konfiguration und Sprach-Mapping.

Testet die Onboarding-Domain-Logik: Sprach-Definitionen, Keyboard-Layout,
Wizard-Texte in verschiedenen Sprachen.
"""

from __future__ import annotations

import pytest

from domain.onboarding import (
    LANGUAGE_KEYBOARD_ROWS,
    VALID_LANGUAGE_CODES,
    WIZARD_LANGUAGES,
    OnboardingState,
    get_auto_detect_text,
    get_language_name,
    get_lets_go_text,
    get_onboarding_hint_text,
    get_skip_wizard_text,
    get_step1_text,
    get_step2_text,
)


class TestWizardLanguages:
    """Tests für die Sprach-Definitionen."""

    def test_exactly_20_languages(self) -> None:
        """Es gibt genau 20 benannte Sprachen."""
        assert len(WIZARD_LANGUAGES) == 20

    def test_valid_codes_include_auto(self) -> None:
        """VALID_LANGUAGE_CODES enthält 'auto' plus alle 20 Sprachen."""
        assert "auto" in VALID_LANGUAGE_CODES
        assert len(VALID_LANGUAGE_CODES) == 21

    def test_all_wizard_languages_in_valid_codes(self) -> None:
        """Alle WIZARD_LANGUAGES sind in VALID_LANGUAGE_CODES."""
        for code in WIZARD_LANGUAGES:
            assert code in VALID_LANGUAGE_CODES

    @pytest.mark.parametrize(
        "code,name",
        [
            ("de", "Deutsch"),
            ("en", "English"),
            ("fr", "Français"),
            ("es", "Español"),
            ("it", "Italiano"),
            ("pt", "Português"),
            ("nl", "Nederlands"),
            ("pl", "Polski"),
            ("sv", "Svenska"),
            ("tr", "Türkçe"),
            ("ru", "Русский"),
            ("uk", "Українська"),
            ("zh", "中文"),
            ("ja", "日本語"),
            ("ko", "한국어"),
            ("ar", "العربية"),
            ("hi", "हिन्दी"),
            ("id", "Bahasa Indo."),
            ("th", "ภาษาไทย"),
            ("vi", "Tiếng Việt"),
        ],
    )
    def test_all_20_languages_valid(self, code: str, name: str) -> None:
        """Jede der 20 Sprachen ist mit korrektem Namen definiert."""
        assert code in WIZARD_LANGUAGES
        assert WIZARD_LANGUAGES[code] == name
        assert code in VALID_LANGUAGE_CODES


class TestKeyboardLayout:
    """Tests für das Keyboard-Layout."""

    def test_5_rows_of_4(self) -> None:
        """Keyboard hat 5 Reihen mit je 4 Buttons."""
        assert len(LANGUAGE_KEYBOARD_ROWS) == 5
        for row in LANGUAGE_KEYBOARD_ROWS:
            assert len(row) == 4

    def test_all_languages_covered(self) -> None:
        """Alle 20 Sprachen kommen im Keyboard-Layout vor."""
        flat = [code for row in LANGUAGE_KEYBOARD_ROWS for code in row]
        assert len(flat) == 20
        assert set(flat) == set(WIZARD_LANGUAGES.keys())

    def test_no_duplicates(self) -> None:
        """Keine Sprache kommt doppelt vor."""
        flat = [code for row in LANGUAGE_KEYBOARD_ROWS for code in row]
        assert len(flat) == len(set(flat))


class TestWizardTexts:
    """Tests für Wizard-UI-Texte."""

    def test_step1_text_de(self) -> None:
        """Step 1 Text in Deutsch."""
        text = get_step1_text("de")
        assert "Willkommen" in text
        assert "Sprache" in text

    def test_step1_text_en(self) -> None:
        """Step 1 Text in Englisch."""
        text = get_step1_text("en")
        assert "Welcome" in text
        assert "language" in text

    def test_step1_text_unknown_falls_back_to_en(self) -> None:
        """Unbekannte Sprache fällt auf EN zurück."""
        text = get_step1_text("xx")
        assert "Welcome" in text

    @pytest.mark.parametrize("lang", list(WIZARD_LANGUAGES.keys()))
    def test_step2_text_exists_for_all_languages(self, lang: str) -> None:
        """Step 2 Text existiert für alle 20 Sprachen."""
        text = get_step2_text(lang, "TestLang")
        assert "TestLang" in text
        assert len(text) > 20  # nicht leer

    def test_step2_text_placeholder_replaced(self) -> None:
        """Step 2 Text ersetzt {lang_name} korrekt."""
        text = get_step2_text("de", "Deutsch")
        assert "Deutsch" in text
        assert "{lang_name}" not in text

    def test_auto_detect_text_de(self) -> None:
        """Auto-Detect Text in DE."""
        text = get_auto_detect_text("de")
        assert "Automatisch" in text
        assert "Empfohlen" in text

    def test_auto_detect_text_en(self) -> None:
        """Auto-Detect Text in EN."""
        text = get_auto_detect_text("en")
        assert "Auto-detect" in text

    def test_skip_wizard_text(self) -> None:
        """Skip-Text existiert in DE und EN."""
        assert (
            "überspringen" in get_skip_wizard_text("de").lower()
            or "skip" in get_skip_wizard_text("de").lower()
        )
        assert "Skip" in get_skip_wizard_text("en")

    def test_lets_go_text(self) -> None:
        """Los-geht's-Text existiert."""
        assert len(get_lets_go_text("de")) > 0
        assert len(get_lets_go_text("en")) > 0

    def test_onboarding_hint_text(self) -> None:
        """Onboarding-Hinweis enthält /onboarding."""
        assert "/onboarding" in get_onboarding_hint_text("de")
        assert "/onboarding" in get_onboarding_hint_text("en")


class TestLanguageName:
    """Tests für get_language_name."""

    def test_known_language(self) -> None:
        """Bekannte Sprache gibt nativen Namen zurück."""
        assert get_language_name("de") == "Deutsch"
        assert get_language_name("ja") == "日本語"

    def test_auto(self) -> None:
        """'auto' gibt 'Auto-detect' zurück."""
        assert get_language_name("auto") == "Auto-detect"

    def test_unknown(self) -> None:
        """Unbekannter Code gibt den Code selbst zurück."""
        assert get_language_name("xx") == "xx"


class TestOnboardingState:
    """Tests für OnboardingState dataclass."""

    def test_defaults(self) -> None:
        """Standard-Werte sind korrekt."""
        state = OnboardingState(user_id=123)
        assert state.user_id == 123
        assert state.onboarded is False
        assert state.wizard_lang is None
        assert state.skip_count == 0
        assert state.hint_shown is False

    def test_custom_values(self) -> None:
        """Benutzerdefinierte Werte."""
        state = OnboardingState(
            user_id=456,
            onboarded=True,
            wizard_lang="en",
            skip_count=3,
            hint_shown=True,
        )
        assert state.onboarded is True
        assert state.wizard_lang == "en"
        assert state.skip_count == 3
        assert state.hint_shown is True

    def test_frozen(self) -> None:
        """State ist immutable (frozen dataclass)."""
        state = OnboardingState(user_id=1)
        with pytest.raises(AttributeError):
            state.onboarded = True  # type: ignore[misc]
