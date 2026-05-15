"""Tests for domain.i18n: centralized i18n strings.

Validates that all i18n dicts cover the same set of language codes
and that get_text() works with all languages and fallback.
"""

from __future__ import annotations

import pytest

from domain.i18n import (
    ALL_LANGUAGES,
    BOOKMARK_DELETE_CONFIRM_TEXTS,
    BOOKMARK_LIST_EMPTY_TEXTS,
    BOOKMARK_REMOVED_TEXTS,
    BOOKMARK_SAVE_HINT_TEXTS,
    BOOKMARK_SAVED_TEXTS,
    FORGET_NOT_FOUND_TEXTS,
    FORGET_SUCCESS_TEXTS,
    FORGET_USAGE_TEXTS,
    INLINE_COMMAND_WARNING_TEXTS,
    LANG_CHANGED_TEXTS,
    MEMORY_EMPTY_TEXTS,
    MEMORY_HEADER_TEXTS,
    MEMORY_SEARCH_HEADER_TEXTS,
    MEMORY_SEARCH_NO_RESULTS_TEXTS,
    REMEMBER_SAVED_TEXTS,
    REMEMBER_USAGE_TEXTS,
    RESET_TEXTS,
    STATUS_TEXTS,
    get_status_text,
    get_text,
)


# All simple (flat) i18n dicts that must cover all 20 languages
_FLAT_DICTS: list[tuple[str, dict[str, str]]] = [
    ("RESET_TEXTS", RESET_TEXTS),
    ("REMEMBER_SAVED_TEXTS", REMEMBER_SAVED_TEXTS),
    ("REMEMBER_USAGE_TEXTS", REMEMBER_USAGE_TEXTS),
    ("FORGET_SUCCESS_TEXTS", FORGET_SUCCESS_TEXTS),
    ("FORGET_NOT_FOUND_TEXTS", FORGET_NOT_FOUND_TEXTS),
    ("FORGET_USAGE_TEXTS", FORGET_USAGE_TEXTS),
    ("MEMORY_EMPTY_TEXTS", MEMORY_EMPTY_TEXTS),
    ("MEMORY_HEADER_TEXTS", MEMORY_HEADER_TEXTS),
    ("MEMORY_SEARCH_NO_RESULTS_TEXTS", MEMORY_SEARCH_NO_RESULTS_TEXTS),
    ("MEMORY_SEARCH_HEADER_TEXTS", MEMORY_SEARCH_HEADER_TEXTS),
    ("BOOKMARK_SAVED_TEXTS", BOOKMARK_SAVED_TEXTS),
    ("BOOKMARK_REMOVED_TEXTS", BOOKMARK_REMOVED_TEXTS),
    ("BOOKMARK_SAVE_HINT_TEXTS", BOOKMARK_SAVE_HINT_TEXTS),
    ("BOOKMARK_LIST_EMPTY_TEXTS", BOOKMARK_LIST_EMPTY_TEXTS),
    ("BOOKMARK_DELETE_CONFIRM_TEXTS", BOOKMARK_DELETE_CONFIRM_TEXTS),
    ("LANG_CHANGED_TEXTS", LANG_CHANGED_TEXTS),
    ("INLINE_COMMAND_WARNING_TEXTS", INLINE_COMMAND_WARNING_TEXTS),
]


class TestI18nCoverage:
    """Every flat i18n dict must have all 20 language keys."""

    @pytest.mark.parametrize("name,d", _FLAT_DICTS, ids=[n for n, _ in _FLAT_DICTS])
    def test_all_languages_present(self, name: str, d: dict[str, str]) -> None:
        missing = set(ALL_LANGUAGES) - set(d.keys())
        assert not missing, f"{name} is missing languages: {missing}"

    @pytest.mark.parametrize("name,d", _FLAT_DICTS, ids=[n for n, _ in _FLAT_DICTS])
    def test_no_empty_values(self, name: str, d: dict[str, str]) -> None:
        for lang, text in d.items():
            assert text.strip(), f"{name}[{lang}] is empty"


class TestStatusTexts:
    """STATUS_TEXTS (nested dict) must cover all 20 languages per key."""

    def test_all_status_keys_have_all_languages(self) -> None:
        for key, texts in STATUS_TEXTS.items():
            missing = set(ALL_LANGUAGES) - set(texts.keys())
            assert not missing, f"STATUS_TEXTS[{key}] missing languages: {missing}"


class TestGetText:
    """get_text() helper function."""

    def test_known_language(self) -> None:
        result = get_text(RESET_TEXTS, "de")
        assert "zurückgesetzt" in result

    def test_english_fallback(self) -> None:
        result = get_text(RESET_TEXTS, "xx_unknown")
        assert "reset" in result.lower()

    def test_format_parameters(self) -> None:
        result = get_text(REMEMBER_SAVED_TEXTS, "en", entry_id="ep_abc123")
        assert "ep_abc123" in result

    def test_format_parameters_all_languages(self) -> None:
        for lang in ALL_LANGUAGES:
            result = get_text(REMEMBER_SAVED_TEXTS, lang, entry_id="test_id")
            assert "test_id" in result, f"Format failed for lang={lang}"


class TestGetStatusText:
    """get_status_text() helper function."""

    def test_german_thinking(self) -> None:
        text = get_status_text("thinking", "de")
        assert "Denke nach" in text

    def test_english_thinking(self) -> None:
        text = get_status_text("thinking", "en")
        assert "Thinking" in text

    def test_fallback_unknown_lang(self) -> None:
        text = get_status_text("thinking", "xx")
        # Should fall back to EN
        assert "Thinking" in text

    def test_memory_loaded_with_n(self) -> None:
        text = get_status_text("memory_loaded", "de", n=5)
        assert "5" in text

    def test_all_languages_thinking(self) -> None:
        for lang in ALL_LANGUAGES:
            text = get_status_text("thinking", lang)
            assert text and len(text) > 0, f"Empty thinking text for {lang}"


class TestForgetBracketStrip:
    """Test that bracket-wrapped entry IDs are handled correctly."""

    def test_bracket_strip(self) -> None:
        raw = "[ep_abc123]"
        cleaned = raw.strip().strip("[]")
        assert cleaned == "ep_abc123"

    def test_no_brackets(self) -> None:
        raw = "ep_abc123"
        cleaned = raw.strip().strip("[]")
        assert cleaned == "ep_abc123"

    def test_single_bracket(self) -> None:
        raw = "[ep_abc123"
        cleaned = raw.strip().strip("[]")
        assert cleaned == "ep_abc123"
