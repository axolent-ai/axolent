"""Tests for i18n key parity, placeholder parity, and t() function behavior.

Covers:
    - All locales have the same keys as en.json
    - All locales have the same placeholders per key
    - t() returns correct strings with kwargs, fallback behavior, strict mode
    - Smoke tests for key commands across multiple languages
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_LOCALES_DIR = Path(__file__).parent.parent.parent / "i18n" / "locales"
_META_FILE = _LOCALES_DIR / "_meta.json"
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _load_locale(lang: str) -> dict:
    """Load a locale JSON file and return the keys dict."""
    path = _LOCALES_DIR / f"{lang}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("keys", {})


def _get_all_locale_codes() -> list[str]:
    """Return all locale codes from _meta.json."""
    with open(_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return list(meta.keys())


# ---------------------------------------------------------------------------
# Test: Key Parity
# ---------------------------------------------------------------------------


class TestI18nKeyParity:
    """All locales must have exactly the same keys as en.json."""

    def test_all_locales_have_same_keys_as_en(self):
        en_keys = set(_load_locale("en").keys())
        all_codes = _get_all_locale_codes()

        for code in all_codes:
            locale_keys = set(_load_locale(code).keys())
            missing = en_keys - locale_keys
            extra = locale_keys - en_keys
            assert not missing, f"{code}.json is missing keys: {missing}"
            assert not extra, f"{code}.json has extra keys: {extra}"

    def test_en_has_expected_key_count(self):
        en_keys = _load_locale("en")
        # 101 keys as of 2026-05-16 (initial 91 + 10 rate_limit/setlimit keys)
        assert len(en_keys) >= 91, f"Expected at least 91 keys, got {len(en_keys)}"

    def test_all_20_locales_present(self):
        all_codes = _get_all_locale_codes()
        assert len(all_codes) == 20


# ---------------------------------------------------------------------------
# Test: Placeholder Parity
# ---------------------------------------------------------------------------


class TestI18nPlaceholderParity:
    """All locales must have the same placeholder names per key as en.json."""

    def test_placeholders_match_en(self):
        en_keys = _load_locale("en")
        all_codes = _get_all_locale_codes()

        for code in all_codes:
            if code == "en":
                continue
            locale_keys = _load_locale(code)
            for key, en_entry in en_keys.items():
                en_placeholders = set(_PLACEHOLDER_RE.findall(en_entry["text"]))
                locale_entry = locale_keys.get(key)
                if locale_entry is None:
                    continue  # covered by parity test
                locale_placeholders = set(_PLACEHOLDER_RE.findall(locale_entry["text"]))
                assert en_placeholders == locale_placeholders, (
                    f"{code}.json key '{key}': "
                    f"EN placeholders {en_placeholders} != "
                    f"{code} placeholders {locale_placeholders}"
                )


# ---------------------------------------------------------------------------
# Test: t() Function
# ---------------------------------------------------------------------------


class TestI18nTFunction:
    """Tests for the main t() translation function."""

    def test_t_returns_string_for_known_key(self):
        from i18n.domain.i18n import t

        result = t("reset.confirmation", "en")
        assert "reset" in result.lower() or "fresh" in result.lower()

    def test_t_with_kwargs(self):
        from i18n.domain.i18n import t

        result = t("remember.saved", "en", entry_id="abc123")
        assert "abc123" in result

    def test_t_fallback_to_en_for_unsupported_lang(self):
        from i18n.domain.i18n import t

        # "xx" is not a supported language
        result = t("reset.confirmation", "xx")
        en_result = t("reset.confirmation", "en")
        assert result == en_result

    def test_t_returns_bracket_key_for_missing_key(self):
        from i18n.domain.i18n import t

        result = t("totally.nonexistent.key", "en")
        assert result == "[totally.nonexistent.key]"

    def test_t_strict_mode_raises(self, monkeypatch):
        import i18n.domain.i18n as i18n_mod

        monkeypatch.setattr(i18n_mod, "_STRICT_MODE", True)

        with pytest.raises(KeyError):
            i18n_mod.t("totally.nonexistent.key", "en")

        # Cleanup
        monkeypatch.setattr(i18n_mod, "_STRICT_MODE", False)

    def test_t_all_languages_return_nonempty(self):
        from i18n.domain.i18n import t

        all_codes = _get_all_locale_codes()
        for code in all_codes:
            result = t("reset.confirmation", code)
            assert len(result) > 0, f"t('reset.confirmation', '{code}') returned empty"


# ---------------------------------------------------------------------------
# Smoke Tests: Key commands across languages
# ---------------------------------------------------------------------------


class TestI18nSmokeCommands:
    """Smoke tests: commands return correctly formatted strings."""

    @pytest.mark.parametrize("lang", ["de", "en", "nl", "fr", "ja", "ar"])
    def test_help_returns_html(self, lang: str):
        from i18n.domain.i18n import t

        title = t("help.title", lang)
        body = t("help.body", lang)
        assert "<b>" in title or "<b>" in body
        assert "/help" in body
        assert "/reset" in body

    @pytest.mark.parametrize("lang", ["de", "en", "nl", "fr", "ja", "ar"])
    def test_setmodel_usage_hint(self, lang: str):
        from i18n.domain.i18n import t

        result = t("setmodel.usage_hint", lang, slots="chat, code, creative")
        assert "/setmodel" in result
        assert "chat, code, creative" in result

    @pytest.mark.parametrize("lang", ["de", "en", "nl", "fr", "ja", "ar"])
    def test_reset_confirmation(self, lang: str):
        from i18n.domain.i18n import t

        result = t("reset.confirmation", lang)
        assert len(result) > 5  # reasonable string length

    @pytest.mark.parametrize("lang", ["de", "en", "id", "th", "vi"])
    def test_new_locales_work(self, lang: str):
        """Tests the newly added id/th/vi locales alongside existing ones."""
        from i18n.domain.i18n import t

        result = t("bookmark.saved", lang)
        assert len(result) > 2

        result2 = t("settings.main_title", lang)
        assert len(result2) > 2
