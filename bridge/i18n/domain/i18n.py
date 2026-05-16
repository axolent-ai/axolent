"""Centralized JSON-based i18n system for Axolent.

Architecture:
    - en.json is the MASTER: only source of truth for new keys
    - All 20 locale files must have exactly the same keys as en.json
    - Each key has a source_hash (sha256[:16] of en text) for staleness detection
    - No silent English fallback for supported languages (logs error, returns EN as emergency)

Public API:
    t(key, lang, **kwargs)           - Main translation function
    get_supported_languages()        - Returns list of language metadata dicts
    is_supported(lang)               - Check if language code is supported
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOCALES_DIR = Path(__file__).parent.parent / "locales"
_META_FILE = _LOCALES_DIR / "_meta.json"

# Environment flag: when set to "1", missing keys raise instead of fallback
_STRICT_MODE = os.environ.get("I18N_STRICT", "0") == "1"

# Placeholder pattern for validation
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# ---------------------------------------------------------------------------
# Cache (loaded once at import time)
# ---------------------------------------------------------------------------

_locales: dict[str, dict[str, dict[str, Any]]] = {}
_meta: dict[str, dict[str, Any]] = {}
_supported_codes: frozenset[str] = frozenset()


def _compute_hash(text: str) -> str:
    """Compute source_hash: sha256 of text, first 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_locales() -> None:
    """Load all locale JSON files into the cache.

    Called once at module import. Subsequent calls are no-ops
    unless _locales is cleared (for testing).
    """
    global _locales, _meta, _supported_codes

    if not _LOCALES_DIR.exists():
        log.warning("i18n locales directory not found: %s", _LOCALES_DIR)
        return

    # Load meta
    if _META_FILE.exists():
        with open(_META_FILE, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        _supported_codes = frozenset(_meta.keys())
    else:
        log.warning("i18n _meta.json not found: %s", _META_FILE)

    # Load each locale file
    for json_file in sorted(_LOCALES_DIR.glob("*.json")):
        if json_file.name.startswith("_"):
            continue  # skip _meta.json
        lang_code = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            _locales[lang_code] = data.get("keys", {})
        except (json.JSONDecodeError, OSError) as e:
            log.error("Failed to load locale %s: %s", json_file.name, e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def t(key: str, lang: str, **kwargs: Any) -> str:
    """Main translation function.

    Looks up the key in the specified language's locale.
    Formats with provided kwargs.

    Fallback behavior:
        - If lang not supported or key missing: log error, return EN text
        - In strict mode (I18N_STRICT=1): raises KeyError instead

    Args:
        key: Dot-separated translation key (e.g. "reset.confirmation")
        lang: ISO 639-1 language code (e.g. "nl", "de", "en")
        **kwargs: Format parameters for placeholders

    Returns:
        Translated and formatted string.

    Raises:
        KeyError: In strict mode, if key is missing for a supported language.
    """
    # Try requested language
    locale_data = _locales.get(lang)
    if locale_data is not None:
        entry = locale_data.get(key)
        if entry is not None:
            text = entry.get("text", "")
            if kwargs:
                try:
                    return text.format(**kwargs)
                except (KeyError, IndexError, ValueError):
                    log.warning(
                        "i18n format error: key=%s lang=%s kwargs=%s",
                        key,
                        lang,
                        kwargs,
                    )
                    return text
            return text

    # Key missing in requested language
    if lang != "en" and lang in _supported_codes:
        # This is a supported language but key is missing: error condition
        if _STRICT_MODE:
            raise KeyError(f"i18n key '{key}' missing for supported language '{lang}'")
        log.error(
            "i18n MISSING KEY: key='%s' lang='%s' (falling back to EN)",
            key,
            lang,
        )

    # Fallback to EN
    en_data = _locales.get("en")
    if en_data is not None:
        entry = en_data.get(key)
        if entry is not None:
            text = entry.get("text", "")
            if kwargs:
                try:
                    return text.format(**kwargs)
                except (KeyError, IndexError, ValueError):
                    return text
            return text

    # Key not found anywhere
    if _STRICT_MODE:
        raise KeyError(f"i18n key '{key}' not found in any locale")
    log.error("i18n key '%s' not found in any locale", key)
    return f"[{key}]"


def get_supported_languages() -> list[dict[str, Any]]:
    """Returns list of supported language metadata.

    Each entry: {"code": "de", "native": "Deutsch", "english": "German", "rtl": false}
    """
    result = []
    for code, info in _meta.items():
        result.append(
            {
                "code": code,
                "native": info.get("native", code),
                "english": info.get("english", code),
                "rtl": info.get("rtl", False),
            }
        )
    return result


def is_supported(lang: str) -> bool:
    """Check if a language code is supported."""
    return lang in _supported_codes


def get_all_keys(lang: str = "en") -> set[str]:
    """Returns all keys for a locale (for validation scripts)."""
    locale_data = _locales.get(lang, {})
    return set(locale_data.keys())


def get_entry(key: str, lang: str) -> dict[str, Any] | None:
    """Returns the full entry dict for a key (for validation scripts)."""
    locale_data = _locales.get(lang)
    if locale_data is None:
        return None
    return locale_data.get(key)


def get_placeholders(text: str) -> set[str]:
    """Extract placeholder names from a text template."""
    return set(_PLACEHOLDER_RE.findall(text))


def reload_locales() -> None:
    """Force reload all locales (for testing)."""
    global _locales, _meta, _supported_codes
    _locales.clear()
    _meta.clear()
    _supported_codes = frozenset()
    _load_locales()


# ---------------------------------------------------------------------------
# Backward-compatibility bridge (used during migration)
# ---------------------------------------------------------------------------


def get_text(texts: dict[str, str], lang: str, **kwargs: Any) -> str:
    """Legacy get_text bridge for gradual migration.

    This function works with the old dict-based approach.
    Will be removed once all call sites use t() directly.
    """
    template = texts.get(lang, texts.get("en", ""))
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return template


def get_status_text(key: str, lang: str = "de", **kwargs: Any) -> str:
    """Get status text via the new t() system.

    Maps old status keys to new i18n keys:
        "memory_loading" -> "status.memory_loading"
        "thinking" -> "status.thinking"
        etc.
    """
    i18n_key = f"status.{key}"
    result = t(i18n_key, lang, **kwargs)
    # If key not found (returns [key]), try old fallback
    if result.startswith("[") and result.endswith("]"):
        return key
    return result


# ---------------------------------------------------------------------------
# Load on import
# ---------------------------------------------------------------------------

_load_locales()
