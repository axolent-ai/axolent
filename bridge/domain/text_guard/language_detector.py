"""Language detection wrapper for Text Guard.

Thin adapter around domain.language to provide text-guard-specific
detection with fallback behavior for unsupported languages.
"""

from __future__ import annotations

import logging

from domain.language import detect_language, detect_language_with_confidence
from domain.text_guard.rules_registry import list_languages

log = logging.getLogger(__name__)


def detect_for_text_guard(text: str) -> str | None:
    """Detect language and return it only if text-guard rules exist.

    Args:
        text: Input text to detect language for.

    Returns:
        ISO 639-1 language code if rules exist, None otherwise.
    """
    lang = detect_language(text)
    available = list_languages()
    if lang in available:
        return lang
    return None


def detect_for_text_guard_with_confidence(
    text: str,
) -> tuple[str | None, float]:
    """Detect language with confidence, only if text-guard rules exist.

    Args:
        text: Input text to detect language for.

    Returns:
        Tuple of (language code or None, confidence 0.0..1.0).
    """
    lang, confidence = detect_language_with_confidence(text)
    available = list_languages()
    if lang in available:
        return lang, confidence
    return None, 0.0
