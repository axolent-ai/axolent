"""Backend protocol and implementations for language detection in the Verifier.

This module isolates the actual language detection library choice behind a
Protocol interface. The ResponseLanguageVerifier and StreamGuard speak ONLY
to LanguageDetectorBackend, never to langdetect or domain.language directly.

Rationale (Codex architecture rule, 2026-05-20):
- domain.language is calibrated for short user inputs (marker-word heuristics).
  It misdetects long LLM outputs (e.g. Dutch as English) because its scoring
  windows are tuned for 5-50 word fragments.
- langdetect (or future Lingua/fast-langdetect) uses n-gram profiles optimized
  for longer text and provides probability distributions, not single-winner results.
- By hiding the backend behind a Protocol, we can swap implementations without
  touching the Verifier or StreamGuard logic.

Implementations:
- LangdetectBackend: Phase 1 default, uses the langdetect library.
- DomainLanguageBackend: Fallback for environments without langdetect installed,
  or as a comparison baseline. NOT the default in production.
"""

from __future__ import annotations

from typing import Protocol


class LanguageDetectorBackend(Protocol):
    """Backend protocol for language detection in the Verifier.

    Implementations must be drop-in-replaceable (e.g. langdetect today,
    Lingua or fast-langdetect in Phase 2).
    """

    def detect_distribution(self, text: str) -> dict[str, float]:
        """Returns {lang_code: probability} for the given text.

        Probabilities should sum to ~1.0. Returns empty dict if text
        is too short or detection fails. Language codes must follow
        AXOLENT conventions (e.g. 'zh' not 'zh-cn', 'nb' not 'no').
        """
        ...


class LangdetectBackend:
    """Phase 1 backend using the langdetect library.

    langdetect is non-deterministic by default (random seed per call).
    We pin the seed to 0 for reproducible results in tests and production.
    """

    def __init__(self) -> None:
        """Initialize with deterministic seed."""
        from langdetect import DetectorFactory

        DetectorFactory.seed = 0

    def detect_distribution(self, text: str) -> dict[str, float]:
        """Detect language distribution using langdetect.

        Args:
            text: Text to analyze.

        Returns:
            Dict of {lang_code: probability}. Empty dict on failure.
        """
        from langdetect import detect_langs

        try:
            results = detect_langs(text)
            return {self._normalize(r.lang): r.prob for r in results}
        except Exception:
            return {}

    @staticmethod
    def _normalize(lang_code: str) -> str:
        """Normalize language codes to AXOLENT conventions.

        Mappings:
        - zh-cn, zh-tw -> zh (unified Chinese)
        - no -> nb (Norwegian Bokmal, ISO standard)
        """
        mapping = {"zh-cn": "zh", "zh-tw": "zh", "no": "nb"}
        return mapping.get(lang_code, lang_code)


class DomainLanguageBackend:
    """Fallback backend using the in-house domain.language detector.

    Kept as a fallback for environments without langdetect installed,
    or as a comparison baseline. NOT the default in production.

    WARNING: This backend is calibrated for short user inputs (5-50 words).
    It will misdetect long LLM outputs. Only use as a fallback when
    langdetect is unavailable.
    """

    def detect_distribution(self, text: str) -> dict[str, float]:
        """Detect language using in-house heuristic detector.

        Args:
            text: Text to analyze.

        Returns:
            Single-entry dict {detected_lang: confidence}.
            Empty dict if detection fails.
        """
        from domain.language import detect_language_with_confidence

        detected, confidence = detect_language_with_confidence(text)
        return {detected: confidence} if detected else {}
