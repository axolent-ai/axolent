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

Phase 2 migration (Step 4/4):
    HC-R4: LangdetectBackend._normalize() now delegates to
    LanguageRegistry.resolve_backend_code(). No more hardcoded
    mapping dict in this module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from application.language.registry import InMemoryLanguageRegistry


@runtime_checkable
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


# Module-level registry for code normalization (HC-R4).
# Read-only and thread-safe. Used by LangdetectBackend._normalize().
_registry = InMemoryLanguageRegistry()


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
        """Normalize language codes to AXOLENT conventions (HC-R4).

        Delegates to LanguageRegistry.resolve_backend_code() for
        centralized backend-code normalization. No hardcoded mapping
        dict in this module.

        Mappings (defined in registry):
        - zh-cn, zh-tw -> zh (unified Chinese)
        - no -> nb (Norwegian Bokmal, ISO standard)
        """
        return _registry.resolve_backend_code(lang_code)


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
