"""Central security input normalizer.

Normalizes text before any security check (InjectionDetector, SecretScanner,
PrivacyPipeline) to prevent bypass via Unicode tricks.

Two-step normalization:
  1. NFKC normalization: handles Compatibility Forms (Fullwidth chars like
     U+FF21 -> A) and canonical decomposition/composition. Does NOT fold
     Cross-Script Confusables (e.g. Cyrillic U+0430 remains distinct from
     Latin 'a'). Phase 1.5 plans UTS-39 Confusables-Skeleton for that.
  2. Strip Unicode format characters (category Cf): Zero-Width Space (U+200B),
     Zero-Width Non-Joiner (U+200C), Zero-Width Joiner (U+200D), BOM (U+FEFF),
     Word Joiner (U+2060), Bidi markers, and all other Cf code points.
     Does NOT strip Variation Selectors (category Mn) or combining marks.

Why Cf stripping after NFKC:
  NFKC does NOT remove Zero-Width characters. An attacker can insert U+200B
  between 'ignore' and 'all' to bypass regex patterns. Stripping Cf after
  NFKC closes this gap.

Usage:
    from application.security.input_normalizer import normalize_for_security_check
    cleaned = normalize_for_security_check(user_text)
    # Now run regex patterns against 'cleaned'
"""

from __future__ import annotations

import unicodedata


def normalize_for_security_check(text: str) -> str:
    """Normalize text for security pattern matching.

    Steps:
      1. NFKC normalization (compatibility decomposition + canonical composition)
      2. Remove all Unicode Cf (Format) category characters

    Args:
        text: Raw user input.

    Returns:
        Normalized text safe for regex pattern matching.
    """
    if not text:
        return text

    # Step 1: NFKC normalization (handles compatibility chars like Fullwidth)
    normalized = unicodedata.normalize("NFKC", text)

    # Step 2: Strip all Cf (Format) category characters
    # This covers: U+200B (ZWSP), U+200C (ZWNJ), U+200D (ZWJ),
    # U+FEFF (BOM/ZWNBS), U+2060 (Word Joiner), U+202A-202E (Bidi),
    # U+2066-2069 (Bidi Isolates), U+00AD (Soft Hyphen), and more.
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")

    return cleaned
