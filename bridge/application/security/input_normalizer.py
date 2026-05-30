"""Central security input normalizer (two-level architecture).

Two normalization levels for security checks:

Level 1 (basic): normalize_for_security_check()
  1. NFKC normalization (Fullwidth chars like U+FF21 -> A)
  2. Strip Cf (Format) category characters (Zero-Width, Bidi, etc.)
  Preserves non-Latin scripts (Russian, Hindi, Thai) for native-script
  pattern matching (InjectionDetector multilingual patterns).

Level 2 (aggressive): normalize_aggressive()
  1. NFD decompose (isolates combining marks from base chars)
  2. Strip Mn (Combining Mark) category including Variation Selectors
  3. Cross-Script Confusables folding (Cyrillic/Greek -> Latin)
  4. NFKC normalize (final compatibility form)
  5. Strip Cf (Format/Zero-Width) category
  Catches mixed-script bypass attacks (Cyrillic 'a' in Latin text)
  AND combining-diacritic bypasses (U+0308 on 'o' to evade 'ignore').
  CAUTION: destroys non-Latin scripts. Used by SecretScanner (Latin
  patterns only), LeakageFilter, HealthcareFilter, and NudgeFilter.

InjectionDetector uses BOTH levels (two-pass matching) to catch both
native-script injections AND mixed-script bypasses.

Usage:
    from application.security.input_normalizer import (
        normalize_for_security_check,   # basic: preserves all scripts
        normalize_aggressive,           # aggressive: folds confusables
    )
"""

from __future__ import annotations

import unicodedata

# ---------------------------------------------------------------------------
# Cross-Script Confusables Map (UTS-39 subset: Cyrillic + Greek -> Latin)
#
# Hardcoded for the security-relevant Latin/Cyrillic/Greek character set.
# Each entry maps a visually confusable character to its Latin equivalent.
# Source: Unicode Technical Standard #39, confusables.txt
# No external dependency required (stdlib-only).
# ---------------------------------------------------------------------------
_CONFUSABLES_MAP: dict[str, str] = {
    # --- Cyrillic lowercase -> Latin lowercase ---
    "а": "a",  # Cyrillic Small Letter A
    "е": "e",  # Cyrillic Small Letter Ie
    "і": "i",  # Cyrillic Small Letter Byelorussian-Ukrainian I
    "о": "o",  # Cyrillic Small Letter O
    "р": "p",  # Cyrillic Small Letter Er
    "с": "c",  # Cyrillic Small Letter Es
    "у": "y",  # Cyrillic Small Letter U (visually ~ y in some fonts)
    "х": "x",  # Cyrillic Small Letter Ha
    "ѕ": "s",  # Cyrillic Small Letter Dze
    "ј": "j",  # Cyrillic Small Letter Je
    "һ": "h",  # Cyrillic Small Letter Shha
    "ї": "i",  # Cyrillic Small Letter Yi (looks like i with diaeresis)
    # --- Cyrillic uppercase -> Latin uppercase ---
    "А": "A",  # Cyrillic Capital Letter A
    "В": "B",  # Cyrillic Capital Letter Ve
    "Е": "E",  # Cyrillic Capital Letter Ie
    "К": "K",  # Cyrillic Capital Letter Ka
    "М": "M",  # Cyrillic Capital Letter Em
    "Н": "H",  # Cyrillic Capital Letter En
    "О": "O",  # Cyrillic Capital Letter O
    "Р": "P",  # Cyrillic Capital Letter Er
    "С": "C",  # Cyrillic Capital Letter Es
    "Т": "T",  # Cyrillic Capital Letter Te
    "Х": "X",  # Cyrillic Capital Letter Ha
    "Ѕ": "S",  # Cyrillic Capital Letter Dze
    "Ј": "J",  # Cyrillic Capital Letter Je
    "Һ": "H",  # Cyrillic Capital Letter Shha
    # --- Greek lowercase -> Latin lowercase ---
    "ο": "o",  # Greek Small Letter Omicron
    "α": "a",  # Greek Small Letter Alpha (visually similar in many fonts)
    "ε": "e",  # Greek Small Letter Epsilon
    "ι": "i",  # Greek Small Letter Iota
    "κ": "k",  # Greek Small Letter Kappa
    "ν": "v",  # Greek Small Letter Nu (visually ~ v)
    "ρ": "p",  # Greek Small Letter Rho
    "τ": "t",  # Greek Small Letter Tau (visually ~ t in some fonts)
    "υ": "u",  # Greek Small Letter Upsilon
    "χ": "x",  # Greek Small Letter Chi
    # --- Greek uppercase -> Latin uppercase ---
    "Α": "A",  # Greek Capital Letter Alpha
    "Β": "B",  # Greek Capital Letter Beta
    "Ε": "E",  # Greek Capital Letter Epsilon
    "Η": "H",  # Greek Capital Letter Eta
    "Ι": "I",  # Greek Capital Letter Iota
    "Κ": "K",  # Greek Capital Letter Kappa
    "Μ": "M",  # Greek Capital Letter Mu
    "Ν": "N",  # Greek Capital Letter Nu
    "Ο": "O",  # Greek Capital Letter Omicron
    "Ρ": "P",  # Greek Capital Letter Rho
    "Τ": "T",  # Greek Capital Letter Tau
    "Υ": "Y",  # Greek Capital Letter Upsilon
    "Χ": "X",  # Greek Capital Letter Chi
    "Ζ": "Z",  # Greek Capital Letter Zeta
}


def normalize_for_security_check(text: str) -> str:
    """Normalize text for security pattern matching (basic level).

    Steps:
      1. NFKC normalization (compatibility decomposition + canonical composition)
      2. Strip all Unicode Cf (Format) category characters

    This is the BASIC normalization level. It does NOT fold confusables or
    strip combining marks (Mn), preserving non-Latin scripts (Russian,
    Hindi, Thai, etc.) for native-script pattern matching.

    For mixed-script bypass detection (Latin patterns with Cyrillic/Greek
    substitutions), use normalize_aggressive() which adds confusables
    folding and Mn stripping.

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


def normalize_aggressive(text: str) -> str:
    """Aggressive normalization: NFD decompose, Mn strip, confusables fold, NFKC.

    Steps (Decompose-First architecture, Phase 1.5.2-Polish):
      1. NFD decompose (separates pre-composed chars into base + combining marks)
      2. Strip Mn (Combining Mark) category (now isolated by NFD)
      3. Cross-Script Confusables folding (Cyrillic/Greek -> Latin via UTS-39)
      4. NFKC normalize (final compatibility form)
      5. Strip Cf (Format/Zero-Width) category

    Why NFD first (Codex finding):
      Old order (NFKC -> confusables -> Mn-strip) failed on combining diacritics:
        'igno' + U+0308 -> NFKC composes to pre-composed char (no longer Mn)
        -> Mn-strip is a no-op -> pattern 'ignore' cannot match 'ignore with umlaut'.
      New order decomposes FIRST so combining marks are always isolated and stripped.

    CAUTION: This destroys non-Latin scripts (Russian, Hindi, Thai).
    Modules that need native-script detection (InjectionDetector multilang
    patterns) must check BOTH normalize_for_security_check() AND
    normalize_aggressive() results.

    Args:
        text: Raw user input.

    Returns:
        Aggressively normalized text for mixed-script bypass detection.
    """
    if not text:
        return text

    # Step 1: NFD decompose - separates pre-composed chars into base + Mn
    normalized = unicodedata.normalize("NFD", text)

    # Step 2: Strip Mn (Combining Mark) category characters
    # Now isolated by NFD: covers combining diacritics (U+0300..U+036F),
    # combining marks, and Variation Selectors (U+FE00..U+FE0F category Mn).
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    # Step 3: Cross-Script Confusables folding (UTS-39 subset)
    normalized = "".join(_CONFUSABLES_MAP.get(ch, ch) for ch in normalized)

    # Step 4: NFKC normalize - final compatibility form
    normalized = unicodedata.normalize("NFKC", normalized)

    # Step 5: Strip Cf (Format/Zero-Width) category
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")

    return cleaned
