"""Language Registry: central, read-only source of truth for language metadata.

This module provides the canonical registry of all languages supported by
the AXOLENT Language Control Plane (LCP). Every LCP component that needs
language metadata (display names, detection tiers, marker words, backend
code mappings) MUST query the registry instead of maintaining its own data.

Architecture rules:
- The registry is read-only at runtime. No filesystem IO, no DB lookups.
- New languages are added by editing _build_default_entries() only.
  No other module (Verifier, Contract, StreamGuard) needs to change.
- Backend-specific codes (e.g. langdetect's "no" for Norwegian) are
  normalized to canonical AXOLENT codes via resolve_backend_code().

Implementation choices (IC-R* from Spec):
- IC-R1: Entries stored as dict[str, LanguageRegistryEntry] for O(1) lookup.
- IC-R2: get() is case-insensitive (normalizes to lowercase before lookup).
- IC-R3: marker_words are a curated subset (10-20 high-signal markers per
  language), not the full set from domain/language.py.
- IC-R4: notes field is populated for languages with detection edge cases.
- IC-R6: Logging at DEBUG level for registry init, WARNING for unknown codes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Protocol

log = logging.getLogger(__name__)


class DetectionTier(Enum):
    """Reliability tier for language detection.

    HIGH: Non-Latin script languages (deterministic Unicode detection)
          or Latin languages with highly distinctive markers.
          Expected accuracy: >95% at >=10 chars.
    MEDIUM: Latin-script languages with moderate marker overlap
            (e.g. DA/NB, PT/ES). Expected accuracy: >85% at >=20 chars.
    LOW: Languages where detection is unreliable below ~50 chars
         or where backend coverage is inconsistent.
         Expected accuracy: >70% at >=50 chars.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class LanguageRegistryEntry:
    """Immutable metadata for a single supported language.

    Attributes:
        code: ISO-639-1 two-letter code (canonical, lowercase).
        name: English display name.
        native_name: Self-referential name in the language itself.
        detection_tier: Reliability classification for detection.
        script: Primary Unicode script family (latin, cyrillic, arabic,
            cjk, devanagari, thai, hangul, kana).
        marker_words: Frozen set of high-signal marker words for
            short-text heuristic detection. Empty for script-detected
            languages.
        langdetect_code: Code returned by langdetect (if different
            from canonical code, e.g. "no" for Norwegian). None if
            identical to code.
        min_chars_reliable: Minimum character count for reliable
            detection by the primary backend.
        notes: Optional free-text notes for edge cases.
    """

    code: str
    name: str
    native_name: str
    detection_tier: DetectionTier
    script: str
    marker_words: FrozenSet[str]
    langdetect_code: Optional[str]
    min_chars_reliable: int
    notes: Optional[str] = None


class LanguageRegistryProtocol(Protocol):
    """Protocol for the language registry.

    All LCP components depend on this protocol, never on the
    concrete implementation. This allows testing with stub registries.
    """

    def get(self, code: str) -> LanguageRegistryEntry:
        """Get entry by ISO-639-1 code. Raises KeyError if unknown.

        Lookup is case-insensitive: get("DE") and get("de") are equivalent.
        """
        ...

    def get_or_none(self, code: str) -> LanguageRegistryEntry | None:
        """Get entry by ISO-639-1 code. Returns None if unknown.

        Lookup is case-insensitive.
        """
        ...

    def is_supported(self, code: str) -> bool:
        """Check if a language code is supported.

        Lookup is case-insensitive.
        """
        ...

    def list_codes(self) -> list[str]:
        """List all supported ISO-639-1 codes, sorted alphabetically."""
        ...

    def list_by_tier(self, tier: DetectionTier) -> list[LanguageRegistryEntry]:
        """List all entries with the given detection tier."""
        ...

    def list_by_script(self, script: str) -> list[LanguageRegistryEntry]:
        """List all entries with the given script family."""
        ...

    def resolve_backend_code(self, backend_code: str) -> str:
        """Map a backend-specific code to canonical AXOLENT code.

        Known mappings:
        - "no" -> "nb" (langdetect returns "no" for Norwegian Bokmal)
        - "zh-cn" -> "zh" (unified Chinese)
        - "zh-tw" -> "zh" (unified Chinese)

        Returns input unchanged if no mapping exists.
        """
        ...

    @property
    def supported_count(self) -> int:
        """Number of supported languages."""
        ...


# ---------------------------------------------------------------------------
# Backend-code normalization table
# ---------------------------------------------------------------------------
# This table centralizes all known divergences between backend output codes
# and canonical AXOLENT ISO-639-1 codes. After Phase 2 migration (Step 4),
# LangdetectBackend._normalize() will delegate to this via the registry.
_BACKEND_CODE_MAP: dict[str, str] = {
    "no": "nb",  # langdetect: Norwegian -> Norwegian Bokmal
    "zh-cn": "zh",  # langdetect: simplified Chinese -> unified Chinese
    "zh-tw": "zh",  # langdetect: traditional Chinese -> unified Chinese
}


class InMemoryLanguageRegistry:
    """Concrete in-memory implementation of the language registry.

    All 23 supported languages are registered as hardcoded entries.
    No filesystem IO or database lookup at runtime.

    Usage::

        registry = InMemoryLanguageRegistry()
        entry = registry.get("de")
        print(entry.name)  # "German"
    """

    def __init__(
        self,
        entries: list[LanguageRegistryEntry] | None = None,
    ) -> None:
        """Initialize with default entries or a custom list.

        Args:
            entries: Custom entry list (for testing). If None, the full
                set of 23 supported languages is loaded.
        """
        if entries is None:
            entries = _build_default_entries()

        self._entries: dict[str, LanguageRegistryEntry] = {e.code: e for e in entries}

        # Build reverse lookup for backend code normalization:
        # langdetect_code -> canonical code for entries that diverge.
        self._backend_map: dict[str, str] = dict(_BACKEND_CODE_MAP)
        for entry in entries:
            if entry.langdetect_code is not None:
                self._backend_map[entry.langdetect_code] = entry.code

        log.debug(
            "LanguageRegistry initialized with %d languages",
            len(self._entries),
        )

    # -- Protocol methods ---------------------------------------------------

    def get(self, code: str) -> LanguageRegistryEntry:
        """Get entry by ISO-639-1 code. Raises KeyError if unknown.

        Lookup is case-insensitive: get("DE") and get("de") are equivalent.
        """
        normalized = code.lower()
        try:
            return self._entries[normalized]
        except KeyError:
            log.warning("Unknown language code requested: %r", code)
            raise KeyError(
                f"Language code {code!r} is not registered. "
                f"Supported: {', '.join(sorted(self._entries))}"
            ) from None

    def get_or_none(self, code: str) -> LanguageRegistryEntry | None:
        """Get entry by ISO-639-1 code. Returns None if unknown.

        Lookup is case-insensitive.
        """
        return self._entries.get(code.lower())

    def is_supported(self, code: str) -> bool:
        """Check if a language code is supported.

        Lookup is case-insensitive.
        """
        return code.lower() in self._entries

    def list_codes(self) -> list[str]:
        """List all supported ISO-639-1 codes, sorted alphabetically."""
        return sorted(self._entries)

    def list_by_tier(self, tier: DetectionTier) -> list[LanguageRegistryEntry]:
        """List all entries with the given detection tier."""
        return [e for e in self._entries.values() if e.detection_tier == tier]

    def list_by_script(self, script: str) -> list[LanguageRegistryEntry]:
        """List all entries with the given script family."""
        normalized = script.lower()
        return [e for e in self._entries.values() if e.script == normalized]

    def resolve_backend_code(self, backend_code: str) -> str:
        """Map a backend-specific code to canonical AXOLENT code.

        Known mappings:
        - "no" -> "nb" (langdetect returns "no" for Norwegian Bokmal)
        - "zh-cn" -> "zh" (unified Chinese)
        - "zh-tw" -> "zh" (unified Chinese)

        Returns input unchanged if no mapping exists.
        """
        return self._backend_map.get(backend_code, backend_code)

    @property
    def supported_count(self) -> int:
        """Number of supported languages."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Default entries: all 23 supported languages
# ---------------------------------------------------------------------------


def _build_default_entries() -> list[LanguageRegistryEntry]:  # noqa: C901
    """Build the full set of 23 language entries.

    Marker words are curated subsets (10-20 high-signal words per language)
    chosen for short-text disambiguation. Script-detected languages (HIGH
    tier, non-Latin) have empty marker sets because Unicode ranges are
    sufficient.

    Returns:
        List of LanguageRegistryEntry for all supported languages.
    """
    return [
        # -- HIGH tier: script-detected + German (exclusive marker) ------
        LanguageRegistryEntry(
            code="de",
            name="German",
            native_name="Deutsch",
            detection_tier=DetectionTier.HIGH,
            script="latin",
            marker_words=frozenset(
                {
                    "der",
                    "die",
                    "das",
                    "ein",
                    "eine",
                    "ist",
                    "sind",
                    "hat",
                    "haben",
                    "wird",
                    "werden",
                    "nicht",
                    "aber",
                    "und",
                    "oder",
                    "ich",
                    "du",
                    "wir",
                    "dass",
                    "mit",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=10,
            notes="HIGH despite Latin script due to exclusive char U+00DF.",
        ),
        LanguageRegistryEntry(
            code="ru",
            name="Russian",
            native_name="Russkij",
            detection_tier=DetectionTier.HIGH,
            script="cyrillic",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=5,
            notes="Cyrillic script. Disambiguated from Ukrainian via markers.",
        ),
        LanguageRegistryEntry(
            code="uk",
            name="Ukrainian",
            native_name="Ukrainska",
            detection_tier=DetectionTier.HIGH,
            script="cyrillic",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=10,
            notes="Cyrillic script. Unique chars: U+0457, U+0456, U+0454, U+0491.",
        ),
        LanguageRegistryEntry(
            code="ar",
            name="Arabic",
            native_name="al-Arabiyya",
            detection_tier=DetectionTier.HIGH,
            script="arabic",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=5,
        ),
        LanguageRegistryEntry(
            code="zh",
            name="Chinese",
            native_name="Zhongwen",
            detection_tier=DetectionTier.HIGH,
            script="cjk",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=3,
            notes="Unified: zh-cn and zh-tw both map to zh.",
        ),
        LanguageRegistryEntry(
            code="ja",
            name="Japanese",
            native_name="Nihongo",
            detection_tier=DetectionTier.HIGH,
            script="kana",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=3,
            notes="Hiragana/Katakana disambiguates from Chinese CJK.",
        ),
        LanguageRegistryEntry(
            code="ko",
            name="Korean",
            native_name="Hangugeo",
            detection_tier=DetectionTier.HIGH,
            script="hangul",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=3,
        ),
        LanguageRegistryEntry(
            code="hi",
            name="Hindi",
            native_name="Hindi",
            detection_tier=DetectionTier.HIGH,
            script="devanagari",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=5,
        ),
        LanguageRegistryEntry(
            code="th",
            name="Thai",
            native_name="Phasa Thai",
            detection_tier=DetectionTier.HIGH,
            script="thai",
            marker_words=frozenset(),
            langdetect_code=None,
            min_chars_reliable=5,
        ),
        # -- MEDIUM tier: Latin-script languages -------------------------
        LanguageRegistryEntry(
            code="en",
            name="English",
            native_name="English",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "the",
                    "is",
                    "are",
                    "was",
                    "have",
                    "has",
                    "will",
                    "would",
                    "could",
                    "should",
                    "been",
                    "this",
                    "that",
                    "what",
                    "which",
                    "who",
                    "how",
                    "not",
                    "but",
                    "for",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=15,
            notes="MEDIUM: high marker overlap with Germanic languages on short text.",
        ),
        LanguageRegistryEntry(
            code="nl",
            name="Dutch",
            native_name="Nederlands",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "het",
                    "een",
                    "van",
                    "voor",
                    "niet",
                    "dat",
                    "zijn",
                    "naar",
                    "ook",
                    "aan",
                    "maar",
                    "nog",
                    "wel",
                    "deze",
                    "werd",
                    "wordt",
                    "hebben",
                    "heeft",
                    "kunnen",
                    "moeten",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=20,
            notes="Often confused with German/English on short inputs.",
        ),
        LanguageRegistryEntry(
            code="fr",
            name="French",
            native_name="Francais",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "le",
                    "la",
                    "les",
                    "un",
                    "une",
                    "des",
                    "est",
                    "sont",
                    "je",
                    "tu",
                    "il",
                    "nous",
                    "vous",
                    "ne",
                    "pas",
                    "mais",
                    "avec",
                    "pour",
                    "dans",
                    "que",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=15,
        ),
        LanguageRegistryEntry(
            code="es",
            name="Spanish",
            native_name="Espanol",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "el",
                    "la",
                    "los",
                    "las",
                    "un",
                    "una",
                    "es",
                    "son",
                    "fue",
                    "tiene",
                    "hay",
                    "yo",
                    "nosotros",
                    "no",
                    "pero",
                    "como",
                    "por",
                    "para",
                    "con",
                    "que",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=15,
        ),
        LanguageRegistryEntry(
            code="it",
            name="Italian",
            native_name="Italiano",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "il",
                    "lo",
                    "la",
                    "gli",
                    "le",
                    "un",
                    "una",
                    "di",
                    "che",
                    "non",
                    "sono",
                    "per",
                    "questo",
                    "come",
                    "anche",
                    "con",
                    "ma",
                    "molto",
                    "bene",
                    "cosa",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=15,
        ),
        LanguageRegistryEntry(
            code="pt",
            name="Portuguese",
            native_name="Portugues",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "o",
                    "a",
                    "os",
                    "as",
                    "um",
                    "uma",
                    "de",
                    "que",
                    "do",
                    "da",
                    "em",
                    "para",
                    "com",
                    "nao",
                    "por",
                    "mais",
                    "como",
                    "mas",
                    "foi",
                    "ao",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=15,
        ),
        LanguageRegistryEntry(
            code="pl",
            name="Polish",
            native_name="Polski",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "jest",
                    "nie",
                    "to",
                    "na",
                    "tak",
                    "ale",
                    "jak",
                    "bardzo",
                    "dobrze",
                    "gdzie",
                    "kiedy",
                    "dlaczego",
                    "jestem",
                    "moze",
                    "tylko",
                    "tego",
                    "jego",
                    "jej",
                    "wszystko",
                    "kazdy",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=10,
        ),
        LanguageRegistryEntry(
            code="sv",
            name="Swedish",
            native_name="Svenska",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "och",
                    "att",
                    "det",
                    "som",
                    "med",
                    "av",
                    "den",
                    "har",
                    "inte",
                    "till",
                    "var",
                    "jag",
                    "kan",
                    "ska",
                    "alla",
                    "mycket",
                    "denna",
                    "efter",
                    "bara",
                    "hur",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=20,
            notes="Shared diacritics with DE/FI; relies on marker words.",
        ),
        LanguageRegistryEntry(
            code="da",
            name="Danish",
            native_name="Dansk",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "og",
                    "er",
                    "det",
                    "en",
                    "af",
                    "til",
                    "den",
                    "som",
                    "med",
                    "har",
                    "ikke",
                    "kan",
                    "skal",
                    "vil",
                    "jeg",
                    "hvad",
                    "hvor",
                    "hvorfor",
                    "fordi",
                    "meget",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=25,
            notes="High overlap with Norwegian Bokmal (nb).",
        ),
        LanguageRegistryEntry(
            code="nb",
            name="Norwegian",
            native_name="Norsk",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "og",
                    "er",
                    "det",
                    "en",
                    "av",
                    "til",
                    "den",
                    "som",
                    "med",
                    "har",
                    "ikke",
                    "kan",
                    "skal",
                    "vil",
                    "jeg",
                    "hva",
                    "hvor",
                    "hvorfor",
                    "fordi",
                    "veldig",
                }
            ),
            langdetect_code="no",
            min_chars_reliable=25,
            notes="langdetect returns 'no'; mapped to 'nb'. High DA overlap.",
        ),
        LanguageRegistryEntry(
            code="fi",
            name="Finnish",
            native_name="Suomi",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "ja",
                    "on",
                    "ei",
                    "se",
                    "kun",
                    "mutta",
                    "tai",
                    "niin",
                    "kuin",
                    "ovat",
                    "olla",
                    "mitä",
                    "miksi",
                    "myös",
                    "vain",
                    "hyvin",
                    "paljon",
                    "kanssa",
                    "sitten",
                    "kaikki",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=15,
        ),
        LanguageRegistryEntry(
            code="tr",
            name="Turkish",
            native_name="Turkce",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "bir",
                    "bu",
                    "olan",
                    "gibi",
                    "daha",
                    "ancak",
                    "sonra",
                    "bunu",
                    "onun",
                    "kadar",
                    "olarak",
                    "evet",
                    "hayir",
                    "burada",
                    "orada",
                    "sadece",
                    "hepsi",
                    "cok",
                    "iyi",
                    "tesekkur",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=10,
        ),
        LanguageRegistryEntry(
            code="id",
            name="Indonesian",
            native_name="Bahasa Indonesia",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "yang",
                    "dan",
                    "untuk",
                    "dengan",
                    "ini",
                    "itu",
                    "adalah",
                    "dari",
                    "pada",
                    "akan",
                    "sudah",
                    "bisa",
                    "ada",
                    "tidak",
                    "juga",
                    "seperti",
                    "mereka",
                    "saya",
                    "sangat",
                    "baik",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=20,
        ),
        LanguageRegistryEntry(
            code="vi",
            name="Vietnamese",
            native_name="Tieng Viet",
            detection_tier=DetectionTier.MEDIUM,
            script="latin",
            marker_words=frozenset(
                {
                    "cua",
                    "khong",
                    "trong",
                    "nhung",
                    "nhu",
                    "cac",
                    "mot",
                    "nay",
                    "duoc",
                    "khi",
                    "cung",
                    "rat",
                    "tot",
                    "bao",
                    "tai",
                    "sao",
                    "noi",
                    "day",
                    "hom",
                    "nay",
                }
            ),
            langdetect_code=None,
            min_chars_reliable=10,
            notes="Heavy use of diacritics; ASCII fallbacks included.",
        ),
    ]
