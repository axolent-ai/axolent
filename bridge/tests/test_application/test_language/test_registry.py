"""Tests for LanguageRegistry (Phase 2 Core, Step 1/4).

Covers:
- Unit tests: all 23 languages present, get/get_or_none, KeyError,
  resolve_backend_code, list_by_tier, list_codes
- Property-based tests: ISO-639-1 invariants, tier/script consistency
- Edge-case tests: empty registry, case-insensitive lookup, all backend
  code mappings

Test naming convention (IC-R5): test_<method>_<scenario>_<expected>.
"""

from __future__ import annotations

import pytest

from application.language.registry import (
    DetectionTier,
    InMemoryLanguageRegistry,
)

# ── Fixtures ──────────────────────────────────────────────────────────

EXPECTED_CODES = sorted(
    [
        "ar",
        "da",
        "de",
        "en",
        "es",
        "fi",
        "fr",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "nb",
        "nl",
        "pl",
        "pt",
        "ru",
        "sv",
        "th",
        "tr",
        "uk",
        "vi",
        "zh",
    ]
)

HIGH_TIER_CODES = sorted(
    [
        "ar",
        "de",
        "hi",
        "ja",
        "ko",
        "ru",
        "th",
        "uk",
        "zh",
    ]
)

MEDIUM_TIER_CODES = sorted(
    [
        "da",
        "en",
        "es",
        "fi",
        "fr",
        "id",
        "it",
        "nb",
        "nl",
        "pl",
        "pt",
        "sv",
        "tr",
        "vi",
    ]
)


@pytest.fixture()
def registry() -> InMemoryLanguageRegistry:
    """Default registry with all 23 languages."""
    return InMemoryLanguageRegistry()


@pytest.fixture()
def empty_registry() -> InMemoryLanguageRegistry:
    """Empty registry for edge-case tests."""
    return InMemoryLanguageRegistry(entries=[])


# ── Unit Tests: Language Presence ─────────────────────────────────────


class TestRegistryPresence:
    """Every expected language must have an entry."""

    def test_all_23_languages_present(self, registry: InMemoryLanguageRegistry) -> None:
        """HC-R1: Registry contains exactly the 23 specified languages."""
        assert registry.list_codes() == EXPECTED_CODES

    def test_supported_count_is_23(self, registry: InMemoryLanguageRegistry) -> None:
        """supported_count matches the number of registered languages."""
        assert registry.supported_count == 23

    @pytest.mark.parametrize("code", EXPECTED_CODES)
    def test_each_language_retrievable(
        self, registry: InMemoryLanguageRegistry, code: str
    ) -> None:
        """Every code in the expected set is retrievable via get()."""
        entry = registry.get(code)
        assert entry.code == code


# ── Unit Tests: get() and get_or_none() ──────────────────────────────


class TestRegistryGet:
    """Tests for get() and get_or_none() methods."""

    def test_get_returns_correct_entry(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """get() returns the correct entry for a valid code."""
        entry = registry.get("de")
        assert entry.name == "German"
        assert entry.native_name == "Deutsch"
        assert entry.detection_tier == DetectionTier.HIGH

    def test_get_raises_keyerror_for_unknown(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """get() raises KeyError for unregistered codes."""
        with pytest.raises(KeyError, match="xx"):
            registry.get("xx")

    def test_get_or_none_returns_entry(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """get_or_none() returns the entry for a valid code."""
        entry = registry.get_or_none("en")
        assert entry is not None
        assert entry.name == "English"

    def test_get_or_none_returns_none_for_unknown(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """get_or_none() returns None for unregistered codes."""
        assert registry.get_or_none("xx") is None


# ── Unit Tests: is_supported() ───────────────────────────────────────


class TestRegistryIsSupported:
    """Tests for is_supported() method."""

    def test_is_supported_for_known_code(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """is_supported() returns True for a registered code."""
        assert registry.is_supported("de") is True

    def test_is_supported_for_unknown_code(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """is_supported() returns False for an unregistered code."""
        assert registry.is_supported("xx") is False

    def test_is_supported_case_insensitive(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """is_supported() is case-insensitive."""
        assert registry.is_supported("DE") is True
        assert registry.is_supported("De") is True


# ── Unit Tests: resolve_backend_code() ───────────────────────────────


class TestResolveBackendCode:
    """Tests for resolve_backend_code() (HC-R4)."""

    def test_no_maps_to_nb(self, registry: InMemoryLanguageRegistry) -> None:
        """langdetect's 'no' maps to canonical 'nb'."""
        assert registry.resolve_backend_code("no") == "nb"

    def test_zh_cn_maps_to_zh(self, registry: InMemoryLanguageRegistry) -> None:
        """langdetect's 'zh-cn' maps to canonical 'zh'."""
        assert registry.resolve_backend_code("zh-cn") == "zh"

    def test_zh_tw_maps_to_zh(self, registry: InMemoryLanguageRegistry) -> None:
        """langdetect's 'zh-tw' maps to canonical 'zh'."""
        assert registry.resolve_backend_code("zh-tw") == "zh"

    def test_unknown_code_passes_through(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Unknown backend codes pass through unchanged."""
        assert registry.resolve_backend_code("en") == "en"
        assert registry.resolve_backend_code("xyz") == "xyz"

    def test_canonical_code_passes_through(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Canonical AXOLENT codes are returned unchanged."""
        assert registry.resolve_backend_code("nb") == "nb"
        assert registry.resolve_backend_code("zh") == "zh"


# ── Unit Tests: list_by_tier() ───────────────────────────────────────


class TestListByTier:
    """Tests for list_by_tier() (HC-R5)."""

    def test_high_tier_contains_script_detected_and_german(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """HIGH tier: all script-detected languages + German."""
        high_entries = registry.list_by_tier(DetectionTier.HIGH)
        high_codes = sorted(e.code for e in high_entries)
        assert high_codes == HIGH_TIER_CODES

    def test_medium_tier_contains_latin_languages(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """MEDIUM tier: all other Latin-script languages."""
        medium_entries = registry.list_by_tier(DetectionTier.MEDIUM)
        medium_codes = sorted(e.code for e in medium_entries)
        assert medium_codes == MEDIUM_TIER_CODES

    def test_low_tier_is_empty(self, registry: InMemoryLanguageRegistry) -> None:
        """LOW tier: no languages in the current set (reserve)."""
        low_entries = registry.list_by_tier(DetectionTier.LOW)
        assert low_entries == []

    def test_tier_counts_add_up(self, registry: InMemoryLanguageRegistry) -> None:
        """Sum of all tier counts equals supported_count."""
        total = sum(len(registry.list_by_tier(tier)) for tier in DetectionTier)
        assert total == registry.supported_count


# ── Unit Tests: list_codes() ─────────────────────────────────────────


class TestListCodes:
    """Tests for list_codes() ordering and uniqueness."""

    def test_list_codes_is_sorted(self, registry: InMemoryLanguageRegistry) -> None:
        """list_codes() returns codes in sorted order."""
        codes = registry.list_codes()
        assert codes == sorted(codes)

    def test_list_codes_no_duplicates(self, registry: InMemoryLanguageRegistry) -> None:
        """list_codes() contains no duplicate entries."""
        codes = registry.list_codes()
        assert len(codes) == len(set(codes))


# ── Unit Tests: list_by_script() ─────────────────────────────────────


class TestListByScript:
    """Tests for list_by_script()."""

    def test_cyrillic_contains_ru_uk(self, registry: InMemoryLanguageRegistry) -> None:
        """Cyrillic script returns Russian and Ukrainian."""
        entries = registry.list_by_script("cyrillic")
        codes = sorted(e.code for e in entries)
        assert codes == ["ru", "uk"]

    def test_latin_contains_14_languages(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Latin script returns the 14 Latin-script languages (incl. DE)."""
        entries = registry.list_by_script("latin")
        # 14 MEDIUM + 1 HIGH (de) = 15 total Latin-script languages
        # Actually: de, en, nl, fr, es, it, pt, pl, sv, da, nb, fi, tr, id, vi
        assert len(entries) == 15

    def test_unknown_script_returns_empty(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Unknown script names return an empty list."""
        assert registry.list_by_script("martian") == []

    def test_script_lookup_case_insensitive(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """list_by_script() normalizes script name to lowercase."""
        entries_lower = registry.list_by_script("cyrillic")
        entries_upper = registry.list_by_script("Cyrillic")
        assert len(entries_lower) == len(entries_upper)


# ── Property-Based Tests ─────────────────────────────────────────────


class TestRegistryProperties:
    """Invariant tests across all entries."""

    def test_all_codes_are_two_chars(self, registry: InMemoryLanguageRegistry) -> None:
        """ISO-639-1 invariant: every code is exactly 2 characters."""
        for code in registry.list_codes():
            entry = registry.get(code)
            assert len(entry.code) == 2, f"{entry.code} is not 2 chars"

    def test_all_codes_are_lowercase(self, registry: InMemoryLanguageRegistry) -> None:
        """Lowercase invariant: every code is lowercase."""
        for code in registry.list_codes():
            entry = registry.get(code)
            assert entry.code == entry.code.lower(), f"{entry.code} is not lowercase"

    def test_high_tier_implies_non_latin_or_german(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """HIGH tier entries are non-Latin script OR German (exclusive char)."""
        for entry in registry.list_by_tier(DetectionTier.HIGH):
            assert entry.script != "latin" or entry.code == "de", (
                f"{entry.code} is HIGH tier + Latin but not German"
            )

    def test_supported_count_equals_list_codes_length(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """supported_count is consistent with list_codes()."""
        assert registry.supported_count == len(registry.list_codes())

    def test_every_entry_has_name(self, registry: InMemoryLanguageRegistry) -> None:
        """Every entry has a non-empty English name."""
        for code in registry.list_codes():
            entry = registry.get(code)
            assert entry.name, f"{code} has empty name"

    def test_every_entry_has_native_name(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Every entry has a non-empty native name."""
        for code in registry.list_codes():
            entry = registry.get(code)
            assert entry.native_name, f"{code} has empty native_name"

    def test_every_entry_has_positive_min_chars(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """min_chars_reliable is always positive."""
        for code in registry.list_codes():
            entry = registry.get(code)
            assert entry.min_chars_reliable > 0, (
                f"{code} has non-positive min_chars_reliable"
            )

    def test_script_detected_languages_have_empty_markers(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Non-Latin HIGH tier languages have empty marker_words."""
        for entry in registry.list_by_tier(DetectionTier.HIGH):
            if entry.script != "latin":
                assert len(entry.marker_words) == 0, (
                    f"{entry.code}: script-detected but has markers"
                )


# ── Edge-Case Tests ──────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: empty registry, case sensitivity, frozen entries."""

    def test_empty_registry_list_codes(
        self, empty_registry: InMemoryLanguageRegistry
    ) -> None:
        """Empty registry returns empty list from list_codes()."""
        assert empty_registry.list_codes() == []

    def test_empty_registry_supported_count(
        self, empty_registry: InMemoryLanguageRegistry
    ) -> None:
        """Empty registry has supported_count of 0."""
        assert empty_registry.supported_count == 0

    def test_empty_registry_get_raises_keyerror(
        self, empty_registry: InMemoryLanguageRegistry
    ) -> None:
        """Empty registry raises KeyError for any code."""
        with pytest.raises(KeyError):
            empty_registry.get("de")

    def test_case_insensitive_get(self, registry: InMemoryLanguageRegistry) -> None:
        """get() works with uppercase input (IC-R2: case-insensitive)."""
        entry_lower = registry.get("de")
        entry_upper = registry.get("DE")
        entry_mixed = registry.get("De")
        assert entry_lower == entry_upper == entry_mixed

    def test_case_insensitive_get_or_none(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """get_or_none() works with uppercase input."""
        entry = registry.get_or_none("FR")
        assert entry is not None
        assert entry.code == "fr"

    def test_entry_is_frozen(self, registry: InMemoryLanguageRegistry) -> None:
        """LanguageRegistryEntry is truly immutable (HC-R3)."""
        entry = registry.get("de")
        with pytest.raises(AttributeError):
            entry.name = "Modified"  # type: ignore[misc]

    def test_entry_has_slots(self, registry: InMemoryLanguageRegistry) -> None:
        """LanguageRegistryEntry uses __slots__ (HC-R3)."""
        entry = registry.get("de")
        assert hasattr(entry, "__slots__") or not hasattr(entry, "__dict__")

    def test_all_backend_code_mappings_covered(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """All known backend code divergences are mapped correctly."""
        mappings = {
            "no": "nb",
            "zh-cn": "zh",
            "zh-tw": "zh",
        }
        for backend_code, expected in mappings.items():
            assert registry.resolve_backend_code(backend_code) == expected

    def test_nb_entry_has_langdetect_code(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Norwegian entry declares its langdetect alias."""
        entry = registry.get("nb")
        assert entry.langdetect_code == "no"


# ── Contract Readiness Tests (Prep for Step 4) ───────────────────────


class TestContractReadiness:
    """Verify registry provides all data contract.py needs for migration."""

    def test_all_contract_language_names_available(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Registry provides English names for all languages in contract.py."""
        # These are the codes currently in contract.py's _LANGUAGE_NAMES
        contract_codes = [
            "de",
            "en",
            "fr",
            "es",
            "it",
            "pt",
            "nl",
            "sv",
            "da",
            "nb",
            "fi",
            "pl",
            "tr",
            "ru",
            "uk",
            "ar",
            "zh",
            "ja",
            "ko",
            "hi",
            "th",
            "id",
            "vi",
        ]
        for code in contract_codes:
            entry = registry.get(code)
            assert entry.name, f"No name for {code}"

    def test_contract_names_match_registry(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Registry names match the current contract.py names (HC-R7 prep)."""
        expected_names = {
            "de": "German",
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "sv": "Swedish",
            "da": "Danish",
            "nb": "Norwegian",
            "fi": "Finnish",
            "pl": "Polish",
            "tr": "Turkish",
            "ru": "Russian",
            "uk": "Ukrainian",
            "ar": "Arabic",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "hi": "Hindi",
            "th": "Thai",
            "id": "Indonesian",
            "vi": "Vietnamese",
        }
        for code, expected_name in expected_names.items():
            entry = registry.get(code)
            assert entry.name == expected_name, (
                f"{code}: expected {expected_name!r}, got {entry.name!r}"
            )
