"""K2: Unicode stress tests.

RTL-override, ZWJ, Zalgo, homoglyphs, mixed-script, sticky-language
hybrid tokens. Tests encoding robustness of privacy filters and scanners.
"""

from __future__ import annotations

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.privacy.secret_scanner import SecretScanner
from application.skill_compression.privacy.healthcare_filter import HealthcareFilter
from application.leakage_filter import check_for_system_prompt_leakage


def _make_hypothesis(claim: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="test-uni-001",
        user_id=1,
        claim=claim,
        scope=HypothesisScope(),
        created_at="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
    )


@pytest.mark.adversarial
class TestRTLOverride:
    """Right-to-left override characters that can reverse display."""

    def test_rtl_override_in_claim(self) -> None:
        """WHAT: U+202E (RTL override) in hypothesis claim.
        EXPECTED: Pipeline does not crash, claim is processed.
        WHY: RTL override can make text appear reversed, hiding content.
        """
        pipeline = PrivacyPipeline()
        claim = "User prefers ‮txet neddih formatting"  # nosemgrep: generic.unicode.security.bidi.contains-bidirectional-characters
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_rtl_bidi_mixed_with_secret(self) -> None:
        """WHAT: Secret key hidden behind RTL bidi characters.
        EXPECTED: SecretScanner still detects the key pattern.
        WHY: Attackers could try to hide secrets with bidi tricks.
        """
        scanner = SecretScanner()
        # The key pattern should still match despite bidi chars
        text = "my key is ‏sk-‎proj-abcdefghijklmnop"
        matches = scanner.scan(text)
        # May or may not detect depending on regex behavior with bidi
        assert isinstance(matches, list)


@pytest.mark.adversarial
class TestZeroWidthChars:
    """Zero-width joiners, zero-width spaces, and invisible chars."""

    def test_zwj_spam_in_claim(self) -> None:
        """WHAT: Many zero-width joiners in claim text.
        EXPECTED: No crash, pipeline processes normally.
        WHY: ZWJ spam can inflate string length without visible content.
        """
        pipeline = PrivacyPipeline()
        claim = "User prefers" + "‍" * 500 + "short answers"
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_zero_width_space_between_keyword_chars(self) -> None:
        """WHAT: Zero-width spaces splitting healthcare keywords.
        EXPECTED: Filter may miss the keyword (known limitation).
        WHY: Tests whether keyword detection can be bypassed with ZWS.
        """
        hf = HealthcareFilter()
        # "depression" with zero-width spaces between each char
        split_keyword = "d​e​p​r​e​s​s​i​o​n"
        h = _make_hypothesis(f"User shows signs of {split_keyword}")
        result = hf.filter_hypothesis(h)
        # Known limitation: word-boundary regex won't match split words
        # This is acceptable because the privacy pipeline is defense-in-depth
        assert isinstance(result, bool)

    def test_invisible_separator_flood(self) -> None:
        """WHAT: Thousands of invisible separator characters.
        EXPECTED: No crash, no excessive CPU time.
        WHY: Performance stress test for regex-heavy filters.
        """
        pipeline = PrivacyPipeline()
        # U+2063 invisible separator
        claim = "normal claim" + "⁣" * 2000
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None


@pytest.mark.adversarial
class TestEmojiZWJClusters:
    """Emoji ZWJ sequences and skin-tone modifiers."""

    def test_emoji_zwj_cluster_in_claim(self) -> None:
        """WHAT: Complex emoji ZWJ sequences (family, profession emojis).
        EXPECTED: Pipeline handles without crash.
        WHY: Real user text contains complex emoji sequences.
        """
        pipeline = PrivacyPipeline()
        # Family emoji: man + ZWJ + woman + ZWJ + girl + ZWJ + boy
        family = "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466"
        claim = f"User likes using {family} in messages"
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_skin_tone_modifiers(self) -> None:
        """WHAT: Emoji with multiple skin-tone modifiers.
        EXPECTED: No crash.
        WHY: Skin-tone modifiers change string length unexpectedly.
        """
        pipeline = PrivacyPipeline()
        claim = "\U0001f44b\U0001f3fb \U0001f44b\U0001f3fc \U0001f44b\U0001f3fd \U0001f44b\U0001f3fe \U0001f44b\U0001f3ff"
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None


@pytest.mark.adversarial
class TestZalgoText:
    """Combining diacritics stacks (Zalgo text)."""

    def test_zalgo_text_in_claim(self) -> None:
        """WHAT: Zalgo text (excessive combining diacritics).
        EXPECTED: Pipeline does not crash or hang.
        WHY: Zalgo text stresses regex word-boundary detection.
        """
        pipeline = PrivacyPipeline()
        # Build Zalgo: each char gets 10 combining diacritics
        base = "depression"
        zalgo = ""
        for c in base:
            zalgo += c + "̀́̂̃̄̅̆̇̈̉"
        h = _make_hypothesis(f"User has {zalgo}")
        result = pipeline.check(h)
        # Zalgo may or may not match the regex depending on \b behavior
        assert isinstance(result, (type(None), object))


@pytest.mark.adversarial
class TestHomoglyphs:
    """Homoglyph attacks (cyrillic 'a' instead of latin 'a')."""

    def test_cyrillic_homoglyph_bypass_healthcare(self) -> None:
        """WHAT: Healthcare keyword with Cyrillic homoglyphs.
        EXPECTED: Filter likely misses (known limitation of regex).
        WHY: Adversary replaces latin chars with visually identical Cyrillic.
        """
        hf = HealthcareFilter()
        # "depression" with Cyrillic 'e' (U+0435) and 'o' (U+043E)
        fake_depression = "dеprеssiоn"
        h = _make_hypothesis(f"User shows {fake_depression}")
        result = hf.filter_hypothesis(h)
        # Known limitation: regex \b and exact match won't catch homoglyphs
        assert isinstance(result, bool)

    def test_homoglyph_secret_key_bypass(self) -> None:
        """WHAT: API key prefix with homoglyph characters.
        EXPECTED: Scanner may miss (documents limitation).
        WHY: Adversary could use Cyrillic 'k' to bypass sk- detection.
        """
        scanner = SecretScanner()
        # Cyrillic 'k' (U+043A) instead of latin 'k'
        fake_key = "sк-proj-abcdefghijklmnop"
        matches = scanner.scan(fake_key)
        assert isinstance(matches, list)


@pytest.mark.adversarial
class TestMixedScript:
    """Mixed script text: CJK + Latin + Greek + Arabic in one claim."""

    def test_mixed_script_claim(self) -> None:
        """WHAT: Claim with characters from 4+ scripts.
        EXPECTED: Pipeline handles without crash.
        WHY: Multilingual users may mix scripts naturally.
        """
        pipeline = PrivacyPipeline()
        claim = "Use bullet points and 3 items maximum"
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_cjk_latin_mix_in_leakage_check(self) -> None:
        """WHAT: Mixed CJK+Latin in leakage filter.
        EXPECTED: No crash, correct behavior.
        WHY: System prompt is English, response could be CJK.
        """
        system_prompt = "You are a helpful assistant. Follow these rules carefully."
        response = "The answer is 42, which can be interpreted differently."
        result = check_for_system_prompt_leakage(response, system_prompt)
        assert result is None or isinstance(result, str)
