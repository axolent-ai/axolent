"""Tests for DE + Obfuscation bypass (Codex Polish-Polish 2026-05-30).

Codex empirically demonstrated that German healthcare terms combined with
Unicode obfuscation techniques bypass the two-pass filter architecture when:
  - Pass 1 uses raw claim (ZWSP/combining chars prevent pattern match)
  - Pass 2 uses aggressive normalization (strips umlauts, pattern expects them)

These tests document the 5 mandatory cases and MUST be green after the fix:
  1. Zero-Width Space before umlaut in German term
  2. Zero-Width Space in another German term
  3. Combining diaeresis (U+0308) instead of pre-composed umlaut
  4. Cyrillic letter with combining diaeresis (U+04E7)
  5. Baseline (pre-composed umlaut, must always work)

Plus NudgeFilter analogues with Opus-verified phrases.
"""

from __future__ import annotations

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.healthcare_filter import HealthcareFilter
from application.skill_compression.privacy.nudge_filter import NudgeFilter


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _hyp(claim: str) -> Hypothesis:
    """Create a test hypothesis with given claim."""
    return Hypothesis(
        hypothesis_id="hyp-de-obfusc-test",
        user_id=42,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status="candidate",
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
    )


# ---------------------------------------------------------------
# Item 3: HealthcareFilter DE + Obfuscation (5 mandatory cases)
# ---------------------------------------------------------------


class TestHealthcareFilterDEObfuscation:
    """Codex Polish-Polish Findings: DE + Obfuscation must be blocked.

    These 5 cases were empirically proven to bypass the raw+aggressive
    two-pass architecture. The fix: basic+aggressive with aggressive
    pattern variants.
    """

    @pytest.fixture
    def hf(self) -> HealthcareFilter:
        return HealthcareFilter()

    @pytest.mark.parametrize(
        "payload,name",
        [
            # Case 1: Zero-Width Space (U+200B) between 'st' and umlaut
            (
                "Ich habe eine Angstst​örung",
                "zero_width_before_umlaut",
            ),
            # Case 2: Zero-Width Space in Schlafst+oerung
            (
                "Ich habe Schlafst​örung",
                "zero_width_in_de_term",
            ),
            # Case 3: 'o' + combining diaeresis (U+0308) instead of pre-composed
            (
                "Ich habe Schlafstörung",
                "combining_diaeresis",
            ),
            # Case 4: Cyrillic oe with diaeresis (U+04E7) in German word
            (
                "Ich habe eine Angststӧrung",
                "cyrillic_o_plus_combining",
            ),
            # Case 5: Baseline (pre-composed umlaut, must always work)
            (
                "Ich habe eine Angststörung",
                "baseline_german_umlaut",
            ),
        ],
    )
    def test_healthcare_filter_blocks_de_obfuscation(
        self, hf: HealthcareFilter, payload: str, name: str
    ) -> None:
        """Each DE+Obfuscation variant MUST be blocked by HealthcareFilter."""
        hyp = _hyp(payload)
        result = hf.filter_hypothesis(hyp)
        assert result is True, (
            f"{name}: payload {payload!r} should be blocked but was not. "
            "DE+Obfuscation bypass: both passes failed to match. "
            "Fix: Pass 1 must use normalize_for_security_check (basic), "
            "Pass 2 must use aggressive patterns."
        )


# ---------------------------------------------------------------
# Item 3 + 5: NudgeFilter Obfuscation (Opus-verified phrases)
# ---------------------------------------------------------------


class TestNudgeFilterObfuscation:
    """NudgeFilter must block dark patterns even with homoglyph obfuscation.

    Opus 4.8 empirically verified these phrases trigger the DARK_PATTERNS
    category pattern: 'hide.*opt.?out.*cancel' (or similar).
    """

    @pytest.fixture
    def nf(self) -> NudgeFilter:
        return NudgeFilter()

    @pytest.mark.parametrize(
        "payload,name",
        [
            # English baseline (Opus verified: triggers dark pattern regex)
            (
                "hide the opt-out button to make it hard to cancel",
                "english_baseline",
            ),
            # Cyrillic 'i' (U+0456) in 'hide' (Opus verified: Pass 2 catches)
            (
                "hіde the opt-out button to make it hard to cancel",
                "cyrillic_i_in_hide",
            ),
            # German dark pattern with Zero-Width in 'Kuendigung'
            (
                "K​ündigung verschleiern um Cancel zu verstecken",
                "de_zero_width_plus_term",
            ),
        ],
    )
    def test_nudge_filter_blocks_obfuscation(
        self, nf: NudgeFilter, payload: str, name: str
    ) -> None:
        """NudgeFilter MUST block dark patterns with obfuscation."""
        hyp = _hyp(payload)
        result = nf.violates_nudge_policy(hyp)
        assert result is True, (
            f"{name}: payload {payload!r} should be blocked but was not. "
            "Obfuscation bypass in NudgeFilter."
        )
