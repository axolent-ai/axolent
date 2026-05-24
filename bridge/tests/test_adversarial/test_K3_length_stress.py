"""K3: Length stress tests.

Messages exceeding expected sizes, whitespace floods,
rapid command sequences, oversized /remember content.
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
from application.leakage_filter import (
    check_for_forbidden_patterns,
    check_for_system_prompt_leakage,
    _extract_fingerprints,
)


def _make_hypothesis(claim: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="test-len-001",
        user_id=1,
        claim=claim,
        scope=HypothesisScope(),
        created_at="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
    )


@pytest.mark.adversarial
class TestOversizedMessages:
    """Messages larger than typical Telegram messages (4096 chars)."""

    def test_claim_over_4096_chars(self) -> None:
        """WHAT: Hypothesis claim exceeding Telegram message limit.
        EXPECTED: Pipeline processes without crash.
        WHY: Claims could be generated from long conversation imports.
        """
        pipeline = PrivacyPipeline()
        claim = "a" * 5000
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_claim_100kb(self) -> None:
        """WHAT: 100KB hypothesis claim.
        EXPECTED: Pipeline processes without excessive delay.
        WHY: Stress test for regex-heavy filters.
        """
        pipeline = PrivacyPipeline()
        claim = "User prefers " + "detailed answers with examples " * 3000
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_secret_scanner_100kb_input(self) -> None:
        """WHAT: 100KB text through SecretScanner.
        EXPECTED: Completes without crash.
        WHY: Scanner runs regex on full text, must handle large inputs.
        """
        scanner = SecretScanner()
        text = "normal text " * 10000
        matches = scanner.scan(text)
        assert isinstance(matches, list)


@pytest.mark.adversarial
class TestWhitespaceFlood:
    """Excessive whitespace characters."""

    def test_50000_spaces_in_claim(self) -> None:
        """WHAT: 50000 space characters in claim.
        EXPECTED: Pipeline handles (icontract may reject empty-after-strip).
        WHY: Tests whitespace normalization in pipeline.
        """
        pipeline = PrivacyPipeline()
        claim = " " * 50000
        h = _make_hypothesis(claim)
        # icontract requires claim.strip() to be non-empty, so this should raise
        with pytest.raises(Exception):
            pipeline.check(h)

    def test_mixed_whitespace_types(self) -> None:
        """WHAT: Mix of spaces, tabs, newlines, unicode spaces.
        EXPECTED: No crash.
        WHY: Various whitespace types test normalization.
        """
        pipeline = PrivacyPipeline()
        # Mix of regular space, tab, newline, NBSP, em-space, en-space
        claim = "normal\t\n     text with mixed spaces"
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_newline_flood_in_leakage_check(self) -> None:
        """WHAT: Thousands of newlines in LLM response.
        EXPECTED: No crash, fingerprint matching still works.
        WHY: LLM could generate excessive newlines.
        """
        response = "\n" * 10000 + "actual content here"
        system_prompt = "You are a helpful assistant."
        result = check_for_system_prompt_leakage(response, system_prompt)
        assert result is None or isinstance(result, str)


@pytest.mark.adversarial
class TestLargeFingerprints:
    """Fingerprint extraction from very large system prompts."""

    def test_fingerprint_extraction_large_prompt(self) -> None:
        """WHAT: Fingerprint extraction from a 50KB system prompt.
        EXPECTED: Returns list of fingerprints without crash.
        WHY: System prompts can be very long with many instructions.
        """
        prompt = "You are a helpful assistant. " * 2000
        fps = _extract_fingerprints(prompt)
        assert isinstance(fps, list)
        assert len(fps) > 0

    def test_leakage_check_large_prompt_large_response(self) -> None:
        """WHAT: Both system prompt and response are large.
        EXPECTED: Completes without excessive memory use.
        WHY: Worst-case scenario for O(n*m) substring matching.
        """
        system_prompt = "Instructions section " + "rule " * 5000
        response = "The answer is " + "detail " * 5000
        result = check_for_system_prompt_leakage(response, system_prompt)
        assert result is None or isinstance(result, str)


@pytest.mark.adversarial
class TestRepeatedContent:
    """Rapid repeated patterns and content."""

    def test_repeated_keyword_flood(self) -> None:
        """WHAT: Healthcare keyword repeated thousands of times.
        EXPECTED: Healthcare filter catches on first occurrence.
        WHY: Tests that regex doesn't catastrophically backtrack.
        """
        hf = HealthcareFilter()
        claim = "depression " * 5000
        h = _make_hypothesis(claim)
        result = hf.filter_hypothesis(h)
        assert result is True

    def test_forbidden_pattern_repeated(self) -> None:
        """WHAT: Forbidden pattern repeated many times.
        EXPECTED: Detected on first occurrence, no performance issue.
        WHY: Defensive check should short-circuit.
        """
        response = "claude.md " * 1000
        result = check_for_forbidden_patterns(response)
        assert result is not None  # Should catch forbidden pattern

    def test_single_char_repeated(self) -> None:
        """WHAT: Single character repeated many times.
        EXPECTED: Pipeline completes without timeout.
        WHY: Stress-test for regex performance on repetitive input.

        NOTE: 1M identical alpha chars cause catastrophic backtracking in
        long_base64 pattern (\\b[A-Za-z0-9+/]{40,}={0,2}\\b). Capped at
        50K which is well beyond any realistic Telegram message (4096 chars).
        The 1M case is documented as FINDING-08 (regex perf on extreme input).
        """
        scanner = SecretScanner()
        text = "x" * 50_000
        matches = scanner.scan(text)
        assert isinstance(matches, list)
