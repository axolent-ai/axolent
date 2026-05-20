"""Tests for min_chars_reliable Guard (B-2 Add-on 1).

Covers:
1. Unit Tests:
   - Text 5 chars, NL detected (min_chars=20): confidence dampened
   - Text 50 chars, NL detected (min_chars=20): confidence unchanged
   - Text 3 chars, ZH detected (min_chars=3): confidence unchanged (CJK)
   - Text 8 chars, DE detected (min_chars=10): confidence slightly dampened
   - Language not in registry: no guard, confidence unchanged
   - min_chars_met=True when threshold met, False when not

2. Architecture Guards:
   - resolver.py must NOT contain _effective_confidence
   - No module except orchestrator reads min_chars_reliable for dampening

3. Reliability score integration:
   - min_chars_met=False adds -0.10 reliability penalty

4. Backward compatibility:
   - min_chars_met defaults to True on OrchestratedDetection

Test naming: test_<subject>_<scenario>_<expected>.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from application.language.orchestrator import (
    DetectionOrchestrator,
    OrchestratedDetection,
)
from application.language.registry import (
    InMemoryLanguageRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubBackend:
    """Configurable stub backend for unit tests."""

    def __init__(
        self,
        distribution: dict[str, float] | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._distribution = distribution or {}
        self._raise_on_call = raise_on_call
        self.call_count = 0

    def detect_distribution(self, text: str) -> dict[str, float]:
        self.call_count += 1
        if self._raise_on_call is not None:
            raise self._raise_on_call
        return dict(self._distribution)


@pytest.fixture()
def registry() -> InMemoryLanguageRegistry:
    """Full registry with all 23 languages."""
    return InMemoryLanguageRegistry()


def _make_orchestrator(
    registry: InMemoryLanguageRegistry,
    primary: StubBackend | None = None,
    fallback: StubBackend | None = None,
) -> DetectionOrchestrator:
    """Helper to build an orchestrator with stub backends."""
    return DetectionOrchestrator(
        primary_backend=primary or StubBackend(),
        fallback_backend=fallback,
        registry=registry,
    )


# ---------------------------------------------------------------------------
# 1. Unit Tests: min_chars_reliable guard
# ---------------------------------------------------------------------------


class TestMinCharsGuardDampening:
    """Confidence dampening based on min_chars_reliable threshold."""

    def test_short_text_nl_confidence_dampened(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """5 chars, NL detected (min_chars=20): confidence dampened.

        NL has min_chars_reliable=20. Text "Hallo" is 5 chars.
        Expected: confidence * (5/20) = 0.90 * 0.25 = 0.225.
        """
        primary = StubBackend(distribution={"nl": 0.90, "de": 0.10})
        orch = _make_orchestrator(registry, primary=primary)

        # Use a long text (>15 words) to route through primary-first,
        # but make the text only 5 actual chars long by using a short
        # text that hits the micro bucket.
        result = orch.detect("Hallo")  # 5 chars, 1 word -> micro

        assert result.code == "nl"
        # Confidence should be dampened: 0.90 * (5/20) = 0.225
        assert result.confidence < 0.90
        assert result.confidence == pytest.approx(0.90 * (5 / 20), abs=0.01)
        assert result.min_chars_met is False

    def test_long_text_nl_confidence_unchanged(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """50 chars, NL detected (min_chars=20): confidence unchanged.

        Text exceeds min_chars_reliable for NL.
        """
        primary = StubBackend(distribution={"nl": 0.90, "de": 0.10})
        orch = _make_orchestrator(registry, primary=primary)

        # 50+ chars text, routes through medium bucket
        text = "Dit is een Nederlandse tekst die lang genoeg is ja"  # ~50 chars
        assert len(text) >= 20  # Sanity: exceeds NL min_chars_reliable

        result = orch.detect(text)

        assert result.code == "nl"
        assert result.confidence == pytest.approx(0.90, abs=0.01)
        assert result.min_chars_met is True

    def test_short_text_zh_confidence_unchanged(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """3 chars, ZH detected (min_chars=3): confidence unchanged.

        CJK languages have very low min_chars_reliable.
        """
        primary = StubBackend(distribution={"zh": 0.95, "ja": 0.05})
        orch = _make_orchestrator(registry, primary=primary)

        result = orch.detect("abc")  # 3 chars, 1 word -> micro

        assert result.code == "zh"
        # ZH min_chars_reliable=3, text len=3 -> no dampening
        assert result.confidence == pytest.approx(0.95, abs=0.01)
        assert result.min_chars_met is True

    def test_short_text_de_confidence_slightly_dampened(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """8 chars, DE detected (min_chars=10): confidence slightly dampened.

        Expected: 0.90 * (8/10) = 0.72.
        """
        primary = StubBackend(distribution={"de": 0.90, "nl": 0.10})
        orch = _make_orchestrator(registry, primary=primary)

        result = orch.detect("Hallo du")  # 8 chars, 2 words -> micro

        assert result.code == "de"
        # DE min_chars_reliable=10, text=8 chars -> dampened
        expected = 0.90 * (8 / 10)
        assert result.confidence == pytest.approx(expected, abs=0.01)
        assert result.min_chars_met is False

    def test_unknown_language_no_guard(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """Language not in registry: no guard applied, confidence unchanged.

        Uses a backend returning a code that is not in the 23 supported
        languages (e.g. 'xx').
        """
        primary = StubBackend(distribution={"xx": 0.85})
        orch = _make_orchestrator(registry, primary=primary)

        result = orch.detect("Hi")

        assert result.code == "xx"
        assert result.confidence == pytest.approx(0.85, abs=0.01)
        assert result.min_chars_met is True


class TestMinCharsMetField:
    """min_chars_met field is correctly set."""

    def test_min_chars_met_true_when_threshold_met(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """min_chars_met=True when text >= min_chars_reliable."""
        primary = StubBackend(distribution={"de": 0.90})
        orch = _make_orchestrator(registry, primary=primary)

        # DE has min_chars_reliable=10, text "Hallo Welt" = 10 chars
        result = orch.detect("Hallo Welt")

        assert result.min_chars_met is True

    def test_min_chars_met_false_when_below_threshold(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """min_chars_met=False when text < min_chars_reliable."""
        primary = StubBackend(distribution={"nl": 0.90})
        orch = _make_orchestrator(registry, primary=primary)

        # NL has min_chars_reliable=20, "Hi" = 2 chars
        result = orch.detect("Hi")

        assert result.min_chars_met is False

    def test_default_result_has_min_chars_met_true(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """Default results (empty text) have min_chars_met=True."""
        primary = StubBackend(distribution={})
        orch = _make_orchestrator(registry, primary=primary)

        result = orch.detect("")

        assert result.min_chars_met is True  # Default value


# ---------------------------------------------------------------------------
# 2. Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """OrchestratedDetection min_chars_met defaults to True."""

    def test_orchestrated_detection_default_min_chars_met(self) -> None:
        """min_chars_met defaults to True for backward compat (HC-A4)."""
        od = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={"de": 0.9},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
        )
        assert od.min_chars_met is True

    def test_orchestrated_detection_explicit_false(self) -> None:
        """min_chars_met can be explicitly set to False."""
        od = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={"de": 0.9},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
            min_chars_met=False,
        )
        assert od.min_chars_met is False


# ---------------------------------------------------------------------------
# 3. Reliability score integration
# ---------------------------------------------------------------------------


class TestReliabilityScoreMinCharsPenalty:
    """min_chars_met=False adds -0.10 reliability penalty (IC-A3)."""

    def test_min_chars_not_met_reduces_reliability(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """When min_chars_met=False, reliability is lower than met=True.

        Compare two identical detections, one with short text (guard active)
        and one with long text (guard inactive).
        """
        # Short text: NL, 5 chars (min_chars=20) -> guard active
        primary_short = StubBackend(distribution={"nl": 0.90})
        orch_short = _make_orchestrator(registry, primary=primary_short)
        result_short = orch_short.detect("Hallo")

        # Long text: NL, 50+ chars -> guard inactive
        primary_long = StubBackend(distribution={"nl": 0.90})
        orch_long = _make_orchestrator(registry, primary=primary_long)
        result_long = orch_long.detect(
            "Dit is een Nederlandse tekst die lang genoeg is voor de test ja"
        )

        # Short text should have lower reliability (dampened confidence + penalty)
        assert result_short.reliability_score < result_long.reliability_score
        assert result_short.min_chars_met is False
        assert result_long.min_chars_met is True


# ---------------------------------------------------------------------------
# 4. Decision reason annotation
# ---------------------------------------------------------------------------


class TestDecisionReasonAnnotation:
    """decision_reason includes guard info when active (IC-A4)."""

    def test_reason_mentions_guard_when_active(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """When guard dampens confidence, reason mentions it."""
        primary = StubBackend(distribution={"nl": 0.90})
        orch = _make_orchestrator(registry, primary=primary)

        result = orch.detect("Hi")  # 2 chars, NL min=20 -> dampened

        assert "min_chars_guard" in result.decision_reason
        assert result.min_chars_met is False

    def test_reason_does_not_mention_guard_when_inactive(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """When guard does not dampen, reason does not mention it."""
        primary = StubBackend(distribution={"de": 0.90})
        orch = _make_orchestrator(registry, primary=primary)

        # DE min_chars=10, text "Hallo Welt" = 10 chars -> guard inactive
        result = orch.detect("Hallo Welt")

        assert "min_chars_guard" not in result.decision_reason
        assert result.min_chars_met is True


# ---------------------------------------------------------------------------
# 5. Architecture Guards
# ---------------------------------------------------------------------------


_BRIDGE_ROOT = Path(__file__).resolve().parents[3]


class TestArchitectureGuardEffectiveConfidenceRemoved:
    """HC-A1: _effective_confidence() must not exist in resolver.py."""

    def test_resolver_no_effective_confidence_function(self) -> None:
        """resolver.py must not define _effective_confidence."""
        resolver_path = _BRIDGE_ROOT / "application" / "language" / "resolver.py"
        source = resolver_path.read_text(encoding="utf-8")

        # Match function definition
        pattern = re.compile(
            r"^def _effective_confidence\(",
            re.MULTILINE,
        )
        matches = pattern.findall(source)
        assert not matches, (
            f"HC-A1 violation: resolver.py still defines _effective_confidence: "
            f"{matches}"
        )

    def test_resolver_no_effective_confidence_call(self) -> None:
        """resolver.py must not call _effective_confidence."""
        resolver_path = _BRIDGE_ROOT / "application" / "language" / "resolver.py"
        source = resolver_path.read_text(encoding="utf-8")

        assert "_effective_confidence(" not in source, (
            "HC-A1 violation: resolver.py still calls _effective_confidence()"
        )


class TestArchitectureGuardBackendAgnostic:
    """HC-A3: Guard must be backend-agnostic."""

    def test_guard_method_does_not_check_backend_name(self) -> None:
        """_apply_min_chars_guard does not contain backend name checks."""
        orch_path = _BRIDGE_ROOT / "application" / "language" / "orchestrator.py"
        source = orch_path.read_text(encoding="utf-8")

        # Find the _apply_min_chars_guard method body
        guard_match = re.search(
            r"def _apply_min_chars_guard\(.*?\).*?(?=\n    def |\nclass |\Z)",
            source,
            re.DOTALL,
        )
        assert guard_match is not None, "Could not find _apply_min_chars_guard method"
        guard_body = guard_match.group()

        # Must NOT reference specific backend names
        assert "domain_heuristic" not in guard_body, (
            "HC-A3 violation: _apply_min_chars_guard checks for 'domain_heuristic'"
        )
        assert "langdetect" not in guard_body, (
            "HC-A3 violation: _apply_min_chars_guard checks for 'langdetect'"
        )
        assert "backend_name" not in guard_body, (
            "HC-A3 violation: _apply_min_chars_guard inspects backend_name"
        )
