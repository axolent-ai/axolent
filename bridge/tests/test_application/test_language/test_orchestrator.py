"""Tests for DetectionOrchestrator (Phase 2 Core, Step 3/4).

Covers:
1. Unit Tests:
   - Short text: domain heuristic first
   - Long text: primary (langdetect) first
   - Primary confidence < fallback_threshold: fallback consulted
   - Primary confidence >= fallback_threshold: fallback NOT consulted
   - Backend exception: OrchestratedDetection with error in candidate
   - No backend delivers: default ("de") with confidence=0.0

2. Integration Tests:
   - Real langdetect backend: German text -> "de", confidence > 0.8
   - Real langdetect backend: Dutch text -> "nl"

3. Property-Based Tests:
   - code always in registry or default
   - len(candidates) >= 1 (except empty input)
   - reliability_score in [0.0, 1.0]
   - text_length_bucket is one of 4 valid values

4. Architecture-Guard Tests:
   - Guard 3: Backend codes normalized (Norwegian "no" -> "nb")
   - Guard 5: Orchestrator delegates code mapping to registry

5. Edge-Case Tests:
   - Empty string, pure emojis, whitespace-only: default
   - Very long text (10k+ words): backend called only once
   - Backend returns empty distribution: default

Test naming convention: test_<subject>_<scenario>_<expected>.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from application.language.context import TEXT_LENGTH_BUCKETS
from application.language.orchestrator import (
    DetectionCandidate,
    DetectionOrchestrator,
    OrchestratedDetection,
)
from application.language.registry import (
    InMemoryLanguageRegistry,
)


def _has_langdetect() -> bool:
    """Check if langdetect is importable (not always in test venv)."""
    try:
        import langdetect  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubBackend:
    """Configurable stub backend for unit tests.

    Implements LanguageDetectorBackend protocol.
    """

    def __init__(
        self,
        distribution: dict[str, float] | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._distribution = distribution or {}
        self._raise_on_call = raise_on_call
        self.call_count = 0
        self.last_text: str | None = None

    def detect_distribution(self, text: str) -> dict[str, float]:
        self.call_count += 1
        self.last_text = text
        if self._raise_on_call is not None:
            raise self._raise_on_call
        return dict(self._distribution)


@pytest.fixture()
def registry() -> InMemoryLanguageRegistry:
    """Full registry with all 23 languages."""
    return InMemoryLanguageRegistry()


@pytest.fixture()
def high_conf_primary() -> StubBackend:
    """Primary backend returning German with high confidence."""
    return StubBackend(distribution={"de": 0.93, "nl": 0.05, "en": 0.02})


@pytest.fixture()
def low_conf_primary() -> StubBackend:
    """Primary backend returning German with low confidence."""
    return StubBackend(distribution={"de": 0.45, "nl": 0.30, "en": 0.25})


@pytest.fixture()
def high_conf_heuristic() -> StubBackend:
    """Fallback heuristic returning German with high confidence."""
    return StubBackend(distribution={"de": 0.85})


@pytest.fixture()
def low_conf_heuristic() -> StubBackend:
    """Fallback heuristic returning Dutch with low confidence."""
    return StubBackend(distribution={"nl": 0.55})


@pytest.fixture()
def failing_backend() -> StubBackend:
    """Backend that always raises an exception."""
    return StubBackend(raise_on_call=RuntimeError("Backend crash"))


@pytest.fixture()
def empty_backend() -> StubBackend:
    """Backend that returns empty distribution."""
    return StubBackend(distribution={})


def _make_orchestrator(
    registry: InMemoryLanguageRegistry,
    primary: StubBackend | None = None,
    fallback: StubBackend | None = None,
    fallback_threshold: float = 0.6,
    short_text_threshold_words: int = 15,
) -> DetectionOrchestrator:
    """Helper to build an orchestrator with stub backends."""
    return DetectionOrchestrator(
        primary_backend=primary or StubBackend(),
        fallback_backend=fallback,
        registry=registry,
        fallback_threshold=fallback_threshold,
        short_text_threshold_words=short_text_threshold_words,
    )


# Short text: fewer than 15 words (micro/short bucket)
SHORT_TEXT = "Hallo Welt"  # 2 words -> micro
SHORT_TEXT_12 = "Dies ist ein kurzer deutscher Text mit genau zwoelf einzelnen Woertern hier"  # 12 words -> short

# Long text: more than 30 words (medium bucket, 31-100 words)
LONG_TEXT = (
    "Dies ist ein laengerer Text der aus mehr als dreissig Woertern besteht "
    "und daher als mittellanger Text klassifiziert werden sollte damit die "
    "Primary-Backend-Strategie zum Einsatz kommt und getestet wird und "
    "damit wir sicherstellen dass der Bucket korrekt ist"
)  # 35 words -> medium


# ---------------------------------------------------------------------------
# 1. Unit Tests
# ---------------------------------------------------------------------------


class TestShortTextRoutesToHeuristic:
    """Spec: short text (<15 words) consults domain heuristic first."""

    def test_short_text_heuristic_first_high_conf(
        self,
        registry: InMemoryLanguageRegistry,
        high_conf_heuristic: StubBackend,
        high_conf_primary: StubBackend,
    ) -> None:
        """Short text + heuristic confident: primary NOT called."""
        orch = _make_orchestrator(
            registry,
            primary=high_conf_primary,
            fallback=high_conf_heuristic,
        )
        result = orch.detect(SHORT_TEXT)

        assert result.code == "de"
        assert result.confidence >= 0.7
        assert high_conf_heuristic.call_count == 1
        assert high_conf_primary.call_count == 0
        assert result.text_length_bucket == "micro"

    def test_short_text_heuristic_low_conf_calls_primary(
        self,
        registry: InMemoryLanguageRegistry,
        low_conf_heuristic: StubBackend,
        high_conf_primary: StubBackend,
    ) -> None:
        """Short text + heuristic not confident: primary also called."""
        orch = _make_orchestrator(
            registry,
            primary=high_conf_primary,
            fallback=low_conf_heuristic,
        )
        result = orch.detect(SHORT_TEXT)

        assert low_conf_heuristic.call_count == 1
        assert high_conf_primary.call_count == 1
        # Primary detected "de", heuristic detected "nl" -> dissent
        assert result.code in ("de", "nl")


class TestLongTextRoutesToPrimary:
    """Spec: long text (>= 15 words) consults primary first."""

    def test_long_text_primary_first(
        self,
        registry: InMemoryLanguageRegistry,
        high_conf_primary: StubBackend,
        high_conf_heuristic: StubBackend,
    ) -> None:
        """Long text: primary called, heuristic NOT called if confident."""
        orch = _make_orchestrator(
            registry,
            primary=high_conf_primary,
            fallback=high_conf_heuristic,
        )
        result = orch.detect(LONG_TEXT)

        assert result.code == "de"
        assert high_conf_primary.call_count == 1
        assert high_conf_heuristic.call_count == 0
        assert result.text_length_bucket == "medium"


class TestFallbackThreshold:
    """Primary confidence vs fallback_threshold controls fallback activation."""

    def test_primary_below_threshold_activates_fallback(
        self,
        registry: InMemoryLanguageRegistry,
        low_conf_primary: StubBackend,
        high_conf_heuristic: StubBackend,
    ) -> None:
        """Primary confidence 0.45 < threshold 0.6: fallback consulted.

        Note: for medium-length text the fallback IS consulted.
        For long text it is NOT (IC-O8).
        """
        # Use a text that puts us in "medium" bucket (31-100 words)
        orch = _make_orchestrator(
            registry,
            primary=low_conf_primary,
            fallback=high_conf_heuristic,
        )
        result = orch.detect(LONG_TEXT)

        assert result.text_length_bucket == "medium"
        assert low_conf_primary.call_count == 1
        assert high_conf_heuristic.call_count == 1

    def test_primary_above_threshold_skips_fallback(
        self,
        registry: InMemoryLanguageRegistry,
        high_conf_primary: StubBackend,
        high_conf_heuristic: StubBackend,
    ) -> None:
        """Primary confidence 0.93 >= threshold 0.6: fallback NOT consulted."""
        orch = _make_orchestrator(
            registry,
            primary=high_conf_primary,
            fallback=high_conf_heuristic,
        )
        result = orch.detect(LONG_TEXT)

        assert high_conf_primary.call_count == 1
        assert high_conf_heuristic.call_count == 0
        assert result.confidence == pytest.approx(0.93, abs=0.01)

    def test_long_text_bucket_skips_fallback_even_below_threshold(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """IC-O8: 'long' bucket (101+ words) never consults fallback."""
        low_primary = StubBackend(distribution={"en": 0.40, "de": 0.30})
        heuristic = StubBackend(distribution={"de": 0.90})
        orch = _make_orchestrator(registry, primary=low_primary, fallback=heuristic)

        # Generate a text with 102 words
        very_long_text = " ".join(["word"] * 102)
        result = orch.detect(very_long_text)

        assert low_primary.call_count == 1
        assert heuristic.call_count == 0
        assert result.text_length_bucket == "long"


class TestBackendException:
    """Backend raising an exception: detection still works."""

    def test_primary_exception_returns_result_with_error(
        self,
        registry: InMemoryLanguageRegistry,
        failing_backend: StubBackend,
        high_conf_heuristic: StubBackend,
    ) -> None:
        """Primary crashes: result has error in candidate, fallback used."""
        orch = _make_orchestrator(
            registry,
            primary=failing_backend,
            fallback=high_conf_heuristic,
        )
        # Use short text so heuristic is called first, then primary fails
        result = orch.detect(LONG_TEXT)

        # At least one candidate should have an error
        errors = [c for c in result.candidates if c.error is not None]
        assert len(errors) >= 1
        assert "RuntimeError" in errors[0].error  # type: ignore[operator]

    def test_both_backends_fail_returns_default(
        self,
        registry: InMemoryLanguageRegistry,
    ) -> None:
        """Both backends fail: default 'de' with confidence 0.0."""
        failing1 = StubBackend(raise_on_call=RuntimeError("crash1"))
        failing2 = StubBackend(raise_on_call=RuntimeError("crash2"))
        orch = _make_orchestrator(registry, primary=failing1, fallback=failing2)
        result = orch.detect(SHORT_TEXT)

        assert result.code == "de"
        assert result.confidence == 0.0


class TestNoBackendDelivers:
    """No backend delivers a result: default returned."""

    def test_empty_backends_return_default(
        self,
        registry: InMemoryLanguageRegistry,
        empty_backend: StubBackend,
    ) -> None:
        """Both backends return empty distribution: default."""
        empty2 = StubBackend(distribution={})
        orch = _make_orchestrator(registry, primary=empty_backend, fallback=empty2)
        result = orch.detect(SHORT_TEXT)

        assert result.code == "de"
        assert result.confidence == 0.0

    def test_no_fallback_configured_returns_default(
        self,
        registry: InMemoryLanguageRegistry,
        empty_backend: StubBackend,
    ) -> None:
        """Primary empty, no fallback: default."""
        orch = _make_orchestrator(registry, primary=empty_backend, fallback=None)
        result = orch.detect(SHORT_TEXT)

        assert result.code == "de"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 2. Integration Tests (real langdetect)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_langdetect(),
    reason="langdetect not installed; skipping integration tests",
)
class TestIntegrationRealBackends:
    """End-to-end tests with real LangdetectBackend."""

    @pytest.fixture()
    def real_orchestrator(
        self, registry: InMemoryLanguageRegistry
    ) -> DetectionOrchestrator:
        """Orchestrator with real langdetect backend."""
        from application.language.backends import (
            DomainLanguageBackend,
            LangdetectBackend,
        )

        return DetectionOrchestrator(
            primary_backend=LangdetectBackend(),
            fallback_backend=DomainLanguageBackend(),
            registry=registry,
        )

    def test_german_text_detected_as_de(
        self, real_orchestrator: DetectionOrchestrator
    ) -> None:
        """German text -> code='de', confidence > 0.8."""
        text = (
            "Dies ist ein deutscher Text der lang genug ist damit "
            "langdetect ihn zuverlaessig als Deutsch erkennen kann "
            "und eine hohe Confidence zurueckgibt."
        )
        result = real_orchestrator.detect(text)
        assert result.code == "de"
        assert result.confidence > 0.8

    def test_dutch_text_detected_as_nl(
        self, real_orchestrator: DetectionOrchestrator
    ) -> None:
        """Dutch text -> code='nl' (the fail-case from Phase-1 RCA)."""
        text = (
            "Dit is een Nederlandse tekst die lang genoeg is zodat "
            "langdetect het betrouwbaar als Nederlands kan herkennen "
            "en een hoge betrouwbaarheidsscore teruggeeft."
        )
        result = real_orchestrator.detect(text)
        assert result.code == "nl"


# ---------------------------------------------------------------------------
# 3. Property-Based Tests
# ---------------------------------------------------------------------------


class TestPropertyBased:
    """Invariant tests across different inputs."""

    @pytest.mark.parametrize(
        "text",
        [
            "Hallo Welt",
            "This is a test",
            "Ceci est un test pour la detection de langue",
            " ".join(["word"] * 200),
            "",
        ],
    )
    def test_code_always_in_registry_or_default(
        self,
        registry: InMemoryLanguageRegistry,
        text: str,
    ) -> None:
        """OrchestratedDetection.code is always registry-known or 'de'."""
        primary = StubBackend(distribution={"en": 0.90, "de": 0.10})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(text)

        assert registry.is_supported(result.code) or result.code == "de"

    @pytest.mark.parametrize(
        "text",
        [
            "Kurzer Text",
            "Ein laengerer Text mit mehr als fuenfzehn Woertern der in den Medium Bucket faellt und primary nutzt",
        ],
    )
    def test_at_least_one_candidate(
        self,
        registry: InMemoryLanguageRegistry,
        text: str,
    ) -> None:
        """len(candidates) >= 1 for non-empty input."""
        primary = StubBackend(distribution={"de": 0.90})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(text)

        assert len(result.candidates) >= 1

    @pytest.mark.parametrize(
        "distribution",
        [
            {"de": 0.99},
            {"nl": 0.50, "de": 0.30, "en": 0.20},
            {"en": 0.01},
        ],
    )
    def test_reliability_score_in_range(
        self,
        registry: InMemoryLanguageRegistry,
        distribution: dict[str, float],
    ) -> None:
        """reliability_score is always in [0.0, 1.0]."""
        primary = StubBackend(distribution=distribution)
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect("Test text for reliability scoring")

        assert 0.0 <= result.reliability_score <= 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "Hi",
            "Short text here",
            " ".join(["word"] * 50),
            " ".join(["word"] * 200),
            "",
        ],
    )
    def test_text_length_bucket_valid(
        self,
        registry: InMemoryLanguageRegistry,
        text: str,
    ) -> None:
        """text_length_bucket is always one of the 4 valid values."""
        primary = StubBackend(distribution={"de": 0.80})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(text)

        assert result.text_length_bucket in TEXT_LENGTH_BUCKETS


# ---------------------------------------------------------------------------
# 4. Architecture-Guard Tests
# ---------------------------------------------------------------------------


class TestArchitectureGuards:
    """Verify code normalization and registry delegation."""

    def test_guard3_norwegian_code_normalized(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Guard 3: Backend returning 'no' gets normalized to 'nb'."""
        # langdetect returns "no" for Norwegian
        primary = StubBackend(distribution={"no": 0.90, "da": 0.10})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(
            "Dette er en norsk tekst som er lang nok til at den burde bli gjenkjent"
        )

        assert result.code == "nb", f"Expected 'nb', got '{result.code}'"
        # Distribution should also have normalized keys
        assert "no" not in result.distribution
        if result.distribution:
            assert "nb" in result.distribution

    def test_guard3_chinese_code_normalized(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Guard 3: Backend returning 'zh-cn' gets normalized to 'zh'."""
        primary = StubBackend(distribution={"zh-cn": 0.85, "ja": 0.15})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(
            "This text is long enough for medium bucket classification test purposes"
        )

        assert result.code == "zh"
        assert "zh-cn" not in result.distribution

    def test_guard5_orchestrator_delegates_to_registry(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Guard 5: Orchestrator uses registry.resolve_backend_code(),
        not its own mapping.
        """
        # Patch resolve_backend_code to track calls
        original_resolve = registry.resolve_backend_code
        resolve_calls: list[str] = []

        def tracking_resolve(code: str) -> str:
            resolve_calls.append(code)
            return original_resolve(code)

        registry.resolve_backend_code = tracking_resolve  # type: ignore[assignment]

        primary = StubBackend(distribution={"no": 0.70, "da": 0.30})
        orch = _make_orchestrator(registry, primary=primary)
        orch.detect("Test tekst for registry delegation checking purposes here")

        # "no" and "da" should both have been resolved through registry
        assert "no" in resolve_calls
        assert "da" in resolve_calls

    def test_no_direct_domain_language_import(self) -> None:
        """HC-O5: orchestrator.py does not import from domain.language.

        Checks actual Python import statements (not docstring prose).
        Uses the same regex approach as the architecture guard in
        test_langdetect_isolation.py.
        """
        repo_root = Path(__file__).resolve().parents[3]  # bridge/
        orchestrator_path = repo_root / "application" / "language" / "orchestrator.py"
        source = orchestrator_path.read_text(encoding="utf-8")

        # Match only real import lines (beginning of line), not docstring text
        pattern = re.compile(
            r"^(?:from domain\.language|import domain\.language)",
            re.MULTILINE,
        )
        matches = pattern.findall(source)
        assert not matches, (
            f"HC-O5 violation: orchestrator.py imports domain.language "
            f"directly: {matches}"
        )


# ---------------------------------------------------------------------------
# 5. Edge-Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty, emojis, whitespace, very long text."""

    def test_empty_string_returns_default(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Empty string: default 'de' with confidence 0.0."""
        primary = StubBackend(distribution={"en": 0.90})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect("")

        assert result.code == "de"
        assert result.confidence == 0.0
        assert primary.call_count == 0

    def test_whitespace_only_returns_default(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Whitespace-only text: default."""
        primary = StubBackend(distribution={"en": 0.90})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect("   \n\t   ")

        assert result.code == "de"
        assert result.confidence == 0.0
        assert primary.call_count == 0

    def test_pure_emojis_treated_as_empty(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Pure emoji text: backends may return empty, should get default.

        Note: emojis are non-empty after strip(), so backends ARE called.
        The test verifies the orchestrator handles the empty-distribution
        case that backends return for emoji-only input.
        """
        primary = StubBackend(distribution={})  # Backend returns nothing for emojis
        fallback = StubBackend(distribution={})
        orch = _make_orchestrator(registry, primary=primary, fallback=fallback)
        result = orch.detect("\U0001f600\U0001f600\U0001f600")

        assert result.code == "de"
        assert result.confidence == 0.0

    def test_very_long_text_backend_called_once(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """10k+ words: backend called exactly once (no chunking)."""
        primary = StubBackend(distribution={"en": 0.95})
        orch = _make_orchestrator(registry, primary=primary)

        huge_text = " ".join(["word"] * 10_001)
        result = orch.detect(huge_text)

        assert primary.call_count == 1
        assert result.text_length_bucket == "long"
        assert result.code == "en"

    def test_backend_empty_distribution_gives_default(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Backend returns empty dict: treated as failure, default returned."""
        primary = StubBackend(distribution={})
        orch = _make_orchestrator(registry, primary=primary, fallback=None)
        result = orch.detect("Some text here")

        assert result.code == "de"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


class TestDataclassInvariants:
    """Verify frozen/slots on DetectionCandidate and OrchestratedDetection."""

    def test_detection_candidate_frozen(self) -> None:
        """HC-O6: DetectionCandidate is frozen."""
        c = DetectionCandidate(
            backend_name="test",
            distribution={"de": 0.9},
            top_lang="de",
            top_confidence=0.9,
            latency_ms=1.0,
        )
        with pytest.raises(AttributeError):
            c.top_lang = "en"  # type: ignore[misc]

    def test_detection_candidate_slots(self) -> None:
        """HC-O6: DetectionCandidate uses slots."""
        c = DetectionCandidate(
            backend_name="test",
            distribution={},
            top_lang="",
            top_confidence=0.0,
            latency_ms=0.0,
        )
        assert not hasattr(c, "__dict__")

    def test_orchestrated_detection_frozen(self) -> None:
        """HC-O6: OrchestratedDetection is frozen."""
        od = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={"de": 0.9},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
        )
        with pytest.raises(AttributeError):
            od.code = "en"  # type: ignore[misc]

    def test_orchestrated_detection_slots(self) -> None:
        """HC-O6: OrchestratedDetection uses slots."""
        od = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
        )
        assert not hasattr(od, "__dict__")

    def test_detection_candidate_succeeded_property(self) -> None:
        """succeeded is True when no error and top_lang is non-empty."""
        good = DetectionCandidate(
            backend_name="test",
            distribution={"de": 0.9},
            top_lang="de",
            top_confidence=0.9,
            latency_ms=1.0,
        )
        bad_error = DetectionCandidate(
            backend_name="test",
            distribution={},
            top_lang="",
            top_confidence=0.0,
            latency_ms=1.0,
            error="failed",
        )
        bad_empty = DetectionCandidate(
            backend_name="test",
            distribution={},
            top_lang="",
            top_confidence=0.0,
            latency_ms=1.0,
        )

        assert good.succeeded is True
        assert bad_error.succeeded is False
        assert bad_empty.succeeded is False


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """DetectionOrchestrator satisfies DetectionOrchestratorProtocol."""

    def test_orchestrator_has_detect(self, registry: InMemoryLanguageRegistry) -> None:
        """detect() method exists and is callable."""
        orch = _make_orchestrator(registry, primary=StubBackend())
        assert callable(orch.detect)

    def test_orchestrator_has_primary_backend_name(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """primary_backend_name property exists."""
        orch = _make_orchestrator(registry, primary=StubBackend())
        assert orch.primary_backend_name == "langdetect"

    def test_orchestrator_registered_backends_with_fallback(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """registered_backends lists both when fallback is configured."""
        orch = _make_orchestrator(
            registry, primary=StubBackend(), fallback=StubBackend()
        )
        assert orch.registered_backends == ["langdetect", "domain_heuristic"]

    def test_orchestrator_registered_backends_no_fallback(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """registered_backends lists only primary when no fallback."""
        orch = _make_orchestrator(registry, primary=StubBackend(), fallback=None)
        assert orch.registered_backends == ["langdetect"]


# ---------------------------------------------------------------------------
# Decision reason (HC-O9)
# ---------------------------------------------------------------------------


class TestDecisionReason:
    """HC-O9: decision_reason is human-readable audit string."""

    def test_reason_contains_backend_name(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Reason mentions which backend made the decision."""
        primary = StubBackend(distribution={"de": 0.93})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(LONG_TEXT)

        assert (
            "langdetect" in result.decision_reason
            or "Primary" in result.decision_reason
        )

    def test_reason_contains_language_code(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Reason mentions the detected language code."""
        primary = StubBackend(distribution={"fr": 0.88})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(LONG_TEXT)

        assert "fr" in result.decision_reason

    def test_reason_contains_confidence(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Reason mentions the confidence value."""
        primary = StubBackend(distribution={"de": 0.93})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(LONG_TEXT)

        assert "0.93" in result.decision_reason

    def test_reason_mentions_dissent(self, registry: InMemoryLanguageRegistry) -> None:
        """When backends disagree, reason mentions dissent."""
        primary = StubBackend(distribution={"de": 0.80})
        fallback = StubBackend(distribution={"nl": 0.60})
        orch = _make_orchestrator(registry, primary=primary, fallback=fallback)
        result = orch.detect(SHORT_TEXT)

        assert (
            "Dissent" in result.decision_reason
            or "disagree" in result.decision_reason.lower()
        )

    def test_reason_for_default_result(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Default result has informative reason."""
        orch = _make_orchestrator(registry, primary=StubBackend())
        result = orch.detect("")

        assert (
            "default" in result.decision_reason.lower()
            or "empty" in result.decision_reason.lower()
        )


# ---------------------------------------------------------------------------
# Consensus vs Dissent aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    """Consensus and dissent scenarios."""

    def test_consensus_uses_max_confidence(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Both backends agree on 'de': confidence = max of both.

        Use short text where heuristic confidence is below 0.7
        so that primary is also consulted. Both agree on 'de'.
        """
        primary = StubBackend(distribution={"de": 0.80})
        fallback = StubBackend(
            distribution={"de": 0.65}
        )  # below 0.7 -> primary also called
        orch = _make_orchestrator(registry, primary=primary, fallback=fallback)
        result = orch.detect(SHORT_TEXT)  # Short -> heuristic first, then primary

        assert result.code == "de"
        assert result.confidence == pytest.approx(0.80, abs=0.01)
        assert "Consensus" in result.decision_reason

    def test_dissent_higher_confidence_wins(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Backends disagree: higher confidence wins."""
        primary = StubBackend(distribution={"en": 0.85})
        fallback = StubBackend(distribution={"de": 0.60})
        orch = _make_orchestrator(registry, primary=primary, fallback=fallback)
        result = orch.detect(SHORT_TEXT)

        # Heuristic (fallback) called first for short text: de 0.60
        # Then primary: en 0.85
        # Primary wins (higher confidence)
        assert result.code == "en"

    def test_dissent_penalizes_reliability(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Dissent reduces reliability_score by 0.20."""
        # For short text: heuristic first, then primary
        primary = StubBackend(distribution={"en": 0.85})
        fallback = StubBackend(distribution={"de": 0.60})

        orch_dissent = _make_orchestrator(registry, primary=primary, fallback=fallback)
        result_dissent = orch_dissent.detect(SHORT_TEXT)

        # Compare with a non-dissent case
        primary2 = StubBackend(distribution={"en": 0.85})
        orch_no_dissent = _make_orchestrator(registry, primary=primary2, fallback=None)
        result_no_dissent = orch_no_dissent.detect(LONG_TEXT)

        # Dissent result should have lower reliability
        # (not a strict 0.20 diff because bucket and tier also factor in)
        assert result_dissent.reliability_score < result_no_dissent.reliability_score


# ---------------------------------------------------------------------------
# Reliability score computation
# ---------------------------------------------------------------------------


class TestReliabilityScore:
    """Verify the reliability_score formula (IC-O4)."""

    def test_high_tier_gets_bonus(self, registry: InMemoryLanguageRegistry) -> None:
        """German (HIGH tier) gets +0.05 tier bonus."""
        primary = StubBackend(distribution={"de": 0.80})
        orch = _make_orchestrator(registry, primary=primary)
        result_de = orch.detect(LONG_TEXT)

        primary2 = StubBackend(distribution={"en": 0.80})
        orch2 = _make_orchestrator(registry, primary=primary2)
        result_en = orch2.detect(LONG_TEXT)

        # DE (HIGH) should have higher reliability than EN (MEDIUM) at same confidence
        assert result_de.reliability_score > result_en.reliability_score

    def test_longer_text_gets_bonus(self, registry: InMemoryLanguageRegistry) -> None:
        """Longer text gets length bonus (+0.05 for long vs -0.05 for micro)."""
        # Same backend and language, different text lengths
        primary1 = StubBackend(distribution={"en": 0.80})
        orch1 = _make_orchestrator(registry, primary=primary1)
        result_micro = orch1.detect("Hello")

        primary2 = StubBackend(distribution={"en": 0.80})
        orch2 = _make_orchestrator(registry, primary=primary2)
        result_long = orch2.detect(" ".join(["hello"] * 102))

        # Long bucket (+0.05) vs micro (-0.05) = 0.10 difference
        assert result_long.reliability_score > result_micro.reliability_score

    def test_reliability_clamped_to_unit_interval(
        self, registry: InMemoryLanguageRegistry
    ) -> None:
        """Score is clamped to [0.0, 1.0] even with extreme inputs."""
        # Very high confidence + HIGH tier + long text could exceed 1.0
        primary = StubBackend(distribution={"de": 0.99})
        orch = _make_orchestrator(registry, primary=primary)
        result = orch.detect(" ".join(["wort"] * 102))

        assert result.reliability_score <= 1.0
        assert result.reliability_score >= 0.0
