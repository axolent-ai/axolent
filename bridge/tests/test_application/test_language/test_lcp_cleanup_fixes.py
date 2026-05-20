"""Tests for LCP Cleanup Bundle fixes (Codex Findings 4-8, Claude Issues).

Covers runtime behavior that the architecture guards in
test_architecture/test_lcp_cleanup.py verify at the source level.

Test groups:
1. Finding 6: Orchestrator dissent distribution consistency
2. Finding 7: StreamGuard report_final_outcome integration
3. Claude Fix 1: Module-level registry singleton in resolver.py
4. Claude Fix 2: detection_tier computation fix
5. Claude Fix 3: Dead _check_time code removed from StreamGuard
"""

from __future__ import annotations


from application.language.context import LanguageContext
from application.language.orchestrator import (
    DetectionOrchestrator,
    OrchestratedDetection,
)
from application.language.registry import InMemoryLanguageRegistry
from application.language.backends import DomainLanguageBackend
from application.language.stream_guard import StreamGuard, StreamGuardStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Stub backend for StreamGuard (avoids langdetect dependency)
_stub_backend = DomainLanguageBackend()


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


# ---------------------------------------------------------------------------
# Finding 6: Orchestrator dissent distribution consistency
# ---------------------------------------------------------------------------


class TestOrchestratorDissentDistribution:
    """Codex Finding 6: distribution must be consistent with winner code."""

    def test_dissent_distribution_matches_winner(self) -> None:
        """When backends disagree, distribution must come from the winner."""
        # Primary says "de" with 0.8, fallback says "nl" with 0.6
        primary = StubBackend(distribution={"de": 0.80, "nl": 0.15, "en": 0.05})
        fallback = StubBackend(distribution={"nl": 0.60, "de": 0.30, "en": 0.10})
        registry = InMemoryLanguageRegistry()

        orch = DetectionOrchestrator(
            primary_backend=primary,
            fallback_backend=fallback,
            registry=registry,
            fallback_threshold=0.9,  # Force fallback consultation
        )

        # Use "medium" length text to trigger long-text strategy
        text = " ".join(["Hallo Welt wie geht es dir heute"] * 10)
        result = orch.detect(text)

        # Winner should be primary ("de" at 0.80 > "nl" at 0.60)
        assert result.code == "de"
        # Distribution must show "de" as top, not "nl"
        assert result.distribution.get("de", 0) >= result.distribution.get("nl", 0)
        # The had_dissent flag must be True
        assert result.had_dissent is True

    def test_consensus_distribution_consistent(self) -> None:
        """When backends agree, distribution matches winner."""
        primary = StubBackend(distribution={"de": 0.93, "nl": 0.05})
        fallback = StubBackend(distribution={"de": 0.85})
        registry = InMemoryLanguageRegistry()

        orch = DetectionOrchestrator(
            primary_backend=primary,
            fallback_backend=fallback,
            registry=registry,
        )

        # Short text to trigger heuristic-first path
        result = orch.detect("Hallo Welt")

        assert result.code == "de"
        assert "de" in result.distribution
        assert result.had_dissent is False

    def test_single_backend_no_dissent(self) -> None:
        """Single backend result has had_dissent=False."""
        primary = StubBackend(distribution={"de": 0.93})
        registry = InMemoryLanguageRegistry()

        orch = DetectionOrchestrator(
            primary_backend=primary,
            fallback_backend=None,
            registry=registry,
        )

        # Long text, no fallback
        text = " ".join(["Dies ist ein Test"] * 30)
        result = orch.detect(text)

        assert result.had_dissent is False
        assert result.code == "de"


# ---------------------------------------------------------------------------
# Finding 7: StreamGuard report_final_outcome integration
# ---------------------------------------------------------------------------


class TestStreamGuardReportFinalOutcome:
    """Codex Finding 7: report_final_outcome must be callable and update stats."""

    def test_report_final_outcome_confirmed_abort(self) -> None:
        """Confirmed abort (bad language) updates confirmed_aborts counter."""
        guard = StreamGuard(expected_lang="de", enabled=True, backend=_stub_backend)
        stats = StreamGuardStats()

        # Simulate: guard checked and aborted
        guard._state.check_performed = True
        guard._state.aborted = True
        guard._state.detected_lang_at_abort = "en"

        guard.report_final_outcome(verification_passed=False, stats=stats)

        assert stats.total_checks == 1
        assert stats.total_aborts == 1
        assert stats.confirmed_aborts == 1
        assert stats.false_positives == 0
        assert stats.consecutive_fp == 0

    def test_report_final_outcome_false_positive(self) -> None:
        """False positive (abort but verification passed) updates FP counter."""
        guard = StreamGuard(expected_lang="de", enabled=True, backend=_stub_backend)
        stats = StreamGuardStats()

        guard._state.check_performed = True
        guard._state.aborted = True

        guard.report_final_outcome(verification_passed=True, stats=stats)

        assert stats.total_aborts == 1
        assert stats.false_positives == 1
        assert stats.consecutive_fp == 1

    def test_report_final_outcome_no_abort_no_stats_change(self) -> None:
        """No abort: total_checks incremented but no abort stats changed."""
        guard = StreamGuard(expected_lang="de", enabled=True, backend=_stub_backend)
        stats = StreamGuardStats()

        guard._state.check_performed = True
        guard._state.aborted = False

        guard.report_final_outcome(verification_passed=True, stats=stats)

        assert stats.total_checks == 1
        assert stats.total_aborts == 0

    def test_report_final_outcome_without_stats_is_noop(self) -> None:
        """Without stats object, report_final_outcome is a no-op."""
        guard = StreamGuard(expected_lang="de", enabled=True, backend=_stub_backend)
        guard._state.check_performed = True
        guard._state.aborted = True

        # Should not raise
        guard.report_final_outcome(verification_passed=True, stats=None)

    def test_auto_disable_after_consecutive_fp(self) -> None:
        """Guard auto-disables after 3 consecutive false positives."""
        stats = StreamGuardStats()

        for _ in range(3):
            guard = StreamGuard(expected_lang="de", enabled=True, backend=_stub_backend)
            guard._state.check_performed = True
            guard._state.aborted = True
            guard.report_final_outcome(verification_passed=True, stats=stats)

        assert stats.consecutive_fp == 3
        assert stats.should_disable is True


# ---------------------------------------------------------------------------
# Claude Fix 2: detection_tier computation
# ---------------------------------------------------------------------------


class TestDetectionTierComputation:
    """Claude Fix 2: detection_tier must use _get_detection_tier, not bucket."""

    def test_detection_tier_is_tier_not_bucket(self) -> None:
        """detection_tier should be a tier value, not a text_length_bucket."""
        from application.language.resolver import _detection_to_context

        # Build a detection with known bucket
        detection = OrchestratedDetection(
            code="de",
            confidence=0.93,
            distribution={"de": 0.93},
            reliability_score=0.88,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
        )

        ctx = _detection_to_context(
            detection=detection,
            request_id="test-123",
            source="detected",
        )

        # detection_tier should be "high" (German is HIGH tier), NOT "medium"
        assert ctx.detection_tier == "high"
        assert ctx.text_length_bucket == "medium"

    def test_detection_tier_none_when_no_bucket(self) -> None:
        """When text_length_bucket is None, detection_tier should be None."""
        from application.language.resolver import _detection_to_context

        detection = OrchestratedDetection(
            code="de",
            confidence=0.93,
            distribution={"de": 0.93},
            reliability_score=0.88,
            candidates=(),
            decision_reason="test",
            text_length_bucket="",  # empty string is falsy
        )

        ctx = _detection_to_context(
            detection=detection,
            request_id="test-123",
            source="detected",
        )

        # Empty bucket -> None tier
        assert ctx.detection_tier is None


# ---------------------------------------------------------------------------
# Claude Fix 3: Dead _check_time removed
# ---------------------------------------------------------------------------


class TestStreamGuardNoCheckTime:
    """Claude Fix 3: _check_time should not exist in StreamGuard."""

    def test_no_check_time_attribute(self) -> None:
        """StreamGuard must not have _check_time attribute."""
        guard = StreamGuard(expected_lang="de", enabled=True, backend=_stub_backend)
        assert not hasattr(guard, "_check_time")


# ---------------------------------------------------------------------------
# Claude Fix 1: Module-level registry singleton
# ---------------------------------------------------------------------------


class TestResolverRegistrySingleton:
    """Claude Fix 1: resolver.py must use module-level _registry."""

    def test_module_level_registry_exists(self) -> None:
        """resolver module must have _registry at module level."""
        from application.language import resolver

        assert hasattr(resolver, "_registry")
        assert isinstance(resolver._registry, InMemoryLanguageRegistry)


# ---------------------------------------------------------------------------
# MappingProxyType compatibility with existing patterns
# ---------------------------------------------------------------------------


class TestMappingProxyTypeCompatibility:
    """Ensure MappingProxyType works with all existing access patterns."""

    def test_items_iteration(self) -> None:
        """dict.items() pattern works on MappingProxyType."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9, "en": 0.1},
        )
        items = list(ctx.detection_distribution.items())
        assert ("de", 0.9) in items

    def test_len(self) -> None:
        """len() works on MappingProxyType."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9, "en": 0.1},
        )
        assert len(ctx.detection_distribution) == 2

    def test_bool_truthy(self) -> None:
        """Non-empty MappingProxyType is truthy."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9},
        )
        assert bool(ctx.detection_distribution) is True

    def test_bool_falsy(self) -> None:
        """Empty MappingProxyType is falsy."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
        )
        assert bool(ctx.detection_distribution) is False

    def test_dict_conversion(self) -> None:
        """dict(mapping_proxy) creates a mutable copy."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9},
        )
        mutable = dict(ctx.detection_distribution)
        mutable["en"] = 0.1  # This must work on the copy
        assert "en" not in ctx.detection_distribution  # Original unchanged

    def test_max_key_pattern(self) -> None:
        """max(dist, key=dist.get) pattern works with MappingProxyType."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9, "en": 0.1},
        )
        top = max(
            ctx.detection_distribution,
            key=ctx.detection_distribution.get,  # type: ignore[arg-type]
        )
        assert top == "de"

    def test_sorted_items_pattern(self) -> None:
        """sorted(dist.items(), key=...) pattern works."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9, "en": 0.1, "nl": 0.05},
        )
        sorted_langs = sorted(
            ctx.detection_distribution.items(),
            key=lambda x: (-x[1], x[0]),
        )
        assert sorted_langs[0] == ("de", 0.9)
