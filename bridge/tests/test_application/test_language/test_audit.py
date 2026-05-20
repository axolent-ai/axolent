"""Tests for Detection Audit Event (B-2 Add-on 3/3).

Covers:
1. Unit Tests:
   - build_audit_event with complete detection + context: all fields populated
   - build_audit_event without detection (override case): detection-derived defaults
   - Frozen invariant: assignment raises FrozenInstanceError / AttributeError
   - Privacy: input text NEVER appears in the event (HC-D1)
   - JSON serialisability: json.dumps(asdict(event)) succeeds (HC-D5)
   - text_length_bucket correctly carried through
   - detection_distribution truncated to top-5 when N > 5
   - min_chars_met correctly passed through

2. Logger Tests:
   - DetectionAuditLogger.log() writes structured JSON
   - Custom logger_name is respected
   - Logger swallows serialisation errors (HC-C7)

3. Integration Tests:
   - Resolver with AuditLogger: log() is called
   - Resolver without AuditLogger: log() is NOT called (no crash)
   - Override source: Event has source="override" and empty candidates

Test naming: test_<subject>_<scenario>_<expected>.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from application.language.audit import (
    DetectionAuditEvent,
    DetectionAuditLogger,
    build_audit_event,
)
from application.language.context import LanguageContext
from application.language.orchestrator import (
    DetectionCandidate,
    OrchestratedDetection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_detection(
    code: str = "de",
    confidence: float = 0.93,
    distribution: dict[str, float] | None = None,
    reliability_score: float = 0.88,
    decision_reason: str = "Primary backend (langdetect) detected 'de' with confidence 0.93.",
    text_length_bucket: str = "medium",
    min_chars_met: bool = True,
    candidates: tuple[DetectionCandidate, ...] | None = None,
) -> OrchestratedDetection:
    """Build a typical OrchestratedDetection for testing."""
    if distribution is None:
        distribution = {"de": 0.93, "nl": 0.05, "en": 0.02}
    if candidates is None:
        candidates = (
            DetectionCandidate(
                backend_name="langdetect",
                distribution={"de": 0.93, "nl": 0.05, "en": 0.02},
                top_lang="de",
                top_confidence=0.93,
                latency_ms=12.5,
            ),
            DetectionCandidate(
                backend_name="domain_heuristic",
                distribution={"de": 0.85},
                top_lang="de",
                top_confidence=0.85,
                latency_ms=0.3,
            ),
        )
    return OrchestratedDetection(
        code=code,
        confidence=confidence,
        distribution=distribution,
        reliability_score=reliability_score,
        candidates=candidates,
        decision_reason=decision_reason,
        text_length_bucket=text_length_bucket,
        min_chars_met=min_chars_met,
    )


_SENTINEL_DIST: dict[str, float] | None = None


def _make_context(
    code: str = "de",
    source: str = "detected",
    confidence: float = 0.93,
    switched_from: str | None = None,
    request_id: str = "abc123def456",
    distribution: dict[str, float] | None = _SENTINEL_DIST,
    reliability_score: float = 0.88,
    text_length_bucket: str | None = "medium",
    backends_consulted: frozenset[str] | None = None,
) -> LanguageContext:
    """Build a LanguageContext for testing."""
    if distribution is _SENTINEL_DIST:
        distribution = {"de": 0.93, "nl": 0.05, "en": 0.02}
    return LanguageContext(
        code=code,
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        switched_from=switched_from,
        request_id=request_id,
        detection_distribution=distribution if distribution is not None else {},
        reliability_score=reliability_score,
        confidence_history=(("langdetect", 0.93), ("domain_heuristic", 0.85)),
        detection_tier="high",
        text_length_bucket=text_length_bucket,
        backends_consulted=backends_consulted
        if backends_consulted is not None
        else frozenset({"langdetect", "domain_heuristic"}),
    )


# ---------------------------------------------------------------------------
# 1. Unit Tests: build_audit_event
# ---------------------------------------------------------------------------


class TestBuildAuditEventWithDetection:
    """build_audit_event with complete detection + context."""

    def test_all_fields_populated(self) -> None:
        """All fields are populated from context + detection."""
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(
            ctx, det, request_id="abc123def456", user_id=42, input_text_length=120
        )

        assert event.request_id == "abc123def456"
        assert event.user_id == 42
        assert event.input_text_length == 120
        assert event.detected_code == "de"
        assert event.confidence == 0.93
        assert event.reliability_score == 0.88
        assert event.source == "detected"
        assert event.text_length_bucket == "medium"
        assert event.min_chars_met is True
        assert event.switched_from is None
        assert len(event.candidates) == 2
        assert len(event.backends_consulted) == 2
        assert event.timestamp  # Non-empty ISO string

    def test_decision_reason_1_to_1_from_detection(self) -> None:
        """HC-D6: decision_reason taken 1:1 from OrchestratedDetection."""
        reason = "Custom reason text for test."
        ctx = _make_context()
        det = _make_detection(decision_reason=reason)
        event = build_audit_event(ctx, det, request_id="r1")

        assert event.decision_reason == reason

    def test_candidates_structure(self) -> None:
        """IC-D2: candidates have correct keys."""
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        for cand in event.candidates:
            assert "backend_name" in cand
            assert "code" in cand
            assert "confidence" in cand
            assert "latency_ms" in cand

    def test_top_alternative_from_context(self) -> None:
        """top_alternative is derived from context.top_alternative."""
        ctx = _make_context(distribution={"de": 0.80, "nl": 0.15, "en": 0.05})
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        assert event.top_alternative == "nl"

    def test_switched_from_populated(self) -> None:
        """switched_from is taken from context when smart-switch happened."""
        ctx = _make_context(switched_from="en")
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        assert event.switched_from == "en"

    def test_min_chars_met_false(self) -> None:
        """min_chars_met=False passed through from detection."""
        ctx = _make_context()
        det = _make_detection(min_chars_met=False)
        event = build_audit_event(ctx, det, request_id="r1")

        assert event.min_chars_met is False

    def test_backends_consulted_from_detection(self) -> None:
        """backends_consulted derived from detection.candidates names."""
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        assert "langdetect" in event.backends_consulted
        assert "domain_heuristic" in event.backends_consulted


class TestBuildAuditEventWithoutDetection:
    """build_audit_event without detection (override/sticky case)."""

    def test_override_case_defaults(self) -> None:
        """Override: candidates empty, decision_reason mentions source."""
        ctx = _make_context(
            source="override",
            confidence=1.0,
            distribution={},
            reliability_score=0.0,
            backends_consulted=frozenset(),
            text_length_bucket=None,
        )
        event = build_audit_event(ctx, detection=None, request_id="r_ov")

        assert event.source == "override"
        assert event.candidates == ()
        assert "override" in event.decision_reason
        assert event.min_chars_met is True
        assert event.confidence == 1.0

    def test_sticky_case_defaults(self) -> None:
        """Sticky without detection: safe defaults."""
        ctx = _make_context(
            source="sticky",
            confidence=1.0,
            distribution={},
            reliability_score=0.0,
            backends_consulted=frozenset(),
        )
        event = build_audit_event(ctx, detection=None, request_id="r_st")

        assert event.source == "sticky"
        assert event.candidates == ()
        assert event.backends_consulted == ()

    def test_user_id_defaults_to_zero(self) -> None:
        """user_id defaults to 0 when not provided."""
        ctx = _make_context(source="override", distribution={})
        event = build_audit_event(ctx, detection=None, request_id="r0")

        assert event.user_id == 0


# ---------------------------------------------------------------------------
# 2. Dataclass invariants
# ---------------------------------------------------------------------------


class TestFrozenInvariant:
    """HC-D2: DetectionAuditEvent must be frozen with slots."""

    def test_frozen_assignment_raises(self) -> None:
        """Assignment to a field of a frozen event raises."""
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        with pytest.raises(dataclasses.FrozenInstanceError):
            event.detected_code = "en"  # type: ignore[misc]

    def test_slots_no_dict(self) -> None:
        """Slots-based dataclass has no __dict__."""
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        assert not hasattr(event, "__dict__")


# ---------------------------------------------------------------------------
# 3. Privacy (HC-D1)
# ---------------------------------------------------------------------------


class TestPrivacy:
    """HC-D1: Input text NEVER appears in the audit event."""

    def test_no_text_field_on_event(self) -> None:
        """DetectionAuditEvent has no 'text' or 'input_text' field."""
        field_names = set(dataclasses.fields(DetectionAuditEvent))
        field_name_strs = {f.name for f in field_names}
        assert "text" not in field_name_strs
        assert "input_text" not in field_name_strs

    def test_text_not_in_serialised_event(self) -> None:
        """The actual input text does not appear in json.dumps output."""
        secret_text = "This_is_the_secret_user_input_message_42!"
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(
            ctx,
            det,
            request_id="r1",
            input_text_length=len(secret_text),
        )
        serialised = json.dumps(asdict(event))

        assert secret_text not in serialised
        assert str(len(secret_text)) in serialised  # Length IS present


# ---------------------------------------------------------------------------
# 4. JSON serialisability (HC-D5)
# ---------------------------------------------------------------------------


class TestJsonSerialisability:
    """HC-D5: json.dumps(asdict(event)) must succeed."""

    def test_full_event_serialisable(self) -> None:
        """Complete event with all fields serialises to valid JSON."""
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(
            ctx, det, request_id="r1", user_id=42, input_text_length=120
        )

        json_str = json.dumps(asdict(event), ensure_ascii=False)
        parsed = json.loads(json_str)

        assert parsed["detected_code"] == "de"
        assert parsed["user_id"] == 42
        assert parsed["input_text_length"] == 120

    def test_override_event_serialisable(self) -> None:
        """Override event (no detection) serialises to valid JSON."""
        ctx = _make_context(source="override", distribution={})
        event = build_audit_event(ctx, detection=None, request_id="r_ov")

        json_str = json.dumps(asdict(event))
        parsed = json.loads(json_str)

        assert parsed["source"] == "override"
        assert parsed["candidates"] == []

    def test_no_frozenset_in_serialised(self) -> None:
        """No FrozenSet or tuple-of-tuples breaks serialisation.

        backends_consulted is stored as tuple[str, ...] (not FrozenSet).
        """
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        # This would fail if backends_consulted were a frozenset
        json_str = json.dumps(asdict(event))
        assert json_str  # Non-empty


# ---------------------------------------------------------------------------
# 5. Text length bucket
# ---------------------------------------------------------------------------


class TestTextLengthBucket:
    """text_length_bucket correctly carried through."""

    @pytest.mark.parametrize("bucket", ["micro", "short", "medium", "long"])
    def test_bucket_from_detection(self, bucket: str) -> None:
        """Bucket from detection is preserved in event."""
        ctx = _make_context(text_length_bucket=bucket)
        det = _make_detection(text_length_bucket=bucket)
        event = build_audit_event(ctx, det, request_id="r1")

        assert event.text_length_bucket == bucket


# ---------------------------------------------------------------------------
# 6. Distribution truncation (IC-D1)
# ---------------------------------------------------------------------------


class TestDistributionTruncation:
    """detection_distribution truncated to top-5 when N > 5."""

    def test_large_distribution_truncated(self) -> None:
        """Distribution with 8 entries: candidates get top-5 per backend."""
        big_dist = {
            "de": 0.30,
            "nl": 0.20,
            "en": 0.15,
            "fr": 0.12,
            "es": 0.10,
            "it": 0.07,
            "pt": 0.04,
            "pl": 0.02,
        }
        det = _make_detection(
            distribution=big_dist,
            candidates=(
                DetectionCandidate(
                    backend_name="langdetect",
                    distribution=big_dist,
                    top_lang="de",
                    top_confidence=0.30,
                    latency_ms=10.0,
                ),
            ),
        )
        ctx = _make_context(distribution=big_dist)
        event = build_audit_event(ctx, det, request_id="r1")

        # candidates holds per-backend info (IC-D2), not truncated distribution.
        # Distribution truncation is tested via _truncate_distribution directly.
        assert len(event.candidates) == 1

    def test_truncate_distribution_caps_at_5(self) -> None:
        """Internal helper: _truncate_distribution returns at most 5."""
        from application.language.audit import _truncate_distribution

        dist = {f"lang{i}": 0.1 for i in range(10)}
        truncated = _truncate_distribution(dist, top_n=5)

        assert len(truncated) == 5

    def test_truncate_distribution_rounds(self) -> None:
        """Internal helper: probabilities rounded to 3 decimal places."""
        from application.language.audit import _truncate_distribution

        dist = {"de": 0.123456789}
        truncated = _truncate_distribution(dist)

        assert truncated[0]["probability"] == 0.123


# ---------------------------------------------------------------------------
# 7. Logger Tests
# ---------------------------------------------------------------------------


class TestDetectionAuditLogger:
    """DetectionAuditLogger structured logging."""

    def test_log_writes_json_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """log() writes a JSON string at INFO level."""
        logger = DetectionAuditLogger()
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        with caplog.at_level(logging.INFO, logger="axolent.language.audit"):
            logger.log(event)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        # Message should be valid JSON
        parsed = json.loads(record.getMessage())
        assert parsed["detected_code"] == "de"

    def test_log_extra_contains_audit_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log() passes extra={'audit_event': ...}."""
        logger = DetectionAuditLogger()
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        with caplog.at_level(logging.INFO, logger="axolent.language.audit"):
            logger.log(event)

        record = caplog.records[0]
        assert hasattr(record, "audit_event")
        assert record.audit_event["detected_code"] == "de"  # type: ignore[attr-defined]

    def test_custom_logger_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Custom logger_name is respected."""
        custom_name = "my.custom.audit"
        logger = DetectionAuditLogger(logger_name=custom_name)
        ctx = _make_context()
        det = _make_detection()
        event = build_audit_event(ctx, det, request_id="r1")

        with caplog.at_level(logging.INFO, logger=custom_name):
            logger.log(event)

        assert len(caplog.records) == 1
        assert caplog.records[0].name == custom_name

    def test_log_swallows_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """HC-C7: Logger swallows errors without raising."""
        logger = DetectionAuditLogger()

        # Create a mock event that raises on asdict
        bad_event = MagicMock(spec=DetectionAuditEvent)

        with (
            patch("application.language.audit.asdict", side_effect=TypeError("boom")),
            caplog.at_level(logging.WARNING, logger="axolent.language.audit"),
        ):
            # Should NOT raise
            logger.log(bad_event)

        # Should have logged a warning about the failure
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# 8. Integration Tests: Resolver + AuditLogger
# ---------------------------------------------------------------------------


class StubBackend:
    """Minimal stub implementing LanguageDetectorBackend protocol."""

    def __init__(self, distribution: dict[str, float] | None = None) -> None:
        self._distribution = distribution or {"de": 0.93, "nl": 0.05, "en": 0.02}

    def detect_distribution(self, text: str) -> dict[str, float]:
        return dict(self._distribution)


class TestResolverAuditIntegration:
    """Resolver with/without AuditLogger."""

    @pytest.fixture(autouse=True)
    def _clear_storage(self) -> None:
        """Reset conversation storage before each test."""
        from infrastructure.conversation_storage import _reset_all_for_tests

        _reset_all_for_tests()

    async def test_resolver_with_audit_logger_calls_log(self) -> None:
        """Resolver with AuditLogger: log() is called once per resolve."""
        from application.language.orchestrator import DetectionOrchestrator
        from application.language.registry import InMemoryLanguageRegistry
        from application.language.resolver import LanguageResolver

        mock_logger = MagicMock(spec=DetectionAuditLogger)
        orch = DetectionOrchestrator(
            primary_backend=StubBackend(),
            fallback_backend=None,
            registry=InMemoryLanguageRegistry(),
        )
        resolver = LanguageResolver(orchestrator=orch, audit_logger=mock_logger)

        await resolver.resolve(
            user_id=1, chat_id=1, text="Hallo Welt wie geht es dir heute"
        )

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert isinstance(event, DetectionAuditEvent)

    async def test_resolver_without_audit_logger_no_crash(self) -> None:
        """Resolver without AuditLogger: no error, no audit call."""
        from application.language.orchestrator import DetectionOrchestrator
        from application.language.registry import InMemoryLanguageRegistry
        from application.language.resolver import LanguageResolver

        orch = DetectionOrchestrator(
            primary_backend=StubBackend(),
            fallback_backend=None,
            registry=InMemoryLanguageRegistry(),
        )
        resolver = LanguageResolver(orchestrator=orch)  # No audit_logger

        ctx = await resolver.resolve(
            user_id=1, chat_id=1, text="Test text for resolver"
        )
        assert ctx.code  # Just verify it works at all

    async def test_resolver_override_source_audit_event(self) -> None:
        """Override: event has source='override' and empty candidates."""
        from application.language.resolver import LanguageResolver

        mock_logger = MagicMock(spec=DetectionAuditLogger)
        resolver = LanguageResolver(audit_logger=mock_logger)

        await resolver.resolve(user_id=1, chat_id=1, text="anything", override="fr")

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event.source == "override"
        assert event.detected_code == "fr"
        assert event.candidates == ()

    async def test_resolve_readonly_with_audit_logger(self) -> None:
        """resolve_readonly with AuditLogger also emits audit event."""
        from application.language.orchestrator import DetectionOrchestrator
        from application.language.registry import InMemoryLanguageRegistry
        from application.language.resolver import LanguageResolver

        mock_logger = MagicMock(spec=DetectionAuditLogger)
        orch = DetectionOrchestrator(
            primary_backend=StubBackend(),
            fallback_backend=None,
            registry=InMemoryLanguageRegistry(),
        )
        resolver = LanguageResolver(orchestrator=orch, audit_logger=mock_logger)

        await resolver.resolve_readonly(
            user_id=1, chat_id=1, text="Hallo Welt test text"
        )

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert isinstance(event, DetectionAuditEvent)
