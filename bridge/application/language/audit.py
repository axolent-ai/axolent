"""Detection Audit Event: structured logging for language detection decisions.

This module is the first real consumer of the Phase 2 fields in
LanguageContext. It produces a frozen, JSON-serialisable audit event
after every language resolution, capturing full detection provenance
without exposing the user's input text (privacy).

Architecture:
- DetectionAuditEvent: frozen dataclass with all detection metadata.
- build_audit_event(): factory that maps LanguageContext +
  OrchestratedDetection into an event.
- DetectionAuditLogger: thin wrapper around Python stdlib logging,
  emitting events as structured JSON on a dedicated logger.

Hard Constraints enforced here:
- HC-D1 [BLOCKER]: Input text is NEVER stored. Only input_text_length.
- HC-D2 [BLOCKER]: frozen=True, slots=True on DetectionAuditEvent.
- HC-D3 [BLOCKER]: AuditLogger is optional (resolver injects or not).
- HC-D4 [BLOCKER]: All Phase 2 fields consumed (detection_distribution
  as top-5 candidates, reliability_score, confidence_history,
  backends_consulted, text_length_bucket, top_alternative, min_chars_met).
- HC-D5 [BLOCKER]: json.dumps(asdict(event)) must succeed.
- HC-D6 [BLOCKER]: decision_reason taken 1:1 from OrchestratedDetection.
- HC-D7 [NICE]: Logger name 'axolent.language.audit' as default.

Implementation choices:
- IC-D1: detection_distribution serialised as top-5 entries, rounded
  to 3 decimal places.
- IC-D2: candidates as list[dict] with keys: backend_name, code,
  confidence, latency_ms.
- IC-D3: Timestamp via datetime.now(UTC).isoformat().
- IC-D4: Log level INFO for all audit events.
- IC-D5: No audit hash (deferred).
- IC-D6: When detection is None (override/sticky), detection-derived
  fields get safe defaults: empty candidates, confidence from context,
  reliability_score from context, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional

from application.language.context import LanguageContext
from application.language.orchestrator import OrchestratedDetection

# Maximum number of entries kept from detection_distribution (IC-D1).
_TOP_N_DISTRIBUTION = 5

# Decimal precision for distribution probabilities (IC-D1).
_DISTRIBUTION_PRECISION = 3


# ---------------------------------------------------------------------------
# Data model (HC-D2: frozen=True, slots=True)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectionAuditEvent:
    """Structured audit event for a language detection decision.

    Used for debugging, AGPL transparency, and post-hoc analysis of
    language detection quality. Privacy-safe: contains text length
    but never the text itself (HC-D1).

    Attributes:
        timestamp: ISO-8601 UTC timestamp of the decision.
        request_id: Correlation ID from the LanguageContext.
        user_id: Telegram user ID (0 if not applicable).
        input_text_length: Length of the input text in characters.
        text_length_bucket: Classification (micro/short/medium/long).
        detected_code: Final language decision.
        confidence: Detection confidence (0.0..1.0).
        reliability_score: Composite reliability metric (0.0..1.0).
        source: How the language was determined
            (override/sticky/detected/default).
        backends_consulted: Names of backends invoked during detection.
        candidates: Per-backend results as serialisable dicts.
        decision_reason: Human-readable explanation from the
            OrchestratedDetection (HC-D6: taken 1:1, never invented).
        switched_from: Previous language if smart-switch occurred.
        top_alternative: Second-most-likely language from distribution.
        min_chars_met: Whether the min_chars_reliable guard passed.
    """

    # WHO / WHEN
    timestamp: str
    request_id: str
    user_id: int

    # INPUT (HC-D1: text length only, NEVER the text itself)
    input_text_length: int
    text_length_bucket: Optional[str]

    # RESULT
    detected_code: str
    confidence: float
    reliability_score: float
    source: str

    # PROVENANCE
    backends_consulted: tuple[str, ...]
    candidates: tuple[dict[str, object], ...]
    decision_reason: str

    # SWITCHING
    switched_from: Optional[str]
    top_alternative: Optional[str]

    # GUARD (Add-on 1)
    min_chars_met: bool


# ---------------------------------------------------------------------------
# Builder (HC-D4, HC-D5, HC-D6)
# ---------------------------------------------------------------------------


def _truncate_distribution(
    distribution: Mapping[str, float],
    top_n: int = _TOP_N_DISTRIBUTION,
    precision: int = _DISTRIBUTION_PRECISION,
) -> list[dict[str, object]]:
    """Convert a detection distribution to a top-N list of dicts.

    Sorted by probability descending, rounded to *precision* decimals.
    Each entry: {"code": str, "probability": float}.

    IC-D1: top-5, rounded to 0.001.
    """
    sorted_entries = sorted(
        distribution.items(),
        key=lambda x: (-x[1], x[0]),
    )[:top_n]
    return [
        {"code": code, "probability": round(prob, precision)}
        for code, prob in sorted_entries
    ]


def _build_candidates_from_detection(
    detection: OrchestratedDetection,
) -> tuple[dict[str, object], ...]:
    """Extract per-backend candidate info as serialisable dicts (IC-D2).

    Keys: backend_name, code, confidence, latency_ms.
    """
    result: list[dict[str, object]] = []
    for c in detection.candidates:
        result.append(
            {
                "backend_name": c.backend_name,
                "code": c.top_lang,
                "confidence": round(c.top_confidence, 4),
                "latency_ms": round(c.latency_ms, 2),
            }
        )
    return tuple(result)


def build_audit_event(
    context: LanguageContext,
    detection: OrchestratedDetection | None,
    request_id: str,
    user_id: int = 0,
    input_text_length: int = 0,
) -> DetectionAuditEvent:
    """Build a DetectionAuditEvent from LanguageContext + OrchestratedDetection.

    If *detection* is None (e.g. source was 'override' or 'sticky' with
    no fresh detection), fills detection-derived fields with safe defaults.

    Args:
        context: The resolved LanguageContext (always available).
        detection: The OrchestratedDetection (None for override/sticky
            without detection).
        request_id: Correlation ID (typically from context.request_id).
        user_id: Telegram user ID. 0 when not applicable.
        input_text_length: Character count of the input text. The text
            itself is NEVER passed into this function (HC-D1 enforcement
            at call-site level).

    Returns:
        A frozen DetectionAuditEvent ready for logging.
    """
    # IC-D3: UTC ISO-8601 timestamp.
    timestamp = datetime.now(timezone.utc).isoformat()

    # Detection-derived fields with defaults for override/sticky.
    if detection is not None:
        candidates = _build_candidates_from_detection(detection)
        decision_reason = detection.decision_reason  # HC-D6: 1:1
        min_chars_met = detection.min_chars_met
        text_length_bucket = detection.text_length_bucket
        backends_consulted = tuple(c.backend_name for c in detection.candidates)
    else:
        candidates = ()
        decision_reason = f"No detection performed (source={context.source})."
        min_chars_met = True
        text_length_bucket = context.text_length_bucket
        backends_consulted = tuple(sorted(context.backends_consulted))

    return DetectionAuditEvent(
        timestamp=timestamp,
        request_id=request_id,
        user_id=user_id,
        input_text_length=input_text_length,
        text_length_bucket=text_length_bucket,
        detected_code=context.code,
        confidence=context.confidence,
        reliability_score=context.reliability_score,
        source=context.source,
        backends_consulted=backends_consulted,
        candidates=candidates,
        decision_reason=decision_reason,
        switched_from=context.switched_from,
        top_alternative=context.top_alternative,
        min_chars_met=min_chars_met,
    )


# ---------------------------------------------------------------------------
# Logger (HC-D3: optional, HC-D7: dedicated logger name)
# ---------------------------------------------------------------------------


class DetectionAuditLogger:
    """Logs DetectionAuditEvent instances via structured JSON.

    Uses a dedicated Python logger (default: 'axolent.language.audit')
    so that audit events can be routed independently from application
    logs (e.g. to a file, to Loki, to ELK).

    HC-D3: This logger is optional. The resolver only calls log() when
    an instance is provided.

    HC-D7: Logger name defaults to 'axolent.language.audit', but can
    be overridden at construction time.
    """

    def __init__(self, logger_name: str = "axolent.language.audit") -> None:
        """Initialize with a dedicated logger.

        Args:
            logger_name: Python logger name. Defaults to
                'axolent.language.audit'.
        """
        self._log = logging.getLogger(logger_name)

    def log(self, event: DetectionAuditEvent) -> None:
        """Log the event as structured JSON at INFO level.

        HC-C7: If logging fails (I/O error, serialisation error), the
        error is swallowed with a warning. The language decision must
        never be delayed or aborted because of audit logging.

        Uses extra={'audit_event': ...} so structured log handlers
        (e.g. python-json-logger) can pick up the event payload.
        """
        try:
            event_dict = asdict(event)
            json_str = json.dumps(event_dict, ensure_ascii=False)
            # IC-D4: INFO level for all audit events.
            self._log.info(
                json_str,
                extra={"audit_event": event_dict},
            )
        except Exception as exc:
            # HC-C7: Never crash the request path.
            self._log.warning(
                "Failed to log audit event: %s: %s",
                type(exc).__name__,
                exc,
            )
