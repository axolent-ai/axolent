"""Layer 3: Evidence Ledger for Skill-Compression.

Collects structured evidence (support and contradiction) for each
hypothesis. Maintains BKT state per hypothesis and computes weighted
evidence summaries that feed into the Pattern Judge (Layer 4).

Evidence flows:
  Layer 1 (Event Normalizer) -> Layer 2 (Algorithmic Candidates)
    -> Layer 3 (Evidence Ledger) -> Layer 4 (Pattern Judge)

The ledger is the single source of truth for hypothesis evidence.
It wraps HypothesisStorage for DB operations and adds:
  1. Typed EvidenceRecord dataclass
  2. BKT-integrated evidence summaries
  3. Session tracking for multi-session requirements
  4. Signal classification (positive vs negative)

HC-LAYER2-1: Evidence Ledger receives signals from Layer 2 algorithms
but does NOT import them directly. It consumes structured EvidenceRecords.

AG: Pattern Judge imports ONLY via EvidenceLedger + Hypothesis,
never directly from N-Gram/Markov/Elo modules.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from application.skill_compression.bkt import (
    BKTState,
    create_initial_state,
    update_bkt_weighted,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------

# Positive signals increase hypothesis confidence
POSITIVE_SIGNALS: frozenset[str] = frozenset(
    {
        "no_correction",
        "bookmark",
        "explicit_confirm",
        "learn_command",
    }
)

# Negative signals decrease hypothesis confidence
NEGATIVE_SIGNALS: frozenset[str] = frozenset(
    {
        "correction",
        "rejection",
    }
)

# All valid signal types
VALID_SIGNAL_TYPES: frozenset[str] = POSITIVE_SIGNALS | NEGATIVE_SIGNALS

# Signal type literal for type checking
SignalType = Literal[
    "no_correction",
    "bookmark",
    "explicit_confirm",
    "correction",
    "rejection",
    "learn_command",
]


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """A single piece of evidence for or against a hypothesis.

    Attributes:
        evidence_id: Unique identifier for this evidence record.
        hypothesis_id: The hypothesis this evidence relates to.
        hypothesis_version: Version of the hypothesis at evidence time.
        episode_id: Conversation episode ID (for session tracking).
        request_id: Original request ID (optional).
        response_id: Bot response ID (optional).
        signal_type: Type of signal observed.
        signal_strength: Strength of the signal [0.0, 1.0].
        created_at: ISO-8601 UTC timestamp.
    """

    evidence_id: str
    hypothesis_id: str
    hypothesis_version: int
    episode_id: str
    request_id: Optional[str]
    response_id: Optional[str]
    signal_type: SignalType
    signal_strength: float
    created_at: str


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    """Aggregated evidence summary for a hypothesis.

    Combines counts, weighted scores, BKT state, and session info.

    Attributes:
        positive_count: Number of positive evidence records.
        negative_count: Number of negative evidence records.
        total_count: Total evidence records.
        weighted_score: BKT-derived confidence (p_knowledge from BKT state).
        bkt_state: Current BKT state after processing all evidence.
        distinct_sessions: Number of distinct episode_ids.
        last_positive_at: Timestamp of last positive evidence.
        last_negative_at: Timestamp of last negative evidence.
    """

    positive_count: int
    negative_count: int
    total_count: int
    weighted_score: float
    bkt_state: BKTState
    distinct_sessions: int
    last_positive_at: Optional[str]
    last_negative_at: Optional[str]


# ---------------------------------------------------------------
# Evidence Ledger
# ---------------------------------------------------------------


def is_positive_signal(signal_type: str) -> bool:
    """Check whether a signal type is positive (supports hypothesis).

    Args:
        signal_type: The signal type string.

    Returns:
        True if positive, False if negative.

    Raises:
        ValueError: If signal_type is not recognized.
    """
    if signal_type in POSITIVE_SIGNALS:
        return True
    if signal_type in NEGATIVE_SIGNALS:
        return False
    msg = f"Unknown signal type: {signal_type!r}. Valid types: {sorted(VALID_SIGNAL_TYPES)}"
    raise ValueError(msg)


class EvidenceLedger:
    """Layer 3: Evidence collection and BKT-based summarization.

    Maintains an in-memory list of EvidenceRecords per hypothesis
    and computes BKT-integrated summaries on demand.

    Thread safety: NOT thread-safe. Designed for single-threaded
    async event loop (Telegram bot context).

    Usage:
        ledger = EvidenceLedger()
        ledger.add_evidence(record)
        summary = ledger.get_summary("hyp_123")
    """

    def __init__(self) -> None:
        """Initialize an empty evidence ledger."""
        # hypothesis_id -> ordered list of evidence records
        self._evidence: dict[str, list[EvidenceRecord]] = {}
        # hypothesis_id -> current BKT state
        self._bkt_states: dict[str, BKTState] = {}

    def add_evidence(self, record: EvidenceRecord) -> None:
        """Add an evidence record and update BKT state.

        The BKT state is updated incrementally with each new record.
        Signal strength modulates the BKT update via weighted update.

        Args:
            record: The evidence record to add.

        Raises:
            ValueError: If signal_type is not valid.
        """
        # Validate signal type
        if record.signal_type not in VALID_SIGNAL_TYPES:
            msg = (
                f"Invalid signal_type: {record.signal_type!r}. "
                f"Valid: {sorted(VALID_SIGNAL_TYPES)}"
            )
            raise ValueError(msg)

        # Validate signal strength
        if not 0.0 <= record.signal_strength <= 1.0:
            log.warning(
                "Signal strength %.3f out of [0, 1] range for evidence %s, clamping",
                record.signal_strength,
                record.evidence_id,
            )

        # Store evidence
        if record.hypothesis_id not in self._evidence:
            self._evidence[record.hypothesis_id] = []
        self._evidence[record.hypothesis_id].append(record)

        # Update BKT state
        positive = is_positive_signal(record.signal_type)
        current_bkt = self._bkt_states.get(
            record.hypothesis_id,
            create_initial_state(),
        )
        new_bkt = update_bkt_weighted(
            current_bkt,
            positive,
            weight=max(0.0, min(1.0, record.signal_strength)),
        )
        self._bkt_states[record.hypothesis_id] = new_bkt

        log.debug(
            "Evidence added: id=%s hyp=%s type=%s strength=%.2f bkt=%.4f->%.4f",
            record.evidence_id,
            record.hypothesis_id,
            record.signal_type,
            record.signal_strength,
            current_bkt.p_knowledge,
            new_bkt.p_knowledge,
        )

    def get_evidence(self, hypothesis_id: str) -> list[EvidenceRecord]:
        """Retrieve all evidence records for a hypothesis, ordered by time.

        Args:
            hypothesis_id: The hypothesis to query.

        Returns:
            Chronologically ordered list of EvidenceRecords.
            Empty list if no evidence exists.
        """
        return list(self._evidence.get(hypothesis_id, []))

    def get_bkt_state(self, hypothesis_id: str) -> BKTState:
        """Get the current BKT state for a hypothesis.

        Args:
            hypothesis_id: The hypothesis to query.

        Returns:
            Current BKTState, or initial state if no evidence.
        """
        return self._bkt_states.get(hypothesis_id, create_initial_state())

    def get_summary(self, hypothesis_id: str) -> EvidenceSummary:
        """Compute an aggregated evidence summary for a hypothesis.

        Counts positive/negative evidence, computes the BKT-derived
        weighted score, counts distinct sessions, and finds the
        latest timestamps for each polarity.

        Args:
            hypothesis_id: The hypothesis to summarize.

        Returns:
            EvidenceSummary with all aggregate metrics.
        """
        records = self._evidence.get(hypothesis_id, [])
        bkt_state = self._bkt_states.get(hypothesis_id, create_initial_state())

        positive_count = 0
        negative_count = 0
        sessions: set[str] = set()
        last_positive_at: Optional[str] = None
        last_negative_at: Optional[str] = None

        for r in records:
            if is_positive_signal(r.signal_type):
                positive_count += 1
                last_positive_at = r.created_at
            else:
                negative_count += 1
                last_negative_at = r.created_at

            if r.episode_id:
                sessions.add(r.episode_id)

        return EvidenceSummary(
            positive_count=positive_count,
            negative_count=negative_count,
            total_count=len(records),
            weighted_score=bkt_state.p_knowledge,
            bkt_state=bkt_state,
            distinct_sessions=len(sessions),
            last_positive_at=last_positive_at,
            last_negative_at=last_negative_at,
        )

    def get_recent_contradictions(
        self,
        hypothesis_id: str,
        limit: int = 10,
    ) -> int:
        """Count negative evidence in the most recent N records.

        Used by Pattern Judge for needs_review detection: if the last
        few observations are mostly negative, the hypothesis may need
        review.

        Args:
            hypothesis_id: The hypothesis to check.
            limit: How many recent records to examine.

        Returns:
            Number of negative signals in the last `limit` records.
        """
        records = self._evidence.get(hypothesis_id, [])
        recent = records[-limit:] if len(records) > limit else records
        return sum(1 for r in recent if r.signal_type in NEGATIVE_SIGNALS)

    def has_evidence(self, hypothesis_id: str) -> bool:
        """Check whether any evidence exists for a hypothesis.

        Args:
            hypothesis_id: The hypothesis to check.

        Returns:
            True if at least one evidence record exists.
        """
        return bool(self._evidence.get(hypothesis_id))

    def clear_hypothesis(self, hypothesis_id: str) -> None:
        """Remove all evidence and BKT state for a hypothesis.

        Used when a hypothesis is retired or tombstoned.

        Args:
            hypothesis_id: The hypothesis to clear.
        """
        self._evidence.pop(hypothesis_id, None)
        self._bkt_states.pop(hypothesis_id, None)
        log.info("Evidence cleared for hypothesis %s", hypothesis_id)
