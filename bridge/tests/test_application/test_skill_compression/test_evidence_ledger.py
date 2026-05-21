"""Tests for Layer 3: Evidence Ledger.

Covers:
  - add_evidence + get_evidence round-trip
  - get_summary with BKT updates
  - Multiple evidence types with correct weighted_score
  - signal_strength respected in BKT updates
  - Chronological ordering stability
  - Session counting
  - Recent contradictions
  - Invalid signal type handling
"""

from __future__ import annotations

import pytest

from application.skill_compression.evidence_ledger import (
    NEGATIVE_SIGNALS,
    POSITIVE_SIGNALS,
    EvidenceLedger,
    EvidenceRecord,
    is_positive_signal,
)


def _make_evidence(
    hypothesis_id: str = "hyp_001",
    signal_type: str = "no_correction",
    signal_strength: float = 1.0,
    episode_id: str = "ep_001",
    evidence_id: str = "",
    created_at: str = "2026-05-20T10:00:00+00:00",
) -> EvidenceRecord:
    """Factory helper for creating EvidenceRecords."""
    if not evidence_id:
        evidence_id = f"ev_{signal_type}_{created_at[-8:-6]}"
    return EvidenceRecord(
        evidence_id=evidence_id,
        hypothesis_id=hypothesis_id,
        hypothesis_version=1,
        episode_id=episode_id,
        request_id="req_001",
        response_id="resp_001",
        signal_type=signal_type,  # type: ignore[arg-type]
        signal_strength=signal_strength,
        created_at=created_at,
    )


class TestEvidenceRoundTrip:
    """Tests for add_evidence + get_evidence round-trip."""

    def test_add_and_retrieve_single(self) -> None:
        """Single evidence record should be retrievable."""
        ledger = EvidenceLedger()
        record = _make_evidence()
        ledger.add_evidence(record)
        result = ledger.get_evidence("hyp_001")
        assert len(result) == 1
        assert result[0].evidence_id == record.evidence_id
        assert result[0].signal_type == "no_correction"

    def test_add_multiple_same_hypothesis(self) -> None:
        """Multiple records for same hypothesis should all be retrievable."""
        ledger = EvidenceLedger()
        for i in range(5):
            ledger.add_evidence(
                _make_evidence(
                    evidence_id=f"ev_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        result = ledger.get_evidence("hyp_001")
        assert len(result) == 5

    def test_different_hypotheses_isolated(self) -> None:
        """Evidence for different hypotheses should not mix."""
        ledger = EvidenceLedger()
        ledger.add_evidence(_make_evidence(hypothesis_id="hyp_A", evidence_id="ev_A"))
        ledger.add_evidence(_make_evidence(hypothesis_id="hyp_B", evidence_id="ev_B"))

        assert len(ledger.get_evidence("hyp_A")) == 1
        assert len(ledger.get_evidence("hyp_B")) == 1
        assert len(ledger.get_evidence("hyp_C")) == 0

    def test_order_preserved(self) -> None:
        """Evidence should be returned in insertion order (chronological)."""
        ledger = EvidenceLedger()
        for i in range(3):
            ledger.add_evidence(
                _make_evidence(
                    evidence_id=f"ev_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        result = ledger.get_evidence("hyp_001")
        assert [r.evidence_id for r in result] == ["ev_0", "ev_1", "ev_2"]


class TestEvidenceSummary:
    """Tests for get_summary with BKT integration."""

    def test_empty_summary(self) -> None:
        """Summary for non-existent hypothesis should have zero counts."""
        ledger = EvidenceLedger()
        summary = ledger.get_summary("nonexistent")
        assert summary.positive_count == 0
        assert summary.negative_count == 0
        assert summary.total_count == 0
        assert summary.weighted_score == 0.5  # BKT prior
        assert summary.distinct_sessions == 0

    def test_positive_evidence_increases_score(self) -> None:
        """Positive evidence should increase weighted_score above 0.5."""
        ledger = EvidenceLedger()
        for i in range(3):
            ledger.add_evidence(
                _make_evidence(
                    signal_type="explicit_confirm",
                    evidence_id=f"ev_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        summary = ledger.get_summary("hyp_001")
        assert summary.positive_count == 3
        assert summary.negative_count == 0
        assert summary.weighted_score > 0.5

    def test_negative_evidence_decreases_score(self) -> None:
        """Negative evidence should decrease weighted_score below 0.5."""
        ledger = EvidenceLedger()
        for i in range(3):
            ledger.add_evidence(
                _make_evidence(
                    signal_type="correction",
                    evidence_id=f"ev_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        summary = ledger.get_summary("hyp_001")
        assert summary.positive_count == 0
        assert summary.negative_count == 3
        assert summary.weighted_score < 0.5

    def test_mixed_evidence_intermediate_score(self) -> None:
        """Mixed evidence should produce intermediate weighted_score."""
        ledger = EvidenceLedger()
        # 3 positive
        for i in range(3):
            ledger.add_evidence(
                _make_evidence(
                    signal_type="no_correction",
                    evidence_id=f"ev_pos_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        # 2 negative
        for i in range(2):
            ledger.add_evidence(
                _make_evidence(
                    signal_type="rejection",
                    evidence_id=f"ev_neg_{i}",
                    created_at=f"2026-05-20T11:0{i}:00+00:00",
                )
            )
        summary = ledger.get_summary("hyp_001")
        assert summary.positive_count == 3
        assert summary.negative_count == 2
        assert summary.total_count == 5

    def test_session_counting(self) -> None:
        """Distinct sessions should be counted correctly."""
        ledger = EvidenceLedger()
        sessions = ["session_A", "session_B", "session_A", "session_C"]
        for i, session in enumerate(sessions):
            ledger.add_evidence(
                _make_evidence(
                    episode_id=session,
                    evidence_id=f"ev_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        summary = ledger.get_summary("hyp_001")
        assert summary.distinct_sessions == 3  # A, B, C

    def test_last_timestamps_tracked(self) -> None:
        """Last positive/negative timestamps should be correct."""
        ledger = EvidenceLedger()
        ledger.add_evidence(
            _make_evidence(
                signal_type="bookmark",
                evidence_id="ev_1",
                created_at="2026-05-20T10:00:00+00:00",
            )
        )
        ledger.add_evidence(
            _make_evidence(
                signal_type="correction",
                evidence_id="ev_2",
                created_at="2026-05-20T11:00:00+00:00",
            )
        )
        ledger.add_evidence(
            _make_evidence(
                signal_type="explicit_confirm",
                evidence_id="ev_3",
                created_at="2026-05-20T12:00:00+00:00",
            )
        )
        summary = ledger.get_summary("hyp_001")
        assert summary.last_positive_at == "2026-05-20T12:00:00+00:00"
        assert summary.last_negative_at == "2026-05-20T11:00:00+00:00"


class TestSignalStrength:
    """Tests for signal_strength modulating BKT updates."""

    def test_strong_signal_bigger_update(self) -> None:
        """Signal strength 1.0 should produce bigger BKT update than 0.3."""
        ledger_strong = EvidenceLedger()
        ledger_weak = EvidenceLedger()

        ledger_strong.add_evidence(
            _make_evidence(signal_strength=1.0, evidence_id="ev_s")
        )
        ledger_weak.add_evidence(
            _make_evidence(signal_strength=0.3, evidence_id="ev_w")
        )

        strong_score = ledger_strong.get_summary("hyp_001").weighted_score
        weak_score = ledger_weak.get_summary("hyp_001").weighted_score

        # Both should be > 0.5 (positive signal) but strong > weak
        assert strong_score > 0.5
        assert weak_score > 0.5
        assert strong_score > weak_score

    def test_zero_strength_no_update(self) -> None:
        """Signal strength 0.0 should not change the BKT score."""
        ledger = EvidenceLedger()
        ledger.add_evidence(_make_evidence(signal_strength=0.0, evidence_id="ev_z"))
        summary = ledger.get_summary("hyp_001")
        # BKT should remain at prior (0.5)
        assert summary.weighted_score == 0.5


class TestRecentContradictions:
    """Tests for recent contradiction counting."""

    def test_all_negative_recent(self) -> None:
        """All recent records negative should return full count."""
        ledger = EvidenceLedger()
        for i in range(5):
            ledger.add_evidence(
                _make_evidence(
                    signal_type="correction",
                    evidence_id=f"ev_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        assert ledger.get_recent_contradictions("hyp_001", limit=5) == 5

    def test_mixed_recent(self) -> None:
        """Mixed signals should count only negatives in window."""
        ledger = EvidenceLedger()
        signals = [
            "no_correction",
            "correction",
            "no_correction",
            "rejection",
            "correction",
        ]
        for i, sig in enumerate(signals):
            ledger.add_evidence(
                _make_evidence(
                    signal_type=sig,
                    evidence_id=f"ev_{i}",
                    created_at=f"2026-05-20T10:0{i}:00+00:00",
                )
            )
        # Last 3: no_correction, rejection, correction = 2 negatives
        assert ledger.get_recent_contradictions("hyp_001", limit=3) == 2


class TestSignalClassification:
    """Tests for is_positive_signal helper."""

    @pytest.mark.parametrize("signal_type", list(POSITIVE_SIGNALS))
    def test_positive_signals(self, signal_type: str) -> None:
        """All positive signal types should return True."""
        assert is_positive_signal(signal_type) is True

    @pytest.mark.parametrize("signal_type", list(NEGATIVE_SIGNALS))
    def test_negative_signals(self, signal_type: str) -> None:
        """All negative signal types should return False."""
        assert is_positive_signal(signal_type) is False

    def test_invalid_signal_raises(self) -> None:
        """Unknown signal type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown signal type"):
            is_positive_signal("invalid_type")


class TestEvidenceLedgerEdgeCases:
    """Edge case tests for the Evidence Ledger."""

    def test_invalid_signal_type_raises(self) -> None:
        """Adding evidence with invalid signal_type should raise."""
        ledger = EvidenceLedger()
        record = EvidenceRecord(
            evidence_id="ev_bad",
            hypothesis_id="hyp_001",
            hypothesis_version=1,
            episode_id="ep_001",
            request_id=None,
            response_id=None,
            signal_type="invalid_type",  # type: ignore[arg-type]
            signal_strength=1.0,
            created_at="2026-05-20T10:00:00+00:00",
        )
        with pytest.raises(ValueError, match="Invalid signal_type"):
            ledger.add_evidence(record)

    def test_has_evidence_false_initially(self) -> None:
        """has_evidence should be False before any evidence is added."""
        ledger = EvidenceLedger()
        assert ledger.has_evidence("hyp_001") is False

    def test_has_evidence_true_after_add(self) -> None:
        """has_evidence should be True after adding evidence."""
        ledger = EvidenceLedger()
        ledger.add_evidence(_make_evidence())
        assert ledger.has_evidence("hyp_001") is True

    def test_clear_hypothesis(self) -> None:
        """clear_hypothesis should remove all evidence and BKT state."""
        ledger = EvidenceLedger()
        ledger.add_evidence(_make_evidence())
        ledger.clear_hypothesis("hyp_001")
        assert ledger.has_evidence("hyp_001") is False
        assert ledger.get_evidence("hyp_001") == []
        # BKT should reset to initial
        bkt = ledger.get_bkt_state("hyp_001")
        assert bkt.p_knowledge == 0.5
