"""Production-path tests for hypothesis status state machine (W4).

Tests the validated transition_hypothesis_status method and
InvalidStatusTransition exception.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from application.skill_compression.hypothesis_storage import (
    ALLOWED_TRANSITIONS,
    Hypothesis,
    HypothesisStorage,
    InvalidStatusTransition,
)


@pytest.fixture
def storage(tmp_path: Path) -> HypothesisStorage:
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_sm.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    s = HypothesisStorage(conn)
    s.init_schema()
    return s


def _insert_hyp(storage: HypothesisStorage, hyp_id: str, status: str) -> None:
    """Insert a hypothesis with given status."""
    now = datetime.now(timezone.utc).isoformat()
    h = Hypothesis(
        hypothesis_id=hyp_id,
        user_id=1,
        type="preference",
        claim="Test claim",
        status=status,
        created_at=now,
        last_seen=now,
    )
    storage.insert_hypothesis(h)


class TestTransitionArchivedToActiveRejected:
    """Edge-Case 3 from Briefing: archived -> active must be rejected."""

    def test_transition_archived_to_active_rejected(
        self, storage: HypothesisStorage
    ) -> None:
        _insert_hyp(storage, "hyp_arch", "archived")
        with pytest.raises(InvalidStatusTransition) as exc_info:
            storage.transition_hypothesis_status("hyp_arch", "active")
        assert exc_info.value.current_status == "archived"
        assert exc_info.value.target_status == "active"

    def test_archived_to_retired_allowed(self, storage: HypothesisStorage) -> None:
        """archived -> retired is the only non-force transition."""
        _insert_hyp(storage, "hyp_arch2", "archived")
        storage.transition_hypothesis_status("hyp_arch2", "retired")
        h = storage.get_hypothesis("hyp_arch2")
        assert h is not None
        assert h.status == "retired"


class TestTransitionRetiredToAnythingRejected:
    """Retired is terminal: no transitions out."""

    def test_retired_to_active_rejected(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_ret", "retired")
        with pytest.raises(InvalidStatusTransition):
            storage.transition_hypothesis_status("hyp_ret", "active")

    def test_retired_to_confirmed_rejected(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_ret2", "retired")
        with pytest.raises(InvalidStatusTransition):
            storage.transition_hypothesis_status("hyp_ret2", "confirmed")

    def test_retired_to_suggested_rejected(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_ret3", "retired")
        with pytest.raises(InvalidStatusTransition):
            storage.transition_hypothesis_status("hyp_ret3", "suggested")


class TestForceTransitionBypassesCheck:
    """force=True allows any transition (admin/migration)."""

    def test_force_transition_bypasses_check(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_force", "retired")
        # This would normally raise InvalidStatusTransition
        storage.transition_hypothesis_status("hyp_force", "active", force=True)
        h = storage.get_hypothesis("hyp_force")
        assert h is not None
        assert h.status == "active"

    def test_force_archived_to_confirmed(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_force2", "archived")
        storage.transition_hypothesis_status("hyp_force2", "confirmed", force=True)
        h = storage.get_hypothesis("hyp_force2")
        assert h is not None
        assert h.status == "confirmed"


class TestAllowedTransitionsStillWork:
    """Verify that all allowed transitions still succeed."""

    def test_candidate_to_suggested(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_cs", "candidate")
        storage.transition_hypothesis_status("hyp_cs", "suggested")
        assert storage.get_hypothesis("hyp_cs").status == "suggested"

    def test_suggested_to_confirmed(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_sc", "suggested")
        storage.transition_hypothesis_status("hyp_sc", "confirmed")
        assert storage.get_hypothesis("hyp_sc").status == "confirmed"

    def test_confirmed_to_active(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_ca", "confirmed")
        storage.transition_hypothesis_status("hyp_ca", "active")
        assert storage.get_hypothesis("hyp_ca").status == "active"

    def test_confirmed_to_paused(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_cp", "confirmed")
        storage.transition_hypothesis_status("hyp_cp", "paused")
        assert storage.get_hypothesis("hyp_cp").status == "paused"

    def test_active_to_paused(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_ap", "active")
        storage.transition_hypothesis_status("hyp_ap", "paused")
        assert storage.get_hypothesis("hyp_ap").status == "paused"

    def test_paused_to_active(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_pa", "paused")
        storage.transition_hypothesis_status("hyp_pa", "active")
        assert storage.get_hypothesis("hyp_pa").status == "active"

    def test_needs_review_to_confirmed(self, storage: HypothesisStorage) -> None:
        _insert_hyp(storage, "hyp_nrc", "needs_review")
        storage.transition_hypothesis_status("hyp_nrc", "confirmed")
        assert storage.get_hypothesis("hyp_nrc").status == "confirmed"


class TestTransitionMatrixComplete:
    """Verify the transition matrix has entries for all statuses."""

    def test_all_statuses_have_transition_entry(self) -> None:
        expected = {
            "candidate",
            "suggested",
            "confirmed",
            "active",
            "needs_review",
            "paused",
            "archived",
            "retired",
        }
        assert set(ALLOWED_TRANSITIONS.keys()) == expected

    def test_retired_has_empty_transitions(self) -> None:
        assert ALLOWED_TRANSITIONS["retired"] == frozenset()

    def test_archived_only_to_retired(self) -> None:
        assert ALLOWED_TRANSITIONS["archived"] == frozenset({"retired"})


class TestInvalidStatusTransitionException:
    """Verify the exception contains useful debugging info."""

    def test_exception_has_fields(self) -> None:
        exc = InvalidStatusTransition("hyp_test", "archived", "active")
        assert exc.hypothesis_id == "hyp_test"
        assert exc.current_status == "archived"
        assert exc.target_status == "active"
        assert "archived" in str(exc)
        assert "active" in str(exc)
