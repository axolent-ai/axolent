"""Tests for CollisionDetector (Layer 5), Commit 4.3.

Covers:
  - Specific scope (2 keys) beats global (0 keys) automatically
  - Medium-specific (1) beats global (0)
  - Tie at equal specificity -> requires_user_decision = True
  - Resolution reason is clearly set
  - resolve_pair convenience method
  - Single candidate: no collision
  - Empty candidates: no collision
  - CollisionDetector has no external dependency

HC-SC-11 [BLOCKER]: Skill Collision Detection.
"""

from __future__ import annotations

import inspect

from application.skill_compression.collision_detector import (
    CollisionDetector,
    compute_specificity,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.skill_matcher import SkillMatch


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_hypothesis(
    *,
    hypothesis_id: str = "hyp-001",
    claim: str = "Test rule",
    project: str = "",
    client: str = "",
    context: tuple[str, ...] = (),
) -> Hypothesis:
    """Create a test hypothesis with scope."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=42,
        type="preference",
        scope=HypothesisScope(project=project, client=client, context=context),
        claim=claim,
        status="confirmed",
        elo_rating=1600.0,
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
    )


def _make_skill_match(hyp: Hypothesis, confidence: float = 0.9) -> SkillMatch:
    """Wrap a hypothesis in a SkillMatch."""
    return SkillMatch(
        hypothesis=hyp,
        confidence=confidence,
        requires_confirmation=True,
        explanation="test match",
    )


# ---------------------------------------------------------------
# Tests: compute_specificity
# ---------------------------------------------------------------


class TestComputeSpecificity:
    """Test specificity scoring (IC-COLLISION-1)."""

    def test_global_scope_is_zero(self) -> None:
        """Empty scope = specificity 0."""
        scope = HypothesisScope()
        assert compute_specificity(scope) == 0

    def test_project_only_is_one(self) -> None:
        """Project-only scope = specificity 1."""
        scope = HypothesisScope(project="ads")
        assert compute_specificity(scope) == 1

    def test_client_only_is_two(self) -> None:
        """Client-only scope = specificity 2."""
        scope = HypothesisScope(client="acme")
        assert compute_specificity(scope) == 2

    def test_project_and_client(self) -> None:
        """Project + client = specificity 3."""
        scope = HypothesisScope(project="ads", client="acme")
        assert compute_specificity(scope) == 3

    def test_context_tags_add_specificity(self) -> None:
        """Context tags add to specificity (max 3 counted)."""
        scope = HypothesisScope(context=("tag1", "tag2"))
        assert compute_specificity(scope) == 2

    def test_context_tags_capped_at_three(self) -> None:
        """Context tags are capped at 3 for specificity."""
        scope = HypothesisScope(context=("a", "b", "c", "d", "e"))
        assert compute_specificity(scope) == 3

    def test_full_scope_max_specificity(self) -> None:
        """Project + client + 3 context = max specificity 6."""
        scope = HypothesisScope(project="ads", client="acme", context=("a", "b", "c"))
        assert compute_specificity(scope) == 6


# ---------------------------------------------------------------
# Tests: CollisionDetector.resolve
# ---------------------------------------------------------------


class TestCollisionDetectorResolve:
    """Test the main resolve method."""

    def test_specific_beats_global(self) -> None:
        """Hypothesis with client+project (spec=3) beats global (spec=0)."""
        detector = CollisionDetector()

        hyp_global = _make_hypothesis(
            hypothesis_id="hyp-global",
            claim="Global rule",
        )
        hyp_specific = _make_hypothesis(
            hypothesis_id="hyp-specific",
            claim="Client-specific rule",
            project="ads",
            client="acme",
        )

        matches = [
            _make_skill_match(hyp_global),
            _make_skill_match(hyp_specific),
        ]

        result = detector.resolve(matches, {})
        assert result.winner is not None
        assert result.winner.hypothesis_id == "hyp-specific"
        assert result.requires_user_decision is False

    def test_medium_beats_global(self) -> None:
        """Hypothesis with project-only (spec=1) beats global (spec=0)."""
        detector = CollisionDetector()

        hyp_global = _make_hypothesis(
            hypothesis_id="hyp-global",
            claim="Global rule",
        )
        hyp_medium = _make_hypothesis(
            hypothesis_id="hyp-medium",
            claim="Project-scoped rule",
            project="ads",
        )

        matches = [
            _make_skill_match(hyp_global),
            _make_skill_match(hyp_medium),
        ]

        result = detector.resolve(matches, {})
        assert result.winner is not None
        assert result.winner.hypothesis_id == "hyp-medium"
        assert result.requires_user_decision is False

    def test_equal_specificity_needs_user_decision(self) -> None:
        """Equal specificity -> requires_user_decision = True."""
        detector = CollisionDetector()

        hyp_a = _make_hypothesis(
            hypothesis_id="hyp-a",
            claim="Rule A",
            client="acme",
        )
        hyp_b = _make_hypothesis(
            hypothesis_id="hyp-b",
            claim="Rule B",
            client="beta",
        )

        matches = [
            _make_skill_match(hyp_a),
            _make_skill_match(hyp_b),
        ]

        result = detector.resolve(matches, {})
        assert result.winner is None
        assert result.requires_user_decision is True
        assert len(result.candidates) == 2

    def test_resolution_reason_set(self) -> None:
        """Resolution reason must be non-empty."""
        detector = CollisionDetector()

        hyp_a = _make_hypothesis(hypothesis_id="hyp-a", claim="A")
        hyp_b = _make_hypothesis(hypothesis_id="hyp-b", claim="B", project="ads")

        matches = [
            _make_skill_match(hyp_a),
            _make_skill_match(hyp_b),
        ]

        result = detector.resolve(matches, {})
        assert result.resolution_reason
        assert len(result.resolution_reason) > 10

    def test_single_candidate_no_collision(self) -> None:
        """Single candidate should return directly without collision."""
        detector = CollisionDetector()
        hyp = _make_hypothesis(claim="Only one")
        matches = [_make_skill_match(hyp)]

        result = detector.resolve(matches, {})
        assert result.winner is not None
        assert result.requires_user_decision is False
        assert result.winner.hypothesis_id == hyp.hypothesis_id

    def test_empty_candidates_no_collision(self) -> None:
        """Empty candidate list should return no winner."""
        detector = CollisionDetector()
        result = detector.resolve([], {})
        assert result.winner is None
        assert result.requires_user_decision is False

    def test_three_way_collision_most_specific_wins(self) -> None:
        """Three-way collision: most specific scope wins."""
        detector = CollisionDetector()

        hyp_global = _make_hypothesis(hypothesis_id="hyp-global", claim="Global")
        hyp_project = _make_hypothesis(
            hypothesis_id="hyp-project", claim="Project", project="ads"
        )
        hyp_full = _make_hypothesis(
            hypothesis_id="hyp-full",
            claim="Full",
            project="ads",
            client="acme",
        )

        matches = [
            _make_skill_match(hyp_global),
            _make_skill_match(hyp_project),
            _make_skill_match(hyp_full),
        ]

        result = detector.resolve(matches, {})
        assert result.winner is not None
        assert result.winner.hypothesis_id == "hyp-full"

    def test_three_way_tie_needs_user(self) -> None:
        """Three hypotheses with same specificity: user decides."""
        detector = CollisionDetector()

        hyp_a = _make_hypothesis(hypothesis_id="hyp-a", claim="A", project="web")
        hyp_b = _make_hypothesis(hypothesis_id="hyp-b", claim="B", project="ads")
        hyp_c = _make_hypothesis(hypothesis_id="hyp-c", claim="C", project="email")

        matches = [
            _make_skill_match(hyp_a),
            _make_skill_match(hyp_b),
            _make_skill_match(hyp_c),
        ]

        result = detector.resolve(matches, {})
        assert result.winner is None
        assert result.requires_user_decision is True
        assert len(result.candidates) == 3


# ---------------------------------------------------------------
# Tests: resolve_pair
# ---------------------------------------------------------------


class TestResolvePair:
    """Test the pairwise convenience method."""

    def test_pair_specific_wins(self) -> None:
        """Pairwise: specific scope wins."""
        detector = CollisionDetector()
        hyp_a = _make_hypothesis(claim="A")
        hyp_b = _make_hypothesis(claim="B", client="acme")

        result = detector.resolve_pair(hyp_a, hyp_b)
        assert result.winner is not None
        assert result.winner.claim == "B"
        assert result.requires_user_decision is False

    def test_pair_equal_needs_user(self) -> None:
        """Pairwise: equal specificity requires user decision."""
        detector = CollisionDetector()
        hyp_a = _make_hypothesis(claim="A", project="x")
        hyp_b = _make_hypothesis(claim="B", project="y")

        result = detector.resolve_pair(hyp_a, hyp_b)
        assert result.winner is None
        assert result.requires_user_decision is True


# ---------------------------------------------------------------
# Tests: Architecture Guards
# ---------------------------------------------------------------


class TestCollisionDetectorArchitectureGuard:
    """CollisionDetector must have no external dependency."""

    def test_no_database_dependency(self) -> None:
        """CollisionDetector requires no constructor arguments (stateless)."""
        # Must be instantiable with zero arguments
        detector = CollisionDetector()
        assert detector is not None

        # Verify resolve method only needs matches + scope (no DB)
        sig = inspect.signature(CollisionDetector.resolve)
        params = [p for p in sig.parameters if p != "self"]
        # Should only need: matches, current_scope
        assert "matches" in params
        assert "current_scope" in params
        # Must NOT have storage/conn/db parameters
        for p in params:
            assert "storage" not in p.lower()
            assert "conn" not in p.lower()
            assert "db" not in p.lower()
