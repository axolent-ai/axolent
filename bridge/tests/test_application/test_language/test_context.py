"""Tests for LanguageContext Phase 2 expansion (Step 2/4).

Covers:
- Backward compatibility: Phase-1-style construction still works
- Phase 2 properties: has_detection_metadata, top_alternative
- Frozen invariant: all Phase 1 + Phase 2 fields immutable
- Property-based tests: override invariant, distribution sum, reliability range
- Edge cases: large distribution, equal probabilities
- Architecture guard: frozen=True + slots=True enforced on all fields
- classify_text_length: HC-C5 bucket boundaries

Test naming convention: test_<subject>_<scenario>_<expected>.
"""

from __future__ import annotations

import dataclasses

import pytest

from application.language.context import (
    TEXT_LENGTH_BUCKETS,
    LanguageContext,
)


# -- Fixtures --------------------------------------------------------------


def _phase1_context(**overrides: object) -> LanguageContext:
    """Build a Phase-1-style LanguageContext (no Phase 2 fields)."""
    defaults: dict[str, object] = {
        "code": "de",
        "source": "detected",
        "confidence": 0.9,
        "switched_from": None,
        "request_id": "test-abc-123",
    }
    defaults.update(overrides)
    return LanguageContext(**defaults)  # type: ignore[arg-type]


def _phase2_context(**overrides: object) -> LanguageContext:
    """Build a fully populated Phase-2 LanguageContext."""
    defaults: dict[str, object] = {
        "code": "de",
        "source": "detected",
        "confidence": 0.93,
        "switched_from": None,
        "request_id": "test-p2-456",
        "detection_distribution": {"de": 0.93, "nl": 0.05, "en": 0.02},
        "reliability_score": 0.88,
        "confidence_history": (("langdetect", 0.93),),
        "detection_tier": "high",
        "text_length_bucket": "medium",
        "backends_consulted": frozenset({"langdetect"}),
    }
    defaults.update(overrides)
    return LanguageContext(**defaults)  # type: ignore[arg-type]


# -- Backward Compatibility (HC-C2) ----------------------------------------


class TestBackwardCompatibility:
    """Phase-1-style construction must still work without any Phase 2 fields."""

    def test_phase1_construction_no_error(self) -> None:
        """HC-C2: LanguageContext with only Phase 1 fields constructs fine."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="abc",
        )
        assert ctx.code == "de"
        assert ctx.source == "detected"
        assert ctx.confidence == 0.9
        assert ctx.switched_from is None
        assert ctx.request_id == "abc"

    def test_phase1_defaults_for_phase2_fields(self) -> None:
        """All Phase 2 fields have their documented defaults."""
        ctx = _phase1_context()
        assert ctx.detection_distribution == {}
        assert ctx.reliability_score == 0.0
        assert ctx.confidence_history == ()
        assert ctx.detection_tier is None
        assert ctx.text_length_bucket is None
        assert ctx.backends_consulted == frozenset()

    def test_phase1_effective_lang_still_works(self) -> None:
        """effective_lang() unchanged from Phase 1."""
        ctx = _phase1_context(code="fr")
        assert ctx.effective_lang() == "fr"

    def test_phase1_was_smart_switched_still_works(self) -> None:
        """was_smart_switched property unchanged from Phase 1."""
        ctx_no = _phase1_context(switched_from=None)
        ctx_yes = _phase1_context(switched_from="en")
        assert ctx_no.was_smart_switched is False
        assert ctx_yes.was_smart_switched is True


# -- has_detection_metadata property ----------------------------------------


class TestHasDetectionMetadata:
    """Tests for has_detection_metadata property."""

    def test_false_for_phase1_style(self) -> None:
        """Phase-1-style context has no detection metadata."""
        ctx = _phase1_context()
        assert ctx.has_detection_metadata is False

    def test_true_with_distribution(self) -> None:
        """Context with populated distribution has detection metadata."""
        ctx = _phase2_context()
        assert ctx.has_detection_metadata is True

    def test_false_for_empty_distribution(self) -> None:
        """Explicitly empty distribution means no metadata."""
        ctx = _phase2_context(detection_distribution={})
        assert ctx.has_detection_metadata is False

    def test_true_for_single_entry_distribution(self) -> None:
        """Even a single entry counts as metadata."""
        ctx = _phase2_context(detection_distribution={"de": 1.0})
        assert ctx.has_detection_metadata is True


# -- top_alternative property -----------------------------------------------


class TestTopAlternative:
    """Tests for top_alternative property."""

    def test_none_for_empty_distribution(self) -> None:
        """Empty distribution returns None."""
        ctx = _phase1_context()
        assert ctx.top_alternative is None

    def test_none_for_single_entry(self) -> None:
        """Single-entry distribution returns None."""
        ctx = _phase2_context(detection_distribution={"de": 1.0})
        assert ctx.top_alternative is None

    def test_returns_second_highest(self) -> None:
        """Returns the second-most-likely language."""
        ctx = _phase2_context(
            detection_distribution={"de": 0.85, "nl": 0.10, "en": 0.05}
        )
        assert ctx.top_alternative == "nl"

    def test_deterministic_with_equal_probabilities(self) -> None:
        """When two languages tie, tiebreak is alphabetical by code."""
        ctx = _phase2_context(
            detection_distribution={"de": 0.50, "nl": 0.25, "en": 0.25}
        )
        # de is top (0.50). en and nl both 0.25. Alphabetically en < nl.
        assert ctx.top_alternative == "en"

    def test_large_distribution_20_languages(self) -> None:
        """top_alternative works correctly with 20+ language entries."""
        dist: dict[str, float] = {}
        codes = [
            "ar",
            "da",
            "de",
            "en",
            "es",
            "fi",
            "fr",
            "hi",
            "id",
            "it",
            "ja",
            "ko",
            "nb",
            "nl",
            "pl",
            "pt",
            "ru",
            "sv",
            "th",
            "tr",
            "uk",
            "vi",
            "zh",
        ]
        # de gets highest, nl second highest, rest distributed
        for i, code in enumerate(codes):
            if code == "de":
                dist[code] = 0.50
            elif code == "nl":
                dist[code] = 0.20
            else:
                dist[code] = 0.30 / (len(codes) - 2)
        ctx = _phase2_context(detection_distribution=dist)
        assert ctx.top_alternative == "nl"

    def test_all_equal_probabilities(self) -> None:
        """All languages with equal probability: tiebreak is alphabetical."""
        dist = {"nl": 0.25, "de": 0.25, "en": 0.25, "fr": 0.25}
        ctx = _phase2_context(detection_distribution=dist)
        # All equal: sorted alphabetically -> de, en, fr, nl
        # top = de, alternative = en
        assert ctx.top_alternative == "en"


# -- Frozen Invariant (HC-C1, Guard 4) -------------------------------------


class TestFrozenInvariant:
    """All fields must be immutable after construction."""

    # Phase 1 fields
    def test_frozen_code(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.code = "en"  # type: ignore[misc]

    def test_frozen_source(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.source = "override"  # type: ignore[misc]

    def test_frozen_confidence(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.confidence = 0.5  # type: ignore[misc]

    def test_frozen_switched_from(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.switched_from = "en"  # type: ignore[misc]

    def test_frozen_request_id(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.request_id = "new-id"  # type: ignore[misc]

    # Phase 2 fields
    def test_frozen_detection_distribution(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.detection_distribution = {}  # type: ignore[misc]

    def test_frozen_reliability_score(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.reliability_score = 0.5  # type: ignore[misc]

    def test_frozen_confidence_history(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.confidence_history = ()  # type: ignore[misc]

    def test_frozen_detection_tier(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.detection_tier = "low"  # type: ignore[misc]

    def test_frozen_text_length_bucket(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.text_length_bucket = "micro"  # type: ignore[misc]

    def test_frozen_backends_consulted(self) -> None:
        ctx = _phase2_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.backends_consulted = frozenset()  # type: ignore[misc]


class TestArchitectureGuardFrozen:
    """Spec Guard 4: comprehensive frozen verification across all fields."""

    def test_language_context_all_fields_frozen(self) -> None:
        """Construct LanguageContext, attempt assignment on EACH field,
        verify FrozenInstanceError. All Phase 1 + Phase 2 fields included.

        Uses setattr() (not object.__setattr__) because frozen=True
        only intercepts normal attribute assignment, not the bypass path.
        """
        ctx = _phase2_context()
        all_field_names = [f.name for f in dataclasses.fields(ctx)]

        # Verify we test ALL fields (catch future additions)
        expected_fields = {
            # Phase 1
            "code",
            "source",
            "confidence",
            "switched_from",
            "request_id",
            # Phase 2
            "detection_distribution",
            "reliability_score",
            "confidence_history",
            "detection_tier",
            "text_length_bucket",
            "backends_consulted",
        }
        assert set(all_field_names) == expected_fields, (
            f"Fields changed! Expected {expected_fields}, got {set(all_field_names)}. "
            f"Update this guard test."
        )

        for field_name in all_field_names:
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(ctx, field_name, "MUTATED")

    def test_language_context_uses_slots(self) -> None:
        """HC-C1: LanguageContext must use __slots__ (no __dict__)."""
        ctx = _phase1_context()
        assert not hasattr(ctx, "__dict__"), (
            "LanguageContext has __dict__; slots=True is not active"
        )


# -- Property-Based Tests --------------------------------------------------


class TestPropertyBased:
    """Invariant-style tests across different LanguageContext configurations."""

    def test_override_source_implies_empty_distribution(self) -> None:
        """source='override' -> detection_distribution is empty, confidence=1.0.

        This is a Phase-1 invariant that must hold after Phase-2 expansion.
        """
        ctx = LanguageContext(
            code="fr",
            source="override",
            confidence=1.0,
            switched_from=None,
            request_id="test-override",
        )
        assert ctx.detection_distribution == {}
        assert ctx.confidence == 1.0

    def test_override_source_with_explicit_empty_distribution(self) -> None:
        """Explicit override with empty distribution is valid."""
        ctx = LanguageContext(
            code="fr",
            source="override",
            confidence=1.0,
            switched_from=None,
            request_id="test-override-2",
            detection_distribution={},
        )
        assert ctx.detection_distribution == {}
        assert ctx.has_detection_metadata is False

    def test_distribution_sum_within_tolerance(self) -> None:
        """Non-empty distribution values sum to roughly 1.0 (float tolerance)."""
        distributions = [
            {"de": 0.93, "nl": 0.05, "en": 0.02},
            {"en": 0.80, "de": 0.15, "fr": 0.05},
            {"ja": 1.0},
            {"nl": 0.333, "de": 0.333, "en": 0.334},
        ]
        for dist in distributions:
            ctx = _phase2_context(detection_distribution=dist)
            total = sum(ctx.detection_distribution.values())
            assert 0.9 <= total <= 1.1, (
                f"Distribution sum {total} outside [0.9, 1.1]: {dist}"
            )

    def test_reliability_score_in_range(self) -> None:
        """reliability_score is always in [0.0, 1.0]."""
        for score in [0.0, 0.5, 0.88, 1.0]:
            ctx = _phase2_context(reliability_score=score)
            assert 0.0 <= ctx.reliability_score <= 1.0

    def test_confidence_history_is_tuple_of_tuples(self) -> None:
        """HC-C4: confidence_history is Tuple[Tuple[str, float], ...]."""
        history = (("langdetect", 0.93), ("domain_heuristic", 0.85))
        ctx = _phase2_context(confidence_history=history)
        assert isinstance(ctx.confidence_history, tuple)
        for entry in ctx.confidence_history:
            assert isinstance(entry, tuple)
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], float)

    def test_backends_consulted_is_frozenset(self) -> None:
        """IC-C4: backends_consulted is FrozenSet[str]."""
        ctx = _phase2_context(
            backends_consulted=frozenset({"langdetect", "domain_heuristic"})
        )
        assert isinstance(ctx.backends_consulted, frozenset)
        assert "langdetect" in ctx.backends_consulted


# -- classify_text_length (HC-C5) ------------------------------------------


class TestClassifyTextLength:
    """Tests for the static text length bucket classifier."""

    def test_micro_zero_words(self) -> None:
        assert LanguageContext.classify_text_length(0) == "micro"

    def test_micro_boundary(self) -> None:
        assert LanguageContext.classify_text_length(10) == "micro"

    def test_short_lower_boundary(self) -> None:
        assert LanguageContext.classify_text_length(11) == "short"

    def test_short_upper_boundary(self) -> None:
        assert LanguageContext.classify_text_length(30) == "short"

    def test_medium_lower_boundary(self) -> None:
        assert LanguageContext.classify_text_length(31) == "medium"

    def test_medium_upper_boundary(self) -> None:
        assert LanguageContext.classify_text_length(100) == "medium"

    def test_long_lower_boundary(self) -> None:
        assert LanguageContext.classify_text_length(101) == "long"

    def test_long_high_value(self) -> None:
        assert LanguageContext.classify_text_length(10000) == "long"

    def test_negative_treated_as_micro(self) -> None:
        """Negative word counts are treated as 0 (micro)."""
        assert LanguageContext.classify_text_length(-5) == "micro"

    def test_all_buckets_in_valid_set(self) -> None:
        """Every bucket returned by classify is in TEXT_LENGTH_BUCKETS."""
        for wc in [0, 1, 10, 11, 30, 31, 100, 101, 500]:
            bucket = LanguageContext.classify_text_length(wc)
            assert bucket in TEXT_LENGTH_BUCKETS, (
                f"Bucket {bucket!r} for word_count={wc} not in valid set"
            )


# -- TEXT_LENGTH_BUCKETS constant -------------------------------------------


class TestTextLengthBucketsConstant:
    """Verify the exported constant matches HC-C5."""

    def test_exactly_four_buckets(self) -> None:
        assert len(TEXT_LENGTH_BUCKETS) == 4

    def test_contains_expected_values(self) -> None:
        assert TEXT_LENGTH_BUCKETS == frozenset({"micro", "short", "medium", "long"})
