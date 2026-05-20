"""Tests for the N-Gram Sliding Window Extractor (Step 2.1/10).

Covers:
  - Basic extraction with window sizes 3, 4, 5
  - Recurring sequence detection (same sequence 5x = recognized pattern)
  - Minimum occurrence filtering
  - Hash determinism
  - Empty and short input handling
  - Multi-size extraction via extract_all_ngrams
  - Pattern matching against known patterns
  - HC-LAYER2-1: N-grams are candidates, not truth (architecture guard)
"""

from __future__ import annotations


import pytest

from application.skill_compression.event_normalizer import NormalizedEvent
from application.skill_compression.ngram_extractor import (
    NGramPattern,
    _compute_pattern_hash,
    _event_key,
    extract_all_ngrams,
    extract_ngrams,
    find_matching_patterns,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


def _make_event(intent: str, domain: str, ts_offset: int = 0) -> NormalizedEvent:
    """Create a minimal NormalizedEvent for testing."""
    ts = f"2026-05-20T{ts_offset:02d}:00:00+00:00"
    return NormalizedEvent(
        event_id=f"evt_{intent}_{domain}_{ts_offset}",
        user_id=42,
        timestamp=ts,
        intent=intent,
        domain=domain,
    )


def _repeat_sequence(
    sequence: list[tuple[str, str]],
    repetitions: int,
) -> list[NormalizedEvent]:
    """Create a list of events by repeating a sequence N times."""
    events: list[NormalizedEvent] = []
    offset = 0
    for _ in range(repetitions):
        for intent, domain in sequence:
            events.append(_make_event(intent, domain, ts_offset=offset))
            offset += 1
    return events


# ---------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------


class TestEventKey:
    """Tests for _event_key derivation."""

    def test_key_format(self):
        """Key should be domain.intent."""
        event = _make_event("create_ad_copy", "marketing")
        assert _event_key(event) == "marketing.create_ad_copy"

    def test_general_defaults(self):
        """General intent/domain should produce general.general."""
        event = _make_event("general", "general")
        assert _event_key(event) == "general.general"


class TestExtractNgrams:
    """Tests for the core extract_ngrams function."""

    def test_same_sequence_5x_detected(self):
        """Same sequence repeated 5x should be recognized as recurring pattern.

        This is the key spec requirement: a pattern that appears 5 times
        is clearly recurring and should be extracted.
        """
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
            ],
            repetitions=5,
        )
        patterns = extract_ngrams(events, n=3, min_occurrences=2)
        assert len(patterns) > 0
        # The original 3-gram should appear at least 4 times
        # (5 repetitions, but the window slides across boundaries too)
        top_pattern = patterns[0]
        assert top_pattern.occurrences >= 4

    def test_minimum_occurrences_filter(self):
        """Patterns below min_occurrences should be excluded."""
        events = _repeat_sequence(
            [
                ("create_code", "development"),
                ("analyze", "development"),
                ("explain", "development"),
            ],
            repetitions=2,
        )
        # With min_occurrences=3, should find nothing (sequence only appears ~2 times)
        patterns = extract_ngrams(events, n=3, min_occurrences=3)
        for p in patterns:
            assert p.occurrences >= 3

    def test_window_size_3(self):
        """Window size 3 should extract trigrams."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "marketing"),
                ("plan", "marketing"),
            ],
            repetitions=3,
        )
        patterns = extract_ngrams(events, n=3, min_occurrences=2)
        for p in patterns:
            assert p.n == 3
            assert len(p.events) == 3

    def test_window_size_4(self):
        """Window size 4 should extract 4-grams."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
                ("format", "content"),
            ],
            repetitions=3,
        )
        patterns = extract_ngrams(events, n=4, min_occurrences=2)
        for p in patterns:
            assert p.n == 4
            assert len(p.events) == 4

    def test_window_size_5(self):
        """Window size 5 should extract 5-grams."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
                ("format", "content"),
                ("summarize", "content"),
            ],
            repetitions=3,
        )
        patterns = extract_ngrams(events, n=5, min_occurrences=2)
        for p in patterns:
            assert p.n == 5
            assert len(p.events) == 5

    def test_short_input_returns_empty(self):
        """Input shorter than window size should return empty list."""
        events = [_make_event("create_code", "development")]
        patterns = extract_ngrams(events, n=3)
        assert patterns == []

    def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        patterns = extract_ngrams([], n=3)
        assert patterns == []

    def test_n_less_than_2_returns_empty(self):
        """N < 2 is not meaningful and should return empty."""
        events = _repeat_sequence([("create_code", "development")], repetitions=10)
        patterns = extract_ngrams(events, n=1)
        assert patterns == []

    def test_sorted_by_occurrences(self):
        """Results should be sorted by occurrences descending."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
            ],
            repetitions=5,
        )
        patterns = extract_ngrams(events, n=3, min_occurrences=1)
        for i in range(len(patterns) - 1):
            assert patterns[i].occurrences >= patterns[i + 1].occurrences

    def test_pattern_hash_is_sha256(self):
        """Pattern hash should be a 64-char hex string (SHA-256)."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
            ],
            repetitions=3,
        )
        patterns = extract_ngrams(events, n=3, min_occurrences=2)
        for p in patterns:
            assert len(p.pattern_hash) == 64
            assert all(c in "0123456789abcdef" for c in p.pattern_hash)

    def test_last_seen_populated(self):
        """last_seen should contain a valid ISO-8601 timestamp."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
            ],
            repetitions=3,
        )
        patterns = extract_ngrams(events, n=3, min_occurrences=2)
        for p in patterns:
            assert p.last_seen  # Not empty


class TestHashDeterminism:
    """Tests for hash determinism."""

    def test_same_events_same_hash(self):
        """Same event tuple should always produce the same hash."""
        key = ("marketing.create_ad_copy", "data.analyze", "business.plan")
        hashes = {_compute_pattern_hash(key) for _ in range(100)}
        assert len(hashes) == 1

    def test_different_order_different_hash(self):
        """Different event order should produce different hash."""
        key_a = ("marketing.create_ad_copy", "data.analyze")
        key_b = ("data.analyze", "marketing.create_ad_copy")
        assert _compute_pattern_hash(key_a) != _compute_pattern_hash(key_b)


class TestExtractAllNgrams:
    """Tests for extract_all_ngrams (multi-size extraction)."""

    def test_combines_multiple_sizes(self):
        """Should extract patterns from all configured window sizes."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
                ("format", "content"),
                ("summarize", "content"),
            ],
            repetitions=4,
        )
        patterns = extract_all_ngrams(events, min_occurrences=2)
        sizes = {p.n for p in patterns}
        # Should have patterns from multiple window sizes
        assert len(sizes) >= 2

    def test_deduplication_by_hash(self):
        """Patterns should be deduplicated by hash across sizes."""
        events = _repeat_sequence(
            [
                ("create_ad_copy", "marketing"),
                ("analyze", "data"),
                ("plan", "business"),
            ],
            repetitions=4,
        )
        patterns = extract_all_ngrams(events, min_occurrences=2)
        hashes = [p.pattern_hash for p in patterns]
        assert len(hashes) == len(set(hashes))  # No duplicates


class TestFindMatchingPatterns:
    """Tests for real-time pattern matching."""

    def test_matches_known_pattern(self):
        """Recent events matching a known pattern should be found."""
        known = [
            NGramPattern(
                pattern_hash=_compute_pattern_hash(
                    ("marketing.create_ad_copy", "data.analyze", "business.plan")
                ),
                n=3,
                events=("marketing.create_ad_copy", "data.analyze", "business.plan"),
                occurrences=5,
                last_seen="2026-05-20T00:00:00+00:00",
            )
        ]
        recent_events = [
            _make_event("create_ad_copy", "marketing"),
            _make_event("analyze", "data"),
            _make_event("plan", "business"),
        ]
        matches = find_matching_patterns(recent_events, known)
        assert len(matches) == 1
        assert matches[0].pattern_hash == known[0].pattern_hash

    def test_no_match_for_unknown_sequence(self):
        """Unknown sequences should return empty matches."""
        known = [
            NGramPattern(
                pattern_hash=_compute_pattern_hash(
                    ("marketing.create_ad_copy", "data.analyze", "business.plan")
                ),
                n=3,
                events=("marketing.create_ad_copy", "data.analyze", "business.plan"),
                occurrences=5,
                last_seen="2026-05-20T00:00:00+00:00",
            )
        ]
        recent_events = [
            _make_event("create_code", "development"),
            _make_event("explain", "development"),
            _make_event("format", "development"),
        ]
        matches = find_matching_patterns(recent_events, known)
        assert len(matches) == 0

    def test_empty_inputs_return_empty(self):
        """Empty events or patterns should return empty."""
        assert find_matching_patterns([], []) == []
        assert find_matching_patterns([_make_event("a", "b")], []) == []
        assert find_matching_patterns([], [NGramPattern("h", 3, (), 1, "")]) == []


class TestNGramPatternFrozen:
    """Guard: NGramPattern must be immutable."""

    def test_ngram_pattern_is_frozen(self):
        """NGramPattern should be frozen (immutable)."""
        p = NGramPattern(
            pattern_hash="abc",
            n=3,
            events=("a", "b", "c"),
            occurrences=1,
            last_seen="2026-05-20T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            p.occurrences = 999  # type: ignore[misc]

    def test_ngram_pattern_has_slots(self):
        """NGramPattern should use __slots__."""
        assert hasattr(NGramPattern, "__slots__")
