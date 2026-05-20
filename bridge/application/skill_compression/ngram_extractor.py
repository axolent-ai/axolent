"""Layer 2: N-Gram Sliding Window Extractor for Skill-Compression.

Slides windows of size 3, 4, and 5 over user action sequences (lists
of NormalizedEvent objects). Extracts recurring subsequences as
skill candidates.

HC-LAYER2-1: N-Gram patterns are candidates, NOT truth. They produce
proposals that feed into the Evidence Ledger (Layer 3), never
directly into SkillMatcher.

IC-NGRAM-1: All three window sizes (3, 4, 5) are extracted. Each
size captures different granularity:
  n=3: catches short workflows (e.g. "create_ad -> review -> publish")
  n=4: catches medium chains with setup steps
  n=5: catches longer procedural patterns

Patterns are identified by SHA-256 hash over the ordered event_id
tuple. Occurrence counting is frequency-based.

No external dependencies. No GPU. Pure Python.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from application.skill_compression.event_normalizer import NormalizedEvent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Default window sizes (IC-NGRAM-1: all three)
DEFAULT_WINDOW_SIZES: tuple[int, ...] = (3, 4, 5)

# Minimum occurrences before a pattern is considered recurring
MIN_OCCURRENCES: int = 2


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NGramPattern:
    """A recurring N-gram pattern extracted from user action sequences.

    Attributes:
        pattern_hash: SHA-256 over the canonical N-gram key sequence.
        n: Window size (3, 4, or 5).
        events: Ordered tuple of canonical event keys (intent.domain).
        occurrences: How many times this pattern was observed.
        last_seen: ISO-8601 UTC timestamp of the last occurrence.
    """

    pattern_hash: str
    n: int
    events: tuple[str, ...]
    occurrences: int
    last_seen: str


# ---------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------


def _event_key(event: NormalizedEvent) -> str:
    """Derive a canonical key from a NormalizedEvent for N-gram comparison.

    Uses intent + domain as the canonical representation. This matches
    the Markov Chain's state representation for consistency across
    Layer 2 algorithms.

    Args:
        event: A normalized event.

    Returns:
        Canonical key string (e.g. 'marketing.create_ad_copy').
    """
    return f"{event.domain}.{event.intent}"


def _compute_pattern_hash(events: tuple[str, ...]) -> str:
    """Compute a deterministic SHA-256 hash for an N-gram event tuple.

    Args:
        events: Ordered tuple of canonical event keys.

    Returns:
        64-character hex SHA-256 hash.
    """
    canonical = json.dumps(events, sort_keys=False, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_ngrams(
    events: list[NormalizedEvent],
    n: int = 3,
    *,
    min_occurrences: int = MIN_OCCURRENCES,
) -> list[NGramPattern]:
    """Extract recurring N-gram patterns from a sequence of events.

    Slides a window of size n over the event list and counts
    occurrences of each unique subsequence. Only patterns with
    at least min_occurrences are returned.

    Args:
        events: Ordered list of NormalizedEvents (chronological).
        n: Window size (typically 3, 4, or 5).
        min_occurrences: Minimum times a pattern must occur to be included.

    Returns:
        List of NGramPattern objects, sorted by occurrences descending.
    """
    if n < 2:
        log.warning("N-gram size %d < 2 is not meaningful, returning empty", n)
        return []

    if len(events) < n:
        return []

    # Slide window and collect canonical keys
    ngram_counter: Counter[tuple[str, ...]] = Counter()
    ngram_last_seen: dict[tuple[str, ...], str] = {}

    for i in range(len(events) - n + 1):
        window = events[i : i + n]
        key_tuple = tuple(_event_key(e) for e in window)
        ngram_counter[key_tuple] += 1
        # Track latest timestamp in this window
        latest_ts = max(e.timestamp for e in window if e.timestamp)
        ngram_last_seen[key_tuple] = latest_ts

    # Filter by minimum occurrences and build results
    results: list[NGramPattern] = []
    for key_tuple, count in ngram_counter.items():
        if count >= min_occurrences:
            pattern = NGramPattern(
                pattern_hash=_compute_pattern_hash(key_tuple),
                n=n,
                events=key_tuple,
                occurrences=count,
                last_seen=ngram_last_seen.get(key_tuple, ""),
            )
            results.append(pattern)

    # Sort by occurrences descending, then by n descending (longer = more specific)
    results.sort(key=lambda p: (p.occurrences, p.n), reverse=True)

    log.debug(
        "Extracted %d N-gram patterns (n=%d) from %d events (min_occ=%d)",
        len(results),
        n,
        len(events),
        min_occurrences,
    )

    return results


def extract_all_ngrams(
    events: list[NormalizedEvent],
    *,
    window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
    min_occurrences: int = MIN_OCCURRENCES,
) -> list[NGramPattern]:
    """Extract N-gram patterns for all configured window sizes.

    Combines results from n=3, n=4, and n=5 into a single list,
    deduplicated by pattern_hash and sorted by occurrences.

    Args:
        events: Ordered list of NormalizedEvents (chronological).
        window_sizes: Tuple of window sizes to use.
        min_occurrences: Minimum occurrences per pattern.

    Returns:
        Combined list of NGramPattern objects, sorted by occurrences.
    """
    seen_hashes: set[str] = set()
    all_patterns: list[NGramPattern] = []

    for n in window_sizes:
        patterns = extract_ngrams(events, n=n, min_occurrences=min_occurrences)
        for p in patterns:
            if p.pattern_hash not in seen_hashes:
                seen_hashes.add(p.pattern_hash)
                all_patterns.append(p)

    all_patterns.sort(key=lambda p: (p.occurrences, p.n), reverse=True)

    log.debug(
        "Total N-gram patterns across sizes %s: %d",
        window_sizes,
        len(all_patterns),
    )

    return all_patterns


def find_matching_patterns(
    events: list[NormalizedEvent],
    existing_patterns: list[NGramPattern],
    n: Optional[int] = None,
) -> list[NGramPattern]:
    """Check if a sequence of events matches any known N-gram patterns.

    Used for real-time matching: given the latest n events, check if
    they form a known pattern. This is the "recognition" side of the
    N-gram extractor.

    Args:
        events: Recent events to check (should be at least n elements).
        existing_patterns: Known patterns to match against.
        n: Window size to check. If None, tries all pattern sizes.

    Returns:
        List of matching NGramPattern objects.
    """
    if not events or not existing_patterns:
        return []

    # Build a hash set of existing patterns for O(1) lookup
    pattern_by_hash: dict[str, NGramPattern] = {
        p.pattern_hash: p for p in existing_patterns
    }

    matches: list[NGramPattern] = []

    # Determine which sizes to check
    sizes_to_check: set[int]
    if n is not None:
        sizes_to_check = {n}
    else:
        sizes_to_check = {p.n for p in existing_patterns}

    for size in sizes_to_check:
        if len(events) < size:
            continue
        # Check the last 'size' events
        recent = events[-size:]
        key_tuple = tuple(_event_key(e) for e in recent)
        candidate_hash = _compute_pattern_hash(key_tuple)
        if candidate_hash in pattern_by_hash:
            matches.append(pattern_by_hash[candidate_hash])

    return matches
