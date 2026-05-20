"""Layer 2 Foundation: Fingerprint Similarity Engine.

Compares two NormalizedEvent objects via weighted structured-field similarity.
NOT output-fulltext similarity (Codex warning). Compares structural intent,
domain, format, constraints, scope, and language.

IC-SC-4: uses lightweight custom similarity (no sklearn, no TF-IDF corpus).
Reason: we compare structured fields, not free-text documents. The field
comparison is fundamentally different from TF-IDF on a corpus. Each field
has a known semantic type with a specific comparison strategy.

Field weights (Sigma's choice, tunable):
  intent:      30%  (most important signal)
  domain:      20%  (narrows the space)
  constraints: 20%  (specific requirements)
  format:      15%  (output format)
  scope:       15%  (context specificity)

Language is a hard filter (different language = not the same pattern),
not a weighted field.

Similarity threshold: > 0.7 = pattern candidate (configurable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from application.skill_compression.event_normalizer import NormalizedEvent

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Field weights (must sum to 1.0)
FIELD_WEIGHTS: dict[str, float] = {
    "intent": 0.30,
    "domain": 0.20,
    "constraints": 0.20,
    "format_type": 0.15,
    "scope": 0.15,
}

# Threshold for pattern candidacy (Spec: > 0.7)
SIMILARITY_THRESHOLD: float = 0.7


# ──────────────────────────────────────────────────────────────
# Result data class
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FingerprintMatch:
    """Result of comparing two NormalizedEvents.

    Attributes:
        event_a_id: ID of the first event.
        event_b_id: ID of the second event.
        similarity_score: Weighted overall similarity [0.0, 1.0].
        field_similarities: Per-field similarity scores.
        is_candidate: Whether similarity exceeds the threshold.
    """

    event_a_id: str = ""
    event_b_id: str = ""
    similarity_score: float = 0.0
    field_similarities: dict[str, float] = field(default_factory=dict)
    is_candidate: bool = False


# ──────────────────────────────────────────────────────────────
# Field comparison strategies
# ──────────────────────────────────────────────────────────────


def _exact_match(a: str, b: str) -> float:
    """Binary exact match: 1.0 if equal, 0.0 otherwise.

    Args:
        a: First string.
        b: Second string.

    Returns:
        1.0 if a == b, 0.0 otherwise.
    """
    return 1.0 if a == b else 0.0


def _prefix_match(a: str, b: str) -> float:
    """Prefix-based similarity for hierarchical labels.

    Examples:
      'create_code' vs 'create_text' = 0.5 (shared 'create_' prefix)
      'create_code' vs 'create_code' = 1.0 (exact)
      'create_code' vs 'analyze'     = 0.0 (no shared prefix)

    Args:
        a: First label.
        b: Second label.

    Returns:
        Similarity [0.0, 1.0].
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    # Split on underscore for hierarchical matching
    parts_a = a.split("_")
    parts_b = b.split("_")
    max_parts = max(len(parts_a), len(parts_b))
    if max_parts == 0:
        return 0.0

    shared = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            shared += 1
        else:
            break

    return shared / max_parts


def _dict_overlap(a: dict, b: dict) -> float:
    """Set-overlap similarity for dict-typed fields.

    Compares both keys and values. Exact key+value match counts as 1,
    key-only match counts as 0.5.

    Examples:
      {'duration': '30s'} vs {'duration': '30s'} = 1.0
      {'duration': '30s'} vs {'duration': '60s'} = 0.5 (key match)
      {'duration': '30s'} vs {'funnel': 'retargeting'} = 0.0
      {} vs {} = 1.0 (both empty = same lack of constraints)

    Args:
        a: First dict.
        b: Second dict.

    Returns:
        Similarity [0.0, 1.0].
    """
    # Both empty = equivalent (no constraints)
    if not a and not b:
        return 1.0
    # One empty, other not = different
    if not a or not b:
        return 0.0

    all_keys = set(a.keys()) | set(b.keys())
    if not all_keys:
        return 1.0

    score = 0.0
    for key in all_keys:
        if key in a and key in b:
            if a[key] == b[key]:
                score += 1.0  # Exact key+value match
            else:
                score += 0.5  # Key match, different value
        # Key in only one dict: 0.0

    return score / len(all_keys)


def _scope_similarity(a: dict, b: dict) -> float:
    """Scope similarity for project/client/context fields.

    More specific matching: project and client are weighted equally,
    empty scopes on both sides count as identical.

    Args:
        a: First scope dict.
        b: Second scope dict.

    Returns:
        Similarity [0.0, 1.0].
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.3  # One has scope, other doesn't = low similarity

    score = 0.0
    total = 0.0

    # Project (weight 0.5)
    proj_a = a.get("project", "")
    proj_b = b.get("project", "")
    total += 0.5
    if proj_a and proj_b:
        score += 0.5 if proj_a == proj_b else 0.0
    elif not proj_a and not proj_b:
        score += 0.5  # Both unset

    # Client (weight 0.5)
    client_a = a.get("client", "")
    client_b = b.get("client", "")
    total += 0.5
    if client_a and client_b:
        score += 0.5 if client_a == client_b else 0.0
    elif not client_a and not client_b:
        score += 0.5  # Both unset

    return score / total if total > 0 else 1.0


# ──────────────────────────────────────────────────────────────
# Main similarity computation
# ──────────────────────────────────────────────────────────────


def compute_similarity(
    event_a: NormalizedEvent,
    event_b: NormalizedEvent,
    *,
    weights: dict[str, float] | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> FingerprintMatch:
    """Compute weighted similarity between two NormalizedEvents.

    Language is a hard filter: if languages differ, similarity is 0.0.

    Each field is compared with a type-appropriate strategy:
      - intent: prefix match (hierarchical labels like create_code)
      - domain: exact match
      - format_type: exact match
      - constraints: dict overlap
      - scope: scope-specific similarity

    Args:
        event_a: First event.
        event_b: Second event.
        weights: Custom field weights (must sum to 1.0). Defaults to FIELD_WEIGHTS.
        threshold: Similarity threshold for candidacy.

    Returns:
        FingerprintMatch with overall and per-field scores.
    """
    if weights is None:
        weights = FIELD_WEIGHTS

    # Hard language filter
    if event_a.language != event_b.language:
        return FingerprintMatch(
            event_a_id=event_a.event_id,
            event_b_id=event_b.event_id,
            similarity_score=0.0,
            field_similarities={
                "intent": 0.0,
                "domain": 0.0,
                "format_type": 0.0,
                "constraints": 0.0,
                "scope": 0.0,
                "language": 0.0,
            },
            is_candidate=False,
        )

    # Compute per-field similarities
    field_sims: dict[str, float] = {}

    field_sims["intent"] = _prefix_match(event_a.intent, event_b.intent)
    field_sims["domain"] = _exact_match(event_a.domain, event_b.domain)
    field_sims["format_type"] = _exact_match(event_a.format_type, event_b.format_type)
    field_sims["constraints"] = _dict_overlap(event_a.constraints, event_b.constraints)
    field_sims["scope"] = _scope_similarity(event_a.scope, event_b.scope)
    field_sims["language"] = 1.0  # Passed hard filter

    # Weighted sum
    total_score = sum(
        field_sims.get(field_name, 0.0) * weight
        for field_name, weight in weights.items()
    )

    # Clamp to [0.0, 1.0]
    total_score = max(0.0, min(1.0, total_score))

    match = FingerprintMatch(
        event_a_id=event_a.event_id,
        event_b_id=event_b.event_id,
        similarity_score=total_score,
        field_similarities=field_sims,
        is_candidate=total_score > threshold,
    )

    log.debug(
        "Fingerprint match: %s vs %s = %.3f (candidate=%s)",
        event_a.event_id[:12] if event_a.event_id else "?",
        event_b.event_id[:12] if event_b.event_id else "?",
        total_score,
        match.is_candidate,
    )

    return match


def find_matches(
    target: NormalizedEvent,
    candidates: list[NormalizedEvent],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
    max_results: int = 10,
) -> list[FingerprintMatch]:
    """Find the most similar events to a target from a candidate list.

    Filters by threshold and returns sorted by similarity (descending).

    Args:
        target: The event to match against.
        candidates: List of candidate events.
        threshold: Minimum similarity for inclusion.
        max_results: Maximum number of results.

    Returns:
        Sorted list of FingerprintMatch objects (highest similarity first).
    """
    matches: list[FingerprintMatch] = []

    for candidate in candidates:
        if candidate.event_id == target.event_id:
            continue  # Skip self-match

        match = compute_similarity(target, candidate, threshold=threshold)
        if match.similarity_score > threshold:
            matches.append(match)

    # Sort by similarity descending
    matches.sort(key=lambda m: m.similarity_score, reverse=True)

    return matches[:max_results]
