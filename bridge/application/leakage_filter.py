"""Leakage filter: checks LLM responses for system prompt leakage (C-3).

Two-layer defense:

Layer 1 (original, C-3): Naive heuristic fingerprint matching.
    Searches for significant substrings of the system prompt
    (>= MIN_SUBSTRING_LENGTH chars) in the LLM response.

Layer 2 (T42/NEU-04 fix): Forbidden-pattern matching.
    Checks the LLM response for known internal marker strings,
    project names, and configuration references that must never
    appear in user-facing output. This catches cases where the
    model paraphrases or comments on the system prompt rather
    than reproducing it verbatim.

Design principles:
    * Conservative: prefer no false positive over being too strict
    * No regex bombs: simple substring/lower match
    * Fast: O(n*m) worst case, but with short chunks
    * Application layer: business rule, no Telegram code
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# Minimum length of a substring to qualify as a leak.
# Short fragments (<40 chars) could coincidentally appear
# in normal responses (false positives).
MIN_SUBSTRING_LENGTH: int = 40

# Step size for substring extraction from the system prompt.
# Step 1 = gapless, no boundary false negatives.
# Performance: for a 2000-char prompt, approx. 2000 chunks of 40 chars each.
# Substring-in checks in CPython are O(n+m) thanks to Boyer-Moore.
_CHUNK_STEP: int = 1

# Replacement text for detected leaks
_REDACTED_TEXT: str = "[Content redacted]"

# Generic refusal response when a leak is detected
REFUSAL_RESPONSE: str = (
    "I cannot share my internal instructions. What else can I help you with?"
)

# ---------------------------------------------------------------------------
# Layer 2: Forbidden patterns (T42 / NEU-04)
# ---------------------------------------------------------------------------
# All patterns are matched case-insensitively against the normalized response.
# Each entry is a tuple: (pattern, category) for logging context.

_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    # System prompt block markers (old bracket style, kept for safety)
    ("language lock", "marker"),
    ("diacritic rule", "marker"),
    ("style rule", "marker"),
    ("format contract", "marker"),
    ("privacy rule", "marker"),
    ("security block", "marker"),
    ("self-awareness", "marker"),
    # Project internals
    ("claude.md", "project_ref"),
    ("axolent ai project", "project_ref"),
    ("axolent project", "project_ref"),
    ("the project we're working on", "project_ref"),
    ("the project we are working on", "project_ref"),
    # Development tooling references
    ("import-linter", "dev_tooling"),
    ("pre-commit hooks", "dev_tooling"),
    ("according to project conventions", "dev_tooling"),
    ("as per claude.md", "dev_tooling"),
    ("in this project we", "dev_tooling"),
    ("production code conventions", "dev_tooling"),
    ("english-only policy", "dev_tooling"),
    # Meta-commentary about injected instructions
    ("injected system-level", "meta_commentary"),
    ("prompt-injection-muster", "meta_commentary"),
    ("prompt-injection pattern", "meta_commentary"),
    ("prompt injection pattern", "meta_commentary"),
    ("injizierte system-level-befehle", "meta_commentary"),
    ("injected system-level commands", "meta_commentary"),
    ("authoritative systembefehle", "meta_commentary"),
    ("authoritative system commands", "meta_commentary"),
]


def _extract_fingerprints(system_prompt: str) -> list[str]:
    """Extract overlapping substrings from the system prompt.

    Normalizes text (lowercase, whitespace reduction) and extracts
    chunks of length MIN_SUBSTRING_LENGTH with step size _CHUNK_STEP.

    Args:
        system_prompt: The full system prompt.

    Returns:
        List of normalized substring fingerprints.
    """
    normalized = " ".join(system_prompt.lower().split())
    if len(normalized) < MIN_SUBSTRING_LENGTH:
        return []

    fingerprints: list[str] = []
    for i in range(0, len(normalized) - MIN_SUBSTRING_LENGTH + 1, _CHUNK_STEP):
        chunk = normalized[i : i + MIN_SUBSTRING_LENGTH]
        fingerprints.append(chunk)
    return fingerprints


def check_for_forbidden_patterns(response: str) -> Optional[str]:
    """Check whether the LLM response contains forbidden internal patterns.

    Layer 2 defense (T42/NEU-04): catches cases where the model
    paraphrases, comments on, or meta-references internal system
    prompt structures rather than reproducing them verbatim.

    Args:
        response: The LLM response to check.

    Returns:
        None if no forbidden pattern was detected.
        REFUSAL_RESPONSE if a forbidden pattern was detected.
    """
    if not response:
        return None

    normalized = " ".join(response.lower().split())

    for pattern, category in _FORBIDDEN_PATTERNS:
        if pattern in normalized:
            log.warning(
                "Forbidden pattern leakage detected: category=%s, "
                "pattern=%r. Response replaced with refusal.",
                category,
                pattern,
            )
            return REFUSAL_RESPONSE

    return None


def check_for_system_prompt_leakage(
    response: str,
    system_prompt: str,
    exclude_texts: Optional[list[str]] = None,
) -> Optional[str]:
    """Check whether the LLM response contains parts of the system prompt.

    Two-layer check:
    1. Forbidden-pattern matching (fast, catches paraphrasing/meta-commentary)
    2. Fingerprint matching (catches verbatim reproduction)

    Args:
        response: The LLM response to check.
        system_prompt: The active system prompt.
        exclude_texts: Optional list of texts whose fingerprints should be
            excluded from leak detection. Used to whitelist memory content
            that was injected into the system prompt: when the LLM correctly
            cites its own memory, that is NOT a leak.

    Returns:
        None if no leak was detected.
        REFUSAL_RESPONSE if a leak was detected.
    """
    if not response:
        return None

    # Layer 2: forbidden patterns (checked first, faster)
    forbidden_result = check_for_forbidden_patterns(response)
    if forbidden_result is not None:
        return forbidden_result

    # Layer 1: fingerprint matching (original C-3 logic)
    if not system_prompt:
        return None

    fingerprints = _extract_fingerprints(system_prompt)
    if not fingerprints:
        return None

    # Build exclusion set from memory/whitelist texts
    excluded_fps: set[str] = set()
    if exclude_texts:
        for text in exclude_texts:
            if text:
                excluded_fps.update(_extract_fingerprints(text))

    normalized_response = " ".join(response.lower().split())

    for fp in fingerprints:
        if fp in excluded_fps:
            continue  # Fingerprint from whitelisted text (e.g. memory)
        if fp in normalized_response:
            log.warning(
                "System prompt leakage detected: %d-char match found. "
                "Response replaced with refusal.",
                len(fp),
            )
            return REFUSAL_RESPONSE

    return None
