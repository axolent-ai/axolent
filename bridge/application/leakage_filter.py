"""Leakage filter: checks LLM responses for system prompt leakage (C-3).

Naive heuristic: searches for significant substrings of the system prompt
(>= MIN_SUBSTRING_LENGTH chars) in the LLM response. If found,
a sanitized response is returned.

Design principles:
    * Conservative: prefer no false positive over being too strict
    * No regex bombs: simple substring match
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


def check_for_system_prompt_leakage(response: str, system_prompt: str) -> Optional[str]:
    """Check whether the LLM response contains parts of the system prompt.

    Compares normalized substrings of the system prompt against the
    normalized response. On match, the refusal response is returned.

    Args:
        response: The LLM response to check.
        system_prompt: The active system prompt.

    Returns:
        None if no leak was detected.
        REFUSAL_RESPONSE if a leak was detected.
    """
    if not response or not system_prompt:
        return None

    fingerprints = _extract_fingerprints(system_prompt)
    if not fingerprints:
        return None

    normalized_response = " ".join(response.lower().split())

    for fp in fingerprints:
        if fp in normalized_response:
            log.warning(
                "System prompt leakage detected: %d-char match found. "
                "Response replaced with refusal.",
                len(fp),
            )
            return REFUSAL_RESPONSE

    return None
