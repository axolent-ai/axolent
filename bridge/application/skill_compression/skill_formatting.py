"""Skill formatting utilities shared between application and presentation layers.

This module lives in the application layer so that chat_service (application)
can import it without violating hexagonal architecture boundaries.
Presentation modules re-export these symbols for backward compatibility.

Functions:
  derive_skill_name: Human-readable skill name from hypothesis fields.
  format_skill_indicator: Append skill-applied indicator to bot response.
"""

from __future__ import annotations

from application.skill_compression.hypothesis_storage import Hypothesis


def derive_skill_name(hypothesis: Hypothesis) -> str:
    """Derive a human-readable skill name from hypothesis fields.

    IC-UI-3: When user hasn't assigned a name, derive from type + claim.
    Truncates to 40 chars max for compact display.

    Args:
        hypothesis: The hypothesis to name.

    Returns:
        Short, human-readable skill name.
    """
    claim = hypothesis.claim.strip()
    if not claim:
        return f"{hypothesis.type} (unnamed)"

    # Use first sentence or first 40 chars of claim
    for sep in (".", ":", ";", ","):
        idx = claim.find(sep)
        if 0 < idx <= 40:
            return claim[:idx]

    if len(claim) <= 40:
        return claim

    # Truncate at word boundary
    truncated = claim[:37]
    last_space = truncated.rfind(" ")
    if last_space > 15:
        return truncated[:last_space] + "..."
    return truncated + "..."


def format_skill_indicator(
    hypothesis: Hypothesis,
    response_text: str,
) -> str:
    """Append skill application indicator to bot response.

    HC-UI-2 [BLOCKER]: Indicator MUST appear on every auto-apply.
    Only shown for 'active' status (auto-apply), NOT for 'confirmed'
    (where "Ask Before" applies and user already knows).

    Format:
        [Bot response]
        -----
        Skill 'Drehkonzepte v2' angewendet

    Args:
        hypothesis: The applied hypothesis.
        response_text: The bot's response text.

    Returns:
        Response with indicator appended.
    """
    name = derive_skill_name(hypothesis)
    version_tag = f" v{hypothesis.version}" if hypothesis.version > 1 else ""
    indicator = f"Skill '{name}{version_tag}' angewendet"

    return f"{response_text}\n─────\n{indicator}"
