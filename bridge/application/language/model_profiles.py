"""Model-Adherence-Profile: enforcement levels per model.

Different models have different instruction-following reliability.
This module maps model IDs to enforcement configurations that
determine how aggressively the Language Control Plane operates.

Enforcement levels:
    - "normal": Model reliably follows language instructions.
      Verifier runs but repair is not triggered unless confidence
      is very high that language is wrong.
    - "strict": Model sometimes drifts. Verifier always runs,
      repair triggered on failure.
    - "strict_with_verify": Model frequently drifts (smaller/local).
      Verifier + repair mandatory, StreamGuard active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EnforcementLevel = Literal["normal", "strict", "strict_with_verify"]


@dataclass(frozen=True, slots=True)
class ModelAdherenceProfile:
    """Language enforcement configuration for a specific model.

    Attributes:
        model_id: Model identifier (or "default" for fallback).
        enforcement_level: How aggressively language is enforced.
        verify_required: Whether ResponseLanguageVerifier must run.
        repair_enabled: Whether RepairService may be triggered on failure.
        stream_guard_enabled: Whether StreamGuard early-abort is active.
        description: Human-readable description of the profile.
    """

    model_id: str
    enforcement_level: EnforcementLevel
    verify_required: bool
    repair_enabled: bool
    stream_guard_enabled: bool
    description: str


# Registry of known model profiles.
# Order: most specific first. Prefix matching is used for model families.
_PROFILES: dict[str, ModelAdherenceProfile] = {
    "claude-opus-4-7": ModelAdherenceProfile(
        model_id="claude-opus-4-7",
        enforcement_level="normal",
        verify_required=False,
        repair_enabled=False,
        stream_guard_enabled=False,
        description="Opus 4.7: very instruction-followy, minimal enforcement needed",
    ),
    "claude-opus-4-6": ModelAdherenceProfile(
        model_id="claude-opus-4-6",
        enforcement_level="normal",
        verify_required=False,
        repair_enabled=False,
        stream_guard_enabled=False,
        description="Opus 4.6: very instruction-followy, minimal enforcement needed",
    ),
    "claude-sonnet-4-6": ModelAdherenceProfile(
        model_id="claude-sonnet-4-6",
        enforcement_level="normal",
        verify_required=False,
        repair_enabled=False,
        stream_guard_enabled=False,
        description="Sonnet 4.6: reliable instruction following",
    ),
    "claude-sonnet-4-5": ModelAdherenceProfile(
        model_id="claude-sonnet-4-5",
        enforcement_level="normal",
        verify_required=False,
        repair_enabled=False,
        stream_guard_enabled=False,
        description="Sonnet 4.5: reliable instruction following",
    ),
    "claude-haiku-4-5": ModelAdherenceProfile(
        model_id="claude-haiku-4-5",
        enforcement_level="strict_with_verify",
        verify_required=True,
        repair_enabled=True,
        stream_guard_enabled=True,
        description="Haiku 4.5: smaller model, less reliable language adherence",
    ),
    "gemini": ModelAdherenceProfile(
        model_id="gemini",
        enforcement_level="strict",
        verify_required=True,
        repair_enabled=True,
        stream_guard_enabled=False,
        description="Gemini models: occasional language drift",
    ),
    "mistral": ModelAdherenceProfile(
        model_id="mistral",
        enforcement_level="strict",
        verify_required=True,
        repair_enabled=True,
        stream_guard_enabled=False,
        description="Mistral models: occasional language drift",
    ),
    "codex": ModelAdherenceProfile(
        model_id="codex",
        enforcement_level="strict",
        verify_required=True,
        repair_enabled=True,
        stream_guard_enabled=False,
        description="OpenAI Codex: code-focused, may default to English",
    ),
    "llama": ModelAdherenceProfile(
        model_id="llama",
        enforcement_level="strict_with_verify",
        verify_required=True,
        repair_enabled=True,
        stream_guard_enabled=True,
        description="Llama local models: frequent language drift in non-English",
    ),
    "default": ModelAdherenceProfile(
        model_id="default",
        enforcement_level="strict",
        verify_required=True,
        repair_enabled=True,
        stream_guard_enabled=False,
        description="Default profile for unknown models: strict enforcement",
    ),
}


def get_profile(model_id: str | None) -> ModelAdherenceProfile:
    """Get the adherence profile for a model.

    Uses exact match first, then prefix match (e.g. "llama-3.1-8b"
    matches "llama"), then falls back to "default".

    Args:
        model_id: Model identifier string (may be None for default).

    Returns:
        Matching ModelAdherenceProfile.
    """
    if not model_id:
        return _PROFILES["default"]

    # Exact match
    if model_id in _PROFILES:
        return _PROFILES[model_id]

    # Prefix match (longest prefix wins)
    best_match: ModelAdherenceProfile | None = None
    best_prefix_len = 0

    for key, profile in _PROFILES.items():
        if key == "default":
            continue
        if model_id.startswith(key) and len(key) > best_prefix_len:
            best_match = profile
            best_prefix_len = len(key)

    if best_match is not None:
        return best_match

    return _PROFILES["default"]


def list_profiles() -> list[ModelAdherenceProfile]:
    """List all registered profiles.

    Returns:
        List of all ModelAdherenceProfile instances.
    """
    return list(_PROFILES.values())
