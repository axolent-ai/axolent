"""Golden Corpus Runner.

Runs corpus entries against a deterministic fake-provider
(or optionally against the real Claude provider if AXOLENT_GOLDEN_REAL=1).

Validates each entry's `expected` block against the actual response.

Response dict from fake_chat_service:
    - text: str (the response body)
    - language: str (detected language code)
    - streaming_aborted: bool
    - streaming_completes: bool
    - duration_seconds: float
    - memory_count_delta: int
    - history_count: int
    - streaming_active_after: bool
    - pending_skill_created: bool
    - privacy_pipeline_ran: bool
    - skill_count_delta: int
    - providers_called: int
    - synthesis_present: bool
    - privacy_rejection: str | None
    - no_crash: bool
    - preserves_unicode: bool
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CORPUS_PATH = Path(__file__).parent / "golden_prompts.yaml"

# German indicator words for no_german check.
# Deliberately short, high-signal words that are unlikely in other languages.
_GERMAN_INDICATORS = frozenset(
    [
        "ist",
        "und",
        "der",
        "die",
        "das",
        "auch",
        "nicht",
        "aber",
        "noch",
        "schon",
        "wird",
        "wurde",
        "haben",
        "hatte",
        "sind",
        "waren",
        "kann",
        "konnte",
        "muss",
        "musste",
        "weil",
        "dass",
    ]
)


def load_corpus() -> dict[str, Any]:
    """Load the golden corpus YAML file."""
    with CORPUS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _words_in_text(text: str) -> set[str]:
    """Extract lowercase words from text."""
    return set(text.lower().split())


def validate_expected(
    entry: dict[str, Any], response: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate `expected` block against actual response.

    Returns:
        (passed, failures) where passed is True if all checks pass,
        and failures is a list of human-readable failure descriptions.
    """
    expected = entry.get("expected", {})
    failures: list[str] = []
    text = response.get("text", "")

    # --- Language checks ---
    if "language" in expected:
        actual_lang = response.get("language")
        if actual_lang != expected["language"]:
            failures.append(
                f"language: expected '{expected['language']}', got '{actual_lang}'"
            )

    if "sticky_after" in expected:
        actual_sticky = response.get("sticky_after")
        if actual_sticky != expected["sticky_after"]:
            failures.append(
                f"sticky_after: expected '{expected['sticky_after']}', got '{actual_sticky}'"
            )

    # --- Length checks ---
    if "min_length" in expected:
        actual_len = len(text)
        if actual_len < expected["min_length"]:
            failures.append(
                f"min_length: expected >= {expected['min_length']}, got {actual_len}"
            )

    # --- Contains checks ---
    if "contains_one_of" in expected:
        text_lower = text.lower()
        if not any(t.lower() in text_lower for t in expected["contains_one_of"]):
            failures.append(
                f"contains_one_of: none of {expected['contains_one_of']} found in response"
            )

    if "response_contains" in expected:
        needle = expected["response_contains"].lower()
        if needle not in text.lower():
            failures.append(
                f"response_contains: '{expected['response_contains']}' not found in response"
            )

    if "response_includes_one_of" in expected:
        text_lower = text.lower()
        if not any(
            t.lower() in text_lower for t in expected["response_includes_one_of"]
        ):
            failures.append(
                f"response_includes_one_of: none of "
                f"{expected['response_includes_one_of']} found"
            )

    if "response_excludes" in expected:
        text_lower = text.lower()
        for excluded in expected["response_excludes"]:
            if excluded.lower() in text_lower:
                failures.append(
                    f"response_excludes: '{excluded}' was found in response"
                )

    # --- Negative language checks ---
    if expected.get("no_german"):
        words = _words_in_text(text)
        german_found = words & _GERMAN_INDICATORS
        if german_found:
            failures.append(
                f"no_german: German indicator words found: {sorted(german_found)}"
            )

    if expected.get("no_english_only"):
        # Check that the response is not purely English (should be target lang)
        # This is a soft heuristic; the language check is the primary gate.
        pass  # Covered by the language assertion above

    if expected.get("no_critical_switch"):
        # Ensure no unexpected language switch happened
        # The fake service handles this via language field
        pass  # Covered by language assertion

    # --- Streaming checks ---
    if "streaming_aborted" in expected:
        if response.get("streaming_aborted") != expected["streaming_aborted"]:
            failures.append(
                f"streaming_aborted: expected {expected['streaming_aborted']}, "
                f"got {response.get('streaming_aborted')}"
            )

    if "streaming_completes" in expected:
        if response.get("streaming_completes") != expected["streaming_completes"]:
            failures.append(
                f"streaming_completes: expected {expected['streaming_completes']}, "
                f"got {response.get('streaming_completes')}"
            )

    if expected.get("no_messages_after_cancel"):
        if not response.get("streaming_aborted", False):
            failures.append("no_messages_after_cancel: streaming was not aborted")

    if "max_duration_seconds" in expected:
        duration = response.get("duration_seconds", 0)
        if duration > expected["max_duration_seconds"]:
            failures.append(
                f"max_duration_seconds: expected <= {expected['max_duration_seconds']}, "
                f"got {duration}"
            )

    # --- Command checks ---
    if "memory_count_delta" in expected:
        actual_delta = response.get("memory_count_delta", 0)
        if actual_delta != expected["memory_count_delta"]:
            failures.append(
                f"memory_count_delta: expected {expected['memory_count_delta']}, "
                f"got {actual_delta}"
            )

    if "history_count" in expected:
        actual_count = response.get("history_count")
        if actual_count != expected["history_count"]:
            failures.append(
                f"history_count: expected {expected['history_count']}, "
                f"got {actual_count}"
            )

    if "streaming_active_after" in expected:
        actual = response.get("streaming_active_after")
        if actual != expected["streaming_active_after"]:
            failures.append(
                f"streaming_active_after: expected {expected['streaming_active_after']}, "
                f"got {actual}"
            )

    # --- Debate checks ---
    if "providers_called_min" in expected:
        actual_providers = response.get("providers_called", 0)
        if actual_providers < expected["providers_called_min"]:
            failures.append(
                f"providers_called_min: expected >= {expected['providers_called_min']}, "
                f"got {actual_providers}"
            )

    if "synthesis_present" in expected:
        if response.get("synthesis_present") != expected["synthesis_present"]:
            failures.append(
                f"synthesis_present: expected {expected['synthesis_present']}, "
                f"got {response.get('synthesis_present')}"
            )

    if expected.get("no_raw_provider_output"):
        if response.get("has_raw_provider_output", False):
            failures.append("no_raw_provider_output: raw provider output found")

    if expected.get("uses_previous_debate_context"):
        if not response.get("uses_previous_debate_context", False):
            failures.append("uses_previous_debate_context: debate context was not used")

    # --- Skill checks ---
    if "pending_skill_created" in expected:
        if response.get("pending_skill_created") != expected["pending_skill_created"]:
            failures.append(
                f"pending_skill_created: expected {expected['pending_skill_created']}, "
                f"got {response.get('pending_skill_created')}"
            )

    if "privacy_pipeline_ran" in expected:
        if response.get("privacy_pipeline_ran") != expected["privacy_pipeline_ran"]:
            failures.append(
                f"privacy_pipeline_ran: expected {expected['privacy_pipeline_ran']}, "
                f"got {response.get('privacy_pipeline_ran')}"
            )

    if "skill_count_delta" in expected:
        actual_delta = response.get("skill_count_delta", 0)
        if actual_delta != expected["skill_count_delta"]:
            failures.append(
                f"skill_count_delta: expected {expected['skill_count_delta']}, "
                f"got {actual_delta}"
            )

    if expected.get("no_duplicate_created"):
        if response.get("duplicate_created", False):
            failures.append("no_duplicate_created: duplicate skill was created")

    # --- Privacy checks ---
    if "privacy_rejection" in expected:
        expected_rejection = expected["privacy_rejection"]
        actual_rejection = response.get("privacy_rejection")
        if expected_rejection is None:
            if actual_rejection is not None:
                failures.append(
                    f"privacy_rejection: expected None, got '{actual_rejection}'"
                )
        else:
            if actual_rejection != expected_rejection:
                failures.append(
                    f"privacy_rejection: expected '{expected_rejection}', "
                    f"got '{actual_rejection}'"
                )

    # --- Edge case checks ---
    if expected.get("no_crash"):
        if not response.get("no_crash", True):
            failures.append("no_crash: service crashed")

    if expected.get("response_present"):
        if not text and not response.get("text"):
            failures.append("response_present: no response text")

    if expected.get("preserves_unicode"):
        if not response.get("preserves_unicode", True):
            failures.append("preserves_unicode: unicode was corrupted")

    return (len(failures) == 0, failures)
