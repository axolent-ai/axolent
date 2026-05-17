"""Personality domain model.

Defines the structure for system prompt and user constitution.
Pure data structure and combination logic, no I/O.

Since Phase 2: self-awareness block uses i18n t() for all 20 languages
instead of hardcoded DE/EN branches.
"""

import logging
from dataclasses import dataclass

from i18n import t

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PersonalityConfig:
    """Configuration for the bot personality.

    Attributes:
        system_prompt: Main instruction for Claude's behavior.
        user_constitution: Additional rules from the user.
    """

    system_prompt: str = ""
    user_constitution: str = ""

    def build_combined_prompt(self) -> str:
        """Combine system prompt and user constitution into one string.

        Format: system prompt + separator + constitution.
        If only one is present, only that one is returned.

        Returns:
            Combined prompt string for --append-system-prompt.
        """
        parts: list[str] = [
            p for p in (self.system_prompt, self.user_constitution) if p
        ]

        if not parts:
            log.warning(
                "Neither system prompt nor user constitution loaded. "
                "Bot starts without personality."
            )
            return ""

        combined = "\n\n---\n\n".join(parts)
        log.info("Combined prompt built: %d chars total", len(combined))
        return combined


@dataclass(frozen=True, slots=True)
class SlotInfo:
    """Occupancy of a single task slot.

    Attributes:
        slot_name: Name of the slot (e.g. "chat", "code").
        model_display_name: Human-readable model name.
        source: Origin of the occupancy ("default", "user-override", "global").
    """

    slot_name: str
    model_display_name: str
    source: str = "default"


def build_self_awareness_block(
    model_display_name: str,
    model_id: str,
    task_slot: str,
    provider: str,
    all_slots: list[SlotInfo] | None = None,
    lang: str = "de",
) -> str:
    """Build the self-awareness block for the system prompt.

    Gives the model factual information about itself so it does not
    hallucinate when the user asks which model is running.
    Since Phase 2: supports all 20 languages via i18n t() system.
    No more hardcoded DE/EN branches.

    Args:
        model_display_name: Human-readable model name (e.g. "Opus 4.7").
        model_id: Technical model ID (e.g. "claude-opus-4-7").
        task_slot: Active task slot (e.g. "chat", "code").
        provider: Provider name (e.g. "anthropic").
        all_slots: Optional list of all 6 slot occupancies in the user context.
        lang: Language code (default: "de"). All 20 languages supported.

    Returns:
        Self-awareness block as string.
    """
    label_model = t("self_awareness.label_model", lang)
    label_slot_heading = t("self_awareness.slot_heading", lang)
    text_precise = t("self_awareness.text_precise", lang)
    text_self_id = t("self_awareness.text_self_id", lang)
    text_no_slots = t("self_awareness.text_no_slots", lang)

    lines = [
        "[SELF-AWARENESS]",
        f"{label_model}: {model_display_name} ({model_id})",
        f"Slot: {task_slot}",
        f"Provider: {provider}",
    ]

    if all_slots:
        lines.append("")
        lines.append(label_slot_heading)
        for slot_info in all_slots:
            lines.append(
                f"- {slot_info.slot_name.upper()}: "
                f"{slot_info.model_display_name} ({slot_info.source})"
            )
        lines.append("")
        lines.append(text_precise)

    lines.append(text_self_id)

    # Anti-hallucination: edge case without slot list
    if not all_slots:
        lines.append(text_no_slots)

    return "\n".join(lines)


def build_effective_prompt(base_prompt: str, language_hint: str = "") -> str:
    """Build the effective system prompt including language lock.

    Always injects a [LANGUAGE LOCK] block for ALL languages (including "de")
    to ensure the LLM never defaults to training-bias language.

    Args:
        base_prompt: The combined base prompt.
        language_hint: Detected language (e.g. "en", "de").

    Returns:
        Effective prompt with language lock and diacritic hint.
    """
    effective = base_prompt

    if language_hint:
        lang_instruction = (
            f"\n\n[LANGUAGE LOCK] You MUST reply in '{language_hint}'. "
            f"This overrides any default language behavior. "
            f"Do not switch languages mid-response."
        )
        effective = (effective + lang_instruction) if effective else lang_instruction

    # Append diacritic instruction for the detected language
    diacritic_hint = _build_diacritic_hint(language_hint or "de")
    if diacritic_hint:
        effective = (effective + diacritic_hint) if effective else diacritic_hint

    return effective


# Diacritic instructions per language, appended to system prompt
# so the LLM is primed to use correct characters from the start.
_DIACRITIC_HINTS: dict[str, str] = {
    "de": (
        "\n\n[DIACRITIC RULE] When responding in German, ALWAYS use real "
        "umlauts and eszett: ä, ö, ü, Ä, Ö, Ü, ß. "
        "NEVER use ASCII substitutes (ae, oe, ue, ss). "
        "Examples: 'für' not 'fuer', 'über' not 'ueber', 'größer' not 'groesser'."
    ),
    "fr": (
        "\n\n[DIACRITIC RULE] When responding in French, ALWAYS use real "
        "accents and cedilla: é, è, ê, ë, à, â, ç, ô, û, î, ï, ù. "
        "NEVER omit them. "
        "Examples: 'être' not 'etre', 'français' not 'francais'."
    ),
    "es": (
        "\n\n[DIACRITIC RULE] When responding in Spanish, ALWAYS use ñ "
        "and accent marks: á, é, í, ó, ú, ñ. NEVER substitute with "
        "plain ASCII. "
        "Examples: 'español' not 'espanol', 'también' not 'tambien'."
    ),
    "it": (
        "\n\n[DIACRITIC RULE] When responding in Italian, ALWAYS use "
        "proper accented vowels: à, è, é, ì, ò, ù. "
        "Examples: 'città' not 'citta', 'perché' not 'perche'."
    ),
    "pt": (
        "\n\n[DIACRITIC RULE] When responding in Portuguese, ALWAYS use "
        "tildes, cedilla, and accents: ã, õ, ç, á, é, í, ó, ú, â, ê, ô. "
        "Examples: 'não' not 'nao', 'você' not 'voce'."
    ),
}


def _build_diacritic_hint(language: str) -> str:
    """Build a diacritic instruction for the system prompt.

    Args:
        language: ISO 639-1 language code.

    Returns:
        Instruction string, empty if no hint for the language.
    """
    return _DIACRITIC_HINTS.get(language, "")
