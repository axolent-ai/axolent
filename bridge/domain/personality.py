"""Personality-Domain-Modell.

Definiert die Struktur für System-Prompt und User-Constitution.
Reine Datenstruktur und Kombinations-Logik, keine I/O.
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PersonalityConfig:
    """Konfiguration der Bot-Persönlichkeit.

    Attributes:
        system_prompt: Hauptanweisung für Claudes Verhalten.
        user_constitution: Zusätzliche Regeln der Benutzerin.
    """

    system_prompt: str = ""
    user_constitution: str = ""

    def build_combined_prompt(self) -> str:
        """Kombiniert System-Prompt und User-Constitution zu einem String.

        Format: System-Prompt + Trennlinie + Constitution.
        Wenn nur einer vorhanden ist, wird nur dieser zurückgegeben.

        Returns:
            Kombinierter Prompt-String für --append-system-prompt.
        """
        parts: list[str] = [
            p for p in (self.system_prompt, self.user_constitution) if p
        ]

        if not parts:
            log.warning(
                "Weder System-Prompt noch User-Constitution geladen. "
                "Bot startet ohne Persönlichkeit."
            )
            return ""

        combined = "\n\n---\n\n".join(parts)
        log.info("Combined Prompt gebaut: %d Zeichen total", len(combined))
        return combined


@dataclass(frozen=True, slots=True)
class SlotInfo:
    """Belegung eines einzelnen Task-Slots.

    Attributes:
        slot_name: Name des Slots (z.B. "chat", "code").
        model_display_name: Menschenlesbarer Modell-Name.
        source: Herkunft der Belegung ("default", "user-override", "global").
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
    """Baut den Self-Awareness-Block für den System-Prompt.

    Gibt dem Modell faktische Informationen über sich selbst,
    damit es nicht halluziniert wenn der User fragt welches Modell läuft.
    Unterstützt DE und EN (alle anderen Sprachen fallen auf EN zurück).

    Args:
        model_display_name: Menschenlesbarer Modell-Name (z.B. "Opus 4.7").
        model_id: Technische Modell-ID (z.B. "claude-opus-4-7").
        task_slot: Aktiver Task-Slot (z.B. "chat", "code").
        provider: Provider-Name (z.B. "anthropic").
        all_slots: Optionale Liste aller 6 Slot-Belegungen im User-Kontext.
        lang: Sprach-Code (default: "de"). Nicht-DE fällt auf EN zurück.

    Returns:
        Self-Awareness-Block als String.
    """
    use_de = lang == "de"

    if use_de:
        label_model = "Modell"
        label_slot_heading = "[Slot-Belegung im System]"
        text_precise = "Antworte praezise mit diesen Werten wenn nach Slot-Belegungen gefragt wird."
        text_self_id = (
            "Wenn der User fragt welches Modell du nutzt, antworte mit diesen Werten. "
            "Spekuliere nicht aus Trainingsdaten."
        )
        text_no_slots = (
            "Wenn du nach anderen Slots gefragt wirst und diese Slot-Liste "
            "nicht hast, antworte ehrlich: 'Ich habe nur Information zu meinem "
            "aktiven Slot.' Spekuliere nicht."
        )
    else:
        label_model = "Current model"
        label_slot_heading = "[Slot occupancy]"
        text_precise = (
            "Answer precisely with these values when asked about slot occupancy."
        )
        text_self_id = (
            "When the user asks which model you are using, answer with these values. "
            "Do not speculate from training data."
        )
        text_no_slots = (
            "If asked about other slots and you do not have this slot list, "
            "answer honestly: 'I only have information about my active slot.' "
            "Do not speculate."
        )

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

    # Anti-Halluzination: Edge-Case ohne Slot-Liste
    if not all_slots:
        lines.append(text_no_slots)

    return "\n".join(lines)


def build_effective_prompt(base_prompt: str, language_hint: str = "") -> str:
    """Baut den effektiven System-Prompt inkl. Sprach-Override.

    Args:
        base_prompt: Der kombinierte Base-Prompt.
        language_hint: Erkannte Sprache (z.B. "en", "de").

    Returns:
        Effektiver Prompt mit optionalem Language-Override.
    """
    effective = base_prompt
    if language_hint and language_hint != "de":
        lang_instruction = (
            f"\n\n[LANGUAGE OVERRIDE] The user's message is in '{language_hint}'. "
            f"You MUST reply in '{language_hint}'. This overrides all other language rules."
        )
        effective = (effective + lang_instruction) if effective else lang_instruction
    return effective
