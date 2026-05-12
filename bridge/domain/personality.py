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


def build_self_awareness_block(
    model_display_name: str,
    model_id: str,
    task_slot: str,
    provider: str,
) -> str:
    """Baut den Self-Awareness-Block für den System-Prompt.

    Gibt dem Modell faktische Informationen über sich selbst,
    damit es nicht halluziniert wenn der User fragt welches Modell läuft.

    Args:
        model_display_name: Menschenlesbarer Modell-Name (z.B. "Opus 4.7").
        model_id: Technische Modell-ID (z.B. "claude-opus-4-7").
        task_slot: Aktiver Task-Slot (z.B. "chat", "code").
        provider: Provider-Name (z.B. "anthropic").

    Returns:
        Self-Awareness-Block als String.
    """
    return (
        "[SELF-AWARENESS]\n"
        f"Modell: {model_display_name} ({model_id})\n"
        f"Slot: {task_slot}\n"
        f"Provider: {provider}\n"
        "Wenn der User fragt welches Modell du nutzt, antworte mit diesen Werten. "
        "Spekuliere nicht aus Trainingsdaten."
    )


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
