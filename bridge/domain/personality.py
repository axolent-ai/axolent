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
