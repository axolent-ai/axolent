"""Task-Slot Domain: Enum und SlotConfig für Task-Klassifikation.

6 Slots für automatisches Modell-Routing:
  CHAT, CODE, REASON, CREATIVE, QUICK, RESEARCH.

Reine Domain-Logik, keine I/O-Abhaengigkeiten.
"""

from __future__ import annotations

from enum import Enum


class TaskSlot(str, Enum):
    """Task-Kategorien für automatisches Modell-Routing.

    Jeder Slot hat ein Default-Modell und eine Heuristik-Konfiguration.
    Die str-Vererbung ermoeglicht direkten Vergleich mit Strings.
    """

    CHAT = "chat"
    CODE = "code"
    REASON = "reason"
    CREATIVE = "creative"
    QUICK = "quick"
    RESEARCH = "research"

    @classmethod
    def from_string(cls, value: str) -> TaskSlot | None:
        """Parst einen String in einen TaskSlot (case-insensitive).

        Args:
            value: Slot-Name als String.

        Returns:
            TaskSlot oder None wenn nicht erkannt.
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            return None

    @classmethod
    def all_names(cls) -> list[str]:
        """Gibt alle Slot-Namen als Strings zurück."""
        return [slot.value for slot in cls]


# Prioritätsreihenfolge bei Score-Gleichstand:
# CODE > REASON > RESEARCH > CREATIVE > QUICK > CHAT
SLOT_PRIORITY: list[TaskSlot] = [
    TaskSlot.CODE,
    TaskSlot.REASON,
    TaskSlot.RESEARCH,
    TaskSlot.CREATIVE,
    TaskSlot.QUICK,
    TaskSlot.CHAT,
]
