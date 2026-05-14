"""Task slot domain: enum and SlotConfig for task classification.

6 slots for automatic model routing:
  CHAT, CODE, REASON, CREATIVE, QUICK, RESEARCH.

Pure domain logic, no I/O dependencies.
"""

from __future__ import annotations

from enum import Enum


class TaskSlot(str, Enum):
    """Task categories for automatic model routing.

    Each slot has a default model and a heuristic configuration.
    The str inheritance enables direct comparison with strings.
    """

    CHAT = "chat"
    CODE = "code"
    REASON = "reason"
    CREATIVE = "creative"
    QUICK = "quick"
    RESEARCH = "research"

    @classmethod
    def from_string(cls, value: str) -> TaskSlot | None:
        """Parse a string into a TaskSlot (case-insensitive).

        Args:
            value: Slot name as string.

        Returns:
            TaskSlot or None if not recognized.
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            return None

    @classmethod
    def all_names(cls) -> list[str]:
        """Return all slot names as strings."""
        return [slot.value for slot in cls]


# Priority order on score tie:
# CODE > REASON > RESEARCH > CREATIVE > QUICK > CHAT
SLOT_PRIORITY: list[TaskSlot] = [
    TaskSlot.CODE,
    TaskSlot.REASON,
    TaskSlot.RESEARCH,
    TaskSlot.CREATIVE,
    TaskSlot.QUICK,
    TaskSlot.CHAT,
]
