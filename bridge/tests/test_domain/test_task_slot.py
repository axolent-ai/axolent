"""Tests fuer TaskSlot Enum und Hilfsfunktionen."""

from __future__ import annotations


from domain.task_slot import SLOT_PRIORITY, TaskSlot


class TestTaskSlotEnum:
    """Tests fuer TaskSlot Enum Grundfunktionalitaet."""

    def test_all_six_slots_exist(self) -> None:
        """Alle 6 Slots sind definiert."""
        assert len(TaskSlot) == 6

    def test_slot_values(self) -> None:
        """Slot-Werte sind lowercase Strings."""
        assert TaskSlot.CHAT.value == "chat"
        assert TaskSlot.CODE.value == "code"
        assert TaskSlot.REASON.value == "reason"
        assert TaskSlot.CREATIVE.value == "creative"
        assert TaskSlot.QUICK.value == "quick"
        assert TaskSlot.RESEARCH.value == "research"

    def test_string_comparison(self) -> None:
        """TaskSlot ist str-basiert, direkter Vergleich funktioniert."""
        assert TaskSlot.CHAT == "chat"
        assert TaskSlot.CODE == "code"

    def test_from_string_valid(self) -> None:
        """from_string erkennt gueltige Slots."""
        assert TaskSlot.from_string("chat") == TaskSlot.CHAT
        assert TaskSlot.from_string("CODE") == TaskSlot.CODE
        assert TaskSlot.from_string("  reason  ") == TaskSlot.REASON
        assert TaskSlot.from_string("Creative") == TaskSlot.CREATIVE

    def test_from_string_invalid(self) -> None:
        """from_string gibt None fuer ungueltige Slots."""
        assert TaskSlot.from_string("unknown") is None
        assert TaskSlot.from_string("") is None
        assert TaskSlot.from_string("debate") is None

    def test_all_names(self) -> None:
        """all_names gibt alle Slot-Werte als Strings."""
        names = TaskSlot.all_names()
        assert len(names) == 6
        assert "chat" in names
        assert "code" in names
        assert "research" in names


class TestSlotPriority:
    """Tests fuer die Prioritaetsreihenfolge."""

    def test_priority_order(self) -> None:
        """CODE hat hoechste Prioritaet, CHAT niedrigste."""
        assert SLOT_PRIORITY[0] == TaskSlot.CODE
        assert SLOT_PRIORITY[-1] == TaskSlot.CHAT

    def test_priority_length(self) -> None:
        """Prioritaetsliste enthaelt alle 6 Slots."""
        assert len(SLOT_PRIORITY) == 6

    def test_all_slots_in_priority(self) -> None:
        """Alle Slots sind in der Prioritaetsliste."""
        priority_set = set(SLOT_PRIORITY)
        slot_set = set(TaskSlot)
        assert priority_set == slot_set
