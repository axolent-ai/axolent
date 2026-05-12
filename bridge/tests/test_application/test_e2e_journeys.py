"""E2E Journey-Tests J20-J23: TaskRouter Auto-Routing Verifikation.

Verifiziert die 4 Auto-Routing-Journeys aus docs/e2e-test-checklist.md
als ausführbare Unit-Tests. Nutzt die Produktions-YAML-Konfiguration
(keine hardcoded Fixtures), damit Regressionen in task_slots.yaml
sofort auffallen.

J20: Code-Frage -> CODE-Slot
J21: Analyse-Frage (>50 Wörter) -> REASON-Slot
J22: "Übersetze: Guten Morgen" -> QUICK-Slot
J23: Standard-Frage -> CHAT-Slot (Fallback)
"""

from __future__ import annotations

import pytest

from application.task_router import TaskRouter, load_slot_configs
from domain.task_slot import TaskSlot


@pytest.fixture
def production_router() -> TaskRouter:
    """TaskRouter mit der echten Produktions-YAML (nicht Test-Fixture)."""
    configs = load_slot_configs()  # Liest bridge/config/task_slots.yaml
    return TaskRouter(slot_configs=configs)


class TestJourneyJ20CodeSlot:
    """J20: Code-Frage mit Code-Block -> CODE-Slot."""

    def test_code_block_with_debug_keyword(self, production_router: TaskRouter) -> None:
        """Code-Block (```) + Keyword 'debugge' ergibt CODE-Slot."""
        prompt = (
            "Debugge diesen Code:\n```python\ndef add(a, b):\n    return a - b\n```"
        )
        result = production_router.classify(prompt)
        assert result.slot == TaskSlot.CODE, (
            f"J20 FAIL: Erwartet CODE, bekam {result.slot.value} "
            f"(score={result.score}, patterns={result.matched_patterns}, "
            f"keywords={result.matched_keywords})"
        )
        # Code-Block muss als Pattern gematcht sein
        assert "```" in result.matched_patterns

    def test_code_slot_resolves_to_opus(self, production_router: TaskRouter) -> None:
        """CODE-Slot Default muss kanonisch claude-opus-4-7 sein."""
        model = production_router.resolve_model(user_id=1, slot=TaskSlot.CODE)
        assert model == "claude-opus-4-7"


class TestJourneyJ21ReasonSlot:
    """J21: Analyse-Frage mit >50 Wörtern -> REASON-Slot.

    Der Prompt in der E2E-Checkliste wurde auf >50 Wörter verlängert,
    da der REASON-Slot min_word_count=50 erfordert.
    """

    def test_analysis_prompt_routes_to_reason(
        self, production_router: TaskRouter
    ) -> None:
        """Langer Analyse-Prompt mit Reason-Keywords ergibt REASON-Slot."""
        # Prompt aus e2e-test-checklist.md (27 Wörter).
        # Wir verwenden den verlängerten Prompt, da min_word_count=50
        # in task_slots.yaml konfiguriert ist.
        prompt = (
            "Analysiere die Vor- und Nachteile von Remote-Arbeit im Vergleich "
            "zu Büroarbeit. Berechne die durchschnittlichen Kosten für einen "
            "Arbeitgeber in beiden Szenarien und leite daraus eine Empfehlung ab. "
            "Berücksichtige dabei Faktoren wie Miete, Nebenkosten, Pendlerpauschale, "
            "Produktivität, Mitarbeiterzufriedenheit und die langfristige Strategie "
            "des Unternehmens im Hinblick auf flexible Arbeitsmodelle und deren "
            "Auswirkung auf Talentakquise."
        )
        word_count = len(prompt.split())
        assert word_count >= 50, f"Prompt hat nur {word_count} Wörter, braucht >50"

        result = production_router.classify(prompt)
        assert result.slot == TaskSlot.REASON, (
            f"J21 FAIL: Erwartet REASON, bekam {result.slot.value} "
            f"(score={result.score}, keywords={result.matched_keywords}, "
            f"word_count={word_count})"
        )

    def test_reason_slot_resolves_to_opus(self, production_router: TaskRouter) -> None:
        """REASON-Slot Default muss kanonisch claude-opus-4-7 sein."""
        model = production_router.resolve_model(user_id=1, slot=TaskSlot.REASON)
        assert model == "claude-opus-4-7"


class TestJourneyJ22QuickSlot:
    """J22: 'Übersetze: Guten Morgen' -> QUICK-Slot."""

    def test_translate_prompt_routes_to_quick(
        self, production_router: TaskRouter
    ) -> None:
        """Kurzer Übersetzungs-Prompt ergibt QUICK-Slot."""
        prompt = "Übersetze: Guten Morgen"
        result = production_router.classify(prompt)
        assert result.slot == TaskSlot.QUICK, (
            f"J22 FAIL: Erwartet QUICK, bekam {result.slot.value} "
            f"(score={result.score}, keywords={result.matched_keywords})"
        )

    def test_uebersetze_variant_also_routes(
        self, production_router: TaskRouter
    ) -> None:
        """'Übersetz das ins Englische' ergibt auch QUICK-Slot."""
        prompt = "Übersetz das ins Englische"
        result = production_router.classify(prompt)
        assert result.slot == TaskSlot.QUICK

    def test_quick_slot_resolves_to_haiku(self, production_router: TaskRouter) -> None:
        """QUICK-Slot Default muss kanonisch claude-haiku-... sein."""
        model = production_router.resolve_model(user_id=1, slot=TaskSlot.QUICK)
        assert "haiku" in model


class TestJourneyJ23ChatFallback:
    """J23: Standard-Frage -> CHAT-Slot (Fallback)."""

    def test_greeting_routes_to_chat(self, production_router: TaskRouter) -> None:
        """Einfache Grußfrage fällt auf CHAT-Slot zurück."""
        prompt = "Wie geht es dir heute?"
        result = production_router.classify(prompt)
        assert result.slot == TaskSlot.CHAT, (
            f"J23 FAIL: Erwartet CHAT, bekam {result.slot.value} "
            f"(score={result.score}, keywords={result.matched_keywords})"
        )

    def test_chat_slot_resolves_to_sonnet(self, production_router: TaskRouter) -> None:
        """CHAT-Slot Default muss kanonisch claude-sonnet-4-6 sein."""
        model = production_router.resolve_model(user_id=1, slot=TaskSlot.CHAT)
        assert model == "claude-sonnet-4-6"
