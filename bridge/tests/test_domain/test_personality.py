"""Tests for domain.personality: personality prompt combination.

Tests build_combined_prompt and build_effective_prompt with language override.
"""

import pytest

from domain.personality import (
    PersonalityConfig,
    SlotInfo,
    build_effective_prompt,
    build_self_awareness_block,
)


@pytest.mark.unit
class TestPersonalityConfig:
    """PersonalityConfig Kombinations-Logik."""

    def test_personality_combined_prompt_format(self) -> None:
        """System-Prompt und Constitution werden mit Trennlinie kombiniert."""
        config = PersonalityConfig(
            system_prompt="Du bist Axolent.",
            user_constitution="Antworte immer freundlich.",
        )
        result = config.build_combined_prompt()
        assert "Du bist Axolent." in result
        assert "Antworte immer freundlich." in result
        assert "---" in result

    def test_only_system_prompt(self) -> None:
        """Wenn nur System-Prompt vorhanden, keine Trennlinie."""
        config = PersonalityConfig(system_prompt="Nur System.", user_constitution="")
        result = config.build_combined_prompt()
        assert result == "Nur System."
        assert "---" not in result

    def test_only_constitution(self) -> None:
        """Wenn nur Constitution vorhanden, wird nur diese zurückgegeben."""
        config = PersonalityConfig(system_prompt="", user_constitution="Nur Regeln.")
        result = config.build_combined_prompt()
        assert result == "Nur Regeln."

    def test_both_empty_returns_empty(self) -> None:
        """Ohne beides wird ein leerer String zurückgegeben."""
        config = PersonalityConfig(system_prompt="", user_constitution="")
        result = config.build_combined_prompt()
        assert result == ""


@pytest.mark.unit
class TestBuildEffectivePrompt:
    """build_effective_prompt mit LANGUAGE LOCK fuer alle Sprachen."""

    def test_language_lock_for_german(self) -> None:
        """Bei 'de' wird LANGUAGE LOCK angehaengt (symmetrisch wie alle Sprachen)."""
        result = build_effective_prompt("Base prompt.", "de")
        assert "[LANGUAGE LOCK]" in result
        assert "'de'" in result
        assert result.startswith("Base prompt.")
        assert "[DIACRITIC RULE]" in result

    def test_language_lock_for_english(self) -> None:
        """Bei 'en' wird ein LANGUAGE LOCK Block angehaengt."""
        result = build_effective_prompt("Base prompt.", "en")
        assert "[LANGUAGE LOCK]" in result
        assert "'en'" in result

    def test_language_lock_for_spanish(self) -> None:
        """Beliebige Sprache loest LANGUAGE LOCK und DIACRITIC RULE aus."""
        result = build_effective_prompt("Base.", "es")
        assert "[LANGUAGE LOCK]" in result
        assert "'es'" in result
        assert "[DIACRITIC RULE]" in result

    def test_empty_language_hint_no_lock(self) -> None:
        """Leerer Language-Hint fuegt keinen LANGUAGE LOCK an, aber DE DIACRITIC RULE."""
        result = build_effective_prompt("Base.", "")
        assert "[LANGUAGE LOCK]" not in result
        assert result.startswith("Base.")
        assert "[DIACRITIC RULE]" in result

    def test_empty_base_prompt_with_language(self) -> None:
        """Auch ohne Base-Prompt wird LANGUAGE LOCK gesetzt."""
        result = build_effective_prompt("", "fr")
        assert "[LANGUAGE LOCK]" in result
        assert "'fr'" in result
        assert "[DIACRITIC RULE]" in result


@pytest.mark.unit
class TestBuildSelfAwarenessBlock:
    """build_self_awareness_block: Modell-Info fuer System-Prompt."""

    def test_contains_all_fields(self) -> None:
        """Block enthaelt Modell-Name, ID, Slot und Provider."""
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="code",
            provider="anthropic",
        )
        assert "[SELF-AWARENESS]" in block
        assert "Opus 4.7" in block
        assert "claude-opus-4-7" in block
        assert "code" in block
        assert "anthropic" in block

    def test_contains_anti_hallucination_instruction(self) -> None:
        """Block enthaelt Anweisung nicht zu halluzinieren."""
        block = build_self_awareness_block(
            model_display_name="Sonnet 4.6",
            model_id="claude-sonnet-4-6",
            task_slot="chat",
            provider="anthropic",
        )
        assert "Spekuliere nicht" in block

    def test_different_models_produce_different_blocks(self) -> None:
        """Verschiedene Modelle produzieren verschiedene Blocks."""
        block_opus = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
        )
        block_sonnet = build_self_awareness_block(
            model_display_name="Sonnet 4.6",
            model_id="claude-sonnet-4-6",
            task_slot="chat",
            provider="anthropic",
        )
        assert block_opus != block_sonnet
        assert "Opus 4.7" in block_opus
        assert "Sonnet 4.6" in block_sonnet

    def test_all_slots_included_in_block(self) -> None:
        """Block enthaelt alle 6 Slot-Belegungen wenn uebergeben."""
        all_slots = [
            SlotInfo(slot_name="chat", model_display_name="Opus 4.7", source="global"),
            SlotInfo(slot_name="code", model_display_name="Opus 4.7", source="global"),
            SlotInfo(
                slot_name="reason", model_display_name="Opus 4.7", source="global"
            ),
            SlotInfo(
                slot_name="creative", model_display_name="Opus 4.7", source="global"
            ),
            SlotInfo(slot_name="quick", model_display_name="Opus 4.7", source="global"),
            SlotInfo(
                slot_name="research", model_display_name="Opus 4.7", source="global"
            ),
        ]
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            all_slots=all_slots,
        )
        assert "[Slot-Belegung im System]" in block
        assert "CHAT: Opus 4.7 (global)" in block
        assert "CODE: Opus 4.7 (global)" in block
        assert "REASON: Opus 4.7 (global)" in block
        assert "CREATIVE: Opus 4.7 (global)" in block
        assert "QUICK: Opus 4.7 (global)" in block
        assert "RESEARCH: Opus 4.7 (global)" in block
        assert "Antworte präzise mit diesen Werten" in block
        # Anti-Halluzination fuer fehlende Slots darf NICHT drin sein
        assert "Ich habe nur Information zu meinem aktiven Slot" not in block

    def test_no_slots_includes_anti_hallucination(self) -> None:
        """Ohne Slot-Liste enthaelt Block Anti-Halluzination fuer andere Slots."""
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            all_slots=None,
        )
        assert "Ich habe nur Information zu meinem aktiven Slot" in block
        assert "[Slot-Belegung im System]" not in block

    def test_mixed_slot_sources_in_block(self) -> None:
        """Block zeigt verschiedene Sources korrekt an."""
        all_slots = [
            SlotInfo(slot_name="chat", model_display_name="Opus 4.7", source="global"),
            SlotInfo(
                slot_name="code", model_display_name="Haiku 4.5", source="user-override"
            ),
            SlotInfo(
                slot_name="reason", model_display_name="Sonnet 4.6", source="default"
            ),
            SlotInfo(
                slot_name="creative", model_display_name="Opus 4.7", source="global"
            ),
            SlotInfo(slot_name="quick", model_display_name="Opus 4.7", source="global"),
            SlotInfo(
                slot_name="research", model_display_name="Opus 4.7", source="global"
            ),
        ]
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            all_slots=all_slots,
        )
        assert "CODE: Haiku 4.5 (user-override)" in block
        assert "REASON: Sonnet 4.6 (default)" in block
        assert "CHAT: Opus 4.7 (global)" in block

    # ── i18n Tests (Fix 6: Self-Awareness EN) ──

    def test_en_block_uses_english_labels(self) -> None:
        """EN-Block nutzt englische Labels statt deutscher."""
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="code",
            provider="anthropic",
            lang="en",
        )
        assert "[SELF-AWARENESS]" in block
        assert "Current model: Opus 4.7" in block
        assert "Do not speculate from training data" in block
        # Deutsche Texte duerfen NICHT drin sein
        assert "Modell:" not in block
        assert "Spekuliere nicht" not in block

    def test_en_block_with_slots(self) -> None:
        """EN-Block mit Slot-Belegung nutzt englische Texte."""
        all_slots = [
            SlotInfo(
                slot_name="chat", model_display_name="Sonnet 4.6", source="default"
            ),
            SlotInfo(slot_name="code", model_display_name="Opus 4.7", source="default"),
        ]
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="code",
            provider="anthropic",
            all_slots=all_slots,
            lang="en",
        )
        assert "[Slot occupancy]" in block
        assert "Answer precisely with these values" in block
        # Deutsche Slot-Heading darf NICHT drin sein
        assert "[Slot-Belegung im System]" not in block

    def test_en_block_without_slots_anti_hallucination(self) -> None:
        """EN-Block ohne Slot-Liste hat englische Anti-Halluzination."""
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            all_slots=None,
            lang="en",
        )
        assert "I only have information about my active slot" in block
        assert "Ich habe nur Information" not in block

    def test_de_is_default(self) -> None:
        """Ohne lang-Parameter wird Deutsch verwendet (Abwärtskompatibilität)."""
        block = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="code",
            provider="anthropic",
        )
        assert "Modell: Opus 4.7" in block
        assert "Spekuliere nicht" in block

    def test_non_de_non_en_uses_native_language(self) -> None:
        """Non-DE/EN languages produce native content (not EN fallback)."""
        block_it = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="code",
            provider="anthropic",
            lang="it",
        )
        block_en = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="code",
            provider="anthropic",
            lang="en",
        )
        # IT must NOT be identical to EN (native i18n content)
        assert block_it != block_en
        # IT block must contain Italian text from i18n
        assert "Modello attuale" in block_it
        assert "[SELF-AWARENESS]" in block_it
