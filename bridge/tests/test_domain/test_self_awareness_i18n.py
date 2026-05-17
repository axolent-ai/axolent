"""Tests for self-awareness block i18n support.

Verifies that the self-awareness block uses i18n t() for all languages,
not just DE/EN. Specifically tests Italian (IT) since that was a known
regression in the root-cause review.
"""

from __future__ import annotations


from domain.personality import SlotInfo, build_self_awareness_block


class TestSelfAwarenessI18n:
    """Tests that self-awareness block works for non-DE/EN languages."""

    def test_self_awareness_block_german(self) -> None:
        """German block uses correct labels via i18n."""
        result = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            lang="de",
        )

        assert "[SELF-AWARENESS]" in result
        assert "Modell: Opus 4.7" in result
        assert "Spekuliere nicht" in result

    def test_self_awareness_block_english(self) -> None:
        """English block uses correct labels via i18n."""
        result = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            lang="en",
        )

        assert "[SELF-AWARENESS]" in result
        assert "Current model: Opus 4.7" in result
        assert "Do not speculate" in result

    def test_self_awareness_block_for_italian(self) -> None:
        """Italian block uses Italian labels, NOT English fallback.

        This was bug T33/Item 9: IT/FR/etc fell back to EN.
        """
        result = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            lang="it",
        )

        assert "[SELF-AWARENESS]" in result
        # Should have Italian text, not English
        assert "Modello attuale: Opus 4.7" in result
        assert "Non speculare" in result

    def test_self_awareness_block_for_french(self) -> None:
        """French block uses French labels."""
        result = build_self_awareness_block(
            model_display_name="Sonnet 4",
            model_id="claude-sonnet-4-20250514",
            task_slot="code",
            provider="anthropic",
            lang="fr",
        )

        assert "[SELF-AWARENESS]" in result
        assert "Modèle actuel: Sonnet 4" in result

    def test_self_awareness_block_with_slots(self) -> None:
        """Block with all_slots includes slot heading in user language."""
        slots = [
            SlotInfo(slot_name="chat", model_display_name="Opus 4.7", source="default"),
            SlotInfo(
                slot_name="code", model_display_name="Sonnet 4", source="user-override"
            ),
        ]

        result = build_self_awareness_block(
            model_display_name="Opus 4.7",
            model_id="claude-opus-4-7",
            task_slot="chat",
            provider="anthropic",
            all_slots=slots,
            lang="de",
        )

        assert "[Slot-Belegung im System]" in result
        assert "CHAT: Opus 4.7 (default)" in result
        assert "CODE: Sonnet 4 (user-override)" in result

    def test_self_awareness_block_for_japanese(self) -> None:
        """Japanese block uses Japanese labels."""
        result = build_self_awareness_block(
            model_display_name="Haiku 3.5",
            model_id="claude-3-5-haiku-20241022",
            task_slot="chat",
            provider="anthropic",
            lang="ja",
        )

        assert "[SELF-AWARENESS]" in result
        # Japanese model label
        assert "現在のモデル: Haiku 3.5" in result
