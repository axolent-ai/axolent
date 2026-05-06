"""Tests fuer infrastructure.conversation_storage: In-Memory Conversation History.

Testet async CRUD-Operationen, MAX_HISTORY eviction und sticky language.
"""

from __future__ import annotations

import pytest

from domain.conversation import ConversationTurn, MAX_HISTORY_TURNS
from infrastructure.conversation_storage import (
    _histories,
    _languages,
    get_history,
    get_language,
    reset_conversation,
    save_turn,
    set_language,
)


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Raeumt den in-memory Storage vor jedem Test auf."""
    _histories.clear()
    _languages.clear()


class TestConversationStorage:
    """Conversation-Storage async Operationen."""

    async def test_save_and_get_history(self) -> None:
        """Gespeicherte Turns werden per get_history zurueckgegeben."""
        turn = ConversationTurn(role="user", content="Hi")
        await save_turn(1, 10, turn)

        history = await get_history(1, 10)
        assert len(history) == 1
        assert history[0].content == "Hi"
        assert history[0].role == "user"

    async def test_max_20_turns_eviction(self) -> None:
        """Nach 20 Turns werden die aeltesten evicted (FIFO)."""
        for i in range(25):
            turn = ConversationTurn(role="user", content=f"Msg {i}")
            await save_turn(1, 10, turn)

        history = await get_history(1, 10)
        assert len(history) == MAX_HISTORY_TURNS  # Genau 20
        # Aelteste (0-4) wurden evicted, neueste (5-24) bleiben
        assert history[0].content == "Msg 5"
        assert history[-1].content == "Msg 24"

    async def test_set_and_get_language(self) -> None:
        """Sticky-Language wird korrekt gesetzt und abgerufen."""
        await set_language(1, 10, "en")
        lang = await get_language(1, 10)
        assert lang == "en"

    async def test_get_language_unset_returns_none(self) -> None:
        """Ohne set_language gibt get_language None zurueck."""
        lang = await get_language(99, 99)
        assert lang is None

    async def test_reset_clears_history_and_language(self) -> None:
        """reset_conversation loescht sowohl History als auch Language."""
        turn = ConversationTurn(role="user", content="Before reset")
        await save_turn(1, 10, turn)
        await set_language(1, 10, "fr")

        await reset_conversation(1, 10)

        history = await get_history(1, 10)
        lang = await get_language(1, 10)
        assert history == []
        assert lang is None

    async def test_different_chats_isolated(self) -> None:
        """Verschiedene (user_id, chat_id) Paare sind isoliert."""
        await save_turn(1, 10, ConversationTurn(role="user", content="Chat A"))
        await save_turn(1, 20, ConversationTurn(role="user", content="Chat B"))

        history_a = await get_history(1, 10)
        history_b = await get_history(1, 20)

        assert len(history_a) == 1
        assert history_a[0].content == "Chat A"
        assert len(history_b) == 1
        assert history_b[0].content == "Chat B"

    async def test_language_per_chat(self) -> None:
        """Language ist pro (user_id, chat_id) gespeichert."""
        await set_language(1, 10, "de")
        await set_language(1, 20, "en")

        assert await get_language(1, 10) == "de"
        assert await get_language(1, 20) == "en"

    async def test_history_returns_copy(self) -> None:
        """get_history gibt eine Kopie zurueck, nicht die interne Liste."""
        await save_turn(1, 10, ConversationTurn(role="user", content="Test"))
        history = await get_history(1, 10)
        history.clear()  # Externe Modifikation

        # Intern muss die History noch intakt sein
        internal = await get_history(1, 10)
        assert len(internal) == 1
