"""Tests for T28: Active Curiosity memory behavior.

Validates that the memory context block injected into the system prompt
contains the correct Active Curiosity instructions. These tests check
prompt construction, not actual LLM output (which is non-deterministic).

T28 design principle: the bot must not extrapolate from stored memory,
but should ask with genuine interest when it notices a gap.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.chat_service import ChatService
from infrastructure.conversation_storage import _reset_all_for_tests
from infrastructure.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _clear_conversation_storage() -> None:
    """Reset conversation storage before each test."""
    _reset_all_for_tests()


def _make_chat_service_with_memory(
    memory_recall_side_effect=None,
) -> tuple[ChatService, MagicMock]:
    """Create a ChatService with mocked ProviderRouter and MemoryService.

    Args:
        memory_recall_side_effect: callable for MemoryService.recall mock.

    Returns:
        Tuple of (ChatService, mock_router).
    """
    mock_router = MagicMock()
    mock_router.route = AsyncMock(
        return_value=ProviderResponse(
            text="Test response",
            duration_seconds=1.0,
            provider_name="claude",
        )
    )

    mock_memory = MagicMock()
    if memory_recall_side_effect:
        mock_memory.recall = MagicMock(side_effect=memory_recall_side_effect)
    else:
        mock_memory.recall = MagicMock(return_value=[])
    mock_memory.list_recent = MagicMock(return_value=[])

    svc = ChatService(
        provider_router=mock_router,
        memory_service=mock_memory,
    )
    return svc, mock_router


class TestActiveCuriosityPromptConstruction:
    """Verify that the memory context block contains Active Curiosity rules."""

    async def test_no_extrapolation_instruction_present(self) -> None:
        """Memory context block must instruct the LLM to not invent reasons."""

        def recall_dolphins(uid, q, layer, limit):
            if layer == "episodic":
                return [{"id": "ep_delfin01", "content": "Ich mag Delfine"}]
            return []

        svc, mock_router = _make_chat_service_with_memory(
            memory_recall_side_effect=recall_dolphins,
        )

        await svc.process_user_message(
            text="Warum mag ich Delfine?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")

        # The no-extrapolation rule must be in the prompt
        assert "never invent" in system_sent.lower(), (
            "Memory context must contain no-extrapolation instruction"
        )
        assert "reasons" in system_sent.lower() or "motivations" in system_sent.lower()

    async def test_active_curiosity_instruction_present(self) -> None:
        """Memory context block must instruct the LLM to ask with genuine interest."""

        def recall_dolphins(uid, q, layer, limit):
            if layer == "episodic":
                return [{"id": "ep_delfin01", "content": "Ich mag Delfine"}]
            return []

        svc, mock_router = _make_chat_service_with_memory(
            memory_recall_side_effect=recall_dolphins,
        )

        await svc.process_user_message(
            text="Warum mag ich Delfine?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")

        # The active curiosity example must be present
        assert "genuine interest" in system_sent.lower(), (
            "Memory context must contain active curiosity instruction"
        )
        # The dolphins example should be in the instruction block
        assert (
            "dolphins" in system_sent.lower() or "what draws you" in system_sent.lower()
        )

    async def test_no_interrogation_instruction_present(self) -> None:
        """Memory context block must limit follow-up questions to one."""

        def recall_simple(uid, q, layer, limit):
            if layer == "semantic":
                return [
                    {
                        "id": "sem_fav01",
                        "content": "Lieblingsfarbe ist Blau",
                        "category": "preference",
                    }
                ]
            return []

        svc, mock_router = _make_chat_service_with_memory(
            memory_recall_side_effect=recall_simple,
        )

        await svc.process_user_message(
            text="Was ist meine Lieblingsfarbe?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")

        # Anti-interrogation rule must be present
        assert "not a list" in system_sent.lower(), (
            "Memory context must contain anti-interrogation instruction"
        )
        assert (
            "one natural" in system_sent.lower()
            or "do not interrogate" in system_sent.lower()
        )

    async def test_curiosity_over_constraint_framing(self) -> None:
        """Instruction must frame curiosity as a feature, not a weakness."""

        def recall_hobby(uid, q, layer, limit):
            if layer == "episodic":
                return [{"id": "ep_hobby01", "content": "Ich spiele gerne Schach"}]
            return []

        svc, mock_router = _make_chat_service_with_memory(
            memory_recall_side_effect=recall_hobby,
        )

        await svc.process_user_message(
            text="Warum spiele ich Schach?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")

        # Positive framing of curiosity
        assert "curiosity as a feature" in system_sent.lower(), (
            "Memory context must frame curiosity positively"
        )
        assert "best guess" in system_sent.lower(), (
            "Memory context must state that user's answer beats guessing"
        )

    async def test_no_extrapolation_across_multiple_entries(self) -> None:
        """Multiple memory entries: prompt must still contain no-extrapolation rule.

        The instruction block is shared across all entries and must not
        be omitted when multiple layers have data.
        """

        def recall_multi(uid, q, layer, limit):
            if layer == "episodic":
                return [
                    {"id": "ep_001", "content": "Ich mag Delfine"},
                    {"id": "ep_002", "content": "Gestern war ich im Kino"},
                ]
            elif layer == "semantic":
                return [
                    {
                        "id": "sem_001",
                        "content": "Lieblingsfarbe ist Blau",
                        "category": "preference",
                    }
                ]
            elif layer == "procedural":
                return [
                    {
                        "id": "pro_001",
                        "content": "Immer pytest nutzen",
                        "skill_name": "testing",
                    }
                ]
            return []

        svc, mock_router = _make_chat_service_with_memory(
            memory_recall_side_effect=recall_multi,
        )

        await svc.process_user_message(
            text="Erzaehl mir ueber meine Hobbys und Vorlieben",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        call_args = mock_router.route.call_args
        system_sent = call_args.kwargs.get("system_prompt", "")

        # All entries must be present
        assert "Delfine" in system_sent
        assert "Kino" in system_sent
        assert "Blau" in system_sent
        assert "pytest" in system_sent

        # Active Curiosity instructions must still be present
        assert "never invent" in system_sent.lower()
        assert "genuine interest" in system_sent.lower()
        assert "not a list" in system_sent.lower()
