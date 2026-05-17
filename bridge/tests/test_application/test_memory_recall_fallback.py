"""Tests for NEU-02 fix: memory recall fallback behavior.

Verifies that when keywords ARE extracted but yield no matches,
the system does NOT fall back to random recent entries.
This prevents "phantom knowledge" where the bot appears to know
things unrelated to the current question.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.chat_service import ChatService
from infrastructure.conversation_storage import _reset_all_for_tests
from infrastructure.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Reset conversation storage before each test."""
    _reset_all_for_tests()


def _make_memory_service(
    recall_returns: list | None = None,
    list_recent_returns: list | None = None,
) -> MagicMock:
    """Create a mock MemoryService.

    Args:
        recall_returns: What recall() should return (empty list = no matches).
        list_recent_returns: What list_recent() should return.
    """
    mock = MagicMock()
    mock.recall.return_value = recall_returns or []
    mock.list_recent.return_value = list_recent_returns or []
    return mock


class TestMemoryRecallNoFallbackWhenKeywordsPresent:
    """NEU-02: No random memory fallback when keywords exist but yield no matches."""

    async def test_keywords_present_no_matches_returns_empty(self) -> None:
        """Keywords extracted but no recall matches -> empty memory context."""
        memory_svc = _make_memory_service(
            recall_returns=[],
            list_recent_returns=[
                {"id": "ep1", "content": "Jessica likes dolphins"},
                {"id": "ep2", "content": "Jessica lives in Vienna"},
            ],
        )

        mock_router = MagicMock()
        mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="Response", duration_seconds=1.0, provider_name="claude"
            )
        )

        svc = ChatService(provider_router=mock_router, memory_service=memory_svc)

        # "Erklaere mir Quantencomputer" has extractable keywords
        # but they won't match any memory entries
        result = await svc.process_user_message(
            text="Erklaere mir bitte Quantencomputer und ihre Funktionsweise",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        assert result.success is True

        # The critical assertion: list_recent should NOT have been called
        # as a fallback since keywords were extracted
        # (list_recent is only for the "no keywords" case)
        # Check the system_prompt passed to route: should NOT contain
        # "dolphins" or "Vienna" (random unrelated memories)
        call_kwargs = mock_router.route.call_args
        system_prompt_used = call_kwargs.kwargs.get(
            "system_prompt", call_kwargs.args[1] if len(call_kwargs.args) > 1 else ""
        )
        # Fallback to checking named arg
        if not system_prompt_used:
            for call in mock_router.route.call_args_list:
                if "system_prompt" in call.kwargs:
                    system_prompt_used = call.kwargs["system_prompt"]
                    break

        assert "dolphins" not in system_prompt_used
        assert "Vienna" not in system_prompt_used

    async def test_no_keywords_short_message_uses_recent(self) -> None:
        """Very short message (no extractable keywords) -> recent fallback OK."""
        memory_svc = _make_memory_service(
            recall_returns=[],
            list_recent_returns=[
                {"id": "ep1", "content": "User likes coffee"},
            ],
        )

        mock_router = MagicMock()
        mock_router.route = AsyncMock(
            return_value=ProviderResponse(
                text="Response", duration_seconds=1.0, provider_name="claude"
            )
        )

        svc = ChatService(provider_router=mock_router, memory_service=memory_svc)

        # "Hi" has no extractable keywords (all words <= 3 chars)
        result = await svc.process_user_message(
            text="Hi",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
        )

        assert result.success is True
        # list_recent SHOULD be called for short messages
        memory_svc.list_recent.assert_called()
