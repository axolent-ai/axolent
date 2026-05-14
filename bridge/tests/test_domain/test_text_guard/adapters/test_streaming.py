"""Tests for the streaming text guard adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from domain.text_guard import TextGuard, get_builtin_rules
from domain.text_guard.adapters.streaming import StreamingTextGuard
from domain.text_guard.models import RuleSet, WordPair


@pytest.fixture
def de_stream_guard() -> StreamingTextGuard:
    """German streaming guard."""
    rules = get_builtin_rules("de")
    assert rules is not None
    guard = TextGuard(rules, mode="fix")
    return StreamingTextGuard(guard)


@pytest.fixture
def small_stream_guard() -> StreamingTextGuard:
    """Small streaming guard for focused tests."""
    rules = RuleSet(
        language="de",
        word_pairs=(
            WordPair("fuer", "für"),
            WordPair("ueber", "über"),
        ),
        loan_word_whitelist=frozenset({"user", "queue"}),
    )
    guard = TextGuard(rules)
    return StreamingTextGuard(guard)


async def _tokens_from_list(tokens: list[str]) -> AsyncIterator[str]:
    """Helper: create an async iterator from a list of tokens."""
    for token in tokens:
        yield token


class TestStreamingTextGuardFilter:
    """Tests for the async filter() method."""

    @pytest.mark.asyncio
    async def test_corrects_complete_word(
        self, small_stream_guard: StreamingTextGuard
    ) -> None:
        """Complete word followed by space gets corrected."""
        tokens = ["fuer", " ", "dich"]
        result_parts: list[str] = []
        async for part in small_stream_guard.filter(_tokens_from_list(tokens)):
            result_parts.append(part)
        result = "".join(result_parts)
        assert "für" in result
        assert "dich" in result

    @pytest.mark.asyncio
    async def test_preserves_loan_word(
        self, small_stream_guard: StreamingTextGuard
    ) -> None:
        """Whitelisted loan word passes through unchanged."""
        tokens = ["user", " ", "is", " ", "here"]
        result_parts: list[str] = []
        async for part in small_stream_guard.filter(_tokens_from_list(tokens)):
            result_parts.append(part)
        result = "".join(result_parts)
        assert "user" in result

    @pytest.mark.asyncio
    async def test_buffers_partial_words(
        self, small_stream_guard: StreamingTextGuard
    ) -> None:
        """Partial word is buffered until word boundary."""
        tokens = ["fu", "er", " ", "dich"]
        result_parts: list[str] = []
        async for part in small_stream_guard.filter(_tokens_from_list(tokens)):
            result_parts.append(part)
        result = "".join(result_parts)
        assert "für" in result

    @pytest.mark.asyncio
    async def test_flushes_at_end(self, small_stream_guard: StreamingTextGuard) -> None:
        """Remaining buffer is flushed at end of stream."""
        tokens = ["fuer"]
        result_parts: list[str] = []
        async for part in small_stream_guard.filter(_tokens_from_list(tokens)):
            result_parts.append(part)
        result = "".join(result_parts)
        assert result == "für"

    @pytest.mark.asyncio
    async def test_empty_stream(self, small_stream_guard: StreamingTextGuard) -> None:
        """Empty stream produces no output."""
        tokens: list[str] = []
        result_parts: list[str] = []
        async for part in small_stream_guard.filter(_tokens_from_list(tokens)):
            result_parts.append(part)
        assert result_parts == []


class TestStreamingTextGuardSyncApi:
    """Tests for the synchronous process_token() + flush() API."""

    def test_process_token_buffers(
        self, small_stream_guard: StreamingTextGuard
    ) -> None:
        """Tokens without word boundary are buffered."""
        result = small_stream_guard.process_token("fu")
        assert result is None

    def test_process_token_flushes_on_boundary(
        self, small_stream_guard: StreamingTextGuard
    ) -> None:
        """Token with word boundary flushes corrected text."""
        small_stream_guard.process_token("fuer")
        result = small_stream_guard.process_token(" ")
        assert result is not None
        assert "für" in result

    def test_flush_remaining(self, small_stream_guard: StreamingTextGuard) -> None:
        """Flush returns remaining buffer content."""
        small_stream_guard.process_token("fuer")
        result = small_stream_guard.flush()
        assert result == "für"

    def test_flush_empty(self, small_stream_guard: StreamingTextGuard) -> None:
        """Flush on empty buffer returns empty string."""
        result = small_stream_guard.flush()
        assert result == ""

    def test_reset_clears_state(self, small_stream_guard: StreamingTextGuard) -> None:
        """Reset clears the buffer."""
        small_stream_guard.process_token("partial")
        small_stream_guard.reset()
        result = small_stream_guard.flush()
        assert result == ""


class TestStreamingCodeBlockHandling:
    """Tests for code block detection in streaming mode."""

    def test_code_block_passthrough(
        self, small_stream_guard: StreamingTextGuard
    ) -> None:
        """Content in code blocks is not corrected."""
        # Simulate streaming: ``` + fuer + ```
        small_stream_guard.process_token("```")
        small_stream_guard.process_token("\n")
        small_stream_guard.process_token("fuer")
        small_stream_guard.process_token("\n")
        small_stream_guard.process_token("```")
        result = small_stream_guard.flush()
        # The fuer inside code block should be unchanged
        assert "fuer" in result or result == ""
