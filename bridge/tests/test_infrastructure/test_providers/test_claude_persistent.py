"""Tests fuer ClaudePersistentProvider.

Verifiziert:
    - is_available() prueft CLI-Verfuegbarkeit
    - query() sammelt Stream-Events zu ProviderResponse
    - query() handelt Fehler-Events korrekt
    - query() handelt leere Responses
    - query_streaming() liefert StreamEvents
    - Capabilities korrekt gesetzt
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from infrastructure.claude_process_pool import ClaudeProcessPool, StreamEvent
from infrastructure.providers.claude_persistent import ClaudePersistentProvider


def _make_pool_mock() -> AsyncMock:
    """Erstellt einen gemockten ClaudeProcessPool."""
    pool = AsyncMock(spec=ClaudeProcessPool)
    pool.is_cli_available = ClaudeProcessPool.is_cli_available
    return pool


class TestClaudePersistentProviderAvailability:
    """Tests fuer is_available()."""

    def test_available_when_claude_in_path(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)
        with patch("shutil.which", return_value="/usr/bin/claude"):
            assert provider.is_available() is True

    def test_not_available_when_claude_missing(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)
        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_name_is_claude_persistent(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)
        assert provider.name == "claude_persistent"

    def test_capabilities_support_streaming(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)
        caps = provider.get_capabilities()
        assert caps.supports_streaming is True
        assert caps.max_context_tokens == 200_000
        assert caps.cost_class == "subscription"


class TestClaudePersistentProviderQuery:
    """Tests fuer query() (non-streaming wrapper)."""

    @pytest.mark.asyncio
    async def test_successful_query(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)

        async def mock_send_message(chat_id, prompt, system_prompt=""):
            yield StreamEvent(event_type="content_delta", text="Hallo ")
            yield StreamEvent(event_type="content_delta", text="Welt")
            yield StreamEvent(
                event_type="result",
                full_text="Hallo Welt",
                is_final=True,
            )

        pool.send_message = mock_send_message

        result = await provider.query(
            prompt="Test", system_prompt="Sei nett", chat_id=100
        )

        assert result.text == "Hallo Welt"
        assert result.error is None
        assert result.provider_name == "claude_persistent"
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_error_event(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)

        async def mock_send_message(chat_id, prompt, system_prompt=""):
            yield StreamEvent(
                event_type="error",
                text="Rate limited",
                is_final=True,
            )

        pool.send_message = mock_send_message

        result = await provider.query(prompt="Test", chat_id=100)

        assert result.text == ""
        assert result.error is not None
        assert "Rate limited" in result.error

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)

        async def mock_send_message(chat_id, prompt, system_prompt=""):
            yield StreamEvent(
                event_type="result",
                full_text="",
                is_final=True,
            )

        pool.send_message = mock_send_message

        result = await provider.query(prompt="Test", chat_id=100)

        assert result.text == ""
        assert result.error == "empty_response"

    @pytest.mark.asyncio
    async def test_runtime_error_handled(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)

        async def mock_send_message_raises(chat_id, prompt, system_prompt=""):
            """Async generator that raises before yielding."""
            if False:
                yield  # pragma: no cover
            raise RuntimeError("Pipe gebrochen")

        pool.send_message = mock_send_message_raises

        result = await provider.query(prompt="Test", chat_id=100)

        assert result.text == ""
        assert result.error is not None
        assert "Pipe gebrochen" in result.error


class TestClaudePersistentProviderStreaming:
    """Tests fuer query_streaming()."""

    @pytest.mark.asyncio
    async def test_streaming_yields_events(self) -> None:
        pool = _make_pool_mock()
        provider = ClaudePersistentProvider(process_pool=pool)

        async def mock_send_message(chat_id, prompt, system_prompt=""):
            yield StreamEvent(event_type="content_delta", text="Token1 ")
            yield StreamEvent(event_type="content_delta", text="Token2")
            yield StreamEvent(
                event_type="result", full_text="Token1 Token2", is_final=True
            )

        pool.send_message = mock_send_message

        events = []
        async for event in provider.query_streaming(prompt="Test", chat_id=200):
            events.append(event)

        assert len(events) == 3
        assert events[0].event_type == "content_delta"
        assert events[0].text == "Token1 "
        assert events[1].event_type == "content_delta"
        assert events[2].event_type == "result"
        assert events[2].full_text == "Token1 Token2"
