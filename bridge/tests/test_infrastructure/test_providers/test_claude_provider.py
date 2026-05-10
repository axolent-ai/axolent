"""Tests für ClaudeProvider.

Verifiziert:
    - is_available() prüft ob claude im PATH ist
    - query() ruft Subprozess korrekt auf (gemockt)
    - Timeout-Handling funktioniert
    - Fehler-Handling bei non-zero exit code
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.providers.claude_cli import ClaudeProvider


class TestClaudeProviderAvailability:
    """Tests für is_available()."""

    def test_available_when_claude_in_path(self) -> None:
        provider = ClaudeProvider()
        with patch("shutil.which", return_value="/usr/bin/claude"):
            assert provider.is_available() is True

    def test_not_available_when_claude_missing(self) -> None:
        provider = ClaudeProvider()
        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_name_is_claude(self) -> None:
        provider = ClaudeProvider()
        assert provider.name == "claude"


class TestClaudeProviderQuery:
    """Tests für query() mit gemocktem Subprozess."""

    @pytest.mark.asyncio
    async def test_successful_query(self) -> None:
        provider = ClaudeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"Claude sagt hallo", b""))
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await provider.query("Hallo", system_prompt="Sei nett")

        assert result.text == "Claude sagt hallo"
        assert result.error is None
        assert result.provider_name == "claude"
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self) -> None:
        provider = ClaudeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: rate limited"))
        mock_proc.returncode = 1
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await provider.query("Test")

        assert result.text == ""
        assert result.error is not None
        assert "exit_code_1" in result.error

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        provider = ClaudeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                # ClaudeProvider handles TimeoutError internally
                result = await provider.query("Test", timeout_seconds=1)

        assert result.error is not None
        assert "timeout" in result.error

    @pytest.mark.asyncio
    async def test_query_accepts_user_id_and_chat_id(self) -> None:
        """Legacy-Provider darf nicht crashen wenn user_id/chat_id uebergeben werden."""
        provider = ClaudeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"OK", b""))
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await provider.query("Test", user_id=12345, chat_id=67890)

        assert result.text == "OK"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_query_accepts_unknown_kwargs(self) -> None:
        """Legacy-Provider darf nicht crashen bei unbekannten kwargs."""
        provider = ClaudeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"Fine", b""))
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await provider.query(
                "Test", user_id=1, chat_id=2, future_param="ignored"
            )

        assert result.text == "Fine"
        assert result.error is None
