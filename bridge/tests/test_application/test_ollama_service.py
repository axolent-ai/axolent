"""Tests for application.ollama_service (Ollama auto-start at bot startup).

Tests:
* Service start is called when Ollama is installed but not running
* Nothing happens when Ollama is already running
* Nothing happens when Ollama is not installed
* Autostart disabled via env var
* Start failure is handled gracefully (no crash)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from application.ollama_service import (
    _find_ollama_executable,
    _is_ollama_running,
    ensure_ollama_running,
)


class TestIsOllamaRunning:
    """Test the HTTP ping check."""

    def test_returns_true_on_200(self) -> None:
        """Returns True when Ollama responds with 200."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(
            "application.ollama_service.urllib.request.urlopen",
            return_value=mock_response,
        ):
            assert _is_ollama_running() is True

    def test_returns_false_on_connection_error(self) -> None:
        """Returns False when connection refused."""
        import urllib.error

        with patch(
            "application.ollama_service.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            assert _is_ollama_running() is False


class TestFindOllamaExecutable:
    """Test executable detection."""

    def test_finds_in_path(self) -> None:
        """Finds ollama via shutil.which."""
        with patch(
            "application.ollama_service.shutil.which", return_value="/usr/bin/ollama"
        ):
            result = _find_ollama_executable()
            assert result == "/usr/bin/ollama"

    def test_returns_none_when_not_installed(self) -> None:
        """Returns None when not in PATH and no Windows paths exist."""
        with (
            patch("application.ollama_service.shutil.which", return_value=None),
            patch("application.ollama_service.platform.system", return_value="Linux"),
        ):
            result = _find_ollama_executable()
            assert result is None


class TestEnsureOllamaRunning:
    """Integration tests for the main entry point."""

    def test_skips_when_autostart_disabled(self) -> None:
        """Does nothing when AXOLENT_OLLAMA_AUTOSTART=false."""
        with patch.dict("os.environ", {"AXOLENT_OLLAMA_AUTOSTART": "false"}):
            with patch("application.ollama_service._is_ollama_running") as mock_ping:
                ensure_ollama_running()
                mock_ping.assert_not_called()

    def test_skips_when_already_running(self) -> None:
        """Does nothing when Ollama is already running."""
        with (
            patch.dict("os.environ", {"AXOLENT_OLLAMA_AUTOSTART": "true"}),
            patch("application.ollama_service._is_ollama_running", return_value=True),
            patch("application.ollama_service._find_ollama_executable") as mock_find,
        ):
            ensure_ollama_running()
            mock_find.assert_not_called()

    def test_skips_when_not_installed(self) -> None:
        """Logs info and returns when Ollama is not found."""
        with (
            patch.dict("os.environ", {"AXOLENT_OLLAMA_AUTOSTART": "true"}),
            patch("application.ollama_service._is_ollama_running", return_value=False),
            patch(
                "application.ollama_service._find_ollama_executable", return_value=None
            ),
            patch("application.ollama_service._start_ollama_serve") as mock_start,
        ):
            ensure_ollama_running()
            mock_start.assert_not_called()

    def test_starts_when_installed_but_not_running(self) -> None:
        """Starts Ollama when installed but service is not running."""
        with (
            patch.dict("os.environ", {"AXOLENT_OLLAMA_AUTOSTART": "true"}),
            patch("application.ollama_service._is_ollama_running", return_value=False),
            patch(
                "application.ollama_service._find_ollama_executable",
                return_value="/usr/bin/ollama",
            ),
            patch(
                "application.ollama_service._start_ollama_serve", return_value=True
            ) as mock_start,
            patch("application.ollama_service._wait_for_ollama", return_value=True),
        ):
            ensure_ollama_running()
            mock_start.assert_called_once_with("/usr/bin/ollama")

    def test_handles_start_failure_gracefully(self) -> None:
        """No crash when subprocess start fails."""
        with (
            patch.dict("os.environ", {"AXOLENT_OLLAMA_AUTOSTART": "true"}),
            patch("application.ollama_service._is_ollama_running", return_value=False),
            patch(
                "application.ollama_service._find_ollama_executable",
                return_value="/usr/bin/ollama",
            ),
            patch("application.ollama_service._start_ollama_serve", return_value=False),
            patch("application.ollama_service._wait_for_ollama") as mock_wait,
        ):
            # Should not raise
            ensure_ollama_running()
            mock_wait.assert_not_called()
