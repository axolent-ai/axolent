"""Ollama auto-start service.

Detects whether Ollama is installed and starts it automatically
at bot startup if configured. Non-blocking, best-effort.

Configuration:
    AXOLENT_OLLAMA_AUTOSTART=true/false (default: true)
    OLLAMA_HOST=http://localhost:11434 (default)
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess  # nosec B404 - used only to start local ollama service
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# Max wait time after starting ollama serve (seconds)
_STARTUP_WAIT_SECONDS: int = 10
_PING_INTERVAL: float = 1.0


def _is_autostart_enabled() -> bool:
    """Check if AXOLENT_OLLAMA_AUTOSTART is enabled (default: true)."""
    raw = os.getenv("AXOLENT_OLLAMA_AUTOSTART", "true").lower()
    return raw in ("true", "1", "yes")


def _get_ollama_host() -> str:
    """Return configured Ollama host URL."""
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _is_ollama_running() -> bool:
    """Check if Ollama service is already running via HTTP ping.

    Makes a GET request to /api/tags with 2s timeout.
    Returns True if status 200.
    """
    host = _get_ollama_host()
    url = f"{host}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")  # nosemgrep
        with urllib.request.urlopen(req, timeout=2) as response:  # nosec B310 # nosemgrep
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def _find_ollama_executable() -> str | None:
    """Find the ollama executable on the system.

    Search order:
    1. PATH (shutil.which)
    2. Windows-specific locations:
       - %LOCALAPPDATA%/Programs/Ollama/ollama.exe
       - %ProgramFiles%/Ollama/ollama.exe
    3. Linux/Mac: PATH only (standard install)

    Returns:
        Full path to ollama executable, or None if not found.
    """
    # 1. Check PATH
    exe = shutil.which("ollama")
    if exe:
        return exe

    # 2. Windows-specific paths
    if platform.system() == "Windows":
        candidates = []
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.append(Path(local_app) / "Programs" / "Ollama" / "ollama.exe")
        program_files = os.environ.get("ProgramFiles", "")
        if program_files:
            candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
        if program_files_x86:
            candidates.append(Path(program_files_x86) / "Ollama" / "ollama.exe")

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    return None


def _start_ollama_serve(exe_path: str) -> bool:
    """Start 'ollama serve' as a detached background process.

    Args:
        exe_path: Full path to the ollama executable.

    Returns:
        True if process was started successfully.
    """
    try:
        if platform.system() == "Windows":
            # DETACHED_PROCESS: no console window, non-blocking
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            subprocess.Popen(  # nosec B603
                [exe_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
            )
        else:
            # Unix: start_new_session detaches from parent
            subprocess.Popen(  # nosec B603
                [exe_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Failed to start ollama serve: %s", exc)
        return False


def _wait_for_ollama(timeout_seconds: int = _STARTUP_WAIT_SECONDS) -> bool:
    """Wait for Ollama to become responsive after starting.

    Polls the HTTP endpoint at 1s intervals.

    Args:
        timeout_seconds: Max wait time.

    Returns:
        True if Ollama responded within timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _is_ollama_running():
            return True
        time.sleep(_PING_INTERVAL)
    return False


def ensure_ollama_running() -> None:
    """Main entry point: ensure Ollama is running at bot startup.

    Logic:
    1. If autostart disabled: skip silently
    2. If already running: log and return
    3. If not installed: log info (not an error) and return
    4. If installed but not running: start it, wait, log result

    This function is synchronous and should be called during
    bot initialization (before async event loop).
    """
    if not _is_autostart_enabled():
        log.debug("Ollama autostart disabled (AXOLENT_OLLAMA_AUTOSTART=false)")
        return

    # Already running?
    if _is_ollama_running():
        log.info("Ollama: already running at %s", _get_ollama_host())
        return

    # Find executable
    exe_path = _find_ollama_executable()
    if exe_path is None:
        log.info(
            "Ollama not installed (not in PATH, not in standard locations). "
            "/debate will use only Claude."
        )
        return

    # Start it
    log.info("Ollama: starting service (%s serve)...", exe_path)
    started = _start_ollama_serve(exe_path)
    if not started:
        log.warning(
            "Ollama: could not start serve process. "
            "/debate will use only Claude. Start Ollama manually if needed."
        )
        return

    # Wait for it to become responsive
    if _wait_for_ollama():
        log.info("Ollama: service started successfully at %s", _get_ollama_host())
    else:
        log.warning(
            "Ollama: started process but not responsive after %ds. "
            "/debate may not include local model. Try starting Ollama manually.",
            _STARTUP_WAIT_SECONDS,
        )
