#!/usr/bin/env python3
"""Platform-portable venv wrapper for pre-commit hooks.

Finds the Python executable in bridge/.venv and runs
the specified module. Works on Windows, Linux, and macOS
without bash dependency.

Usage:
    python scripts/run_with_venv.py <module> [args...]

Examples:
    python scripts/run_with_venv.py pytest -q --no-header
    python scripts/run_with_venv.py pip_audit
    python scripts/run_with_venv.py lint_imports
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_venv_python() -> Path:
    """Finds the Python executable in bridge/.venv.

    Checks Windows path (.venv/Scripts/python.exe) and
    Unix path (.venv/bin/python).

    Returns:
        Path to the venv Python executable.

    Raises:
        SystemExit: If no venv is found.
    """
    bridge_dir = Path(__file__).resolve().parent.parent / "bridge"
    candidates = [
        bridge_dir / ".venv" / "Scripts" / "python.exe",  # Windows
        bridge_dir / ".venv" / "bin" / "python",  # Linux/macOS
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    print(
        f"ERROR: No venv found in {bridge_dir / '.venv'}",
        file=sys.stderr,
    )
    sys.exit(1)


def find_venv_executable(name: str) -> Path | None:
    """Finds an executable in the venv Scripts/bin directory.

    Args:
        name: Name of the executable (without extension).

    Returns:
        Path to the executable or None.
    """
    bridge_dir = Path(__file__).resolve().parent.parent / "bridge"
    candidates = [
        bridge_dir / ".venv" / "Scripts" / f"{name}.exe",  # Windows
        bridge_dir / ".venv" / "bin" / name,  # Linux/macOS
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    """Main logic: run module inside venv."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <module|executable> [args...]", file=sys.stderr)
        sys.exit(1)

    module_or_exe = sys.argv[1]
    extra_args = sys.argv[2:]
    bridge_dir = Path(__file__).resolve().parent.parent / "bridge"

    # Special case: lint-imports (executable, not a Python module)
    if module_or_exe == "lint-imports":
        exe = find_venv_executable("lint-imports")
        if exe is None:
            print("ERROR: lint-imports not found in venv", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run(
            [str(exe)] + extra_args,
            cwd=str(bridge_dir),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        sys.exit(result.returncode)

    # Special case: semgrep (executable)
    if module_or_exe == "semgrep":
        exe = find_venv_executable("semgrep")
        if exe is None:
            print("ERROR: semgrep not found in venv", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run(
            [str(exe)] + extra_args,
            cwd=str(bridge_dir),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        sys.exit(result.returncode)

    # Default: Python -m <module>
    venv_python = find_venv_python()
    result = subprocess.run(  # noqa: S603
        [str(venv_python), "-m", module_or_exe] + extra_args,
        cwd=str(bridge_dir),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
