#!/usr/bin/env python3
"""Plattform-portabler venv-Wrapper für Pre-Commit-Hooks.

Findet die Python-Executable in der bridge/.venv und führt
das angegebene Modul aus. Funktioniert auf Windows, Linux und macOS
ohne bash-Abhängigkeit.

Usage:
    python scripts/run_with_venv.py <module> [args...]

Beispiele:
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
    """Findet die Python-Executable in bridge/.venv.

    Prüft Windows-Pfad (.venv/Scripts/python.exe) und
    Unix-Pfad (.venv/bin/python).

    Returns:
        Pfad zur venv-Python-Executable.

    Raises:
        SystemExit: Wenn kein venv gefunden wird.
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
        f"FEHLER: Kein venv gefunden in {bridge_dir / '.venv'}",
        file=sys.stderr,
    )
    sys.exit(1)


def find_venv_executable(name: str) -> Path | None:
    """Findet eine Executable im venv Scripts/bin Ordner.

    Args:
        name: Name der Executable (ohne Extension).

    Returns:
        Pfad zur Executable oder None.
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
    """Hauptlogik: Modul im venv ausführen."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <module|executable> [args...]", file=sys.stderr)
        sys.exit(1)

    module_or_exe = sys.argv[1]
    extra_args = sys.argv[2:]
    bridge_dir = Path(__file__).resolve().parent.parent / "bridge"

    # Spezialfall: lint-imports (Executable, kein Python-Modul)
    if module_or_exe == "lint-imports":
        exe = find_venv_executable("lint-imports")
        if exe is None:
            print("FEHLER: lint-imports nicht im venv gefunden", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run(
            [str(exe)] + extra_args,
            cwd=str(bridge_dir),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        sys.exit(result.returncode)

    # Spezialfall: semgrep (Executable)
    if module_or_exe == "semgrep":
        exe = find_venv_executable("semgrep")
        if exe is None:
            print("FEHLER: semgrep nicht im venv gefunden", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run(
            [str(exe)] + extra_args,
            cwd=str(bridge_dir),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        sys.exit(result.returncode)

    # Standard: Python -m <module>
    venv_python = find_venv_python()
    result = subprocess.run(  # noqa: S603
        [str(venv_python), "-m", module_or_exe] + extra_args,
        cwd=str(bridge_dir),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
