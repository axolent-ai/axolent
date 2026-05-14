"""Coverage-Report-Generator für Axolent Bridge.

Führt pytest mit Coverage-Messung aus und generiert:
  - Terminal-Report: % Coverage pro Modul
  - HTML-Report: htmlcov/index.html als visuelle Coverage-Map

Nutzung:
  python scripts/pytest_coverage.py

Konfiguration:
  .coveragerc im bridge/-Ordner steuert Excludes (.venv, tests).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Projekt-Root: zwei Ebenen hoch (scripts/ -> axolent/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = PROJECT_ROOT / "bridge"
VENV_PYTHON = BRIDGE_DIR / ".venv" / "Scripts" / "python.exe"

# Fallback: sys.executable wenn kein venv Python gefunden
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def main() -> int:
    """Führt pytest mit Coverage aus und zeigt Ergebnisse."""
    cmd = [
        PYTHON,
        "-m",
        "pytest",
        "--cov=.",
        "--cov-config=.coveragerc",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-q",
        "--no-header",
    ]

    print(f"[Coverage] Starte Coverage-Report in {BRIDGE_DIR}")
    print(f"[Coverage] Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(BRIDGE_DIR))

    if result.returncode == 0:
        htmlcov = BRIDGE_DIR / "htmlcov" / "index.html"
        print()
        print(f"[Coverage] HTML-Report: {htmlcov}")
        print("[Coverage] Im Browser öffnen für visuelle Coverage-Map")
    else:
        print()
        print(f"[Coverage] Tests fehlgeschlagen (exit code: {result.returncode})")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
