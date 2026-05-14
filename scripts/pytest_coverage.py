"""Coverage report generator for Axolent Bridge.

Runs pytest with coverage measurement and generates:
  - Terminal report: % coverage per module
  - HTML report: htmlcov/index.html as visual coverage map

Usage:
  python scripts/pytest_coverage.py

Configuration:
  .coveragerc in the bridge/ directory controls excludes (.venv, tests).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Project root: two levels up (scripts/ -> axolent/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = PROJECT_ROOT / "bridge"
VENV_PYTHON = BRIDGE_DIR / ".venv" / "Scripts" / "python.exe"

# Fallback: sys.executable if no venv Python found
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def main() -> int:
    """Runs pytest with coverage and displays results."""
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

    print(f"[Coverage] Starting coverage report in {BRIDGE_DIR}")
    print(f"[Coverage] Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(BRIDGE_DIR))

    if result.returncode == 0:
        htmlcov = BRIDGE_DIR / "htmlcov" / "index.html"
        print()
        print(f"[Coverage] HTML report: {htmlcov}")
        print("[Coverage] Open in browser for visual coverage map")
    else:
        print()
        print(f"[Coverage] Tests failed (exit code: {result.returncode})")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
