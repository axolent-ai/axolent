"""Pytest wrapper for the auto-smoke-test script.

Ensures the smoke test stays green in CI and local pytest runs.
If any of the 15 scenarios fail, this test fails with details.

The smoke_test module lives in ``<repo>/scripts/``, outside the
``bridge/`` pythonpath.  We add it to sys.path at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ sits one level above bridge/
_scripts_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


@pytest.mark.asyncio
async def test_smoke_script_all_scenarios_pass() -> None:
    """All 15 smoke scenarios must pass without exceptions."""
    from smoke_test import run_all_scenarios

    results = await run_all_scenarios()

    failed = [r for r in results if not r.passed]
    assert all(r.passed for r in results), failed
