"""Smoke tests for scripts/dev_check.py.

Verifies the CLI contract:
    - --help exits with code 0
    - --fast flag is accepted without error
    - A failing subprocess causes the overall exit code to be 1
    - All-pass subprocesses produce exit code 0
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEV_CHECK = _REPO_ROOT / "scripts" / "dev_check.py"


class TestDevCheckHelp:
    """--help must print usage and exit 0."""

    def test_dev_check_help_exits_zero(self) -> None:
        """python scripts/dev_check.py --help returns exit code 0."""
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(_DEV_CHECK), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, (
            f"Expected 0, got {result.returncode}.\n{result.stdout}"
        )

    def test_dev_check_help_mentions_fast(self) -> None:
        """--help output must describe the --fast flag."""
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(_DEV_CHECK), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert "--fast" in result.stdout


def _load_dev_check():
    """Load dev_check module via importlib, registering it in sys.modules.

    Required because dev_check.py uses @dataclass which needs the module
    to be present in sys.modules during class creation.
    """
    import importlib.util
    import sys

    mod_name = "_dev_check_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, _DEV_CHECK)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestDevCheckFastMode:
    """--fast flag must be accepted without crashing."""

    def test_fast_flag_recognized(self) -> None:
        """_build_steps(fast=True) returns fewer steps than fast=False."""
        mod = _load_dev_check()

        full_steps = mod._build_steps(fast=False)
        fast_steps = mod._build_steps(fast=True)

        assert len(fast_steps) < len(full_steps), (
            f"fast mode should produce fewer steps; "
            f"got {len(fast_steps)} vs {len(full_steps)}"
        )
        # The optional bandit/ruff steps must be absent in fast mode
        fast_labels = {s.label for s in fast_steps}
        assert "bandit security linter" not in fast_labels
        assert "ruff linter" not in fast_labels


@pytest.mark.unit
class TestDevCheckFailurePropagation:
    """A non-zero subprocess exit code must make dev_check exit 1."""

    def test_exits_nonzero_on_subprocess_failure(self) -> None:
        """When _run_step returns (False, ...), main() must exit with code 1."""
        mod = _load_dev_check()

        # Patch _run_step so the first step always fails
        def _always_fail(step, verbose):  # noqa: ANN001
            return False, "mock failure output"

        with patch.object(mod, "_run_step", side_effect=_always_fail):
            with patch("sys.argv", ["dev_check.py", "--fast"]):
                with pytest.raises(SystemExit) as exc_info:
                    mod.main()

        assert exc_info.value.code == 1, (
            f"Expected SystemExit(1), got SystemExit({exc_info.value.code})"
        )


@pytest.mark.unit
class TestDevCheckAllPassProducesZero:
    """When every step passes, main() must exit with code 0."""

    def test_exits_zero_when_all_pass(self) -> None:
        """When _run_step always returns (True, ''), main() exits 0."""
        mod = _load_dev_check()

        def _always_pass(step, verbose):  # noqa: ANN001
            return True, ""

        # Run in fast mode to avoid needing bandit/ruff installed
        with patch.object(mod, "_run_step", side_effect=_always_pass):
            with patch("sys.argv", ["dev_check.py", "--fast"]):
                with pytest.raises(SystemExit) as exc_info:
                    mod.main()

        assert exc_info.value.code == 0, (
            f"Expected SystemExit(0), got SystemExit({exc_info.value.code})"
        )
