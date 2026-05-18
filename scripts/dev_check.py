#!/usr/bin/env python3
"""One-command quality gate for Axolent Bridge contributors.

Runs all pre-push checks in order and reports clear Pass/Fail for each step.
Exits with code 0 only when every mandatory step passes.

Usage:
    python scripts/dev_check.py            # full check
    python scripts/dev_check.py --fast     # skip bandit + ruff (quick loop)
    python scripts/dev_check.py --verbose  # show full subprocess output

Steps:
    1. i18n parity check          (scripts/i18n_check.py)
    2. i18n AST scan              (scripts/i18n_scan.py)
    3. English-only production    (scripts/check_en_only_production.py)
    4. No fake umlauts            (scripts/check_no_fake_umlauts.py)
    5. pytest                     (scripts/run_with_venv.py pytest -q --no-header)
    6. bandit security linter     (python -m bandit -r bridge/)  [skipped with --fast]
    7. ruff linter                (python -m ruff check bridge/) [skipped with --fast]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Repository root detection
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_BRIDGE_DIR = _REPO_ROOT / "bridge"


# ---------------------------------------------------------------------------
# Step definition
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """Definition of a single check step."""

    label: str
    cmd: list[str]
    cwd: Path
    optional: bool = False
    description: str = ""


def _build_steps(fast: bool) -> list[Step]:
    """Return the ordered list of steps.

    Args:
        fast: When True, the optional bandit/ruff steps are excluded.

    Returns:
        Ordered list of Step objects.
    """
    python = sys.executable
    run_with_venv = str(_SCRIPTS_DIR / "run_with_venv.py")

    mandatory: list[Step] = [
        Step(
            label="i18n parity check",
            cmd=[python, str(_SCRIPTS_DIR / "i18n_check.py")],
            cwd=_REPO_ROOT,
            description="Verifies all locale JSON files have identical keys and source hashes.",
        ),
        Step(
            label="i18n AST scan",
            cmd=[python, str(_SCRIPTS_DIR / "i18n_scan.py")],
            cwd=_REPO_ROOT,
            description="AST scanner for hardcoded user-facing strings not using t().",
        ),
        Step(
            label="English-only production check",
            cmd=[python, str(_SCRIPTS_DIR / "check_en_only_production.py")],
            cwd=_REPO_ROOT,
            description="Detects German text in production source files.",
        ),
        Step(
            label="No fake umlauts check",
            cmd=[python, str(_SCRIPTS_DIR / "check_no_fake_umlauts.py")],
            cwd=_REPO_ROOT,
            description="Detects ASCII umlaut substitutions (ae/oe/ue/ss) in production files.",
        ),
        Step(
            label="pytest",
            cmd=[python, run_with_venv, "pytest", "-q", "--no-header"],
            cwd=_REPO_ROOT,
            description="Full test suite via venv pytest.",
        ),
    ]

    optional: list[Step] = [
        Step(
            label="bandit security linter",
            cmd=[python, "-m", "bandit", "-r", str(_BRIDGE_DIR)],
            cwd=_REPO_ROOT,
            optional=True,
            description="Static security analysis. Install: pip install bandit",
        ),
        Step(
            label="ruff linter",
            cmd=[python, "-m", "ruff", "check", str(_BRIDGE_DIR)],
            cwd=_REPO_ROOT,
            optional=True,
            description="Fast Python linter. Install: pip install ruff",
        ),
    ]

    if fast:
        return mandatory
    return mandatory + optional


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _print_header(step_number: int, total: int, label: str) -> None:
    """Print a clearly visible step header.

    Args:
        step_number: 1-based index of the current step.
        total: Total number of steps.
        label: Human-readable step name.
    """
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Step {step_number}/{total}: {label}")
    print(f"{bar}")


def _run_step(step: Step, verbose: bool) -> tuple[bool, str]:
    """Run a single step and return (passed, output).

    Args:
        step: The step to run.
        verbose: If True, streams output live; otherwise captures it.

    Returns:
        Tuple of (passed: bool, captured_output: str).
        When verbose=True the output has already been printed; the string
        is still returned for the summary.
    """
    if verbose:
        result = subprocess.run(  # noqa: S603
            step.cmd,
            cwd=str(step.cwd),
        )
        return result.returncode == 0, ""
    else:
        result = subprocess.run(  # noqa: S603
            step.cmd,
            cwd=str(step.cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output


def _print_result(passed: bool, output: str, verbose: bool, step: Step) -> None:
    """Print the pass/fail result for a completed step.

    Args:
        passed: Whether the step succeeded.
        output: Captured subprocess output (empty when verbose=True).
        verbose: If True, output was already streamed live.
        step: The step that was just run.
    """
    if passed:
        print(f"  OK  {step.label}")
    elif not passed and step.optional:
        print(f"  SKIP  {step.label}  (optional, not installed)")
        if not verbose and output:
            print()
            print("  --- Output ---")
            for line in output.splitlines():
                print(f"  {line}")
            print()
        print(f"  Hint: {step.description}")
    else:
        print(f"  FAIL  {step.label}")
        if not verbose and output:
            print()
            print("  --- Output ---")
            for line in output.splitlines():
                print(f"  {line}")
            print()
        print("  Hint: Fix the above errors then re-run: python scripts/dev_check.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="One-command quality gate for Axolent Bridge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip bandit and ruff (faster feedback loop, skips steps 6 and 7).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full subprocess output for every step in real time.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all quality-gate steps and exit with 0 on full pass, 1 on any failure."""
    args = _parse_args()
    steps = _build_steps(fast=args.fast)
    total = len(steps)

    mode_note = " (fast mode: bandit + ruff skipped)" if args.fast else ""
    print(f"\nAxolent dev_check{mode_note}")
    print(f"Running {total} steps...\n")

    failures: list[tuple[int, str]] = []

    for i, step in enumerate(steps, start=1):
        _print_header(i, total, step.label)

        passed, output = _run_step(step, verbose=args.verbose)
        _print_result(passed, output, verbose=args.verbose, step=step)

        if not passed and not step.optional:
            failures.append((i, step.label))

    # Final summary
    print("\n" + "=" * 60)
    if not failures:
        print("  All checks passed. Ready for commit/push.")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"  {len(failures)} check(s) FAILED:")
        for step_num, label in failures:
            print(f"    Step {step_num}: {label}")
        print()
        print("  Fix the failures above, then re-run:")
        print("    python scripts/dev_check.py")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
