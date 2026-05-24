"""Performance-budget test helpers.

Provides:
    - budgets fixture: loads budgets.yaml (session-scoped)
    - PerfTimer: context manager that fails if average-per-iteration > budget
    - perf_timer fixture: factory for PerfTimer from budget name
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml


BUDGETS_PATH = Path(__file__).parent / "budgets.yaml"

# Default number of iterations inside the timed block.
# The budget is compared against (elapsed / iterations).
DEFAULT_ITERATIONS = 100


@pytest.fixture(scope="session")
def budgets() -> dict:
    """Load performance budgets from YAML (session-scoped, loaded once)."""
    with BUDGETS_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["budgets"]


class PerfTimer:
    """Context manager that measures elapsed time and fails on budget violation.

    The budget is compared against the AVERAGE per iteration:
        avg_ms = total_elapsed_ms / iterations

    Usage:
        with PerfTimer("op", budget_ms=10.0, iterations=100) as timer:
            for _ in range(100):
                do_stuff()
        # Fails if avg per iteration > 10ms

    Attributes:
        label: Human-readable operation name.
        budget_ms: Maximum allowed milliseconds PER ITERATION.
        iterations: Number of iterations inside the block.
        elapsed_ms: Total elapsed milliseconds (set on __exit__).
        avg_ms: Average per-iteration milliseconds (set on __exit__).
    """

    __slots__ = ("label", "budget_ms", "iterations", "elapsed_ms", "avg_ms", "_start")

    def __init__(
        self, label: str, budget_ms: float, iterations: int = DEFAULT_ITERATIONS
    ) -> None:
        self.label = label
        self.budget_ms = budget_ms
        self.iterations = iterations
        self.elapsed_ms: float | None = None
        self.avg_ms: float | None = None
        self._start: float = 0.0

    def __enter__(self) -> "PerfTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.avg_ms = self.elapsed_ms / self.iterations
        if self.avg_ms > self.budget_ms:
            pytest.fail(
                f"PERF BUDGET EXCEEDED: {self.label}: "
                f"avg {self.avg_ms:.2f}ms/iter > {self.budget_ms}ms budget "
                f"(total {self.elapsed_ms:.2f}ms over {self.iterations} iterations)"
            )


@pytest.fixture
def perf_timer(budgets: dict):
    """Factory fixture: create a PerfTimer for a named budget.

    Usage in tests:
        def test_something(perf_timer):
            with perf_timer("language_detection_short_text"):
                for _ in range(100):
                    detect_language("hello")

    The budget from YAML is per-iteration. The timer divides total
    elapsed by iterations (default 100) before comparing.
    """

    def _make(label: str, iterations: int = DEFAULT_ITERATIONS) -> PerfTimer:
        if label not in budgets:
            pytest.fail(
                f"Unknown budget '{label}'. Available: {sorted(budgets.keys())}"
            )
        budget = budgets[label]
        return PerfTimer(label, budget["max_ms"], iterations=iterations)

    return _make
