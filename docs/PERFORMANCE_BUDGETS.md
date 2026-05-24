# Performance Budgets

## What Are Performance Budgets?

Performance budgets define the maximum acceptable latency for each critical operation in AXOLENT Bridge. If an operation exceeds its budget, it indicates either:

1. A performance regression introduced by a code change
2. An architectural issue that needs investigation
3. A budget that needs recalibration (with documented justification)

## Why Performance Budgets?

- **User Experience:** Operations like language detection (every message) and rate limiting (every request) must be imperceptible to users. A 50ms delay on every message compounds into noticeable sluggishness.
- **Regression Detection:** Without explicit budgets, performance degrades silently over time ("boiling frog"). Budgets make regressions immediately visible.
- **Architecture Accountability:** Forces engineers to think about the performance cost of abstractions before merging them.

## Budget Definitions

All budgets are defined in `bridge/tests/test_performance/budgets.yaml`.

| Operation | Budget (ms) | Rationale |
|-----------|-------------|-----------|
| `language_detection_short_text` | 10 | Called on every user message; must be imperceptible |
| `language_resolver_resolve` | 20 | Includes detection + async storage lookup + write |
| `language_resolver_resolve_readonly` | 10 | Read-only path, no storage write |
| `privacy_pipeline_check` | 50 | Full 3-filter pipeline; runs on hypothesis promotion |
| `healthcare_filter_check` | 15 | Individual filter budget |
| `secret_scanner_check` | 15 | Individual filter budget |
| `nudge_filter_check` | 15 | Individual filter budget |
| `sentry_before_send` | 1 | Called on every error event; must never delay error reporting |
| `sqlite_connection_query_simple` | 5 | Simple indexed SELECT (hot path) |
| `memory_storage_retrieve_user` | 30 | Read + filter + sort for up to 100 entries |
| `rate_limiter_check` | 2 | Called on every incoming message |
| `stream_guard_classify` | 10 | Full abort cycle: check_early + classify (2x langdetect calls) |

## Initial Baseline Measurements (2026-05-24, local dev machine)

| Operation | Measured (avg/call) | Budget | Headroom |
|-----------|--------------------:|-------:|---------:|
| `language_detection_short_text` | 0.009ms | 10ms | 1111x |
| `language_resolver_resolve_readonly` | 0.055ms | 10ms | 182x |
| `language_resolver_resolve` | 6.972ms | 20ms | 2.9x |
| `privacy_pipeline_check` | 0.071ms | 50ms | 704x |
| `healthcare_filter_check` | 0.019ms | 15ms | 789x |
| `secret_scanner_check` | 0.012ms | 15ms | 1250x |
| `nudge_filter_check` | 0.026ms | 15ms | 577x |
| `sentry_before_send` | 0.038ms | 1ms | 26x |
| `sqlite_connection_query_simple` | 0.017ms | 5ms | 294x |
| `memory_storage_retrieve_user` | 0.867ms | 30ms | 35x |
| `rate_limiter_check` | 0.004ms | 2ms | 500x |
| `stream_guard_classify` | 5.577ms | 10ms | 1.8x |

Notes:
- `language_resolver_resolve` is the tightest (2.9x headroom) because it includes the langdetect library call on each iteration
- `stream_guard_classify` has minimal headroom (1.8x) because it calls langdetect twice (check + classify)
- Most pure-Python operations have massive headroom (>100x); budgets are set for UX-acceptability, not current measurement

## How Budgets Were Established

1. **Measure baseline:** Run the operation 100 times on a representative machine, take P95.
2. **Apply headroom:** Set budget to 2x P95 (accounts for system load variance).
3. **Validate UX-acceptability:** Verify the budget is within acceptable user experience thresholds (e.g., language detection under 10ms is imperceptible in a chat interface).

## How Tests Work

### Test Structure

```
bridge/tests/test_performance/
    __init__.py
    budgets.yaml          # Budget definitions (source of truth)
    conftest.py           # PerfTimer helper + fixtures
    test_budgets.py       # One test per budget (12+ tests)
```

### Running Locally

```bash
cd bridge
pytest tests/test_performance/ -v -m performance
```

### CI Behavior

Performance tests run in `.github/workflows/performance.yml`:
- **Weekly schedule:** Every Monday 05:00 UTC
- **Path-triggered PRs:** When `bridge/domain/`, `bridge/application/`, or `bridge/infrastructure/` changes
- **Manual dispatch:** Via `workflow_dispatch`
- **Not blocking:** `continue-on-error: true` (informational)

Performance tests are explicitly NOT in `pr-check.yml` because shared CI runners have unpredictable load, causing false positives.

### Test Design

- **Warmup:** Each test runs the operation 1-2 times before measurement to avoid cold-start costs (imports, JIT, cache warming).
- **100-iteration averaging:** Statistical robustness against single-run outliers.
- **`@pytest.mark.performance`:** Tests are excluded from default `pytest` runs via marker.

## How to Adjust a Budget

1. **Measure:** Run the test 5 times locally. Document P50, P95, P99.
2. **Justify:** Why does the operation need more time? Is it a new feature, additional safety check, or genuine regression?
3. **Update `budgets.yaml`:** Change `max_ms` with a comment explaining why.
4. **Review:** Get a second engineer to review the change.
5. **Document:** Add a note in the YAML comment above the budget entry.

Example:
```yaml
  # Increased from 10ms to 15ms after adding Phase 2 orchestrator
  # fallback logic (2026-05-24). Baseline P95 = 7ms, budget = 2x.
  language_detection_short_text:
    max_ms: 15
    description: "detect_language() for text under 100 chars"
```

## What Happens on Budget Violation

### Locally

The test fails with a clear message:
```
FAILED: PERF BUDGET EXCEEDED: language_detection_short_text: 12.34ms > 10.00ms budget
```

### In CI (weekly/PR)

The workflow logs the violation but does NOT block the PR (`continue-on-error: true`). This is intentional:
- Shared CI runners have variable performance
- A 15% overshoot on CI may be runner noise, not a real regression
- Engineers should investigate violations but are not blocked

### When to Treat a Violation as Critical

- Violation is **3x+ over budget** consistently (not runner noise)
- Violation appears in **local runs** (not just CI)
- Violation correlates with a specific code change (bisectable)
- Multiple budgets fail simultaneously (systemic regression)

## Known Limitations

1. **Shared CI variance:** GitHub Actions runners share resources. Performance tests may flap. This is why `continue-on-error: true` is set.
2. **System load sensitivity:** Local runs on a busy machine may exceed budgets. Close heavy applications before running perf tests.
3. **Windows vs Linux:** Some operations (file I/O, process spawning) differ by platform. Budgets are calibrated for the development machine (Windows) but CI runs on Linux.
4. **Async overhead:** `asyncio.get_event_loop().run_until_complete()` adds ~0.1ms overhead per call in tests. This is accounted for in the budgets.
