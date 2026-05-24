# Mutation Testing

## What is Mutation Testing?

Mutation testing systematically introduces small code changes (mutants) and checks
whether the existing test suite detects them. Unlike line coverage which only
measures "was this code executed", mutation testing measures "was this logic
actually verified by assertions".

A **killed** mutant means tests caught the change. A **survived** mutant means
no test noticed the change, revealing a test gap.

Tool: [mutmut](https://github.com/boxed/mutmut) (v2.5.x, pinned below v3 for
Windows compatibility).

## Targeted Modules

We run mutation testing on 5 critical modules rather than the entire codebase
(which would take hours and produce excessive noise):

| # | Module | Rationale |
|---|--------|-----------|
| 1 | `application/language/` | Language detection + sticky logic (core UX) |
| 2 | `application/language/stream_guard.py` | Real-time stream language enforcement |
| 3 | `main.py` (`_sentry_before_send`) | Privacy filter preventing PII leaks to Sentry |
| 4 | `application/skill_compression/privacy/` | 3-filter pipeline (healthcare, nudge, secrets) |
| 5 | `application/rate_limiter.py` | Abuse prevention, quota enforcement |

## Running Locally

From repo root:

```bash
# Single module
bash scripts/mutation_test_language.sh
bash scripts/mutation_test_stream_guard.sh
bash scripts/mutation_test_sentry.sh
bash scripts/mutation_test_privacy.sh
bash scripts/mutation_test_rate_limiter.sh
```

On Windows (without WSL), run directly:

```powershell
cd bridge
$env:PYTHONIOENCODING = "utf-8"
mutmut run --paths-to-mutate "application/language/" --tests-dir "tests/" --runner "python -m pytest tests/test_application/test_language/ -x -q --tb=no"
mutmut results
```

## CI Schedule

`.github/workflows/mutation-testing.yml` runs weekly on Sundays at 06:00 UTC.
Each module runs in a separate job (parallel). Results are uploaded as GitHub
Actions artifacts under `mutation-cache-*`.

**Important:** `continue-on-error: true` ensures mutation testing never blocks
merges. It is reporting-only.

## Triaging Survived Mutants

When reviewing survived mutants, classify each as:

### 1. Real Test Gap
The mutation changes behavior that should be tested but is not.
**Action:** Add a test in the next sprint.

### 2. Equivalent Mutant (False Positive)
The mutation produces semantically identical behavior.
Examples:
- `range(n)` -> `range(0, n)`
- `x >= 0` -> `x > -1` (for integers)
- Reordering commutative operations

**Action:** Mark as false positive, no test needed.

### 3. Real Bug Found
The mutation reveals that production code has a logic error that tests
also miss (both original and mutant pass tests because the logic was
already wrong).
**Action:** STOP. File a bug. Fix production code AND add test.

## Mutation Score Goals

Target: >= 80% killed per module.

| Module | Score | Killed | Survived | Timeout | Date |
|--------|-------|--------|----------|---------|------|
| language/ (core) | 38.6%* | 32 | 51 | 0 | 2026-05-24 |
| stream_guard.py | 59.6% | 155 | 105 | 0 | 2026-05-24 |
| main.py (sentry only) | 95.8% | 69 | 3 | 0 | 2026-05-24 |
| privacy/ | 38.0% | 174 | 284 | 0 | 2026-05-24 |
| rate_limiter.py | 68.5% | 265 | 122 | 0 | 2026-05-24 |

*Language core score is partial (83/374 mutants tested due to run time constraints
and pre-existing enforcement test failures). Full run estimated at 60+ minutes.

## How to Extend

To add a new module to mutation testing:

1. Create `scripts/mutation_test_<module>.sh`
2. Add a job in `.github/workflows/mutation-testing.yml`
3. Update the score table above
4. Document survived mutants in `docs/MUTATION_FINDINGS.md`
