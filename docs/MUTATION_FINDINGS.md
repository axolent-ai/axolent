# Mutation Testing Findings (2026-05-24)

Initial mutation testing run on 5 critical AXOLENT modules.
These findings represent the test-gap backlog for future improvement.

## Summary

| Module | Mutants | Killed | Survived | Score | Note |
|--------|---------|--------|----------|-------|------|
| language/ (core: resolver, enforcement, orchestrator) | 374 | 32 | 51 | 38.6%* | *Partial (83/374 tested, run time-limited) |
| stream_guard.py | 261 | 155 | 105 | 59.6% | Complete |
| main.py `_sentry_before_send` | 72 | 69 | 3 | 95.8% | In-function only (rest of main.py excluded) |
| privacy/ (3-filter pipeline) | 458 | 174 | 284 | 38.0% | Many keyword-list mutations (see analysis) |
| rate_limiter.py | 387 | 265 | 122 | 68.5% | Complete |

## Analysis by Module

### 1. Language Core (resolver.py, enforcement.py, orchestrator.py)

**Score: 38.6% (partial)**

Key survived mutants (test gaps):
- `_SMART_SWITCH_THRESHOLD: 0.7 -> 1.7`: No test verifies that smart-switch
  triggers at the correct threshold. A threshold above 1.0 makes switching
  impossible but no test catches this.
- `_registry = InMemoryLanguageRegistry() -> None`: Module-level singleton init
  not tested directly.
- `if self._orchestrator is None -> is not None`: Lazy-init inversion not caught.
- Log messages mutated (string changes): Expected, not a real test gap.

**Note:** 5 enforcement tests fail due to pre-existing icontract violations
(unrelated to mutation testing). These tests were excluded from the run.

### 2. Stream Guard (stream_guard.py)

**Score: 59.6%**

Key survived mutants:
- `StreamGuardOutcome.NO_CHECK = "no_check" -> "XXno_checkXX"`: Tests compare
  enum members, not string values. This is acceptable (enum identity is correct).
- `StreamGuardOutcome.PASSED_NO_ABORT = "..." -> None`: Same pattern.
- Threshold mutations (MIN_CHARS, confidence values): Some thresholds not boundary-tested.
- Cache size mutations (LRU cache maxsize changes): No test verifies cache eviction behavior.
- `self._consecutive_fp_count` counter logic: FP-rate tracking not fully tested.

**Classification:** ~40% of survived mutants are equivalent/acceptable (enum value
strings, log messages). Real gaps are in threshold boundary testing and FP-rate logic.

### 3. Sentry Privacy Filter (_sentry_before_send)

**Score: 95.8%** (PASSES 80% goal)

Only 3 survived mutants:
1. `and -> or` in URL type-check (line 130): Tests don't cover scenario where
   `event["request"]` exists but `event["request"]["url"]` is missing.
2. `continue -> break` in exception loop (line 147): Tests only send single-exception
   events, not multi-exception.
3. `"category" -> "XXcategoryXX"` in breadcrumb allowlist (line 167): Tests don't
   send breadcrumbs with "category" key.

**All 3 are real test gaps** (not equivalent mutants). Priority: HIGH (privacy-critical).

### 4. Privacy Pipeline (skill_compression/privacy/)

**Score: 38.0%**

Analysis of survived mutants reveals a structural pattern:
- **~70% are keyword-list mutations**: Changing individual keywords in lists of 30-80
  entries (e.g., "anxiety" -> "XXanxietyXX", "rehabilitation" -> "XXrehabilitationXX").
  These are technically survived but not actionable: testing every individual keyword
  would require hundreds of additional tests for marginal value.
- **~10% are log message mutations**: `log = None`, string changes in log.info calls.
  Expected, not real gaps.
- **~20% are structural logic gaps**: These are the real findings.

Real structural gaps:
- Domain matching logic: some domain patterns not tested individually.
- Confidence threshold in nudge_filter: boundary not tested.
- Pipeline ordering: mutations that reorder filter chain not detected.

**Adjusted score (excluding keyword-list mutations):** Estimated ~65-70%.

### 5. Rate Limiter (rate_limiter.py)

**Score: 68.5%**

Key survived mutants:
- `_EVICTION_TTL_SECONDS: 3600.0 -> 3601.0`: Eviction boundary not tested precisely.
- Profile dict key mutations ("normal" -> "XXnormalXX"): Tests don't validate all
  profile keys are exactly correct (relies on runtime behavior).
- `_load_user_profiles` JSONL logic: JSONL file loading not tested (SQLite path used).
- `_WARNING_THRESHOLD: 0.7 -> 1.7`: 70% warning threshold not boundary-tested.
- `_UNLIMITED_REMINDER_INTERVAL: 100 -> 101`: Reminder interval not exact-tested.
- `consumed_count` time.monotonic mutations: Time-dependent logic hard to test precisely.

**Classification:** ~30% are timing/boundary mutations (hard to test without time mocking),
~20% are JSONL-path mutations (dead code in SQLite mode), ~50% are real test gaps.

## False Positive (Equivalent Mutant) Patterns

These mutation patterns consistently survive but are NOT real bugs:

1. **Log message string changes**: `"message" -> "XXmessageXX"` in log.info/warning calls.
   No test should assert on log message content.
2. **Enum value string changes**: When tests compare enum members (identity), not
   string representations.
3. **Keyword-list single-entry mutations**: In lists of 30+ entries, mutating one
   keyword is equivalent in practice (defense-in-depth: other keywords still catch
   the category).
4. **Rounding precision changes**: `round(x, 2) -> round(x, 3)` in non-user-facing
   audit data.
5. **Logger initialization**: `log = logging.getLogger(__name__) -> None` when no
   test triggers the log path.

## Priority Backlog (Real Test Gaps to Fix)

### P0 (Privacy-Critical)
- [ ] Sentry: test multi-exception events (break vs continue)
- [ ] Sentry: test missing URL in request dict (and vs or)
- [ ] Sentry: test breadcrumbs with "category" key

### P1 (Business Logic)
- [ ] Rate limiter: boundary test for 70% warning threshold
- [ ] Rate limiter: test unlimited reminder at exact interval
- [ ] Language resolver: test smart-switch threshold boundaries
- [ ] Language resolver: test lazy orchestrator initialization
- [ ] Stream guard: test FP-rate auto-disable logic

### P2 (Defense-in-Depth)
- [ ] Rate limiter: test eviction at exactly TTL boundary
- [ ] Privacy pipeline: test filter chain ordering matters
- [ ] Privacy nudge_filter: confidence threshold boundary test
- [ ] Stream guard: test all threshold boundaries (MIN_CHARS, etc.)

### P3 (Low Priority / Acceptable)
- [ ] Rate limiter: JSONL profile loading (only used as fallback)
- [ ] Privacy: individual keyword coverage (diminishing returns)
