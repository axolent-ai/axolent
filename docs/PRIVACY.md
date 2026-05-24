# Privacy: Logging, Audit, Reports & CI Artifacts

This document specifies what AXOLENT logs, stores, and exposes through
each output channel. It serves as a binding contract for privacy-by-design.

## 1. Audit Log (`bridge/logs/audit.jsonl`)

### What IS stored

| Field | Description | Example |
|-------|-------------|---------|
| `timestamp` | ISO-8601 UTC | `2026-05-24T10:00:00+00:00` |
| `user_id` | Telegram user ID (int) | `123456789` |
| `chat_id` | Telegram chat ID (int) | `-100123456` |
| `username` | Telegram username | `@example_user` |
| `prompt_length` | Character count of user message | `142` |
| `response_length` | Character count of bot response | `580` |
| `duration_seconds` | Processing time | `2.31` |
| `detected_language` | Language code | `de` |
| `provider` | LLM provider name | `claude` |
| `task_slot` | Task classification | `CHAT` |
| `task_score` | Classification confidence | `85` |
| `error` | Error type (technical) | `timeout` |
| `error_id` | Error correlation ID | `a1b2c3d4` |

### What is NEVER stored

- Raw user message text (only `prompt_length`)
- Raw bot response text (only `response_length`)
- System prompts
- Memory content
- Conversation history
- API keys or tokens

### Rotation

- Max 10 MB per file, 5 backup files (RotatingFileHandler)
- Total maximum disk usage: ~60 MB
- Old entries are overwritten, not archived

## 2. Privacy Audit Log (Skill Compression, in-memory)

The `PrivacyAuditLog` in `privacy_pipeline.py` is an in-memory ring buffer
that tracks privacy filter rejections.

### What IS stored (per rejection)

| Field | Description |
|-------|-------------|
| `hypothesis_id` | Unique ID of the rejected hypothesis |
| `source` | Which filter rejected (`healthcare_filter`, `secret_scanner`, `nudge_filter`) |
| `reason` | Generic filter description (e.g., "Health-related domain in scope: 'medical'") |
| `timestamp` | ISO-8601 UTC |

### What is NEVER stored

- The hypothesis claim text (the actual user-derived content)
- The original user message that generated the hypothesis
- Matched keywords or patterns from user text

### Persistence

- In-memory only (not persisted to disk or DB)
- Max 1000 entries, rotated at 50% when full
- Lost on process restart (by design)

## 3. Language Detection Audit (`axolent.language.audit` logger)

Hard Constraint HC-D1: **Input text is NEVER stored.**

### What IS logged

- `input_text_length` (character count only)
- `text_length_bucket` (micro/short/medium/long)
- `detected_code`, `confidence`, `reliability_score`
- `backends_consulted`, `candidates` (backend metadata)
- `source` (override/sticky/detected/default)
- `decision_reason` (from detection engine, not user text)

### What is NEVER logged

- The input text itself (enforced at call-site: text is never passed to `build_audit_event()`)

## 4. Stream Guard Audit

### What IS stored

- `event_type`, `expected_lang`, `check_performed`, `aborted`, `disabled`
- `detected_at_abort`, `abort_confidence`, `partial_length`
- `outcome`, `partial_verification_confidence`, `disable_reason`

### What is NEVER stored

- `accumulated_text` (explicitly excluded, documented in source)
- Any portion of the streaming response text

## 5. Application Logger (Python `logging`)

### Design Principles

- User message text is never logged at INFO or WARNING level
- User IDs (integers) are logged for operator debugging (intentional)
- Exception messages (`str(e)`) may appear in ERROR logs; these originate
  from providers/libraries and do not echo user input
- Debug-level logs with `exc_info=True` may contain stack traces but are
  disabled in production (`logging.WARNING` default)

### Known Acceptable Patterns

- `log.info("Language for chat: '%s' (source=%s)", lang, source)` - language code, not text
- `log.info("Skill applied: hyp=%s confidence=%.3f", hyp_id, conf)` - IDs, not claims
- `log.debug("Judge raw response (%d chars): %s", len(text), text[:500])` - LLM judge output at DEBUG only

## 6. Exported Reports

**AXOLENT does not currently export HTML, JSON, or Markdown reports to disk.**

The only file-based output is the audit JSONL log (Section 1).

If report export is added in the future, it must:
1. Pass the `TestExportedReportsPrivacy` test suite
2. Exclude all raw user text
3. Use only aggregated metrics and IDs

## 7. GitHub Actions Artifacts

### Workflow Output Policy

- `daily-smoke.yml`: Runs pytest with `--tb=short -q`. Output contains only
  test names and pass/fail status. No user text in test fixtures (all synthetic).
- No `upload-artifact` steps that persist logs or audit data.
- `pytest` xfail reasons are generic technical descriptions.
- Smoke test (`scripts/smoke_test.py`) uses synthetic test messages only.

### CI Secret Protection

- Bot tokens and API keys are in GitHub Secrets (never in logs)
- `secret-scan.yml` (TruffleHog) runs on every push to detect leaked secrets
- Sentry `_sentry_before_send` strips all user-controlled data from events
  before transmission (tested in `test_K8_sentry_pii.py`)

## 8. Audit Key Registry (chat_service.py)

All keys written to the audit log from `chat_service.py` are enumerated
and tested in `test_privacy_logs.py::test_no_pii_leak_across_chat_service_to_audit`.

Adding a new audit key requires:
1. Adding the key to the `safe_patterns` set in the test
2. Confirming the value does not contain user text
3. If the key carries user-derived content, it MUST be rejected

Current registry (as of 2026-05-24):
```
timestamp, user_id, chat_id, username, prompt_length, response_length,
duration_seconds, detected_language, history_turns, memory_entries_loaded,
provider, error, error_id, leakage_attempt, language_enforced,
task_slot, task_score, task_matched_patterns, task_matched_keywords,
resolved_model, fallback_used, fallback_level, fallback_provider,
skill_applied, skill_confidence, skill_matched_ask_before,
skill_applied_streaming, skill_matched_ask_before_streaming,
plan_type, plan_provider_chain, plan_memory_ids
```
