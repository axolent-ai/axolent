# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AXOLENT AI, please report it responsibly.

**Email:** security@axolent.ai

**What to include:**

* Description of the vulnerability
* Steps to reproduce
* Potential impact assessment
* Suggested fix (if any)

**Response timeline:**

* Acknowledgment within 48 hours
* Initial assessment within 5 business days
* Fix timeline communicated within 10 business days

**Please do NOT:**

* Open a public GitHub issue for security vulnerabilities
* Exploit the vulnerability beyond what is necessary for a proof of concept
* Access or modify other users' data

We will credit reporters in the release notes (unless they prefer to remain anonymous).

## Supported Versions

| Version | Supported |
|---------|-----------|
| Phase 1 (current) | Yes |
| Pre-release / dev | Best effort |

## Security Architecture

### Mode B: No Token Hijacking

AXOLENT AI operates in **Mode B**: a local CLI wrapper that spawns the official
`claude` CLI as a subprocess on your machine. Your existing Claude Pro/Max
subscription handles inference.

This means:

* **No OAuth token hijacking.** The bot never reads, stores, or proxies your
  Anthropic OAuth tokens.
* **No API key in code.** There is no Anthropic API key anywhere in the codebase.
* **No cloud proxy.** There is no server between you and Anthropic. All inference
  runs locally via your own subscription.
* **No SDK imports.** The codebase does not import the Anthropic Python SDK. All
  interaction with Claude goes through the CLI subprocess.

See [docs/adr/0001-mode-b-local-cli-wrapper.md](docs/adr/0001-mode-b-local-cli-wrapper.md)
for the architectural decision record.

### Access Control

* **Whitelist (fail-closed):** Only explicitly authorized Telegram user IDs can
  interact with the bot. Configured via `WHITELIST_USER_IDS` in `.env`.
* **ALLOW_ALL_USERS safeguard:** Opening the bot to all users requires setting
  both `ALLOW_ALL_USERS=true` and `AXOLENT_DEV_MODE=true`. Without the companion
  flag, the bot refuses to start (tripwire guard).
* **Private chat enforcement:** Sensitive commands are restricted to 1:1 chats
  via the `@require_private_chat` decorator.

### Rate Limiting

Four profiles (Light, Normal, Power, Unlimited) with per-user fixed-window counters.
Prevents abuse even from whitelisted users. Profiles persist across restarts.

### Audit Logging

Every LLM call, command invocation, and rate-limit event is written to a JSONL audit
log with rotation. Audit entries include `request_id` for correlation across the
request lifecycle.

### Data Protection

* **System prompt leakage guard:** Two-layer defense (instruction in prompt plus
  substring-based output filter) prevents the system prompt from appearing in
  bot responses.
* **Error redaction:** Users see only generic error IDs. Stack traces and internal
  details appear only in the local log file.
* **No data exfiltration:** All storage (SQLite, JSONL) is local to your machine.

### Supply Chain Security

* **Pre-commit hooks (17 active):** Every commit is scanned by ruff, bandit, semgrep,
  pip-audit, import-linter, and pytest before it can land.
* **gitleaks configuration:** `.gitleaks.toml` is configured for secrets scanning.
  The pre-commit hook is currently disabled on Windows due to an upstream wasm bug,
  but manual scanning is recommended before releases:
  ```bash
  gitleaks detect --source . --config .gitleaks.toml
  ```
* **Dependency auditing:** `pip-audit` checks all dependencies for known vulnerabilities
  on every commit.
* **SAST:** Bandit (Python-specific) and Semgrep (2000+ rules) scan for security
  anti-patterns.

### Consciously Accepted Risks (Phase 1)

These are documented in detail in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md):

* **R-1:** SQLite storage without encryption-at-rest (Phase 1+: Fernet).
* **R-2:** Conversation history is in-memory only (lost on restart).
* **R-3:** Audit log without tamper protection (Phase 1+: hash chain).
* **R-5:** System prompt leakage guard is heuristic (~60-70% coverage).
* **R-7:** Memory translation sends content to LLM provider (user's own subscription,
  disableable via `AXOLENT_MEMORY_TRANSLATION=false`).

## Threat Model

The full threat model, including adversary model, trust boundaries, and planned
mitigations, is documented in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## OWASP LLM Top 10 Test Suite

AXOLENT includes a dedicated security test suite mapped to the
[OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

### Critical Categories for AXOLENT (Mode-B Telegram Bot)

| OWASP ID | Category | Tests | Status |
|----------|----------|-------|--------|
| LLM01 | Prompt Injection | 7 | All pass |
| LLM02 | Insecure Output Handling | 5 | All pass |
| LLM06 | Sensitive Information Disclosure | 6 | All pass |
| LLM07 | System Prompt Leakage | 5 | All pass |
| LLM08 | Vector and Embedding Weaknesses | 4 | All pass |
| LLM09 | Misinformation | 4 | All pass |
| LLM10 | Unbounded Consumption | 5 | All pass |

### Less Critical (covered by other mechanisms)

* **LLM03 Supply Chain:** Covered by TruffleHog (Item 3), Dependabot (Item 7),
  pip-audit, and gitleaks in pre-commit hooks.
* **LLM04 Data Poisoning:** Not applicable (Mode-B, no custom model training).
* **LLM05 Improper Output Handling:** Subset of LLM02, covered there.

### Running the Security Tests

```bash
cd bridge
pytest tests/test_security/ -v -m security
```

### Test Location

All security tests reside in `bridge/tests/test_security/`:

* `test_owasp_llm01_prompt_injection.py` (7 tests)
* `test_owasp_llm02_insecure_output.py` (5 tests)
* `test_owasp_llm06_sensitive_disclosure.py` (6 tests)
* `test_owasp_llm07_system_prompt_leak.py` (5 tests)
* `test_owasp_llm08_vector_embedding.py` (4 tests)
* `test_owasp_llm09_misinformation.py` (4 tests)
* `test_owasp_llm10_unbounded_consumption.py` (5 tests)
* `conftest.py` (shared fixtures: injection payloads, PII samples, isolated stores)

### CI Integration

The security suite runs as a dedicated step in `.github/workflows/pr-check.yml`
after the full test suite. Both must pass for a PR to merge.

### XFAIL Policy

Tests marked `@pytest.mark.xfail` represent known gaps with documented reasons.
Each XFAIL entry includes:

* A clear `reason` string explaining the gap
* `strict=True` so the test fails if the gap is accidentally fixed (prompting
  the XFAIL to be removed)
* A roadmap reference for when the fix is planned

Currently: **0 XFAIL tests** (all defenses are active and passing).
