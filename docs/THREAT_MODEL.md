# AXOLENT AI Threat Model

As of: 2026-05-09
Scope: Bridge (Telegram bot) in Mode B (local CLI wrapper)

## 1. Threat Model

### Trust Boundary

The bot runs on user hardware (laptop/desktop). The user is simultaneously
operator, configurator, and primary stakeholder. There is no central
server, no hosting, no shared infrastructure.

### Adversary Model

| Adversary | Assessment |
|-----------|------------|
| External Telegram users (not on whitelist) | Actively addressed: whitelist blocks |
| Whitelisted user with malicious intent (insider) | Low probability, partially addressed |
| Local malware / other software on the laptop | Out of scope (OS level) |
| Anthropic as LLM provider | Trusted (trust anchor, Mode B compliance) |
| Telegram as transport layer | Trusted (E2E encryption in private chat) |
| Supply chain attacks (dependencies) | Partially addressed (pip-audit, bandit, semgrep) |

## 2. Implemented Mitigations

### Access Control

| Mitigation | Layer | Description |
|------------|-------|-------------|
| Whitelist (WHITELIST_USER_IDS) | Presentation | Only explicitly authorized Telegram user IDs |
| ALLOW_ALL_USERS safeguard (C-1) | Main | Blocks bot start when ALLOW_ALL_USERS=true without AXOLENT_DEV_MODE=true |
| require_private_chat | Presentation | Sensitive commands only in 1:1 chat, not in groups |

### Rate Limiting and Abuse Protection

| Mitigation | Layer | Description |
|------------|-------|-------------|
| Per-user rate limiting (C-2) | Application | 4-profile fixed-window counter: Light (17/min, 100/h, 400/day), Normal (25/min, 350/h, 1500/day), Power (60/min, 900/h, 10000/day), Unlimited |
| Per-user lock | Presentation | Max 1 concurrent LLM call per user |
| Global semaphore | Presentation | Max 4 concurrent Claude processes total |

### Data Protection and Leakage Prevention

| Mitigation | Layer | Description |
|------------|-------|-------------|
| System prompt leakage guard (C-3) | Application | Instruction in prompt + substring-based output filter |
| Error redaction | Presentation/Application | User sees only generic error IDs, details only in log |
| Audit log | Infrastructure | Every LLM call, every command, every rate-limit event is logged |

### LLM Routing and Isolation

| Mitigation | Layer | Description |
|------------|-------|-------------|
| Tuple routing (user_id, chat_id) | Application/Infrastructure | Each user has their own Claude subprocess |
| Provider error isolation | Application | ProviderError hierarchy, no stack traces to user |
| Mode B compliance | Architecture | No API token in code, CLI subprocess uses user subscription |

### Code Quality and Supply Chain

| Mitigation | Layer | Description |
|------------|-------|-------------|
| Pre-commit hooks | CI | ruff, bandit, semgrep, pip-audit on every commit |
| Import linter | CI | Hexagonal layer contracts are enforced |
| Typing | Codebase | Type hints throughout, mypy-compatible |

## 3. Consciously Accepted Risks

### R-1: Storage Without Encryption-at-Rest

**Status:** Partially mitigated (Phase 1)
**Risk:** Bookmarks and memory are stored as plaintext in SQLite on disk, audit logs as JSONL.
**Mitigation Phase 1:** SQLite storage with WAL mode and FTS5 is implemented (C-4). Bookmarks and memory are migrated.
**Rationale:** Local laptop, single user. Full-disk encryption (BitLocker/FileVault) is the OS's responsibility.
**Mitigation Phase 1+:** Fernet encryption on SQLite DB.

### R-2: Conversation History In-Memory Only

**Status:** Accepted (Phase 1)
**Risk:** History is lost on bot restart.
**Rationale:** Privacy by default: no unintended storage of conversation contents.
**Planned mitigation:** Phase 1+: Opt-in SQLite persistence (R07).

### R-3: No Audit Log Tamper Protection

**Status:** Accepted (Phase 1)
**Risk:** JSONL file can be manually edited.
**Rationale:** Local single-user context, tamper protection only becomes relevant with multiple users.
**Planned mitigation:** Phase 1+: Hash chain or HMAC signing.

### R-4: _reset_all_for_tests Reachable in Production

**Status:** Accepted (Low Risk)
**Risk:** Module function that clears internal state.
**Rationale:** Requires direct Python import, not reachable via Telegram or CLI.
**Planned mitigation:** Can be secured via `if TYPE_CHECKING` or conditional import.

### R-5: System Prompt Leakage Guard Is Heuristic

**Status:** Accepted (Phase 1)
**Risk:** Substring-based filter has ~60-70% coverage. Creative reformulations can get through.
**Rationale:** Deterministic filter as first line of defense. Prompt instruction as primary protection.
**Planned mitigation:** Phase 2: Embedding-based similarity check, classifier-based detection.

### R-6: Rate Limiting Partially Persistent

**Status:** Partially mitigated (Phase 1)
**Risk:** Bot restart resets rate-limit counters (counters are in-memory). Profiles are persistent (JSONL/SQLite).
**Rationale:** Whitelist restricts to trusted users. No abuse scenario where counter reset is a problem. Profiles (light/normal/power/unlimited) survive restarts.
**Planned mitigation:** Phase 1+: SQLite-based rate limiting with persistent counters.

### R-7: Memory Translation Sends Content to LLM Provider

**Status:** Documented and accepted (Phase 2)
**Risk:** When a user retrieves memory entries via /memory in a language different from the
stored entry, the memory content is sent to the LLM provider (Claude Haiku) for translation.
This means user memory content leaves the local process and is seen by the LLM provider.
**Mitigation:** The call runs in Mode B (user's own Anthropic subscription via CLI subprocess).
No separate API key is used. The content is processed transiently by the provider and subject
to Anthropic's data retention policy (which does not train on API inputs).
**User control:** Set `AXOLENT_MEMORY_TRANSLATION=false` in environment to disable translation
entirely. Memory entries will then always be shown in their original stored language.
**Privacy note:** Translation requests do not include the user's Telegram ID or username in
the prompt. Only the memory text content and target language are sent.
**Multi-user consideration:** Cache keys include (entry_id, target_lang, user_id) to prevent
cross-user cache hits in future multi-user scenarios.

## 4. Out of Scope

| Area | Rationale |
|------|-----------|
| Server hardening | No server, purely local |
| Multi-tenancy | Single-user model (Phase 1+: multi-user with own subprocess pool) |
| GDPR/SOC2/ISO27001 compliance | Personal tool, no third-party user data (Phase 2+) |
| DDoS protection | Telegram as transport layer absorbs DDoS |
| Backup and recovery | Local laptop, user is responsible for backups |
| Browser/extension security | Not part of the bridge architecture |

## 5. Next Steps (Top 5 from Phase 1+ Roadmap)

1. **Fernet encryption on SQLite DB** (next session)
   Bookmarks and memory are already in SQLite (C-4 done). Next step: Fernet encryption-at-rest.

2. **Audit log hash chain**
   Each entry contains the hash of the previous entry, making tampering detectable.

3. **Input sanitization layer**
   Prompt injection detection before the LLM call (classifier or rule-based).

4. **Persistent rate limiting**
   SQLite-backed rate limits that survive bot restarts.

5. **Multi-user subprocess isolation**
   Dedicated Claude subprocess pool per user with resource limits.
