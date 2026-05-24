# AXOLENT Threat Model

**Version:** 2.0
**Date:** 2026-05-24
**Methodology:** STRIDE per component
**Scope:** AXOLENT Mode-B Telegram bot pre-public-switch
**Authors:** Cosmo (Deep Researcher), reviewed by Atlas

## 1. Architecture Overview

```
                        TRUST BOUNDARY A
                        (Internet / Telegram Cloud)
                        ========================
                              |
                    [Telegram API Servers]
                              |
                        TRUST BOUNDARY B
                        (Telegram <-> Local Process)
                        ========================
                              |
                    [python-telegram-bot lib]
                              |
               +-----------------------------+
               |   PRESENTATION LAYER        |
               |   handlers.py, callbacks.py |
               |   decorators.py, render.py  |
               +-----------------------------+
                              |
               +-----------------------------+
               |   APPLICATION LAYER         |
               |   ChatService, MemoryService|
               |   RateLimiter, LCP,         |
               |   SkillCompression,         |
               |   ExecutionKernel           |
               +-----------------------------+
                    |                   |
        +-----------+         +--------+---------+
        |                     |                  |
+-------v-------+    +-------v-------+  +-------v-------+
| INFRASTRUCTURE|    | INFRASTRUCTURE|  | INFRASTRUCTURE|
| SQLite/       |    | Claude CLI    |  | Sentry SDK    |
| SQLCipher     |    | Subprocess    |  | (error report)|
| (local disk)  |    | (stdin pipe)  |  | (cloud)       |
+---------------+    +---------------+  +---------------+
                              |
                        TRUST BOUNDARY C
                        (Local Process <-> Claude CLI)
                        ========================
                              |
                    [Claude Code CLI]
                              |
                        TRUST BOUNDARY D
                        (Local <-> Anthropic Cloud)
                        ========================
                              |
                    [Anthropic API / Model]
```

### Data Flow (Happy Path)

1. User sends Telegram message
2. Telegram API delivers to python-telegram-bot webhook/polling
3. Presentation layer: whitelist check, rate limit, command routing
4. Application layer: context resolution, memory injection, prompt composition
5. Infrastructure: message sent via stdin pipe to Claude CLI subprocess
6. Claude CLI communicates with Anthropic API (user's own subscription)
7. Response streamed back via stdout pipe
8. Application layer: language enforcement, leakage filter, text guard
9. Presentation layer: chunking, HTML conversion, Telegram send
10. Audit log written (JSONL, local disk)

## 2. Trust Boundaries

| ID | Boundary | Description | Risk Level |
|----|----------|-------------|------------|
| TB-A | Internet to Telegram | Messages traverse Telegram infrastructure | Medium |
| TB-B | Telegram to Local Bot | python-telegram-bot library processes incoming updates | High |
| TB-C | Bot to Claude CLI | Subprocess stdin/stdout pipe, local IPC | Medium |
| TB-D | Local to Anthropic Cloud | Claude CLI connects to Anthropic API | Low (trusted) |
| TB-E | Bot to SQLite | File I/O to local database | Medium |
| TB-F | Bot to Sentry | Error telemetry sent to cloud service | Medium |
| TB-G | Bot to OS Keyring | Encryption key retrieval via DPAPI/Keychain | Low |
| TB-H | Bot to Audit Log | JSONL file writes to local filesystem | Low |

## 3. STRIDE Analysis per Component

### 3.1 Telegram Bot Handler (Presentation Layer)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.1.1 | Unauthorized user access | Spoofing | Attacker uses a Telegram account not on whitelist | **Active**: WHITELIST_USER_IDS fail-closed + ALLOW_ALL_USERS safeguard (C-1) |
| 3.1.2 | Bot command injection via slash | Tampering | Malicious `/command` embedded in bot response triggers Telegram auto-link | **Active**: sanitize_telegram_slashes() replaces / before letters with fraction slash U+2044 |
| 3.1.3 | Missing audit for rejected requests | Repudiation | Unauthorized access attempts not logged for forensics | **Gap**: Only log.warning(), no structured audit entry for rejected whitelist checks |
| 3.1.4 | User message content in Telegram logs | Information Disclosure | python-telegram-bot debug logging may log raw user messages | **Active**: Logging level set to INFO (not DEBUG), but no explicit scrubbing of Update objects |
| 3.1.5 | Callback data injection | Tampering | Attacker crafts arbitrary callback_data in inline keyboard responses | **Active**: Pattern-based routing (regex match), but no HMAC signature on callback payloads |
| 3.1.6 | Concurrent request flooding from whitelisted user | Denial of Service | Single whitelisted user sends burst exceeding semaphore | **Active**: Per-user lock + global semaphore (4) + rate limiter. Adequate. |
| 3.1.7 | Group chat information exposure | Information Disclosure | Bot responds in group chats where other members can see conversation | **Active**: require_private_chat decorator on sensitive commands |
| 3.1.8 | Typing indicator resource exhaustion | Denial of Service | Typing keepalive loop runs indefinitely if response never completes | **Active**: Timeout on Claude subprocess (120s default), cancel propagation via /stop |

### 3.2 Chat Service (Application Layer)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.2.1 | System prompt extraction via prompt injection | Spoofing | User crafts input that makes LLM output its system prompt | **Active**: LeakageFilter (substring-based, ~60-70% coverage) + instruction in system prompt |
| 3.2.2 | Conversation history poisoning | Tampering | Malicious user injects instructions via earlier turns that persist in context | **Active**: Max 20 turns in context window, /reset clears. No turn-level validation. |
| 3.2.3 | Cross-user context leakage via process pool | Information Disclosure | Subprocess shares state between users | **Active**: Per-(user_id, chat_id, model) routing key. Processes never shared. |
| 3.2.4 | Memory injection amplification | Elevation of Privilege | User stores malicious memory entry via /remember that is injected into every future prompt | **Gap**: Memory content injected into system prompt without sanitization beyond truncation (MAX_MEMORY_CHARS_PER_ENTRY=400) |
| 3.2.5 | Unbounded token generation | Denial of Service | LLM generates extremely long response consuming resources | **Active**: Timeout (120s), TELEGRAM_CHUNK_SIZE limits output per message, but no explicit max-tokens parameter sent to CLI |
| 3.2.6 | Provider fallback information leakage | Information Disclosure | Error messages from failed providers may contain internal details | **Active**: ProviderError hierarchy, generic error IDs shown to user |
| 3.2.7 | Language enforcement bypass | Tampering | User forces LLM to respond in unintended language by manipulating context | **Active**: LanguageEnforcement with Verifier + RepairService + StreamGuard |

### 3.3 SQLite Storage (Infrastructure Layer)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.3.1 | Database file theft | Information Disclosure | Attacker copies axolent.db from disk (malware, backup leak, shared drive) | **Partial**: Optional SQLCipher via CryptoConnection + keyring. Default is plaintext SQLite. |
| 3.3.2 | WAL file contains uncommitted data | Information Disclosure | SQLite WAL (-wal) and SHM (-shm) files contain recent writes in plaintext even if main DB is encrypted | **Gap**: No explicit WAL file cleanup on shutdown. WAL may persist with user data. |
| 3.3.3 | SQL injection via memory content | Tampering | User stores content with SQL metacharacters in /remember | **Active**: Parameterized queries throughout (? placeholders). No f-string SQL with user input. |
| 3.3.4 | Schema migration race condition | Denial of Service | Two bot instances starting simultaneously corrupt the schema | **Active**: threading.Lock in SqliteConnection, single-process architecture. But no file-level locking for multi-instance scenario. |
| 3.3.5 | Backup file (.bak) exposure | Information Disclosure | Migration creates .jsonl.bak and .plaintext.bak files containing user data in plaintext | **Gap**: Backup files are never auto-deleted. Public boundary scanner does not check for .bak files at runtime. |
| 3.3.6 | _reset_all_for_tests reachable in production | Tampering | Test-only methods exist in production code, callable via import | **Active**: Documented as R-4 (accepted risk). Not reachable via Telegram commands. |
| 3.3.7 | FTS5 content tokenization reveals search patterns | Information Disclosure | FTS5 index structure could reveal what terms appear in memory without decrypting main content | **Gap**: FTS5 index is stored alongside main DB. If main DB is encrypted via SQLCipher, FTS5 content is also encrypted. But in plaintext mode, FTS5 leaks token metadata. |

### 3.4 Sentry Integration (Infrastructure Layer)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.4.1 | PII leakage via exception messages | Information Disclosure | Exception raised with user text (e.g. ValueError("invalid: <user message>")) | **Active**: CFV-02 redacts ALL exception values. before_send strips all. |
| 3.4.2 | Telegram bot token in breadcrumb URLs | Information Disclosure | HTTP requests to api.telegram.org include bot token in URL path | **Active**: _redact_sensitive_url() with regex replacement in before_send |
| 3.4.3 | Environment variables in crash context | Information Disclosure | Sentry SDK might capture env vars containing TELEGRAM_BOT_TOKEN | **Active**: include_local_variables=False + frame locals stripped + send_default_pii=False |
| 3.4.4 | Sentry DSN exposure | Spoofing | If Sentry DSN is leaked, attacker can send fake error events polluting dashboard | **Gap**: DSN is in .env (not in code). Public boundary scanner blocks .env files. But DSN itself is not rotatable without Sentry project recreation. |
| 3.4.5 | Side-channel timing via Sentry events | Information Disclosure | Error frequency patterns reveal usage patterns (when user is active, how often errors occur) | **Active**: traces_sample_rate=0.0, profiles_sample_rate=0.0. Only errors are sent. Accepted risk for error monitoring value. |
| 3.4.6 | before_send bypass via new SDK features | Information Disclosure | Future Sentry SDK updates may add new data fields not covered by the allowlist | **Active**: Allowlist approach (primary) + blocklist (secondary). New fields are stripped by default. |

### 3.5 Skill Compression / Memory (Application Layer)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.5.1 | Healthcare data stored as skill hypothesis | Information Disclosure | User discusses medical condition; system learns it as a "preference" | **Active**: HealthcareFilter (HC-SC-14) blocks health-related hypotheses |
| 3.5.2 | Secret/credential stored as pattern | Information Disclosure | User pastes API key in chat; system learns it as a "code pattern" | **Active**: SecretScanner (HC-SC-13) detects token/key patterns |
| 3.5.3 | Privacy audit log overflow | Denial of Service | Attacker triggers thousands of privacy rejections filling in-memory log | **Active**: PrivacyAuditLog has max_entries=1000 with rotation (keeps newest 500) |
| 3.5.4 | Nudge filter bypass via Unicode homoglyphs | Tampering | User uses lookalike characters to bypass NudgeFilter regex patterns | **Gap**: NudgeFilter likely uses simple regex. No Unicode normalization before pattern matching documented. |
| 3.5.5 | Hypothesis collision enables data overwrite | Tampering | Two different user statements produce same hypothesis_id, causing data overwrite | **Active**: CollisionDetector exists. But collision is based on content hash, not cryptographic guarantee. |
| 3.5.6 | Conversation import from untrusted source | Tampering | User imports a ChatGPT/Claude export that contains injected system prompts | **Gap**: ImportOrchestrator parses external JSON/Markdown. No validation that imported content does not contain prompt injection payloads. |
| 3.5.7 | ELO rating manipulation via crafted feedback | Elevation of Privilege | User deliberately provides inconsistent feedback to skew skill rankings | **Active**: BKT + FSRS decay + evidence ledger provide multi-signal validation. Manipulation requires sustained effort. |

### 3.6 Claude CLI Subprocess (Infrastructure Layer)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.6.1 | Prompt visible in process listing | Information Disclosure | System prompt passed via stdin, but process argv shows `claude -p --model X` | **Active**: Prompt sent via stdin PIPE, not as command-line argument. Privacy note in claude_cli.py. |
| 3.6.2 | Subprocess escape via LLM output | Elevation of Privilege | LLM output contains shell metacharacters that could be interpreted | **Active**: stdout is read as bytes and decoded. No shell interpretation of output. |
| 3.6.3 | Process pool memory accumulation | Denial of Service | Warm subprocesses accumulate memory over time (context window growth) | **Active**: 60-minute inactivity timeout + LRU eviction at POOL_MAX_SIZE=20. |
| 3.6.4 | Dirty process state after cancellation | Tampering | Cancelled stream leaves stale data in stdout pipe; next request reads old response | **Active**: is_dirty flag on ManagedProcess. Dirty processes are killed and restarted on next use. |
| 3.6.5 | Claude CLI update changes behavior silently | Tampering | User updates Claude CLI and new version has different output format or new flags | **Gap**: No version pinning or version check for claude CLI binary. Stream JSON parsing could break silently. |
| 3.6.6 | Environment variable exfiltration via prompt | Information Disclosure | Crafted prompt asks LLM to output env vars (TELEGRAM_BOT_TOKEN, SENTRY_DSN) | **Active**: Claude CLI subprocess inherits env vars. However, Claude Code does not have shell access in -p mode. LLM cannot execute commands to read env. But in --dangerously-skip-permissions mode, this would be exploitable. |
| 3.6.7 | Zombie process accumulation | Denial of Service | Subprocesses that fail to terminate properly accumulate as zombies | **Active**: asyncio.wait_for with timeout + proc.kill() on timeout. _cleanup_loop checks health every 60s. |

### 3.7 Language Control Plane (Application Layer)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.7.1 | Language detection poisoning | Spoofing | Attacker sends text in mixed languages to confuse the detector, causing wrong language to stick | **Active**: Confidence threshold >= 0.7 for language switch. Min 15 chars for detection. |
| 3.7.2 | Immutable context bypass | Tampering | If LanguageContext mutability is broken, attacker could inject language override | **Active**: MappingProxyType makes LanguageContext immutable. typeguard on constructors. |
| 3.7.3 | Repair service amplification | Denial of Service | RepairService calls the LLM provider again (re-translation), doubling resource usage | **Active**: Repair is bounded (single retry). FP-Detection prevents unnecessary repairs. |
| 3.7.4 | StreamGuard false positive causes response truncation | Denial of Service | StreamGuard incorrectly identifies response as wrong language and truncates | **Active**: FP-Detection layer reduces false positives. Known limitation documented. |
| 3.7.5 | Language audit log reveals user language preferences | Information Disclosure | DetectionAuditLogger stores language detections which could profile user nationality | **Active**: In-memory only, not persisted to disk. Rotated per session. |
| 3.7.6 | @lcp_aware decorator creates new resolver per call | Denial of Service | Each command with @lcp_aware creates a new LanguageResolver() instance | **Gap**: Resolver instantiated per invocation in decorator. Low severity (lightweight object) but wasteful pattern. |

### 3.8 Pre-Commit Hooks (Development Security)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.8.1 | Hook bypass via --no-verify | Tampering | Developer commits with `git commit --no-verify`, skipping all security hooks | **Gap**: No server-side enforcement. CI catches it post-push, but secrets could be in git history. |
| 3.8.2 | Gitleaks disabled on Windows | Information Disclosure | Gitleaks pre-commit hook is disabled due to upstream wasm bug | **Active**: TruffleHog v3 hook is active as replacement. Gitleaks runs in CI. Manual scanning recommended pre-release. |
| 3.8.3 | Semgrep rule coverage gaps | Tampering | Custom semgrep rules (3) cover specific patterns but cannot catch all anti-patterns | **Active**: 3 custom rules + semgrep auto config (2000+ community rules). |
| 3.8.4 | pip-audit window between scan and install | Tampering | Vulnerability disclosed after commit but before deployment | **Active**: Daily smoke test re-runs dependency check. |
| 3.8.5 | Bandit false positive suppression hiding real issues | Tampering | nosec/nosemgrep comments could accidentally suppress real findings | **Gap**: No automated audit of suppression comments. Each nosec should have a justification comment (most do, but no enforcement). |
| 3.8.6 | Pre-commit hook dependencies not pinned | Tampering | Upstream pre-commit repos could be compromised (supply chain attack on hooks themselves) | **Active**: rev pinned for all repos. But no hash verification of hook source. |

### 3.9 GitHub Actions (CI/CD Security)

| # | Threat | Category | Description | Mitigation Status |
|---|--------|----------|-------------|-------------------|
| 3.9.1 | Workflow injection via PR title/body | Tampering | Attacker opens PR with malicious title that gets interpolated into shell commands | **Active**: No ${{ github.event.pull_request.title }} interpolation in run steps. Uses actions/checkout@v4. |
| 3.9.2 | Secret exfiltration in CI logs | Information Disclosure | GITHUB_TOKEN or other secrets printed to logs | **Active**: Only GITHUB_TOKEN used (in gitleaks action). TELEGRAM_BOT_TOKEN not in CI. |
| 3.9.3 | Dependency confusion attack | Tampering | Malicious package with same name published to PyPI, pulled during pip install | **Gap**: No hash pinning in requirements. Uses pip install -e ".[dev,test]" from pyproject.toml. |
| 3.9.4 | Actions version pinning to tag (not SHA) | Tampering | actions/checkout@v4 resolves to a tag that could be force-pushed | **Gap**: All actions pinned to version tags (v4, v5), not commit SHAs. Standard practice but less secure than SHA pinning. |
| 3.9.5 | Public boundary scanner not enforced on direct pushes | Repudiation | Main branch allows direct push without PR (if configured) | **Active**: public-boundary.yml runs on push to main AND pull_request. |
| 3.9.6 | CI timeout allows long-running resource abuse | Denial of Service | Attacker opens PR that triggers 15-minute test suite repeatedly | **Active**: timeout-minutes: 15 on quality job. PR from forks require approval. |

## 4. Identified Gaps (Sigma-Test-Backlog)

### STRIDE-GAP-01: WAL File Plaintext Leak

| Field | Value |
|-------|-------|
| Component | SQLite Storage |
| Category | Information Disclosure |
| Description | SQLite WAL (-wal) and SHM (-shm) journal files may contain recent writes in plaintext. Even when SQLCipher is enabled for the main DB file, the WAL is part of the encrypted DB (SQLCipher handles this). However, in plaintext mode (default for development), these files persist after unclean shutdown containing user data. Additionally, if the bot crashes, WAL checkpoint may not complete, leaving data in the WAL file accessible separately. |
| Current Status | No explicit WAL file management on shutdown. SqliteConnection.close() does not force a WAL checkpoint. |
| Proposed Test | Test that after SqliteConnection.close(), a PRAGMA wal_checkpoint(TRUNCATE) is issued. Verify no -wal or -shm files remain after clean shutdown. Test that CryptoConnection WAL is also encrypted by SQLCipher (verify by trying to read .db-wal with plain sqlite3). |
| Severity | **MEDIUM** |

### STRIDE-GAP-02: Backup File (.bak) Exposure

| Field | Value |
|-------|-------|
| Component | SQLite Storage |
| Category | Information Disclosure |
| Description | Migration process creates .jsonl.bak and .plaintext.bak files that contain full user data (bookmarks, memory entries, profiles) in plaintext. These files are never automatically deleted and are not covered by the public boundary scanner's runtime checks. If the data directory is accidentally synced, backed up to cloud, or included in a deployment artifact, user data is exposed. |
| Current Status | .bak files created during migration, never cleaned up. check_public_boundary.py blocks *.db but not *.bak or *.jsonl.bak from git. |
| Proposed Test | (1) Add *.bak and *.jsonl.bak to public_boundary.yaml forbidden patterns. (2) Test that migration log includes a warning about plaintext backup persistence. (3) Add a startup check that warns if .bak files older than 7 days exist in data/. |
| Severity | **MEDIUM** |

### STRIDE-GAP-03: Conversation Import Injection

| Field | Value |
|-------|-------|
| Component | Skill Compression |
| Category | Tampering / Elevation of Privilege |
| Description | ImportOrchestrator accepts ChatGPT exports, Claude exports, and Markdown files for skill learning. These external files could contain crafted content designed to inject prompt instructions that get stored as "skills" and later injected into future system prompts. Unlike regular user messages (which are transient), imported content becomes persistent knowledge. |
| Current Status | ImportOrchestrator parses file content. PrivacyPipeline runs on hypothesis promotion (catches secrets/health data) but does NOT check for prompt injection patterns in imported content. |
| Proposed Test | (1) Create test fixtures with ChatGPT export JSON containing known prompt injection payloads (e.g., "Ignore all previous instructions and..."). (2) Verify that ImportOrchestrator either rejects or sanitizes these entries. (3) Add a PromptInjectionFilter to the PrivacyPipeline or as a pre-filter in ImportOrchestrator. |
| Severity | **HIGH** |

### STRIDE-GAP-04: Claude CLI Version Drift

| Field | Value |
|-------|-------|
| Component | Claude CLI Subprocess |
| Category | Tampering |
| Description | The process pool expects Claude CLI to output stream-json format events. No version check is performed at startup. If the user updates their Claude CLI and the output format changes (new event types, changed field names, removed fields), the bot may silently produce wrong results, crash mid-stream, or leak internal state through error messages. |
| Current Status | No version detection. No format validation beyond basic JSON parsing. Stream events are parsed optimistically. |
| Proposed Test | (1) Add a startup version check: run `claude --version` and log/assert minimum version. (2) Add a schema validator for StreamEvent JSON (reject unknown event types gracefully). (3) Test that malformed stream JSON produces a clean error message (not a stack trace or partial response). |
| Severity | **MEDIUM** |

### STRIDE-GAP-05: Memory Injection Amplification

| Field | Value |
|-------|-------|
| Component | Chat Service / Memory Service |
| Category | Elevation of Privilege |
| Description | User can store arbitrary text via /remember (up to 400 chars per entry, 4000 chars total). This text is injected verbatim into the system prompt on every subsequent LLM call. A malicious or compromised user could store prompt injection payloads as "memories" that persist across conversations and are automatically loaded, effectively gaining persistent prompt injection. |
| Current Status | Only truncation applied (MAX_MEMORY_CHARS_PER_ENTRY=400). No content validation, no injection detection on memory entries. PrivacyPipeline only applies to skill compression hypotheses, not to /remember entries. |
| Proposed Test | (1) Test that known prompt injection payloads stored via /remember are either rejected at storage time or sanitized/escaped at injection time. (2) Add a lightweight injection detector to MemoryService.store() that flags suspicious patterns (e.g., "ignore previous", "system:", "you are now"). (3) Verify that memory entries are clearly delimited in the system prompt (e.g., wrapped in XML tags or quotes) so the LLM can distinguish memory from instructions. |
| Severity | **HIGH** |

### STRIDE-GAP-06: Callback Data Forgery

| Field | Value |
|-------|-------|
| Component | Telegram Bot Handler |
| Category | Tampering |
| Description | Inline keyboard callbacks (bm_show:X, bm_del:X, settings_Y, skill_Z) use plain-text identifiers that a technically skilled user could forge by crafting raw Telegram API requests with arbitrary callback_data. While Telegram only delivers callbacks to users who saw the keyboard, a modified Telegram client could send arbitrary callback_data strings. This could allow accessing/deleting other users' bookmarks if the handler does not verify ownership. |
| Current Status | Pattern-based routing via regex. Bookmark handlers check user_id ownership on read. But ownership check should be verified for ALL callback handlers (settings, skills, import). No HMAC or nonce on callback payloads. |
| Proposed Test | (1) Unit test that sends forged callback_data with a different user's bookmark ID and verify access is denied. (2) Audit all CallbackQueryHandlers for ownership verification. (3) Consider adding user_id to callback_data pattern (e.g., bm_del:{user_id}:{msg_id}) as defense-in-depth. |
| Severity | **MEDIUM** |

### STRIDE-GAP-07: Audit Log Missing Tamper Detection

| Field | Value |
|-------|-------|
| Component | Audit Log (Infrastructure) |
| Category | Repudiation |
| Description | Audit log is written as plain JSONL with RotatingFileHandler. An attacker with local filesystem access (malware, physical access) can edit or delete audit entries without detection. This means malicious actions cannot be reliably attributed. For multi-user scenarios or compliance requirements, this is insufficient. |
| Current Status | Documented as R-3 (accepted risk). No hash chain, no HMAC signing, no append-only filesystem flag. |
| Proposed Test | (1) Implement and test a hash-chain mechanism: each entry includes SHA-256 of previous entry. (2) Test that tampering with any entry breaks the chain validation. (3) Add a /audit-verify command (admin-only) that validates chain integrity. |
| Severity | **LOW** (single-user context reduces impact; escalates to MEDIUM in multi-user) |

### STRIDE-GAP-08: NudgeFilter Unicode Bypass

| Field | Value |
|-------|-------|
| Component | Skill Compression Privacy |
| Category | Tampering |
| Description | The NudgeFilter uses pattern matching to detect policy-violating content (e.g., manipulation, dark patterns). Attackers can use Unicode homoglyphs (Cyrillic "a" instead of Latin "a"), zero-width characters, or Unicode normalization tricks to bypass regex-based filters while the text remains visually identical to a human reader. |
| Current Status | No explicit Unicode normalization (NFKC/NFKD) before pattern matching in privacy filters. SecretScanner and HealthcareFilter may have the same vulnerability. |
| Proposed Test | (1) Create test cases with homoglyph-obfuscated strings (e.g., using Cyrillic, fullwidth, mathematical symbols). (2) Verify filters detect them after normalization. (3) Add unicodedata.normalize('NFKC', text) as a preprocessing step in all three privacy filters. |
| Severity | **MEDIUM** |

### STRIDE-GAP-09: Pre-Commit Bypass via --no-verify

| Field | Value |
|-------|-------|
| Component | Pre-Commit Hooks |
| Category | Tampering / Repudiation |
| Description | All 17 pre-commit hooks (including bandit, semgrep, trufflehog, pip-audit) can be skipped with `git commit --no-verify`. A developer in a rush could commit secrets, vulnerable code, or non-English strings that bypass all local security gates. While CI catches this post-push, secrets in git history are permanent (require history rewrite). |
| Current Status | CI runs all checks on PR. But between commit and PR creation, secrets could be pushed to remote. TruffleHog CI scans full history (mitigates partially). |
| Proposed Test | (1) Document in CONTRIBUTING.md that --no-verify is prohibited. (2) Add a CI check that verifies the most recent commit passes all pre-commit hooks (retroactive enforcement). (3) Consider a server-side pre-receive hook if GitHub Enterprise is used (not applicable for GitHub.com free/pro). |
| Severity | **LOW** (solo developer context; escalates to MEDIUM with team) |

### STRIDE-GAP-10: GitHub Actions SHA Pinning

| Field | Value |
|-------|-------|
| Component | GitHub Actions |
| Category | Tampering |
| Description | All GitHub Actions are pinned to version tags (e.g., actions/checkout@v4) instead of commit SHAs. Tags are mutable: a compromised action maintainer could force-push a malicious version under an existing tag. This is a supply chain risk for CI. |
| Current Status | Standard practice. No SHA pinning. 4 workflows with 6 action references total. |
| Proposed Test | (1) Replace all action version tags with full commit SHAs (e.g., actions/checkout@<sha>). (2) Add Dependabot configuration for GitHub Actions to get automatic SHA update PRs. (3) Document the pinning policy in CONTRIBUTING.md. |
| Severity | **LOW** |

### STRIDE-GAP-11: Environment Variable Leakage via Subprocess

| Field | Value |
|-------|-------|
| Component | Claude CLI Subprocess |
| Category | Information Disclosure |
| Description | Claude CLI subprocess inherits the full process environment (including TELEGRAM_BOT_TOKEN, SENTRY_DSN, and any other secrets in .env). While Claude in -p mode cannot execute shell commands, the Claude Code CLI in other modes (interactive, with --dangerously-skip-permissions) could potentially be instructed to read environment variables. If the subprocess invocation is ever changed or if a bug in Claude CLI allows command execution, env vars are at risk. |
| Current Status | No environment scrubbing for subprocess. Full env inherited. |
| Proposed Test | (1) Modify subprocess creation to pass explicit env dict with only required variables (PATH, HOME/USERPROFILE, ANTHROPIC variables). (2) Test that TELEGRAM_BOT_TOKEN is NOT in subprocess environment. (3) Verify claude CLI still functions with the restricted env. |
| Severity | **HIGH** |

### STRIDE-GAP-12: Whitelist Rejection Not Audited

| Field | Value |
|-------|-------|
| Component | Telegram Bot Handler |
| Category | Repudiation |
| Description | When an unauthorized user attempts to access the bot, only a log.warning() is emitted. No structured audit log entry is created. In a scenario where someone probes the bot repeatedly, there is no persistent, queryable record of these attempts. The audit log only captures successful interactions. |
| Current Status | require_whitelist decorator logs to Python logger but does not call write_audit_log(). |
| Proposed Test | (1) Add write_audit_log() call in require_whitelist for rejected access attempts (include timestamp, user_id, username if available, attempted command). (2) Test that unauthorized access attempts appear in audit.jsonl. (3) Consider rate-limiting the warning log to prevent log flooding from persistent probing. |
| Severity | **LOW** |

### STRIDE-GAP-13: Dependency Confusion in CI

| Field | Value |
|-------|-------|
| Component | GitHub Actions |
| Category | Tampering |
| Description | CI installs dependencies via `pip install -e ".[dev,test]"` from pyproject.toml. If any dependency name has a private/internal variant that could be registered on PyPI by an attacker, a dependency confusion attack is possible. Additionally, no hash verification (--require-hashes) is used, so a compromised PyPI mirror could serve malicious packages. |
| Current Status | Standard pip install without hash pinning. All dependencies are public PyPI packages. No private registry used (reduces confusion risk). |
| Proposed Test | (1) Verify all package names in pyproject.toml are registered on PyPI by the expected maintainers. (2) Consider generating a requirements.txt with hashes for CI (pip-compile --generate-hashes). (3) Add pip install --require-hashes in CI for production dependencies. |
| Severity | **LOW** |

### Summary Table

| Gap ID | Component | Category | Severity | Status |
|--------|-----------|----------|----------|--------|
| STRIDE-GAP-01 | SQLite Storage | Information Disclosure | MEDIUM | TESTED (xfail, Post-Switch) |
| STRIDE-GAP-02 | SQLite Storage | Information Disclosure | MEDIUM | TESTED (xfail, Post-Switch) |
| STRIDE-GAP-03 | Skill Compression | Tampering / EoP | **HIGH** | **FIXED** |
| STRIDE-GAP-04 | Claude CLI Subprocess | Tampering | MEDIUM | TESTED (partial) |
| STRIDE-GAP-05 | Chat Service / Memory | Elevation of Privilege | **HIGH** | **FIXED** |
| STRIDE-GAP-06 | Telegram Bot Handler | Tampering | MEDIUM | TESTED (ownership verified) |
| STRIDE-GAP-07 | Audit Log | Repudiation | LOW | TESTED (xfail, Post-Switch) |
| STRIDE-GAP-08 | Skill Compression Privacy | Tampering | MEDIUM | TESTED (xfail, Post-Switch) |
| STRIDE-GAP-09 | Pre-Commit Hooks | Tampering / Repudiation | LOW | TESTED (xfail, Post-Switch) |
| STRIDE-GAP-10 | GitHub Actions | Tampering | LOW | TESTED (xfail, Post-Switch) |
| STRIDE-GAP-11 | Claude CLI Subprocess | Information Disclosure | **HIGH** | **FIXED** |
| STRIDE-GAP-12 | Telegram Bot Handler | Repudiation | LOW | TESTED (xfail, Post-Switch) |
| STRIDE-GAP-13 | GitHub Actions | Tampering | LOW | TESTED (xfail, Post-Switch) |

## 5. Out of Scope

| Area | Rationale |
|------|-----------|
| Telegram platform compromise | Telegram server-side security is Telegram's responsibility. Mode-B cannot mitigate a compromised Telegram infrastructure. |
| Anthropic API compromise | Claude model behavior is Anthropic's responsibility. Mode-B trusts the model output as-is (post-filtering). |
| OS-level threats (keylogger, rootkit) | Local malware with kernel access can read any process memory. Full-disk encryption and OS hardening are the user's responsibility. |
| Physical access to machine | If an attacker has physical access, all local data is compromised regardless of application-level protections. |
| Network-level MITM | Telegram uses TLS. Claude CLI uses HTTPS. Certificate pinning is the library's responsibility. |
| Social engineering of the user | User voluntarily sharing their bot token or DB file is not an application-level threat. |
| Anthropic rate limits / billing | User's own subscription. Cost management is user's responsibility. |
| Mobile Telegram client vulnerabilities | Client-side vulnerabilities in Telegram apps are out of scope. |

## 6. Risk Acceptance

| Risk ID | Description | Acceptance Rationale |
|---------|-------------|---------------------|
| RA-01 | System prompt leakage guard is heuristic (~60-70%) | Deterministic filter as first line. Prompt instruction is primary defense. Full mitigation requires embedding-based classifier (Phase 2). |
| RA-02 | Audit log without tamper protection | Single-user, local-only deployment. Tamper protection becomes relevant only with multi-user or compliance requirements. |
| RA-03 | Rate limit counters partially in-memory | SQLiteRateLimitStorage persists counters since Phase 1. Remaining gap: very brief window between count increment and DB write. Acceptable for trusted whitelist users. |
| RA-04 | _reset_all_for_tests in production code | Not callable via any Telegram command or external interface. Requires direct Python import. No security boundary crossed. |
| RA-05 | Conversation history in-memory only (lost on restart) | Privacy-by-default design. Prevents unintended long-term storage of sensitive conversations. |
| RA-06 | Memory translation sends content to LLM provider | Mode-B: user's own Anthropic subscription. No third-party key. Disableable via env var. Content is transient per Anthropic policy. |
| RA-07 | Single-user architecture without multi-tenancy isolation | Current deployment is single-user (owner-operated). Multi-user scenarios will require additional isolation (separate DB per user, process-level sandboxing). |
| RA-08 | SQLCipher is optional (plaintext default) | Development/testing convenience. Production mode (AXOLENT_PRODUCTION=true) enforces encryption via AG-SC-7 guard. |

## 7. References

| Resource | URL |
|----------|-----|
| STRIDE Methodology (Microsoft) | https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats |
| OWASP Top 10 for LLM Applications (2025) | https://owasp.org/www-project-top-10-for-large-language-model-applications/ |
| OWASP Threat Modeling | https://owasp.org/www-community/Threat_Modeling |
| SQLCipher Security | https://www.zetetic.net/sqlcipher/design/ |
| SQLite WAL Mode | https://www.sqlite.org/wal.html |
| Telegram Bot API Security | https://core.telegram.org/bots#security |
| Supply Chain Security (SLSA) | https://slsa.dev/ |
| GitHub Actions Security Hardening | https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions |
| Unicode Security Considerations | https://unicode.org/reports/tr36/ |
| Prompt Injection Taxonomy | https://arxiv.org/abs/2302.12173 |

---

## Appendix A: Existing Mitigations Summary

For reference, these mitigations are already active and tested:

### Access Control
- WHITELIST_USER_IDS (fail-closed)
- ALLOW_ALL_USERS + AXOLENT_DEV_MODE tripwire (C-1)
- require_private_chat decorator
- Per-user asyncio.Lock (max 1 concurrent LLM call)
- Global asyncio.Semaphore(4) (max 4 Claude processes)
- Rate limiter (4 profiles, persistent counters)

### Data Protection
- Sentry before_send with allowlist + blocklist + frame-locals-strip
- Telegram bot token URL redaction
- send_default_pii=False, include_local_variables=False
- LeakageFilter (substring-based system prompt protection)
- sanitize_telegram_slashes (bot command injection prevention)
- ALLOWED_URL_SCHEMES (no javascript: in markdown-to-HTML)
- Error redaction (generic error IDs to user)

### Privacy
- PrivacyPipeline (3 filters: Healthcare, Secret, Nudge)
- Pattern-Judge blocks promotion without privacy pipeline (Semgrep enforced)
- In-memory privacy audit log (rotated per session)
- Memory token budget (MAX_MEMORY_CHARS_PER_ENTRY=400, total 4000)
- stdin pipe for prompts (not argv, prevents ps/top exposure)

### Code Quality / Supply Chain
- 17 pre-commit hooks (ruff, bandit, semgrep, pip-audit, import-linter, trufflehog, etc.)
- Import-linter hexagonal contracts
- 3 custom Semgrep rules
- icontract pre/post-conditions on 4 pipelines
- typeguard on 13 constructors
- TruffleHog v3 full-history scan (CI + pre-commit)
- Public boundary scanner (blocks .env, *.sqlite, *.db, secrets patterns)
- English-only production code hook
- 127 adversarial tests (K1-K10)
- 36 OWASP LLM Top 10 tests
- 11 Hypothesis property-based tests
- Auto smoke test (15 scenarios)

### Architecture
- Hexagonal architecture (domain/application/infrastructure/presentation)
- Per-(user_id, chat_id, model) subprocess isolation
- LanguageContext immutability (MappingProxyType)
- StreamGuard + FP-Detection for language enforcement
- Optional SQLCipher encryption (AG-SC-7 enforces in production)
- Keyring-based key management (OS credential vault)

---

## 8. Resolution Status (Updated 2026-05-24)

| Gap | Severity | Status | Resolution |
|-----|----------|--------|------------|
| GAP-03 | HIGH | FIXED (9f4a24e) | ImportOrchestrator injection detection + InjectionDetector pre-filter on all imported user messages |
| GAP-05 | HIGH | FIXED (9f4a24e) | /remember injection-pattern check (InjectionDetector) + `<user_memory>` delimiter wrap in system prompt |
| GAP-11 | HIGH | FIXED (9f4a24e) | Claude CLI subprocess env-scrubbing via allowlist (`build_scrubbed_env()`) in both ProcessPool and ClaudeProvider |
| GAP-01 | MEDIUM | TESTED (xfail) | WAL checkpoint test added. Standard close() handles it on most platforms; explicit PRAGMA recommended |
| GAP-02 | MEDIUM | TESTED (xfail) | Test verifies public_boundary.yaml coverage. *.bak pattern not yet added to forbidden list |
| GAP-04 | MEDIUM | TESTED (partial) | StreamEvent schema validation passes. Version check not yet implemented |
| GAP-06 | MEDIUM | TESTED | BookmarkService.remove_bookmark() requires user_id (ownership check verified) |
| GAP-07 | LOW | TESTED (xfail) | Hash-chain not implemented. Accepted risk RA-02 for single-user context |
| GAP-08 | MEDIUM | TESTED (xfail) | Homoglyph tests reveal NFKC normalization gap in privacy filters. Post-switch fix |
| GAP-09 | LOW | TESTED (xfail) | No retroactive pre-commit enforcement in CI yet |
| GAP-10 | LOW | TESTED (xfail) | All 10 actions pinned to version tags, not SHAs. Standard practice but less secure |
| GAP-12 | LOW | TESTED (xfail) | require_whitelist only uses log.warning(), no structured audit entry |
| GAP-13 | LOW | TESTED (xfail) | No hash-pinned requirements file. Post-switch task |

### Implementation Details (HIGH Fixes)

**GAP-11 (Env Scrubbing):**
- New module: `bridge/application/security/env_scrubber.py`
- Allowlist: PATH, HOME, USERPROFILE, ANTHROPIC_*, CLAUDE_*, system paths, TLS cert paths
- Applied in: `infrastructure/claude_process_pool.py` (_spawn_process) and `infrastructure/providers/claude_cli.py` (query)
- Verified: TELEGRAM_BOT_TOKEN, SENTRY_DSN, DATABASE_URL never reach subprocess

**GAP-05 (Memory Injection):**
- New module: `bridge/application/security/injection_detector.py`
- 15 regex patterns (EN + DE) with NFKC normalization pre-processing
- Applied in: `presentation/handlers.py` (handle_remember_command) before storage
- Defense-in-depth: `<user_memory>` delimiters in `chat_service.py` memory formatting
- Audit trail: blocked attempts logged via write_raw_audit()

**GAP-03 (Import Injection):**
- Same InjectionDetector used in ImportOrchestrator._extract_and_store_patterns()
- Per-message check on all imported user_messages
- Rejected messages counted in ImportResult.injection_rejections
- Logged with source_path, pattern_name, matched_text

---

## Appendix B: Threat Model Maintenance

This document should be updated:
1. When new components are added (new provider, new storage layer, new feature)
2. When a gap is mitigated (move from Gap to Active mitigation)
3. After any security incident (add lessons learned)
4. Quarterly review (minimum) to catch architectural drift

Next review date: 2026-08-24
