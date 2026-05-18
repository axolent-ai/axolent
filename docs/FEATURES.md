# AXOLENT AI Feature Overview

As of: 2026-05-10

> **Stop sending your private thoughts to someone else's server.**
> Your AI. Your device. Your data. Period.

## Implemented

### Architecture (R00, Foundation)

* Hexagonal Architecture (Variant A): Domain / Application / Infrastructure / Presentation
* import-linter with 3/3 layer contracts
* Pre-commit hooks (13 hooks: ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, check-merge-conflict, mixed-line-ending, bandit, import-linter, pytest, pip-audit, semgrep)
* Mode B: local CLI wrapper for claude, no token hijacking
* UTF-8 enforced across all components (helper in bridge/utils/encoding.py)
* Audit logging with rotation
* Commits: 821aaba (Initial scaffold) .. f613961 (Mode B Bot)

### Memory System

* Trinity-Memory (Episodic, Semantic, Procedural)
* SQLite storage with FTS5 full-text search (since C-4/R03)
* LIKE fallback on FTS5 zero results (substring contract)
* Stop-word filtering + punctuation strip in keyword extraction
* Memory token budget (400 chars/entry, 4000 chars total)
* User isolation per user_id
* Consolidation hook (consolidator.py stub)
* T28 Active Curiosity: memory context instructs the LLM to reference only stored facts, never extrapolate, and ask with genuine interest when gaps are noticed (replaces passive constraint approach)
* Commit: 92be2da, 9954df8

### Persistent Pipe + Streaming (R04)

* ClaudeProcessPool: Persistent stdin pipe per (user_id, chat_id)
* 60-minute inactivity timeout (previously 5 min)
* Crash recovery, graceful shutdown, health check
* NDJSON parser for stream events
* StreamingHandler with adaptive throttle (1.5s default, backoff up to 10s)
* Markdown smart trim (formatted text during streaming)
* Multi-message split (responses >4096 characters)
* Final edit priority (retried on 429)
* 74% faster than cold-start subprocess
* Commit: e07a7ef

### Typing Keepalive (R02-A)

* Background asyncio task triggers TYPING indicator every 4 seconds
* Commit: 051bc96

### Status Manager (R02-B)

* Contextual status indicators: Memory loading, Thinking, Formatting
* Language-aware (DE + EN based on sticky language)
* Rate-limited with phase-change bypass (new phase bypasses throttle)
* StatusSession architecture with callback protocol
* Automatic deactivation on stream start

### Bookmarks

* Reply-based /save
* /bookmarks with inline buttons (full text / remove)
* chat_id scoping
* BookmarkService with constructor injection (SQLite default, JSONL legacy adapter)
* Backward-compat migration for legacy entries (JSONL -> SQLite on first start)
* Commits: 605d8ca .. 029e7eb

### Conversation History

* Sticky language per chat_id
* Max 20 turns
* Static bot responses in history (for /start, /help, /reset, /lang)
* Reply-to-bot-message context
* Currently in-memory only (see R07 for persistence)

### Privacy & Safety

* Whitelist fail-closed with ALLOW_ALL_USERS override
* AXOLENT_DEV_MODE mandatory companion (tripwire guard)
* @require_whitelist + @require_private_chat decorators
* Group chat block (privacy)
* System prompt leakage guard (2 layers: instruction + output filter)
* Generic error messages with error_id (no stack trace leak)
* Commit: 029e7eb (Track-C hardening)

### Rate Limiting

* 4 profiles: Light (100/h, 400/day), Normal (350/h, 1500/day), Power (900/h, 10k/day), Unlimited
* /usage command with progress bars + reset times
* /setlimit command (two-step for Unlimited)
* 70% warning once per window
* Unlimited reminder every 100 requests
* Profiles persistent across bot restart
* Commit: 029e7eb

### Security Tools

* Local tools (pre-commit hooks): Bandit, Semgrep, pip-audit
* gitleaks configured (pre-commit hook commented out due to Windows go-re2 bug, binary not installed locally; config .gitleaks.toml exists for later use)
* SECURITY.md, .gitleaks.toml, .semgrepignore
* pip-audit + semgrep: venv-relative paths, work without activated venv
* Commit: 02b1da0

### Markdown Rendering

* Markdown-to-Telegram-HTML converter
* UUID sentinel for links (double-escape fix)
* URL scheme whitelist (http, https, tg, mailto)
* Commits: eeb89c5 .. bc3a05c

### Multi-AI Debate (R10, MVP)

* /debate command: queries all available providers in parallel
* DebateOrchestrator with asyncio.gather + return_exceptions=True
* Crash resilience: one crashing provider does not stop the others
* Per-provider 60s timeout (configurable)
* Consensus/dissent heuristic (Jaccard word overlap analysis)
* DEBATE_PROVIDERS env var for provider selection
* Audit log with event_type "debate"
* Privacy guard + rate limiting active
* Multi-message split for long outputs

### Provider Failover (FallbackResolver)

* Automatic provider failover for non-streaming operations (/debate, self-awareness)
* Per-slot fallback chains configurable via AXOLENT_FALLBACK_CHAIN_* env vars
* User notice when fallback is used (configurable threshold)
* In-memory metrics (attempts, failures, reasons per provider)
* **Known limitation (v1.0):** Streaming /chat does NOT have automatic fallback. If the primary provider returns 429 during streaming, the user sees an error. Streaming fallback is planned for v1.1.
* Commit: f09a7f7

### Code Quality

* 1900+ tests passing
* 8 code reviews: V1, V2, V3, V4, V5, V6 (Codex + Claude), Feature Review, Language Re-Review
* Threat model documented (docs/THREAT_MODEL.md)
* Commits: 916d234, eeb89c5, bc3a05c, 9954df8

## Killer Features (Roadmap Highlights)

### R10: Multi-AI Debate (Implemented, MVP)

Multiple AI models answer a question in parallel. The user sees responses
side-by-side with consensus/dissent analysis. Crash-resilient: if one provider
fails, the others still respond. Foundation for NLnet application.

### R11: Cross-Provider Memory

Memory entries work across all providers.
Switch from Claude to another model: your context stays.
No vendor lock-in at the knowledge level.

### R12: Smart Privacy Routing

Automatic detection of sensitive content (health, finances, password queries).
Sensitive requests are processed locally, non-critical ones may go to the cloud.
Privacy by design without user overhead.

### R15: Skill Marketplace with Premium + Revenue Sharing

Community-created skills (prompts, workflows, tool chains) as a marketplace.
Creators earn revenue, premium skills for power users.
An ecosystem that grows with the community.

### R18: User Model Override (Phase 1 complete)

Phase 1: `/setmodel`, `/resetmodel`, `/models` commands. User can switch between
Opus, Sonnet and Haiku. SQLite-persisted, alias resolution,
model mismatch detection in the process pool.
Phase 2 (planned): TaskRouter with 6 slots (chat, code, etc.).

## Roadmap (all items R01-R18)

| Item | Name | Description | Status |
|------|------|-------------|--------|
| R01 | Faster Responses | System prompt optimization, multi-provider routing | Planned |
| R02-A | Typing Keepalive | TYPING indicator every 4s during processing | Done |
| R02-B | Status Manager | Contextual status indicators (Memory, Thinking, Formatting) | Done |
| R03 | SQLite Migration | JSONL to SQLite/FTS5 (Memory + Bookmarks) | Done |
| R04 | Persistent Pipe + Streaming | Live token streaming via persistent stdin pipe | Done |
| R05 | Desktop Client | Local desktop client (Electron/Tauri) | Planned |
| R06 | Anthropic API Premium | API-key-based premium provider (pay-per-use) | Planned |
| R07 | Persistent History | Conversation history survives bot restart (SQLite) | Planned |
| R08 | Burst Edit Pattern | Faster time-to-first-visible-text for Telegram + Desktop | Planned |
| R09 | /forget History Cleanup | /forget also cleans conversation history | Planned |
| R10 | Multi-AI Debate | Query multiple models in parallel, side-by-side + consensus analysis | Done (MVP) |
| R11 | Cross-Provider Memory | Memory portable across all providers | Planned |
| R12 | Smart Privacy Routing | Sensitive requests local, non-critical to the cloud | Planned |
| R13 | Voice Input/Output | Voice input + voice output (Whisper + TTS) | Planned |
| R14 | Plugin System | Extensible tool architecture for community plugins | Planned |
| R15 | Skill Marketplace | Community skills with premium + revenue sharing | Planned |
| R16 | Team Collaboration | Shared conversations + memory for teams | Planned |
| R17 | Self-Hosted Enterprise | On-premise deployment for businesses | Planned |
| R18 | User Model Override | /setmodel + /models + model switching logic in pool | Phase 1 done |

## Consciously Accepted Risks (Phase 1)

* SQLite storage without encryption-at-rest (Phase 1+: Fernet)
* Audit log without hash chain (tamper protection Phase 1+)
* `_creation_locks` map grows unbounded (irrelevant for <10 users)
* `_reset_all_for_tests` technically reachable in production (low risk: module function)

## Statistics (as of 2026-05-10)

* Tests: 602+ passing
* Production code: ~9,700 LOC
* Test code: ~10,300 LOC
* Layer contracts: 3/3 kept
* Pre-commit hooks: 13/13 passing
* Local commits since Tier 1: 15
