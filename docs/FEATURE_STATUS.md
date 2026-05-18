# Feature Status

Overview of all features, their status, and current implementation owner.
Updated as features progress through development.

## Status Legend

| Status | Meaning |
|--------|---------|
| **Active** | In production, working |
| **Planned** | On roadmap, not yet started |
| **Experimental** | Implemented but may change |
| **Deprecated** | Being phased out |

## Feature Matrix

| Feature | Status | Owner | Notes |
|---------|--------|-------|-------|
| Telegram Bridge | Active | `presentation/` | Main user interface, long-polling |
| Hexagonal Architecture | Active | `bridge/` | 4 layers, 3 import-linter contracts |
| Mode B (CLI Wrapper) | Active | `infrastructure/` | No OAuth hijacking, subprocess-based |
| Multi-AI Debate | Active | `application/debate_orchestrator` | Parallel queries, Jaccard consensus analysis |
| Memory System (Trinity) | Active | `domain/memory/`, `application/memory_service` | Episodic, semantic, procedural; SQLite + FTS5 |
| Streaming Handler | Active | `application/streaming_handler` | Live-Rollover since commit d2fad5c, burst-then-throttle |
| Language Detection | Active | `domain/language` | 20 languages, Unicode + marker-word heuristics |
| Language Resolution | Active | `application/language_resolver` | Sticky, smart-switch, confidence-based |
| i18n System | Active | `bridge/i18n/` | 20 languages, 154+ keys, source_hash tracking |
| Rate Limiting | Active | `application/rate_limiter` | 4 profiles: Light, Normal, Power, Unlimited |
| Provider Failover | Active | `application/fallback_resolver` | Auto-failover for non-streaming operations |
| Execution Kernel | Active | `application/execution/` | ContextKernel + InstructionCompiler + ExecutionContext |
| Personality Features (P1-P6) | Active | `domain/personality` | User-configurable personality traits |
| Active Curiosity (T28) | Active | `application/`, `domain/` | Memory-aware LLM instructions, genuine interest |
| Bookmarks | Active | `application/bookmark_service` | Reply-based /save, inline buttons, SQLite |
| Conversation History | Active | `infrastructure/conversation_storage` | In-memory, 20 turns max, sticky language |
| Audit Logging | Active | `infrastructure/audit_log` | JSONL with rotation, request_id correlation |
| User Model Override | Active | `application/model_service` | /setmodel, /models, /resetmodel |
| Typing Keepalive | Active | `presentation/` | TYPING indicator every 4s |
| Status Manager | Active | `application/status_manager` | Phase-aware status indicators |
| Markdown Rendering | Active | `domain/markdown` | Telegram HTML, UUID sentinels, URL whitelist |
| Privacy Guards | Active | `presentation/decorators` | Whitelist, private chat, leakage guard |
| Onboarding Wizard | Active | `presentation/onboarding_callbacks` | 20-language wizard (interim design) |
| Desktop App | Planned | `desktop/` | Tauri-based shell |
| Mini App | Planned | `mini-app/` | Telegram Mini App |
| Plugin SDK | Planned | N/A | Community plugin interface |
| Cross-Provider Memory | Planned | N/A | Memory portable across providers (R11) |
| Smart Privacy Routing | Planned | N/A | Sensitive content routed locally (R12) |
| Persistent History | Planned | N/A | SQLite-based conversation persistence (R07) |
| Streaming Failover | Planned | N/A | Auto-failover during streaming (v1.1) |
| Encryption at Rest | Planned | N/A | Fernet encryption on SQLite DB |
| Audit Hash Chain | Planned | N/A | Tamper detection for audit log |
| JSONL Bookmark Adapter | Deprecated | `infrastructure/bookmark_storage` | Replaced by SQLite, migration path exists |

## Roadmap Reference

For the full R01-R18 roadmap with descriptions, see [FEATURES.md](FEATURES.md).

## Related Documents

* [FEATURES.md](FEATURES.md): Detailed feature descriptions
* [ARCHITECTURE.md](ARCHITECTURE.md): System structure
* [README.md](../README.md): Project overview and roadmap table
