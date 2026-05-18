# Architecture

AXOLENT AI uses **Hexagonal Architecture** (Ports and Adapters) with four layers,
enforced by `import-linter` contracts. This document describes the system structure,
data flow, and design decisions.

## Table of Contents

* [System Overview](#system-overview)
* [Layer Rules](#layer-rules)
* [Request Flow](#request-flow)
* [Composition Root](#composition-root)
* [Streaming Flow](#streaming-flow)
* [Execution Kernel](#execution-kernel)
* [Language Resolution Flow](#language-resolution-flow)
* [Audit Flow](#audit-flow)
* [Module Map](#module-map)
* [Where to Add New Features](#where-to-add-new-features)
* [Related Documents](#related-documents)

## System Overview

```
+-------------------+
|   Telegram User   |
+--------+----------+
         |
         | Telegram Bot API (long-polling)
         v
+--------+----------+     +--------------------------+
| presentation/     |     | Composition Root         |
|   handlers.py     |     |   main.py                |
|   render.py       |     |   (wires all layers)     |
|   decorators.py   |     +--------------------------+
|   callbacks.py    |
+--------+----------+
         |
         | Calls application services
         v
+--------+----------+
| application/      |
|   chat_service    |-----> execution/
|   debate_orch.    |       * ContextKernel
|   bookmark_svc    |       * InstructionCompiler
|   memory_svc      |       * ExecutionContext
|   rate_limiter    |       * RequestEnvelope
|   streaming_hndlr |
|   language_rslvr  |
|   provider_router |
|   fallback_rslvr  |
|   status_manager  |
+--------+----------+
         |
    +----+----+
    |         |
    v         v
+---+---+ +---+--------+
|domain/| |infra/       |
| pure  | | claude_pool |
| logic | | sqlite      |
|       | | audit_log   |
|       | | providers/  |
+-------+ +------------+
```

### High-Level Data Flow

```
Telegram Update
  -> presentation/handlers.py       (parse, validate access, rate-limit check)
  -> application/chat_service.py    (orchestrate: history, memory, language)
  -> application/execution/kernel   (build ExecutionContext from RequestEnvelope)
  -> application/execution/compiler (assemble system + user prompts)
  -> application/provider_router    (select LLM provider)
  -> infrastructure/claude_pool     (spawn CLI subprocess, stream tokens)
  -> application/streaming_handler  (aggregate tokens, throttle edits)
  -> presentation/render.py         (convert Markdown to Telegram HTML, chunk, send)
```

## Layer Rules

The architecture enforces strict import boundaries via three `import-linter`
contracts defined in `bridge/.importlinter`:

### Contract 1: Hexagonal Layers

```
presentation  (top)
application
infrastructure
domain        (bottom)
```

Higher layers may import from lower layers. Lower layers must never import
from higher layers.

### Contract 2: Domain Purity

`domain/` must not import from `application/`, `infrastructure/`, or `presentation/`.
Domain modules contain only pure business logic: data structures, validation rules,
language detection, markdown conversion, personality models. No I/O, no framework
imports, no side effects.

**Domain modules:**

| Module | Responsibility |
|--------|---------------|
| `language.py` | Language detection via Unicode scripts and marker-word heuristics |
| `conversation.py` | Conversation turn model and context-block builder |
| `bookmark.py` | Bookmark formatting and validation |
| `personality.py` | Personality feature definitions (P1 through P6) and prompt assembly |
| `markdown.py` | Markdown to Telegram HTML conversion, URL scheme whitelist |
| `onboarding.py` | Onboarding wizard step definitions |
| `task_slot.py` | Task slot definitions for provider routing |
| `i18n.py` | i18n domain types |
| `text_guard/` | Text guard domain types |
| `memory/` | Memory domain models (episodic, semantic, procedural) |

### Contract 3: Presentation Isolation

`presentation/` must not import directly from `infrastructure/`. All cross-layer
access goes through Application Services. This ensures that swapping the transport
layer (e.g., from Telegram to a Desktop app) does not require changes to
infrastructure code.

**Presentation modules:**

| Module | Responsibility |
|--------|---------------|
| `handlers.py` | Telegram command and message handlers |
| `render.py` | Chunking, Markdown-to-HTML conversion, response caching |
| `decorators.py` | `@require_whitelist`, `@require_private_chat` |
| `callbacks.py` | Inline keyboard callback handlers (bookmarks) |
| `onboarding_callbacks.py` | Onboarding wizard callback handlers |
| `settings_callbacks.py` | Settings menu callback handlers |

## Request Flow

A typical `/chat` message follows this path:

```
1. Telegram sends an Update to the bot via long-polling.

2. presentation/handlers.py::handle_message()
   * Validates whitelist access (@require_whitelist)
   * Checks rate limit
   * Acquires per-user concurrency lock
   * Creates RequestEnvelope with user_id, chat_id, text, metadata

3. application/execution/kernel.py::ContextKernel.build()
   * Runs resolver pipeline:
     a. LanguageResolverAdapter  (sticky/detected/override/default)
     b. ChannelResolver          (Telegram capabilities)
     c. TimeResolver             (localized time context)
   * Produces frozen ExecutionContext

4. application/execution/instruction_compiler.py::InstructionCompiler.compile_chat()
   * Assembles system prompt in fixed block order:
     [1] Security / Non-disclosure
     [2] Privacy / Tool restrictions
     [3] User language lock
     [4] Task objective (base prompt)
     [5] Time / location / channel context
     [6] Memory with provenance
     [7] Style / personality (P1-P6 features)
     [8] Output format contract
   * Produces CompiledPrompt (system_prompt + user_prompt + metadata)

5. application/chat_service.py::process_user_message_streaming()
   * Loads conversation history (last 20 turns)
   * Loads relevant memory entries (episodic + semantic)
   * Passes CompiledPrompt to ProviderRouter

6. application/provider_router.py::ProviderRouter.query()
   * Selects provider (default: claude_persistent)
   * Delegates to infrastructure provider

7. infrastructure/claude_process_pool.py::ClaudeProcessPool
   * Reuses persistent subprocess keyed by (user_id, chat_id)
   * Sends prompt via stdin pipe
   * Yields StreamEvent objects (NDJSON parsed)

8. application/streaming_handler.py::StreamingSession
   * Aggregates tokens with burst-then-throttle curve
   * Edits Telegram message in-place during streaming
   * Converts final Markdown to Telegram HTML

9. presentation/render.py
   * Splits responses >4096 chars at sensible boundaries
   * Sends via Telegram Bot API
   * Caches response for bookmark saving
```

## Composition Root

`bridge/main.py` is the single wiring point for all dependencies. It:

* Loads environment variables from `.env`
* Validates the ALLOW_ALL_USERS safeguard
* Creates infrastructure instances (SQLite connections, process pool, audit log)
* Runs legacy migrations (JSONL to SQLite)
* Creates application services with constructor injection
* Registers Telegram handlers, command handlers, and callback handlers
* Starts long-polling

No business logic lives in `main.py`. It only connects layers.

## Streaming Flow

AXOLENT AI uses live token streaming with adaptive throttling (T23 Live-Rollover).

```
ClaudeProcessPool (NDJSON stream)
  |
  | StreamEvent(type, text, metadata)
  v
StreamingSession
  |
  | Burst mode: first 5 edits at 0.2s intervals
  | Throttle mode: graduated curve up to 1.5s
  | Flood control: on Telegram 429, pause + exponential backoff
  v
Telegram editMessageText()
  |
  | Intermediate: Markdown rendered live, incomplete tokens trimmed
  | Final: full Markdown-to-HTML conversion, multi-message split
  v
User sees tokens appear in real-time
```

**Streaming modes** (configurable via `AXOLENT_STREAMING_MODE`):

* `telegram` (default): Burst-then-throttle optimized for Telegram rate limits
* `local`: No throttling (for desktop app or direct socket connections)

**Final edit priority:** The last edit (complete response) is retried on Telegram
429 errors to ensure the user always sees the full response.

## Execution Kernel

The Execution Kernel (introduced in Phase 0, commits 766f3a3 through fd2d98e) is
the central pipeline that eliminates scattered context resolution.

```
RequestEnvelope (raw input)
  |
  v
ContextKernel.build()
  |
  | Resolvers (ordered pipeline):
  |   1. LanguageResolverAdapter
  |   2. ChannelResolver
  |   3. TimeResolver
  |
  v
ExecutionContext (frozen dataclass, single source of truth)
  |
  v
InstructionCompiler.compile_chat()
  |
  | Reads: ExecutionContext + ExecutionPlan
  | Produces: CompiledPrompt
  |
  v
CompiledPrompt (system_prompt + user_prompt + metadata)
```

**Design principles:**

* `ExecutionContext` is a frozen dataclass. Once built, it cannot drift.
* No downstream component may resolve context independently.
* The `InstructionCompiler` assembles prompts in a fixed block order for
  auditability and consistency.
* `request_id` flows through the entire pipeline for audit correlation.

See [docs/adr/0003-execution-kernel-architecture.md](docs/adr/0003-execution-kernel-architecture.md)
for the decision rationale.

## Language Resolution Flow

Language detection and resolution is handled by a single entry point:
`application/language_resolver.py::LanguageResolver`.

```
User sends message
  |
  v
LanguageResolver.resolve(user_id, chat_id, text)
  |
  | Priority:
  |   1. Override (explicit /lang command)     -> confidence 1.0
  |   2. Sticky (stored per chat_id)           -> confidence 1.0
  |   3. Detected (domain/language.py)         -> confidence 0.0-1.0
  |   4. Default fallback ("de")               -> confidence 1.0
  |
  | Smart-switch: detection must exceed 0.7 confidence
  | to override the sticky language.
  |
  v
LanguageContext (frozen dataclass)
  * code: ISO-639-1 (guaranteed non-empty)
  * source: "override" | "sticky" | "detected" | "default"
  * confidence: float
  * request_id: UUID for audit
```

**Detection strategy** (`domain/language.py`):

1. Non-Latin scripts (Arabic, Chinese, Japanese, Korean, Hindi, Thai, Cyrillic)
   are detected deterministically via Unicode ranges.
2. Cyrillic text is further classified as Russian vs Ukrainian via distinctive
   markers.
3. Latin-script languages are scored by frequency of marker words.
4. Marker-precedence logic: shared diacritical characters (e.g., `a`/`o` in
   Swedish vs German) are resolved by highest word-marker score.

See [docs/adr/0004-language-resolution-contract.md](docs/adr/0004-language-resolution-contract.md)
for the marker-precedence decision and [docs/I18N.md](docs/I18N.md) for the
i18n system.

## Audit Flow

Every request generates audit entries for observability and debugging.

```
Request arrives
  |
  | request_id = UUID (generated in LanguageResolver or handler)
  v
presentation/handlers.py
  | log_command_audit(action, user_id, chat_id, request_id)
  v
application/chat_service.py
  | write_audit_log({event_type: "llm_call", request_id, ...})
  v
infrastructure/audit_log.py
  | Append to JSONL file with rotation
  | Fields: timestamp, event_type, user_id, chat_id, request_id,
  |         provider, model, duration, token_count, language, ...
```

All audit entries share the same `request_id`, enabling end-to-end correlation
from the incoming Telegram update through language resolution, prompt compilation,
LLM call, and response delivery.

## Module Map

### Application Layer (Orchestration)

| Module | Responsibility |
|--------|---------------|
| `chat_service.py` | Main LLM call orchestration (history, memory, prompt, stream) |
| `execution/kernel.py` | ContextKernel: builds ExecutionContext |
| `execution/instruction_compiler.py` | Assembles system + user prompts |
| `execution/context.py` | ExecutionContext (frozen, single source of truth) |
| `execution/envelope.py` | RequestEnvelope (raw input container) |
| `execution/plan.py` | ExecutionPlan (what to do with the context) |
| `execution/resolvers.py` | Pipeline resolvers (Language, Channel, Time) |
| `debate_orchestrator.py` | Multi-AI debate: parallel queries, consensus analysis |
| `language_resolver.py` | Single-entry language resolution |
| `provider_router.py` | LLM provider selection |
| `fallback_resolver.py` | Automatic provider failover |
| `streaming_handler.py` | Token aggregation and Telegram edit throttling |
| `rate_limiter.py` | Per-user rate limiting (4 profiles) |
| `status_manager.py` | Contextual status indicators during processing |
| `prompt_composer.py` | Legacy prompt builder (being replaced by InstructionCompiler) |
| `memory_service.py` | Memory retrieval and injection |
| `bookmark_service.py` | Bookmark CRUD operations |
| `model_service.py` | User model selection (/setmodel, /models) |
| `audit_service.py` | Audit logging use-case wrapper |
| `style_adaption_service.py` | Response style adaptation |
| `task_router.py` | Task classification for provider routing |
| `text_guard_service.py` | Text guard coordination |
| `consolidator.py` | Memory consolidation hook (episodic dedup, semantic promotion, aging/decay) |
| `leakage_filter.py` | Checks LLM responses for system prompt leakage (C-3 countermeasure) |
| `memory_translation_service.py` | On-the-fly translation of memory entries for /memory display |
| `model_registry.py` | Static model registry: loads model metadata from YAML config |
| `ollama_service.py` | Ollama auto-start: detects and starts local Ollama at bot startup |
| `proactive_trigger_service.py` | Proactive memory nudges and time-based triggers (P1, P5 personality) |
| `self_awareness_service.py` | Builds self-awareness block for system prompt (model identity, slot occupancy) |

### Infrastructure Layer (I/O)

| Module | Responsibility |
|--------|---------------|
| `claude_process_pool.py` | Persistent subprocess pool, NDJSON streaming |
| `sqlite_storage.py` | SQLite storage for bookmarks, memory, profiles, rate limits |
| `audit_log.py` | JSONL audit log with rotation |
| `conversation_storage.py` | In-memory conversation history storage |
| `memory_storage.py` | Memory storage abstraction |
| `bookmark_storage.py` | Legacy JSONL bookmark adapter + migration |
| `onboarding_storage.py` | Onboarding state storage |
| `personality_loader.py` | Personality prompt file loader |
| `encoding.py` | UTF-8 encoding helper (errors=replace, ensure_ascii=False) |
| `providers/base.py` | LLMProvider protocol + ProviderResponse |
| `providers/claude_persistent.py` | Claude Mode B via persistent subprocess |
| `providers/claude_cli.py` | Claude Mode B via one-shot subprocess |
| `providers/ollama_local.py` | Ollama local provider |
| `providers/gemini_cli.py` | Gemini provider |
| `providers/openai_codex_cli.py` | OpenAI Codex provider |
| `providers/mistral_vibe_cli.py` | Mistral Vibe provider |

## Where to Add New Features

| Type of change | Where to put it |
|---------------|-----------------|
| New Telegram command | `presentation/handlers.py` + register in `main.py` |
| New business rule (no I/O) | `domain/` (new module or extend existing) |
| New LLM provider | `infrastructure/providers/` (implement `LLMProvider` protocol) |
| New storage backend | `infrastructure/` (implement storage protocol) |
| New orchestration logic | `application/` (new service or extend existing) |
| New i18n key | `bridge/i18n/locales/en.json` + run sync scripts |

## Related Documents

* [README.md](../README.md): Project overview, setup, and vision
* [CONTRIBUTING.md](../CONTRIBUTING.md): How to contribute
* [docs/DEVELOPMENT.md](DEVELOPMENT.md): Development setup and debugging
* [docs/TESTING.md](TESTING.md): Test conventions and pre-commit hooks
* [docs/I18N.md](I18N.md): Internationalization system
* [docs/THREAT_MODEL.md](THREAT_MODEL.md): Security threat model
* [docs/FEATURE_STATUS.md](FEATURE_STATUS.md): Feature status and roadmap
* [docs/PUBLIC_PRIVATE_BOUNDARY.md](PUBLIC_PRIVATE_BOUNDARY.md): Public vs private boundary
* [docs/adr/](adr/): Architecture Decision Records
