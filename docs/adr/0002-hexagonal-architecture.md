# ADR-0002: Hexagonal Architecture

**Status:** Accepted
**Date:** 2026-03-15
**Decision makers:** Jessica (AXOLENT AI)

## Context

As AXOLENT AI grew from a simple Telegram bot to a multi-provider, multi-feature
assistant, the risk of spaghetti code increased. Business logic was mixing with
Telegram API calls, storage operations were scattered across handlers, and testing
required standing up the full application.

The project needed clear boundaries to:

* Keep domain logic testable without I/O
* Allow swapping the transport layer (Telegram today, Desktop tomorrow)
* Prevent accidental coupling between layers
* Enable onboarding of external contributors who can immediately understand
  where code belongs

## Decision

Adopt Hexagonal Architecture (Ports and Adapters) with four layers, enforced
by `import-linter` contracts.

### Layer Definitions

```
presentation/      Telegram-specific: handlers, rendering, decorators
application/       Use-case orchestration: services, resolvers, compilers
infrastructure/    I/O adapters: CLI subprocess, SQLite, audit log, providers
domain/            Pure business logic: no I/O, no framework imports
```

### Import Contracts (enforced by import-linter)

Three contracts defined in `bridge/.importlinter`:

1. **Hexagonal Layers:** `presentation > application > infrastructure > domain`.
   Higher layers may import from lower layers, never the reverse.
2. **Domain Purity:** `domain/` must not import from `application/`,
   `infrastructure/`, or `presentation/`.
3. **Presentation Isolation:** `presentation/` must not import directly from
   `infrastructure/`. All cross-layer access goes through Application Services.

### Composition Root

`bridge/main.py` is the single wiring point. It creates all instances,
injects dependencies, and starts the application. No business logic lives
in `main.py`.

## Consequences

### Positive

* **Testability:** Domain tests are pure functions. Application tests mock
  infrastructure. Presentation tests mock application services.
* **Clarity:** Every module has a clear home. New contributors know immediately
  where to put code.
* **Swappability:** Adding a Desktop app means adding a new `presentation/`
  variant without touching application or domain logic.
* **Automated enforcement:** `import-linter` runs on every commit. Layer
  violations are caught before they land.

### Negative

* **Boilerplate:** Some orchestration code in `application/` exists solely
  to pass data between layers.
* **Learning curve:** Contributors must understand the layer rules before
  making changes.
* **Indirection:** Presentation cannot call infrastructure directly, which
  sometimes adds an extra service method.

### Rules for Contributors

* `domain/` may only contain pure logic: dataclasses, validation, detection,
  formatting. No `import os`, no `import sqlite3`, no Telegram imports.
* `presentation/` handles Telegram specifics. To access storage, call an
  Application Service (e.g., `BookmarkService.save()` instead of
  `SqliteBookmarkStorage.insert()`).
* `infrastructure/` implements I/O: CLI subprocess, SQLite, file system,
  audit logging. It may import from `domain/` for data structures.
* New features go in the appropriate layer. When unsure, ask: "Does this
  need I/O?" If yes, it is not domain. "Does this need Telegram?" If yes,
  it is presentation.

## Verification

```bash
python scripts/run_with_venv.py lint-imports
```

This checks all three contracts. A violation produces a clear error message
identifying the offending import.

## References

* [docs/ARCHITECTURE.md](../ARCHITECTURE.md): Full system overview
* [bridge/.importlinter](../../bridge/.importlinter): Contract definitions
* [.pre-commit-config.yaml](../../.pre-commit-config.yaml): Hook configuration
