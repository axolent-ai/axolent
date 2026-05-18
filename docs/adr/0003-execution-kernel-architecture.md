# ADR-0003: Execution Kernel Architecture

**Status:** Accepted (Phase 0 complete)
**Date:** 2026-05-10
**Decision makers:** Jessica (AXOLENT AI)
**Commits:** 766f3a3 through fd2d98e

## Context

Before the Execution Kernel, language resolution and context assembly were
scattered across multiple components:

* `handlers.py` resolved language via `get_language()` and `detect_language()`
* `chat_service.py` resolved language again with different fallback logic
* `debate_orchestrator.py` had its own language detection path
* `prompt_composer.py` assembled prompts with inconsistent context

This led to a **whack-a-mole bug pattern**: fixing a language bug in one
component would introduce a regression in another. The root cause was
architectural: there was no single source of truth for request context.

Specific problems:

* **Sticky language contamination:** A German sticky language would leak
  into English-initiated conversations because different components read
  the sticky value at different times.
* **Inconsistent fallbacks:** Some paths fell back to `"de"`, others to `"en"`,
  others to whatever `detect_language()` returned.
* **No audit correlation:** There was no `request_id` flowing through the
  entire request lifecycle, making debugging difficult.

## Decision

Introduce the **Execution Kernel**: a central pipeline that builds an
immutable `ExecutionContext` from a `RequestEnvelope`, with a fixed resolver
pipeline and a single-path prompt compiler.

### Components

```
RequestEnvelope          Raw input (user_id, chat_id, text, metadata)
      |
      v
ContextKernel            Runs resolver pipeline, produces ExecutionContext
      |
      | Resolvers (ordered):
      |   1. LanguageResolverAdapter
      |   2. ChannelResolver
      |   3. TimeResolver
      v
ExecutionContext          Frozen dataclass, single source of truth
      |
      v
InstructionCompiler      Assembles prompts in fixed block order
      |
      v
CompiledPrompt           system_prompt + user_prompt + metadata
```

### Design Principles

1. **Single source of truth:** `ExecutionContext` is a frozen dataclass.
   Once built by `ContextKernel`, it cannot be modified. All downstream
   components receive it as a parameter.

2. **No independent resolution:** No component downstream of the kernel
   may resolve language, time, or channel context independently. They
   must use the values from `ExecutionContext`.

3. **Fixed block order:** The `InstructionCompiler` assembles the system
   prompt in a strict order:
   ```
   [1] Security / Non-disclosure
   [2] Privacy / Tool restrictions
   [3] User language lock
   [4] Task objective (base prompt)
   [5] Time / location / channel context
   [6] Memory with provenance
   [7] Style / personality
   [8] Output format contract
   ```
   This order is never violated. It ensures auditability and consistency.

4. **request_id correlation:** A UUID is generated at the start of every
   request and flows through the entire pipeline, from `LanguageResolver`
   through `ExecutionContext` to audit log entries.

5. **Resolver pipeline is ordered:** `LanguageResolverAdapter` must run
   before `TimeResolver` because the weekday name depends on the resolved
   language.

## Consequences

### Positive

* **Whack-a-mole eliminated:** Language is resolved exactly once, in
  exactly one place. No component can override it.
* **Debuggable:** `request_id` enables end-to-end correlation in the
  audit log.
* **Auditable prompts:** The `InstructionCompiler` produces a
  `CompiledPrompt` with metadata showing exactly which blocks were
  included and why.
* **Testable:** `ContextKernel` and `InstructionCompiler` are independently
  testable with mock resolvers and contexts.
* **Extensible:** Adding a new context dimension (e.g., user preferences,
  device type) means adding a new resolver to the pipeline.

### Negative

* **Migration overhead:** Existing code paths (`chat_service`, `debate_orchestrator`)
  needed to be updated to accept `ExecutionContext` instead of resolving
  context independently. This was done incrementally across Phase 0.
* **Two code paths during migration:** Until all consumers are migrated,
  both the old `PromptComposer` and the new `InstructionCompiler` coexist.
  The `PromptComposer` is being phased out.

### Phase 0 Scope

Phase 0 implemented blocks 3 through 7 of the `InstructionCompiler`:

* [3] User language lock
* [4] Task objective (base prompt)
* [5] Time / location / channel context
* [6] Memory with provenance
* [7] Style / personality

Blocks 1 (Security), 2 (Privacy), and 8 (Output format) are stubs,
to be implemented in Phase 1.

## References

* [docs/ARCHITECTURE.md](../ARCHITECTURE.md): Execution Kernel in system context
* `bridge/application/execution/kernel.py`: ContextKernel implementation
* `bridge/application/execution/instruction_compiler.py`: InstructionCompiler
* `bridge/application/execution/context.py`: ExecutionContext frozen dataclass
* `bridge/application/execution/envelope.py`: RequestEnvelope
* `bridge/application/execution/resolvers.py`: Resolver pipeline
