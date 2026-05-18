# ADR-0005: Public-Private Boundary

**Status:** Accepted
**Date:** 2026-05-18
**Deciders:** AXOLENT core team

## Context

AXOLENT is published as open source under AGPL-3.0. To enable sustainable
development while preserving commercial differentiation, some future
intelligence components may be developed as proprietary modules that
integrate with the public codebase via Protocol interfaces.

External contributors and users need clarity on:

* What is open source today
* What may be optional proprietary additions in the future
* How the two integrate technically

Without explicit boundaries, contributors risk building features that
later become proprietary, and users risk assuming the public repository
depends on closed-source modules.

## Decision

1. Everything visible in this repository is open source under AGPL-3.0.

2. The public repository must always be fully functional without any
   proprietary modules. No mandatory closed-source dependency.

3. Proprietary modules, if developed, integrate via documented Protocol
   interfaces. These interfaces are public; implementations may be
   public or proprietary.

4. The boundary is documented in
   [docs/PUBLIC_PRIVATE_BOUNDARY.md](../PUBLIC_PRIVATE_BOUNDARY.md)
   in user-facing language.

5. Proprietary modules live in separate packages or repositories. They
   are never committed to this repository.

## Consequences

**Positive:**

* External contributors have clarity about scope
* AGPL-3.0 license is respected: derivative works of the public code
  must share their modifications
* The project remains genuinely open source, not a teaser for proprietary
  software
* Commercial sustainability via optional proprietary modules is preserved

**Negative:**

* Maintaining Protocol interfaces requires discipline (no leaky
  abstractions into proprietary internals)
* Two test suites required if proprietary modules exist (public tests
  pass without proprietary code; proprietary tests live elsewhere)

**Mitigations:**

* The base InstructionCompiler and ExecutionKernel are public, providing
  a complete out-of-the-box experience
* Protocol interfaces use Python's `typing.Protocol` for structural
  subtyping: no need for inheritance from proprietary base classes

## Alternatives Considered

| Alternative | Rejected because |
|---|---|
| Fully open source, no proprietary modules ever | Eliminates commercial differentiation; sustainability risk |
| Fully proprietary, no open source | Contradicts the project's open source mission |
| Public stubs, proprietary implementations only | Public users would have non-functional baseline |
| Detailed boundary listing naming specific modules | Leaks roadmap and implementation details prematurely |

## References

* [docs/PUBLIC_PRIVATE_BOUNDARY.md](../PUBLIC_PRIVATE_BOUNDARY.md)
* [LICENSE](../../LICENSE)
* [docs/adr/0002-hexagonal-architecture.md](0002-hexagonal-architecture.md):
  Protocol-based integration aligns with hexagonal architecture's
  ports-and-adapters pattern
* [docs/adr/0003-execution-kernel-architecture.md](0003-execution-kernel-architecture.md):
  Execution Kernel as the public baseline
