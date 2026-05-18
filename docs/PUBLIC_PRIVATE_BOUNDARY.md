# Public vs Private Boundary

This document explains what is open source in this repository and what
may exist as optional proprietary components in the future.

## TL;DR

* Everything visible in this repository is open source under AGPL-3.0
* The repository is and will remain fully functional on its own
* AXOLENT may develop optional proprietary modules that integrate via
  documented Protocol interfaces; these are enhancements, not core
  dependencies

## What is Public

Everything currently in this repository:

* Telegram bridge and presentation layer
* Provider adapters (Claude CLI subprocess, future provider integrations)
* Hexagonal architecture: domain / application / infrastructure / presentation
* Execution Kernel (RequestEnvelope, ContextKernel, InstructionCompiler)
* Memory system (Trinity: episodic, semantic, procedural)
* Streaming handler with live multi-message rollover
* Language detection and resolution (20 languages)
* i18n infrastructure
* Rate limiting (4 profiles)
* Audit logging
* All tests in `bridge/tests/`
* All documentation in `docs/`

## What May Be Private

AXOLENT reserves the right to develop optional proprietary modules that
extend the public codebase. These would integrate via Protocol interfaces
defined in the public repo. Examples of integration points where future
proprietary modules may dock:

* Custom prompt assembly logic (beyond the base InstructionCompiler)
* Advanced output validation pipelines
* User-personalization heuristics
* Domain-specific scoring or ranking systems

The Protocol interfaces themselves remain public. Only specific
implementations of these interfaces may exist as separate proprietary
packages.

## Guarantees

1. **Functional baseline:** The public repository must always be fully
   functional on its own. No proprietary module is required to run the
   bot, the Telegram bridge, or the test suite.

2. **No proprietary content in public commits:** No proprietary logic,
   scoring tables, or training/evaluation datasets will be committed
   to this repository.

3. **Protocol stability:** Once a Protocol interface is documented for
   third-party (or future proprietary) implementations, breaking changes
   require a major version bump and migration notes.

4. **Open source first:** Improvements that benefit the public codebase
   (better tests, cleaner abstractions, security fixes) land here, not
   in any proprietary fork.

## How to Recognize Public vs Private

If you see a file in this repository: it is public.

If a future proprietary module exists, it will:

* Live in a separate package or repository
* Be installable optionally (e.g. via `pip install axolent-pro`)
* Implement Protocol interfaces defined here
* Not be required for the public repo to work

## For Contributors

* You may freely modify, extend, and redistribute the public codebase
  under AGPL-3.0.
* You should not assume any planned proprietary modules. Focus your
  contributions on what is in this repo today.
* If you want to build your own proprietary or open-source extensions:
  use the documented Protocol interfaces as integration points.

## Related Documents

* [ARCHITECTURE.md](ARCHITECTURE.md): System architecture and layer rules
* [adr/0005-public-private-boundary.md](adr/0005-public-private-boundary.md):
  ADR for this boundary decision
* [../LICENSE](../LICENSE): AGPL-3.0 full text
