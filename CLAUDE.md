# AXOLENT AI, Coding Conventions

**Scope:** This file defines project-specific conventions for AXOLENT AI.
For Atlas-orchestrated agents this overrides any conflicting global rule.

---

## 1. Language Rule, English-Only

This project is **English-only** for all code, docs, and user-facing strings.
This explicitly overrides any external project rule that says "German in
comments, English in identifiers".

### Mandatory English

* All docstrings
* All inline comments
* All log messages (`log.info()`, `log.warning()`, `log.error()`, etc.)
* All exception messages (`raise ValueError("...")`)
* All variable, function, class, module names (was already global rule)
* All test names, test docstrings, assertion messages
* All README, FEATURES, THREAT_MODEL and other public-facing documentation
* All commit messages and pull request descriptions
* All YAML configuration keys and structural elements

### Mandatory German (intentional exceptions)

* `bridge/config/task_slots.yaml`, German keywords are a **feature**, not a bug.
  They enable the TaskRouter to classify German user inputs.
  English keywords must be added alongside, not replacing.
* Internal `docs/` files that are not public-facing (code reviews, briefings,
  brand-naming history, internal research), these are historical context and
  may stay German. They will be removed from the public repository via
  `git filter-repo` before public release.

### User-facing Strings (Bot Responses)

* Default language: English
* Multi-language support: via i18n key system, NOT hardcoded strings
* Localizations live in dedicated translation files / dictionaries
* When adding new user-facing text: add to i18n dictionary, default to English

---

## 2. Style Rules

* **No em-dashes (,)** or en-dashes (-). Use commas, colons, parentheses, or
  periods instead.
* **Bullets:** use `*` or numbered lists. Never `-` as bullet marker.
* **Comments:** prefer no comments. Only add when the WHY is non-obvious
  (hidden constraint, subtle invariant, workaround for a specific bug,
  surprising behavior). Do not explain WHAT the code does.

---

## 3. Architecture Rules (Hexagonal)

These come from the Layer-Linter contracts and must not be broken:

* `domain/` may import nothing from `application/`, `infrastructure/`,
  or `presentation/`
* `presentation/` may not import directly from `infrastructure/`
* All cross-layer access goes through Application Services

Run `python scripts/run_with_venv.py lint-imports` to verify.

---

## 4. Mode B Reminder

AXOLENT AI uses **Mode B**: a local CLI wrapper that spawns the official
`claude` CLI as a subprocess. The user has their own Claude Pro/Max
subscription. There is no OAuth token hijacking, no custom API key, no
proxy server. Anthropic explicitly permits this pattern.

Never confuse Mode B (allowed) with Mode A (OAuth token hijacking, forbidden).

When reviewing or extending code: ensure no provider call ever bypasses the
subprocess wrapper. No direct HTTP requests to Anthropic, no SDK imports,
no token reading from `~/.claude`.

---

## 5. Test Conventions

* Test file naming: `test_<module>.py`
* Test class naming: `TestSomething` (PascalCase, English)
* Test method naming: `test_does_something_when_condition` (snake_case, English)
* Test docstrings: English, describing the WHY of the test
* Use pytest fixtures for setup, avoid setUp/tearDown unless using unittest.TestCase

---

## 6. Pre-Commit Hooks (active)

* ruff (lint)
* ruff-format (formatter)
* trailing-whitespace
* end-of-file-fixer
* check-yaml
* check-added-large-files
* check-merge-conflict
* mixed-line-ending
* bandit (security scan)
* import-linter (architecture contracts)
* pytest (test suite)
* pip-audit (dependency vulnerabilities)
* semgrep (semantic security patterns)
* no-fake-umlauts (prevents ASCII umlaut regression in any remaining German text)
* en-only-production, blocks new German strings/comments in production code

All hooks must pass before commit.

---

## 7. Brand and Terminology Reference

* Brand name: **AXOLENT AI**
* Acronym: **A**daptive e**X**cellent **O**pen **L**ocal **E**ngaging
  **N**eural **T**hinker
* Mascot: **Axolent** (the axolotl)
* Domain: `axolent.ai`
* Repository: `github.com/axolent-ai/axolent`
* License: AGPL-3.0-only
* Architecture term: **Mode B** (lowercase "o")
* Tagline: "AIs that argue. Files that never leave. Privacy that holds.
  Memory that lasts."

---

## 8. Repository Hygiene

* `.env` and all secrets: never committed, gitignored
* Personal user configuration (`system_prompt.md`, `user_constitution.md`):
  never committed, gitignored. Use `.example.md` templates as canonical.
* Test artifacts (`pytest_tmp_*`, `htmlcov/`, `.pytest_cache/`, `.ruff_cache/`):
  gitignored
* Build artifacts (`__pycache__/`, `*.pyc`, `dist/`, `build/`): gitignored
* `bridge/data/` and `bridge/logs/`: gitignored (runtime data)

---

## 9. Briefing Note for Coding Agents

When an Atlas-delegated agent (Sigma, Dev, etc.) works in this project,
the following overrides apply automatically by reading this file:

* Write code, comments, docstrings, and log messages in English
* Do not introduce German into production code paths
* Do not break Mode B architecture (no SDK imports, no OAuth, no token read)
* Do not modify `docs/` files marked as "historical" by Codex audit unless
  explicitly instructed
* Run the full pre-commit suite before committing
