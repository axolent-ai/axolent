# Contributing to AXOLENT AI

Thank you for your interest in contributing to AXOLENT AI. This guide covers
setup, testing, code conventions, and the submission process.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Local Setup](#local-setup)
* [Running Tests](#running-tests)
* [Architecture Rules](#architecture-rules)
* [Branch Naming](#branch-naming)
* [Commit and PR Guidelines](#commit-and-pr-guidelines)
* [What Not to Commit](#what-not-to-commit)
* [Internationalization (i18n)](#internationalization-i18n)
* [Pre-Commit Hooks](#pre-commit-hooks)
* [Code of Conduct](#code-of-conduct)

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | 3.12 recommended |
| Claude Code CLI | Installed and logged in with your own Pro/Max subscription |
| Telegram Bot Token | Create via [@BotFather](https://t.me/BotFather) |
| Git | For cloning and committing |
| pre-commit | `pip install pre-commit` |

## Local Setup

```bash
# Clone the repository
git clone https://github.com/axolent-ai/axolent.git
cd axolent/bridge

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install with dev + test dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your Telegram bot token and user ID
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed setup instructions and
debugging tips.

## Running Tests

```bash
cd bridge

# Full test suite
pytest

# Specific layer
pytest tests/test_domain/
pytest tests/test_application/
pytest tests/test_presentation/
pytest tests/test_infrastructure/

# Quick run (no verbose output)
pytest -q --no-header
```

See [docs/TESTING.md](docs/TESTING.md) for test conventions, markers, and
how to write new tests.

## Architecture Rules

AXOLENT AI follows **Hexagonal Architecture** with four layers. These rules are
enforced by `import-linter` and will fail your commit if violated.

| Layer | Directory | May Import From | Must Not Import From |
|-------|-----------|-----------------|----------------------|
| Domain | `domain/` | Nothing external | `application/`, `infrastructure/`, `presentation/` |
| Application | `application/` | `domain/` | `presentation/` (infrastructure via DI) |
| Infrastructure | `infrastructure/` | `domain/`, `application/` | `presentation/` |
| Presentation | `presentation/` | `domain/`, `application/` | `infrastructure/` (directly) |

**Key rules:**

* `domain/` is pure business logic. No I/O, no framework imports, no side effects.
* `presentation/` handles Telegram specifics. All cross-layer access goes through
  Application Services.
* `infrastructure/` performs all I/O (CLI calls, storage, audit logging).
* `main.py` is the Composition Root. It wires everything together.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system overview and
[docs/adr/0002-hexagonal-architecture.md](docs/adr/0002-hexagonal-architecture.md) for
the decision rationale.

**Public vs Private:** This repository is fully open source under AGPL-3.0. Optional
proprietary modules may exist separately in the future but are never required. See
[docs/PUBLIC_PRIVATE_BOUNDARY.md](docs/PUBLIC_PRIVATE_BOUNDARY.md) for details.

## Branch Naming

```
feature/<short-description>    New features
fix/<short-description>        Bug fixes
refactor/<short-description>   Refactoring without behavior change
docs/<short-description>       Documentation only
test/<short-description>       Test additions or improvements
```

Examples: `feature/plugin-sdk`, `fix/streaming-429-retry`, `docs/architecture-update`.

## Commit and PR Guidelines

* Write commit messages in English.
* Keep the subject line under 72 characters.
* Reference relevant issues or ADRs where applicable.
* One logical change per commit.
* PR descriptions should explain the **why**, not just the **what**.

## What Not to Commit

The following are gitignored and must never be committed:

* **Secrets:** `.env`, API keys, tokens, credentials
* **Personal config:** `bridge/config/system_prompt.md`, `bridge/config/user_constitution.md`
  (use `.example.md` templates instead)
* **Runtime data:** `bridge/data/`, `bridge/logs/`
* **Build artifacts:** `__pycache__/`, `*.pyc`, `dist/`, `build/`, `*.egg-info/`
* **Test artifacts:** `.pytest_cache/`, `htmlcov/`, `.coverage`, `pytest_tmp_*/`
* **Cache files:** `.ruff_cache/`, `.hypothesis/`, `.import_linter_cache/`

## Internationalization (i18n)

AXOLENT AI supports 20 languages via a JSON-based i18n system.

**When adding new user-facing text:**

1. Add the English key to `bridge/i18n/locales/en.json`.
2. Run `python scripts/i18n_sync.py` to propagate to all locale files.
3. Run `python scripts/i18n_bootstrap_hashes.py` to update source hashes.
4. The pre-commit hooks (`i18n_check.py`, `i18n_scan.py`) will verify parity.

**Never hardcode user-facing strings.** Use `t(key, lang)` from the `i18n` package.
The `i18n_scan.py` AST scanner will block commits with hardcoded strings in
`presentation/` and `application/`.

See [docs/I18N.md](docs/I18N.md) for the full i18n system documentation.

## Pre-Commit Hooks

All hooks must pass before a commit is accepted. There are currently 17 hooks:

```bash
# Run all hooks manually
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files
pre-commit run pytest --all-files
```

See [docs/TESTING.md](docs/TESTING.md) for the complete list and what each hook checks.

## Production-Path Tests (mandatory)

Every new feature **must** include a production-path test. This is non-negotiable.

**What counts as a production-path test:**

* Uses real wrapper classes (e.g. `SqliteConnection`, not `FakeConn` or `MagicMock`)
* Imports through the composition root (`main.py` imports), not directly from
  `application/` internals where possible
* Exercises the full call chain from the entry point consumers actually use
* Verifies wiring: the component is instantiated, injected, and callable in the
  production path

**Why:** Mock-only tests can pass while the production path is broken (wiring bugs).
The Codex review found multiple instances where components existed but were never
called in the real code path. Production-path tests catch this class of bugs.

**Example (good):**

```python
# Uses real SqliteConnection, calls through the service interface
conn = SqliteConnection(tmp_path / "test.db")
storage = HypothesisStorage(conn)
storage.init_schema()
service = SkillLearningService(storage, PrivacyPipeline())
result = service.learn("Always respond in German", user_id=1, source="learn_command")
assert result.success
```

**Example (insufficient on its own):**

```python
# Mock-only test: passes even if wiring is broken
storage = MagicMock()
service = SkillLearningService(storage, MagicMock())
# This can't catch real wiring bugs
```

## Architecture Guards

When adding a new service or component that gets wired in `main.py`:

1. Add an AST-based architecture test in `tests/test_architecture/` that verifies
   the component is actually instantiated in `main.py`
2. Add an `inspect.signature` check that validates constructor kwargs match what
   `main.py` passes
3. This prevents "code exists but never runs" bugs

## Pre-Commit Compliance

All 17 pre-commit hooks must pass. No exceptions.

* **Semgrep warnings:** Either fix the finding or explicitly suppress with a
  `# nosemgrep` comment that includes a justification
* **Phase 0 / TODO / FIXME markers:** Do not leave these in committed code.
  Create a GitHub Issue instead and reference it in a comment
* **import-linter:** Zero contract violations. The hexagonal layer boundaries
  are enforced at commit time

## CodeQL (GitHub SAST)

CodeQL runs automatically on every push to `main` and every pull request.

* **Critical/High findings block merge.** Fix them before requesting review.
* **Medium/Low findings** are tracked but do not block merge.
* Findings appear in the PR's "Security" checks and in the repository's
  Security > Code scanning tab.

**Triaging findings:**

1. Click the finding in the PR checks or Security tab
2. Review the data-flow path (source to sink)
3. Fix the vulnerability, or dismiss with a reason if it is a false positive

**Suppressing false positives inline:**

```python
value = cursor.execute(query)  # codeql[py/sql-injection] query is a compile-time constant
```

See [docs/CODEQL.md](docs/CODEQL.md) for full documentation on severity levels,
custom queries, and the relationship to Bandit/Semgrep.

## Design-by-Contract (icontract)

New pipeline-level methods must include `icontract` pre/post-conditions:

* `@icontract.require()` for parameter validation at the boundary
* `@icontract.ensure()` for return value guarantees
* Tests in `tests/test_application/test_contracts/` that verify
  `ViolationError` is raised on boundary violations

## Code Style

* **Language:** English only for all code, comments, docstrings, log messages, and
  documentation. See [CLAUDE.md](CLAUDE.md) for the full language policy.
* **Formatting:** `ruff format` (Black-compatible).
* **Linting:** `ruff` with project-specific rules.
* **Type hints:** Required on all public functions and methods.
* **Bullets:** Use `*` or numbered lists in Markdown. Never `-` as a bullet marker.
* **Comments:** Only when the WHY is non-obvious. Do not explain what the code does.

## Dogfood Day

Once per week, contributors are encouraged to use AXOLENT as a normal
user (not as a tester) for 30 minutes. This finds UX/UI bugs that
automated tests cannot find. See [docs/DOGFOOD.md](docs/DOGFOOD.md)
for the full protocol and reporting format.

## Code of Conduct

Be respectful, constructive, and professional. Technical disagreements are welcome
when backed by reasoning. Personal attacks, harassment, or discrimination are not
tolerated.

## Questions?

Open a [GitHub Issue](https://github.com/axolent-ai/axolent/issues) or start a
[Discussion](https://github.com/axolent-ai/axolent/discussions).
