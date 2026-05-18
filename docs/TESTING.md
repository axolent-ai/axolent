# Testing Guide

AXOLENT AI has 1900+ tests organized by architectural layer. This guide
covers running tests, writing new tests, and the pre-commit hook system.

## Table of Contents

* [Running Tests](#running-tests)
* [Test Structure](#test-structure)
* [Test Conventions](#test-conventions)
* [Pytest Markers](#pytest-markers)
* [Writing New Tests](#writing-new-tests)
* [Pre-Commit Hooks](#pre-commit-hooks)
* [Coverage](#coverage)

## Running Tests

All commands assume you are in the `bridge/` directory with the virtual
environment activated.

### Full Suite

```bash
pytest
```

### By Layer

```bash
pytest tests/test_domain/           # Pure business logic
pytest tests/test_application/      # Service orchestration
pytest tests/test_presentation/     # Telegram handlers
pytest tests/test_infrastructure/   # Storage, CLI, providers
pytest tests/test_e2e/              # End-to-end tests
pytest tests/test_scripts/          # Script tests
```

### Single File or Test

```bash
pytest tests/test_domain/test_language.py
pytest tests/test_application/test_chat_service.py::TestChatService::test_streaming
```

### Quick Run (CI-style)

```bash
pytest -q --no-header
```

### By Marker (when available)

```bash
pytest -m unit             # Fast, isolated unit tests
pytest -m integration      # Tests with I/O or subprocess
pytest -m i18n             # Internationalization tests
pytest -m streaming        # Streaming and throttle tests
pytest -m security         # Security-related tests
```

**Note:** Pytest markers are being introduced incrementally. Not all tests
are marked yet. Running without `-m` always executes the full suite.

## Test Structure

```
bridge/tests/
    conftest.py                 Shared fixtures
    test_main.py                Entry-point tests
    test_domain/                Pure domain logic
        test_language.py
        test_bookmark.py
        test_conversation.py
        test_personality.py
        test_markdown.py
        ...
    test_application/           Service orchestration
        test_chat_service.py
        test_debate_orchestrator.py
        test_language_resolver.py
        test_streaming_handler.py
        test_rate_limiter.py
        test_execution/
            test_kernel.py
            test_instruction_compiler.py
            test_context.py
            ...
        ...
    test_presentation/          Handler tests
        test_handlers.py
        test_render.py
        test_decorators.py
        ...
    test_infrastructure/        Storage and provider tests
        test_sqlite_storage.py
        test_audit_log.py
        ...
    test_e2e/                   End-to-end tests
        ...
    test_scripts/               Pre-commit script tests
        ...
```

## Test Conventions

### Naming

* Test files: `test_<module>.py`
* Test classes: `TestSomething` (PascalCase)
* Test methods: `test_does_something_when_condition` (snake_case)
* Test docstrings: English, describe the WHY

### AAA Pattern

Every test follows **Arrange, Act, Assert**:

```python
def test_detects_german_from_marker_words():
    """German marker words should produce 'de' with high confidence."""
    # Arrange
    text = "Ich habe eine Frage zu diesem Thema"

    # Act
    result = detect_language_with_confidence(text)

    # Assert
    assert result.language == "de"
    assert result.confidence > 0.7
```

### Mocking

* Use `pytest-mock` (`mocker` fixture) for all mocking.
* Mock at the boundary: mock infrastructure when testing application,
  mock application when testing presentation.
* Never mock domain logic. Domain tests must be pure.

```python
def test_chat_service_calls_provider(mocker):
    """ChatService should route to the selected provider."""
    # Arrange
    mock_router = mocker.MagicMock()
    mock_router.query.return_value = ProviderResponse(text="Hello")
    service = ChatService(provider_router=mock_router, ...)

    # Act
    result = await service.process_message(...)

    # Assert
    mock_router.query.assert_called_once()
```

### Async Tests

`pytest-asyncio` is configured with `asyncio_mode = "auto"`. Async test
functions are detected and run automatically:

```python
async def test_kernel_builds_context():
    """ContextKernel should produce a frozen ExecutionContext."""
    kernel = ContextKernel.create_default()
    ctx = await kernel.build(envelope)
    assert ctx.language.code == "en"
```

### Fixtures

Shared fixtures live in `tests/conftest.py`. Layer-specific fixtures can be
placed in `tests/test_<layer>/conftest.py`.

### Snapshot Testing

`syrupy` is available for snapshot testing of complex outputs (e.g., rendered
Telegram HTML). Snapshots are stored in `__snapshots__/` directories.

### Property-Based Testing

`hypothesis` is available for property-based testing, particularly useful for
domain logic (language detection edge cases, markdown conversion).

## Pytest Markers

The following markers are being introduced to allow targeted test execution:

| Marker | Description | Example |
|--------|-------------|---------|
| `@pytest.mark.unit` | Fast, isolated, no I/O | Domain logic, pure functions |
| `@pytest.mark.integration` | Involves I/O or subprocess | SQLite, Claude process pool |
| `@pytest.mark.i18n` | i18n key parity, translation | Locale file validation |
| `@pytest.mark.streaming` | Streaming and throttle logic | StreamingSession, burst mode |
| `@pytest.mark.security` | Security-related | Whitelist, leakage guard, rate limiting |

Register markers in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "unit: Fast isolated tests without I/O",
    "integration: Tests involving I/O or subprocess",
    "i18n: Internationalization tests",
    "streaming: Streaming and throttle tests",
    "security: Security-related tests",
]
```

## Writing New Tests

### For Domain Layer

Domain tests must be pure: no mocking, no I/O, no async (unless the domain
function is async). Test inputs and expected outputs only.

```python
# tests/test_domain/test_language.py
class TestLanguageDetection:
    def test_detects_arabic_script(self):
        """Arabic Unicode characters should be detected deterministically."""
        result = detect_language_with_confidence("مرحبا بالعالم")
        assert result.language == "ar"
        assert result.confidence == 1.0
```

### For Application Layer

Mock infrastructure dependencies. Test orchestration logic.

```python
# tests/test_application/test_memory_service.py
class TestMemoryService:
    async def test_loads_relevant_entries(self, mocker):
        """MemoryService should load and filter entries by user_id."""
        mock_storage = mocker.MagicMock()
        mock_storage.search.return_value = [...]
        service = MemoryService(storage=mock_storage)

        result = await service.get_relevant(user_id=42, query="test")

        mock_storage.search.assert_called_once_with(user_id=42, query="test")
```

### For Presentation Layer

Mock application services. Test that handlers parse input correctly and call
the right service methods.

```python
# tests/test_presentation/test_handlers.py
class TestHandleMessage:
    async def test_rate_limited_user_gets_error(self, mocker):
        """Rate-limited users should see a localized error message."""
        mock_limiter = mocker.MagicMock()
        mock_limiter.check.return_value = RateLimitResult(allowed=False, ...)
        ...
```

### For Infrastructure Layer

These tests may involve real I/O (in-memory SQLite, temp files).
Use `tmp_path` fixture for file-based tests.

```python
# tests/test_infrastructure/test_sqlite_storage.py
class TestSqliteBookmarkStorage:
    def test_save_and_retrieve(self, tmp_path):
        """Bookmarks should survive save/load cycle."""
        db_path = tmp_path / "test.db"
        storage = SqliteBookmarkStorage(str(db_path))
        storage.save(user_id=1, chat_id=1, text="test bookmark")
        result = storage.get_all(user_id=1, chat_id=1)
        assert len(result) == 1
```

## Pre-Commit Hooks

All 17 hooks must pass before a commit is accepted. They run automatically
on `git commit` after `pre-commit install`.

### Standard Hooks (from pre-commit/pre-commit-hooks)

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with a newline |
| `check-yaml` | Validate YAML syntax |
| `check-added-large-files` | Block files >500KB |
| `check-merge-conflict` | Detect unresolved merge conflicts |
| `mixed-line-ending` | Enforce consistent line endings |

### Code Quality Hooks

| Hook | Purpose |
|------|---------|
| `ruff` | Python linting with auto-fix |
| `ruff-format` | Python formatting (Black-compatible) |

### Security Hooks

| Hook | Purpose |
|------|---------|
| `bandit` | Python SAST (security anti-patterns) |
| `pip-audit` | Dependency vulnerability scanner |
| `semgrep` | Semantic SAST (2000+ rules) |

**Note:** `gitleaks` is configured (`.gitleaks.toml`) but disabled as a
pre-commit hook on Windows due to an upstream wasm bug. Run manually:

```bash
gitleaks detect --source . --config .gitleaks.toml
```

### Architecture Hooks

| Hook | Purpose |
|------|---------|
| `import-linter` | Enforce hexagonal layer contracts (3 contracts) |

### Test Hooks

| Hook | Purpose |
|------|---------|
| `pytest` | Run full test suite |
| `pytest-coverage-report` | Generate coverage report (manual stage only) |

### i18n Hooks

| Hook | Purpose |
|------|---------|
| `i18n-check` | Validate locale file key parity and source hash integrity |
| `i18n-scan` | AST scanner for hardcoded strings in presentation/application |

### Language Policy Hooks

| Hook | Purpose |
|------|---------|
| `en-only-production` | Block new German strings in production code |
| `no-fake-umlauts` | Prevent ASCII umlaut substitutions (ae/oe/ue/ss) in remaining German text |

### Running Hooks Manually

```bash
# All hooks
pre-commit run --all-files

# Specific hook
pre-commit run ruff --all-files
pre-commit run import-linter --all-files
pre-commit run i18n-check --all-files

# Only on staged files
pre-commit run
```

## Coverage

Generate a coverage report (manual stage, not on every commit):

```bash
pre-commit run pytest-coverage-report --hook-stage manual

# Or directly:
python scripts/pytest_coverage.py
```

Coverage HTML report is generated to `bridge/htmlcov/` (gitignored).

## Related Documents

* [ARCHITECTURE.md](ARCHITECTURE.md): Layer rules that inform test boundaries
* [CONTRIBUTING.md](../CONTRIBUTING.md): Contribution guidelines
* [DEVELOPMENT.md](DEVELOPMENT.md): Local setup and debugging
* [CLAUDE.md](../CLAUDE.md): Test naming conventions
