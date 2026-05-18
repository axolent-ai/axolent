# Development Guide

This guide covers local setup, running the bot, debugging, and common issues.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the Bot](#running-the-bot)
* [Development Workflow](#development-workflow)
* [Debugging Tips](#debugging-tips)
* [Hot Reload](#hot-reload)
* [Common Errors](#common-errors)
* [Environment Variables](#environment-variables)

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | 3.12 recommended |
| Claude Code CLI | Latest | Must be installed and logged in (`claude --version`) |
| Telegram Bot Token | N/A | Create via [@BotFather](https://t.me/BotFather) |
| Git | Latest | For cloning and contributing |

**Claude CLI check:**

```bash
claude --version
```

If this fails, install the Claude Code CLI and log in with your Pro/Max
subscription first.

## Installation

```bash
# Clone
git clone https://github.com/axolent-ai/axolent.git
cd axolent/bridge

# Virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows cmd)
.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source .venv/bin/activate

# Install with all development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

**Verify installation:**

```bash
python -c "import domain; import application; import infrastructure; import presentation; print('All layers importable')"
```

## Configuration

```bash
# Copy the environment template
cp .env.example .env
```

Edit `.env` with your values:

```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
WHITELIST_USER_IDS=your_telegram_user_id

# Optional: dev mode (required for ALLOW_ALL_USERS)
# AXOLENT_DEV_MODE=true
# ALLOW_ALL_USERS=true
```

**Getting your Telegram user ID:** Message [@userinfobot](https://t.me/userinfobot)
on Telegram.

**Personal configuration files** (optional):

* `bridge/config/system_prompt.md`: Custom system prompt (gitignored)
* `bridge/config/user_constitution.md`: Custom user constitution (gitignored)
* Templates: `bridge/config/system_prompt.example.md`, `bridge/config/user_constitution.example.md`

## Running the Bot

```bash
cd bridge
python main.py
```

The bot starts in long-polling mode. Send a message to your bot on Telegram
and Claude will respond.

**Expected startup output:**

```
INFO axolent: AXOLENT AI starting...
INFO axolent: Whitelist: [your_user_id]
INFO axolent: Providers registered: claude_persistent, ollama_local, ...
INFO axolent: Long-polling started.
```

## Development Workflow

1. **Make changes** in the appropriate layer (see [ARCHITECTURE.md](ARCHITECTURE.md))
2. **Run tests** for the affected layer:
   ```bash
   pytest tests/test_domain/         # Pure logic tests
   pytest tests/test_application/    # Service orchestration tests
   pytest tests/test_presentation/   # Handler tests
   ```
3. **Run all pre-commit hooks:**
   ```bash
   pre-commit run --all-files
   ```
4. **Commit** (pre-commit hooks run automatically)

## Debugging Tips

### Log Levels

Set the log level in your environment or at the top of `main.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

Useful loggers:

* `axolent`: Main bot logger
* `application.chat_service`: LLM call orchestration
* `application.language_resolver`: Language detection decisions
* `infrastructure.claude_process_pool`: Subprocess lifecycle
* `application.streaming_handler`: Token streaming and throttling

### Audit Log

Every LLM call and command is logged to `bridge/logs/audit.jsonl`.
Each entry includes a `request_id` for end-to-end correlation:

```bash
# View recent audit entries (PowerShell)
Get-Content bridge/logs/audit.jsonl -Tail 10

# Search by request_id (PowerShell)
Select-String -Path bridge/logs/audit.jsonl -Pattern "your-request-id"

# View recent audit entries (bash)
tail -n 10 bridge/logs/audit.jsonl
```

### Interactive Testing

For testing specific services without Telegram:

```python
import asyncio
from application.language_resolver import LanguageResolver

resolver = LanguageResolver(conv_storage=None)
ctx = asyncio.run(resolver.resolve(user_id=123, chat_id=456, text="Hello world"))
print(ctx)  # LanguageContext(code='en', source='detected', confidence=0.95, ...)
```

### Import Linter

Verify layer contracts at any time:

```bash
python scripts/run_with_venv.py lint-imports
```

## Hot Reload

AXOLENT AI does not have built-in hot reload for the Telegram bot. The
recommended development pattern:

1. Stop the bot (Ctrl+C)
2. Make changes
3. Run relevant tests
4. Restart: `python main.py`

For rapid iteration on non-Telegram logic (domain, application), use
pytest or interactive Python instead of running the full bot.

## Common Errors

### `ALLOW_ALL_USERS is active but AXOLENT_DEV_MODE is not set`

The bot refuses to start because `ALLOW_ALL_USERS=true` is set without the
companion `AXOLENT_DEV_MODE=true`. This is a safety tripwire.

**Fix:** Either remove `ALLOW_ALL_USERS` from `.env` (recommended for
production), or add `AXOLENT_DEV_MODE=true` (development only).

### `ModuleNotFoundError: No module named 'domain'`

You are running from the wrong directory. The bridge expects to be run from
within `bridge/`:

```bash
cd bridge
python main.py
```

### `telegram.error.InvalidToken`

Your `TELEGRAM_BOT_TOKEN` in `.env` is invalid or missing. Get a new token
from [@BotFather](https://t.me/BotFather).

### `Claude subprocess failed` or `claude: command not found`

The Claude Code CLI is not installed or not on your PATH. Install it and
verify with `claude --version`.

### Pre-commit hook failures

```bash
# See which hook failed
pre-commit run --all-files

# Run a specific hook with verbose output
pre-commit run ruff --all-files --verbose
pre-commit run pytest --all-files --verbose
```

### `UnicodeDecodeError` or mojibake in output

All files must be UTF-8 encoded. The encoding helper at
`bridge/infrastructure/encoding.py` provides safe defaults with
`errors='replace'` and `ensure_ascii=False`. If you see garbled characters,
check that your editor and terminal use UTF-8.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | N/A | Telegram Bot API token |
| `WHITELIST_USER_IDS` | Yes | N/A | Comma-separated Telegram user IDs |
| `ALLOW_ALL_USERS` | No | `false` | Allow any Telegram user (requires DEV_MODE) |
| `AXOLENT_DEV_MODE` | No | `false` | Enable development mode |
| `CLAUDE_SUBPROCESS_TTL_SECONDS` | No | `3600` | Subprocess timeout (seconds) |
| `CLAUDE_POOL_MODEL` | No | `claude-sonnet-4-6` | Model for Claude process pool |
| `AXOLENT_STREAMING_MODE` | No | `telegram` | Streaming mode (`telegram` or `local`) |
| `DEBATE_PROVIDERS` | No | All available | Comma-separated providers for /debate |
| `AXOLENT_OLLAMA_AUTOSTART` | No | `true` | Auto-start Ollama at bot startup |
| `AXOLENT_MEMORY_TRANSLATION` | No | `true` | Enable memory translation |
| `AXOLENT_FALLBACK_CHAIN_*` | No | N/A | Per-slot fallback provider chains |
| `AXOLENT_FALLBACK_TIMEOUT_SECONDS` | No | `30` | Fallback resolver timeout |

## Related Documents

* [ARCHITECTURE.md](ARCHITECTURE.md): System overview and layer rules
* [TESTING.md](TESTING.md): Test conventions and pre-commit hooks
* [CONTRIBUTING.md](../CONTRIBUTING.md): Contribution guidelines
* [CLAUDE.md](../CLAUDE.md): Coding conventions
