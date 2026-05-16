# Bridge Service

Backend of AXOLENT AI (Axolent). Telegram bot that spawns Claude Code CLI as a local subprocess (Mode B). Hexagonal Architecture, 1636+ tests, UTF-8 throughout.

## Architecture (Hexagonal)

```
[Telegram User]
      |
      v
[presentation/handlers.py]   Telegram-specific: Commands, Messages, Callbacks
      |
      v
[application/services]        Use-cases: chat_service, bookmark_service
      |                \
      v                 v
[domain/]            [infrastructure/]
  Pure Logic           I/O Adapters
  bookmark.py          claude_cli.py      (Claude Code CLI subprocess)
  language.py          bookmark_storage.py (JSONL legacy backend)
  conversation.py      sqlite_storage.py   (SQLite: BookmarkService, MemoryService)
  personality.py       conversation_storage.py
  markdown.py          audit_log.py       (Audit with rotation)
                       encoding.py        (UTF-8 helper)
                       personality_loader.py
```

**Data flow:** Telegram message arrives -> presentation parses and validates -> application orchestrates the use-case -> domain contains business logic -> infrastructure performs I/O (CLI call, filesystem, logging).

## Directories

| Folder | Contents |
|--------|----------|
| `domain/` | Pure business logic. No I/O imports allowed. |
| `application/` | Use-case orchestration (chat_service, bookmark_service) |
| `infrastructure/` | I/O adapters: Claude CLI, Storage, Audit, Encoding |
| `presentation/` | Telegram handlers, Decorators (Whitelist), Rendering |
| `config/` | system_prompt.md, user_constitution.md |
| `data/` | axolent.db (SQLite), user_profiles.jsonl (runtime data) |
| `logs/` | audit.jsonl (with rotation) |
| `tests/` | 1636+ pytest tests |

## Setup

### Prerequisites

1. Python 3.11+ (3.12 recommended)
2. Claude Code CLI installed and logged in (own Pro/Max subscription)
3. Telegram Bot Token (via @BotFather)

### Installation

```bash
cd bridge
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### Create .env

Create a `.env` file in the `bridge/` folder:

```env
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_here
WHITELIST_USER_IDS=YOUR_TELEGRAM_USER_ID

# Optional (development only!)
ALLOW_ALL_USERS=false
```

## .env Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from @BotFather |
| `WHITELIST_USER_IDS` | Yes* | Comma-separated Telegram User IDs |
| `ALLOW_ALL_USERS` | No | `true` = anyone can use the bot (dev only!) |

*Required when `ALLOW_ALL_USERS` is not set to `true`.

## Start the Bot

```bash
python main.py
```

Expected log output:

```
2026-05-06 10:00:00 [INFO] axolent-bridge: Axolent Bridge starting, Mode B (Claude Code CLI subprocess)
2026-05-06 10:00:00 [INFO] axolent-bridge: Whitelist active: yes
2026-05-06 10:00:00 [INFO] axolent-bridge: Bookmarks feature active (reply-based via /save)
2026-05-06 10:00:00 [INFO] axolent-bridge: Conversation history active (max 20 turns, /reset to clear)
```

The bot now polls Telegram. Every message to the bot is forwarded to Claude Code CLI.

## Telegram Commands

| Command | Description |
|---------|-------------|
| Normal text | Starts Claude query with conversation history |
| `/save` (as reply) | Save or remove bookmark (toggle) |
| `/bookmarks` | List all saved bookmarks |
| `/bookmarks search <term>` | Search bookmarks |
| `/remember <text>` | Save a note (considered in future responses) |
| `/memory` | Show saved notes |
| `/memory search <query>` | Search notes |
| `/forget <id>` | Delete a note |
| `/usage` | Show current usage and profile |
| `/setlimit <profile>` | Switch rate-limit profile (light, normal, power, unlimited) |
| `/setmodel <model>` | Switch AI model (opus, sonnet, haiku or full ID) |
| `/resetmodel` | Reset model to default |
| `/models` | Show current model and available options |
| `/lang <code>` | Set language (de, en, es, fr, etc.) |
| `/reset` or `/new` | Reset conversation and language |
| `/help` | Command overview |
| `/start` | Welcome message |

## Running Tests

```bash
# All tests
python -m pytest

# With verbose output (default via pyproject.toml)
python -m pytest -v

# Single module
python -m pytest tests/test_bookmark.py

# Update snapshots (after UI changes)
python -m pytest --snapshot-update
```

Currently: **1636+ tests**, all passing, runtime ~3 seconds.

## Generate Coverage Report

```bash
# Via script (generates terminal + HTML report)
python scripts/pytest_coverage.py

# Or via pre-commit (manual, not on every commit)
pre-commit run pytest-coverage-report --hook-stage manual

# Or directly
python -m pytest --cov=bridge --cov-config=.coveragerc --cov-report=term-missing --cov-report=html:htmlcov
```

HTML report is then at `bridge/htmlcov/index.html`.
Configuration: `bridge/.coveragerc` (excludes .venv and tests).

## Architecture Rules (non-negotiable)

| Layer | May import from |
|-------|-----------------|
| `domain/` | Nothing (pure, no external deps) |
| `infrastructure/` | `domain/` |
| `application/` | `domain/`, `infrastructure/` |
| `presentation/` | `domain/`, `application/` |
| `main.py` | Everything (Composition Root) |

**Golden Rule:** domain/ NEVER imports from infrastructure/ or presentation/. If you break this rule, the tests break.

## Style Rules

1. Comments and documentation in public-facing files: English
2. Code identifiers (variables, functions, classes): English
3. Umlauts always correct in German text: ä, ö, ü, ß (never ae, oe, ue, ss)
4. No em-dashes in outputs
5. Bullets as dot or numbered, never as hyphen
6. Type hints throughout (all functions, all parameters)
7. Docstrings: what goes in, what comes out, WHY (not WHAT the code does)
8. Encoding: always explicit UTF-8 + errors="replace" + ensure_ascii=False

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `claude: command not found` | Install Claude Code CLI and log in (`claude login`) |
| `WHITELIST_USER_IDS not set` | Set in `.env` or use `ALLOW_ALL_USERS=true` for dev |
| Mojibake in bot output | Set `PYTHONIOENCODING=utf-8` (main.py does this automatically) |
| Bot won't start | Run `pip install -e .` again, test with `python -c "import main"` |
| Tests fail | Is `.venv` active? Run `pip install -e ".[test]"` |
| Bookmark not saved | `/save` must be sent as reply to a bot message |
| Claude not responding | Test CLI: run `claude "test"` directly in terminal |
