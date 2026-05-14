# AXOLENT AI

**Open-Source AI Personal Assistant**

> AIs that argue. Files that never leave. Privacy that holds. Memory that lasts.

![Phase](https://img.shields.io/badge/phase-1%20(active)-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![Python](https://img.shields.io/badge/python-3.11%2B-yellow)
![Tests](https://img.shields.io/badge/tests-1190%2B%20passing-brightgreen)

## TL;DR

AXOLENT AI is a local AI personal assistant that runs on YOUR machine, using YOUR own Claude Pro/Max subscription. No cloud middleman, no token hijacking, no SaaS lock-in. Multi-AI debates, local files, true privacy, endless memory. Talk to it via Telegram today, Desktop and Mini App coming soon.

Built for people who want a powerful AI assistant without giving up control over their data or their subscription.

## Vision

1. Open-source AI personal assistant via Telegram + Desktop
2. Runs locally as a subprocess wrapper around Claude Code CLI (Mode B)
3. Multi-provider support planned (Claude, GPT, Gemini, local models)
4. Multi-AI debate: let multiple AIs answer the same question and compare
5. Persistent memory, bookmarks, conversation history, personality system
6. Your keys, your data, your machine

## Mode B: What It Is (and What It Isn't)

**What it is:** A local CLI wrapper that spawns `claude` as a subprocess on your machine. Your existing Claude Pro/Max subscription handles the inference. Axolent is just the interface layer.

**What it is NOT:** Token hijacking, OAuth abuse, or cloud-hosted proxy. There is no server between you and Anthropic. Anthropic explicitly permits local CLI tool usage with your own subscription.

**Architecture:** Telegram message -> Axolent bridge (local) -> Claude Code CLI (local) -> Anthropic API (your subscription)

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | 3.12 recommended |
| Claude Code CLI | Installed and logged in with your own Pro/Max subscription |
| Telegram Bot Token | Create via [@BotFather](https://t.me/BotFather) |
| Git | For cloning |

Optional (Phase 1+):
- Rust + Tauri (for Desktop app)
- Node.js (for Mini App)

## Setup in 5 Minutes

```bash
# 1. Clone
git clone https://github.com/axolent-ai/axolent.git
cd axolent/bridge

# 2. Virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install
pip install -e .

# 4. Configure
cp .env.example .env
# Edit .env with your values:
#   TELEGRAM_BOT_TOKEN=your_token_here
#   WHITELIST_USER_IDS=your_telegram_user_id

# 5. Run
python main.py
```

The bot starts long-polling. Send it a message on Telegram and Claude responds.

## Repository Structure

```
axolent/
  bridge/             Backend service (Hexagonal Architecture, Python)
    domain/           Pure business logic (no I/O imports)
    application/      Use-case orchestration
    infrastructure/   I/O adapters (CLI, storage, audit)
    presentation/     Telegram-specific handlers & rendering
    config/           System prompt, user constitution
    tests/            1190+ pytest tests
    main.py           Entry point
  mini-app/           Telegram Mini App (planned, Phase 1+)
  desktop/            Desktop App via Tauri (planned, Phase 1+)
  shared/             Shared UI components (planned, Phase 1+)
  docs/               Technical documentation
```

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Telegram bridge with Mode B, bookmarks, history, personality | Active |
| **1+** | Multi-provider, persistent memory, Mini App, Desktop App | Planned |
| **2** | User acquisition, marketing, community | Future |
| **3** | App store release, premium tier | Future |

## License: AGPL-3.0

This project uses AGPL-3.0. Why?

The AGPL ensures that if anyone takes this code, hosts it as a service, and offers it to users, they MUST open-source their modifications. This prevents the "Amazon pattern" where cloud providers take open-source projects, wrap them in a service, and give nothing back.

You can run it locally, modify it, fork it. But if you host it for others: share your code.

See [LICENSE](LICENSE) for the full text.

## Contributing

Issues and pull requests are welcome.

1. Fork the repo
2. Create a feature branch
3. Write tests (we have 1190+ and counting)
4. Submit a PR with a clear description

Code style: Python with type hints everywhere, Black-formatted, hexagonal architecture rules enforced. See `bridge/README.md` for architecture details.

## Status

Phase 1 is under active development. The Telegram bridge works, Claude responds, bookmarks and conversation history are functional. Not production-ready yet for public deployment.

Feedback, bug reports, and ideas are welcome via GitHub Issues.
