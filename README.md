# AXOLENT AI

**Open-Source AI Personal Assistant**

> AIs that argue. Files that never leave. Privacy that holds. Memory that lasts.

![Phase](https://img.shields.io/badge/phase-1%20(active)-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![Python](https://img.shields.io/badge/python-3.11%2B-yellow)
![Tests](https://img.shields.io/badge/tests-1900%2B%20passing-brightgreen)

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

## Why Language Handling Is Different

Most AI assistants promise multi-lingual support and hope the model follows
the instruction. AXOLENT treats language as a system-level constraint, not
a suggestion. Every response is verified, and drift is repaired before
the user sees it.

### The Problem

All major AI assistants (proprietary and open-source alike) handle language
via prompt engineering: they inject "respond in German" into the system prompt
and rely on the model's instruction-following. This works most of the time.
But it breaks silently when the model encounters code, English-language
documentation, long contexts, or context-window compaction. The user receives
a wrong-language response with no warning and no correction.

### AXOLENT's Language Control Plane

AXOLENT implements language handling as an architecturally isolated subsystem
(`bridge/application/language/`) with five distinct components that no other
major AI assistant has at the time of writing:

#### 1. Output Verifier

Every model response is checked against the declared target language after
generation. The verifier uses n-gram-profile-based detection (via a pluggable
backend protocol) to produce a three-level verdict: **PASS**, **WARN**, or
**FAIL**. Code blocks, URLs, and technical terms are stripped before
detection to minimize false positives.

Code: `bridge/application/language/verifier.py`

#### 2. Repair Loop

When the verifier returns FAIL, the system automatically re-queries the
same provider with a reinforced language contract. One repair attempt,
hard-capped. No infinite retry loops, no unbounded token waste. For
outputs above 5000 characters, sample verification replaces full rewrite
to protect latency.

Code: `bridge/application/language/repair_service.py`

#### 3. Stream Guard

For streaming responses, language drift is detected early (between 200-400
characters) instead of streaming an entire wrong-language response to the
user. Uses a very high confidence threshold (0.85) and automatic
self-calibration (disables after 3 consecutive false positives) to
avoid destroying the streaming experience.

Code: `bridge/application/language/stream_guard.py`

#### 4. Immutable Language Context

The target language is decided once per request (via a priority cascade:
explicit override > sticky preference > detected > default) and frozen
into a `LanguageContext` dataclass. This object is immutable (`frozen=True,
slots=True`). No downstream component can mutate it. No mid-pipeline drift.
No context-compaction reset.

Code: `bridge/application/language/context.py`

#### 5. Architecturally Isolated Subsystem

The entire Language Control Plane lives in `bridge/application/language/`
with strict boundaries:

- Detection libraries (langdetect, Lingua, etc.) are confined to a single
  backend module (`backends.py`) behind the `LanguageDetectorBackend` Protocol.
- An architecture test (`tests/test_architecture/test_langdetect_isolation.py`)
  scans the entire codebase on every CI run and fails if detection libraries
  leak outside the backend module.
- Domain-level language detection (`domain/language.py`, calibrated for short
  user inputs) is architecturally separated from output verification
  (calibrated for long LLM outputs). They never cross-import.

This is not a multi-lingual promise. This is a language reliability layer.

For a detailed comparison of how other AI assistants handle language at the
time of writing, see [`docs/COMPETITION_LANGUAGE.md`](docs/COMPETITION_LANGUAGE.md).

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
      language/       Language Control Plane (verifier, repair, stream guard)
    infrastructure/   I/O adapters (CLI, storage, audit)
    presentation/     Telegram-specific handlers & rendering
    config/           System prompt, user constitution
    tests/            1900+ pytest tests
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
3. Write tests (we have 1900+ and counting)
4. Submit a PR with a clear description

Code style: Python with type hints everywhere, Black-formatted, hexagonal architecture rules enforced. See `bridge/README.md` for architecture details.

## Known Limitations (v1.0)

1. **Streaming fallback:** The normal /chat path uses streaming and does NOT have automatic provider failover. If the primary provider returns a rate limit error during streaming, the user sees an error message. Automatic fallback is only active for non-streaming operations (e.g., /debate). Streaming fallback is planned for v1.1.

2. **Memory translation:** When /memory is accessed in a language different from stored entries, content is sent to the LLM provider for translation. This runs via your own Mode B subscription, but memory content does leave the local process. Disable with `AXOLENT_MEMORY_TRANSLATION=false`.

3. **Translations:** Non-English UI strings are LLM-generated and not yet reviewed by native speakers. Community corrections welcome via PR.

## Status

Phase 1 is under active development. The Telegram bridge works, Claude responds, bookmarks and conversation history are functional. Not production-ready yet for public deployment.

Feedback, bug reports, and ideas are welcome via GitHub Issues.
