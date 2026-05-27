---
title: AXOLENT
description: Local-first, privacy-preserving AI assistant
---

# AXOLENT

**Local-first, privacy-preserving AI assistant.**

> AIs that argue. Files that never leave. Privacy that holds. Memory that lasts.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.11%2B-yellow)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-4354%2B%20passing-brightgreen)](https://github.com/axolent-ai/axolent)

## What is AXOLENT?

AXOLENT is a Telegram-bot wrapper that runs locally on your device, routing queries across multiple LLM providers (Claude, OpenAI, Gemini, Mistral, Ollama, Groq) using your own subscriptions. Memory, conversation history, and pseudonym mappings stay on your device. No central AXOLENT cloud account needed.

## Key Features

- **Mode B Architecture**: Bring your own LLM subscriptions, no vendor lock-in
- **Privacy by Design**: All personal data stays local, no cloud upload
- **Multi-Provider Routing**: Claude, OpenAI, Gemini, Mistral, Ollama, Groq
- **PII Control Plane** (in development): structured pseudonymization before LLM calls
- **4000+ Tests**: comprehensive test coverage with security gates
- **AGPL-3.0**: open-source, auditable, reproducible
- **20 Languages**: i18n framework with full DE/EN support

## Documentation

- [Architecture Overview](./ARCHITECTURE.md)
- [Features](./FEATURES.md)
- [Development Guide](./DEVELOPMENT.md)
- [Internationalization (i18n)](./I18N.md)
- [Privacy & PII](./GOLDEN_CORPUS.md)
- [CI/CD Security](./CICD_SECURITY.md)
- [Dependabot Setup](./DEPENDABOT.md)
- [License Compliance](./LICENSE_COMPLIANCE.md)
- [Defensive Publication](./DEFENSIVE_PUBLICATION.md)
- [E2E Telegram Tests](./E2E_TELEGRAM_TESTS.md)

## What's Next: PII Control Plane

The next development phase builds a local PII Control Plane that automatically detects and pseudonymizes personal data (names, addresses, API keys, financial identifiers) before any LLM call. The pseudonym mapping stays local in an encrypted vault.

A funding application for this work has been submitted to NLnet (NGI Zero Commons Fund). Status: under review.

## Get Started

Visit the [GitHub repository](https://github.com/axolent-ai/axolent) for installation instructions, source code, and issue tracking.

## License

AGPL-3.0. See [LICENSE](https://github.com/axolent-ai/axolent/blob/main/LICENSE) for full text.

The Core (this repository) is AGPL-3.0. An advanced skill system is developed as a separate private project with its own license.

---

Made with care for privacy. 2026.
