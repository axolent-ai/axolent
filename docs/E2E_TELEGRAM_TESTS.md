# E2E Telegram Tests

## Overview

End-to-end tests that interact with a **real running AXOLENT bot instance**
through Telegram's API using [tgintegration](https://github.com/JosXa/tgintegration).

These tests verify complete user journeys from message send to response
receive, testing the full stack: Telegram API -> Bot handlers -> LLM ->
Response formatting -> Telegram delivery.

## What They Test

| # | Journey | What's Verified |
|---|---------|-----------------|
| 1 | First-time setup wizard | /start -> welcome -> first question -> response |
| 2 | Long response + /stop cancel | Streaming cancellation works, bot stays responsive |
| 3 | Debate multi-provider | /debate -> synthesis -> follow-up retains context |
| 4 | Memory lifecycle | /remember -> /memory lists it -> recall in conversation |
| 5 | Skill learn + apply | /learn pattern -> trigger -> pattern applied |
| 6 | Language sticky | Swedish messages -> all responses in Swedish |
| 7 | /reset clears history | Context completely wiped after reset |
| 8 | Privacy-filter healthcare | Sensitive health data blocked from memory |
| 9 | Injection detection | Prompt injection via /remember is blocked |
| 10 | Slash-command sanitize | Bot commands in responses don't auto-link |

## Prerequisites

- Python 3.11+
- `pip install tgintegration pyrogram` (or `pip install -e ".[e2e]"`)
- A **test bot** (separate from production!)
- A **test Telegram account** (NOT your personal account)

## Setup Steps

### 1. Create a Test Bot

1. Open Telegram, message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`
3. Name it something like "AXOLENT Test Bot"
4. Username: e.g., `axolent_test_bot`
5. Copy the token -> this is your `TELEGRAM_BOT_TOKEN_TEST`

### 2. Get API Credentials

1. Go to [my.telegram.org](https://my.telegram.org)
2. Log in with your **test account** phone number
3. Go to "API development tools"
4. Create an application (any name/description)
5. Copy `api_id` -> `TELEGRAM_API_ID`
6. Copy `api_hash` -> `TELEGRAM_API_HASH`

### 3. Set Up Test Telegram Account

**Important:** Use a separate Telegram account for testing, NOT your
personal account. You can use a secondary phone number or a VoIP number.

### 4. Generate Session String

Run the interactive helper script:

```bash
python scripts/generate_test_session.py
```

This will:
1. Ask for your API ID and API Hash
2. Ask for the phone number of your test account
3. Send an auth code to that phone via Telegram
4. Generate a session string

Copy the output -> `TELEGRAM_TEST_ACCOUNT_SESSION`

### 5. Configure Environment

Create a `.env.test` file in the project root (this file is gitignored):

```env
TELEGRAM_BOT_TOKEN_TEST=123456:ABC-DEF...
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
TELEGRAM_TEST_ACCOUNT_SESSION=BQC7...longstring...
TELEGRAM_BOT_USERNAME_TEST=axolent_test_bot
```

### 6. Run the Tests

```bash
# From the bridge/ directory:
cd bridge

# Run E2E tests only:
pytest -m e2e_telegram -v

# Run with extra output for debugging:
pytest -m e2e_telegram -v -s --tb=long
```

## CI Behavior

These tests are **automatically skipped** in CI when the environment
variables are not set. No workflow changes are needed.

The test output will show:
```
SKIPPED [10] tests/test_e2e_telegram/conftest.py: E2E Telegram env not configured...
```

## Adding New Journeys

1. Add a new test class in `tests/test_e2e_telegram/test_user_journeys.py`
2. Use the `fresh_chat` fixture to get a clean bot state
3. Follow the pattern:
   - Send message/command via `controller.send_command()` or `controller.client.send_message()`
   - Collect response via `async with controller.collect() as response:`
   - Assert on `response.full_text`, `response.num_messages`, etc.
4. Mark with `@pytest.mark.e2e_telegram`

### Example:

```python
class TestNewFeature:
    async def test_my_new_journey(self, fresh_chat):
        """Description of what this tests."""
        controller = fresh_chat

        async with controller.collect(max_wait=20.0) as response:
            await controller.send_command("mycommand")

        assert "expected" in response.full_text.lower()
```

## Troubleshooting

### "E2E Telegram env not configured"

All 4 required env vars must be set. Check with:
```bash
echo $TELEGRAM_BOT_TOKEN_TEST
echo $TELEGRAM_API_ID
echo $TELEGRAM_API_HASH
echo $TELEGRAM_TEST_ACCOUNT_SESSION
```

### "Bot process exited prematurely"

The test bot couldn't start. Check:
- Is `TELEGRAM_BOT_TOKEN_TEST` valid?
- Is another instance already running with the same token?
- Are all bot dependencies installed?

### Timeout errors

LLM responses can be slow. The default `max_wait` is 30 seconds.
For debate/long responses, use `max_wait=60.0`.

### "FloodWait" errors

Telegram rate-limits API calls. Add delays between tests or use
`global_action_delay` in the BotController configuration.

### Session string expired

Pyrogram session strings can expire. Regenerate with:
```bash
python scripts/generate_test_session.py
```

## Architecture

```
tests/test_e2e_telegram/
  __init__.py          - Package docs
  conftest.py          - Fixtures (bot process, controller, fresh_chat)
  test_user_journeys.py - 10 user-journey test scenarios
  test_real_telegram_flow.py - Legacy skeleton (superseded)
```

The `conftest.py` manages:
- Environment variable checking (auto-skip if not configured)
- Bot subprocess lifecycle (start before tests, kill after)
- Pyrogram client + BotController initialization
- Per-test state reset via `/reset` command
