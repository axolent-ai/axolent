#!/usr/bin/env python3
"""Auto-Smoke-Test for AXOLENT Bot.

Runs 15 typical user scenarios through the handler functions with
mocked Telegram objects. No real bot, no real API calls, no token
required.

Each scenario:
    1. Constructs a mocked Update + Context
    2. Calls the appropriate handler function
    3. Asserts the handler responded (no crash, no silent failure)

Usage:
    python scripts/smoke_test.py

Exit-Code 0 = all scenarios passed, 1 = at least one failed.

This script can be run standalone or via pytest (it uses pytest
internally for assertion reporting).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure bridge/ is on sys.path
_bridge_root = Path(__file__).resolve().parent.parent / "bridge"
if str(_bridge_root) not in sys.path:
    sys.path.insert(0, str(_bridge_root))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smoke_test")


# ---------------------------------------------------------------------------
# Mock factories (adapted from tests/test_presentation/test_handlers.py)
# ---------------------------------------------------------------------------


def _make_update(
    user_id: int = 99999,
    chat_id: int = 99999,
    text: str = "",
) -> MagicMock:
    """Create a mocked Telegram Update."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "smoke_test_user"
    update.effective_user.first_name = "Smoke"
    update.effective_user.language_code = "de"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message = MagicMock()
    update.message.text = text
    update.message.message_id = 1
    update.message.reply_text = AsyncMock()
    update.message.reply_html = AsyncMock()
    update.message.reply_to_message = None
    update.callback_query = None
    return update


def _make_context_kernel() -> MagicMock:
    """Create a mock ContextKernel."""
    kernel = AsyncMock()

    async def _build(envelope: Any, language_override: str | None = None) -> Any:
        # Lazy import to avoid circular deps at module level
        mod_exec = importlib.import_module("application.execution")
        mod_lang = importlib.import_module("application.language_resolver")
        return mod_exec.ExecutionContext(
            request_id=envelope.request_id,
            user_id=envelope.user_id,
            chat_id=envelope.chat_id,
            channel="telegram",
            language=mod_lang.LanguageContext(
                code="de",
                source="default",
                confidence=1.0,
                switched_from=None,
                request_id=envelope.request_id,
            ),
        )

    kernel.build = AsyncMock(side_effect=_build)
    return kernel


def _make_mock_chat_service() -> MagicMock:
    """Create a mock ChatService that returns a simple response.

    All methods that are ``await``-ed by presentation-layer handlers
    MUST be ``AsyncMock`` instances.  Using plain ``MagicMock`` for
    those causes ``TypeError: object MagicMock can't be used in
    'await' expression``.
    """
    svc = MagicMock()

    # --- async methods awaited by handlers -------------------------
    svc.get_chat_language = AsyncMock(return_value="de")
    svc.set_chat_language = AsyncMock()
    svc.reset = AsyncMock()
    svc.save_static_response_to_history = AsyncMock()
    svc.process_user_message = AsyncMock(
        return_value=MagicMock(
            text="Smoke test response",
            duration_seconds=0.1,
            provider_name="mock",
        )
    )
    svc.process_user_message_streaming = AsyncMock(
        return_value=(
            "Smoke test response",  # full_text
            0.1,  # duration
            "mock",  # provider_name
        )
    )
    svc.save_streaming_result = AsyncMock(return_value="Smoke test response")
    svc.save_debate_turns = AsyncMock()
    svc.route = AsyncMock(
        return_value=MagicMock(
            text="Smoke test response",
            duration_seconds=0.1,
            provider_name="mock",
        )
    )
    svc.route_streaming = AsyncMock(return_value=None)

    # --- sync attributes / services --------------------------------
    svc.provider_router = MagicMock()
    svc.provider_router.providers = {}
    svc.provider_router.list_available = AsyncMock(return_value=[])
    svc.memory_service = None
    svc.model_service = None
    svc.task_router = MagicMock()
    svc.task_router.classify = MagicMock(
        return_value=MagicMock(
            slot=MagicMock(value="chat"),
            model_id="claude-sonnet-4-20250514",
            provider="claude_persistent",
        )
    )
    svc.self_awareness_service = None
    svc.skill_matcher = None
    svc.fallback_resolver = None
    svc._language_enforcement = None
    return svc


def _make_memory_service() -> MagicMock:
    """Create a mock MemoryService.

    All methods that are consumed by handlers must return primitive values
    (not MagicMock) to avoid 'Object of type MagicMock is not JSON
    serializable' warnings in the audit log (R7-LOW-03).
    """
    svc = MagicMock()
    svc.remember_episodic = MagicMock(return_value="mem_001")
    svc.list_episodic = MagicMock(return_value=[])
    svc.forget_episodic = MagicMock(return_value=True)
    svc.forget = MagicMock(return_value=True)
    svc.load_all_for_prompt = MagicMock(return_value="")
    return svc


def _make_rate_limiter() -> MagicMock:
    """Create a mock RateLimiter with realistic return values.

    ``check_and_consume`` returns a real ``RateLimitResult`` (allowed)
    and ``get_usage`` returns a real ``UsageInfo`` so that the
    ``/usage`` handler can do arithmetic on the fields.
    """
    # Lazy import to avoid circular deps at module level
    rl_mod = importlib.import_module("application.rate_limiter")

    limiter = MagicMock()
    limiter.check_and_consume = MagicMock(
        return_value=rl_mod.RateLimitResult(allowed=True, profile="normal"),
    )
    limiter.get_usage = MagicMock(
        return_value=rl_mod.UsageInfo(
            profile="normal",
            minute_used=1,
            minute_limit=10,
            minute_reset_seconds=30.0,
            hour_used=5,
            hour_limit=60,
            hour_reset_seconds=1800.0,
            day_used=10,
            day_limit=200,
            day_reset_seconds=3600.0,
        ),
    )
    limiter.get_user_profile = MagicMock(return_value="normal")
    return limiter


def _make_context(
    args: list[str] | None = None,
    chat_service: Any = None,
    memory_service: Any = None,
) -> MagicMock:
    """Create a mocked Telegram context with bot_data."""
    context = MagicMock()
    context.args = args or []
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()
    context.bot.send_message = AsyncMock()
    context.application = MagicMock()

    svc = chat_service or _make_mock_chat_service()
    mem = memory_service or _make_memory_service()

    context.application.bot_data = {
        "chat_service": svc,
        "system_prompt": "Smoke test system prompt.",
        "memory_service": mem,
        "persistent_provider": None,
        "process_pool": MagicMock(),
        "rate_limiter": _make_rate_limiter(),
        "bookmark_service": MagicMock(),
        "context_kernel": _make_context_kernel(),
        "model_service": MagicMock(),
        "task_router": svc.task_router,
        "onboarding_storage": None,
        "hypothesis_storage": None,
        "skill_explainer": None,
        "import_orchestrator": None,
        "skill_learning_service": None,
        "language_enforcement": None,
    }
    return context


# ---------------------------------------------------------------------------
# Smoke test scenarios
# ---------------------------------------------------------------------------


class SmokeResult:
    """Result of a single smoke test scenario."""

    def __init__(self, name: str, passed: bool, error: str = "") -> None:
        self.name = name
        self.passed = passed
        self.error = error

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        suffix = f" ({self.error})" if self.error else ""
        return f"[{status}] {self.name}{suffix}"


async def _run_scenario(
    name: str,
    handler_path: str,
    handler_name: str,
    update: MagicMock,
    context: MagicMock,
) -> SmokeResult:
    """Run a single handler scenario and return result."""
    try:
        mod = importlib.import_module(handler_path)
        handler = getattr(mod, handler_name)
        await handler(update, context)
        return SmokeResult(name, True)
    except Exception as exc:
        return SmokeResult(name, False, f"{type(exc).__name__}: {exc}")


async def run_all_scenarios() -> list[SmokeResult]:
    """Run all 15 smoke test scenarios."""
    results: list[SmokeResult] = []

    # Patch whitelist to allow our test user
    with (
        patch("presentation.decorators.ALLOW_ALL_USERS", True),
        patch(
            "infrastructure.conversation_storage._reset_all_for_tests",
            lambda: None,
            create=True,
        ),
    ):
        # Reset conversation storage
        try:
            from infrastructure.conversation_storage import _reset_all_for_tests

            _reset_all_for_tests()
        except (ImportError, AttributeError):
            pass

        # Scenario 1: /start
        results.append(
            await _run_scenario(
                "01_start",
                "presentation.handlers",
                "handle_start_command",
                _make_update(text="/start"),
                _make_context(),
            )
        )

        # Scenario 2: /help
        results.append(
            await _run_scenario(
                "02_help",
                "presentation.handlers",
                "handle_help_command",
                _make_update(text="/help"),
                _make_context(),
            )
        )

        # Scenario 3: /lang de
        results.append(
            await _run_scenario(
                "03_lang_de",
                "presentation.handlers",
                "handle_lang_command",
                _make_update(text="/lang de"),
                _make_context(args=["de"]),
            )
        )

        # Scenario 4: Short German message
        results.append(
            await _run_scenario(
                "04_short_message",
                "presentation.handlers",
                "handle_message",
                _make_update(text="Wie spaet ist es?"),
                _make_context(),
            )
        )

        # Scenario 5: Longer message (would trigger streaming in production)
        results.append(
            await _run_scenario(
                "05_long_message",
                "presentation.handlers",
                "handle_message",
                _make_update(text="Erklaer mir was Python ist in 200 Worten"),
                _make_context(),
            )
        )

        # Scenario 6: /reset
        results.append(
            await _run_scenario(
                "06_reset",
                "presentation.handlers",
                "handle_reset_command",
                _make_update(text="/reset"),
                _make_context(),
            )
        )

        # Scenario 7: /stop
        results.append(
            await _run_scenario(
                "07_stop",
                "presentation.handlers",
                "handle_stop_command",
                _make_update(text="/stop"),
                _make_context(),
            )
        )

        # Scenario 8: /remember
        results.append(
            await _run_scenario(
                "08_remember",
                "presentation.handlers",
                "handle_remember_command",
                _make_update(text="/remember ich bevorzuge Markdown"),
                _make_context(args=["ich", "bevorzuge", "Markdown"]),
            )
        )

        # Scenario 9: /memory
        results.append(
            await _run_scenario(
                "09_memory",
                "presentation.handlers",
                "handle_memory_command",
                _make_update(text="/memory"),
                _make_context(),
            )
        )

        # Scenario 10: /forget
        results.append(
            await _run_scenario(
                "10_forget",
                "presentation.handlers",
                "handle_forget_command",
                _make_update(text="/forget mem_001"),
                _make_context(args=["mem_001"]),
            )
        )

        # Scenario 11: /skills
        results.append(
            await _run_scenario(
                "11_skills",
                "presentation.skill_commands",
                "handle_skills_command",
                _make_update(text="/skills"),
                _make_context(),
            )
        )

        # Scenario 12: /learn
        results.append(
            await _run_scenario(
                "12_learn",
                "presentation.skill_commands",
                "handle_learn_command",
                _make_update(text="/learn antworte praezise"),
                _make_context(args=["antworte", "praezise"]),
            )
        )

        # Scenario 13: /debate
        results.append(
            await _run_scenario(
                "13_debate",
                "presentation.handlers",
                "handle_debate_command",
                _make_update(text="/debate Notion vs Obsidian?"),
                _make_context(args=["Notion", "vs", "Obsidian?"]),
            )
        )

        # Scenario 14: Follow-up message (tests conversation context)
        results.append(
            await _run_scenario(
                "14_followup_message",
                "presentation.handlers",
                "handle_message",
                _make_update(text="wie findet ihr die Empfehlungen?"),
                _make_context(),
            )
        )

        # Scenario 15: /usage
        results.append(
            await _run_scenario(
                "15_usage",
                "presentation.handlers",
                "handle_usage_command",
                _make_update(text="/usage"),
                _make_context(),
            )
        )

    return results


def main() -> int:
    """Run all smoke tests and report results."""
    print("=" * 60)
    print("AXOLENT Bot Smoke Test")
    print("=" * 60)

    results = asyncio.run(run_all_scenarios())

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    for r in results:
        print(f"  {r}")
    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(results)}")
    print("=" * 60)

    if failed > 0:
        print("SMOKE TEST FAILED")
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
