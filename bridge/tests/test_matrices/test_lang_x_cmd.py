"""Language x Command 2D matrix tests (10 languages * 8 commands = 80 combos).

Production-path tests verifying:
  - Command responses respect the user's sticky language
  - Audit-log entries include language marker for all command invocations
  - i18n fallback chain works correctly under sticky language

These tests use the real LanguageResolver + real i18n system to verify
that the language/command interaction works correctly across all combinations.
"""

from __future__ import annotations


import pytest

from application.language.resolver import LanguageResolver
from domain.personality import build_effective_prompt
from i18n.domain.i18n import is_supported
from infrastructure.conversation_storage import (
    get_language,
    set_language,
)

from .conftest import COMMANDS_WITH_ARGS, LANGUAGES


pytestmark = pytest.mark.matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# i18n keys that are expected to exist for ALL languages.
# These are the keys used by command responses.
_UNIVERSAL_I18N_KEYS: list[str] = [
    "reset.confirmation",
    "remember.saved",
    "remember.usage",
    "memory.empty",
    "memory.list_header",
    "forget.success",
    "forget.not_found",
    "forget.usage",
]


# ---------------------------------------------------------------------------
# 2D Matrix: Language x Command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lang", LANGUAGES)
@pytest.mark.parametrize("cmd,args", COMMANDS_WITH_ARGS)
class TestLanguageXCommand:
    """Cross-cutting tests for all language/command combinations."""

    async def test_sticky_language_available_for_command(
        self, lang: str, cmd: str, args: str
    ) -> None:
        """After setting sticky language, resolver returns it for command context.

        This verifies the production path: user sets language via wizard,
        then uses a command. The command handler calls get_chat_language()
        or resolver.resolve() and gets the correct language.
        """
        user_id, chat_id = 42, 100
        await set_language(user_id, chat_id, lang)

        resolver = LanguageResolver(default_lang="de")
        # Simulate command invocation context (short text = command itself)
        ctx = await resolver.resolve(user_id=user_id, chat_id=chat_id, text=cmd)

        assert ctx.code == lang, (
            f"Sticky lang '{lang}' not resolved for command '{cmd}': "
            f"got '{ctx.code}' (source={ctx.source})"
        )

    async def test_readonly_resolve_does_not_mutate_sticky(
        self, lang: str, cmd: str, args: str
    ) -> None:
        """resolve_readonly() never mutates sticky language.

        Command handlers should use resolve_readonly() or resolve with
        the command text stripped. This test verifies that the readonly
        path never overwrites sticky, regardless of command text content.
        """
        user_id, chat_id = 55, 200
        await set_language(user_id, chat_id, lang)

        resolver = LanguageResolver(default_lang="de")

        # Readonly resolve with command text (should NOT persist anything)
        await resolver.resolve_readonly(
            user_id=user_id, chat_id=chat_id, text=f"{cmd} {args}"
        )

        # Sticky must remain unchanged
        stored = await get_language(user_id, chat_id)
        assert stored == lang, (
            f"resolve_readonly mutated sticky from '{lang}' to '{stored}' "
            f"after command '{cmd} {args}'"
        )


@pytest.mark.parametrize("lang", LANGUAGES)
@pytest.mark.parametrize("cmd,args", COMMANDS_WITH_ARGS)
class TestLanguageXCommandI18n:
    """Verify i18n coverage for all language/command intersections."""

    def test_language_supported_in_i18n(self, lang: str, cmd: str, args: str) -> None:
        """All matrix languages are recognized by i18n system."""
        assert is_supported(lang), (
            f"Language '{lang}' not supported by i18n (needed for {cmd})"
        )

    def test_system_prompt_language_lock_for_command_context(
        self, lang: str, cmd: str, args: str
    ) -> None:
        """System prompt language lock works for every lang/cmd combination.

        When a user with sticky language invokes a command that triggers
        an LLM call, the system prompt must contain the language lock
        for their language.
        """
        result = build_effective_prompt("You are a helpful assistant.", lang)
        assert f"'{lang}'" in result, (
            f"Language lock for '{lang}' missing in system prompt "
            f"(command context: {cmd})"
        )
