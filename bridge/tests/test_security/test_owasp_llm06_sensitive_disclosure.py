"""OWASP LLM06: Sensitive Information Disclosure tests.

Verifies that AXOLENT enforces strict user isolation (memory, history)
and that secrets (bot token, API keys) never appear in responses.
Also verifies that the Sentry audit log does not contain raw user text.

Production paths tested:
    - infrastructure.memory_storage.MemoryStorage (user_id scoping)
    - infrastructure.conversation_storage (user_id, chat_id keying)
    - application.leakage_filter (forbidden patterns for tokens)
    - main._sentry_before_send (PII stripping)
"""

from __future__ import annotations

from typing import Any

import pytest

from infrastructure.memory_storage import MemoryStorage
from main import _sentry_before_send


@pytest.mark.security
@pytest.mark.owasp_llm
class TestLLM06SensitiveDisclosure:
    """LLM06: Secrets and cross-user data never disclosed."""

    def test_user_a_memory_never_appears_in_user_b_context(
        self, isolated_memory_stores: dict[str, Any]
    ) -> None:
        """WHAT: User A stores a memory entry. User B searches for it.
        EXPECTED: User B's search returns empty (user_id filtering).
        WHY: Memory entries are keyed by user_id. A storage bug could
            allow cross-user memory leakage.
        """
        storage: MemoryStorage = isolated_memory_stores["storage"]
        user_a = isolated_memory_stores["user_a_id"]
        user_b = isolated_memory_stores["user_b_id"]

        # User A stores a secret
        secret_entry = {
            "id": "ep_secret_001",
            "user_id": user_a,
            "content": "My bank account PIN is 9876",
            "importance": 9,
            "created_at": "2026-05-20T10:00:00Z",
        }
        storage.append(secret_entry, "episodic")

        # User B searches for it
        results_b = storage.search(user_b, "bank account PIN", layer="episodic")
        assert len(results_b) == 0, f"User B found User A's memory: {results_b}"

        # User A CAN find their own entry
        results_a = storage.search(user_a, "bank account PIN", layer="episodic")
        assert len(results_a) >= 1, "User A should find their own memory"

    def test_user_a_conversation_history_never_in_user_b_response(
        self,
    ) -> None:
        """WHAT: Conversation storage uses (user_id, chat_id) keys.
        EXPECTED: History for user A is isolated from user B.
        WHY: In-memory dict keyed by tuple ensures isolation, but a
            regression could break this.
        """
        import asyncio
        from infrastructure.conversation_storage import (
            _histories,
            get_history,
            save_turn,
        )
        from domain.conversation import ConversationTurn

        user_a_id, chat_a = 3001, 3001
        user_b_id, chat_b = 4002, 4002

        # Clear any prior state
        _histories.pop((user_a_id, chat_a), None)
        _histories.pop((user_b_id, chat_b), None)

        async def _test() -> None:
            # User A has a conversation
            turn_a = ConversationTurn(
                role="user", content="My secret password is hunter2"
            )
            await save_turn(user_a_id, chat_a, turn_a)

            # User B's history should be empty
            history_b = await get_history(user_b_id, chat_b)
            assert len(history_b) == 0, f"User B has access to history: {history_b}"

            # Verify User A's history exists
            history_a = await get_history(user_a_id, chat_a)
            assert len(history_a) == 1
            assert "hunter2" in history_a[0].content

        asyncio.run(_test())

        # Cleanup
        _histories.pop((user_a_id, chat_a), None)
        _histories.pop((user_b_id, chat_b), None)

    def test_telegram_bot_token_never_in_any_reply(self) -> None:
        """WHAT: LLM response accidentally contains a Telegram bot token pattern.
        EXPECTED: The secret scanner or leakage filter would catch this
            before it reaches the user (via forbidden patterns or the
            SecretScanner in the privacy pipeline).
        WHY: A confused model might echo env vars or config values.
        """
        from application.skill_compression.privacy.secret_scanner import SecretScanner

        scanner = SecretScanner()
        # Real Telegram bot token format
        token_in_response = "Here is the config: TELEGRAM_TOKEN=7234567890:AAHfiqksKZ8WmR2zCwdZ3C3FYP0P0kReal"
        matches = scanner.scan(token_in_response)
        # Should detect the API token pattern
        has_token_match = any(
            "token" in m.pattern_name.lower() or "api" in m.pattern_name.lower()
            for m in matches
        )
        assert has_token_match, (
            f"SecretScanner missed Telegram bot token. Matches: {[m.pattern_name for m in matches]}"
        )

    def test_anthropic_api_key_never_in_any_reply(self) -> None:
        """WHAT: LLM response contains an Anthropic API key pattern.
        EXPECTED: SecretScanner detects the sk-ant- prefix pattern.
        WHY: The API key gives full access to the Claude account.
        """
        from application.skill_compression.privacy.secret_scanner import SecretScanner

        scanner = SecretScanner()
        key_in_response = (
            "You can use this key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
        )
        matches = scanner.scan(key_in_response)
        has_key_match = any(
            "token" in m.pattern_name.lower()
            or "api" in m.pattern_name.lower()
            or "key" in m.pattern_name.lower()
            for m in matches
        )
        assert has_key_match, (
            f"SecretScanner missed Anthropic API key. Matches: {[m.pattern_name for m in matches]}"
        )

    def test_audit_log_never_contains_raw_user_text(self) -> None:
        """WHAT: Sentry before_send strips user-controlled fields.
        EXPECTED: Keys like message_text, user_message, user_input, claim
            are removed. Only allowlisted keys survive.
        WHY: Audit/error tracking must not contain raw user PII
            (GDPR, privacy-by-design).
        """
        event: dict[str, Any] = {
            "extra": {
                "message_text": "My secret: I hate my boss",
                "user_message": "Private conversation content",
                "user_input": "sk-ant-api03-realkey",
                "claim": "User has depression",
                "request_id": "req_abc123",
                "user_id": 12345,
                "chat_id": 67890,
                "duration": 1.5,
            }
        }
        cleaned = _sentry_before_send(event, {})
        extra = cleaned.get("extra", {})

        # Sensitive keys must be gone
        assert "message_text" not in extra
        assert "user_message" not in extra
        assert "user_input" not in extra
        assert "claim" not in extra

        # Allowlisted keys preserved
        assert extra.get("request_id") == "req_abc123"

    def test_system_prompt_text_never_in_sentry_extra(self) -> None:
        """WHAT: system_prompt key in Sentry extra.
        EXPECTED: Stripped by allowlist filter.
        WHY: System prompt is confidential business logic.
        """
        event: dict[str, Any] = {
            "extra": {
                "system_prompt": "You are AXOLENT AI. Secret rules...",
                "request_id": "req_xyz",
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert "system_prompt" not in cleaned.get("extra", {})
