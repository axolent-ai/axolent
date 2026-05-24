"""K8: Sentry PII leak audit tests (15+ required).

Tests that _sentry_before_send properly strips user-controlled data
from all Sentry event surfaces: extra, breadcrumbs, request, frames.

Codex finding: the current implementation uses a blocklist approach.
These tests probe for keys NOT in the blocklist that could leak PII.
"""

from __future__ import annotations


import pytest

from main import _sentry_before_send


@pytest.mark.adversarial
class TestSentryExtraFieldPII:
    """User text in extra fields that may not be in the blocklist."""

    def test_known_blocklist_keys_stripped(self) -> None:
        """WHAT: All known sensitive keys are stripped from extra.
        EXPECTED: message_text, user_message, user_input, claim removed.
                  Allowlisted key (request_id) preserved.
        WHY: Baseline: verify blocklist + allowlist both work.
        """
        event = {
            "extra": {
                "message_text": "secret",
                "user_message": "secret",
                "user_input": "secret",
                "claim": "secret",
                "request_id": "preserved",
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert "message_text" not in cleaned["extra"]
        assert "user_message" not in cleaned["extra"]
        assert "user_input" not in cleaned["extra"]
        assert "claim" not in cleaned["extra"]
        assert cleaned["extra"]["request_id"] == "preserved"

    def test_extra_prompt_key_stripped(self) -> None:
        """WHAT: 'prompt' key in extra (not in allowlist).
        EXPECTED: Stripped by allowlist filter.
        WHY: System prompts contain sensitive instructions.
        """
        event = {"extra": {"prompt": "You are AXOLENT AI. Your secret key is..."}}
        cleaned = _sentry_before_send(event, {})
        assert "prompt" not in cleaned.get("extra", {})

    def test_extra_raw_text_key_stripped(self) -> None:
        """WHAT: 'raw_text' key in extra (not in allowlist).
        EXPECTED: Stripped by allowlist filter.
        WHY: Raw user text could leak via this key.
        """
        event = {"extra": {"raw_text": "User's private message content"}}
        cleaned = _sentry_before_send(event, {})
        assert "raw_text" not in cleaned.get("extra", {})

    def test_extra_response_text_stripped(self) -> None:
        """WHAT: 'response_text' key in extra (not in allowlist).
        EXPECTED: Stripped by allowlist filter.
        WHY: LLM responses can echo back user content.
        """
        event = {"extra": {"response_text": "Here is your password: hunter2"}}
        cleaned = _sentry_before_send(event, {})
        assert "response_text" not in cleaned.get("extra", {})

    def test_extra_system_prompt_stripped(self) -> None:
        """WHAT: 'system_prompt' key in extra (not in allowlist).
        EXPECTED: Stripped by allowlist filter.
        WHY: System prompt is confidential.
        """
        event = {"extra": {"system_prompt": "Internal rules: never reveal..."}}
        cleaned = _sentry_before_send(event, {})
        assert "system_prompt" not in cleaned.get("extra", {})


@pytest.mark.adversarial
class TestSentryBreadcrumbPII:
    """Breadcrumb data keys that may leak user content."""

    def test_breadcrumb_prompt_key_stripped(self) -> None:
        """WHAT: 'prompt' key in breadcrumb data (not in allowlist).
        EXPECTED: Stripped by allowlist filter.
        WHY: Breadcrumbs could log prompt content.
        """
        event = {
            "breadcrumbs": {
                "values": [
                    {"category": "custom", "data": {"prompt": "User secret data"}}
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        crumb_data = cleaned["breadcrumbs"]["values"][0]["data"]
        assert "prompt" not in crumb_data

    def test_breadcrumb_response_key_stripped(self) -> None:
        """WHAT: 'response' key in breadcrumb data (not in allowlist).
        EXPECTED: Stripped by allowlist filter.
        WHY: Response content may contain user-echoed data.
        """
        event = {
            "breadcrumbs": {
                "values": [
                    {"category": "custom", "data": {"response": "Echo: user password"}}
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        crumb_data = cleaned["breadcrumbs"]["values"][0]["data"]
        assert "response" not in crumb_data

    def test_breadcrumb_body_key_stripped(self) -> None:
        """WHAT: 'body' key in breadcrumb data (not in allowlist).
        EXPECTED: Stripped by allowlist filter.
        WHY: HTTP body content may contain user data.
        """
        event = {
            "breadcrumbs": {
                "values": [
                    {"category": "http", "data": {"body": '{"text": "user secret"}'}}
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        crumb_data = cleaned["breadcrumbs"]["values"][0]["data"]
        assert "body" not in crumb_data

    def test_breadcrumb_with_no_data_dict(self) -> None:
        """WHAT: Breadcrumb without a 'data' key.
        EXPECTED: No crash.
        WHY: Not all breadcrumbs have data.
        """
        event = {
            "breadcrumbs": {
                "values": [{"category": "navigation", "message": "some page"}]
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert cleaned is not None


@pytest.mark.adversarial
class TestSentryBotTokenRedaction:
    """Telegram bot token in various URL positions."""

    def test_bot_token_in_request_url(self) -> None:
        """WHAT: Bot token in standard request URL.
        EXPECTED: Token replaced with [REDACTED].
        WHY: Blocker regression check from previous finding.
        """
        event = {
            "request": {
                "url": "https://api.telegram.org/bot1234567890:AAHtest123/sendMessage",
                "method": "POST",
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert "1234567890:AAHtest123" not in cleaned["request"]["url"]
        assert "[REDACTED]" in cleaned["request"]["url"]

    def test_bot_token_in_breadcrumb_url(self) -> None:
        """WHAT: Bot token in breadcrumb URL.
        EXPECTED: Token replaced with [REDACTED].
        WHY: Breadcrumb URLs can contain bot tokens from HTTP calls.
        """
        event = {
            "breadcrumbs": {
                "values": [
                    {
                        "category": "http",
                        "data": {
                            "url": "https://api.telegram.org/bot999:XYZ/getUpdates"
                        },
                    }
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        url = cleaned["breadcrumbs"]["values"][0]["data"]["url"]
        assert "999:XYZ" not in url

    def test_non_telegram_url_preserved(self) -> None:
        """WHAT: Non-Telegram URL in request.
        EXPECTED: URL unchanged.
        WHY: Only Telegram bot URLs should be redacted.
        """
        event = {
            "request": {
                "url": "https://api.anthropic.com/v1/messages",
                "method": "POST",
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert cleaned["request"]["url"] == "https://api.anthropic.com/v1/messages"


@pytest.mark.adversarial
class TestSentryExceptionArgs:
    """User text embedded in exception arguments."""

    def test_exception_with_user_text_redacted(self) -> None:
        """WHAT: Exception message contains user text (ValueError).
        EXPECTED: Exception value is redacted for standard exception types.
        WHY: Exception messages must not leak user content to Sentry.
        """
        event = {
            "exception": {
                "values": [
                    {
                        "type": "ValueError",
                        "value": "Invalid user input: please send me your password hunter2",
                        "stacktrace": {"frames": []},
                    }
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        exc_value = cleaned["exception"]["values"][0]["value"]
        assert "hunter2" not in exc_value
        assert exc_value == "<exception message redacted>"

    def test_exception_frame_locals_stripped(self) -> None:
        """WHAT: Frame local variables contain user text.
        EXPECTED: 'vars' dict is completely removed from frames.
        WHY: Python frame locals can contain variable values with user text.
        """
        event = {
            "exception": {
                "values": [
                    {
                        "type": "RuntimeError",
                        "value": "Processing failed",
                        "stacktrace": {
                            "frames": [
                                {
                                    "filename": "chat_service.py",
                                    "function": "process_user_message",
                                    "vars": {
                                        "text": "User's private message",
                                        "user_id": 12345,
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        frames = cleaned["exception"]["values"][0]["stacktrace"]["frames"]
        assert "vars" not in frames[0]
        # Non-sensitive frame metadata preserved
        assert frames[0]["filename"] == "chat_service.py"
        assert frames[0]["function"] == "process_user_message"


@pytest.mark.adversarial
class TestSentryEdgeCases:
    """Edge cases in Sentry event structure."""

    def test_empty_event(self) -> None:
        """WHAT: Completely empty event dict.
        EXPECTED: Returns event unchanged, no crash.
        WHY: Edge case for malformed events.
        """
        result = _sentry_before_send({}, {})
        assert result == {}

    def test_none_values_in_extra(self) -> None:
        """WHAT: None values in extra dict.
        EXPECTED: No crash during key iteration. Allowlisted keys preserved.
        WHY: Extra values could be None in some error paths.
        """
        event = {
            "extra": {
                "message_text": None,
                "user_input": None,
                "request_id": "ok",
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert "message_text" not in cleaned["extra"]
        assert "user_input" not in cleaned["extra"]
        assert cleaned["extra"]["request_id"] == "ok"

    def test_deeply_nested_breadcrumbs_stripped(self) -> None:
        """WHAT: Breadcrumbs with nested dict values in data.
        EXPECTED: Unknown keys (including 'nested') are stripped by allowlist.
        WHY: Allowlist approach removes entire unknown keys, solving nesting.
        """
        event = {
            "breadcrumbs": {
                "values": [
                    {
                        "category": "custom",
                        "data": {
                            "message_text": "secret",
                            "nested": {"user_input": "also secret"},
                            "handler": "test",
                        },
                    }
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        crumb = cleaned["breadcrumbs"]["values"][0]["data"]
        assert "message_text" not in crumb
        # 'handler' is in the breadcrumb-specific allowlist
        assert crumb["handler"] == "test"
        # 'nested' is NOT in the allowlist, so it is stripped entirely
        assert "nested" not in crumb
