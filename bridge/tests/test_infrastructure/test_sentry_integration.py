"""Tests for Sentry SDK integration in main.py.

Validates:
1. _sentry_before_send strips user-controlled text from events
2. _sentry_before_send preserves non-sensitive data
3. Breadcrumb data is also sanitized
4. Telegram bot tokens in request/breadcrumb URLs are redacted
"""

from __future__ import annotations

import copy

from main import _redact_sensitive_url, _sentry_before_send


class TestSentryBeforeSend:
    """Tests for the _sentry_before_send privacy filter."""

    def test_strips_user_message_from_extra(self) -> None:
        """User-controlled keys are removed from extra context."""
        event = {
            "extra": {
                "message_text": "secret user message",
                "user_message": "another secret",
                "user_input": "yet another secret",
                "claim": "user claim text",
                "request_id": "abc123",
            },
        }
        cleaned = _sentry_before_send(event, {})

        assert cleaned is not None
        assert "message_text" not in cleaned["extra"]
        assert "user_message" not in cleaned["extra"]
        assert "user_input" not in cleaned["extra"]
        assert "claim" not in cleaned["extra"]
        # Non-sensitive data preserved
        assert cleaned["extra"]["request_id"] == "abc123"

    def test_strips_request_data(self) -> None:
        """Request body and query string are removed."""
        event = {
            "request": {
                "data": "secret body",
                "query_string": "foo=bar",
                "url": "https://api.telegram.org/bot/sendMessage",
                "method": "POST",
            },
        }
        cleaned = _sentry_before_send(event, {})

        assert cleaned is not None
        assert "data" not in cleaned["request"]
        assert "query_string" not in cleaned["request"]
        # Non-sensitive request metadata preserved
        assert cleaned["request"]["url"] == "https://api.telegram.org/bot/sendMessage"
        assert cleaned["request"]["method"] == "POST"

    def test_strips_breadcrumb_data(self) -> None:
        """User-controlled keys are removed from breadcrumb data."""
        event = {
            "breadcrumbs": {
                "values": [
                    {
                        "category": "telegram",
                        "data": {
                            "message_text": "secret",
                            "user_input": "also secret",
                            "text": "raw text",
                            "handler": "handle_message",
                        },
                    },
                    {
                        "category": "http",
                        "data": {
                            "url": "https://example.com",
                            "status_code": 200,
                        },
                    },
                ],
            },
        }
        cleaned = _sentry_before_send(event, {})

        assert cleaned is not None
        crumbs = cleaned["breadcrumbs"]["values"]
        # First breadcrumb: user data stripped, handler preserved
        assert "message_text" not in crumbs[0]["data"]
        assert "user_input" not in crumbs[0]["data"]
        assert "text" not in crumbs[0]["data"]
        assert crumbs[0]["data"]["handler"] == "handle_message"
        # Second breadcrumb: untouched (no sensitive keys)
        assert crumbs[1]["data"]["url"] == "https://example.com"
        assert crumbs[1]["data"]["status_code"] == 200

    def test_returns_event_not_none(self) -> None:
        """before_send always returns the event (never drops it)."""
        event = {"message": "SomeError occurred", "level": "error"}
        result = _sentry_before_send(event, {})
        assert result is not None
        assert result["message"] == "SomeError occurred"

    def test_handles_missing_sections_gracefully(self) -> None:
        """Events without request/extra/breadcrumbs don't crash."""
        event = {"event_id": "abc123", "level": "error"}
        result = _sentry_before_send(event, {})
        assert result is not None
        assert result["event_id"] == "abc123"

    def test_handles_empty_breadcrumbs(self) -> None:
        """Empty breadcrumbs list is handled without error."""
        event = {"breadcrumbs": {"values": []}}
        result = _sentry_before_send(event, {})
        assert result is not None
        assert result["breadcrumbs"]["values"] == []

    def test_does_not_mutate_hint(self) -> None:
        """The hint dict is not modified."""
        event = {"extra": {"message_text": "secret"}}
        hint = {"exc_info": "some_traceback_info", "mechanism": "test"}
        hint_copy = copy.deepcopy(hint)
        _sentry_before_send(event, hint)
        assert hint == hint_copy

    # --- R6-SEC-01: Telegram bot token redaction ---

    def test_redacts_telegram_bot_token_from_request_url(self) -> None:
        """Bot tokens in request URLs are replaced with [REDACTED]."""
        event = {
            "request": {
                "url": "https://api.telegram.org/bot123:ABC/sendMessage",
                "method": "POST",
            }
        }
        cleaned = _sentry_before_send(event, {})

        assert "123:ABC" not in cleaned["request"]["url"]
        assert (
            cleaned["request"]["url"]
            == "https://api.telegram.org/bot[REDACTED]/sendMessage"
        )

    def test_redacts_telegram_bot_token_from_breadcrumb_url(self) -> None:
        """Bot tokens in breadcrumb data URLs are replaced with [REDACTED]."""
        event = {
            "breadcrumbs": {
                "values": [
                    {
                        "category": "http",
                        "data": {
                            "url": "https://api.telegram.org/bot789:XYZ/getUpdates",
                            "status_code": 200,
                        },
                    },
                ],
            },
        }
        cleaned = _sentry_before_send(event, {})

        crumb_url = cleaned["breadcrumbs"]["values"][0]["data"]["url"]
        assert "789:XYZ" not in crumb_url
        assert crumb_url == "https://api.telegram.org/bot[REDACTED]/getUpdates"

    def test_preserves_non_telegram_urls(self) -> None:
        """Non-Telegram URLs are not modified by the redaction."""
        event = {
            "request": {
                "url": "https://example.com/api/v1/data",
                "method": "GET",
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert cleaned["request"]["url"] == "https://example.com/api/v1/data"


class TestSentryAllowlistHardening:
    """Tests for FIX-01/02/03: Allowlist, exception redaction, frame stripping."""

    def test_allowlist_strips_unknown_extra_keys(self) -> None:
        """Unknown extra keys are stripped by the allowlist filter."""
        event = {
            "extra": {
                "request_id": "abc-123",  # allowed
                "model_id": "claude-4",  # allowed
                "prompt": "System secret instructions",  # NOT allowed
                "raw_text": "User private message",  # NOT allowed
                "some_random_key": "should vanish",  # NOT allowed
            },
        }
        cleaned = _sentry_before_send(event, {})

        # Allowed keys survive
        assert cleaned["extra"]["request_id"] == "abc-123"
        assert cleaned["extra"]["model_id"] == "claude-4"
        # Non-allowed keys are removed
        assert "prompt" not in cleaned["extra"]
        assert "raw_text" not in cleaned["extra"]
        assert "some_random_key" not in cleaned["extra"]

    def test_allowlist_preserves_all_allowed_keys(self) -> None:
        """All keys in the allowlist are preserved."""
        from main import _SENTRY_EXTRA_ALLOWLIST

        event = {"extra": {k: f"value-{k}" for k in _SENTRY_EXTRA_ALLOWLIST}}
        cleaned = _sentry_before_send(event, {})

        for key in _SENTRY_EXTRA_ALLOWLIST:
            assert key in cleaned["extra"], f"Allowlisted key '{key}' was stripped"
            assert cleaned["extra"][key] == f"value-{key}"

    def test_exception_value_redacted_for_standard_types(self) -> None:
        """Exception messages are redacted for ValueError, TypeError, etc."""
        for exc_type in (
            "ValueError",
            "TypeError",
            "RuntimeError",
            "KeyError",
            "AttributeError",
        ):
            event = {
                "exception": {
                    "values": [
                        {
                            "type": exc_type,
                            "value": f"Bad input: user-secret-data-{exc_type}",
                            "stacktrace": {"frames": []},
                        }
                    ]
                }
            }
            cleaned = _sentry_before_send(event, {})
            assert (
                cleaned["exception"]["values"][0]["value"]
                == "<exception message redacted>"
            )

    def test_frame_locals_stripped(self) -> None:
        """Frame vars are removed from all stack frames."""
        event = {
            "exception": {
                "values": [
                    {
                        "type": "RuntimeError",
                        "value": "fail",
                        "stacktrace": {
                            "frames": [
                                {
                                    "filename": "handler.py",
                                    "function": "handle",
                                    "vars": {"user_text": "secret", "x": 42},
                                },
                                {
                                    "filename": "service.py",
                                    "function": "process",
                                    "vars": {"msg": "private"},
                                },
                            ]
                        },
                    }
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        frames = cleaned["exception"]["values"][0]["stacktrace"]["frames"]
        for frame in frames:
            assert "vars" not in frame
        # Non-sensitive metadata preserved
        assert frames[0]["filename"] == "handler.py"
        assert frames[1]["function"] == "process"

    def test_breadcrumb_allowlist_symmetry(self) -> None:
        """Breadcrumb data uses same allowlist logic; unknown keys are stripped."""
        event = {
            "breadcrumbs": {
                "values": [
                    {
                        "category": "http",
                        "data": {
                            "url": "https://example.com",
                            "status_code": 200,
                            "method": "POST",
                            "handler": "test",
                            # These should be stripped:
                            "body": '{"text":"secret"}',
                            "response": "echo secret",
                            "raw_text": "private",
                        },
                    }
                ]
            }
        }
        cleaned = _sentry_before_send(event, {})
        data = cleaned["breadcrumbs"]["values"][0]["data"]
        # Allowed
        assert data["url"] == "https://example.com"
        assert data["status_code"] == 200
        assert data["method"] == "POST"
        assert data["handler"] == "test"
        # Stripped
        assert "body" not in data
        assert "response" not in data
        assert "raw_text" not in data

    def test_exception_value_redacted_for_generic_exception(self) -> None:
        """CFV-02: Generic Exception values are also redacted (privacy-by-default)."""
        event = {
            "exception": {
                "values": [{"type": "Exception", "value": "private user text"}]
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert (
            cleaned["exception"]["values"][0]["value"] == "<exception message redacted>"
        )

    def test_exception_value_redacted_for_custom_exception(self) -> None:
        """CFV-02: Custom app exceptions are also redacted (privacy-by-default)."""
        event = {
            "exception": {
                "values": [{"type": "MyAppException", "value": "user leaked data here"}]
            }
        }
        cleaned = _sentry_before_send(event, {})
        assert (
            cleaned["exception"]["values"][0]["value"] == "<exception message redacted>"
        )

    def test_raw_user_id_not_in_allowlist(self) -> None:
        """CFV-03: Raw user_id is stripped from Sentry extra (PII)."""
        event = {"extra": {"user_id": 123456789, "chat_id_hash": "abc"}}
        cleaned = _sentry_before_send(event, {})
        assert "user_id" not in cleaned["extra"]
        assert cleaned["extra"].get("chat_id_hash") == "abc"

    def test_bot_token_url_still_redacted_after_hardening(self) -> None:
        """Bot token redaction works correctly after allowlist changes."""
        event = {
            "request": {
                "url": "https://api.telegram.org/bot555:TOKEN/sendMessage",
                "method": "POST",
            },
            "breadcrumbs": {
                "values": [
                    {
                        "category": "http",
                        "data": {
                            "url": "https://api.telegram.org/bot777:SEC/getUpdates",
                            "status_code": 200,
                        },
                    }
                ]
            },
        }
        cleaned = _sentry_before_send(event, {})
        assert "555:TOKEN" not in cleaned["request"]["url"]
        assert "[REDACTED]" in cleaned["request"]["url"]
        crumb_url = cleaned["breadcrumbs"]["values"][0]["data"]["url"]
        assert "777:SEC" not in crumb_url
        assert "[REDACTED]" in crumb_url


class TestRedactSensitiveUrl:
    """Direct tests for the _redact_sensitive_url helper."""

    def test_redacts_standard_bot_url(self) -> None:
        url = "https://api.telegram.org/bot123456:ABCDEF/sendMessage"
        assert (
            _redact_sensitive_url(url)
            == "https://api.telegram.org/bot[REDACTED]/sendMessage"
        )

    def test_no_match_returns_original(self) -> None:
        url = "https://example.com/some/path"
        assert _redact_sensitive_url(url) == url

    def test_redacts_long_token(self) -> None:
        url = "https://api.telegram.org/bot1234567890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/getMe"
        result = _redact_sensitive_url(url)
        assert "1234567890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in result
        assert result == "https://api.telegram.org/bot[REDACTED]/getMe"
