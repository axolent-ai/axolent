"""4-Path tests for SecretRedactionFilter.

Paths tested:
  1. Happy: normal messages pass through unchanged.
  2. Malicious/Secret: messages containing secrets get redacted.
  3. Rejection/Edge: empty/None/non-string args don't crash.
  4. Privacy/E2E: secrets never reach captured log output.

IMPORTANT: Test tokens are built programmatically (never literal constants).
"""

from __future__ import annotations

import logging
import time

import pytest

from infrastructure.log_redaction import (
    SecretRedactionFilter,
    _redact_string,
    install_secret_redaction_filter,
)


# ---------------------------------------------------------------------------
# Helpers: programmatically assembled test tokens (never literal constants)
# ---------------------------------------------------------------------------


def _make_telegram_bot_token() -> str:
    """Build a realistic-looking Telegram bot token for testing."""
    # bot<numeric_id>:<alphanumeric_hash>
    numeric_id = "123456789"
    hash_part = "A" * 35  # 35 chars, well above the 20-char minimum
    return f"{numeric_id}:{hash_part}"


def _make_telegram_bot_url() -> str:
    """Build a realistic httpx-style Telegram API URL with embedded token."""
    token = _make_telegram_bot_token()
    return f"https://api.telegram.org/bot{token}/getUpdates"


def _make_sentry_dsn() -> str:
    """Build a realistic Sentry DSN for testing."""
    key = "a" * 32  # 32 hex chars
    return f"https://{key}@o123456.ingest.sentry.io/7654321"


def _make_anthropic_key() -> str:
    """Build a realistic Anthropic API key for testing."""
    suffix = "x" * 40  # well above 20-char minimum
    return f"sk-ant-{suffix}"


def _make_openai_key() -> str:
    """Build a realistic OpenAI API key for testing."""
    suffix = "y" * 48  # 48 chars, above 40-char minimum
    return f"sk-{suffix}"


def _make_bearer_token() -> str:
    """Build a realistic Bearer token for testing."""
    token_value = "Z" * 30  # 30 chars, above 20-char minimum
    return f"Bearer {token_value}"


# ---------------------------------------------------------------------------
# PATH 1: Happy Path -- normal messages pass through unchanged
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Normal log entries without secrets should be untouched."""

    def test_normal_message_unchanged(self) -> None:
        msg = "Application started successfully on port 8080"
        assert _redact_string(msg) == msg

    def test_url_without_secret_unchanged(self) -> None:
        msg = "HTTP Request: GET https://example.com/api/v1/users HTTP/1.1 200 OK"
        assert _redact_string(msg) == msg

    def test_short_token_like_string_not_redacted(self) -> None:
        # Short strings that look vaguely token-like but don't match patterns
        msg = "session_id=abc123:xyz"
        assert _redact_string(msg) == msg

    def test_sk_prefix_too_short_not_redacted(self) -> None:
        # sk- followed by less than 40 chars should NOT match OpenAI pattern
        msg = "key: sk-shortvalue"
        assert _redact_string(msg) == msg


# ---------------------------------------------------------------------------
# PATH 2: Malicious/Secret Path -- secrets get redacted
# ---------------------------------------------------------------------------


class TestSecretRedaction:
    """Messages containing secrets must be properly redacted."""

    def test_telegram_bot_url_redacted(self) -> None:
        url = _make_telegram_bot_url()
        result = _redact_string(f"HTTP Request: POST {url} HTTP/1.1 200 OK")
        assert "bot<REDACTED>" in result
        assert _make_telegram_bot_token() not in result

    def test_standalone_telegram_token_redacted(self) -> None:
        token = _make_telegram_bot_token()
        result = _redact_string(f"Token loaded: {token}")
        # Either bot<REDACTED> or <REDACTED-TELEGRAM-TOKEN> depending on context
        assert token not in result

    def test_sentry_dsn_redacted(self) -> None:
        dsn = _make_sentry_dsn()
        result = _redact_string(f"Sentry DSN: {dsn}")
        assert "<REDACTED-SENTRY-DSN>" in result
        assert dsn not in result

    def test_anthropic_key_redacted(self) -> None:
        key = _make_anthropic_key()
        result = _redact_string(f"Using API key: {key}")
        assert "<REDACTED-ANTHROPIC-KEY>" in result
        assert key not in result

    def test_openai_key_redacted(self) -> None:
        key = _make_openai_key()
        result = _redact_string(f"OpenAI key: {key}")
        assert "<REDACTED-OPENAI-KEY>" in result
        assert key not in result

    def test_bearer_token_redacted(self) -> None:
        bearer = _make_bearer_token()
        result = _redact_string(f"Authorization: {bearer}")
        assert "Bearer <REDACTED>" in result
        assert "Z" * 30 not in result

    def test_multiple_secrets_in_one_string(self) -> None:
        url = _make_telegram_bot_url()
        key = _make_anthropic_key()
        msg = f"Request to {url} with key {key}"
        result = _redact_string(msg)
        assert _make_telegram_bot_token() not in result
        assert key not in result
        assert "bot<REDACTED>" in result
        assert "<REDACTED-ANTHROPIC-KEY>" in result

    def test_record_args_tuple_redacted(self) -> None:
        """Secrets in record.args (lazy formatting) are also redacted."""
        filt = SecretRedactionFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="HTTP Request: POST %s %s",
            args=(_make_telegram_bot_url(), "HTTP/1.1 200 OK"),
            exc_info=None,
        )
        filt.filter(record)
        assert isinstance(record.args, tuple)
        assert _make_telegram_bot_token() not in record.args[0]
        assert "bot<REDACTED>" in record.args[0]

    def test_record_args_dict_redacted(self) -> None:
        """Secrets in dict-style args are redacted."""
        filt = SecretRedactionFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Key: %(api_key)s",
            args=None,
            exc_info=None,
        )
        # Set args to dict after construction (avoids LogRecord constructor
        # issue with dict args and the internal len() check).
        record.args = {"api_key": _make_anthropic_key()}
        filt.filter(record)
        assert isinstance(record.args, dict)
        assert "<REDACTED-ANTHROPIC-KEY>" in record.args["api_key"]


# ---------------------------------------------------------------------------
# PATH 3: Rejection/Edge Cases -- no crashes on weird input
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases must not crash the filter."""

    def test_empty_message(self) -> None:
        filt = SecretRedactionFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=None,
            exc_info=None,
        )
        result = filt.filter(record)
        assert result is True
        assert record.msg == ""

    def test_none_message(self) -> None:
        """record.msg can technically be None in edge cases."""
        filt = SecretRedactionFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="placeholder",
            args=None,
            exc_info=None,
        )
        # Force msg to None (unusual but possible)
        record.msg = None  # type: ignore[assignment]
        result = filt.filter(record)
        assert result is True  # no crash

    def test_non_string_args_int(self) -> None:
        """Integer args should pass through without crash."""
        filt = SecretRedactionFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Count: %d",
            args=(42,),
            exc_info=None,
        )
        result = filt.filter(record)
        assert result is True
        assert record.args == (42,)

    def test_non_string_args_mixed(self) -> None:
        """Mixed args (int + string with secret) are handled correctly."""
        filt = SecretRedactionFilter()
        key = _make_anthropic_key()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Status %d, key %s",
            args=(200, key),
            exc_info=None,
        )
        result = filt.filter(record)
        assert result is True
        assert isinstance(record.args, tuple)
        assert record.args[0] == 200  # int unchanged
        assert key not in record.args[1]  # string redacted

    def test_large_string_performance(self) -> None:
        """10kB string should be processed in under 5ms."""
        large_msg = "x" * 10_000
        start = time.perf_counter()
        result = _redact_string(large_msg)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert result == large_msg  # no secrets, unchanged
        assert elapsed_ms < 5.0, f"Took {elapsed_ms:.2f}ms, expected <5ms"


# ---------------------------------------------------------------------------
# PATH 4: Privacy/E2E -- secrets never appear in captured log output
# ---------------------------------------------------------------------------


class TestPrivacyEndToEnd:
    """End-to-end: secrets must not appear in actual log output."""

    def test_caplog_telegram_url_redacted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Attach filter to logger, log a Telegram URL, verify no token in output."""
        logger = logging.getLogger("test_privacy_telegram")
        logger.handlers = []
        logger.propagate = True
        install_secret_redaction_filter(logger)

        url = _make_telegram_bot_url()
        with caplog.at_level(logging.INFO, logger="test_privacy_telegram"):
            logger.info("HTTP Request: POST %s HTTP/1.1 200 OK", url)

        # Token must NOT be in any captured record
        for record in caplog.records:
            formatted = record.getMessage()
            assert _make_telegram_bot_token() not in formatted
            assert "bot<REDACTED>" in formatted

    def test_caplog_multiple_secrets_redacted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Multiple secret types in one message are all redacted."""
        logger = logging.getLogger("test_privacy_multi")
        logger.handlers = []
        logger.propagate = True
        install_secret_redaction_filter(logger)

        anthropic_key = _make_anthropic_key()
        bearer = _make_bearer_token()

        with caplog.at_level(logging.INFO, logger="test_privacy_multi"):
            logger.info("Key=%s Auth=%s", anthropic_key, bearer)

        for record in caplog.records:
            formatted = record.getMessage()
            assert anthropic_key not in formatted
            assert "Z" * 30 not in formatted
            assert "<REDACTED-ANTHROPIC-KEY>" in formatted
            assert "Bearer <REDACTED>" in formatted

    def test_caplog_openai_key_redacted(self, caplog: pytest.LogCaptureFixture) -> None:
        """OpenAI key is redacted in captured output."""
        logger = logging.getLogger("test_privacy_openai")
        logger.handlers = []
        logger.propagate = True
        install_secret_redaction_filter(logger)

        key = _make_openai_key()
        with caplog.at_level(logging.INFO, logger="test_privacy_openai"):
            logger.info("Using key: %s", key)

        for record in caplog.records:
            formatted = record.getMessage()
            assert key not in formatted
            assert "<REDACTED-OPENAI-KEY>" in formatted


# ---------------------------------------------------------------------------
# PATH 4+: Production-Path Integration Test
# ---------------------------------------------------------------------------


class TestProductionPath:
    """Production-path test: realistic httpx log format with Telegram URL."""

    def test_log_redaction_blocks_real_telegram_url(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Simulates the exact httpx log format that triggered this fix.

        The httpx logger emits:
          HTTP Request: POST https://api.telegram.org/bot<TOKEN>/getUpdates "HTTP/1.1 200 OK"

        After SecretRedactionFilter is installed, the token portion must be
        replaced with bot<REDACTED> in the final formatted output.
        """
        # Use the httpx logger name (matches production)
        logger = logging.getLogger("httpx")
        # Ensure our filter is on it
        install_secret_redaction_filter(logger)

        # Build realistic message (exact httpx format)
        token = _make_telegram_bot_token()
        httpx_msg = (
            f"HTTP Request: POST https://api.telegram.org/bot{token}"
            f'/getMe "HTTP/1.1 200 OK"'
        )

        with caplog.at_level(logging.INFO, logger="httpx"):
            logger.info(httpx_msg)

        # Verify: token NOT in output, bot<REDACTED> IS in output
        assert len(caplog.records) > 0, "Expected at least one log record"
        for record in caplog.records:
            formatted = record.getMessage()
            assert token not in formatted, (
                f"Token leaked! Found in: {formatted[:80]}..."
            )
            assert "bot<REDACTED>" in formatted, (
                f"Expected 'bot<REDACTED>' in: {formatted[:80]}..."
            )

    def test_log_redaction_handles_url_object_in_args(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """httpx passes URL objects (not strings) in record.args.

        This is the actual production scenario: httpx logs with:
          msg = "HTTP Request: %s %s \"%s %d %s\""
          args = ("POST", URL("https://...bot<TOKEN>/getMe"), "HTTP/1.1", 200, "OK")

        The URL object's str() contains the token. The filter must detect
        this and replace the URL object with a redacted string.
        """
        logger = logging.getLogger("httpx_url_obj_test")
        logger.handlers = []
        logger.propagate = True
        install_secret_redaction_filter(logger)

        # Simulate httpx's URL object (any object with __str__ containing token)
        token = _make_telegram_bot_token()

        class FakeURL:
            """Mimics httpx.URL which has a __str__ containing the full URL."""

            def __init__(self, url: str) -> None:
                self._url = url

            def __str__(self) -> str:
                return self._url

            def __repr__(self) -> str:
                return f"URL('{self._url}')"

        url_obj = FakeURL(f"https://api.telegram.org/bot{token}/getMe")

        with caplog.at_level(logging.INFO, logger="httpx_url_obj_test"):
            logger.info(
                'HTTP Request: %s %s "%s %d %s"',
                "POST",
                url_obj,
                "HTTP/1.1",
                200,
                "OK",
            )

        assert len(caplog.records) > 0
        for record in caplog.records:
            formatted = record.getMessage()
            assert token not in formatted, (
                f"Token leaked via URL object! Found in: {formatted[:100]}"
            )
            assert "bot<REDACTED>" in formatted

    def test_install_idempotent(self) -> None:
        """Calling install_secret_redaction_filter multiple times is safe."""
        logger = logging.getLogger("test_idempotent")
        logger.filters = []

        install_secret_redaction_filter(logger)
        install_secret_redaction_filter(logger)
        install_secret_redaction_filter(logger)

        # Should only have one filter instance
        redaction_filters = [
            f for f in logger.filters if isinstance(f, SecretRedactionFilter)
        ]
        assert len(redaction_filters) == 1
