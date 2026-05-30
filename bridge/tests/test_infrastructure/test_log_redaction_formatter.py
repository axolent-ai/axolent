"""Tests for RedactingFormatter and traceback redaction (Finding 6).

4-Path: Happy + Malicious + Rejection + Privacy.
Production-Path: through actual logging setup.
"""

from __future__ import annotations

import logging
import sys

import pytest

from infrastructure.log_redaction import (
    RedactingFormatter,
    install_secret_redaction_filter,
)


@pytest.fixture
def logger_with_formatter() -> logging.Logger:
    """Create a test logger with RedactingFormatter installed."""
    logger = logging.getLogger("test_redaction_formatter")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.filters = []

    handler = logging.StreamHandler(stream=None)
    handler.setFormatter(RedactingFormatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


class TestRedactingFormatterHappy:
    """Happy path: normal log messages pass through."""

    def test_normal_message(self, logger_with_formatter: logging.Logger) -> None:
        handler = logger_with_formatter.handlers[0]
        record = logger_with_formatter.makeRecord(
            "test", logging.INFO, "test.py", 1, "Hello world", (), None
        )
        output = handler.format(record)
        assert "Hello world" in output

    def test_message_with_secret_in_msg(
        self, logger_with_formatter: logging.Logger
    ) -> None:
        """Secret in msg field is redacted."""
        record = logger_with_formatter.makeRecord(
            "test",
            logging.INFO,
            "test.py",
            1,
            "Token: bot123456789:AAFakeTokenThatIsLongEnough1234",
            (),
            None,
        )
        handler = logger_with_formatter.handlers[0]
        output = handler.format(record)
        assert "AAFakeToken" not in output
        assert "REDACTED" in output


class TestRedactingFormatterMalicious:
    """Malicious: secrets in tracebacks are redacted."""

    def test_telegram_token_in_traceback(
        self, logger_with_formatter: logging.Logger
    ) -> None:
        """Bot token in exception traceback is redacted."""
        handler = logger_with_formatter.handlers[0]

        try:
            raise RuntimeError(
                "Connection failed: "
                "https://api.telegram.org/bot123456789:AAFakeTokenLongEnough12345/sendMessage"
            )
        except RuntimeError:
            exc_info = sys.exc_info()

        record = logger_with_formatter.makeRecord(
            "test",
            logging.ERROR,
            "test.py",
            1,
            "Request failed",
            (),
            exc_info,
        )
        output = handler.format(record)
        assert "AAFakeToken" not in output
        assert "REDACTED" in output
        # Traceback structure is preserved
        assert "RuntimeError" in output

    def test_openai_key_in_traceback(
        self, logger_with_formatter: logging.Logger
    ) -> None:
        """OpenAI key in traceback is redacted."""
        handler = logger_with_formatter.handlers[0]

        try:
            raise ValueError(
                "Invalid key: sk-proj-abc123def456ghi789jkl012mno345pqr678"
            )
        except ValueError:
            exc_info = sys.exc_info()

        record = logger_with_formatter.makeRecord(
            "test",
            logging.ERROR,
            "test.py",
            1,
            "Auth failed",
            (),
            exc_info,
        )
        output = handler.format(record)
        assert "sk-proj-abc123" not in output
        assert "REDACTED" in output

    def test_anthropic_key_in_traceback(
        self, logger_with_formatter: logging.Logger
    ) -> None:
        """Anthropic key in traceback is redacted."""
        handler = logger_with_formatter.handlers[0]

        try:
            raise ConnectionError(
                "API error with key sk-ant-abc123456789012345678901234567890"
            )
        except ConnectionError:
            exc_info = sys.exc_info()

        record = logger_with_formatter.makeRecord(
            "test",
            logging.ERROR,
            "test.py",
            1,
            "Connection error",
            (),
            exc_info,
        )
        output = handler.format(record)
        assert "sk-ant-abc123" not in output
        assert "REDACTED" in output


class TestRedactingFormatterProductionPath:
    """Production-Path: through install_secret_redaction_filter."""

    def test_installed_formatter_redacts_exception(self) -> None:
        """Full production-path: install filter, log exception, verify redacted."""
        logger = logging.getLogger("test_production_redaction")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.filters = []

        # Add a handler (simulating production setup)
        handler = logging.StreamHandler(stream=None)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        # Install redaction (production call)
        install_secret_redaction_filter(logger)

        # Now the handler should have RedactingFormatter
        assert isinstance(handler.formatter, RedactingFormatter)

        # Log an exception with a secret in traceback
        try:
            raise RuntimeError(
                "Error at https://api.telegram.org/bot999888777:BBSecretTokenHere1234567890/getMe"
            )
        except RuntimeError:
            exc_info = sys.exc_info()

        record = logger.makeRecord(
            "test",
            logging.ERROR,
            "test.py",
            1,
            "Bot error",
            (),
            exc_info,
        )
        output = handler.format(record)
        assert "BBSecretToken" not in output
        assert "REDACTED" in output


class TestRedactingFormatterPrivacy:
    """Privacy: formatter output never contains known secret patterns."""

    GOLDEN_CORPUS: list[tuple[str, str]] = [
        # (secret, identifying_fragment_that_must_NOT_appear)
        ("bot123456789:AAExampleSecretTokenValue12345", "AAExampleSecret"),
        ("sk-ant-supersecret12345678901234567890", "supersecret"),
        ("sk-proj-abc123def456ghi789jkl012mno345pqr678", "abc123def456"),
        ("sk-svcacct-longserviceaccountkey123456", "longserviceacc"),
        ("sk-admin-administratorkey1234567890ab", "administratork"),
        ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature", "eyJhbGciOiJI"),
        (
            "https://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa@o1234.ingest.sentry.io/5678",
            "aaaaaaaaaaaa@o1234",
        ),
    ]

    @pytest.mark.parametrize("secret,fragment", GOLDEN_CORPUS)
    def test_golden_corpus_never_in_output(self, secret: str, fragment: str) -> None:
        """No secret from the golden corpus ever appears in formatted output."""
        formatter = RedactingFormatter("%(message)s")

        try:
            raise RuntimeError(f"Error with {secret}")
        except RuntimeError:
            exc_info = sys.exc_info()

        logger = logging.getLogger("golden_test")
        record = logger.makeRecord(
            "test", logging.ERROR, "test.py", 1, f"msg: {secret}", (), exc_info
        )
        output = formatter.format(record)
        assert fragment not in output, (
            f"Secret fragment '{fragment}' found in formatted output"
        )
