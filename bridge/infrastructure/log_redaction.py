"""Global log-record secret redaction filter.

Masks API tokens, DSNs, and bearer tokens in all log records BEFORE
they reach any handler (file, stdout, Sentry breadcrumbs, etc.).

Primary motivation: httpx logs the full Telegram bot-API URL at INFO
level, which includes the bot token in cleartext. This filter catches
that and any other secret pattern that might appear in log output.

Usage (early in startup, after logging.basicConfig):

    from infrastructure.log_redaction import install_secret_redaction_filter
    install_secret_redaction_filter()
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Pattern definitions: (compiled regex, replacement string)
# Order matters: more specific patterns first to avoid partial matches.
# ---------------------------------------------------------------------------

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 1. Telegram bot token in URL path: bot<id>:<hash>
    (re.compile(r"bot\d{6,}:[A-Za-z0-9_-]{20,}"), "bot<REDACTED>"),
    # 2. Standalone Telegram bot token: <id>:<hash> (30+ chars after colon)
    (re.compile(r"\d{6,}:[A-Za-z0-9_-]{30,}"), "<REDACTED-TELEGRAM-TOKEN>"),
    # 3. Sentry DSN
    (
        re.compile(r"https://[a-f0-9]{32,}@[a-z0-9.]+\.ingest\.sentry\.io/\d+"),
        "<REDACTED-SENTRY-DSN>",
    ),
    # 4. Anthropic API key
    (re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"), "<REDACTED-ANTHROPIC-KEY>"),
    # 5. OpenAI API key (sk- followed by 40+ alphanum, but NOT sk-ant-)
    (re.compile(r"sk-(?!ant-)[a-zA-Z0-9]{40,}"), "<REDACTED-OPENAI-KEY>"),
    # 6. Generic Bearer token in headers
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"), "Bearer <REDACTED>"),
]


def _redact_string(text: str) -> str:
    """Apply all secret patterns to a string, returning redacted version."""
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_single_arg(arg: object) -> object:
    """Redact a single log-record argument.

    Handles strings directly. For non-string objects (e.g. httpx.URL),
    checks str(obj) for secrets and replaces the arg with the redacted
    string if a secret is found. This covers httpx's pattern of passing
    URL objects in log args which get %s-formatted at emit time.

    Integers and floats are never redacted (fast path, no secrets).
    """
    if isinstance(arg, str):
        return _redact_string(arg)

    # Fast path: numeric types never contain secrets
    if isinstance(arg, (int, float, bool, type(None))):
        return arg

    # For any other object: check its string representation
    try:
        str_repr = str(arg)
    except Exception:
        return arg

    redacted = _redact_string(str_repr)
    if redacted != str_repr:
        # Secret found in string representation -- return redacted string
        # instead of the original object
        return redacted

    return arg


def _redact_args(args: object) -> object:
    """Redact secrets in log record args (tuple, dict, or single value).

    Log records use lazy formatting: ``record.msg % record.args``.
    We must redact both msg and any string values in args. This includes
    non-string objects whose str() representation contains secrets
    (e.g. httpx.URL objects containing bot tokens).

    Returns the (possibly modified) args. Non-string values pass through
    unchanged unless their string representation contains a secret.
    """
    if args is None:
        return args

    if isinstance(args, dict):
        return {k: _redact_single_arg(v) for k, v in args.items()}

    if isinstance(args, tuple):
        return tuple(_redact_single_arg(a) for a in args)

    # Single arg (string or other object)
    return _redact_single_arg(args)


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts secrets from record.msg and record.args.

    Always returns True (never drops records). Only modifies content.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact any secret patterns found in the log record.

        Args:
            record: The log record to inspect/modify.

        Returns:
            Always True (record is never suppressed).
        """
        # Redact msg (may be a format string or pre-formatted)
        if record.msg and isinstance(record.msg, str):
            record.msg = _redact_string(record.msg)

        # Redact args (lazy-formatting values)
        if record.args is not None:
            record.args = _redact_args(record.args)

        return True


# ---------------------------------------------------------------------------
# Logger names that should have the filter installed (defense-in-depth:
# root logger catches everything, but explicit sub-loggers ensure coverage
# even if propagate=False is set somewhere).
# ---------------------------------------------------------------------------

_TARGET_LOGGER_NAMES: Sequence[str] = (
    "httpx",
    "httpcore",
    "telegram",
    "telegram.ext",
    "anthropic",
    "openai",
)


def install_secret_redaction_filter(logger: logging.Logger | None = None) -> None:
    """Install SecretRedactionFilter on loggers AND their handlers.

    Python logging caveat: filters on a parent logger do NOT apply to
    records that propagate up from child loggers. Records propagated from
    child loggers go directly to parent handlers, bypassing parent filters.

    Therefore we install the filter in THREE places:
      1. On the root logger itself (catches records logged directly on root).
      2. On all root-logger handlers (catches ALL propagated records before
         they are emitted -- this is the critical layer).
      3. On key sub-loggers (defense-in-depth for propagate=False cases).

    Safe to call multiple times (idempotent): checks whether the filter
    class is already attached before adding.

    Args:
        logger: If provided, install only on this logger (and its handlers).
                If None (default), install on root + handlers + sub-loggers.
    """
    redaction_filter = SecretRedactionFilter()

    def _add_filter_if_missing(target: logging.Logger | logging.Handler) -> None:
        if not any(isinstance(f, SecretRedactionFilter) for f in target.filters):
            target.addFilter(redaction_filter)

    if logger is not None:
        # Single-logger mode (useful for tests)
        _add_filter_if_missing(logger)
        # Also install on this logger's handlers
        for handler in logger.handlers:
            _add_filter_if_missing(handler)
        return

    # Install on root logger
    root = logging.getLogger()
    _add_filter_if_missing(root)

    # Install on ALL root-logger handlers (critical: catches propagated records)
    for handler in root.handlers:
        _add_filter_if_missing(handler)

    # Install on specific sub-loggers (defense-in-depth for propagate=False)
    for name in _TARGET_LOGGER_NAMES:
        sub_logger = logging.getLogger(name)
        _add_filter_if_missing(sub_logger)
        # Also install on any handlers directly on the sub-logger
        for handler in sub_logger.handlers:
            _add_filter_if_missing(handler)
