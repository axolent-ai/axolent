"""LLM Output Property Validators.

Pure functions that validate properties of LLM responses.
Used by property-based and golden-corpus tests.

Design rationale: LLM outputs are non-deterministic. Exact string matching
is brittle and produces false failures. Instead, we validate PROPERTIES
of the output: language correctness, absence of secrets, no system-prompt
leakage, valid markdown structure, sane length, and Telegram compatibility.

Each validator returns (passed: bool, reason: str).
When passed=True, reason is always "".
When passed=False, reason explains the violation.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------


def validate_language(text: str, expected_lang: str) -> tuple[bool, str]:
    """Validate that detected language matches expected language.

    Uses the production domain.language.detect_language function,
    ensuring test assertions match real routing behavior.

    Args:
        text: LLM response text.
        expected_lang: ISO-639-1 code (e.g. "en", "de").

    Returns:
        (passed, reason) tuple.
    """
    from domain.language import detect_language

    detected = detect_language(text)
    if detected != expected_lang:
        return (False, f"expected language {expected_lang}, detected {detected}")
    return (True, "")


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"\d+:[A-Za-z0-9_-]{35,}"),  # Telegram bot token
    re.compile(r"sk-ant-[A-Za-z0-9_-]+"),  # Anthropic API key
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI API key
    re.compile(r"gsk_[A-Za-z0-9]{30,}"),  # Groq API key
    re.compile(r"https://[a-z0-9]+@o[0-9]+\.ingest\.sentry"),  # Sentry DSN
    re.compile(r"AKIA[A-Z0-9]{16}"),  # AWS access key
    re.compile(r"gh[pous]_[A-Za-z0-9]{20,}"),  # GitHub token
]


def validate_no_secrets(text: str) -> tuple[bool, str]:
    """Check that no secret/token patterns appear in the text.

    Covers: Telegram bot tokens, Anthropic keys, OpenAI keys,
    Groq keys, Sentry DSNs, AWS access keys, GitHub tokens.

    Args:
        text: LLM response text.

    Returns:
        (passed, reason) tuple.
    """
    for pattern in _SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            return (False, f"secret pattern matched: {pattern.pattern[:40]}...")
    return (True, "")


# ---------------------------------------------------------------------------
# System Prompt Leak
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_LEAK_PATTERNS = [
    re.compile(r"You are AXOLENT", re.IGNORECASE),
    re.compile(r"system_prompt", re.IGNORECASE),
    re.compile(r"\[SYSTEM\]", re.IGNORECASE),
    re.compile(r"<\|im_start\|>system"),
    re.compile(r"^Instructions:.*follow", re.IGNORECASE | re.MULTILINE),
]


def validate_no_system_prompt_leak(text: str) -> tuple[bool, str]:
    """Check that the response does not leak system prompt content.

    Detects common patterns that indicate system-prompt exfiltration:
    - Direct mentions of "You are AXOLENT"
    - References to system_prompt variable
    - [SYSTEM] markers
    - ChatML special tokens
    - "Instructions:" preambles

    Args:
        text: LLM response text.

    Returns:
        (passed, reason) tuple.
    """
    for pattern in _SYSTEM_PROMPT_LEAK_PATTERNS:
        if pattern.search(text):
            return (False, f"system prompt leak pattern: {pattern.pattern[:40]}")
    return (True, "")


# ---------------------------------------------------------------------------
# Markdown Validity
# ---------------------------------------------------------------------------


def validate_markdown_balanced(text: str) -> tuple[bool, str]:
    """Check that markdown formatting markers are balanced.

    Validates:
    - Even number of ``` (code blocks)
    - Even number of ** (bold markers)
    - Equal count of [ and ] (link brackets)

    Note: This is a lightweight structural check, not a full
    markdown parser. It catches the most common LLM formatting errors.

    Args:
        text: LLM response text.

    Returns:
        (passed, reason) tuple.
    """
    code_blocks = text.count("```")
    if code_blocks % 2 != 0:
        return (False, f"unbalanced code blocks: {code_blocks}")

    bold = text.count("**")
    if bold % 2 != 0:
        return (False, f"unbalanced bold: {bold} ** markers")

    # Simple bracket check (ignoring those inside code blocks)
    open_brackets = text.count("[")
    close_brackets = text.count("]")
    if open_brackets != close_brackets:
        return (
            False,
            f"unbalanced brackets: {open_brackets} open vs {close_brackets} close",
        )

    return (True, "")


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


def validate_length_in_range(
    text: str, min_len: int = 1, max_len: int = 1_000_000
) -> tuple[bool, str]:
    """Check that response length is within acceptable bounds.

    Default bounds: at least 1 character, at most 1M characters.
    Callers should pass stricter bounds for their use case.

    Args:
        text: LLM response text.
        min_len: Minimum acceptable length (inclusive).
        max_len: Maximum acceptable length (inclusive).

    Returns:
        (passed, reason) tuple.
    """
    length = len(text)
    if length < min_len:
        return (False, f"length {length} below min {min_len}")
    if length > max_len:
        return (False, f"length {length} above max {max_len}")
    return (True, "")


# ---------------------------------------------------------------------------
# Telegram Compatibility
# ---------------------------------------------------------------------------

TELEGRAM_MAX_MESSAGE_CHARS = 4096


def validate_telegram_chunk_size(
    text: str, max_chunk_size: int = TELEGRAM_MAX_MESSAGE_CHARS
) -> tuple[bool, str]:
    """Check that a single message does not exceed Telegram's limit.

    Telegram Bot API rejects messages longer than 4096 characters.
    If the text exceeds this, it MUST be chunked before sending.

    Args:
        text: LLM response text (single message, not pre-chunked).
        max_chunk_size: Maximum chars per message (default: 4096).

    Returns:
        (passed, reason) tuple.
    """
    if len(text) > max_chunk_size:
        return (
            False,
            f"single message {len(text)} > {max_chunk_size} chars (must be chunked)",
        )
    return (True, "")


def validate_no_telegram_bot_command_inline(text: str) -> tuple[bool, str]:
    """Check that no unsanitized /commands appear that Telegram would auto-link.

    Telegram auto-links patterns like /start, /help, /reset when they appear
    at the start of a line. LLM outputs should use sanitize_telegram_slashes()
    before sending. This validator catches missed sanitization.

    Args:
        text: LLM response text.

    Returns:
        (passed, reason) tuple.
    """
    pattern = re.compile(r"^/[a-zA-Z][a-zA-Z0-9_]+", re.MULTILINE)
    matches = pattern.findall(text)
    if matches:
        return (False, f"unsanitized telegram commands: {matches[:3]}")
    return (True, "")


# ---------------------------------------------------------------------------
# Composite: run all validators
# ---------------------------------------------------------------------------


def validate_all(
    text: str,
    expected_lang: Optional[str] = None,
    max_len: int = 1_000_000,
    telegram_single_message: bool = False,
) -> list[tuple[str, bool, str]]:
    """Run all validators and return results.

    Convenience function for integration tests that want to check
    multiple properties at once.

    Args:
        text: LLM response text.
        expected_lang: If provided, validates language detection.
        max_len: Maximum acceptable length.
        telegram_single_message: If True, checks Telegram chunk size.

    Returns:
        List of (validator_name, passed, reason) tuples.
    """
    results: list[tuple[str, bool, str]] = []

    if expected_lang:
        passed, reason = validate_language(text, expected_lang)
        results.append(("language", passed, reason))

    passed, reason = validate_no_secrets(text)
    results.append(("no_secrets", passed, reason))

    passed, reason = validate_no_system_prompt_leak(text)
    results.append(("no_system_prompt_leak", passed, reason))

    passed, reason = validate_markdown_balanced(text)
    results.append(("markdown_balanced", passed, reason))

    passed, reason = validate_length_in_range(text, min_len=1, max_len=max_len)
    results.append(("length_in_range", passed, reason))

    if telegram_single_message:
        passed, reason = validate_telegram_chunk_size(text)
        results.append(("telegram_chunk_size", passed, reason))

    passed, reason = validate_no_telegram_bot_command_inline(text)
    results.append(("no_telegram_bot_command", passed, reason))

    return results
