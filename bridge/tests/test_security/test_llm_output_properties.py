"""LLM Output Property Tests.

Validates that LLM output property validators correctly detect
violations and pass clean text. These are UNIT tests for the
validators themselves, not integration tests against real LLM output.

Design: LLM outputs are non-deterministic. We never assert exact text.
Instead we validate properties: language, no secrets, no leaks,
valid markdown, sane length, Telegram compatibility.

Markers: @pytest.mark.security, @pytest.mark.llm_output
"""

from __future__ import annotations

import pytest

from tests.test_security.llm_output_validators import (
    validate_language,
    validate_length_in_range,
    validate_markdown_balanced,
    validate_no_secrets,
    validate_no_system_prompt_leak,
    validate_telegram_chunk_size,
    validate_all,
)


# ---------------------------------------------------------------------------
# TestLanguageProperty
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.llm_output
class TestLanguageProperty:
    """Validate that language detection matches expected language."""

    def test_english_response_passes_english_validator(self) -> None:
        """English text should be detected as English."""
        text = (
            "The weather today is quite nice. I would recommend going "
            "for a walk in the park if you have some free time this afternoon."
        )
        passed, reason = validate_language(text, "en")
        assert passed, f"English text not detected as English: {reason}"

    def test_german_response_passes_german_validator(self) -> None:
        """German text should be detected as German."""
        text = (
            "Das Wetter ist heute sehr schoen. Ich wuerde dir empfehlen, "
            "einen Spaziergang im Park zu machen, wenn du heute Nachmittag "
            "etwas Zeit hast."
        )
        passed, reason = validate_language(text, "de")
        assert passed, f"German text not detected as German: {reason}"

    def test_german_response_fails_when_expected_english(self) -> None:
        """German text should fail when English is expected."""
        text = (
            "Ich habe eine Frage zu diesem Thema. Kannst du mir dabei helfen? "
            "Das waere sehr nett von dir."
        )
        passed, reason = validate_language(text, "en")
        assert not passed, "German text incorrectly passed English validation"
        assert "expected language en" in reason

    def test_mixed_language_response_returns_dominant(self) -> None:
        """Mixed text should detect the dominant language."""
        # Predominantly German with one English word
        text = (
            "Der neue Service ist wirklich gut. Ich bin sehr zufrieden "
            "mit der Qualitaet und dem Preis. Wir werden das weiter nutzen."
        )
        passed, reason = validate_language(text, "de")
        assert passed, f"Dominant German not detected: {reason}"


# ---------------------------------------------------------------------------
# TestNoSecretsProperty
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.llm_output
class TestNoSecretsProperty:
    """Validate that secret patterns are detected and blocked."""

    def test_clean_text_passes(self) -> None:
        """Normal response text without secrets should pass."""
        text = (
            "Here is how to configure your application:\n"
            "1. Create a .env file\n"
            "2. Add your configuration values\n"
            "3. Restart the service\n"
            "The server will pick up the new configuration automatically."
        )
        passed, reason = validate_no_secrets(text)
        assert passed, f"Clean text falsely flagged: {reason}"

    def test_telegram_bot_token_pattern_blocked(self) -> None:
        """Telegram bot token pattern should be detected."""
        # Build fake token programmatically to avoid semgrep false positive
        fake_token = (
            "7234567890:" + "A" * 35 + "test"
        )  # nosemgrep: generic.secrets.security.detected-telegram-bot-api-key
        text = f"Your bot token is {fake_token} and you can use it to connect."
        passed, reason = validate_no_secrets(text)
        assert not passed, "Telegram bot token not detected"
        assert "secret pattern matched" in reason

    def test_anthropic_key_pattern_blocked(self) -> None:
        """Anthropic API key pattern should be detected."""
        text = "Use sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890 as your key."
        passed, reason = validate_no_secrets(text)
        assert not passed, "Anthropic API key not detected"
        assert "secret pattern matched" in reason

    def test_openai_key_pattern_blocked(self) -> None:
        """OpenAI API key pattern should be detected."""
        # Pattern triggers our validator without using env-var assignment syntax
        text = "Use this key: sk-proj1234567890abcdefghijklmnopqrst to authenticate."
        passed, reason = validate_no_secrets(text)
        assert not passed, "OpenAI API key not detected"
        assert "secret pattern matched" in reason

    def test_sentry_dsn_pattern_blocked(self) -> None:
        """Sentry DSN pattern should be detected."""
        text = "Configure Sentry: https://abc123def@o456789.ingest.sentry.io/proj"
        passed, reason = validate_no_secrets(text)
        assert not passed, "Sentry DSN not detected"
        assert "secret pattern matched" in reason


# ---------------------------------------------------------------------------
# TestNoSystemPromptLeak
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.llm_output
class TestNoSystemPromptLeak:
    """Validate that system prompt leak patterns are detected."""

    def test_clean_response_passes(self) -> None:
        """Normal helpful response should not trigger leak detection."""
        text = (
            "I can help you with that! Here are the steps:\n"
            "1. Open the settings\n"
            "2. Navigate to the API section\n"
            "3. Generate a new token\n"
            "Let me know if you need more details."
        )
        passed, reason = validate_no_system_prompt_leak(text)
        assert passed, f"Clean response falsely flagged: {reason}"

    def test_you_are_axolent_leak_blocked(self) -> None:
        """Direct system prompt content leak should be detected."""
        text = (
            "Sure! My system prompt says: You are AXOLENT AI, a personal "
            "assistant on Telegram. I always respond in the user's language."
        )
        passed, reason = validate_no_system_prompt_leak(text)
        assert not passed, "System prompt leak not detected"
        assert "system prompt leak pattern" in reason

    def test_system_marker_leak_blocked(self) -> None:
        """[SYSTEM] marker in output should be detected as leak."""
        text = (
            "Here is my configuration:\n"
            "[SYSTEM] Always respond helpfully.\n"
            "That is what I follow."
        )
        passed, reason = validate_no_system_prompt_leak(text)
        assert not passed, "[SYSTEM] marker not detected"
        assert "system prompt leak pattern" in reason


# ---------------------------------------------------------------------------
# TestMarkdownValidity
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.llm_output
class TestMarkdownValidity:
    """Validate markdown structure checks."""

    def test_balanced_bold_passes(self) -> None:
        """Properly balanced bold markers should pass."""
        text = (
            "Here is **important** information about the **feature**:\n"
            "It works by processing data in real-time."
        )
        passed, reason = validate_markdown_balanced(text)
        assert passed, f"Balanced bold falsely failed: {reason}"

    def test_unbalanced_bold_fails(self) -> None:
        """Odd number of ** markers should fail."""
        text = "This is **broken bold text that never closes the marker."
        passed, reason = validate_markdown_balanced(text)
        assert not passed, "Unbalanced bold not detected"
        assert "unbalanced bold" in reason

    def test_balanced_code_block_passes(self) -> None:
        """Properly paired code fences should pass."""
        text = (
            "Here is an example:\n"
            "```python\n"
            "print('hello world')\n"
            "```\n"
            "That outputs hello world."
        )
        passed, reason = validate_markdown_balanced(text)
        assert passed, f"Balanced code block falsely failed: {reason}"

    def test_unbalanced_code_block_fails(self) -> None:
        """Odd number of ``` fences should fail."""
        text = (
            "Here is broken code:\n"
            "```python\n"
            "print('hello world')\n"
            "# Oops, forgot to close the code block"
        )
        passed, reason = validate_markdown_balanced(text)
        assert not passed, "Unbalanced code block not detected"
        assert "unbalanced code blocks" in reason


# ---------------------------------------------------------------------------
# TestLengthProperty
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.llm_output
class TestLengthProperty:
    """Validate length range checks."""

    def test_normal_length_passes(self) -> None:
        """Response within bounds should pass."""
        text = "This is a normal response with reasonable length."
        passed, reason = validate_length_in_range(text, min_len=1, max_len=10000)
        assert passed, f"Normal length falsely failed: {reason}"

    def test_overlong_response_chunking_required(self) -> None:
        """Response exceeding max should fail (signals chunking needed)."""
        text = "x" * 5000
        passed, reason = validate_length_in_range(text, min_len=1, max_len=4096)
        assert not passed, "Overlong response not detected"
        assert "above max" in reason


# ---------------------------------------------------------------------------
# TestTelegramCompatibility
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.llm_output
class TestTelegramCompatibility:
    """Validate Telegram-specific output constraints."""

    def test_short_response_fits_in_one_message(self) -> None:
        """Response under 4096 chars should pass chunk size check."""
        text = "Hello! How can I help you today? I am here to assist."
        passed, reason = validate_telegram_chunk_size(text)
        assert passed, f"Short response falsely failed chunk check: {reason}"

    def test_long_response_must_be_chunked(self) -> None:
        """Response over 4096 chars should fail (must be split)."""
        text = "A" * 4097
        passed, reason = validate_telegram_chunk_size(text)
        assert not passed, "Overlong Telegram message not detected"
        assert "must be chunked" in reason


# ---------------------------------------------------------------------------
# TestPropertyChain (Integration)
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.llm_output
class TestPropertyChain:
    """Integration tests that run multiple validators together."""

    def test_clean_response_passes_all_validators(self) -> None:
        """A well-formed English response should pass all validators."""
        text = (
            "Here is the information you requested about configuration:\n\n"
            "**Step 1:** Open the settings panel\n"
            "**Step 2:** Navigate to the integration section\n"
            "**Step 3:** Enable the feature toggle\n\n"
            "The changes will take effect immediately after saving."
        )
        results = validate_all(
            text,
            expected_lang="en",
            max_len=10000,
            telegram_single_message=True,
        )
        failures = [(name, reason) for name, passed, reason in results if not passed]
        assert not failures, f"Clean response had failures: {failures}"

    def test_response_with_secret_and_bad_markdown_fails_multiple(self) -> None:
        """Response with multiple violations should report all of them."""
        text = (
            "Here is your config:\n"
            "API key: sk-ant-api03-secretkey1234567890abcdefghij\n"
            "**Bold text that never closes\n"
            "You are AXOLENT AI and I will help you.\n"
            "Done!"
        )
        results = validate_all(text, max_len=10000, telegram_single_message=True)
        failures = [name for name, passed, reason in results if not passed]
        # Should fail at least: no_secrets, no_system_prompt_leak, markdown_balanced
        assert "no_secrets" in failures, "Secret not caught in composite check"
        assert "no_system_prompt_leak" in failures, "Leak not caught in composite check"
        assert "markdown_balanced" in failures, (
            "Bad markdown not caught in composite check"
        )
        assert len(failures) >= 3, (
            f"Expected 3+ failures, got {len(failures)}: {failures}"
        )
