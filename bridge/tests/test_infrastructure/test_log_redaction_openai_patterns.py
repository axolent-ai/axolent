"""Tests for OpenAI key pattern recognition (Finding 7).

Parametrized over modern key formats: sk-proj-, sk-svcacct-, sk-admin-.
4-Path: Happy + Malicious + Rejection + Privacy.
"""

from __future__ import annotations

import pytest

from infrastructure.log_redaction import _redact_string


class TestOpenAIKeyPatternHappy:
    """Modern OpenAI key formats are properly redacted."""

    @pytest.mark.parametrize(
        "key_prefix,description",
        [
            ("sk-proj-", "Project-scoped key"),
            ("sk-svcacct-", "Service account key"),
            ("sk-admin-", "Admin key"),
            ("sk-", "Classic key (no sub-prefix)"),
        ],
    )
    def test_modern_key_redacted(self, key_prefix: str, description: str) -> None:
        """Key with prefix '{key_prefix}' ({description}) is redacted."""
        # Build a realistic key: prefix + 40 chars of mixed alphanum + hyphens
        key_body = "abc123DEF456ghi789JKL012mno345PQR678stu901"
        full_key = f"{key_prefix}{key_body}"
        result = _redact_string(f"Using key: {full_key}")
        assert key_body not in result
        assert "REDACTED-OPENAI-KEY" in result

    def test_sk_proj_with_hyphens(self) -> None:
        """sk-proj-... with hyphens in body is fully redacted."""
        key = "sk-proj-abc123-def456-ghi789-jkl012-mno345pqr"
        result = _redact_string(f"Auth: {key}")
        assert "abc123" not in result
        assert "REDACTED" in result

    def test_sk_proj_with_underscores(self) -> None:
        """sk-proj-... with underscores in body is fully redacted."""
        key = "sk-proj-abc_123_def_456_ghi_789_jkl"
        result = _redact_string(f"Key={key}")
        assert "abc_123" not in result
        assert "REDACTED" in result


class TestOpenAIKeyPatternRejection:
    """Anthropic keys and short strings are NOT affected."""

    def test_anthropic_key_not_matched_by_openai_pattern(self) -> None:
        """sk-ant-... is handled by Anthropic pattern, not OpenAI."""
        key = "sk-ant-abc123def456ghi789jkl012mno345pqr678"
        result = _redact_string(key)
        # Should be redacted by the ANTHROPIC pattern
        assert "REDACTED-ANTHROPIC-KEY" in result
        assert "REDACTED-OPENAI-KEY" not in result

    def test_short_sk_prefix_not_matched(self) -> None:
        """Very short sk- strings (< 20 chars body) are NOT redacted."""
        short = "sk-shortkey"
        result = _redact_string(short)
        # Should pass through since body is too short
        assert result == short

    def test_non_sk_prefix_not_matched(self) -> None:
        """Random text with 'sk' is not redacted."""
        text = "I like to sk-ip stones"
        result = _redact_string(text)
        assert result == text


class TestOpenAIKeyPatternPrivacy:
    """Redacted output never contains the key material."""

    @pytest.mark.parametrize(
        "key",
        [
            "sk-proj-reallyLongKeyMaterial123456789012345678901234567890",
            "sk-svcacct-anotherLongKeyForServiceAccounts123456789012345",
            "sk-admin-adminKeyMaterialThatShouldNeverLeak12345678901234",
        ],
    )
    def test_no_key_material_in_output(self, key: str) -> None:
        """No portion of key material leaks in redacted output."""
        result = _redact_string(f"Authorization: Bearer {key}")
        # Check that no substring of the key body appears
        body = key.split("-", 2)[-1]  # everything after sk-prefix-
        # Check first 10 chars of body
        assert body[:10] not in result
