"""Tests for SecretScanner (HC-SC-13, Step 8).

AG-SC-2 [GUARD]: test_no_secret_patterns_in_hypotheses verifies the
  scanner blocks token/price/IBAN leaks in skill storage.

Covers:
  - Layer 2: Regex patterns (API tokens, prices, emails, IBANs, passwords)
  - Layer 3: Heuristic filters (long digits, user paths, third-party PII)
  - Allowlist: legitimate skill content passes
  - Integration: check_secret_content wrapper still works
"""

from __future__ import annotations

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.secret_scanner import (
    SecretScanner,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def scanner() -> SecretScanner:
    return SecretScanner()


def _hyp(claim: str) -> Hypothesis:
    """Create a test hypothesis with given claim."""
    return Hypothesis(
        hypothesis_id="hyp-secret-test",
        user_id=42,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status="candidate",
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
    )


# ---------------------------------------------------------------
# Layer 2: Regex patterns - API tokens
# ---------------------------------------------------------------


class TestSecretTokenPatterns:
    """Tests for API token/key detection."""

    def test_openai_sk_token(self, scanner: SecretScanner) -> None:
        """OpenAI-style sk- token should be detected."""
        matches = scanner.scan("use sk-1234567890abcdef1234567890abcdef")
        assert len(matches) >= 1
        assert any(m.pattern_name == "api_token" for m in matches)

    def test_github_pat(self, scanner: SecretScanner) -> None:
        """GitHub PAT should be detected."""
        matches = scanner.scan("token ghp_abcdefghijklmnopqrstuvwxyz12345678")
        assert len(matches) >= 1

    def test_bearer_token(self, scanner: SecretScanner) -> None:
        """Bearer token should be detected."""
        matches = scanner.scan("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abc")
        assert len(matches) >= 1

    def test_aws_key(self, scanner: SecretScanner) -> None:
        """AWS access key should be detected."""
        matches = scanner.scan("AKIAIOSFODNN7EXAMPLE1")
        assert len(matches) >= 1
        assert any(m.pattern_name == "aws_key" for m in matches)

    def test_generic_api_key_assignment(self, scanner: SecretScanner) -> None:
        """Generic api_key= assignment should be detected."""
        matches = scanner.scan("api_key=abc123def456ghi789jkl")
        assert len(matches) >= 1

    def test_long_hex_string(self, scanner: SecretScanner) -> None:
        """Long hex string should be detected."""
        hex_str = "a" * 32
        matches = scanner.scan(f"key is {hex_str}")
        assert len(matches) >= 1

    def test_long_base64_string(self, scanner: SecretScanner) -> None:
        """Long base64 string should be detected."""
        b64 = "A" * 40
        matches = scanner.scan(f"token: {b64}")
        assert len(matches) >= 1

    def test_url_with_credentials(self, scanner: SecretScanner) -> None:
        """URL with embedded credentials should be detected."""
        matches = scanner.scan("https://user:password@example.com/api")
        assert len(matches) >= 1


# ---------------------------------------------------------------
# Layer 2: Regex patterns - Financial data
# ---------------------------------------------------------------


class TestSecretFinancialPatterns:
    """Tests for financial data detection."""

    def test_price_euro_symbol(self, scanner: SecretScanner) -> None:
        """Euro price should be detected."""
        matches = scanner.scan("Price is 29.99 EUR")
        assert len(matches) >= 1

    def test_price_dollar_sign(self, scanner: SecretScanner) -> None:
        """Dollar price should be detected."""
        matches = scanner.scan("costs $150.00")
        assert len(matches) >= 1

    def test_price_pound(self, scanner: SecretScanner) -> None:
        """British pound price should be detected."""
        matches = scanner.scan("total: 199.99 GBP")
        assert len(matches) >= 1

    def test_iban_german(self, scanner: SecretScanner) -> None:
        """German IBAN should be detected."""
        matches = scanner.scan("IBAN: DE89 3704 0044 0532 0130 00")
        assert len(matches) >= 1

    def test_iban_austrian(self, scanner: SecretScanner) -> None:
        """Austrian IBAN should be detected."""
        matches = scanner.scan("AT61 1904 3002 3457 3201")
        assert len(matches) >= 1

    def test_tax_format(self, scanner: SecretScanner) -> None:
        """Tax format should be detected."""
        matches = scanner.scan("MwSt: 19.0%")
        assert len(matches) >= 1


# ---------------------------------------------------------------
# Layer 2: Regex patterns - Personal identifiers
# ---------------------------------------------------------------


class TestSecretPersonalIdentifiers:
    """Tests for personal identifier detection."""

    def test_email_address(self, scanner: SecretScanner) -> None:
        """Email address should be detected."""
        matches = scanner.scan("contact: user@example.com")
        assert len(matches) >= 1
        assert any(m.pattern_name == "email" for m in matches)

    def test_phone_number_international(self, scanner: SecretScanner) -> None:
        """International phone number should be detected."""
        matches = scanner.scan("Call +49 170 1234 5678")
        assert len(matches) >= 1

    def test_social_security(self, scanner: SecretScanner) -> None:
        """SSN format should be detected."""
        matches = scanner.scan("SSN: 123-45-6789")
        assert len(matches) >= 1

    def test_password_keyword(self, scanner: SecretScanner) -> None:
        """Password-adjacent content should be detected."""
        matches = scanner.scan("passwort: meinGeheim123!")
        assert len(matches) >= 1

    def test_kennwort_keyword(self, scanner: SecretScanner) -> None:
        """German 'Kennwort' should be detected."""
        matches = scanner.scan("kennwort=secretvalue123")
        assert len(matches) >= 1


# ---------------------------------------------------------------
# Layer 3: Heuristic filters
# ---------------------------------------------------------------


class TestSecretHeuristics:
    """Tests for heuristic-based detection."""

    def test_long_digit_sequence(self, scanner: SecretScanner) -> None:
        """Long digit sequence should be detected."""
        matches = scanner.scan("account 1234567890123")
        assert len(matches) >= 1
        assert any(m.pattern_name == "long_digit_sequence" for m in matches)

    def test_user_path_unix(self, scanner: SecretScanner) -> None:
        """Unix user path should be detected."""
        matches = scanner.scan("file at /home/john/documents/private.txt")
        assert len(matches) >= 1

    def test_user_path_windows(self, scanner: SecretScanner) -> None:
        """Windows user path should be detected."""
        matches = scanner.scan(r"file at C:\Users\john\Desktop\data.csv")
        assert len(matches) >= 1

    def test_third_party_name(self, scanner: SecretScanner) -> None:
        """Third-party personal name should be detected."""
        matches = scanner.scan("Herr Schmidt has specific preferences")
        assert len(matches) >= 1

    def test_third_party_mrs(self, scanner: SecretScanner) -> None:
        """Mrs. name pattern should be detected."""
        matches = scanner.scan("Mrs. Johnson asked for a different format")
        assert len(matches) >= 1


# ---------------------------------------------------------------
# Allowlist: legitimate content passes
# ---------------------------------------------------------------


class TestSecretAllowlist:
    """Tests that legitimate skill content passes."""

    def test_clean_instruction(self, scanner: SecretScanner) -> None:
        """Normal instruction should pass."""
        matches = scanner.scan("Always use bullet points in summaries")
        assert len(matches) == 0

    def test_clean_preference(self, scanner: SecretScanner) -> None:
        """Normal preference should pass."""
        matches = scanner.scan("User prefers formal tone in business emails")
        assert len(matches) == 0

    def test_clean_workflow(self, scanner: SecretScanner) -> None:
        """Workflow description should pass."""
        matches = scanner.scan("First analyze root cause, then suggest fix")
        assert len(matches) == 0

    def test_clean_german_instruction(self, scanner: SecretScanner) -> None:
        """German instruction should pass."""
        matches = scanner.scan("Verwende immer Markdown-Tabellen bei Vergleichen")
        assert len(matches) == 0

    def test_short_numbers_pass(self, scanner: SecretScanner) -> None:
        """Short numbers should pass."""
        matches = scanner.scan("Chapter 3 of 10 sections")
        assert len(matches) == 0

    def test_code_keyword_pass(self, scanner: SecretScanner) -> None:
        """Code-related text without actual secrets should pass."""
        matches = scanner.scan("User prefers Python type hints everywhere")
        assert len(matches) == 0


# ---------------------------------------------------------------
# Hypothesis-level check
# ---------------------------------------------------------------


class TestSecretHypothesisBlock:
    """Tests for hypothesis-level blocking."""

    def test_block_hypothesis_with_token(self, scanner: SecretScanner) -> None:
        """Hypothesis with API token should be blocked."""
        h = _hyp("Use sk-1234567890abcdef1234567890ab as default key")
        assert scanner.block_if_secrets(h) is True

    def test_pass_clean_hypothesis(self, scanner: SecretScanner) -> None:
        """Clean hypothesis should pass."""
        h = _hyp("User prefers markdown tables")
        assert scanner.block_if_secrets(h) is False

    def test_block_reason_with_token(self, scanner: SecretScanner) -> None:
        """get_block_reason should return description for blocked items."""
        h = _hyp("passwort: secret123value")
        reason = scanner.get_block_reason(h)
        assert reason is not None
        assert "Passwort" in reason

    def test_block_reason_none_for_clean(self, scanner: SecretScanner) -> None:
        """get_block_reason should return None for clean items."""
        h = _hyp("User prefers bullet points")
        reason = scanner.get_block_reason(h)
        assert reason is None


# ---------------------------------------------------------------
# Integration: check_secret_content wrapper
# ---------------------------------------------------------------


class TestCheckSecretContentWrapper:
    """Tests that the skill_commands wrapper still works."""

    def test_wrapper_detects_token(self) -> None:
        """Wrapper should detect API tokens."""
        from presentation.skill_commands import check_secret_content

        result = check_secret_content("use sk-1234567890abcdef1234567890abcdef")
        assert result is not None

    def test_wrapper_passes_clean(self) -> None:
        """Wrapper should pass clean text."""
        from presentation.skill_commands import check_secret_content

        result = check_secret_content("User prefers bullet points")
        assert result is None

    def test_wrapper_detects_email(self) -> None:
        """Wrapper should detect email addresses."""
        from presentation.skill_commands import check_secret_content

        result = check_secret_content("user@example.com preference")
        assert result is not None

    def test_wrapper_detects_iban(self) -> None:
        """Wrapper should detect IBANs."""
        from presentation.skill_commands import check_secret_content

        result = check_secret_content("IBAN: DE89 3704 0044 0532 0130 00")
        assert result is not None
