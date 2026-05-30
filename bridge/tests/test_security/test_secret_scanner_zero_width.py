"""Tests for SecretScanner Zero-Width bypass (Finding 8, secondary).

Verifies that Zero-Width characters cannot bypass secret detection.
4-Path: Happy + Malicious + Rejection + Privacy.
"""

from __future__ import annotations

import pytest

from application.security.secret_scanner import SecretScanner


@pytest.fixture
def scanner() -> SecretScanner:
    return SecretScanner()


class TestSecretScannerZeroWidth:
    """Zero-Width chars inserted into secrets must still be detected."""

    @pytest.mark.parametrize(
        "zw_char,name",
        [
            ("​", "ZERO WIDTH SPACE"),
            ("‌", "ZERO WIDTH NON-JOINER"),
            ("‍", "ZERO WIDTH JOINER"),
            ("﻿", "BOM"),
            ("⁠", "WORD JOINER"),
        ],
    )
    def test_anthropic_key_with_zero_width(
        self, scanner: SecretScanner, zw_char: str, name: str
    ) -> None:
        """Anthropic key with ZW char is still detected."""
        # Insert ZW between 'sk-ant-' and the rest
        key = f"sk-ant-{zw_char}abc123def456ghi789jkl012mno345"
        matches = scanner.scan(key)
        assert len(matches) > 0, (
            f"Zero-Width ({name}) bypass: SecretScanner failed to detect key"
        )

    @pytest.mark.parametrize(
        "zw_char,name",
        [
            ("​", "ZERO WIDTH SPACE"),
            ("‌", "ZERO WIDTH NON-JOINER"),
            ("‍", "ZERO WIDTH JOINER"),
        ],
    )
    def test_email_with_zero_width(
        self, scanner: SecretScanner, zw_char: str, name: str
    ) -> None:
        """Email with ZW char is still detected."""
        email = f"user{zw_char}@example.com"
        matches = scanner.scan(email)
        assert len(matches) > 0, (
            f"Zero-Width ({name}) bypass: SecretScanner failed to detect email"
        )

    def test_password_with_multiple_zw(self, scanner: SecretScanner) -> None:
        """Password pattern with multiple ZW chars is detected."""
        text = "pass​wor‌d: my​secret​123"
        matches = scanner.scan(text)
        assert len(matches) > 0

    def test_clean_text_not_blocked(self, scanner: SecretScanner) -> None:
        """Normal text without secrets passes."""
        matches = scanner.scan("I prefer dark mode")
        assert len(matches) == 0
