"""Tests for LanguageEnforcement integration facade."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from application.language.context import LanguageContext
from application.language.enforcement import LanguageEnforcement


def _make_ctx(code: str = "de") -> LanguageContext:
    """Helper to create a LanguageContext."""
    return LanguageContext(
        code=code,
        source="sticky",
        confidence=1.0,
        switched_from=None,
        request_id="test-enforce",
    )


class TestEnforcementNormalProfile:
    """Tests with normal enforcement (Claude Opus/Sonnet)."""

    async def test_normal_profile_skips_verification(self) -> None:
        """Normal profile does not run verification."""
        enforcement = LanguageEnforcement()
        ctx = _make_ctx("de")

        result = await enforcement.enforce(
            output="Some text",
            ctx=ctx,
            model_id="claude-opus-4-7",
        )

        assert result.final_output == "Some text"
        assert result.verification is None
        assert result.repair is None
        assert result.was_enforced is False
        assert result.model_profile.enforcement_level == "normal"


class TestEnforcementStrictProfile:
    """Tests with strict enforcement (Gemini, Mistral)."""

    async def test_strict_profile_verifies(self) -> None:
        """Strict profile runs verification."""
        enforcement = LanguageEnforcement()
        ctx = _make_ctx("de")

        # German text should pass
        german_text = (
            "Dies ist ein deutscher Text der lang genug ist um die "
            "Verifikation zu bestehen. Er enthält genug Wörter um "
            "eine zuverlässige Spracherkennung zu ermöglichen "
            "und sollte als Deutsch erkannt werden."
        )

        result = await enforcement.enforce(
            output=german_text,
            ctx=ctx,
            model_id="gemini-2.0-flash",
        )

        assert result.final_output == german_text
        assert result.verification is not None
        assert result.was_enforced is False

    async def test_strict_profile_short_text_passes(self) -> None:
        """Short text passes verification (skipped)."""
        enforcement = LanguageEnforcement()
        ctx = _make_ctx("de")

        result = await enforcement.enforce(
            output="Ja, stimmt.",
            ctx=ctx,
            model_id="gemini-2.0-flash",
        )

        assert result.final_output == "Ja, stimmt."
        assert result.verification is not None
        assert result.verification.skipped is True


class TestEnforcementStrictWithVerifyProfile:
    """Tests with strict_with_verify enforcement (Haiku, Llama)."""

    async def test_wrong_language_triggers_repair(self) -> None:
        """Wrong language in output triggers repair attempt."""
        # Mock the provider router for repair
        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = (
            "Dies ist der reparierte Text der jetzt auf Deutsch "
            "geschrieben ist und die Verifikation bestehen sollte "
            "weil er genug deutsche Wörter enthält um als Deutsch "
            "erkannt zu werden vom Spracherkennungssystem."
        )
        mock_response.error = None
        mock_router.route = AsyncMock(return_value=mock_response)

        enforcement = LanguageEnforcement(provider_router=mock_router)
        ctx = _make_ctx("de")

        english_text = (
            "This is an English response that should not have been "
            "generated because the user expects German. The model "
            "has drifted to English which is a common problem with "
            "smaller models that need strict enforcement."
        )

        result = await enforcement.enforce(
            output=english_text,
            ctx=ctx,
            model_id="llama-3.1-8b",
        )

        # Verification should have failed
        assert result.verification is not None
        # Repair should have been attempted
        if result.verification.passed is False:
            assert result.repair is not None
            assert result.was_enforced is True


class TestEnforcementAudit:
    """Tests for audit event emission via AuditLogPort (Codex Finding 5)."""

    async def test_audit_emitted_on_verification(self) -> None:
        """Audit event is written when verification runs via injected port."""
        audit_calls: list[dict] = []

        def mock_audit(entry: dict) -> None:
            audit_calls.append(entry)

        enforcement = LanguageEnforcement(audit_log=mock_audit)
        ctx = _make_ctx("de")

        german_text = (
            "Dies ist ein deutscher Text der lang genug ist um die "
            "Verifikation zu bestehen und einen Audit-Eintrag "
            "auszulösen der dann in den Logs erscheint "
            "damit wir die Sprachkontrolle überwachen können."
        )

        await enforcement.enforce(
            output=german_text,
            ctx=ctx,
            model_id="gemini-2.0-flash",
            request_id="audit-test-123",
        )

        # At least one audit call for verification
        assert len(audit_calls) > 0
        assert audit_calls[0]["event_type"] == "language_verification_performed"
        assert audit_calls[0]["request_id"] == "audit-test-123"
