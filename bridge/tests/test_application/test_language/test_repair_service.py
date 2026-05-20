"""Tests for RepairService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from application.language.context import LanguageContext
from application.language.repair_service import RepairService
from application.language.verifier import (
    VerificationResult,
    VerificationStatus,
)


def _make_ctx(code: str = "de") -> LanguageContext:
    """Helper to create a LanguageContext."""
    return LanguageContext(
        code=code,
        source="sticky",
        confidence=1.0,
        switched_from=None,
        request_id="test-repair",
    )


def _make_failed_verification(
    expected: str = "de", detected: str = "en"
) -> VerificationResult:
    """Helper to create a failed verification result."""
    return VerificationResult(
        expected_lang=expected,
        detected_lang=detected,
        confidence=0.9,
        foreign_share=0.8,
        target_language_ratio=0.2,
        status=VerificationStatus.FAIL,
        reason=f"Expected '{expected}' but detected '{detected}'",
    )


class TestRepairServiceNoRouter:
    """Tests when no provider_router is available."""

    async def test_no_router_returns_original(self) -> None:
        """Without a router, repair returns the original output."""
        service = RepairService()
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        result = await service.repair(
            original_output="This is English text",
            ctx=ctx,
            verification_result=verification,
            provider_router=None,
        )

        assert result.repaired_output == "This is English text"
        assert result.was_repaired is False
        assert result.attempts_used == 0
        assert result.repair_failed is False


class TestRepairServiceLongOutput:
    """Tests for output length protection."""

    async def test_long_output_skips_rewrite(self) -> None:
        """Outputs longer than 5000 chars are not rewritten."""
        service = RepairService()
        ctx = _make_ctx("de")
        verification = _make_failed_verification()
        long_text = "x" * 6000

        result = await service.repair(
            original_output=long_text,
            ctx=ctx,
            verification_result=verification,
            provider_router=MagicMock(),
        )

        assert result.repaired_output == long_text
        assert result.was_repaired is False
        assert result.attempts_used == 0


class TestRepairServiceSuccess:
    """Tests for successful repair."""

    async def test_successful_repair(self) -> None:
        """Successful repair returns the new text."""
        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        # Mock provider router
        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = (
            "Dies ist der reparierte deutsche Text der jetzt korrekt "
            "in der richtigen Sprache verfasst wurde und genug Wörter "
            "enthält um die Verifikation zu bestehen. Dieser Text ist "
            "definitiv auf Deutsch geschrieben worden."
        )
        mock_response.error = None
        mock_router.route = AsyncMock(return_value=mock_response)

        result = await service.repair(
            original_output="This is English text that should be German",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        assert result.was_repaired is True
        assert result.attempts_used == 1
        assert result.repair_failed is False
        assert (
            "deutsch" in result.repaired_output.lower()
            or "reparierte" in result.repaired_output.lower()
        )


class TestRepairServiceFailure:
    """Tests for failed repair attempts."""

    async def test_repair_fails_returns_original(self) -> None:
        """When repair attempt also fails verification, return original."""
        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        # Mock: router returns English again
        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = (
            "This is still English text even after the repair attempt "
            "which means the model did not follow the instruction. "
            "The text remains in English throughout this response "
            "and should trigger a repair failure."
        )
        mock_response.error = None
        mock_router.route = AsyncMock(return_value=mock_response)

        result = await service.repair(
            original_output="Original English text",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        assert result.repaired_output == "Original English text"
        assert result.was_repaired is False
        assert result.repair_failed is True
        assert result.attempts_used == 1

    async def test_router_error_returns_original(self) -> None:
        """When router raises exception, return original."""
        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        mock_router = AsyncMock()
        mock_router.route = AsyncMock(side_effect=RuntimeError("Provider down"))

        result = await service.repair(
            original_output="Original text",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        assert result.repaired_output == "Original text"
        assert result.was_repaired is False
        assert result.repair_failed is True

    async def test_empty_response_tries_again_or_fails(self) -> None:
        """Empty response from router counts as failed attempt."""
        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = ""
        mock_response.error = None
        mock_router.route = AsyncMock(return_value=mock_response)

        result = await service.repair(
            original_output="Original text",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        assert result.repaired_output == "Original text"
        assert result.was_repaired is False


class TestRepairServiceMaxAttempts:
    """Tests for max_attempts enforcement."""

    async def test_max_attempts_capped(self) -> None:
        """max_attempts cannot exceed hard cap."""
        service = RepairService(max_attempts=99)
        assert service._max_attempts <= 2  # _HARD_MAX_ATTEMPTS

    async def test_attempts_counted_correctly(self) -> None:
        """Attempts are counted correctly."""
        service = RepairService(max_attempts=2)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        # Mock: always returns English (both attempts fail)
        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = (
            "Still English after multiple attempts at repair "
            "which demonstrates that the model is not following "
            "the language instruction despite being asked "
            "multiple times to switch to German."
        )
        mock_response.error = None
        mock_router.route = AsyncMock(return_value=mock_response)

        result = await service.repair(
            original_output="English text",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        assert result.attempts_used == 2
        assert result.repair_failed is True


class TestRepairServiceAudit:
    """Tests for audit entry generation."""

    def test_audit_entry_on_success(self) -> None:
        """Successful repair generates correct audit entry."""
        from application.language.repair_service import RepairResult

        result = RepairResult(
            original_output="English",
            repaired_output="Deutsch",
            was_repaired=True,
            attempts_used=1,
            verification_before=_make_failed_verification(),
            verification_after=VerificationResult(
                expected_lang="de",
                detected_lang="de",
                confidence=0.95,
                foreign_share=0.0,
                target_language_ratio=1.0,
                status=VerificationStatus.PASS,
                reason=None,
            ),
            latency_ms=150.0,
            repair_failed=False,
        )

        service = RepairService()
        entry = service.build_audit_entry(result)

        assert entry["event_type"] == "language_repair_succeeded"
        assert entry["target_lang"] == "de"
        assert entry["original_detected_lang"] == "en"
        assert entry["attempts_used"] == 1

    def test_audit_entry_on_failure(self) -> None:
        """Failed repair generates correct audit entry."""
        from application.language.repair_service import RepairResult

        result = RepairResult(
            original_output="English",
            repaired_output="English",
            was_repaired=False,
            attempts_used=1,
            verification_before=_make_failed_verification(),
            verification_after=_make_failed_verification(),
            latency_ms=200.0,
            repair_failed=True,
        )

        service = RepairService()
        entry = service.build_audit_entry(result)

        assert entry["event_type"] == "language_repair_failed"
