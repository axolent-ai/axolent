"""Tests for RepairService."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from application.language.context import LanguageContext
from application.language.repair_service import RepairService, _REPAIR_TIMEOUT_SECONDS
from application.language.verifier import (
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    import pytest


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


class TestRepairServiceLongOutputAudit:
    """Issue 3: long output skip produces distinct audit event."""

    async def test_long_output_has_skipped_reason(self) -> None:
        """Outputs >5000 chars get skipped_reason='output_too_long'."""
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

        assert result.skipped_reason == "output_too_long"
        assert result.was_repaired is False
        assert result.repair_failed is False

    def test_long_output_audit_entry_has_distinct_event_type(self) -> None:
        """Audit entry for long-output skip uses language_repair_skipped_too_long."""
        from application.language.repair_service import RepairResult

        result = RepairResult(
            original_output="x" * 6000,
            repaired_output="x" * 6000,
            was_repaired=False,
            attempts_used=0,
            verification_before=_make_failed_verification(),
            verification_after=None,
            latency_ms=0.1,
            repair_failed=False,
            skipped_reason="output_too_long",
        )

        service = RepairService()
        entry = service.build_audit_entry(result)

        assert entry["event_type"] == "language_repair_skipped_too_long"
        assert entry["skipped_reason"] == "output_too_long"
        assert entry["output_length"] == 6000

    def test_normal_skip_has_no_skipped_reason(self) -> None:
        """Normal repair skip (no router) has skipped_reason=None."""
        from application.language.repair_service import RepairResult

        result = RepairResult(
            original_output="short",
            repaired_output="short",
            was_repaired=False,
            attempts_used=0,
            verification_before=_make_failed_verification(),
            verification_after=None,
            latency_ms=0.0,
            repair_failed=False,
        )

        assert result.skipped_reason is None
        service = RepairService()
        entry = service.build_audit_entry(result)
        assert entry["event_type"] == "language_repair_skipped"
        assert "skipped_reason" not in entry


class TestRepairServiceTimeout:
    """Issue 4: repair re-query uses explicit timeout."""

    async def test_repair_passes_timeout_to_provider(self) -> None:
        """RepairService passes timeout_seconds=15 to provider_router.route()."""

        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = (
            "Dies ist der reparierte deutsche Text der jetzt korrekt "
            "in der richtigen Sprache verfasst wurde und genug Woerter "
            "enthaelt um die Verifikation zu bestehen. Dieser Text ist "
            "definitiv auf Deutsch geschrieben worden."
        )
        mock_response.error = None
        mock_router.route = AsyncMock(return_value=mock_response)

        await service.repair(
            original_output="This is English text that should be German",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        # Assert route() was called with timeout_seconds=15
        mock_router.route.assert_called_once()
        call_kwargs = mock_router.route.call_args
        assert call_kwargs.kwargs.get("timeout_seconds") == _REPAIR_TIMEOUT_SECONDS
        assert _REPAIR_TIMEOUT_SECONDS == 15


class TestRepairServiceProviderName:
    """Issue 2: repair uses provider_name, not model ID."""

    async def test_provider_name_passed_through(self) -> None:
        """provider_name argument is forwarded to provider_router.route()."""
        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = (
            "Dies ist der reparierte deutsche Text der jetzt korrekt "
            "in der richtigen Sprache verfasst wurde und genug Woerter "
            "enthaelt um die Verifikation zu bestehen. Dieser Text ist "
            "definitiv auf Deutsch geschrieben worden."
        )
        mock_response.error = None
        mock_router.route = AsyncMock(return_value=mock_response)

        await service.repair(
            original_output="This is English",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
            provider_name="claude_persistent",
            model="claude-sonnet-4-6",
        )

        call_kwargs = mock_router.route.call_args
        # provider_name must be the actual provider, not the model ID
        assert call_kwargs.kwargs.get("provider_name") == "claude_persistent"
        # model must be the model ID
        assert call_kwargs.kwargs.get("model") == "claude-sonnet-4-6"


class TestRepairCancelStaleOutputProtection:
    """Edge case: repair timeout + stale output from previous cancelled call.

    Scenario:
    1. User sends message, repair path triggers.
    2. Provider hangs >15s (e.g. Claude subprocess is slow).
    3. asyncio.wait_for cancels the repair after _REPAIR_TIMEOUT_SECONDS.
    4. User quickly sends another message on the same (user, chat, model) key.
    5. WITHOUT the fix: the next call reads stale output from the cancelled request.
    6. WITH the fix: the pool marks the process as dirty on incomplete stream,
       get_or_create recycles it, and the next call gets fresh output.

    This test verifies the protection end-to-end through RepairService.
    """

    async def test_stale_output_not_leaked_after_repair_timeout(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        """After repair timeout, subsequent call must not read stale output."""
        import application.language.repair_service as repair_mod

        monkeypatch.setattr(repair_mod, "_REPAIR_TIMEOUT_SECONDS", 2)

        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        # Track call count to differentiate first (hanging) vs second (fast) call
        call_count = 0

        async def _mock_route(**kwargs):  # noqa: ANN003
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: hang forever (will be cancelled by wait_for)
                await asyncio.sleep(60)
                # Should never reach here
                return MagicMock(success=True, text="stale-content-XYZ", error=None)
            # Second call: return fresh content quickly
            return MagicMock(
                success=True,
                text=(
                    "Dies ist frischer deutscher Text der nach dem "
                    "Timeout korrekt geliefert wird und die Verifikation "
                    "besteht weil er auf Deutsch geschrieben wurde."
                ),
                error=None,
            )

        mock_router = MagicMock()
        mock_router.route = _mock_route

        # First repair: will timeout after 2s
        result1 = await service.repair(
            original_output="This is English text",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        # Must have timed out (repair failed, original returned)
        assert result1.was_repaired is False
        assert result1.repair_failed is True
        assert call_count == 1

        # Second repair: must succeed with fresh content, NOT stale
        result2 = await service.repair(
            original_output="Another English text",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )

        assert call_count == 2
        # The critical assertion: second call must have gotten fresh
        # content, not stale content from the first cancelled call.
        assert result2.was_repaired is True
        assert "stale-content-XYZ" not in result2.repaired_output
        assert "deutsch" in result2.repaired_output.lower()


class TestRepairServiceTimeoutEnforced:
    """Codex Finding 2: repair timeout MUST be enforced via asyncio.wait_for.

    Before the fix, RepairService passed timeout_seconds=15 to
    provider_router.route(), but ClaudePersistentProvider.generate()
    ignored that parameter. The underlying ClaudeProcessPool had
    a hardcoded 120s timeout. So a real repair could hang for ~120s.

    The fix wraps the entire provider_router.route() call in
    asyncio.wait_for(timeout=_REPAIR_TIMEOUT_SECONDS) so the repair
    call is forcibly cancelled after 15s regardless of what the
    provider does internally.
    """

    async def test_hanging_provider_gets_cancelled(
        self, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        """A provider that sleeps 60s is cancelled within ~2s (monkeypatched)."""
        # Monkeypatch _REPAIR_TIMEOUT_SECONDS to 2 for fast test
        import application.language.repair_service as repair_mod

        monkeypatch.setattr(repair_mod, "_REPAIR_TIMEOUT_SECONDS", 2)

        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        # Mock provider that hangs forever (simulates a provider
        # that ignores timeout_seconds)
        async def _hanging_route(**kwargs):  # noqa: ANN003
            await asyncio.sleep(60)  # Will be cancelled by wait_for
            # Should never reach here
            return MagicMock(success=True, text="Should not reach", error=None)

        mock_router = MagicMock()
        mock_router.route = _hanging_route

        t_start = time.monotonic()
        result = await service.repair(
            original_output="This is English text",
            ctx=ctx,
            verification_result=verification,
            provider_router=mock_router,
        )
        elapsed = time.monotonic() - t_start

        # Must complete within ~3s (2s timeout + overhead), not 60s
        assert elapsed < 5.0, (
            f"Repair took {elapsed:.1f}s, expected <5s. "
            "asyncio.wait_for is not enforcing the timeout."
        )

        # Result must indicate failure (timeout = no repair)
        assert result.was_repaired is False
        assert result.repair_failed is True
        assert result.attempts_used == 1

    async def test_hanging_provider_logs_timeout_marker(
        self, monkeypatch: "pytest.MonkeyPatch", caplog: "pytest.LogCaptureFixture"
    ) -> None:
        """Timeout produces a 'repair_service.timeout' log marker."""
        import application.language.repair_service as repair_mod

        monkeypatch.setattr(repair_mod, "_REPAIR_TIMEOUT_SECONDS", 1)

        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        async def _hanging_route(**kwargs):  # noqa: ANN003
            await asyncio.sleep(60)

        mock_router = MagicMock()
        mock_router.route = _hanging_route

        with caplog.at_level(logging.WARNING):
            await service.repair(
                original_output="English text",
                ctx=ctx,
                verification_result=verification,
                provider_router=mock_router,
            )

        # The timeout handler logs "repair_service.timeout"
        assert any(
            "repair_service.timeout" in record.message for record in caplog.records
        ), (
            "Expected 'repair_service.timeout' in log output. "
            "This fails without the asyncio.TimeoutError handler."
        )

    async def test_successful_repair_still_works_with_wait_for(self) -> None:
        """A fast provider still succeeds normally (no regression)."""
        service = RepairService(max_attempts=1)
        ctx = _make_ctx("de")
        verification = _make_failed_verification()

        mock_router = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.text = (
            "Dies ist der reparierte deutsche Text der jetzt korrekt "
            "in der richtigen Sprache verfasst wurde und genug Woerter "
            "enthaelt um die Verifikation zu bestehen. Dieser Text ist "
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

        # Fast provider returns successfully
        assert result.was_repaired is True
        assert result.repair_failed is False
