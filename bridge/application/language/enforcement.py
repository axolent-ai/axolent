"""LanguageEnforcement: integration facade for the Language Control Plane.

Provides a single entry point that ChatService and DebateOrchestrator
can call to run the full verify+repair pipeline on any LLM output.

This avoids scattering verification/repair logic across multiple files.
Consumers call:
    result = await enforcement.enforce(output, ctx, model_id, ...)
    final_text = result.final_output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from application.language.context import LanguageContext
from application.language.model_profiles import ModelAdherenceProfile, get_profile
from application.language.repair_service import RepairResult, RepairService
from application.language.verifier import ResponseLanguageVerifier, VerificationResult
from infrastructure.audit_log import write_audit_log

if TYPE_CHECKING:
    from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnforcementResult:
    """Result of the full language enforcement pipeline.

    Attributes:
        final_output: The text to deliver to the user (original or repaired).
        verification: Verification result (None if skipped by profile).
        repair: Repair result (None if no repair was needed/attempted).
        was_enforced: Whether any enforcement action was taken.
        model_profile: The profile used for enforcement decisions.
    """

    final_output: str
    verification: VerificationResult | None
    repair: RepairResult | None
    was_enforced: bool
    model_profile: ModelAdherenceProfile


class LanguageEnforcement:
    """Integration facade for verify + repair pipeline.

    Usage by ChatService:
        enforcement = LanguageEnforcement(provider_router=self.provider_router)
        result = await enforcement.enforce(
            output=response_text,
            ctx=execution_context.language,
            model_id=resolved_model,
        )
        final_response = result.final_output
    """

    def __init__(
        self,
        verifier: ResponseLanguageVerifier | None = None,
        repair_service: RepairService | None = None,
        provider_router: "ProviderRouter | None" = None,
    ) -> None:
        """Initialize enforcement facade.

        Args:
            verifier: Custom verifier (creates default if None).
            repair_service: Custom repair service (creates default if None).
            provider_router: Router for repair re-queries.
        """
        self._verifier = verifier or ResponseLanguageVerifier()
        self._repair = repair_service or RepairService()
        self._provider_router = provider_router

    async def enforce(
        self,
        output: str,
        ctx: LanguageContext,
        model_id: str | None = None,
        provider_name: str | None = None,
        user_id: int = 0,
        chat_id: int = 0,
        system_prompt_base: str = "",
        request_id: str = "",
    ) -> EnforcementResult:
        """Run the full enforcement pipeline on LLM output.

        Steps:
        1. Look up model profile
        2. If verify_required: run verifier
        3. If verification fails AND repair_enabled: run repair
        4. Emit audit events
        5. Return final output

        Args:
            output: Raw LLM response text.
            ctx: Resolved LanguageContext for this request.
            model_id: Model ID (for profile lookup).
            provider_name: Provider to use for repair re-query.
            user_id: User ID for provider calls.
            chat_id: Chat ID for provider calls.
            system_prompt_base: Base system prompt for repair context.
            request_id: Correlation ID for audit.

        Returns:
            EnforcementResult with final output and diagnostics.
        """
        profile = get_profile(model_id)

        # Step 1: Check if verification is required
        if not profile.verify_required:
            return EnforcementResult(
                final_output=output,
                verification=None,
                repair=None,
                was_enforced=False,
                model_profile=profile,
            )

        # Step 2: Verify
        verification = self._verifier.verify(output, ctx.code)

        # Audit: verification performed
        self._audit_verification(verification, request_id, model_id)

        # Step 3: If passed or skipped, return as-is
        if verification.passed:
            return EnforcementResult(
                final_output=output,
                verification=verification,
                repair=None,
                was_enforced=False,
                model_profile=profile,
            )

        # Step 4: Verification failed. Repair if enabled.
        if not profile.repair_enabled:
            log.info(
                "Language verification failed but repair disabled for model=%s",
                model_id,
            )
            return EnforcementResult(
                final_output=output,
                verification=verification,
                repair=None,
                was_enforced=False,
                model_profile=profile,
            )

        # Step 5: Run repair
        repair_result = await self._repair.repair(
            original_output=output,
            ctx=ctx,
            verification_result=verification,
            provider_router=self._provider_router,
            provider_name=provider_name,
            user_id=user_id,
            chat_id=chat_id,
            model=model_id,
            system_prompt_base=system_prompt_base,
        )

        # Audit: repair attempt
        self._audit_repair(repair_result, request_id, model_id)

        return EnforcementResult(
            final_output=repair_result.repaired_output,
            verification=verification,
            repair=repair_result,
            was_enforced=repair_result.was_repaired,
            model_profile=profile,
        )

    def _audit_verification(
        self,
        result: VerificationResult,
        request_id: str,
        model_id: str | None,
    ) -> None:
        """Write audit log for verification."""
        entry: dict[str, Any] = {
            "event_type": "language_verification_performed",
            "request_id": request_id,
            "model_id": model_id or "unknown",
            "target_lang": result.expected_lang,
            "detected_lang": result.detected_lang,
            "confidence": round(result.confidence, 3),
            "foreign_share": round(result.foreign_share, 3),
            "passed": result.passed,
            "skipped": result.skipped,
        }
        if result.reason:
            entry["reason"] = result.reason
        write_audit_log(entry)

    def _audit_repair(
        self,
        result: RepairResult,
        request_id: str,
        model_id: str | None,
    ) -> None:
        """Write audit log for repair attempt."""
        entry = self._repair.build_audit_entry(result)
        entry["request_id"] = request_id
        entry["model_id"] = model_id or "unknown"
        write_audit_log(entry)
