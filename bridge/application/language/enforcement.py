"""LanguageEnforcement: integration facade for the Language Control Plane.

Provides a single entry point that ChatService and DebateOrchestrator
can call to run the full verify+repair pipeline on any LLM output.

This avoids scattering verification/repair logic across multiple files.
Consumers call:
    result = await enforcement.enforce(output, ctx, model_id, ...)
    final_text = result.final_output

Architecture note (Codex Finding 5):
    This module lives in the application layer and must NOT import
    infrastructure modules directly. Audit logging is abstracted
    behind the AuditLogPort protocol and injected via constructor.
    The concrete write_audit_log adapter is wired in main.py.

Verification note (Codex Finding 8):
    Verification only runs for models with verify_required=True per
    their ModelAdherenceProfile. Models like Claude Opus/Sonnet have
    verify_required=False because they reliably follow language
    instructions. This is NOT a gap in enforcement but a deliberate
    performance optimization. The Defensive Publication's claim that
    "every response is verified" describes the capability, not the
    runtime behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import icontract
from typeguard import typechecked

from application.language.context import LanguageContext
from application.language.model_profiles import ModelAdherenceProfile, get_profile
from application.language.repair_service import RepairResult, RepairService
from application.language.verifier import ResponseLanguageVerifier, VerificationResult

if TYPE_CHECKING:
    from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)


@runtime_checkable
class AuditLogPort(Protocol):
    """Application-layer port for audit logging.

    Decouples enforcement from infrastructure.audit_log (hexagonal rule).
    The concrete adapter (write_audit_log) is injected via constructor.
    """

    def __call__(self, entry: dict[str, Any]) -> None:
        """Write an audit log entry."""
        ...


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

    @typechecked
    def __init__(
        self,
        verifier: ResponseLanguageVerifier | None = None,
        repair_service: RepairService | None = None,
        provider_router: "ProviderRouter | None" = None,
        audit_log: AuditLogPort | None = None,
    ) -> None:
        """Initialize enforcement facade.

        Args:
            verifier: Custom verifier (creates default if None).
            repair_service: Custom repair service (creates default if None).
            provider_router: Router for repair re-queries.
            audit_log: Audit log writer (injected from infrastructure via
                main.py). When None, audit entries are logged via stdlib
                logger as fallback (no silent data loss).
        """
        self._verifier = verifier or ResponseLanguageVerifier()
        self._repair = repair_service or RepairService()
        self._provider_router = provider_router
        self._audit_log = audit_log

    @icontract.require(
        lambda ctx: isinstance(ctx, LanguageContext),
        "ctx must be a LanguageContext instance (immutable)",
    )
    @icontract.require(
        lambda ctx: ctx.code and ctx.code.strip(),
        "ctx.code (language code) must not be empty",
    )
    @icontract.ensure(
        lambda result: isinstance(result, EnforcementResult),
        "result must be an EnforcementResult",
    )
    @icontract.ensure(
        lambda result: result.final_output is not None,
        "result.final_output must not be None",
    )
    @typechecked
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

        Contracts:
            Pre: ctx is an immutable LanguageContext with non-empty code.
            Post: returns EnforcementResult with non-None final_output.

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

    def _write_audit(self, entry: dict[str, Any]) -> None:
        """Write audit entry via injected port or fallback to logger."""
        if self._audit_log is not None:
            self._audit_log(entry)
        else:
            log.info("audit_log (no port): %s", entry)

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
        self._write_audit(entry)

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
        self._write_audit(entry)
