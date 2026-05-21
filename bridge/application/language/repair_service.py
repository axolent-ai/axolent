"""RepairService: automatic re-query when language verification fails.

When the ResponseLanguageVerifier detects that an LLM response is in
the wrong language, RepairService attempts to fix it by re-querying
the same provider with a reinforced language instruction.

Design constraints:
- max_attempts=1 (hard limit to prevent token waste)
- Only triggers when profile.repair_enabled=True AND verifier fails
- For very long outputs (>5000 chars): sample-verify, do not full rewrite
- Emits audit events for monitoring repair frequency

Token-cost protection:
- A repair costs ~1 additional API call
- Only active for models that need it (strict/strict_with_verify profiles)
- Configurable max_attempts (default 1, hard-capped)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from application.language.context import LanguageContext
from application.language.contract import LanguageContract
from application.language.verifier import ResponseLanguageVerifier, VerificationResult

if TYPE_CHECKING:
    from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)

# Maximum output length for which a full repair rewrite is attempted.
# Above this, only sample-verification is done (no full rewrite due to
# latency and token cost).
_MAX_REPAIR_OUTPUT_LENGTH = 5000

# Hard cap on repair attempts (even if caller requests more)
_HARD_MAX_ATTEMPTS = 2

# Timeout for repair re-query (Issue 4).
# Shorter than the initial query (120s) because if the model already
# produced wrong-language output, a slow repair just adds user frustration.
_REPAIR_TIMEOUT_SECONDS = 15


@dataclass(frozen=True, slots=True)
class RepairResult:
    """Result of a repair attempt.

    Attributes:
        original_output: The original LLM response.
        repaired_output: The repaired response (same as original if no repair).
        was_repaired: Whether a repair was actually performed.
        attempts_used: Number of repair attempts made.
        verification_before: Verification result that triggered repair.
        verification_after: Verification result after repair (None if not repaired).
        latency_ms: Total repair latency in milliseconds.
        repair_failed: True if repair was attempted but still failed.
        skipped_reason: Why repair was skipped (e.g. "output_too_long").
            None when repair was attempted or not needed.
    """

    original_output: str
    repaired_output: str
    was_repaired: bool
    attempts_used: int
    verification_before: VerificationResult
    verification_after: VerificationResult | None
    latency_ms: float
    repair_failed: bool
    skipped_reason: str | None = None


class RepairService:
    """Re-queries the LLM when language verification fails.

    Strategy:
    1. Re-query with reinforced system prompt (LanguageContract.build_repair_contract)
    2. Verify the re-query result
    3. If still failed: return original output + audit as repair_failed
    4. Never enter unbounded loops (max_attempts capped at 2)
    """

    def __init__(
        self,
        verifier: ResponseLanguageVerifier | None = None,
        max_attempts: int = 1,
    ) -> None:
        """Initialize RepairService.

        Args:
            verifier: ResponseLanguageVerifier instance (creates default if None).
            max_attempts: Maximum repair attempts (capped at _HARD_MAX_ATTEMPTS).
        """
        self._verifier = verifier or ResponseLanguageVerifier()
        self._max_attempts = min(max_attempts, _HARD_MAX_ATTEMPTS)

    async def repair(
        self,
        original_output: str,
        ctx: LanguageContext,
        verification_result: VerificationResult,
        provider_router: "ProviderRouter | None" = None,
        provider_name: str | None = None,
        user_id: int = 0,
        chat_id: int = 0,
        model: str | None = None,
        system_prompt_base: str = "",
    ) -> RepairResult:
        """Attempt to repair a language-violated response.

        Args:
            original_output: The failed LLM response.
            ctx: Target LanguageContext.
            verification_result: The failed verification result.
            provider_router: Router for re-query (None = no repair possible).
            provider_name: Which provider to re-query.
            user_id: User ID for the provider call.
            chat_id: Chat ID for the provider call.
            model: Model ID for the provider call.
            system_prompt_base: Original system prompt (for context).

        Returns:
            RepairResult with the outcome.
        """
        t_start = time.perf_counter()

        # Guard: no router means no repair possible
        if provider_router is None:
            log.debug("RepairService: no provider_router, skipping repair")
            return RepairResult(
                original_output=original_output,
                repaired_output=original_output,
                was_repaired=False,
                attempts_used=0,
                verification_before=verification_result,
                verification_after=None,
                latency_ms=0.0,
                repair_failed=False,
            )

        # Guard: output too long for full rewrite (Issue 3).
        # Log a clear warning and return a distinct event_type so the
        # audit trail captures that repair was skipped due to length,
        # not because verification passed.
        if len(original_output) > _MAX_REPAIR_OUTPUT_LENGTH:
            log.warning(
                "RepairService: output too long (%d chars > %d), "
                "skipping repair. User receives wrong-language output.",
                len(original_output),
                _MAX_REPAIR_OUTPUT_LENGTH,
            )
            return RepairResult(
                original_output=original_output,
                repaired_output=original_output,
                was_repaired=False,
                attempts_used=0,
                verification_before=verification_result,
                verification_after=None,
                latency_ms=_elapsed_ms(t_start),
                repair_failed=False,
                skipped_reason="output_too_long",
            )

        # Build repair prompt
        repair_contract = LanguageContract.build_repair_contract(
            ctx=ctx,
            original_detected_lang=verification_result.detected_lang,
        )

        # Combine with base system prompt
        repair_system_prompt = (
            f"{system_prompt_base}\n\n{repair_contract}"
            if system_prompt_base
            else repair_contract
        )

        # Re-query: ask model to rewrite in correct language
        repair_prompt = (
            f"Please rewrite the following text entirely in "
            f"{ctx.code}. Preserve all meaning and formatting:\n\n"
            f"{original_output}"
        )

        attempts_used = 0
        final_verification: VerificationResult | None = None

        for attempt in range(1, self._max_attempts + 1):
            attempts_used = attempt
            log.info(
                "RepairService: attempt %d/%d for lang=%s "
                "(detected=%s, confidence=%.2f)",
                attempt,
                self._max_attempts,
                ctx.code,
                verification_result.detected_lang,
                verification_result.confidence,
            )

            try:
                # Issue 4: explicit short timeout for repair re-query.
                # 15s is shorter than the initial query (120s default)
                # because a repair means the model already proved unreliable
                # and we want to fail fast rather than accumulate latency.
                result = await provider_router.route(
                    prompt=repair_prompt,
                    system_prompt=repair_system_prompt,
                    provider_name=provider_name,
                    user_id=user_id,
                    chat_id=chat_id,
                    model=model,
                    timeout_seconds=_REPAIR_TIMEOUT_SECONDS,
                )

                if not result.success or not result.text.strip():
                    log.warning(
                        "RepairService: re-query failed (attempt %d): %s",
                        attempt,
                        result.error or "empty response",
                    )
                    continue

                # Verify the repair result
                repaired_text = result.text.strip()
                final_verification = self._verifier.verify(repaired_text, ctx.code)

                if final_verification.passed:
                    log.info(
                        "RepairService: repair succeeded on attempt %d "
                        "(detected=%s, confidence=%.2f)",
                        attempt,
                        final_verification.detected_lang,
                        final_verification.confidence,
                    )
                    return RepairResult(
                        original_output=original_output,
                        repaired_output=repaired_text,
                        was_repaired=True,
                        attempts_used=attempts_used,
                        verification_before=verification_result,
                        verification_after=final_verification,
                        latency_ms=_elapsed_ms(t_start),
                        repair_failed=False,
                    )

                log.warning(
                    "RepairService: attempt %d still wrong language (detected=%s)",
                    attempt,
                    final_verification.detected_lang,
                )

            except Exception as exc:
                log.error(
                    "RepairService: exception during repair attempt %d: %s",
                    attempt,
                    exc,
                )
                continue

        # All attempts exhausted
        log.warning(
            "RepairService: all %d attempts exhausted, returning original",
            self._max_attempts,
        )
        return RepairResult(
            original_output=original_output,
            repaired_output=original_output,
            was_repaired=False,
            attempts_used=attempts_used,
            verification_before=verification_result,
            verification_after=final_verification,
            latency_ms=_elapsed_ms(t_start),
            repair_failed=True,
        )

    def build_audit_entry(self, result: RepairResult) -> dict[str, Any]:
        """Build an audit log entry for a repair attempt.

        Args:
            result: The RepairResult to audit.

        Returns:
            Dict suitable for write_audit_log().
        """
        # Issue 3: distinguish "skipped due to length" from generic skip
        if result.skipped_reason == "output_too_long":
            event_type = "language_repair_skipped_too_long"
        elif result.was_repaired:
            event_type = "language_repair_succeeded"
        elif result.repair_failed:
            event_type = "language_repair_failed"
        else:
            event_type = "language_repair_skipped"

        entry: dict[str, Any] = {
            "event_type": event_type,
            "target_lang": result.verification_before.expected_lang,
            "original_detected_lang": result.verification_before.detected_lang,
            "attempts_used": result.attempts_used,
            "latency_ms": round(result.latency_ms, 1),
        }

        if result.skipped_reason is not None:
            entry["skipped_reason"] = result.skipped_reason
            entry["output_length"] = len(result.original_output)

        if result.verification_after is not None:
            entry["final_detected_lang"] = result.verification_after.detected_lang
            entry["final_confidence"] = round(result.verification_after.confidence, 3)

        return entry


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since start.

    Args:
        start: time.perf_counter() value at start.

    Returns:
        Elapsed milliseconds.
    """
    return (time.perf_counter() - start) * 1000
