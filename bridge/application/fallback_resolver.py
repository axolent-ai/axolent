"""Fallback resolver: automatic provider failover for LLM requests.

Sits between TaskRouter and ProviderRouter. When a primary provider
fails (rate limit, timeout, exception, 5xx), the resolver transparently
tries alternative providers from a per-slot fallback chain.

Design decisions (confirmed 2026-05-09 in axolent-tagebuch):
  - Own class in application layer (complexity warrants separation)
  - Per-slot fallback chains (CHAT may differ from CODE)
  - User-facing notice above configurable threshold
  - All failures logged with reason and provider name
  - Metrics counters per provider/slot/failure-reason
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from i18n.domain.i18n import t
from infrastructure.providers.base import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
    ProviderUnavailable,
)

if TYPE_CHECKING:
    from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_USER_NOTICE_THRESHOLD = 2  # show notice from this fallback level


# ---------------------------------------------------------------------------
# Fallback trigger classification
# ---------------------------------------------------------------------------


class FallbackTrigger(Exception):
    """Raised when a provider failure should trigger fallback to next provider.

    Attributes:
        provider_name: Which provider failed.
        reason: Why it failed (rate_limit, timeout, exception, unavailable).
    """

    def __init__(self, provider_name: str, reason: str, detail: str = "") -> None:
        self.provider_name = provider_name
        self.reason = reason
        self.detail = detail
        super().__init__(f"{provider_name}: {reason} ({detail})")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class FallbackMetrics:
    """Simple in-memory counters for fallback events.

    Not persisted across restarts (session-level observability).
    """

    # provider_name -> count
    attempts: dict[str, int] = field(default_factory=dict)
    # provider_name -> count of failures
    failures: dict[str, int] = field(default_factory=dict)
    # (provider_name, reason) -> count
    failure_reasons: dict[tuple[str, str], int] = field(default_factory=dict)
    # slot_name -> count of fallback activations
    slot_fallbacks: dict[str, int] = field(default_factory=dict)

    def record_attempt(self, provider: str) -> None:
        self.attempts[provider] = self.attempts.get(provider, 0) + 1

    def record_failure(self, provider: str, reason: str, slot: str) -> None:
        self.failures[provider] = self.failures.get(provider, 0) + 1
        key = (provider, reason)
        self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1
        self.slot_fallbacks[slot] = self.slot_fallbacks.get(slot, 0) + 1

    def get_failure_count(self, provider: str) -> int:
        return self.failures.get(provider, 0)

    def get_attempt_count(self, provider: str) -> int:
        return self.attempts.get(provider, 0)


# ---------------------------------------------------------------------------
# Resolve result
# ---------------------------------------------------------------------------


@dataclass
class ResolveResult:
    """Result of a fallback-resolved provider call.

    Attributes:
        response: The provider response (always set, even on total failure).
        fallback_used: True if a non-primary provider answered.
        fallback_level: 0 = primary, 1 = first fallback, etc.
        provider_name: Which provider ultimately answered.
        user_notice: Optional notice text for the user (i18n).
    """

    response: ProviderResponse
    fallback_used: bool = False
    fallback_level: int = 0
    provider_name: str = ""
    user_notice: str = ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FallbackResolver:
    """Resolves LLM requests with automatic provider failover.

    Wraps ProviderRouter: tries providers in chain order, falls back
    on rate-limit, timeout, or provider errors.

    Args:
        provider_router: The ProviderRouter instance for actual calls.
        fallback_chains: Per-slot ordered list of provider names.
            Example: {"chat": ["claude_persistent", "ollama_local"],
                      "code": ["claude_persistent", "ollama_local"]}
        timeout_seconds: Max wait per provider attempt.
        user_notice_threshold: Fallback level at which user gets a notice.
    """

    def __init__(
        self,
        provider_router: "ProviderRouter",
        fallback_chains: Optional[dict[str, list[str]]] = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        user_notice_threshold: int = _DEFAULT_USER_NOTICE_THRESHOLD,
    ) -> None:
        self.provider_router = provider_router
        self._chains = fallback_chains or {}
        self._timeout_seconds = timeout_seconds
        self._user_notice_threshold = user_notice_threshold
        self.metrics = FallbackMetrics()

        log.info(
            "FallbackResolver initialized. Chains: %s, timeout: %ds, notice_threshold: %d",
            {k: v for k, v in self._chains.items()},
            self._timeout_seconds,
            self._user_notice_threshold,
        )

    def _get_fallback_chain(self, slot: str) -> list[str]:
        """Get the fallback chain for a slot.

        Falls back to default chain if slot has no specific chain.
        If no chain is configured at all, returns the router default.

        Args:
            slot: Task slot name (e.g. "chat", "code").

        Returns:
            Ordered list of provider names to try.
        """
        chain = self._chains.get(slot)
        if chain:
            return list(chain)

        # Try "default" chain
        default_chain = self._chains.get("default")
        if default_chain:
            return list(default_chain)

        # Ultimate fallback: just the router default
        return [self.provider_router.default]

    async def resolve(
        self,
        slot: str,
        prompt: str,
        system_prompt: str = "",
        user_id: int | None = None,
        chat_id: int | None = None,
        user_lang: str = "en",
        model: str | None = None,
    ) -> ResolveResult:
        """Try primary provider, fall back to alternatives on failure.

        Args:
            slot: Task slot name (e.g. "chat", "code").
            prompt: User message / context prompt.
            system_prompt: System prompt.
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            user_lang: User language for i18n notices.
            model: Optional model override (used for primary only).

        Returns:
            ResolveResult with response and fallback metadata.
        """
        chain = self._get_fallback_chain(slot)

        for level, provider_name in enumerate(chain):
            self.metrics.record_attempt(provider_name)

            try:
                response = await self._try_provider(
                    provider_name=provider_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    user_id=user_id,
                    chat_id=chat_id,
                    # Only pass model override to primary provider
                    model=model if level == 0 else None,
                )

                # Success
                user_notice = ""
                if level >= self._user_notice_threshold:
                    user_notice = t(
                        "fallback.notice",
                        user_lang,
                        backup=provider_name,
                    )

                return ResolveResult(
                    response=response,
                    fallback_used=level > 0,
                    fallback_level=level,
                    provider_name=provider_name,
                    user_notice=user_notice,
                )

            except FallbackTrigger as trigger:
                self.metrics.record_failure(provider_name, trigger.reason, slot)
                log.warning(
                    "Fallback triggered [slot=%s, level=%d/%d]: %s",
                    slot,
                    level + 1,
                    len(chain),
                    trigger,
                )
                continue

        # All providers in chain failed
        log.error("All providers failed for slot '%s'. Chain: %s", slot, chain)
        return self._all_failed_result(user_lang)

    async def _try_provider(
        self,
        provider_name: str,
        prompt: str,
        system_prompt: str,
        user_id: int | None,
        chat_id: int | None,
        model: str | None,
    ) -> ProviderResponse:
        """Attempt a single provider call, raising FallbackTrigger on failure.

        Args:
            provider_name: Provider to try.
            prompt: User prompt.
            system_prompt: System prompt.
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            model: Optional model override.

        Returns:
            ProviderResponse on success.

        Raises:
            FallbackTrigger: If the provider fails in a way that warrants fallback.
        """
        try:
            response = await self.provider_router.route(
                prompt=prompt,
                system_prompt=system_prompt,
                provider_name=provider_name,
                timeout_seconds=self._timeout_seconds,
                user_id=user_id,
                chat_id=chat_id,
                model=model,
            )

            # Check for error in response (provider returned but with error)
            if response.error:
                error_lower = response.error.lower()
                if "rate" in error_lower or "429" in error_lower:
                    raise FallbackTrigger(provider_name, "rate_limit", response.error)
                if "timeout" in error_lower:
                    raise FallbackTrigger(provider_name, "timeout", response.error)
                # Generic provider error in response
                raise FallbackTrigger(provider_name, "provider_error", response.error)

            return response

        except ProviderTimeout as e:
            raise FallbackTrigger(provider_name, "timeout", str(e)) from e

        except ProviderUnavailable as e:
            raise FallbackTrigger(provider_name, "unavailable", str(e)) from e

        except ProviderError as e:
            # Retryable errors (rate limits, server errors) trigger fallback
            reason = "rate_limit" if "rate" in str(e).lower() else "provider_error"
            raise FallbackTrigger(provider_name, reason, str(e)) from e

        except (ConnectionError, OSError) as e:
            raise FallbackTrigger(provider_name, "connection_error", str(e)) from e

        except TimeoutError as e:
            raise FallbackTrigger(provider_name, "timeout", str(e)) from e

    def _all_failed_result(self, user_lang: str) -> ResolveResult:
        """Build a ResolveResult when all providers have failed.

        Args:
            user_lang: User language for i18n error message.

        Returns:
            ResolveResult with error response.
        """
        error_text = t("fallback.all_failed", user_lang)
        return ResolveResult(
            response=ProviderResponse(
                text="",
                duration_seconds=0.0,
                provider_name="fallback_resolver",
                error=error_text,
            ),
            fallback_used=True,
            fallback_level=-1,
            provider_name="none",
            user_notice=error_text,
        )


# ---------------------------------------------------------------------------
# Factory: load configuration from environment
# ---------------------------------------------------------------------------


def load_fallback_config_from_env() -> dict[str, Any]:
    """Load FallbackResolver configuration from environment variables.

    Reads:
        AXOLENT_FALLBACK_CHAIN_CHAT: comma-separated provider names for chat slot
        AXOLENT_FALLBACK_CHAIN_CODE: comma-separated provider names for code slot
        AXOLENT_FALLBACK_CHAIN_REASON: comma-separated provider names
        AXOLENT_FALLBACK_CHAIN_CREATIVE: comma-separated provider names
        AXOLENT_FALLBACK_CHAIN_QUICK: comma-separated provider names
        AXOLENT_FALLBACK_CHAIN_RESEARCH: comma-separated provider names
        AXOLENT_FALLBACK_CHAIN_DEFAULT: default chain for unconfigured slots
        AXOLENT_FALLBACK_TIMEOUT_SECONDS: timeout per provider attempt
        AXOLENT_FALLBACK_USER_NOTICE_THRESHOLD: level at which user sees notice

    Returns:
        Dict with keys: fallback_chains, timeout_seconds, user_notice_threshold.
    """
    chains: dict[str, list[str]] = {}

    slot_names = ["chat", "code", "reason", "creative", "quick", "research", "default"]
    for slot in slot_names:
        env_key = f"AXOLENT_FALLBACK_CHAIN_{slot.upper()}"
        raw = os.environ.get(env_key, "").strip()
        if raw:
            providers = [p.strip() for p in raw.split(",") if p.strip()]
            if providers:
                chains[slot] = providers

    timeout = int(
        os.environ.get(
            "AXOLENT_FALLBACK_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)
        )
    )
    notice_threshold = int(
        os.environ.get(
            "AXOLENT_FALLBACK_USER_NOTICE_THRESHOLD",
            str(_DEFAULT_USER_NOTICE_THRESHOLD),
        )
    )

    return {
        "fallback_chains": chains,
        "timeout_seconds": timeout,
        "user_notice_threshold": notice_threshold,
    }
