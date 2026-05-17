"""Tests for FallbackResolver: automatic provider failover.

10 test cases covering:
  1. Primary succeeds without fallback
  2. Rate limit (429) triggers fallback
  3. Timeout triggers fallback
  4. Provider exception triggers fallback
  5. All providers failed returns error
  6. User notice above threshold
  7. Logging records fallback
  8. Metrics per provider
  9. Chain configuration from env
  10. Per-slot chain differentiation
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.fallback_resolver import (
    FallbackResolver,
    load_fallback_config_from_env,
)
from infrastructure.providers.base import (
    ProviderResponse,
    ProviderTimeout,
    ProviderUnavailable,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider_router():
    """Create a mock ProviderRouter with configurable route behavior."""
    router = MagicMock()
    router.default = "claude_persistent"
    router.route = AsyncMock()
    return router


@pytest.fixture
def resolver_with_chain(mock_provider_router):
    """Create a FallbackResolver with a standard 3-provider chain."""
    chains = {
        "chat": ["claude_persistent", "ollama_local", "haiku_fallback"],
        "code": ["claude_persistent", "ollama_local"],
    }
    return FallbackResolver(
        provider_router=mock_provider_router,
        fallback_chains=chains,
        timeout_seconds=30,
        user_notice_threshold=2,
    )


def _success_response(provider: str = "claude_persistent") -> ProviderResponse:
    """Helper: create a successful ProviderResponse."""
    return ProviderResponse(
        text="Hello, world!",
        duration_seconds=1.5,
        provider_name=provider,
        model="claude-sonnet-4-6",
        error=None,
    )


def _error_response(provider: str, error: str) -> ProviderResponse:
    """Helper: create a ProviderResponse with an error."""
    return ProviderResponse(
        text="",
        duration_seconds=0.5,
        provider_name=provider,
        error=error,
    )


# ---------------------------------------------------------------------------
# Test 1: Primary succeeds, no fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_succeeds_no_fallback(resolver_with_chain, mock_provider_router):
    """When primary provider succeeds, no fallback is triggered."""
    mock_provider_router.route.return_value = _success_response("claude_persistent")

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.response.success
    assert result.response.text == "Hello, world!"
    assert result.fallback_used is False
    assert result.fallback_level == 0
    assert result.provider_name == "claude_persistent"
    assert result.user_notice == ""
    # Only one call to route
    assert mock_provider_router.route.call_count == 1


# ---------------------------------------------------------------------------
# Test 2: Rate limit triggers fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_triggers_fallback(resolver_with_chain, mock_provider_router):
    """429 rate limit from primary triggers fallback to secondary."""
    # First call: rate limit error in response
    # Second call: success from ollama
    mock_provider_router.route.side_effect = [
        _error_response("claude_persistent", "Rate limit exceeded (429)"),
        _success_response("ollama_local"),
    ]

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.response.success
    assert result.fallback_used is True
    assert result.fallback_level == 1
    assert result.provider_name == "ollama_local"
    assert mock_provider_router.route.call_count == 2


# ---------------------------------------------------------------------------
# Test 3: Timeout triggers fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_triggers_fallback(resolver_with_chain, mock_provider_router):
    """ProviderTimeout exception triggers fallback to next provider."""
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        _success_response("ollama_local"),
    ]

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.response.success
    assert result.fallback_used is True
    assert result.fallback_level == 1
    assert result.provider_name == "ollama_local"


# ---------------------------------------------------------------------------
# Test 4: Provider exception triggers fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_exception_triggers_fallback(
    resolver_with_chain, mock_provider_router
):
    """Generic ProviderError triggers fallback."""
    mock_provider_router.route.side_effect = [
        ProviderUnavailable("claude_persistent", reason="CLI not found"),
        _success_response("ollama_local"),
    ]

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.response.success
    assert result.fallback_used is True
    assert result.fallback_level == 1


# ---------------------------------------------------------------------------
# Test 5: All providers failed returns error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_failed_returns_error(resolver_with_chain, mock_provider_router):
    """When all providers in chain fail, return clear error response."""
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        ProviderTimeout("ollama_local", timeout_seconds=30),
        ProviderTimeout("haiku_fallback", timeout_seconds=30),
    ]

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.response.success is False
    assert result.response.error is not None
    assert result.fallback_used is True
    assert result.fallback_level == -1
    assert result.provider_name == "none"
    # All 3 providers attempted
    assert mock_provider_router.route.call_count == 3


# ---------------------------------------------------------------------------
# Test 6: User notice above threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_notice_above_threshold(resolver_with_chain, mock_provider_router):
    """When fallback level >= threshold, user gets a notice."""
    # threshold is 2, so level 2 (third provider) should show notice
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        ProviderTimeout("ollama_local", timeout_seconds=30),
        _success_response("haiku_fallback"),
    ]

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.response.success
    assert result.fallback_level == 2
    assert result.user_notice != ""
    assert "haiku_fallback" in result.user_notice


@pytest.mark.asyncio
async def test_no_user_notice_below_threshold(
    resolver_with_chain, mock_provider_router
):
    """When fallback level < threshold, no user notice."""
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        _success_response("ollama_local"),
    ]

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.fallback_level == 1
    assert result.user_notice == ""


# ---------------------------------------------------------------------------
# Test 7: Logging records fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logging_records_fallback(resolver_with_chain, mock_provider_router):
    """Each fallback trigger is logged with reason."""
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        _success_response("ollama_local"),
    ]

    with patch("application.fallback_resolver.log") as mock_log:
        await resolver_with_chain.resolve(
            slot="chat",
            prompt="Hello",
            user_lang="en",
        )

        # Check that warning was logged for the fallback
        mock_log.warning.assert_called()
        call_args = mock_log.warning.call_args[0]
        assert "Fallback triggered" in call_args[0]
        assert "chat" in str(call_args)


# ---------------------------------------------------------------------------
# Test 8: Metrics per provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_per_provider(resolver_with_chain, mock_provider_router):
    """Metrics counters increment correctly per provider and reason."""
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        _success_response("ollama_local"),
    ]

    await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    metrics = resolver_with_chain.metrics

    # claude_persistent: 1 attempt, 1 failure
    assert metrics.get_attempt_count("claude_persistent") == 1
    assert metrics.get_failure_count("claude_persistent") == 1

    # ollama_local: 1 attempt, 0 failures
    assert metrics.get_attempt_count("ollama_local") == 1
    assert metrics.get_failure_count("ollama_local") == 0

    # Failure reason tracked
    assert metrics.failure_reasons[("claude_persistent", "timeout")] == 1

    # Slot fallback tracked
    assert metrics.slot_fallbacks["chat"] == 1


# ---------------------------------------------------------------------------
# Test 9: Chain configuration from env
# ---------------------------------------------------------------------------


def test_chain_configuration_from_env():
    """Environment variables are parsed into fallback chains correctly."""
    env_vars = {
        "AXOLENT_FALLBACK_CHAIN_CHAT": "claude_persistent,ollama_local,haiku",
        "AXOLENT_FALLBACK_CHAIN_CODE": "claude_persistent,ollama_local",
        "AXOLENT_FALLBACK_CHAIN_DEFAULT": "claude_persistent",
        "AXOLENT_FALLBACK_TIMEOUT_SECONDS": "45",
        "AXOLENT_FALLBACK_USER_NOTICE_THRESHOLD": "3",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = load_fallback_config_from_env()

    assert config["fallback_chains"]["chat"] == [
        "claude_persistent",
        "ollama_local",
        "haiku",
    ]
    assert config["fallback_chains"]["code"] == ["claude_persistent", "ollama_local"]
    assert config["fallback_chains"]["default"] == ["claude_persistent"]
    assert config["timeout_seconds"] == 45
    assert config["user_notice_threshold"] == 3


def test_chain_configuration_defaults():
    """Without env vars, config returns empty chains and defaults."""
    env_vars = {}
    with patch.dict(os.environ, env_vars, clear=True):
        config = load_fallback_config_from_env()

    assert config["fallback_chains"] == {}
    assert config["timeout_seconds"] == 30
    assert config["user_notice_threshold"] == 2


# ---------------------------------------------------------------------------
# Test 10: Per-slot chain differentiation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_slot_chain(mock_provider_router):
    """Different slots use different fallback chains."""
    chains = {
        "chat": ["claude_persistent", "ollama_local"],
        "code": ["claude_persistent", "deepseek_code"],
    }
    resolver = FallbackResolver(
        provider_router=mock_provider_router,
        fallback_chains=chains,
        timeout_seconds=30,
    )

    # Chat slot: timeout on primary -> uses ollama_local
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        _success_response("ollama_local"),
    ]
    result = await resolver.resolve(slot="chat", prompt="Hello", user_lang="en")
    assert result.provider_name == "ollama_local"

    # Reset mock
    mock_provider_router.route.reset_mock()

    # Code slot: timeout on primary -> uses deepseek_code
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        _success_response("deepseek_code"),
    ]
    result = await resolver.resolve(slot="code", prompt="/code fix", user_lang="en")
    assert result.provider_name == "deepseek_code"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_error_triggers_fallback(
    resolver_with_chain, mock_provider_router
):
    """ConnectionError (network down) triggers fallback."""
    mock_provider_router.route.side_effect = [
        ConnectionError("Connection refused"),
        _success_response("ollama_local"),
    ]

    result = await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
    )

    assert result.response.success
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_unknown_slot_uses_default_chain(mock_provider_router):
    """Unknown slot falls back to 'default' chain if configured."""
    chains = {
        "default": ["claude_persistent", "ollama_local"],
    }
    resolver = FallbackResolver(
        provider_router=mock_provider_router,
        fallback_chains=chains,
    )

    mock_provider_router.route.return_value = _success_response("claude_persistent")

    result = await resolver.resolve(slot="unknown_slot", prompt="Hello", user_lang="en")
    assert result.response.success
    assert result.provider_name == "claude_persistent"


@pytest.mark.asyncio
async def test_no_chain_configured_uses_router_default(mock_provider_router):
    """With no chains configured, uses single router default."""
    resolver = FallbackResolver(
        provider_router=mock_provider_router,
        fallback_chains={},
    )

    mock_provider_router.route.return_value = _success_response("claude_persistent")

    result = await resolver.resolve(slot="chat", prompt="Hello", user_lang="en")
    assert result.response.success
    assert result.provider_name == "claude_persistent"


@pytest.mark.asyncio
async def test_model_override_only_for_primary(
    resolver_with_chain, mock_provider_router
):
    """Model override is only passed to the primary provider, not fallbacks."""
    mock_provider_router.route.side_effect = [
        ProviderTimeout("claude_persistent", timeout_seconds=30),
        _success_response("ollama_local"),
    ]

    await resolver_with_chain.resolve(
        slot="chat",
        prompt="Hello",
        user_lang="en",
        model="claude-opus-4-7",
    )

    # First call (primary): model passed
    first_call = mock_provider_router.route.call_args_list[0]
    assert first_call.kwargs.get("model") == "claude-opus-4-7"

    # Second call (fallback): model is None
    second_call = mock_provider_router.route.call_args_list[1]
    assert second_call.kwargs.get("model") is None
