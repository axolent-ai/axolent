"""Tests for ContextKernel: resolver pipeline execution."""

from __future__ import annotations

import pytest

from application.execution.context import (
    ExecutionContext,
)
from application.execution.envelope import RequestEnvelope
from application.execution.kernel import ContextKernel
from application.execution.resolvers import (
    BaseResolver,
    ChannelResolver,
    LanguageResolverAdapter,
    TimeResolver,
)
from application.language_resolver import LanguageContext


class _MockLanguageResolver:
    """Mock for LanguageResolver that returns a fixed language."""

    def __init__(self, lang: str = "en", source: str = "detected") -> None:
        self.lang = lang
        self.source = source
        self.resolve_count = 0

    async def resolve(self, user_id, chat_id, text, override=None):
        self.resolve_count += 1
        return LanguageContext(
            code=override or self.lang,
            source="override" if override else self.source,
            confidence=0.99,
            switched_from=None,
            request_id="mock123",
        )


class TestContextKernelResolvesPipeline:
    """Test that ContextKernel runs resolvers in sequence."""

    @pytest.mark.asyncio
    async def test_build_produces_execution_context(self) -> None:
        """build() returns a frozen ExecutionContext."""
        mock_lr = _MockLanguageResolver(lang="de")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="Hallo")
        ctx = await kernel.build(env)

        assert isinstance(ctx, ExecutionContext)
        assert ctx.request_id == env.request_id
        assert ctx.language.code == "de"
        assert ctx.user_id == 1
        assert ctx.chat_id == 2

    @pytest.mark.asyncio
    async def test_language_resolved_once(self) -> None:
        """Language is resolved exactly once via the kernel."""
        mock_lr = _MockLanguageResolver(lang="it")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=10, chat_id=20, text="Ciao mondo")
        ctx = await kernel.build(env)

        assert ctx.language.code == "it"
        assert mock_lr.resolve_count == 1

    @pytest.mark.asyncio
    async def test_language_override_passed_through(self) -> None:
        """language_override is forwarded to the resolver."""
        mock_lr = _MockLanguageResolver(lang="de")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="Hello")
        ctx = await kernel.build(env, language_override="fr")

        assert ctx.language.code == "fr"
        assert ctx.language.source == "override"

    @pytest.mark.asyncio
    async def test_time_context_filled(self) -> None:
        """TimeResolver fills the time context."""
        mock_lr = _MockLanguageResolver(lang="en")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(
            user_id=1, chat_id=2, text="What time is it?"
        )
        ctx = await kernel.build(env)

        assert ctx.time.timezone_name == "UTC"
        assert ctx.time.weekday >= 0
        assert ctx.time.time_of_day in ("morning", "afternoon", "evening", "night")

    @pytest.mark.asyncio
    async def test_channel_capabilities_telegram(self) -> None:
        """Telegram channel gets correct capabilities."""
        mock_lr = _MockLanguageResolver(lang="en")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="hi")
        ctx = await kernel.build(env)

        assert ctx.channel_capabilities.streaming_supported is True
        assert ctx.channel_capabilities.max_message_length == 4096

    @pytest.mark.asyncio
    async def test_request_id_correlation(self) -> None:
        """request_id flows from envelope through to context."""
        mock_lr = _MockLanguageResolver(lang="en")
        kernel = ContextKernel.create_default(language_resolver=mock_lr)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="x")
        ctx = await kernel.build(env)

        assert ctx.request_id == env.request_id
        assert ctx.language.request_id == env.request_id

    @pytest.mark.asyncio
    async def test_custom_resolver_pipeline(self) -> None:
        """Kernel accepts custom resolver list."""

        class TagResolver(BaseResolver):
            async def resolve(self, partial):
                partial.audit_tags["custom"] = True
                return partial

        mock_lr = _MockLanguageResolver(lang="en")
        resolvers = [
            LanguageResolverAdapter(mock_lr),
            TimeResolver(),
            ChannelResolver(),
            TagResolver(),
        ]
        kernel = ContextKernel(resolvers)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        ctx = await kernel.build(env)

        assert ctx.audit_tags.get("custom") is True
