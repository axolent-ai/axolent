"""ContextKernel: the central pipeline that builds ExecutionContext.

Entry point for every request. Runs the resolver pipeline and
produces the immutable ExecutionContext that all downstream
components consume.

Usage:
    kernel = ContextKernel.create_default()
    ctx = await kernel.build(envelope)
"""

from __future__ import annotations

import logging
from typing import Optional

from application.execution.context import ExecutionContext, PartialExecutionContext
from application.execution.envelope import RequestEnvelope
from application.execution.resolvers import (
    BaseResolver,
    ChannelResolver,
    LanguageResolverAdapter,
    TimeResolver,
)
from application.language_resolver import LanguageResolver

log = logging.getLogger(__name__)


class ContextKernel:
    """Central pipeline for building ExecutionContext from RequestEnvelope.

    Runs a sequence of resolvers that each fill one part of the
    context. Order matters: LanguageResolver must run before TimeResolver
    (weekday name depends on language).

    Thread safety: stateless after construction, safe for concurrent use.
    """

    def __init__(self, resolvers: list[BaseResolver]) -> None:
        """Initialize with an ordered list of resolvers.

        Args:
            resolvers: Resolvers to run in sequence.
        """
        self._resolvers = resolvers

    async def build(
        self,
        envelope: RequestEnvelope,
        language_override: Optional[str] = None,
    ) -> ExecutionContext:
        """Build a complete ExecutionContext from a RequestEnvelope.

        Runs all resolvers in sequence, then freezes the result.

        Args:
            envelope: The incoming request envelope.
            language_override: Optional explicit language (e.g. from /lang).

        Returns:
            Frozen, immutable ExecutionContext.
        """
        partial = PartialExecutionContext.from_envelope(
            envelope, language_override=language_override
        )

        for resolver in self._resolvers:
            partial = await resolver.resolve(partial)

        ctx = partial.freeze()

        log.debug(
            "ExecutionContext built: request_id=%s, lang=%s, channel=%s, "
            "time_of_day=%s",
            ctx.request_id,
            ctx.language.code,
            ctx.channel,
            ctx.time.time_of_day,
        )

        return ctx

    @classmethod
    def create_default(
        cls,
        language_resolver: Optional[LanguageResolver] = None,
    ) -> "ContextKernel":
        """Factory: create a ContextKernel with the standard Phase-0 resolvers.

        Resolver order:
            1. LanguageResolverAdapter (needs user_id, chat_id, text)
            2. TimeResolver (needs language for weekday name)
            3. ChannelResolver (independent)

        Args:
            language_resolver: Optional custom LanguageResolver instance.

        Returns:
            Configured ContextKernel instance.
        """
        resolvers: list[BaseResolver] = [
            LanguageResolverAdapter(language_resolver),
            TimeResolver(),
            ChannelResolver(),
        ]
        return cls(resolvers)
