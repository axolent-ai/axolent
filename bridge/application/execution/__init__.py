"""Execution Kernel: central request pipeline for Axolent.

Phase 0 MVP: RequestEnvelope, ExecutionContext, ContextKernel,
ExecutionPlan, ExecutionPlanner, InstructionCompiler.

Every user request passes through this pipeline before any
provider call, tool call, or visible output.
"""

from application.execution.context import (
    ChannelCapabilities,
    ExecutionContext,
    TimeContext,
)
from application.execution.envelope import RequestEnvelope
from application.execution.instruction_compiler import (
    CompiledPrompt,
    InstructionCompiler,
)
from application.execution.kernel import ContextKernel
from application.execution.plan import ExecutionPlan, ExecutionPlanner
from application.execution.resolvers import (
    BaseResolver,
    ChannelResolver,
    LanguageResolverAdapter,
    TimeResolver,
)

__all__ = [
    "BaseResolver",
    "ChannelCapabilities",
    "ChannelResolver",
    "CompiledPrompt",
    "ContextKernel",
    "ExecutionContext",
    "ExecutionPlan",
    "ExecutionPlanner",
    "InstructionCompiler",
    "LanguageResolverAdapter",
    "RequestEnvelope",
    "TimeContext",
    "TimeResolver",
]
