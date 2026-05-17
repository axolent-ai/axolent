"""ExecutionPlan: structured plan describing what Axolent will do.

Every action is auditable, testable, and debuggable because it was
planned before execution. The plan is the contract between the
kernel and the execution layer.

Phase 0: minimal plan for chat and debate.
Future phases add tool_call, plugin_call, memory_write, scheduled_task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from application.execution.context import ExecutionContext


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable plan for a single request execution.

    Describes what task will be performed, in which language,
    via which providers, and with what audit requirements.

    Attributes:
        request_id: Correlation ID (from ExecutionContext).
        task_type: What kind of task this is.
        language: Resolved language code for this execution.
        provider_chain: Ordered list of providers to try.
        memory_used: IDs of memory entries injected into context.
        verifier_profile: Which verification profile to apply.
        audit_required: Whether this execution must be audited.
    """

    request_id: str = ""
    task_type: Literal[
        "answer_chat",
        "debate",
        "tool_call",
        "plugin_call",
        "memory_write",
        "scheduled_task",
    ] = "answer_chat"
    language: str = "de"
    provider_chain: tuple[str, ...] = field(default_factory=tuple)
    memory_used: tuple[str, ...] = field(default_factory=tuple)
    verifier_profile: str = "standard"
    audit_required: bool = True

    def to_audit_dict(self) -> dict:
        """Convert plan to a dict suitable for audit logging.

        Returns:
            Dictionary with all plan fields (tuples as lists for JSON).
        """
        return {
            "request_id": self.request_id,
            "task_type": self.task_type,
            "language": self.language,
            "provider_chain": list(self.provider_chain),
            "memory_used": list(self.memory_used),
            "verifier_profile": self.verifier_profile,
            "audit_required": self.audit_required,
        }


class PolicyEngine:
    """Stub PolicyEngine for Phase 0.

    Always returns ALLOW. Will be replaced with a real
    implementation in Phase 1.
    """

    def evaluate(self, ctx: ExecutionContext) -> str:
        """Evaluate policy for a context.

        Phase 0: always allows.

        Args:
            ctx: The execution context to evaluate.

        Returns:
            "ALLOW" (always in Phase 0).
        """
        return "ALLOW"


class ExecutionPlanner:
    """Creates ExecutionPlans from ExecutionContext.

    Uses the PolicyEngine (stub in Phase 0) and fallback
    configuration to determine the provider chain.
    """

    def __init__(
        self,
        default_provider_chain: list[str] | tuple[str, ...] | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        """Initialize the planner.

        Args:
            default_provider_chain: Default providers to try in order.
            policy_engine: PolicyEngine instance (stub in Phase 0).
        """
        self._default_chain = (
            tuple(default_provider_chain)
            if default_provider_chain
            else ("claude_persistent",)
        )
        self._policy = policy_engine or PolicyEngine()

    def plan_chat(
        self,
        ctx: ExecutionContext,
        memory_ids: list[str] | tuple[str, ...] | None = None,
    ) -> ExecutionPlan:
        """Create an execution plan for a chat request.

        Args:
            ctx: Resolved execution context.
            memory_ids: IDs of memory entries to include.

        Returns:
            ExecutionPlan for a chat task.
        """
        return ExecutionPlan(
            request_id=ctx.request_id,
            task_type="answer_chat",
            language=ctx.language.code,
            provider_chain=self._default_chain,
            memory_used=tuple(memory_ids) if memory_ids else (),
            verifier_profile="standard",
            audit_required=True,
        )

    def plan_debate(
        self,
        ctx: ExecutionContext,
    ) -> ExecutionPlan:
        """Create an execution plan for a debate request.

        Args:
            ctx: Resolved execution context.

        Returns:
            ExecutionPlan for a debate task.
        """
        return ExecutionPlan(
            request_id=ctx.request_id,
            task_type="debate",
            language=ctx.language.code,
            provider_chain=self._default_chain,
            memory_used=(),
            verifier_profile="standard",
            audit_required=True,
        )
