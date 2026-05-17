"""Tests for ExecutionPlan and ExecutionPlanner."""

from __future__ import annotations

import pytest

from application.execution.context import ExecutionContext
from application.execution.plan import ExecutionPlan, ExecutionPlanner, PolicyEngine
from application.language_resolver import LanguageContext


def _make_context(lang: str = "de", request_id: str = "test-req") -> ExecutionContext:
    """Helper to create a minimal ExecutionContext."""
    return ExecutionContext(
        request_id=request_id,
        user_id=1,
        chat_id=2,
        language=LanguageContext(
            code=lang,
            source="detected",
            confidence=0.95,
            switched_from=None,
            request_id=request_id,
        ),
    )


class TestExecutionPlan:
    """Test ExecutionPlan dataclass."""

    def test_default_values(self) -> None:
        """Plan has sensible defaults."""
        plan = ExecutionPlan()
        assert plan.task_type == "answer_chat"
        assert plan.language == "de"
        assert plan.provider_chain == ()
        assert plan.audit_required is True

    def test_frozen(self) -> None:
        """ExecutionPlan is immutable."""
        plan = ExecutionPlan(request_id="x")
        with pytest.raises(Exception):
            plan.request_id = "y"  # type: ignore[misc]

    def test_to_audit_dict(self) -> None:
        """to_audit_dict returns all fields as JSON-friendly lists."""
        plan = ExecutionPlan(
            request_id="req1",
            task_type="debate",
            language="en",
            provider_chain=("claude_persistent", "ollama"),
            memory_used=("mem_1", "mem_2"),
            verifier_profile="strict",
        )
        d = plan.to_audit_dict()
        assert d["request_id"] == "req1"
        assert d["task_type"] == "debate"
        assert d["language"] == "en"
        assert d["provider_chain"] == ["claude_persistent", "ollama"]
        assert d["memory_used"] == ["mem_1", "mem_2"]
        assert d["verifier_profile"] == "strict"
        assert d["audit_required"] is True


class TestPolicyEngine:
    """Test Phase-0 PolicyEngine stub."""

    def test_always_allows(self) -> None:
        """Phase 0 stub always returns ALLOW."""
        engine = PolicyEngine()
        ctx = _make_context()
        assert engine.evaluate(ctx) == "ALLOW"


class TestExecutionPlanner:
    """Test ExecutionPlanner plan creation."""

    def test_plan_chat_basic(self) -> None:
        """plan_chat creates correct plan type."""
        planner = ExecutionPlanner()
        ctx = _make_context(lang="en", request_id="r1")
        plan = planner.plan_chat(ctx)

        assert plan.request_id == "r1"
        assert plan.task_type == "answer_chat"
        assert plan.language == "en"
        assert plan.audit_required is True
        assert "claude_persistent" in plan.provider_chain

    def test_plan_chat_with_memory_ids(self) -> None:
        """plan_chat includes memory IDs."""
        planner = ExecutionPlanner()
        ctx = _make_context()
        plan = planner.plan_chat(ctx, memory_ids=["ep_1", "sem_2"])

        assert plan.memory_used == ("ep_1", "sem_2")

    def test_plan_debate(self) -> None:
        """plan_debate creates debate plan."""
        planner = ExecutionPlanner()
        ctx = _make_context(lang="it", request_id="r2")
        plan = planner.plan_debate(ctx)

        assert plan.request_id == "r2"
        assert plan.task_type == "debate"
        assert plan.language == "it"
        assert plan.memory_used == ()

    def test_custom_provider_chain(self) -> None:
        """Planner uses configured default provider chain."""
        planner = ExecutionPlanner(
            default_provider_chain=["ollama_local", "claude_persistent"]
        )
        ctx = _make_context()
        plan = planner.plan_chat(ctx)

        assert plan.provider_chain == ("ollama_local", "claude_persistent")

    def test_plan_uses_context_language(self) -> None:
        """Plan language comes from context, not hardcoded."""
        planner = ExecutionPlanner()
        ctx = _make_context(lang="fr")
        plan = planner.plan_chat(ctx)
        assert plan.language == "fr"

    def test_plan_debate_uses_context_language(self) -> None:
        """Debate plan uses language from the actual question context."""
        planner = ExecutionPlanner()
        ctx = _make_context(lang="ja")
        plan = planner.plan_debate(ctx)
        assert plan.language == "ja"


class TestPlanImmutability:
    """EK-06: Verify plan tuples cannot be mutated after creation."""

    def test_provider_chain_is_tuple(self) -> None:
        """provider_chain is a tuple, not a list."""
        planner = ExecutionPlanner()
        ctx = _make_context()
        plan = planner.plan_chat(ctx)
        assert isinstance(plan.provider_chain, tuple)

    def test_memory_used_is_tuple(self) -> None:
        """memory_used is a tuple, not a list."""
        planner = ExecutionPlanner()
        ctx = _make_context()
        plan = planner.plan_chat(ctx, memory_ids=["m1"])
        assert isinstance(plan.memory_used, tuple)

    def test_provider_chain_mutation_raises(self) -> None:
        """Attempting to append to provider_chain raises AttributeError."""
        plan = ExecutionPlan(provider_chain=("claude_persistent",))
        with pytest.raises(AttributeError):
            plan.provider_chain.append("x")  # type: ignore[attr-error]

    def test_memory_used_mutation_raises(self) -> None:
        """Attempting to append to memory_used raises AttributeError."""
        plan = ExecutionPlan(memory_used=("m1",))
        with pytest.raises(AttributeError):
            plan.memory_used.append("x")  # type: ignore[attr-error]
