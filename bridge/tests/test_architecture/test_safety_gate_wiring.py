"""Architecture Guards: One Safety Gate + Edit State Machine wiring.

Ensures:
  1. All persist() calls in LearnFlowService have _validate_contract_safety before them
  2. Follow-up message handler registered in main.py
  3. Callback action 'edit_trigger'/'edit_instruction' registered in skill_commands.py
  4. PendingEditStore imported and used in main.py
  5. handle_learn_followup_message imported and registered

These are AST/semantic guards that prevent "half-wired" regressions.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BRIDGE_ROOT = Path(__file__).resolve().parents[2]


def _read_source(relative_path: str) -> str:
    full = _BRIDGE_ROOT / relative_path
    return full.read_text(encoding="utf-8")


def _parse_ast(relative_path: str) -> ast.Module:
    source = _read_source(relative_path)
    return ast.parse(source, filename=relative_path)


# ---------------------------------------------------------------
# One Safety Gate: all persist calls have safety before them
# ---------------------------------------------------------------


class TestOneSafetyGateArchitecture:
    """Guard: _validate_contract_safety is the ONE safety gate."""

    def test_learn_flow_has_validate_contract_safety(self) -> None:
        """LearnFlowService defines _validate_contract_safety."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        assert "_validate_contract_safety" in source

    def test_save_draft_calls_safety_before_persist(self) -> None:
        """save_draft must call _validate_contract_safety before ContractStore.persist."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        # Find positions: safety check should come BEFORE persist
        safety_pos = source.find("_validate_contract_safety")
        assert safety_pos > 0

        # In save_draft method: safety must be before _contract_store.persist
        save_draft_start = source.find("async def save_draft(")
        assert save_draft_start > 0

        # After save_draft_start: find _validate_contract_safety and persist
        save_draft_section = source[save_draft_start:]
        safety_in_save = save_draft_section.find("_validate_contract_safety")
        persist_in_save = save_draft_section.find("_contract_store.persist")

        assert safety_in_save > 0, "save_draft must call _validate_contract_safety"
        assert persist_in_save > 0, "save_draft must call _contract_store.persist"
        assert safety_in_save < persist_in_save, (
            "safety check must come BEFORE persist in save_draft"
        )

    def test_start_learn_calls_safety_before_persist(self) -> None:
        """start_learn must call _validate_contract_safety before _persist_contract."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        start_learn_start = source.find("async def start_learn(")
        assert start_learn_start > 0

        start_learn_section = source[start_learn_start:]
        safety_in_start = start_learn_section.find("_validate_contract_safety")
        persist_in_start = start_learn_section.find("_persist_contract")

        assert safety_in_start > 0, "start_learn must call _validate_contract_safety"
        assert persist_in_start > 0, "start_learn must call _persist_contract"
        assert safety_in_start < persist_in_start, (
            "safety check must come BEFORE _persist_contract in start_learn"
        )

    def test_edit_trigger_calls_safety(self) -> None:
        """edit_trigger must call _validate_contract_safety."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        edit_trigger_start = source.find("async def edit_trigger(")
        assert edit_trigger_start > 0
        edit_trigger_section = source[edit_trigger_start:]
        # Limit to next method
        next_method = edit_trigger_section.find("async def edit_instruction(")
        if next_method > 0:
            edit_trigger_section = edit_trigger_section[:next_method]
        assert "_validate_contract_safety" in edit_trigger_section, (
            "edit_trigger must call _validate_contract_safety"
        )

    def test_edit_instruction_calls_safety(self) -> None:
        """edit_instruction must call _validate_contract_safety."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        edit_instr_start = source.find("async def edit_instruction(")
        assert edit_instr_start > 0
        edit_instr_section = source[edit_instr_start:]
        # Limit to next method
        next_method = edit_instr_section.find("async def set_pending_edit(")
        if next_method > 0:
            edit_instr_section = edit_instr_section[:next_method]
        assert "_validate_contract_safety" in edit_instr_section, (
            "edit_instruction must call _validate_contract_safety"
        )

    def test_privacy_pipeline_used_not_just_secret_scanner(self) -> None:
        """The safety gate must use PrivacyPipeline, not just SecretScanner."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        # Find _validate_contract_safety method
        method_start = source.find("def _validate_contract_safety(")
        assert method_start > 0
        # Find end of method (next def at same or lower indent)
        next_def = source.find("\n    async def ", method_start + 1)
        if next_def < 0:
            next_def = len(source)
        method_section = source[method_start:next_def]
        # Must call self._privacy.check (full pipeline)
        assert "self._privacy.check" in method_section, (
            "_validate_contract_safety must use full PrivacyPipeline, not just SecretScanner"
        )


# ---------------------------------------------------------------
# Edit State Machine: wiring in main.py
# ---------------------------------------------------------------


class TestEditStateMachineWiring:
    """Guard: Edit state machine properly wired in main.py."""

    def test_main_imports_handle_learn_followup_message(self) -> None:
        source = _read_source("main.py")
        assert "handle_learn_followup_message" in source, (
            "main.py must import handle_learn_followup_message"
        )

    def test_main_imports_pending_edit_store(self) -> None:
        source = _read_source("main.py")
        assert "PendingEditStore" in source, "main.py must import PendingEditStore"

    def test_main_registers_followup_handler_before_generic(self) -> None:
        """Follow-up handler must be in group 0, generic in group 1."""
        source = _read_source("main.py")
        # Follow-up registration
        followup_pos = source.find("handle_learn_followup_message")
        assert followup_pos > 0
        # Generic handle_message registration
        generic_pos = source.rfind("handle_message")
        assert generic_pos > 0
        assert followup_pos < generic_pos, (
            "handle_learn_followup_message must be registered BEFORE handle_message"
        )

    def test_main_uses_handler_groups(self) -> None:
        """main.py must use group=0 and group=1 for priority."""
        source = _read_source("main.py")
        assert "group=0" in source, "main.py must register follow-up handler in group=0"
        assert "group=1" in source, "main.py must register handle_message in group=1"

    def test_main_instantiates_pending_edit_store(self) -> None:
        tree = _parse_ast("main.py")
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
        assert "PendingEditStore" in calls, "main.py must call PendingEditStore()"


# ---------------------------------------------------------------
# Skill commands: edit_trigger / edit_instruction callbacks
# ---------------------------------------------------------------


class TestSkillCommandsEditCallbacks:
    """Guard: skill_commands.py handles edit_trigger and edit_instruction."""

    def test_skill_commands_handles_edit_trigger_action(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert '"edit_trigger"' in source or "'edit_trigger'" in source, (
            "skill_commands.py must handle edit_trigger action"
        )
        assert "skill_learn:edit_trigger:" in source, (
            "skill_commands.py must build skill_learn:edit_trigger callback data"
        )

    def test_skill_commands_handles_edit_instruction_action(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert '"edit_instruction"' in source or "'edit_instruction'" in source, (
            "skill_commands.py must handle edit_instruction action"
        )
        assert "skill_learn:edit_instruction:" in source, (
            "skill_commands.py must build skill_learn:edit_instruction callback data"
        )

    def test_skill_commands_has_followup_handler(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert "handle_learn_followup_message" in source, (
            "skill_commands.py must define handle_learn_followup_message"
        )

    def test_skill_commands_imports_application_handler_stop(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert "ApplicationHandlerStop" in source, (
            "skill_commands.py must import ApplicationHandlerStop for follow-up handler"
        )

    def test_skill_commands_raises_application_handler_stop(self) -> None:
        """Follow-up handler must raise ApplicationHandlerStop when consuming."""
        source = _read_source("presentation/skill_commands.py")
        # Find the handler
        handler_start = source.find("async def handle_learn_followup_message(")
        assert handler_start > 0
        handler_section = source[handler_start:]
        assert "raise ApplicationHandlerStop" in handler_section, (
            "handle_learn_followup_message must raise ApplicationHandlerStop"
        )


# ---------------------------------------------------------------
# Dual-write atomicity: legacy result evaluated
# ---------------------------------------------------------------


class TestDualWriteAtomicityGuard:
    """Guard: legacy_result is evaluated before contract persist."""

    def test_save_draft_evaluates_legacy_result(self) -> None:
        """save_draft must check legacy_result.success."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        save_start = source.find("async def save_draft(")
        assert save_start > 0
        save_section = source[save_start:]
        # Limit to cancel_draft
        next_method = save_section.find("async def cancel_draft(")
        if next_method > 0:
            save_section = save_section[:next_method]

        assert "legacy_result" in save_section, (
            "save_draft must capture legacy_service.learn() result"
        )
        assert "legacy_result.success" in save_section, (
            "save_draft must check legacy_result.success"
        )

    def test_persist_contract_evaluates_legacy_result(self) -> None:
        """_persist_contract must check legacy_result.success."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        persist_start = source.find("async def _persist_contract(")
        assert persist_start > 0
        persist_section = source[persist_start:]

        assert "legacy_result" in persist_section, (
            "_persist_contract must capture legacy_service.learn() result"
        )
        assert "legacy_result.success" in persist_section, (
            "_persist_contract must check legacy_result.success"
        )

    def test_legacy_check_before_contract_persist_in_save(self) -> None:
        """In save_draft: legacy check must happen BEFORE contract persist."""
        source = _read_source("application/skill_compression/learn_flow_service.py")
        save_start = source.find("async def save_draft(")
        save_section = source[save_start:]
        next_method = save_section.find("async def cancel_draft(")
        if next_method > 0:
            save_section = save_section[:next_method]

        legacy_pos = save_section.find("legacy_result.success")
        persist_pos = save_section.find("self._contract_store.persist")

        assert legacy_pos > 0
        assert persist_pos > 0
        assert legacy_pos < persist_pos, (
            "legacy_result.success check must come BEFORE _contract_store.persist"
        )
