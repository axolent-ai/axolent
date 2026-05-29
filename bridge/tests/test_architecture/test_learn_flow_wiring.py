"""Architecture guards: Learn Flow (v2) production wiring integrity.

Ensures the new ContractBuilder/DraftStore/ContractStore/LearnFlowService
are properly wired in production code (main.py, skill_commands.py).

Analogous to test_skill_compression_wiring.py, prevents the Etappe 3
NO-GO from recurring: modules exist but are never production-wired.
"""

from __future__ import annotations

import ast
from pathlib import Path

# bridge/ root
_BRIDGE_ROOT = Path(__file__).resolve().parents[2]


def _read_source(relative_path: str) -> str:
    """Read a source file relative to bridge root."""
    full = _BRIDGE_ROOT / relative_path
    return full.read_text(encoding="utf-8")


def _parse_ast(relative_path: str) -> ast.Module:
    """Parse a source file into AST."""
    source = _read_source(relative_path)
    return ast.parse(source, filename=relative_path)


def _find_all_call_names(tree: ast.Module) -> list[str]:
    """Find all function/class call names in an AST."""
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name):
            names.append(callee.id)
        elif isinstance(callee, ast.Attribute):
            names.append(callee.attr)
    return names


class TestLearnFlowComponentsImportedOutsideTests:
    """Guard: ContractBuilder/DraftStore/ContractStore used in production code."""

    def test_contract_builder_imported_in_main(self) -> None:
        source = _read_source("main.py")
        assert "ContractBuilder" in source, "main.py must import ContractBuilder"

    def test_draft_store_imported_in_main(self) -> None:
        source = _read_source("main.py")
        assert "DraftStore" in source, "main.py must import DraftStore"

    def test_contract_store_imported_in_main(self) -> None:
        source = _read_source("main.py")
        assert "ContractStore" in source, "main.py must import ContractStore"

    def test_learn_flow_service_imported_in_main(self) -> None:
        source = _read_source("main.py")
        assert "LearnFlowService" in source, "main.py must import LearnFlowService"


class TestMainInitializesLearnFlowComponents:
    """Guard: main.py instantiates all learn flow components."""

    def test_main_instantiates_contract_store(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "ContractStore" in calls, "main.py must call ContractStore(...)"

    def test_main_instantiates_draft_store(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "DraftStore" in calls, "main.py must call DraftStore()"

    def test_main_instantiates_learn_flow_service(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "LearnFlowService" in calls, "main.py must call LearnFlowService(...)"

    def test_main_calls_contract_store_init_schema(self) -> None:
        source = _read_source("main.py")
        assert "contract_store.init_schema()" in source, (
            "main.py must call contract_store.init_schema()"
        )


class TestMainSetsBotDataLearnFlowKeys:
    """Guard: main.py sets learn flow bot_data keys."""

    def test_main_sets_contract_store_bot_data(self) -> None:
        source = _read_source("main.py")
        assert '"contract_store"' in source, (
            'main.py must set bot_data["contract_store"]'
        )

    def test_main_sets_draft_store_bot_data(self) -> None:
        source = _read_source("main.py")
        assert '"draft_store"' in source, 'main.py must set bot_data["draft_store"]'

    def test_main_sets_learn_flow_service_bot_data(self) -> None:
        source = _read_source("main.py")
        assert '"learn_flow_service"' in source, (
            'main.py must set bot_data["learn_flow_service"]'
        )


class TestMainRegistersLearnCallbackHandler:
    """Guard: main.py registers skill_learn: callback pattern."""

    def test_main_registers_skill_learn_callback(self) -> None:
        source = _read_source("main.py")
        assert 'pattern=r"^skill_learn:"' in source, (
            "main.py must register skill_learn: callback handler"
        )

    def test_main_imports_handle_learn_callback(self) -> None:
        source = _read_source("main.py")
        assert "handle_learn_callback" in source, (
            "main.py must import handle_learn_callback from skill_commands"
        )


class TestSkillCommandsUsesLearnFlowService:
    """Guard: skill_commands.py uses LearnFlowService (not only legacy)."""

    def test_skill_commands_has_learn_flow_service_getter(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert "learn_flow_service" in source, (
            "skill_commands.py must reference learn_flow_service"
        )

    def test_skill_commands_has_handle_learn_callback(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert "handle_learn_callback" in source, (
            "skill_commands.py must define handle_learn_callback"
        )

    def test_skill_commands_uses_i18n_preview_keys(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert "skill.learn_preview_header" in source, (
            "skill_commands.py must use skill.learn_preview_header i18n key"
        )

    def test_skill_commands_has_quick_flag_parsing(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert "--quick" in source, (
            "skill_commands.py must parse --quick flag in /learn handler"
        )

    def test_skill_commands_builds_save_edit_cancel_buttons(self) -> None:
        source = _read_source("presentation/skill_commands.py")
        assert "skill_learn:save:" in source, (
            "skill_commands.py must build skill_learn:save callback data"
        )
        assert "skill_learn:edit:" in source, (
            "skill_commands.py must build skill_learn:edit callback data"
        )
        assert "skill_learn:cancel:" in source, (
            "skill_commands.py must build skill_learn:cancel callback data"
        )
