"""Architecture guards: Skill-Compression wiring integrity.

Analogous to test_lcp_wiring.py. Prevents regression of the
three show-stopper findings from the Skill-Compression review:

1. main.py must instantiate all Skill-Compression components
2. main.py must inject skill_matcher into ChatService
3. main.py must register all Skill-Compression commands + callbacks
4. PatternJudge must NEVER be instantiated without privacy_pipeline
5. ChatService must receive skill_matcher keyword argument

These are AST/source-level checks that run without starting the bot.
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


def _find_call_keywords(tree: ast.Module, func_name: str) -> list[list[str]]:
    """Find all calls to 'func_name(...)' and return their keyword names."""
    results: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = ""
        if isinstance(callee, ast.Name):
            name = callee.id
        elif isinstance(callee, ast.Attribute):
            name = callee.attr
        if name == func_name:
            kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
            results.append(kw_names)
    return results


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


class TestMainImportsSkillCompressionModules:
    """SC-01 Guard 1: main.py imports all required Skill-Compression modules."""

    def test_main_imports_hypothesis_storage(self) -> None:
        source = _read_source("main.py")
        assert "HypothesisStorage" in source, "main.py must import HypothesisStorage"

    def test_main_imports_privacy_pipeline(self) -> None:
        source = _read_source("main.py")
        assert "PrivacyPipeline" in source, "main.py must import PrivacyPipeline"

    def test_main_imports_pattern_judge(self) -> None:
        source = _read_source("main.py")
        assert "PatternJudge" in source, "main.py must import PatternJudge"

    def test_main_imports_skill_matcher(self) -> None:
        source = _read_source("main.py")
        assert "SkillMatcher" in source, "main.py must import SkillMatcher"

    def test_main_imports_skill_explainer(self) -> None:
        source = _read_source("main.py")
        assert "SkillExplainer" in source, "main.py must import SkillExplainer"

    def test_main_imports_import_orchestrator(self) -> None:
        source = _read_source("main.py")
        assert "ImportOrchestrator" in source, "main.py must import ImportOrchestrator"


class TestMainInitializesAllSkillComponents:
    """SC-01 Guard 2: main.py instantiates all Skill-Compression components."""

    def test_main_instantiates_hypothesis_storage(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "HypothesisStorage" in calls, "main.py must call HypothesisStorage(...)"

    def test_main_calls_init_schema(self) -> None:
        source = _read_source("main.py")
        assert (
            "hypothesis_storage.init_schema()" in source or "init_schema()" in source
        ), "main.py must call init_schema() on HypothesisStorage"

    def test_main_instantiates_privacy_pipeline(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "PrivacyPipeline" in calls, "main.py must call PrivacyPipeline()"

    def test_main_instantiates_pattern_judge_with_privacy(self) -> None:
        """PatternJudge must be instantiated WITH privacy_pipeline kwarg."""
        tree = _parse_ast("main.py")
        calls = _find_call_keywords(tree, "PatternJudge")
        assert calls, "PatternJudge(...) call not found in main.py"
        has_privacy = any("privacy_pipeline" in kw_list for kw_list in calls)
        assert has_privacy, (
            "PatternJudge(...) in main.py MUST include privacy_pipeline kwarg. "
            "Without it, all privacy filters are silently skipped."
        )

    def test_main_instantiates_skill_matcher(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "SkillMatcher" in calls, "main.py must call SkillMatcher(...)"

    def test_main_instantiates_skill_explainer(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "SkillExplainer" in calls, "main.py must call SkillExplainer(...)"

    def test_main_instantiates_import_orchestrator(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "ImportOrchestrator" in calls, (
            "main.py must call ImportOrchestrator(...)"
        )

    def test_main_instantiates_skill_learning_service(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_all_call_names(tree)
        assert "SkillLearningService" in calls, (
            "main.py must call SkillLearningService(...)"
        )


class TestMainRegisterAllSkillCommands:
    """SC-01 Guard 3: main.py registers all Skill-Compression command handlers."""

    def test_main_registers_skills_command(self) -> None:
        source = _read_source("main.py")
        assert 'CommandHandler("skills"' in source, (
            "main.py must register /skills command"
        )

    def test_main_registers_skill_command(self) -> None:
        source = _read_source("main.py")
        assert 'CommandHandler("skill"' in source, (
            "main.py must register /skill command"
        )

    def test_main_registers_skillforget_command(self) -> None:
        source = _read_source("main.py")
        assert 'CommandHandler("skillforget"' in source, (
            "main.py must register /skillforget command"
        )

    def test_main_registers_learn_command(self) -> None:
        source = _read_source("main.py")
        assert 'CommandHandler("learn"' in source, (
            "main.py must register /learn command"
        )

    def test_main_registers_explain_command(self) -> None:
        source = _read_source("main.py")
        assert 'CommandHandler("explain"' in source, (
            "main.py must register /explain command"
        )

    def test_main_registers_import_command(self) -> None:
        source = _read_source("main.py")
        assert 'CommandHandler("import"' in source, (
            "main.py must register /import command"
        )


class TestMainRegistersSkillCallbackHandlers:
    """SC-01 Guard 4: main.py registers skill_ and import_ callback handlers."""

    def test_main_registers_skill_callback_handler(self) -> None:
        source = _read_source("main.py")
        assert 'pattern=r"^skill_"' in source, (
            "main.py must register skill_ callback pattern"
        )

    def test_main_registers_import_callback_handler(self) -> None:
        source = _read_source("main.py")
        assert 'pattern=r"^import_"' in source, (
            "main.py must register import_ callback pattern"
        )


class TestChatServiceReceivesSkillMatcher:
    """SC-01 Guard 5: ChatService in main.py receives skill_matcher kwarg."""

    def test_chatservice_receives_skill_matcher_in_main(self) -> None:
        tree = _parse_ast("main.py")
        calls = _find_call_keywords(tree, "ChatService")
        assert calls, "ChatService(...) call not found in main.py"
        has_matcher = any("skill_matcher" in kw_list for kw_list in calls)
        assert has_matcher, (
            "ChatService(...) in main.py MUST include skill_matcher kwarg. "
            "Without it, Skill-Compression is dead in the non-streaming path."
        )


class TestMainSetsBotDataSkillKeys:
    """SC-01 Guard 6: main.py sets all required bot_data keys."""

    def test_main_sets_hypothesis_storage_bot_data(self) -> None:
        source = _read_source("main.py")
        assert '"hypothesis_storage"' in source, (
            'main.py must set bot_data["hypothesis_storage"]'
        )

    def test_main_sets_skill_explainer_bot_data(self) -> None:
        source = _read_source("main.py")
        assert '"skill_explainer"' in source, (
            'main.py must set bot_data["skill_explainer"]'
        )

    def test_main_sets_import_orchestrator_bot_data(self) -> None:
        source = _read_source("main.py")
        assert '"import_orchestrator"' in source, (
            'main.py must set bot_data["import_orchestrator"]'
        )

    def test_main_sets_skill_learning_service_bot_data(self) -> None:
        source = _read_source("main.py")
        assert '"skill_learning_service"' in source, (
            'main.py must set bot_data["skill_learning_service"]'
        )


class TestNoBarePatternJudgeInProductionWiring:
    """SC-01 Guard 7: No PatternJudge() without privacy_pipeline in main.py."""

    def test_no_pattern_judge_without_privacy_pipeline_in_production_wiring(
        self,
    ) -> None:
        """Ensures that PatternJudge is NEVER called without privacy_pipeline.

        The anti-pattern is: PatternJudge() with no arguments.
        This silently skips ALL privacy filters.
        """
        tree = _parse_ast("main.py")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = ""
            if isinstance(callee, ast.Name):
                name = callee.id
            elif isinstance(callee, ast.Attribute):
                name = callee.attr
            if name == "PatternJudge":
                # Must have at least one keyword argument
                kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
                assert "privacy_pipeline" in kw_names, (
                    "PatternJudge() in main.py called WITHOUT privacy_pipeline kwarg. "
                    "This would silently disable all privacy filters."
                )
