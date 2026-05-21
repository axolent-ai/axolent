"""Architecture guards: Skill-Compression wiring integrity.

Analogous to test_lcp_wiring.py. Prevents regression of the
three show-stopper findings from the Skill-Compression review:

1. main.py must instantiate all Skill-Compression components
2. main.py must inject skill_matcher into ChatService
3. main.py must register all Skill-Compression commands + callbacks
4. PatternJudge must NEVER be instantiated without privacy_pipeline
5. ChatService must receive skill_matcher keyword argument
6. R2-SC-01 Guard: All Skill-Compression component kwargs in main.py
   must match the real __init__ signatures (inspect.signature check)

These are AST/source-level checks that run without starting the bot.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from application.skill_compression.conversation_import.orchestrator import (
    ImportOrchestrator,
)
from application.skill_compression.hypothesis_storage import HypothesisStorage
from application.skill_compression.pattern_judge import PatternJudge
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_explainer import SkillExplainer
from application.skill_compression.skill_learning_service import SkillLearningService
from application.skill_compression.skill_matcher import SkillMatcher

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


class TestHandlerWritesEvidenceAfterStreaming:
    """RISK-2 Guard: chat_service.py writes evidence in streaming post-save path."""

    def test_handler_writes_evidence_after_streaming(self) -> None:
        """AST-Check: save_streaming_result contains _write_skill_evidence call."""
        source = _read_source("application/chat_service.py")
        tree = ast.parse(source, filename="chat_service.py")

        # Find save_streaming_result method
        found_evidence_call = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                if node.name == "save_streaming_result":
                    # Search inside this method for _write_skill_evidence
                    method_source = ast.dump(node)
                    if "_write_skill_evidence" in method_source:
                        found_evidence_call = True

        assert found_evidence_call, (
            "save_streaming_result() in chat_service.py must call "
            "_write_skill_evidence for streaming evidence writes (RISK-2 fix)"
        )


class TestHandlerHandlesAskBeforeApplyCallback:
    """RISK-3 Guard: handlers.py implements ask-before-apply pre-check."""

    def test_handler_handles_ask_before_apply_callback(self) -> None:
        """AST-Check: handlers.py contains pre_match_skill and
        skill_confirm keyboard builder in the message handler."""
        source = _read_source("presentation/handlers.py")

        # Must contain pre_match_skill call
        assert "pre_match_skill" in source, (
            "handlers.py must call pre_match_skill for ask-before-apply (RISK-3)"
        )
        # Must contain skill_confirm keyboard
        assert "build_skill_confirm_keyboard" in source, (
            "handlers.py must use build_skill_confirm_keyboard (RISK-3)"
        )
        # Must contain confirmation question i18n key
        assert "skill.confirm_apply_question" in source, (
            "handlers.py must show skill.confirm_apply_question (RISK-3)"
        )

    def test_skill_commands_handles_confirm_callback(self) -> None:
        """AST-Check: skill_commands.py routes skill_confirm: callbacks."""
        source = _read_source("presentation/skill_commands.py")

        assert "skill_confirm:" in source, (
            "skill_commands.py must handle skill_confirm: callback pattern (RISK-3)"
        )
        assert "_handle_skill_confirm_inline" in source, (
            "skill_commands.py must define _handle_skill_confirm_inline (RISK-3)"
        )


# ---------------------------------------------------------------
# R2-SC-01 Guard: inspect.signature kwarg validation
# ---------------------------------------------------------------

# Map of class name (as used in main.py AST) -> actual class object
_SKILL_COMPONENT_CLASSES: dict[str, type] = {
    "HypothesisStorage": HypothesisStorage,
    "PrivacyPipeline": PrivacyPipeline,
    "PatternJudge": PatternJudge,
    "SkillMatcher": SkillMatcher,
    "SkillExplainer": SkillExplainer,
    "SkillLearningService": SkillLearningService,
    "ImportOrchestrator": ImportOrchestrator,
}


class TestSkillComponentKwargsMatchSignature:
    """R2-SC-01 Guard: main.py kwargs must match real __init__ signatures.

    The existing AST guards verify that SkillMatcher(...) IS called in
    main.py, but they did NOT verify that the keyword argument names
    match the real __init__ signature. This caused a TypeError at
    runtime (judge= vs pattern_judge=) that the 28 existing guards
    missed entirely.

    This test class uses inspect.signature() on each of the 7
    Skill-Compression components, extracts the valid parameter names,
    then walks the main.py AST to verify every kwarg used in the
    constructor call is a real parameter.
    """

    def test_all_skill_component_kwargs_match_signatures(self) -> None:
        """Parametric check: every kwarg in main.py matches __init__."""
        tree = _parse_ast("main.py")
        errors: list[str] = []

        for class_name, cls in _SKILL_COMPONENT_CLASSES.items():
            # Get valid params from real __init__
            sig = inspect.signature(cls.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}

            # Find all calls to this class in main.py
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                callee = node.func
                name = ""
                if isinstance(callee, ast.Name):
                    name = callee.id
                elif isinstance(callee, ast.Attribute):
                    name = callee.attr
                if name != class_name:
                    continue

                # Check each keyword argument
                for kw in node.keywords:
                    if kw.arg is None:
                        continue  # **kwargs expansion
                    if kw.arg not in valid_params:
                        errors.append(
                            f"main.py: {class_name}({kw.arg}=...) "
                            f"but real __init__ params are {sorted(valid_params)}. "
                            f"TypeError at runtime!"
                        )

        assert not errors, (
            "Keyword argument mismatch in main.py constructor calls:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


class TestChatServiceNoPrivateAttributeAccess:
    """Claude Beob 2 Guard: ChatService must not access SkillMatcher._storage.

    ChatService should use the public .storage property instead of
    reaching into the private _storage attribute.
    """

    def test_chat_service_does_not_access_skill_matcher_private_storage(
        self,
    ) -> None:
        """No '._storage' access on skill_matcher in chat_service.py."""
        source = _read_source("application/chat_service.py")
        # Check that no line contains skill_matcher._storage
        # The pattern we're looking for is accessing _storage on self.skill_matcher
        assert "skill_matcher._storage" not in source, (
            "chat_service.py accesses SkillMatcher._storage directly. "
            "Use the public .storage property instead."
        )


class TestHandlersEscapeHtmlInSkillClaim:
    """R2-SC-02 Guard: handlers.py must html-escape skill claims."""

    def test_handlers_uses_html_escape_for_skill_claim(self) -> None:
        """The ask-before-apply block must escape _hyp.claim for HTML."""
        source = _read_source("presentation/handlers.py")
        assert "html_mod.escape" in source or "html.escape" in source, (
            "handlers.py must use html.escape() on skill claim text "
            "to prevent Telegram HTML parse errors (R2-SC-02)"
        )
