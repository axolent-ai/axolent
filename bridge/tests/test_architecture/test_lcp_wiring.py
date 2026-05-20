"""Architecture guards: LCP (Language Control Plane) wiring integrity.

Codex blocker rule (2026-05-20): these tests prevent regression of
the three show-stopper findings from the LCP review:

1. main.py must inject LanguageEnforcement into ChatService
2. main.py must inject DetectionAuditLogger into LanguageResolver
3. handlers.py must pass language data to save_streaming_result

These are AST/source-level checks that run without starting the bot.
They verify the wiring exists in code, not at runtime.
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
    """Find all calls to 'func_name(...)' and return their keyword names.

    Returns a list of keyword-name-lists, one per call site found.
    E.g. for ChatService(provider_router=..., language_enforcement=...)
    returns [["provider_router", "language_enforcement"]].
    """
    results: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match Class(...) or module.Class(...)
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


class TestMainInjectsLanguageEnforcement:
    """Finding 1: main.py must wire LanguageEnforcement into ChatService."""

    def test_chatservice_receives_language_enforcement(self) -> None:
        """ChatService(...) in main.py must include language_enforcement kwarg."""
        tree = _parse_ast("main.py")
        calls = _find_call_keywords(tree, "ChatService")
        assert calls, "ChatService(...) call not found in main.py"
        # At least one ChatService call must have language_enforcement
        has_enforcement = any("language_enforcement" in kw_list for kw_list in calls)
        assert has_enforcement, (
            "ChatService(...) in main.py is missing language_enforcement kwarg. "
            "LCP enforcement will not run in production."
        )

    def test_language_enforcement_is_instantiated(self) -> None:
        """LanguageEnforcement must be instantiated somewhere in main.py."""
        source = _read_source("main.py")
        assert "LanguageEnforcement(" in source, (
            "LanguageEnforcement is never instantiated in main.py. "
            "Import + construction required for LCP wiring."
        )


class TestMainInjectsDetectionAuditLogger:
    """Finding 3: main.py must wire DetectionAuditLogger into LanguageResolver."""

    def test_language_resolver_receives_audit_logger(self) -> None:
        """LanguageResolver(...) in main.py must include audit_logger kwarg."""
        tree = _parse_ast("main.py")
        calls = _find_call_keywords(tree, "LanguageResolver")
        assert calls, "LanguageResolver(...) call not found in main.py"
        has_audit = any("audit_logger" in kw_list for kw_list in calls)
        assert has_audit, (
            "LanguageResolver(...) in main.py is missing audit_logger kwarg. "
            "Detection audit events will not be logged."
        )

    def test_detection_audit_logger_is_instantiated(self) -> None:
        """DetectionAuditLogger must be instantiated somewhere in main.py."""
        source = _read_source("main.py")
        assert "DetectionAuditLogger(" in source, (
            "DetectionAuditLogger is never instantiated in main.py. "
            "Import + construction required for audit trail."
        )


class TestHandlersPassLanguageDataToStreamingSave:
    """Finding 2: handlers.py must pass language data to save_streaming_result."""

    _REQUIRED_KWARGS = frozenset(
        {
            "language_code",
            "language_ctx",
            "user_model",
            "provider_name",
        }
    )

    def test_save_streaming_result_receives_language_params(self) -> None:
        """save_streaming_result() call must include all 4 language kwargs."""
        tree = _parse_ast("presentation/handlers.py")
        calls = _find_call_keywords(tree, "save_streaming_result")
        assert calls, "save_streaming_result(...) call not found in handlers.py"
        # Check the first (and normally only) call site
        kw_set = set(calls[0])
        missing = self._REQUIRED_KWARGS - kw_set
        assert not missing, (
            f"save_streaming_result() in handlers.py is missing kwargs: {missing}. "
            "LCP post-stream enforcement needs these to verify language compliance."
        )

    def test_debate_orchestrator_receives_language_enforcement(self) -> None:
        """DebateOrchestrator(...) in handlers.py must include language_enforcement."""
        tree = _parse_ast("presentation/handlers.py")
        calls = _find_call_keywords(tree, "DebateOrchestrator")
        assert calls, "DebateOrchestrator(...) call not found in handlers.py"
        has_enforcement = any("language_enforcement" in kw_list for kw_list in calls)
        assert has_enforcement, (
            "DebateOrchestrator(...) in handlers.py is missing "
            "language_enforcement kwarg. Debate responses will skip "
            "language verification."
        )
