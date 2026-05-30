"""Generalized wiring guard: CallbackQuery + Command + Message handler coverage.

Phase 1.5, Item 3: Extends the settings_v2-specific wiring guard to cover
ALL handler registrations in main.py. Catches:
  - Callback data prefixes emitted in presentation/ without a matching handler
  - Command handlers registered but not importable
  - Message handler registration order violations (learn_followup before handle_message)
  - Specific-before-generic pattern ordering for callback handlers

Doc-Lock: if any new callback_data prefix is introduced without a handler,
or handlers are reordered, this test goes red.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

_BRIDGE_ROOT = Path(__file__).resolve().parents[2]


def _read_main_source() -> str:
    """Read main.py source."""
    return (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")


def _extract_registered_callback_patterns(source: str) -> list[tuple[str, int]]:
    """Extract CallbackQueryHandler patterns and their line numbers from main.py.

    Returns list of (pattern_string, line_number) in registration order.
    """
    results = []
    for i, line in enumerate(source.splitlines(), 1):
        m = re.search(r'CallbackQueryHandler\([^,]+,\s*pattern=r"([^"]+)"', line)
        if m:
            results.append((m.group(1), i))
    return results


def _extract_emitted_callback_prefixes() -> set[str]:
    """Scan presentation/**/*.py for all callback_data=... string prefixes.

    Uses AST-based extraction (Phase 1.5 Polish-Polish Item 6) to catch:
      - ast.Constant: literal strings (single/double quoted)
      - ast.JoinedStr: f-strings (leading literal parts before first variable)
      - ast.BinOp: simple string concatenation (left-side literal, best-effort)

    Module-level string constants (ast.Name) are NOT resolved; this is a
    known limitation acceptable because the codebase currently uses only
    inline literals for callback_data. If constants are introduced later,
    the extractor can be extended.

    Returns set of prefixes (the part before the first ':' or '{').
    """
    pres_dir = _BRIDGE_ROOT / "presentation"
    prefixes: set[str] = set()

    for py_file in pres_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Find keyword argument callback_data=...
            if not isinstance(node, ast.keyword):
                continue
            if node.arg != "callback_data":
                continue

            value = node.value
            prefix = _extract_prefix_from_ast_node(value, content)
            if prefix:
                prefixes.add(prefix)

    return prefixes


def _extract_prefix_from_ast_node(node: ast.expr, source: str = "") -> Optional[str]:
    """Extract callback_data prefix from an AST value node.

    Handles:
      - ast.Constant (literal strings)
      - ast.JoinedStr (f-strings: extracts leading literal parts)
      - ast.BinOp with Str+... (string concatenation, best-effort)

    Returns the prefix (before first ':' or '{'), or None.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # Plain string literal (single or double quoted)
        return re.split(r"[{:]", node.value)[0] or None

    if isinstance(node, ast.JoinedStr):
        # f-string: extract leading Constant parts before first FormattedValue
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            else:
                # Hit a FormattedValue (variable), stop collecting
                break
        if parts:
            joined = "".join(parts)
            prefix = re.split(r"[{:]", joined)[0]
            return prefix or None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # String concatenation: "prefix" + variable
        left_prefix = _extract_prefix_from_ast_node(node.left, source)
        if left_prefix:
            return left_prefix

    return None


def _extract_registered_commands(source: str) -> list[str]:
    """Extract all CommandHandler command names from main.py."""
    return re.findall(r'CommandHandler\("([^"]+)"', source)


class TestCallbackQueryPrefixCoverage:
    """Every callback_data prefix emitted in presentation/ must have a handler."""

    def test_all_emitted_prefixes_have_handlers(self) -> None:
        """Each emitted callback_data prefix matches at least one registered pattern."""
        source = _read_main_source()
        registered = _extract_registered_callback_patterns(source)
        emitted = _extract_emitted_callback_prefixes()

        # Compile registered patterns
        compiled = [re.compile(pat) for pat, _ in registered]

        uncovered: list[str] = []
        for prefix in sorted(emitted):
            # Check if any registered pattern matches this prefix
            # The prefix is the start of a callback_data value, so we test
            # if the pattern would match a typical callback_data string
            test_value = f"{prefix}:test_value"
            if not any(p.match(test_value) or p.match(prefix) for p in compiled):
                uncovered.append(prefix)

        assert not uncovered, (
            f"Callback data prefixes emitted in presentation/ but no matching "
            f"CallbackQueryHandler registered in main.py: {uncovered}"
        )

    def test_emitted_prefixes_nontrivial(self) -> None:
        """Sanity: we should find a non-trivial number of emitted prefixes."""
        emitted = _extract_emitted_callback_prefixes()
        # main.py has 9 CallbackQueryHandlers, presentation emits many prefixes
        assert len(emitted) >= 15, (
            f"Expected at least 15 emitted callback prefixes, found {len(emitted)}. "
            f"Extraction logic may be broken."
        )


class TestCallbackQueryPatternOrdering:
    """Specific patterns must be registered before their generic catch-all."""

    # Known specific-before-generic pairs (from main.py)
    # Each tuple: (specific_pattern, generic_pattern, description)
    _ORDERING_INVARIANTS: list[tuple[str, str, str]] = [
        (
            r"^settings_v2_",
            r"^settings_",
            "settings_v2_ must come before settings_ (Etappe 1.5.1)",
        ),
        (
            r"^skill_learn:",
            r"^skill_",
            "skill_learn: must come before skill_ (catch-all)",
        ),
    ]

    def test_specific_before_generic(self) -> None:
        """Each specific pattern is registered before its generic catch-all."""
        source = _read_main_source()
        registered = _extract_registered_callback_patterns(source)

        errors: list[str] = []
        for specific, generic, desc in self._ORDERING_INVARIANTS:
            specific_line = None
            generic_line = None
            for pat, lineno in registered:
                if pat == specific:
                    specific_line = lineno
                if pat == generic:
                    generic_line = lineno

            if specific_line is None:
                errors.append(f"Specific pattern {specific!r} not found in main.py")
            elif generic_line is None:
                errors.append(f"Generic pattern {generic!r} not found in main.py")
            elif specific_line > generic_line:
                errors.append(
                    f"ORDERING VIOLATION: {desc}. "
                    f"Specific ({specific!r}) at line {specific_line} "
                    f"AFTER generic ({generic!r}) at line {generic_line}."
                )

        assert not errors, "CallbackQueryHandler ordering violations:\n" + "\n".join(
            f"  - {e}" for e in errors
        )


class TestCommandHandlerCoverage:
    """Every registered CommandHandler must be importable and not dead."""

    def test_all_command_handlers_importable(self) -> None:
        """Every CommandHandler function referenced in main.py must be importable."""
        source = _read_main_source()
        tree = ast.parse(source, filename="main.py")

        # Collect all from...import statements
        imports: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = module

        # Find all CommandHandler("cmd", handler_func) calls and their handler names
        handler_funcs: list[tuple[str, str]] = []
        for m in re.finditer(r'CommandHandler\("(\w+)",\s*(\w+)\)', source):
            handler_funcs.append((m.group(1), m.group(2)))

        errors: list[str] = []
        for cmd, func_name in handler_funcs:
            if func_name not in imports:
                errors.append(f"/{cmd}: handler {func_name} not found in imports")

        assert not errors, (
            "CommandHandler functions not imported in main.py:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def test_command_handlers_nontrivial(self) -> None:
        """Sanity: we should find at least 20 command handlers."""
        source = _read_main_source()
        commands = _extract_registered_commands(source)
        assert len(commands) >= 20, (
            f"Expected at least 20 command handlers, found {len(commands)}. "
            f"Extraction may be broken."
        )


class TestMessageHandlerOrdering:
    """MessageHandler registration order invariants.

    learn_followup_message in group 0 MUST come before handle_message in group 1.
    """

    def test_learn_followup_before_handle_message(self) -> None:
        """handle_learn_followup_message (group 0) before handle_message (group 1)."""
        source = _read_main_source()

        followup_pos = source.find("handle_learn_followup_message")
        message_pos = source.find("handle_message)")
        # Find the group= assignments near each
        assert followup_pos != -1, "handle_learn_followup_message not found in main.py"
        assert message_pos != -1, "handle_message not found in main.py"

        # handle_learn_followup_message must appear BEFORE handle_message
        # Get group numbers
        followup_group_match = re.search(
            r"handle_learn_followup_message.*?group=(\d+)",
            source,
            re.DOTALL,
        )
        message_group_match = re.search(
            r"handle_message\).*?group=(\d+)",
            source,
            re.DOTALL,
        )

        assert followup_group_match is not None, (
            "Could not find group= for handle_learn_followup_message"
        )
        assert message_group_match is not None, (
            "Could not find group= for handle_message"
        )

        followup_group = int(followup_group_match.group(1))
        message_group = int(message_group_match.group(1))

        assert followup_group < message_group, (
            f"handle_learn_followup_message (group {followup_group}) must be in a "
            f"LOWER group than handle_message (group {message_group}). "
            f"Lower group = higher priority in python-telegram-bot."
        )

    def test_both_message_handlers_registered(self) -> None:
        """Both handle_learn_followup_message and handle_message are registered."""
        source = _read_main_source()
        assert "handle_learn_followup_message" in source
        assert "handle_message" in source
        # Both must be in MessageHandler calls
        assert re.search(r"MessageHandler.*handle_learn_followup_message", source), (
            "handle_learn_followup_message must be in a MessageHandler"
        )
        assert re.search(r"MessageHandler.*handle_message\b", source), (
            "handle_message must be in a MessageHandler"
        )


class TestDecoratorStackConsistency:
    """Every CommandHandler function MUST carry at least one permission-granting
    decorator from the explicit AUTHORIZATION_DECORATORS allowlist.

    Missing decorators are silent privilege gaps: a handler without an
    authorization decorator allows ANY user to execute that command,
    bypassing the whitelist check entirely.

    AUTHORIZATION_DECORATORS: explicit allowlist. Adding a new permission-granting
    decorator (e.g. @conditional_whitelist) REQUIRES updating this list.
    Without update, the guard gives false security by accepting handlers without
    a recognized authorization decorator.

    Detection method: AST-scan of the source file containing each handler
    to verify that a recognized decorator appears in the decorator chain above
    the function definition.
    """

    # Explicit allowlist of permission-granting decorator names.
    # Adding a new authorization decorator to the codebase REQUIRES adding it here.
    # See docs/CONVENTIONS.md for the update obligation.
    AUTHORIZATION_DECORATORS: frozenset[str] = frozenset(
        [
            "require_whitelist",
            # Add new permission-granting decorators here with a comment explaining
            # their authorization semantics. Example:
            # "require_admin",  # Only bot admins can execute this handler
        ]
    )

    # Handlers that are intentionally public (if any) go here with
    # justification. Currently: none. All handlers require whitelist.
    EXEMPT_HANDLERS: set[str] = set()

    def test_all_command_handlers_have_authorization_decorator(self) -> None:
        """Every CommandHandler function has a recognized authorization decorator."""
        source = _read_main_source()

        # Extract handler function names from CommandHandler("cmd", func)
        handler_funcs = re.findall(r'CommandHandler\("\w+",\s*(\w+)\)', source)
        assert len(handler_funcs) >= 20, (
            f"Expected at least 20 handlers, found {len(handler_funcs)}"
        )

        # Resolve each handler to its source file and check decorators
        errors: list[str] = []

        # Build import map: func_name -> module_path
        import_map: dict[str, str] = {}
        tree = ast.parse(source, filename="main.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    import_map[name] = module

        for func_name in handler_funcs:
            if func_name in self.EXEMPT_HANDLERS:
                continue

            module_path = import_map.get(func_name)
            if not module_path:
                errors.append(f"{func_name}: not found in main.py imports")
                continue

            # Resolve module file
            module_file = _BRIDGE_ROOT / module_path.replace(".", "/")
            # Could be a package or a .py file
            if module_file.with_suffix(".py").exists():
                module_file = module_file.with_suffix(".py")
            elif (module_file / "__init__.py").exists():
                module_file = module_file / "__init__.py"
            else:
                errors.append(f"{func_name}: module file not found for {module_path}")
                continue

            mod_source = module_file.read_text(encoding="utf-8")

            # Find the function definition and check decorators above it.
            lines = mod_source.splitlines()
            func_def_pattern = re.compile(
                rf"^\s*(async\s+)?def\s+{re.escape(func_name)}\s*\("
            )
            found_def = False
            for i, line in enumerate(lines):
                if func_def_pattern.match(line):
                    found_def = True
                    # Check preceding lines for authorization decorators
                    # (decorators are in the lines immediately above def)
                    decorator_lines: list[str] = []
                    j = i - 1
                    while j >= 0 and (
                        lines[j].strip().startswith("@") or lines[j].strip() == ""
                    ):
                        if lines[j].strip().startswith("@"):
                            decorator_lines.append(lines[j].strip())
                        j -= 1

                    has_authz = any(
                        any(
                            deco_name in d
                            for deco_name in self.AUTHORIZATION_DECORATORS
                        )
                        for d in decorator_lines
                    )
                    if not has_authz:
                        errors.append(
                            f"{func_name} in {module_path}: MISSING authorization "
                            f"decorator. Found: {decorator_lines}. "
                            f"Allowed: {sorted(self.AUTHORIZATION_DECORATORS)}."
                        )
                    break

            if not found_def:
                errors.append(
                    f"{func_name}: function definition not found in {module_path}"
                )

        assert not errors, (
            "CommandHandler functions missing authorization decorator:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
