#!/usr/bin/env python3
"""AST-based scanner: detect hardcoded user-facing strings in Telegram calls.

Walks the AST of presentation/ and application/ Python files to find
string literals passed directly to Telegram API methods that should
use the i18n t() system instead.

Watched calls:
  - reply_text(...)
  - edit_message_text(...)
  - answer(text=...)
  - send_message(...)
  - InlineKeyboardButton(...) (first arg or text= kwarg)

Whitelist rules:
  - Variable references (not string literals) are always OK
  - Calls where the argument is a t() result or wrapper are OK
  - Pure ASCII symbols/markers (e.g. "...", "---", arrow chars) are OK
  - Lines ending with `# i18n: ok` suppress the finding

Usage:
    python scripts/i18n_scan.py

Exit code 0 = no violations, 1 = hardcoded strings found (blocks commit).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BRIDGE_DIR = Path(__file__).parent.parent / "bridge"
SCAN_DIRS = [
    BRIDGE_DIR / "presentation",
    BRIDGE_DIR / "application",
]

# Function/method names to watch for hardcoded string arguments
WATCHED_METHODS: set[str] = {
    "reply_text",
    "edit_message_text",
    "send_message",
}

# Methods where we check the `text=` keyword argument
WATCHED_KEYWORD_METHODS: set[str] = {
    "answer",
}

# Classes where first arg or text= kwarg is user-facing
WATCHED_CONSTRUCTORS: set[str] = {
    "InlineKeyboardButton",
}

# Regex for strings that are pure symbols/markers (not natural language)
# Matches: "...", "---", single unicode symbols, empty strings
SYMBOL_PATTERN = re.compile(
    r"^[\s\-_.,:;!?#*=<>|/\\@&+~`─-╿▀-▟"
    r"■-◿←-⇿⤀-⥿⬀-⯿"
    r" -/:-@[-`{-~"
    r"\U0001f000-\U0001ffff]*$"
)

# Suppress comment pattern
SUPPRESS_COMMENT = "# i18n: ok"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Violation(NamedTuple):
    """A single hardcoded string finding."""

    file: Path
    line: int
    col: int
    method: str
    string_value: str


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------


class I18nStringVisitor(ast.NodeVisitor):
    """Visits Call nodes and checks for hardcoded string arguments."""

    def __init__(self, source_lines: list[str], file_path: Path) -> None:
        self.source_lines = source_lines
        self.file_path = file_path
        self.violations: list[Violation] = []

    def _is_suppressed(self, lineno: int) -> bool:
        """Check if the line has a # i18n: ok suppress comment."""
        if lineno < 1 or lineno > len(self.source_lines):
            return False
        line = self.source_lines[lineno - 1]
        return SUPPRESS_COMMENT in line

    def _is_symbol_only(self, value: str) -> bool:
        """Check if string is pure symbols/markers (not natural language)."""
        return bool(SYMBOL_PATTERN.match(value))

    def _is_t_call(self, node: ast.expr) -> bool:
        """Check if node is a call to t() or similar i18n wrapper."""
        if isinstance(node, ast.Call):
            func = node.func
            # Direct t() call
            if isinstance(func, ast.Name) and func.id in ("t", "get_text"):
                return True
            # module.t() call
            if isinstance(func, ast.Attribute) and func.attr in ("t", "get_text"):
                return True
        return False

    def _is_string_literal(self, node: ast.expr) -> bool:
        """Check if node is a string literal (Constant or JoinedStr/f-string)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return True
        if isinstance(node, ast.JoinedStr):
            return True
        return False

    def _get_string_value(self, node: ast.expr) -> str:
        """Extract string value from a literal node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            # For f-strings, reconstruct a representative value
            parts: list[str] = []
            for val in node.values:
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    parts.append(val.value)
                else:
                    parts.append("{...}")
            return "".join(parts)
        return ""

    def _get_method_name(self, node: ast.Call) -> str | None:
        """Extract method name from a Call node."""
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return None

    def _check_argument(
        self, arg_node: ast.expr, method_name: str, call_node: ast.Call
    ) -> None:
        """Check a single argument node for hardcoded strings."""
        # Skip if it's a t() call result
        if self._is_t_call(arg_node):
            return

        # Skip if it's not a string literal (variable, function call, etc.)
        if not self._is_string_literal(arg_node):
            return

        # Get the string value
        value = self._get_string_value(arg_node)

        # Skip pure symbol/marker strings
        if self._is_symbol_only(value):
            return

        # Skip empty strings
        if not value.strip():
            return

        # Check for suppress comment on the line
        lineno = arg_node.lineno
        if self._is_suppressed(lineno):
            return

        # Also check the call line itself (for multi-line calls)
        if self._is_suppressed(call_node.lineno):
            return

        self.violations.append(
            Violation(
                file=self.file_path,
                line=lineno,
                col=arg_node.col_offset,
                method=method_name,
                string_value=value[:60],
            )
        )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Visit a function/method call and check for hardcoded strings."""
        method_name = self._get_method_name(node)

        if method_name is None:
            self.generic_visit(node)
            return

        # Check standard watched methods (first positional arg)
        if method_name in WATCHED_METHODS:
            if node.args:
                self._check_argument(node.args[0], method_name, node)

        # Check keyword-based methods (text= kwarg)
        elif method_name in WATCHED_KEYWORD_METHODS:
            for kw in node.keywords:
                if kw.arg == "text":
                    self._check_argument(kw.value, method_name, node)
                    break

        # Check constructors (first arg or text= kwarg)
        elif method_name in WATCHED_CONSTRUCTORS:
            checked = False
            # Check text= keyword first
            for kw in node.keywords:
                if kw.arg == "text":
                    self._check_argument(kw.value, method_name, node)
                    checked = True
                    break
            # If no text= keyword, check first positional arg
            if not checked and node.args:
                self._check_argument(node.args[0], method_name, node)

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def scan_file(file_path: Path) -> list[Violation]:
    """Scan a single Python file for hardcoded i18n violations.

    Args:
        file_path: Path to the .py file.

    Returns:
        List of violations found.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"WARNING: Cannot read {file_path}: {exc}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        print(f"WARNING: Syntax error in {file_path}: {exc}", file=sys.stderr)
        return []

    source_lines = source.splitlines()
    visitor = I18nStringVisitor(source_lines, file_path)
    visitor.visit(tree)
    return visitor.violations


def scan_directories(dirs: list[Path]) -> list[Violation]:
    """Scan all .py files in the given directories.

    Args:
        dirs: List of directory paths to scan.

    Returns:
        All violations found across all files.
    """
    all_violations: list[Violation] = []

    for scan_dir in dirs:
        if not scan_dir.exists():
            continue
        for py_file in sorted(scan_dir.rglob("*.py")):
            # Skip __pycache__ and test files
            if "__pycache__" in str(py_file) or "test" in py_file.name.lower():
                continue
            violations = scan_file(py_file)
            all_violations.extend(violations)

    return all_violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the scanner and report findings.

    Returns:
        0 if no violations, 1 if violations found.
    """
    violations = scan_directories(SCAN_DIRS)

    if not violations:
        print("i18n scan: PASSED (no hardcoded user-facing strings found)")
        return 0

    # Group by file for readable output
    by_file: dict[Path, list[Violation]] = {}
    for v in violations:
        by_file.setdefault(v.file, []).append(v)

    print("i18n scan: FAILED")
    print(f"Found {len(violations)} hardcoded string(s) that should use t():\n")

    for file_path, file_violations in sorted(by_file.items()):
        rel_path = file_path.relative_to(BRIDGE_DIR.parent)
        print(f"  {rel_path}:")
        for v in sorted(file_violations, key=lambda x: x.line):
            preview = v.string_value.replace("\n", "\\n")
            print(f'    L{v.line}: {v.method}("{preview}")')
        print()

    print("Fix: Replace hardcoded strings with t('key', lang) calls.")
    print("Suppress false positives with: # i18n: ok")
    return 1


if __name__ == "__main__":
    # Ensure UTF-8 output on Windows (avoid cp1252 encoding errors)
    import io

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    sys.exit(main())
