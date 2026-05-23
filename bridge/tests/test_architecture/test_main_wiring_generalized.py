"""Generalized inspect.signature guard for ALL main.py constructors.

Walks main.py AST, finds every Call(Name) where Name matches a known
class, looks up the class via importlib, compares the Call's keyword
arguments against inspect.signature(cls.__init__).parameters.

Catches the SkillMatcher(judge=...) bug class for any future
component, not just skill compression.

This is the generalized version of TestSkillComponentKwargsMatchSignature
from test_skill_compression_wiring.py.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

# bridge/ root
_BRIDGE_ROOT = Path(__file__).resolve().parents[2]


def _read_main_source() -> str:
    """Read main.py source."""
    return (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")


def _parse_main_ast() -> ast.Module:
    """Parse main.py into AST."""
    return ast.parse(_read_main_source(), filename="main.py")


def _extract_imports(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Extract all 'from X import Y' statements from main.py AST.

    Returns a dict: class_name -> (module_path, class_name).
    Only collects names that start with uppercase (likely classes).
    """
    imports: dict[str, tuple[str, str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                # Only track uppercase-starting names (classes)
                if name and name[0].isupper():
                    imports[name] = (module, alias.name)

    return imports


def _extract_constructor_calls(
    tree: ast.Module,
    known_classes: set[str],
) -> list[tuple[str, list[str], int]]:
    """Find all ClassName(...) calls in main.py where ClassName is imported.

    Returns list of (class_name, [kwarg_names], line_number).
    """
    calls: list[tuple[str, list[str], int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        callee = node.func
        name = ""
        if isinstance(callee, ast.Name):
            name = callee.id
        elif isinstance(callee, ast.Attribute):
            name = callee.attr

        if name in known_classes:
            kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
            calls.append((name, kw_names, getattr(node, "lineno", 0)))

    return calls


def _resolve_class(module_path: str, class_name: str) -> type | None:
    """Try to import and return the class. Returns None on failure."""
    try:
        mod = importlib.import_module(
            module_path
        )  # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import - test only, imports our own modules from main.py AST walk
        return getattr(mod, class_name, None)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


# Classes to skip: factory methods, not direct constructors, or
# classes where beartype_packages call would be incorrectly matched.
_SKIP_CLASSES = frozenset(
    {
        # Not a constructor: Application.builder().token(...).build()
        "Application",
        # telegram.ext handler wrappers (positional args, not kwarg-based)
        "CommandHandler",
        "MessageHandler",
        "CallbackQueryHandler",
        # Standard library / third-party classes with complex signatures
        "Path",
        "FileLock",
        # Internal factory patterns (create_default is a classmethod)
        # ContextKernel.create_default() is a classmethod, not __init__
    }
)


class TestMainWiringGeneralized:
    """Generalized guard: every kwarg in main.py constructor calls
    must match the real __init__ signature of the target class.

    This is the catch-all version of the Skill-Compression-specific
    TestSkillComponentKwargsMatchSignature. It covers ALL classes
    instantiated in main.py.
    """

    def test_all_constructor_kwargs_match_signatures(self) -> None:
        """Every kwarg in every constructor call in main.py must be valid."""
        tree = _parse_main_ast()
        imports = _extract_imports(tree)
        known_classes = set(imports.keys()) - _SKIP_CLASSES
        calls = _extract_constructor_calls(tree, known_classes)

        errors: list[str] = []

        for class_name, kw_names, lineno in calls:
            if class_name not in imports:
                continue

            module_path, real_name = imports[class_name]
            cls = _resolve_class(module_path, real_name)

            if cls is None:
                # Cannot resolve (optional dependency, conditional import)
                continue

            # Get valid parameter names from __init__
            try:
                sig = inspect.signature(cls.__init__)
            except (ValueError, TypeError):
                continue

            valid_params = set(sig.parameters.keys()) - {"self"}

            for kw in kw_names:
                if kw not in valid_params:
                    errors.append(
                        f"main.py line {lineno}: {class_name}({kw}=...) "
                        f"but {class_name}.__init__ accepts "
                        f"{sorted(valid_params)}. TypeError at runtime!"
                    )

        assert not errors, (
            "Keyword argument mismatch in main.py constructor calls:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def test_discovered_classes_are_not_empty(self) -> None:
        """Sanity check: we should find a non-trivial number of classes."""
        tree = _parse_main_ast()
        imports = _extract_imports(tree)
        known_classes = set(imports.keys()) - _SKIP_CLASSES
        calls = _extract_constructor_calls(tree, known_classes)

        # main.py has 20+ constructor calls. If we find fewer than 10,
        # something is wrong with our AST extraction.
        assert len(calls) >= 10, (
            f"Expected at least 10 constructor calls in main.py, "
            f"found {len(calls)}. AST extraction may be broken."
        )

    def test_no_unresolvable_imports(self) -> None:
        """All imported classes used in constructor calls must be resolvable.

        If a class cannot be imported, the kwarg check is silently skipped.
        This test ensures we know about any unresolvable classes.
        """
        tree = _parse_main_ast()
        imports = _extract_imports(tree)
        known_classes = set(imports.keys()) - _SKIP_CLASSES
        calls = _extract_constructor_calls(tree, known_classes)

        # Collect unique class names from calls
        called_classes = {name for name, _, _ in calls}

        unresolvable: list[str] = []
        for class_name in sorted(called_classes):
            if class_name not in imports:
                continue
            module_path, real_name = imports[class_name]
            cls = _resolve_class(module_path, real_name)
            if cls is None:
                unresolvable.append(f"{class_name} (from {module_path})")

        # Allow up to 2 unresolvable (optional deps like bertopic)
        assert len(unresolvable) <= 2, (
            f"Too many unresolvable classes in main.py constructor calls: "
            f"{unresolvable}. Check imports and optional dependencies."
        )


class TestMainFactoryMethodKwargs:
    """Guard for factory method calls like ContextKernel.create_default(...)."""

    def test_context_kernel_create_default_kwargs(self) -> None:
        """ContextKernel.create_default() kwargs must match signature."""
        tree = _parse_main_ast()
        errors: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match: ContextKernel.create_default(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "create_default"
                and isinstance(func.value, ast.Name)
                and func.value.id == "ContextKernel"
            ):
                kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]

                # Import the class and check
                try:
                    from application.execution import ContextKernel

                    sig = inspect.signature(ContextKernel.create_default)
                    valid_params = set(sig.parameters.keys()) - {"cls", "self"}
                    for kw in kw_names:
                        if kw not in valid_params:
                            errors.append(
                                f"ContextKernel.create_default({kw}=...) "
                                f"but valid params are {sorted(valid_params)}"
                            )
                except (ImportError, AttributeError) as exc:
                    errors.append(f"Cannot import ContextKernel.create_default: {exc}")

        assert not errors, "Factory method kwarg mismatch:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
