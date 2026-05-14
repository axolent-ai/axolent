"""Text Guard CLI: check and fix diacritic issues in files.

Usage:
    python -m domain.text_guard check <file_or_dir> [--lang <code>]
    python -m domain.text_guard fix <file_or_dir> [--lang <code>]
    python -m domain.text_guard report <directory> [--lang <code>]
    python -m domain.text_guard languages

Examples:
    python -m domain.text_guard check document.md --lang de
    python -m domain.text_guard fix ./content/ --lang fr
    python -m domain.text_guard report ./docs/ --lang de
    python -m domain.text_guard languages
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from domain.text_guard import get_builtin_rules, list_languages
from domain.text_guard.guard import TextGuard
from domain.text_guard.models import Issue

# File extensions the CLI processes
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".html",
        ".htm",
        ".xml",
        ".csv",
    }
)


def _collect_files(path: Path) -> list[Path]:
    """Collect all supported files from a path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            return [path]
        return []
    if path.is_dir():
        files: list[Path] = []
        for ext in _SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
        return sorted(files)
    return []


def _check_file(filepath: Path, guard: TextGuard) -> list[tuple[Path, Issue]]:
    """Check a single file and return issues with file path."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  Error reading {filepath}: {exc}", file=sys.stderr)
        return []

    issues = guard.check(content)
    return [(filepath, issue) for issue in issues]


def _fix_file(filepath: Path, guard: TextGuard) -> bool:
    """Fix a single file in-place. Returns True if changes made."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  Error reading {filepath}: {exc}", file=sys.stderr)
        return False

    fixed = guard.fix(content)
    if fixed != content:
        filepath.write_text(fixed, encoding="utf-8")
        return True
    return False


def cmd_check(args: argparse.Namespace) -> int:
    """Run check mode: report issues without modifying files."""
    rules = get_builtin_rules(args.lang)
    if rules is None:
        print(f"No rules for language: {args.lang}", file=sys.stderr)
        return 1

    guard = TextGuard(rules, mode="check")
    target = Path(args.target)
    files = _collect_files(target)

    if not files:
        print(f"No supported files found in: {target}")
        return 0

    total_issues: list[tuple[Path, Issue]] = []
    for filepath in files:
        total_issues.extend(_check_file(filepath, guard))

    if not total_issues:
        print(f"All clean. Checked {len(files)} file(s), language={args.lang}.")
        return 0

    print(f"\n{'=' * 70}")
    print(f"  Text Guard: {len(total_issues)} issue(s) in {len(files)} file(s)")
    print(f"{'=' * 70}\n")

    for filepath, issue in total_issues:
        print(f"  {filepath}:{issue.line}:{issue.column}")
        print(f"    Found: '{issue.ascii_form}' -> '{issue.correct_form}'")
        print(f"    Line:  {issue.excerpt[:120]}")
        print()

    return 1


def cmd_fix(args: argparse.Namespace) -> int:
    """Run fix mode: correct issues in-place."""
    rules = get_builtin_rules(args.lang)
    if rules is None:
        print(f"No rules for language: {args.lang}", file=sys.stderr)
        return 1

    guard = TextGuard(rules, mode="fix")
    target = Path(args.target)
    files = _collect_files(target)

    if not files:
        print(f"No supported files found in: {target}")
        return 0

    fixed_count = 0
    for filepath in files:
        if _fix_file(filepath, guard):
            print(f"  Fixed: {filepath}")
            fixed_count += 1

    print(f"\nDone. {fixed_count}/{len(files)} file(s) modified, language={args.lang}.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Run report mode: detailed report of all issues."""
    rules = get_builtin_rules(args.lang)
    if rules is None:
        print(f"No rules for language: {args.lang}", file=sys.stderr)
        return 1

    guard = TextGuard(rules, mode="check")
    target = Path(args.target)
    files = _collect_files(target)

    if not files:
        print(f"No supported files found in: {target}")
        return 0

    # Group issues by file
    file_issues: dict[Path, list[Issue]] = {}
    for filepath in files:
        issues = guard.check(filepath.read_text(encoding="utf-8", errors="replace"))
        if issues:
            file_issues[filepath] = issues

    print(f"\n{'=' * 70}")
    print("  Text Guard Report")
    print(f"  Language: {args.lang} | Files scanned: {len(files)}")
    print(f"  Files with issues: {len(file_issues)}")
    total = sum(len(v) for v in file_issues.values())
    print(f"  Total issues: {total}")
    print(f"{'=' * 70}\n")

    for filepath, issues in file_issues.items():
        print(f"  {filepath} ({len(issues)} issue(s))")
        for issue in issues:
            print(
                f"    L{issue.line}:C{issue.column} "
                f"'{issue.ascii_form}' -> '{issue.correct_form}'"
            )
        print()

    return 0 if not file_issues else 1


def cmd_languages(_args: argparse.Namespace) -> int:
    """List available languages."""
    langs = list_languages()
    print("Available Text Guard languages:")
    for lang in langs:
        rules = get_builtin_rules(lang)
        count = len(rules.word_pairs) if rules else 0
        print(f"  {lang}: {count} word pairs")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="text-guard",
        description="Text Guard: fix ASCII diacritic substitutions in text files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check
    p_check = subparsers.add_parser("check", help="Check files for issues")
    p_check.add_argument("target", help="File or directory to check")
    p_check.add_argument("--lang", default="de", help="Language code (default: de)")
    p_check.set_defaults(func=cmd_check)

    # fix
    p_fix = subparsers.add_parser("fix", help="Fix issues in-place")
    p_fix.add_argument("target", help="File or directory to fix")
    p_fix.add_argument("--lang", default="de", help="Language code (default: de)")
    p_fix.set_defaults(func=cmd_fix)

    # report
    p_report = subparsers.add_parser("report", help="Generate detailed report")
    p_report.add_argument("target", help="Directory to scan")
    p_report.add_argument("--lang", default="de", help="Language code (default: de)")
    p_report.set_defaults(func=cmd_report)

    # languages
    p_lang = subparsers.add_parser("languages", help="List available languages")
    p_lang.set_defaults(func=cmd_languages)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
