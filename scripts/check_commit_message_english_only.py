"""Pre-commit hook (commit-msg stage): block German text in commit messages.

Detects:
  - German umlauts: ae, oe, ue, ss (as standalone words)
  - Actual umlaut characters: [aouAOU]  and ss
  - Common German stop-words (case-insensitive)

Whitelists:
  - Git trailers (Co-Authored-By, Signed-off-by, etc.)
  - Fenced code blocks (```...```)
  - Inline code in backticks (`...`)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# German stop-words that are unlikely in English commit messages.
# Each must be a complete word (\b boundaries).
GERMAN_STOPWORDS: set[str] = {
    "auch",
    "können",
    "müssen",
    "während",
    "werden",
    "sollte",
    "nicht",
    "wenn",
    "wird",
    "geht",
    "ist",
    "und",
    "das",
    "für",
    "über",
    "aber",
    "wir",
    "euch",
    "sie",
    "sind",
    "haben",
}
# NOTE: "die", "hat", "der" removed to avoid English false-positives:
#   "die" (EN: to die, die-cast), "hat" (EN: a hat), "der" (EN: name/suffix).
# Kept: "ist" (EN "specialist" has \b protection; standalone "ist" is rare),
#   "sie" (extremely rare standalone in EN), "das" (rare standalone in EN).

# Regex: umlaut characters
RE_UMLAUTS = re.compile(r"[äöüÄÖÜß]")

# Regex: ASCII substitutes for umlauts as standalone words
# "ae", "oe", "ue" as full words; "ss" as a full word (not inside words like "pass")
RE_ASCII_UMLAUTS = re.compile(r"\b(ae|oe|ue|ss)\b", re.IGNORECASE)

# Regex: German stop-words (built dynamically)
RE_STOPWORDS = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(GERMAN_STOPWORDS)) + r")\b",
    re.IGNORECASE,
)

# Trailer prefixes to ignore (case-insensitive start of line)
TRAILER_PREFIXES: tuple[str, ...] = (
    "co-authored-by:",
    "signed-off-by:",
    "reviewed-by:",
    "acked-by:",
    "tested-by:",
    "reported-by:",
    "helped-by:",
    "cc:",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_inline_code(line: str) -> str:
    """Remove inline backtick-quoted segments from a line."""
    return re.sub(r"`[^`]*`", "", line)


def is_trailer(line: str) -> bool:
    """Check if a line is a Git trailer."""
    lower = line.lower().strip()
    return any(lower.startswith(prefix) for prefix in TRAILER_PREFIXES)


def is_comment(line: str) -> bool:
    """Check if a line is a Git comment (starts with #)."""
    return line.lstrip().startswith("#")


def check_line(line: str, line_number: int) -> list[str]:
    """Check a single line for German indicators. Returns list of issues."""
    issues: list[str] = []

    # Strip inline code before checking
    cleaned = strip_inline_code(line)

    # Check umlauts
    match = RE_UMLAUTS.search(cleaned)
    if match:
        issues.append(
            f"  Line {line_number}: umlaut character '{match.group()}' found in: {line.strip()}"
        )

    # Check ASCII umlaut substitutes
    match = RE_ASCII_UMLAUTS.search(cleaned)
    if match:
        issues.append(
            f"  Line {line_number}: ASCII umlaut substitute '{match.group()}' found in: {line.strip()}"
        )

    # Check German stop-words
    match = RE_STOPWORDS.search(cleaned)
    if match:
        issues.append(
            f"  Line {line_number}: German stop-word '{match.group()}' found in: {line.strip()}"
        )

    return issues


def check_commit_message(filepath: str) -> list[str]:
    """Check entire commit message file. Returns list of issues."""
    path = Path(filepath)
    if not path.exists():
        return [f"ERROR: Commit message file not found: {filepath}"]

    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    issues: list[str] = []
    in_code_block = False

    for i, line in enumerate(lines, start=1):
        # Toggle code block state
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        # Skip code blocks
        if in_code_block:
            continue

        # Skip comments
        if is_comment(line):
            continue

        # Skip trailers
        if is_trailer(line):
            continue

        # Skip empty lines
        if not line.strip():
            continue

        issues.extend(check_line(line, i))

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_commit_message_english_only.py <commit-msg-file>")
        return 1

    filepath = sys.argv[1]
    issues = check_commit_message(filepath)

    if issues:
        print(
            "\n[english-only-commit-msg] BLOCKED: German text detected in commit message!\n"
        )
        print("Issues found:")
        for issue in issues:
            print(issue)
        print("\nPlease write your commit message in English only.")
        print(
            "Hint: Trailers (Co-Authored-By, etc.) and code blocks are whitelisted.\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
