"""Obsidian umlaut checker: scans Obsidian 11_axolent folder for fake umlauts.

Scans all .md, .html, .txt files recursively for German words where
ASCII substitutes (ae/oe/ue/ss) are used instead of real umlauts.

Designed for manual execution by Atlas, NOT as a pre-commit hook
(target folder is outside the repo).

Usage:
    python scripts/check_umlauts_in_obsidian.py [--path <folder>]

Default path: D:\\Obsidian\\Command Center\\SELBSTSTAENDIG\\11_axolent\\

Exit code 0: all clean.
Exit code 1: findings detected.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Word-level pattern: German stems with ASCII umlaut substitutions.
# Covers ae->ae, oe->oe, ue->ue, ss->ss for common German words.
# Each stem uses \\w* suffix so inflected forms are caught.
# ---------------------------------------------------------------------------

_FAKE_UMLAUT_PATTERN = re.compile(
    r"\b("
    # ae statt ae
    r"[Aa]endern\w*|[Aa]enderung\w*|[Aa]ehnlich\w*|"
    r"[Bb]estaetigung\w*|"
    r"[Ee]igenstae?ndig\w*|[Ee]nthaelt\w*|[Ee]rklaer\w*|"
    r"[Gg]aeb\w*|[Gg]aengig\w*|[Gg]efaehrlich\w*|"
    r"[Hh]aelt\w*|[Hh]aett\w*|[Hh]aeufig\w*|"
    r"[Ll]aedt\w*|[Ll]aesst\w*|[Ll]aeuft\w*|"
    r"[Mm]aechtig\w*|"
    r"[Nn]aechst\w*|"
    r"[Pp]raeferenz\w*|[Pp]raefix\w*|[Pp]raezis\w*|[Pp]rioritaet\w*|"
    r"[Qq]ualitaets\w*|"
    r"[Ss]aetz\w*|[Ss]aeule\w*|[Ss]chaerfer\w*|[Ss]chwaeche\w*|[Ss]paeter\w*|[Ss]taerke\w*|"
    r"[Tt]aeglich\w*|"
    r"[Vv]ollstaendig\w*|[Vv]orschlaeg\w*|"
    r"[Ww]aehl\w*|[Ww]aehrend\w*|[Ww]aehrung\w*|"
    # oe statt oe
    r"[Bb]oese\w*|"
    r"[Gg]ehoer\w*|[Gg]eloescht\w*|[Gg]ewoehnlich\w*|[Gg]roess\w*|"
    r"[Hh]oeflich\w*|[Hh]oer\w*|"
    r"[Kk]oenn\w*|"
    r"[Ll]oeschung\w*|[Ll]oesung\w*|"
    r"[Mm]oeglich\w*|[Mm]oecht\w*|"
    r"[Nn]oetig\w*|"
    r"[Oo]effentlich\w*|"
    r"[Pp]ersoenlich\w*|"
    r"[Ss]toerung\w*|"
    r"[Uu]ebersicht\w*|"
    r"[Vv]oellig\w*|"
    r"[Ww]oert\w*|"
    # ue statt ue
    r"[Aa]usfuehr\w*|"
    r"[Bb]egruend\w*|"
    r"[Dd]urchfuehr\w*|"
    r"[Ee]infuehr\w*|[Ee]ingefuehr\w*|"
    r"[Ff]uehr\w*|[Ff]uer\b|"
    r"[Gg]eprueft\w*|[Gg]ewuenscht\w*|[Gg]lueck\w*|[Gg]ruend\w*|"
    r"[Gg]ueltig\w*|"
    r"[Mm]ued\w*|[Mm]uess\w*|"
    r"[Nn]uetzlich\w*|"
    r"[Pp]ruef\w*|"
    r"[Ss]chluessel\w*|[Ss]tueck\w*|"
    r"[Uu]eber\w*|"
    r"[Uu]ebrig\w*|[Uu]ngueltig\w*|"
    r"[Vv]erfuegbar\w*|"
    r"[Ww]uerd\w*|"
    r"[Zz]urueck\w*|"
    # ss statt ss (selective: only unambiguous German words)
    r"[Aa]usserdem\w*|[Aa]usserhalb\w*|[Gg]emaess\w*|"
    r"[Ss]chliess\w*|[Ss]trass\w*|[Gg]ruesse\w*|"
    r"[Bb]egruessung\w*"
    r")\b",
    re.UNICODE,
)

# Suggestion map: fake prefix -> correct replacement
_SUGGESTIONS: dict[str, str] = {
    "ae": "ae",
    "oe": "oe",
    "ue": "ue",
    "ss": "ss",
}

# File extensions to scan
_SCAN_EXTENSIONS: frozenset[str] = frozenset({".md", ".html", ".txt", ".htm"})

# Lines containing these markers are skipped (code blocks, URLs, CSS)
_SKIP_MARKERS: tuple[str, ...] = (
    "```",
    "http://",
    "https://",
    "font-family",
    "border-radius",
    "box-shadow",
    "margin",
    "padding",
    "background",
    "color:",
    "display:",
    "position:",
    "width:",
    "height:",
    "overflow:",
    "transition:",
)


def _suggest_fix(word: str) -> str:
    """Generate a correction suggestion for a fake-umlaut word.

    Simple heuristic: replace the first ae/oe/ue/ss occurrence
    with the real umlaut character.

    Args:
        word: The detected word with fake umlaut.

    Returns:
        Suggested correction.
    """
    lower = word.lower()
    replacements = [
        ("ae", "ä"),
        ("oe", "ö"),
        ("ue", "ü"),
    ]
    result = word
    for fake, real in replacements:
        if fake in lower:
            # Case-aware replacement
            idx = lower.find(fake)
            if idx >= 0:
                original_chars = result[idx : idx + 2]
                if original_chars[0].isupper():
                    real_char = real.upper()
                else:
                    real_char = real
                result = result[:idx] + real_char + result[idx + 2 :]
                break
    # Handle ss -> ss (only for specific stems like Strass -> Strass)
    if result == word and "ss" in lower:
        idx = lower.find("ss")
        if idx > 0:
            result = result[:idx] + "ß" + result[idx + 2 :]
    return result


def scan_file(filepath: Path) -> list[tuple[int, str, str, str]]:
    """Scan a single file for fake umlaut words.

    Args:
        filepath: Path to the file.

    Returns:
        List of (line_number, found_word, suggestion, line_preview).
    """
    findings: list[tuple[int, str, str, str]] = []
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    for line_no, line in enumerate(text.splitlines(), start=1):
        # Skip lines that look like code/CSS/URLs
        stripped = line.strip()
        if any(marker in stripped.lower() for marker in _SKIP_MARKERS):
            continue

        for match in _FAKE_UMLAUT_PATTERN.finditer(line):
            word = match.group(0)
            suggestion = _suggest_fix(word)
            preview = stripped[:120]
            findings.append((line_no, word, suggestion, preview))

    return findings


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check Obsidian 11_axolent folder for fake German umlauts."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=r"D:\Obsidian\Command Center\SELBSTSTÄNDIG\11_axolent",
        help="Root folder to scan (default: 11_axolent in Obsidian)",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"ERROR: Path does not exist: {root}", file=sys.stderr)
        return 1

    all_findings: list[tuple[str, int, str, str, str]] = []

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in _SCAN_EXTENSIONS:
            continue

        for line_no, word, suggestion, preview in scan_file(filepath):
            rel_path = str(filepath.relative_to(root))
            all_findings.append((rel_path, line_no, word, suggestion, preview))

    if not all_findings:
        print(f"OK: No fake umlauts found in {root}")
        return 0

    print(f"\n{'=' * 78}")
    print(f"  FAKE UMLAUTS FOUND: {len(all_findings)} occurrence(s)")
    print(f"{'=' * 78}\n")

    current_file = ""
    for rel_path, line_no, word, suggestion, preview in all_findings:
        if rel_path != current_file:
            current_file = rel_path
            print(f"--- {rel_path} ---")
        print(f"  Line {line_no}: '{word}' -> '{suggestion}'")
        print(f"    {preview}")
        print()

    print(f"Total: {len(all_findings)} fake umlaut(s) in {root}")
    print("Fix: Replace ASCII substitutes with real umlauts.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
