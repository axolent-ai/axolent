"""Pre-commit hook: detects ASCII umlaut substitutions in production code.

Prevents regressions like fuer/ueber/zurueck/praezise/etc. in German text.
Only for production files (tests are excluded, separate iteration).

Structural approach: word stems with ``\\w*`` suffix instead of individual word forms,
so inflections (uebergeben, uebergibt, uebergabe etc.) are reliably detected.

Exit code 0: all clean.
Exit code 1: matches found (blocks commit).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Pattern: German word stems with ASCII umlaut substitutions.
# Each stem ends with \w* instead of \b so inflections match reliably.
# Catches both upper and lower case.
_FAKE_UMLAUT_PATTERNS = re.compile(
    r"\b("
    # ae instead of ä (alphabetically sorted by stem)
    r"[Aa]ehnlich\w*|[Aa]endern\w*|[Aa]enderung\w*|"
    r"[Ee]igenstae?ndig\w*|[Ee]nthaelt\w*|[Ee]rklaer\w*|"
    r"[Gg]aeb\w*|[Gg]aengig\w*|[Gg]aenzig\w*|[Gg]efaehrlich\w*|"
    r"[Hh]aelt\w*|[Hh]aett\w*|[Hh]aeufig\w*|"
    r"[Ll]aedt\w*|[Ll]aesst\w*|[Ll]aeuft\w*|"
    r"[Mm]aechtig\w*|[Mm]odellabhaengig\w*|"
    r"[Nn]aechst\w*|"
    r"[Pp]raeferenz\w*|[Pp]raefix\w*|[Pp]raezis\w*|[Pp]rioritaet\w*|"
    r"[Ss]aetz\w*|[Ss]chaerfer\w*|[Ss]chwaeche\w*|[Ss]paeter\w*|[Ss]taerke\w*|"
    r"[Vv]ollstaendig\w*|[Vv]orschlaeg\w*|"
    r"[Ww]aehl\w*|[Ww]aehrend\w*|[Ww]aehrung\w*|"
    # oe instead of ö (alphabetically sorted by stem)
    r"[Bb]oese\w*|"
    r"[Gg]ehoer\w*|[Gg]eloescht\w*|"
    r"[Gg]ewoehnlich\w*|[Gg]oettin\w*|[Gg]roess\w*|"
    r"[Hh]oeflich\w*|[Hh]oer\w*|"
    r"[Kk]oenn\w*|"
    r"[Mm]oeglich\w*|[Mm]oecht\w*|"
    r"[Nn]oetig\w*|"
    r"[Oo]effentlich\w*|"
    r"[Ss]toerung\w*|"
    r"[Vv]oellig\w*|"
    r"[Ww]oert\w*|"
    r"[Zz]uhoerer\w*|"
    # ue instead of ü (alphabetically sorted by stem)
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
    # ss instead of ß (selective, only unambiguous cases)
    r"[Aa]usserdem\w*|[Aa]usserhalb\w*|[Gg]emaess\w*|[Ss]chliess\w*|[Ss]trass\w*"
    r")\b",
    re.UNICODE,
)

# Suppress token: lines containing this string are skipped.
# Composed so ruff does not flag it (ruff recognizes "# noqa: X" as a directive).
_SUPPRESS_TOKEN = "# noqa: " + "fake-umlaut"

# Whitelist: files/patterns that are NOT checked
_EXCLUDED_PATHS = {
    "scripts/check_no_fake_umlauts.py",  # This script itself
    # Text Guard module: word pairs, docstring examples, and YAML rule files
    # contain ASCII diacritic forms by design (they ARE the detection targets).
    "domain/text_guard/",
    # DIACRITIC RULE hints contain intentional ASCII negative examples
    # (e.g. "'für' not 'fuer'") to prime the LLM.
    "domain/personality.py",
}


def _is_excluded(path: Path) -> bool:
    """Checks whether a path is excluded."""
    posix = path.as_posix()
    for excluded in _EXCLUDED_PATHS:
        if posix.endswith(excluded) or excluded in posix:
            return True
    return False


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Checks a file for ASCII umlaut substitutions.

    Returns:
        List of (line_number, found_word, line).
    """
    if _is_excluded(filepath):
        return []

    findings: list[tuple[int, str, str]] = []
    try:
        text = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    for line_no, line in enumerate(text.splitlines(), start=1):
        # Inline suppress: skip lines with the suppress token
        if _SUPPRESS_TOKEN in line:
            continue
        # Comments and strings treated equally (everything is production text)
        for match in _FAKE_UMLAUT_PATTERNS.finditer(line):
            findings.append((line_no, match.group(0), line.strip()))

    return findings


def main() -> int:
    """Main function: checks all passed files."""
    files = sys.argv[1:]
    if not files:
        return 0

    total_findings: list[tuple[str, int, str, str]] = []
    for filepath_str in files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            continue
        for line_no, word, line in check_file(filepath):
            total_findings.append((filepath_str, line_no, word, line))

    if total_findings:
        print(f"\n{'=' * 70}")
        print(f"  no-fake-umlauts: {len(total_findings)} ASCII umlaut(s) found!")
        print(f"{'=' * 70}\n")
        for filepath_str, line_no, word, line in total_findings:
            print(f"  {filepath_str}:{line_no}")
            print(f"    Found: '{word}'")
            print(f"    Line:  {line[:120]}")
            print()
        print("Please use real umlauts: ae->ä, oe->ö, ue->ü, ss->ß")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
