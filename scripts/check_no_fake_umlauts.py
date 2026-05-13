"""Pre-Commit Hook: erkennt ASCII-Umlaut-Umschreibungen in Production-Code.

Verhindert Regressionen wie fuer/ueber/zurueck/praezise/etc. in deutschem Text.
Nur fuer Production-Dateien (Tests sind ausgenommen, separate Iteration).

Strukturelle Lösung: Wortstämme mit ``\\w*`` Suffix statt einzelner Wortformen,
damit Flexionen (uebergeben, uebergibt, uebergabe etc.) zuverlässig erkannt werden.

Exit-Code 0: alles sauber.
Exit-Code 1: Treffer gefunden (blockiert Commit).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Pattern: deutsche Wortstämme mit ASCII-Umlaut-Umschreibung.
# Jeder Stamm endet auf \w* statt \b, damit Flexionen zuverlässig matchen.
# Erfasst sowohl Groß- als auch Kleinschreibung.
_FAKE_UMLAUT_PATTERNS = re.compile(
    r"\b("
    # ae statt ä  (alphabetisch nach Stamm sortiert)
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
    # oe statt ö  (alphabetisch nach Stamm sortiert)
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
    # ue statt ü  (alphabetisch nach Stamm sortiert)
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
    # ss statt ß  (selektiv, nur eindeutige Fälle)
    r"[Aa]usserdem\w*|[Aa]usserhalb\w*|[Gg]emaess\w*|[Ss]chliess\w*|[Ss]trass\w*"
    r")\b",
    re.UNICODE,
)

# Suppress-Token: Zeilen die diesen String enthalten werden uebersprungen.
# Zusammengesetzt damit ruff nicht warnt (ruff erkennt "# noqa: X" als Directive).
_SUPPRESS_TOKEN = "# noqa: " + "fake-umlaut"

# Whitelist: Dateien/Patterns die NICHT geprueft werden
_EXCLUDED_PATHS = {
    "scripts/check_no_fake_umlauts.py",  # Dieses Script selbst
}


def _is_excluded(path: Path) -> bool:
    """Prueft ob ein Pfad ausgeschlossen ist."""
    posix = path.as_posix()
    for excluded in _EXCLUDED_PATHS:
        if posix.endswith(excluded):
            return True
    return False


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Prueft eine Datei auf ASCII-Umlaut-Umschreibungen.

    Returns:
        Liste von (Zeilennummer, gefundenes Wort, Zeile).
    """
    if _is_excluded(filepath):
        return []

    findings: list[tuple[int, str, str]] = []
    try:
        text = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    for line_no, line in enumerate(text.splitlines(), start=1):
        # Inline-Suppress: Zeilen mit dem Suppress-Token ueberspringen
        if _SUPPRESS_TOKEN in line:
            continue
        # Kommentare und Strings gleich behandeln (alles ist Production-Text)
        for match in _FAKE_UMLAUT_PATTERNS.finditer(line):
            findings.append((line_no, match.group(0), line.strip()))

    return findings


def main() -> int:
    """Hauptfunktion: prueft alle uebergebenen Dateien."""
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
        print(f"  no-fake-umlauts: {len(total_findings)} ASCII-Umlaut(e) gefunden!")
        print(f"{'=' * 70}\n")
        for filepath_str, line_no, word, line in total_findings:
            print(f"  {filepath_str}:{line_no}")
            print(f"    Gefunden: '{word}'")
            print(f"    Zeile:    {line[:120]}")
            print()
        print("Bitte echte Umlaute verwenden: ae->ä, oe->ö, ue->ü, ss->ß")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
