"""Pre-Commit Hook: erkennt ASCII-Umlaut-Umschreibungen in Production-Code.

Verhindert Regressionen wie fuer/ueber/zurueck/Prueft/etc. in deutschem Text.
Nur fuer Production-Dateien (Tests sind ausgenommen, separate Iteration).

Exit-Code 0: alles sauber.
Exit-Code 1: Treffer gefunden (blockiert Commit).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Pattern: typische deutsche Woerter mit ASCII-Umlaut-Umschreibung.
# Erfasst sowohl Gross- als auch Kleinschreibung.
_FAKE_UMLAUT_PATTERNS = re.compile(
    r"\b("
    # ae statt ä  (alphabetisch sortiert)
    r"[Aa]ehnlich|[Aa]endern|[Aa]enderung|[Aa]enderungen|"
    r"[Ee]nthaelt|[Ee]rklaeren|[Ee]rklaerung|"
    r"[Gg]aebe|[Gg]aengig|[Gg]aenzig|[Gg]efaehrlich|"
    r"[Hh]aelt|[Hh]aette|[Hh]aeufig|"
    r"[Ll]aedt|[Ll]aesst|[Ll]aeuft|"
    r"[Mm]aechtig|"
    r"[Nn]aechst|[Nn]aechste|"
    r"[Pp]raeferenz|[Pp]raefix|[Pp]raefixe|[Pp]rioritaet|[Pp]rioritaeten|"
    r"[Ss]aetze|[Ss]chaerfer|[Ss]chwaechen|[Ss]paeter|[Ss]taerken|"
    r"[Vv]orschlaege|"
    r"[Ww]aehlen|[Ww]aehrend|[Ww]aehrung|"
    # oe statt ö  (alphabetisch sortiert)
    r"[Bb]oese|"
    r"[Gg]ehoert|[Gg]eloescht|[Gg]eloeschte|[Gg]eloeschter|[Gg]eloeschtes|"
    r"[Gg]ewoehnlich|[Gg]oettin|[Gg]roesse|[Gg]roesser|"
    r"[Hh]oeflich|[Hh]oeren|"
    r"[Kk]oennen|[Kk]oennte|"
    r"[Mm]oeglich|[Mm]oechte|"
    r"[Nn]oetig|"
    r"[Oo]effentlich|"
    r"[Ss]toerung|"
    r"[Vv]oellig|"
    r"[Ww]oerter|[Ww]oertlich|"
    r"[Zz]uhoerer|"
    # ue statt ü  (alphabetisch sortiert)
    r"[Aa]usfuehr|"
    r"[Bb]egruend|"
    r"[Dd]urchfuehr|"
    r"[Ee]infuehr|[Ee]ingefuehrt|"
    r"[Ff]uehren|[Ff]uehrt|[Ff]uer|"
    r"[Gg]eprueft|[Gg]ewuenscht|[Gg]lueck|[Gg]ruende|[Gg]ruenden|"
    r"[Gg]ueltig|"
    r"[Mm]odellabhaengig|[Mm]uede|[Mm]uessen|"
    r"[Nn]uetzlich|"
    r"[Pp]ruefen|[Pp]rueft|"
    r"[Ss]chluessel|[Ss]tueck|"
    r"[Uu]eber|[Uu]ebergeb|[Uu]ebergibt|[Uu]eberpr|[Uu]eberpruef|"
    r"[Uu]eberschreib|[Uu]ebersetz|[Uu]ebersetze|[Uu]ebersicht|"
    r"[Uu]ebersprungen|[Uu]ebrigens|[Uu]ngueltig|"
    r"[Vv]erfuegbar|[Vv]ollstaendig|"
    r"[Ww]uerde|[Ww]uerden|"
    r"[Zz]urueck|[Zz]urueckgeb|[Zz]uruecksetzen|"
    r"[Ee]igenstaendig|[Aa]usserhalb|"
    # ss statt ß  (selektiv, nur eindeutige Fälle)
    r"[Aa]usserdem|[Gg]emaess|[Ss]chliessen|[Ss]trasse"
    r")\b",
    re.UNICODE,
)

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
        # Inline-Suppress: Zeilen mit "# noqa: fake-umlaut" ueberspringen
        if "# noqa: fake-umlaut" in line:
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
