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
    # ae statt ä
    r"[Aa]endern|[Aa]enderung|[Aa]enderungen|[Ww]aehrung|[Ww]aehrend|"
    r"[Ss]paeter|[Ee]rklaerung|[Ee]rklaeren|[Ss]taerken|[Ss]chwaechen|"
    r"[Ss]aetze|[Ss]paeter|[Pp]raeferenz|[Nn]aechst|[Nn]aechste|"
    r"[Hh]aeufig|[Gg]efaehrlich|[Aa]ehnlich|[Ss]chaerfer|"
    r"[Ww]aehlen|[Mm]aechtig|[Gg]aengig|[Gg]aenzig|"
    r"[Ll]aesst|[Ll]aeuft|[Gg]aebe|[Hh]aelt|[Hh]aette|"
    # oe statt ö
    r"[Mm]oeglich|[Hh]oeflich|[Oo]effentlich|[Gg]roesse|[Gg]roesser|"
    r"[Vv]oellig|[Gg]ewoehnlich|[Bb]oese|[Ss]toerung|[Gg]oettin|"
    r"[Kk]oennen|[Kk]oennte|[Mm]oechte|[Hh]oeren|[Gg]ehoert|"
    r"[Ww]oertlich|[Ww]oerter|[Nn]oetig|[Zz]uhoerer|"
    # ue statt ü
    r"[Ff]uer|[Uu]eber|[Uu]eberpr|[Uu]eberpruef|[Uu]eberschreib|"
    r"[Uu]ebersicht|[Uu]ebergibt|[Uu]ebergeb|[Zz]urueck|[Zz]urueckgeb|"
    r"[Ww]uerden|[Ww]uerde|[Vv]erfuegbar|[Gg]ueltig|[Uu]ngueltig|"
    r"[Gg]ewuenscht|[Aa]usserhalb|[Nn]uetzlich|"
    r"[Ff]uehrt|[Ff]uehren|[Aa]usfuehr|[Dd]urchfuehr|[Ee]infuehr|"
    r"[Ee]ingefuehrt|[Gg]ruende|[Gg]ruenden|[Bb]egruend|"
    r"[Ss]chluessel|[Ss]tueck|[Gg]lueck|[Mm]uessen|[Mm]uede|"
    r"[Pp]rueft|[Pp]ruefen|[Gg]eprueft|[Uu]ebrigens|"
    r"[Mm]odellabhaengig|[Ee]igenstaendig|[Vv]ollstaendig|"
    # ss statt ß (selective, nur eindeutige Faelle)
    r"[Aa]usserdem|[Ss]chliessen|[Gg]emaess|[Ss]trasse"
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
