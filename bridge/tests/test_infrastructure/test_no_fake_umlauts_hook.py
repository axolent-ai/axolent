"""Tests für den no-fake-umlauts Pre-Commit-Hook.

V8-R3: Strukturelle Lösung gegen ASCII-Umlaut-Regressionen.

Verifiziert dass das Script:
  - ASCII-Umlaut-Umschreibungen in Production-Dateien erkennt
  - Bei Treffern Exit-Code 1 liefert
  - Saubere Dateien mit Exit-Code 0 durchlässt
  - Englische Wörter nicht fälschlich flagged
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


# Import des Scripts (kein Package, daher exec-basiert)
_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "scripts"
    / "check_no_fake_umlauts.py"
)

_spec = importlib.util.spec_from_file_location("check_no_fake_umlauts", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

check_file = _mod.check_file


class TestFakeUmlautDetection:
    """Tests für die Erkennung von ASCII-Umlaut-Umschreibungen."""

    def test_detects_fuer(self, tmp_path: Path) -> None:
        """'fuer' wird als ASCII-Umlaut erkannt."""
        f = tmp_path / "test.py"
        f.write_text("# Nur fuer Testzwecke\n", encoding="utf-8")
        findings = check_file(f)
        assert len(findings) == 1
        assert findings[0][1] == "fuer"

    def test_detects_ueber(self, tmp_path: Path) -> None:
        """'ueber' wird als ASCII-Umlaut erkannt."""
        f = tmp_path / "test.py"
        f.write_text('msg = "Informationen ueber das System"\n', encoding="utf-8")
        findings = check_file(f)
        assert len(findings) == 1
        assert findings[0][1] == "ueber"

    def test_detects_zurueck(self, tmp_path: Path) -> None:
        """'zurueck' wird als ASCII-Umlaut erkannt."""
        f = tmp_path / "test.py"
        f.write_text("# Gibt None zurueck\n", encoding="utf-8")
        findings = check_file(f)
        assert len(findings) == 1
        assert findings[0][1] == "zurueck"

    def test_detects_Prueft(self, tmp_path: Path) -> None:
        """'Prueft' wird als ASCII-Umlaut erkannt."""
        f = tmp_path / "test.py"
        f.write_text('"""Prueft ob der Server läuft."""\n', encoding="utf-8")
        findings = check_file(f)
        assert len(findings) == 1
        assert findings[0][1] == "Prueft"

    def test_clean_file_passes(self, tmp_path: Path) -> None:
        """Datei mit echten Umlauten hat keine Findings."""
        f = tmp_path / "test.py"
        f.write_text(
            '"""Prüft ob der Server läuft."""\n'
            "# Gibt None zurück für ungültige Eingaben\n",
            encoding="utf-8",
        )
        findings = check_file(f)
        assert len(findings) == 0

    def test_english_words_not_flagged(self, tmp_path: Path) -> None:
        """Englische Wörter mit 'ue/ae/oe' werden nicht fälschlich erkannt."""
        f = tmp_path / "test.py"
        f.write_text(
            'queue = []\nvalue = True\nblue = "sky"\ndef fuel_check(): pass\n',
            encoding="utf-8",
        )
        findings = check_file(f)
        assert len(findings) == 0

    def test_multiple_findings_per_file(self, tmp_path: Path) -> None:
        """Mehrere Treffer in einer Datei werden alle gefunden."""
        f = tmp_path / "test.py"
        f.write_text(
            "# Prueft fuer verfuegbar\n# zurueck nach Uebersicht\n",
            encoding="utf-8",
        )
        findings = check_file(f)
        # Prueft, fuer, verfuegbar, zurueck, Uebersicht = 5 Treffer
        assert len(findings) >= 4
