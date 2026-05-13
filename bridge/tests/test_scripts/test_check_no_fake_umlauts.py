"""Tests for the no-fake-umlauts pre-commit hook.

Verifies that the hook correctly detects ASCII-umlaut substitutions
across all known word stems and their flexions, preventing pattern regressions.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Add scripts dir to path so we can import the hook
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from check_no_fake_umlauts import check_file  # noqa: E402


class TestFakeUmlautDetection:
    """Verifies the hook detects all known fake-umlaut patterns."""

    def _write_and_check(self, tmp_path: Path, content: str) -> list[str]:
        """Helper: write content to a temp .py file and return found words."""
        f = tmp_path / "test_sample.py"
        f.write_text(content, encoding="utf-8")
        findings = check_file(f)
        return [word for _, word, _ in findings]

    def test_uebergeben_flexion(self, tmp_path: Path) -> None:
        """uebergeben (flexion of ueberg-Stamm) must be detected."""
        words = self._write_and_check(tmp_path, 'msg = "Wird uebergeben"')
        assert any("uebergeb" in w.lower() for w in words)

    def test_uebergibt_flexion(self, tmp_path: Path) -> None:
        """uebergibt must be detected."""
        words = self._write_and_check(tmp_path, 'msg = "Er uebergibt es"')
        assert any("ueberg" in w.lower() for w in words)

    def test_praezise(self, tmp_path: Path) -> None:
        """praezise must be detected."""
        words = self._write_and_check(tmp_path, 'x = "praezise Angabe"')
        assert "praezise" in [w.lower() for w in words]

    def test_praezision(self, tmp_path: Path) -> None:
        """Praezision must be detected (flexion of praezis-Stamm)."""
        words = self._write_and_check(tmp_path, 'x = "Praezision ist wichtig"')
        assert any("praezis" in w.lower() for w in words)

    def test_praefix(self, tmp_path: Path) -> None:
        """Praefix must be detected."""
        words = self._write_and_check(tmp_path, 'x = "Das Praefix"')
        assert any("praefix" in w.lower() for w in words)

    def test_geloeschter(self, tmp_path: Path) -> None:
        """geloeschter (flexion) must be detected."""
        words = self._write_and_check(tmp_path, 'x = "ein geloeschter Eintrag"')
        assert any("geloescht" in w.lower() for w in words)

    def test_fuer_standalone(self, tmp_path: Path) -> None:
        """fuer as standalone word must be detected."""
        words = self._write_and_check(tmp_path, 'x = "Das ist fuer dich"')
        assert "fuer" in [w.lower() for w in words]

    def test_ausfuehrlich(self, tmp_path: Path) -> None:
        """ausfuehrlich (flexion of ausfuehr-) must be detected."""
        words = self._write_and_check(tmp_path, 'x = "ausfuehrlich erklaert"')
        assert any("ausfuehr" in w.lower() for w in words)

    def test_schliessen(self, tmp_path: Path) -> None:
        """schliessen (ss statt ß) must be detected."""
        words = self._write_and_check(tmp_path, 'x = "Fenster schliessen"')
        assert any("schliess" in w.lower() for w in words)

    def test_strasse(self, tmp_path: Path) -> None:
        """Strasse must be detected."""
        words = self._write_and_check(tmp_path, 'x = "Die Strasse"')
        assert any("strass" in w.lower() for w in words)

    def test_multiple_on_one_line(self, tmp_path: Path) -> None:
        """Multiple fake umlauts on one line all get detected."""
        words = self._write_and_check(tmp_path, 'x = "fuer uebergeben praezise"')
        assert len(words) >= 3

    def test_clean_file_no_findings(self, tmp_path: Path) -> None:
        """File with correct umlauts produces no findings."""
        words = self._write_and_check(
            tmp_path,
            'x = "für übergeben präzise gelöscht gültig Präfix"',
        )
        assert words == []

    def test_noqa_suppression(self, tmp_path: Path) -> None:
        """Lines with # noqa: fake-umlaut are skipped."""
        words = self._write_and_check(
            tmp_path,
            'x = "fuer"  # noqa: fake-umlaut',
        )
        assert words == []

    def test_self_exclusion(self) -> None:
        """The hook script itself is excluded from checking."""
        hook_path = _SCRIPTS_DIR / "check_no_fake_umlauts.py"
        findings = check_file(hook_path)
        assert findings == []

    def test_uebersetzt_flexion(self, tmp_path: Path) -> None:
        r"""uebersetzt must be detected (via ueber-Stamm + \w*)."""
        words = self._write_and_check(tmp_path, 'x = "wird uebersetzt"')
        assert any("ueber" in w.lower() for w in words)

    def test_ungueltige_flexion(self, tmp_path: Path) -> None:
        """ungueltige must be detected."""
        words = self._write_and_check(tmp_path, 'x = "ungueltige Eingabe"')
        assert any("ungueltig" in w.lower() for w in words)

    def test_zuruecksetzen(self, tmp_path: Path) -> None:
        """zuruecksetzen must be detected."""
        words = self._write_and_check(tmp_path, 'x = "Werte zuruecksetzen"')
        assert any("zurueck" in w.lower() for w in words)

    def test_vollstaendige_flexion(self, tmp_path: Path) -> None:
        """vollstaendige must be detected."""
        words = self._write_and_check(tmp_path, 'x = "vollstaendige Liste"')
        assert any("vollstaendig" in w.lower() for w in words)

    def test_erklaerung(self, tmp_path: Path) -> None:
        """Erklaerung must be detected via erklaer-Stamm."""
        words = self._write_and_check(tmp_path, 'x = "Eine Erklaerung"')
        assert any("erklaer" in w.lower() for w in words)

    def test_prioritaeten_flexion(self, tmp_path: Path) -> None:
        """Prioritaeten must be detected."""
        words = self._write_and_check(tmp_path, 'x = "Prioritaeten setzen"')
        assert any("prioritaet" in w.lower() for w in words)
