"""Tests for the en-only-production pre-commit hook.

Verifies that the hook correctly detects German text in production code
and respects whitelisted files and the noqa suppress marker.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Add scripts dir to path so we can import the hook
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from check_en_only_production import (  # noqa: E402
    REPO_ROOT,
    _is_whitelisted,
    _scan_file,
    main,
)


class TestGermanDetectionPositive:
    """Verify that German text IS detected (5 positive cases)."""

    def _write_and_scan(
        self, tmp_path: Path, content: str
    ) -> list[tuple[int, str, str]]:
        """Helper: write content to a temp .py file and scan it."""
        f = tmp_path / "sample.py"
        f.write_text(content, encoding="utf-8")
        return _scan_file(f)

    def test_detects_real_umlauts(self, tmp_path: Path) -> None:
        """Real umlauts (ä, ö, ü, ß) are detected."""
        hits = self._write_and_scan(tmp_path, "# Prüfe den Wert\nx = 1\n")
        assert len(hits) >= 1
        assert any("ü" in token for _, token, _ in hits)

    def test_detects_german_marker_words(self, tmp_path: Path) -> None:
        """Common German marker words trigger detection."""
        hits = self._write_and_scan(tmp_path, "# Diese Funktion wird nicht verwendet\n")
        assert len(hits) >= 1

    def test_detects_und_as_marker(self, tmp_path: Path) -> None:
        """The word 'und' is detected as a German marker."""
        hits = self._write_and_scan(tmp_path, "# Lese und schreibe Daten\n")
        assert len(hits) >= 1
        assert any("und" in token.lower() for _, token, _ in hits)

    def test_detects_uppercase_umlauts(self, tmp_path: Path) -> None:
        """Uppercase umlauts (Ä, Ö, Ü) are detected."""
        hits = self._write_and_scan(tmp_path, "# Änderungen speichern\n")
        assert len(hits) >= 1
        assert any("Ä" in token for _, token, _ in hits)

    def test_detects_eszett(self, tmp_path: Path) -> None:
        """The letter ß is detected."""
        hits = self._write_and_scan(tmp_path, "# Die Straße ist lang\n")
        assert len(hits) >= 1
        assert any("ß" in token for _, token, _ in hits)


class TestGermanDetectionNegative:
    """Verify that clean English text is NOT flagged (5 negative cases)."""

    def _write_and_scan(
        self, tmp_path: Path, content: str
    ) -> list[tuple[int, str, str]]:
        """Helper: write content to a temp .py file and scan it."""
        f = tmp_path / "sample.py"
        f.write_text(content, encoding="utf-8")
        return _scan_file(f)

    def test_clean_english_comment(self, tmp_path: Path) -> None:
        """Standard English comment produces no hits."""
        hits = self._write_and_scan(tmp_path, "# Read the configuration file\nx = 1\n")
        assert len(hits) == 0

    def test_clean_english_docstring(self, tmp_path: Path) -> None:
        """Standard English docstring produces no hits."""
        hits = self._write_and_scan(
            tmp_path,
            '"""Load all user profiles from the database."""\n',
        )
        assert len(hits) == 0

    def test_python_code_no_false_positive(self, tmp_path: Path) -> None:
        """Pure Python code (no comments) produces no hits."""
        code = "def process_data(items: list) -> dict:\n    return {}\n"
        hits = self._write_and_scan(tmp_path, code)
        assert len(hits) == 0

    def test_urls_are_skipped(self, tmp_path: Path) -> None:
        """Lines starting with http/https are not scanned."""
        hits = self._write_and_scan(
            tmp_path,
            "https://example.com/über/straße\n",
        )
        assert len(hits) == 0

    def test_noqa_suppresses_detection(self, tmp_path: Path) -> None:
        """Lines with '# noqa: en-only' are not flagged."""
        hits = self._write_and_scan(
            tmp_path,
            "# implements German 'duzen' behavior  # noqa: en-only\n",
        )
        assert len(hits) == 0


class TestWhitelist:
    """Verify whitelisting logic (3 whitelist tests)."""

    def test_task_slots_yaml_whitelisted(self) -> None:
        """bridge/config/task_slots.yaml is whitelisted."""
        path = REPO_ROOT / "bridge" / "config" / "task_slots.yaml"
        assert _is_whitelisted(path)

    def test_test_directory_whitelisted(self) -> None:
        """Files under bridge/tests/ are whitelisted."""
        path = REPO_ROOT / "bridge" / "tests" / "test_handlers.py"
        assert _is_whitelisted(path)

    def test_production_file_not_whitelisted(self) -> None:
        """A regular production file is NOT whitelisted."""
        path = REPO_ROOT / "bridge" / "infrastructure" / "sqlite_storage.py"
        assert not _is_whitelisted(path)

    def test_i18n_file_whitelisted(self) -> None:
        """i18n lookup files are whitelisted."""
        path = REPO_ROOT / "bridge" / "domain" / "onboarding.py"
        assert _is_whitelisted(path)


class TestMainEntryPoint:
    """Integration test for the main() entry point."""

    def test_main_returns_zero_on_clean_codebase(self) -> None:
        """Running the hook on the full codebase returns 0 (all clean)."""
        result = main(["check_en_only_production.py"])
        assert result == 0

    def test_main_returns_one_on_german_file(self, tmp_path: Path) -> None:
        """Running the hook on a file with German text returns 1."""
        de_file = tmp_path / "bad.py"
        de_file.write_text(
            '"""Diese Funktion löscht alle Daten."""\n', encoding="utf-8"
        )
        result = main(["check_en_only_production.py", str(de_file)])
        assert result == 1
