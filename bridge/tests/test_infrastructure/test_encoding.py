"""Tests for infrastructure.encoding: UTF-8 helper.

Tests that all I/O operations consistently use UTF-8
and that Unicode characters are correctly preserved.
"""

import json
from pathlib import Path

from infrastructure.encoding import (
    append_jsonl_utf8,
    open_utf8,
    read_json_utf8,
    run_subprocess_utf8,
    write_json_utf8,
)


class TestOpenUtf8:
    """open_utf8 Helper-Tests."""

    def test_open_utf8_reads_umlauts(self, tmp_path: Path) -> None:
        """Umlaute und Sonderzeichen werden korrekt gelesen."""
        test_file = tmp_path / "umlauts.txt"
        test_file.write_text("Hallo aeoeueAeOeUe und ss", encoding="utf-8")

        with open_utf8(test_file, "r") as f:
            content = f.read()
        assert "aeoeue" in content
        assert "ss" in content

    def test_open_utf8_writes_unicode(self, tmp_path: Path) -> None:
        """Unicode-Zeichen werden korrekt geschrieben."""
        test_file = tmp_path / "unicode.txt"
        with open_utf8(test_file, "w") as f:
            f.write("Japanisch: こんにちは")

        content = test_file.read_text(encoding="utf-8")
        assert "こんにちは" in content

    def test_open_utf8_errors_replace(self, tmp_path: Path) -> None:
        """Ungültige Bytes werden durch Replacement-Character ersetzt (kein Crash)."""
        test_file = tmp_path / "invalid.txt"
        # Schreibe ungültige UTF-8 Bytes
        test_file.write_bytes(b"Valid \xff\xfe Invalid")

        with open_utf8(test_file, "r") as f:
            content = f.read()
        # Kein Crash, Replacement-Characters statt Exception
        assert "Valid" in content


class TestRunSubprocessUtf8:
    """run_subprocess_utf8 Tests."""

    def test_run_subprocess_utf8_returns_decoded_output(self) -> None:
        """Subprocess-Output wird als UTF-8-String zurückgegeben."""
        # echo gibt auf allen Plattformen etwas zurück
        result = run_subprocess_utf8(["python", "-c", "print('Hallo Welt')"])
        assert "Hallo Welt" in result.stdout
        assert result.returncode == 0

    def test_run_subprocess_utf8_unicode_output(self) -> None:
        """Unicode im Subprocess-Output wird korrekt dekodiert.

        Erzwingt PYTHONUTF8=1 im Subprozess damit stdout auf Windows
        tatsächlich UTF-8 schreibt (nicht cp1252).
        """
        import os

        env = {**os.environ, "PYTHONUTF8": "1"}
        script = (
            'import sys; sys.stdout.buffer.write("Unicode: äöü\\n".encode("utf-8"))'
        )
        result = run_subprocess_utf8(["python", "-c", script], env=env)
        assert "äöü" in result.stdout


class TestWriteJsonUtf8:
    """write_json_utf8 und read_json_utf8 Tests."""

    def test_write_json_utf8_no_ascii_escape(self, tmp_path: Path) -> None:
        """ensure_ascii=False: Unicode-Zeichen werden NICHT escaped."""
        test_file = tmp_path / "test.json"
        data = {"name": "Muenchen", "text": "äöü"}
        write_json_utf8(data, test_file)

        raw_content = test_file.read_text(encoding="utf-8")
        # Darf KEINE \\u Escapes enthalten
        assert "\\u" not in raw_content
        assert "äöü" in raw_content

    def test_read_json_utf8_roundtrip(self, tmp_path: Path) -> None:
        """Schreiben und Lesen ergibt identische Daten."""
        test_file = tmp_path / "roundtrip.json"
        original = {"key": "Wert mit äöüß", "num": 42}
        write_json_utf8(original, test_file)
        loaded = read_json_utf8(test_file)
        assert loaded == original


class TestAppendJsonlUtf8:
    """append_jsonl_utf8 Tests."""

    def test_append_jsonl_utf8_appends_unicode_safely(self, tmp_path: Path) -> None:
        """Mehrere Einträge werden korrekt als JSONL angehängt."""
        test_file = tmp_path / "sub" / "test.jsonl"
        # Verzeichnis wird automatisch erstellt
        append_jsonl_utf8({"msg": "Erste Zeile"}, test_file)
        append_jsonl_utf8({"msg": "Zweite mit ü"}, test_file)

        lines = test_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["msg"] == "Erste Zeile"
        assert json.loads(lines[1])["msg"] == "Zweite mit ü"

    def test_append_jsonl_utf8_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Fehlende Parent-Verzeichnisse werden automatisch erstellt."""
        deep_path = tmp_path / "a" / "b" / "c" / "test.jsonl"
        append_jsonl_utf8({"test": True}, deep_path)
        assert deep_path.exists()
