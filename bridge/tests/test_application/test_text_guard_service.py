"""Tests for the TextGuardService application service."""

from __future__ import annotations

from pathlib import Path

import pytest

from application.text_guard_service import TextGuardService


@pytest.fixture
def service() -> TextGuardService:
    """Fresh TextGuardService instance."""
    return TextGuardService()


class TestGetGuard:
    """Tests for get_guard()."""

    def test_returns_guard_for_supported_language(
        self, service: TextGuardService
    ) -> None:
        """Returns a TextGuard for a supported language."""
        guard = service.get_guard("de")
        assert guard is not None
        assert guard.language == "de"

    def test_returns_none_for_unsupported_language(
        self, service: TextGuardService
    ) -> None:
        """Returns None for unsupported language."""
        guard = service.get_guard("xx")
        assert guard is None

    def test_caches_guard(self, service: TextGuardService) -> None:
        """Same guard is returned on repeated calls."""
        g1 = service.get_guard("de")
        g2 = service.get_guard("de")
        assert g1 is g2

    def test_different_modes_different_guards(self, service: TextGuardService) -> None:
        """Different modes produce different guard instances."""
        g_fix = service.get_guard("de", mode="fix")
        g_check = service.get_guard("de", mode="check")
        assert g_fix is not g_check


class TestGetStreamingGuard:
    """Tests for get_streaming_guard()."""

    def test_returns_streaming_guard(self, service: TextGuardService) -> None:
        """Returns a StreamingTextGuard for supported language."""
        sg = service.get_streaming_guard("de")
        assert sg is not None

    def test_returns_none_for_unsupported(self, service: TextGuardService) -> None:
        """Returns None for unsupported language."""
        sg = service.get_streaming_guard("xx")
        assert sg is None


class TestCheckString:
    """Tests for check_string()."""

    def test_finds_german_issues(self, service: TextGuardService) -> None:
        """Detects German ASCII umlaut issues."""
        issues = service.check_string("Das ist fuer dich.", "de")
        assert len(issues) >= 1

    def test_unsupported_language(self, service: TextGuardService) -> None:
        """Unsupported language returns empty list."""
        issues = service.check_string("fuer", "xx")
        assert issues == []


class TestFixString:
    """Tests for fix_string()."""

    def test_fixes_german(self, service: TextGuardService) -> None:
        """Fixes German ASCII umlauts."""
        result = service.fix_string("Das ist fuer dich.", "de")
        assert result == "Das ist für dich."

    def test_fixes_french(self, service: TextGuardService) -> None:
        """Fixes French missing accents."""
        result = service.fix_string("Le francais est beau.", "fr")
        assert "français" in result

    def test_fixes_spanish(self, service: TextGuardService) -> None:
        """Fixes Spanish missing tildes."""
        result = service.fix_string("Hablo espanol.", "es")
        assert "español" in result

    def test_unsupported_language_passthrough(self, service: TextGuardService) -> None:
        """Unsupported language returns text unchanged."""
        text = "fuer ueber"
        result = service.fix_string(text, "xx")
        assert result == text

    def test_english_noop(self, service: TextGuardService) -> None:
        """English (no-op rules) returns text unchanged."""
        text = "This fuer text passes through."
        result = service.fix_string(text, "en")
        assert result == text


class TestCheckFile:
    """Tests for check_file() with text files."""

    def test_checks_markdown(self, service: TextGuardService, tmp_path: Path) -> None:
        """Checks a markdown file."""
        filepath = tmp_path / "test.md"
        filepath.write_text("Das ist fuer dich.", encoding="utf-8")
        issues = service.check_file(filepath, "de")
        assert len(issues) >= 1

    def test_unsupported_extension(
        self, service: TextGuardService, tmp_path: Path
    ) -> None:
        """Unsupported file extension returns empty list."""
        filepath = tmp_path / "test.bin"
        filepath.write_bytes(b"fuer")
        issues = service.check_file(filepath, "de")
        assert issues == []


class TestFixFile:
    """Tests for fix_file() with text files."""

    def test_fixes_markdown(self, service: TextGuardService, tmp_path: Path) -> None:
        """Fixes a markdown file in-place."""
        filepath = tmp_path / "test.md"
        filepath.write_text("Das ist fuer dich.", encoding="utf-8")
        changed = service.fix_file(filepath, "de")
        assert changed is True
        content = filepath.read_text(encoding="utf-8")
        assert "für" in content

    def test_no_change_on_clean(
        self, service: TextGuardService, tmp_path: Path
    ) -> None:
        """Clean file is not modified."""
        filepath = tmp_path / "clean.md"
        filepath.write_text("Das ist für dich.", encoding="utf-8")
        changed = service.fix_file(filepath, "de")
        assert changed is False

    def test_fixes_json_file(self, service: TextGuardService, tmp_path: Path) -> None:
        """Fixes a JSON file."""
        filepath = tmp_path / "test.json"
        filepath.write_text('{"msg": "fuer dich"}', encoding="utf-8")
        changed = service.fix_file(filepath, "de")
        assert changed is True
        content = filepath.read_text(encoding="utf-8")
        assert "für" in content
