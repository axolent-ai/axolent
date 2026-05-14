"""Tests for the plain text adapter."""

from __future__ import annotations

import json

import pytest

from domain.text_guard import TextGuard, get_builtin_rules
from domain.text_guard.adapters.text import (
    check_text,
    fix_json_string,
    fix_json_values,
    fix_text,
)


@pytest.fixture
def de_guard() -> TextGuard:
    """German guard for adapter tests."""
    rules = get_builtin_rules("de")
    assert rules is not None
    return TextGuard(rules, mode="fix")


class TestCheckText:
    """Tests for check_text()."""

    def test_finds_issues(self, de_guard: TextGuard) -> None:
        """Finds ASCII umlaut issues in plain text."""
        issues = check_text("Das ist fuer dich.", de_guard)
        assert len(issues) >= 1
        assert any(i.ascii_form == "fuer" for i in issues)

    def test_clean_text(self, de_guard: TextGuard) -> None:
        """Clean text returns no issues."""
        issues = check_text("Das ist für dich.", de_guard)
        assert issues == []


class TestFixText:
    """Tests for fix_text()."""

    def test_fixes_issues(self, de_guard: TextGuard) -> None:
        """Fixes ASCII umlaut issues in plain text."""
        result = fix_text("Das ist fuer dich.", de_guard)
        assert result == "Das ist für dich."

    def test_preserves_clean_text(self, de_guard: TextGuard) -> None:
        """Clean text is returned unchanged."""
        text = "Das ist für dich."
        assert fix_text(text, de_guard) == text


class TestFixJsonValues:
    """Tests for fix_json_values() recursive fixer."""

    def test_fixes_string(self, de_guard: TextGuard) -> None:
        """String values are corrected."""
        assert fix_json_values("fuer", de_guard) == "für"

    def test_fixes_dict_values(self, de_guard: TextGuard) -> None:
        """Dict string values are corrected, keys unchanged."""
        data = {"fuer": "fuer dich", "count": 42}
        result = fix_json_values(data, de_guard)
        assert result["fuer"] == "für dich"  # key unchanged
        assert result["count"] == 42

    def test_fixes_list_items(self, de_guard: TextGuard) -> None:
        """List string items are corrected."""
        data = ["fuer", "ueber", 42, None]
        result = fix_json_values(data, de_guard)
        assert result[0] == "für"
        assert result[1] == "über"
        assert result[2] == 42
        assert result[3] is None

    def test_nested_structure(self, de_guard: TextGuard) -> None:
        """Deeply nested structures are handled."""
        data = {"a": {"b": [{"c": "fuer"}]}}
        result = fix_json_values(data, de_guard)
        assert result["a"]["b"][0]["c"] == "für"

    def test_non_string_passthrough(self, de_guard: TextGuard) -> None:
        """Non-string, non-container types pass through."""
        assert fix_json_values(42, de_guard) == 42
        assert fix_json_values(True, de_guard) is True
        assert fix_json_values(None, de_guard) is None


class TestFixJsonString:
    """Tests for fix_json_string()."""

    def test_fixes_json_string(self, de_guard: TextGuard) -> None:
        """JSON string values are corrected."""
        data = {"message": "Das ist fuer dich."}
        json_str = json.dumps(data)
        result = fix_json_string(json_str, de_guard)
        parsed = json.loads(result)
        assert parsed["message"] == "Das ist für dich."

    def test_invalid_json_falls_back(self, de_guard: TextGuard) -> None:
        """Invalid JSON is treated as plain text."""
        result = fix_json_string("fuer dich", de_guard)
        assert result == "für dich"

    def test_ensure_ascii_false(self, de_guard: TextGuard) -> None:
        """Output JSON preserves Unicode (no \\u escapes)."""
        data = {"text": "fuer"}
        json_str = json.dumps(data)
        result = fix_json_string(json_str, de_guard)
        assert "\\u" not in result
        assert "für" in result
