"""Tests for the core TextGuard engine."""

from __future__ import annotations

import pytest

from domain.text_guard.guard import TextGuard
from domain.text_guard.models import RuleSet, WordPair


@pytest.fixture
def de_guard() -> TextGuard:
    """German text guard with a small representative rule set."""
    rules = RuleSet(
        language="de",
        word_pairs=(
            WordPair("fuer", "für"),
            WordPair("ueber", "über"),
            WordPair("moeglich", "möglich"),
            WordPair("erklaere", "erkläre"),
            WordPair("erklaeren", "erklären"),
            WordPair("erklaert", "erklärt"),
            WordPair("groesser", "größer"),
            WordPair("koennte", "könnte"),
            WordPair("wuerde", "würde"),
            WordPair("ausserdem", "außerdem"),
            WordPair("strasse", "Straße"),
            WordPair("natuerlich", "natürlich"),
        ),
        loan_word_whitelist=frozenset({"queue", "blue", "true", "user", "module"}),
        code_block_skip=True,
    )
    return TextGuard(rules, mode="fix")


@pytest.fixture
def de_check_guard() -> TextGuard:
    """German text guard in check mode."""
    rules = RuleSet(
        language="de",
        word_pairs=(
            WordPair("fuer", "für"),
            WordPair("ueber", "über"),
            WordPair("natuerlich", "natürlich"),
        ),
        loan_word_whitelist=frozenset({"user", "true"}),
        code_block_skip=True,
    )
    return TextGuard(rules, mode="check")


class TestTextGuardFix:
    """Tests for the fix() method."""

    def test_simple_replacement(self, de_guard: TextGuard) -> None:
        """Single word gets corrected."""
        assert de_guard.fix("Das ist fuer dich.") == "Das ist für dich."

    def test_multiple_replacements(self, de_guard: TextGuard) -> None:
        """Multiple words in one sentence get corrected."""
        result = de_guard.fix("Ich erklaere dir das fuer dich.")
        assert result == "Ich erkläre dir das für dich."

    def test_case_preservation_lowercase(self, de_guard: TextGuard) -> None:
        """Lowercase input stays lowercase in replacement."""
        assert de_guard.fix("moeglich") == "möglich"

    def test_case_preservation_titlecase(self, de_guard: TextGuard) -> None:
        """Title case input produces title case replacement."""
        assert de_guard.fix("Natuerlich ist das so.") == "Natürlich ist das so."

    def test_case_preservation_uppercase(self, de_guard: TextGuard) -> None:
        """All-caps input produces all-caps replacement."""
        assert de_guard.fix("FUER") == "FÜR"

    def test_loan_word_not_replaced(self, de_guard: TextGuard) -> None:
        """English loan words in the whitelist stay unchanged."""
        assert de_guard.fix("The user is in the queue.") == "The user is in the queue."

    def test_loan_word_case_insensitive(self, de_guard: TextGuard) -> None:
        """Loan word whitelist is case-insensitive."""
        assert de_guard.fix("The User joined.") == "The User joined."

    def test_code_block_skip(self, de_guard: TextGuard) -> None:
        """Content inside fenced code blocks is not modified."""
        text = "Normal fuer text.\n```\nfuer in code\n```\nMore fuer text."
        result = de_guard.fix(text)
        assert "```\nfuer in code\n```" in result
        assert result.startswith("Normal für text.")
        assert result.endswith("More für text.")

    def test_inline_code_skip(self, de_guard: TextGuard) -> None:
        """Content inside inline code is not modified."""
        text = "Use `fuer` as variable name, but fuer users."
        result = de_guard.fix(text)
        assert "`fuer`" in result
        assert "für users" in result

    def test_empty_text(self, de_guard: TextGuard) -> None:
        """Empty text returns empty text."""
        assert de_guard.fix("") == ""

    def test_whitespace_only(self, de_guard: TextGuard) -> None:
        """Whitespace-only text returns unchanged."""
        assert de_guard.fix("   \n  ") == "   \n  "

    def test_no_matches(self, de_guard: TextGuard) -> None:
        """Text without any ASCII diacritic issues returns unchanged."""
        text = "This is English text without any issues."
        assert de_guard.fix(text) == text

    def test_word_boundary_respected(self, de_guard: TextGuard) -> None:
        """Replacement only happens at word boundaries."""
        rules = RuleSet(
            language="de",
            word_pairs=(WordPair("fuer", "für"),),
            loan_word_whitelist=frozenset(),
        )
        guard = TextGuard(rules)
        # "fuerstlich" should NOT be corrected (fuer is a substring)
        # Actually, with word-boundary regex, "fuer" in "fuerstlich" won't match
        # because \b matches at the start but not before "stlich"
        # Wait: "fuerstlich" starts with "fuer" but "fuer" has \b at end,
        # and "stlich" continues without boundary. So it won't match.
        assert guard.fix("fuerstlich") == "fuerstlich"

    def test_eszett_replacement(self, de_guard: TextGuard) -> None:
        """ss -> eszett replacement for known words."""
        assert de_guard.fix("Die Strasse ist lang.") == "Die Straße ist lang."

    def test_multiline_text(self, de_guard: TextGuard) -> None:
        """Corrections work across multiple lines."""
        text = "Zeile 1: fuer\nZeile 2: ueber\nZeile 3: moeglich"
        result = de_guard.fix(text)
        assert "für" in result
        assert "über" in result
        assert "möglich" in result


class TestTextGuardCheck:
    """Tests for the check() method."""

    def test_detects_issues(self, de_check_guard: TextGuard) -> None:
        """Check mode detects issues without modifying text."""
        issues = de_check_guard.check("Das ist fuer dich und ueber alles.")
        assert len(issues) == 2
        assert issues[0].ascii_form == "fuer"
        assert issues[0].correct_form == "für"
        assert issues[1].ascii_form == "ueber"

    def test_returns_line_and_column(self, de_check_guard: TextGuard) -> None:
        """Check mode returns correct line and column."""
        issues = de_check_guard.check("OK\nDas ist fuer dich.")
        assert len(issues) == 1
        assert issues[0].line == 2
        assert issues[0].column == 8

    def test_no_issues_for_clean_text(self, de_check_guard: TextGuard) -> None:
        """Clean text returns empty issue list."""
        issues = de_check_guard.check("Das ist für dich und über alles.")
        assert issues == []

    def test_skips_loan_words(self, de_check_guard: TextGuard) -> None:
        """Check mode does not flag whitelisted loan words."""
        issues = de_check_guard.check("The user is true to form.")
        assert issues == []

    def test_skips_code_blocks(self, de_check_guard: TextGuard) -> None:
        """Check mode skips code block content."""
        text = "Real fuer text.\n```\nfuer in code\n```"
        issues = de_check_guard.check(text)
        assert len(issues) == 1
        assert issues[0].line == 1


class TestTextGuardFixWord:
    """Tests for the fix_word() method (used by streaming adapter)."""

    def test_fixes_known_word(self, de_guard: TextGuard) -> None:
        """Known ASCII word gets corrected."""
        assert de_guard.fix_word("fuer") == "für"

    def test_preserves_unknown_word(self, de_guard: TextGuard) -> None:
        """Unknown word passes through unchanged."""
        assert de_guard.fix_word("hello") == "hello"

    def test_preserves_loan_word(self, de_guard: TextGuard) -> None:
        """Whitelisted loan word passes through unchanged."""
        assert de_guard.fix_word("queue") == "queue"

    def test_empty_word(self, de_guard: TextGuard) -> None:
        """Empty string returns empty string."""
        assert de_guard.fix_word("") == ""


class TestTextGuardProperties:
    """Tests for guard properties."""

    def test_language_property(self, de_guard: TextGuard) -> None:
        """Language property returns the rule set language."""
        assert de_guard.language == "de"

    def test_rule_count_property(self, de_guard: TextGuard) -> None:
        """Rule count property returns correct count."""
        assert de_guard.rule_count == 12


class TestEmptyRuleSet:
    """Tests with empty (no-op) rule set."""

    def test_english_noop(self) -> None:
        """English rule set passes text through unchanged."""
        rules = RuleSet(language="en", word_pairs=())
        guard = TextGuard(rules)
        text = "This fuer text should pass through unchanged."
        assert guard.fix(text) == text

    def test_empty_check(self) -> None:
        """Empty rule set reports no issues."""
        rules = RuleSet(language="en", word_pairs=())
        guard = TextGuard(rules, mode="check")
        assert guard.check("fuer ueber moeglich") == []
