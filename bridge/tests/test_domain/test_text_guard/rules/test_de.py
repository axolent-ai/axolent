"""Tests for German (de) Text Guard rules.

Coverage target: 15+ positive cases, 10+ negative cases.
"""

from __future__ import annotations

import pytest

from domain.text_guard import TextGuard, get_builtin_rules


@pytest.fixture
def de_guard() -> TextGuard:
    """Full German guard from built-in rules."""
    rules = get_builtin_rules("de")
    assert rules is not None
    return TextGuard(rules, mode="fix")


@pytest.fixture
def de_checker() -> TextGuard:
    """Full German guard in check mode."""
    rules = get_builtin_rules("de")
    assert rules is not None
    return TextGuard(rules, mode="check")


class TestDePositiveCases:
    """Words that MUST be corrected."""

    def test_fuer_to_fuer_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Das ist fuer dich.") == "Das ist für dich."

    def test_ueber_to_ueber_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Ich denke ueber das nach.") == "Ich denke über das nach."

    def test_moeglich_to_moeglich_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Ist das moeglich?") == "Ist das möglich?"

    def test_erklaeren_to_erklaeren_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Ich werde erklaeren.") == "Ich werde erklären."

    def test_groesser_to_groesser_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Viel groesser.") == "Viel größer."

    def test_koennte_to_koennte_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Das koennte sein.") == "Das könnte sein."

    def test_wuerde_to_wuerde_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Ich wuerde sagen.") == "Ich würde sagen."

    def test_natuerlich_to_natuerlich_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Ja natuerlich.") == "Ja natürlich."

    def test_verfuegbar_to_verfuegbar_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Noch verfuegbar.") == "Noch verfügbar."

    def test_zurueck_to_zurueck_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Komm zurueck.") == "Komm zurück."

    def test_ausserdem_to_ausserdem_eszett(self, de_guard: TextGuard) -> None:
        assert (
            de_guard.fix("Ausserdem wollte ich sagen.") == "Außerdem wollte ich sagen."
        )

    def test_strasse_to_strasse_eszett(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Die Strasse ist breit.") == "Die Straße ist breit."

    def test_schliessen_to_schliessen_eszett(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Bitte schliessen.") == "Bitte schließen."

    def test_haette_to_haette_umlaut(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Das haette ich gewusst.") == "Das hätte ich gewusst."

    def test_moeglichkeit_to_moeglichkeit_umlaut(self, de_guard: TextGuard) -> None:
        assert (
            de_guard.fix("Eine Moeglichkeit waere gut.") == "Eine Möglichkeit wäre gut."
        )

    def test_multiple_corrections_in_sentence(self, de_guard: TextGuard) -> None:
        """Multiple corrections in one complex sentence."""
        text = "Ich wuerde dir natuerlich erklaeren warum das fuer uns moeglich waere."
        result = de_guard.fix(text)
        assert "würde" in result
        assert "natürlich" in result
        assert "erklären" in result
        assert "für" in result
        assert "möglich" in result
        assert "wäre" in result


class TestDeNegativeCases:
    """Words that must NOT be corrected (false positive protection)."""

    def test_queue_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Add to the queue.") == "Add to the queue."

    def test_blue_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("The sky is blue.") == "The sky is blue."

    def test_true_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("That is true.") == "That is true."

    def test_user_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("The user logged in.") == "The user logged in."

    def test_module_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Import the module.") == "Import the module."

    def test_continue_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Please continue.") == "Please continue."

    def test_issue_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("Open an issue.") == "Open an issue."

    def test_value_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("The value is 42.") == "The value is 42."

    def test_feature_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("New feature added.") == "New feature added."

    def test_unique_unchanged(self, de_guard: TextGuard) -> None:
        assert de_guard.fix("This is unique.") == "This is unique."

    def test_code_block_preserved(self, de_guard: TextGuard) -> None:
        """Code blocks must never be modified."""
        text = '```python\nfuer = "value"\nueber = True\n```'
        assert de_guard.fix(text) == text

    def test_inline_code_preserved(self, de_guard: TextGuard) -> None:
        """Inline code must never be modified."""
        assert "`fuer`" in de_guard.fix("Use `fuer` as variable name.")

    def test_already_correct_umlauts(self, de_guard: TextGuard) -> None:
        """Already correct text stays unchanged."""
        text = "Das ist für dich und über alles möglich."
        assert de_guard.fix(text) == text


class TestDeCheckMode:
    """Check mode tests for German rules."""

    def test_detects_all_issues(self, de_checker: TextGuard) -> None:
        """Check mode finds all issues in a sentence."""
        text = "Ich erklaere fuer dich warum das moeglich waere."
        issues = de_checker.check(text)
        ascii_forms = {i.ascii_form for i in issues}
        assert "fuer" in ascii_forms
        assert "moeglich" in ascii_forms

    def test_issue_has_correct_suggestion(self, de_checker: TextGuard) -> None:
        """Each issue has the correct replacement suggestion."""
        issues = de_checker.check("Das ist fuer dich.")
        assert len(issues) >= 1
        fuer_issue = next(i for i in issues if i.ascii_form == "fuer")
        assert fuer_issue.correct_form == "für"


class TestDeRuleSetStats:
    """Verify the German rule set has sufficient coverage."""

    def test_minimum_word_pairs(self) -> None:
        """German rules have at least 100 word pairs."""
        rules = get_builtin_rules("de")
        assert rules is not None
        assert len(rules.word_pairs) >= 100

    def test_minimum_whitelist(self) -> None:
        """German whitelist has at least 50 entries."""
        rules = get_builtin_rules("de")
        assert rules is not None
        assert len(rules.loan_word_whitelist) >= 50

    def test_ae_coverage(self) -> None:
        """German rules cover ae->ae_umlaut substitutions."""
        rules = get_builtin_rules("de")
        assert rules is not None
        ae_pairs = [p for p in rules.word_pairs if "ae" in p.ascii_form]
        assert len(ae_pairs) >= 20

    def test_oe_coverage(self) -> None:
        """German rules cover oe->oe_umlaut substitutions."""
        rules = get_builtin_rules("de")
        assert rules is not None
        oe_pairs = [p for p in rules.word_pairs if "oe" in p.ascii_form]
        assert len(oe_pairs) >= 20

    def test_ue_coverage(self) -> None:
        """German rules cover ue->ue_umlaut substitutions."""
        rules = get_builtin_rules("de")
        assert rules is not None
        ue_pairs = [p for p in rules.word_pairs if "ue" in p.ascii_form]
        assert len(ue_pairs) >= 30
