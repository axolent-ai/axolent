"""Tests for French (fr) Text Guard rules.

Coverage target: 10+ positive cases, 5+ negative cases.
"""

from __future__ import annotations

import pytest

from domain.text_guard import TextGuard, get_builtin_rules


@pytest.fixture
def fr_guard() -> TextGuard:
    """Full French guard from built-in rules."""
    rules = get_builtin_rules("fr")
    assert rules is not None
    return TextGuard(rules, mode="fix")


class TestFrPositiveCases:
    """Words that MUST be corrected."""

    def test_etre_to_etre_accent(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Il veut etre ici.") == "Il veut être ici."

    def test_etait_to_etait_accent(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Il etait la.") == "Il était la."

    def test_ete_to_ete_accent(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Il a ete la.") == "Il a été la."

    def test_etat_to_etat_accent(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("L'etat de la France.") == "L'état de la France."

    def test_ecole_to_ecole_accent(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("L'ecole est fermee.") == "L'école est fermee."

    def test_francais_to_francais_cedilla(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Le francais est beau.") == "Le français est beau."

    def test_garcon_to_garcon_cedilla(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Le garcon mange.") == "Le garçon mange."

    def test_lecon_to_lecon_cedilla(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Une lecon importante.") == "Une leçon importante."

    def test_deja_to_deja_accent(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("C'est deja fait.") == "C'est déjà fait."

    def test_hopital_to_hopital_circumflex(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("L'hopital est proche.") == "L'hôpital est proche."

    def test_hotel_to_hotel_circumflex(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("L'hotel est beau.") == "L'hôtel est beau."

    def test_meme_to_meme_circumflex(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("C'est le meme chose.") == "C'est le même chose."

    def test_multiple_corrections(self, fr_guard: TextGuard) -> None:
        """Multiple corrections in one French sentence."""
        text = "Le francais etait tres important pour l'ecole."
        result = fr_guard.fix(text)
        assert "français" in result
        assert "était" in result
        assert "école" in result


class TestFrNegativeCases:
    """Words that must NOT be corrected."""

    def test_age_unchanged(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("What is your age?") == "What is your age?"

    def test_file_unchanged(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Open the file.") == "Open the file."

    def test_page_unchanged(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Turn the page.") == "Turn the page."

    def test_table_unchanged(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Create a table.") == "Create a table."

    def test_module_unchanged(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Import the module.") == "Import the module."

    def test_node_unchanged(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Start the node.") == "Start the node."

    def test_code_unchanged(self, fr_guard: TextGuard) -> None:
        assert fr_guard.fix("Read the code.") == "Read the code."


class TestFrRuleSetStats:
    """Verify the French rule set has sufficient coverage."""

    def test_minimum_word_pairs(self) -> None:
        """French rules have at least 30 word pairs."""
        rules = get_builtin_rules("fr")
        assert rules is not None
        assert len(rules.word_pairs) >= 30

    def test_minimum_whitelist(self) -> None:
        """French whitelist has at least 10 entries."""
        rules = get_builtin_rules("fr")
        assert rules is not None
        assert len(rules.loan_word_whitelist) >= 10
