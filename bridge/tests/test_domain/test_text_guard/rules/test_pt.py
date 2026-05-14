"""Tests for Portuguese (pt) Text Guard rules.

Coverage target: 10+ positive cases, 5+ negative cases.
"""

from __future__ import annotations

import pytest

from domain.text_guard import TextGuard, get_builtin_rules


@pytest.fixture
def pt_guard() -> TextGuard:
    """Full Portuguese guard from built-in rules."""
    rules = get_builtin_rules("pt")
    assert rules is not None
    return TextGuard(rules, mode="fix")


class TestPtPositiveCases:
    """Words that MUST be corrected."""

    def test_nao_to_nao_tilde(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Eu nao sei.") == "Eu não sei."

    def test_sao_to_sao_tilde(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Eles sao bons.") == "Eles são bons."

    def test_entao_to_entao_tilde(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Entao vamos.") == "Então vamos."

    def test_informacao_to_informacao_tilde(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("A informacao esta aqui.") == "A informação esta aqui."

    def test_voce_to_voce_cedilla(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Voce sabe?") == "Você sabe?"

    def test_preco_to_preco_cedilla(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("O preco e alto.") == "O preço e alto."

    def test_servico_to_servico_cedilla(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Bom servico.") == "Bom serviço."

    def test_agua_to_agua_accent(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Preciso de agua.") == "Preciso de água."

    def test_tambem_to_tambem_accent(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Eu tambem.") == "Eu também."

    def test_familia_to_familia_accent(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Minha familia.") == "Minha família."

    def test_musica_to_musica_accent(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Eu amo musica.") == "Eu amo música."

    def test_pais_to_pais_accent(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Meu pais e lindo.") == "Meu país e lindo."


class TestPtNegativeCases:
    """Words that must NOT be corrected."""

    def test_area_unchanged(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("The area is large.") == "The area is large."

    def test_data_unchanged(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Load the data.") == "Load the data."

    def test_extra_unchanged(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Extra features.") == "Extra features."

    def test_radio_unchanged(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("Turn on the radio.") == "Turn on the radio."

    def test_normal_unchanged(self, pt_guard: TextGuard) -> None:
        assert pt_guard.fix("This is normal.") == "This is normal."


class TestPtRuleSetStats:
    """Verify the Portuguese rule set has sufficient coverage."""

    def test_minimum_word_pairs(self) -> None:
        """Portuguese rules have at least 30 word pairs."""
        rules = get_builtin_rules("pt")
        assert rules is not None
        assert len(rules.word_pairs) >= 30

    def test_minimum_whitelist(self) -> None:
        """Portuguese whitelist has at least 10 entries."""
        rules = get_builtin_rules("pt")
        assert rules is not None
        assert len(rules.loan_word_whitelist) >= 10

    def test_tilde_coverage(self) -> None:
        """Portuguese rules include tilde corrections."""
        rules = get_builtin_rules("pt")
        assert rules is not None
        tilde_pairs = [
            p
            for p in rules.word_pairs
            if "ã" in p.correct_form or "õ" in p.correct_form
        ]
        assert len(tilde_pairs) >= 5

    def test_cedilla_coverage(self) -> None:
        """Portuguese rules include cedilla corrections."""
        rules = get_builtin_rules("pt")
        assert rules is not None
        cedilla_pairs = [p for p in rules.word_pairs if "ç" in p.correct_form]
        assert len(cedilla_pairs) >= 3
