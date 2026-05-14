"""Tests for Spanish (es) Text Guard rules.

Coverage target: 10+ positive cases, 5+ negative cases.
"""

from __future__ import annotations

import pytest

from domain.text_guard import TextGuard, get_builtin_rules


@pytest.fixture
def es_guard() -> TextGuard:
    """Full Spanish guard from built-in rules."""
    rules = get_builtin_rules("es")
    assert rules is not None
    return TextGuard(rules, mode="fix")


class TestEsPositiveCases:
    """Words that MUST be corrected."""

    def test_ano_to_ano_tilde(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Este ano es bueno.") == "Este año es bueno."

    def test_espanol_to_espanol_tilde(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Hablo espanol.") == "Hablo español."

    def test_manana_to_manana_tilde(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Hasta manana.") == "Hasta mañana."

    def test_nino_to_nino_tilde(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("El nino juega.") == "El niño juega."

    def test_senor_to_senor_tilde(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Hola senor.") == "Hola señor."

    def test_tambien_to_tambien_accent(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Yo tambien.") == "Yo también."

    def test_despues_to_despues_accent(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Nos vemos despues.") == "Nos vemos después."

    def test_aqui_to_aqui_accent(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Estoy aqui.") == "Estoy aquí."

    def test_facil_to_facil_accent(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Es muy facil.") == "Es muy fácil."

    def test_informacion_to_informacion_accent(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Necesito informacion.") == "Necesito información."

    def test_numero_to_numero_accent(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("El numero es 5.") == "El número es 5."

    def test_pequeno_to_pequeno_tilde(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Es muy pequeno.") == "Es muy pequeño."


class TestEsNegativeCases:
    """Words that must NOT be corrected."""

    def test_area_unchanged(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("The area is large.") == "The area is large."

    def test_auto_unchanged(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Set to auto mode.") == "Set to auto mode."

    def test_data_unchanged(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Load the data.") == "Load the data."

    def test_extra_unchanged(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Extra features.") == "Extra features."

    def test_radio_unchanged(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("Turn on the radio.") == "Turn on the radio."

    def test_normal_unchanged(self, es_guard: TextGuard) -> None:
        assert es_guard.fix("This is normal.") == "This is normal."


class TestEsRuleSetStats:
    """Verify the Spanish rule set has sufficient coverage."""

    def test_minimum_word_pairs(self) -> None:
        """Spanish rules have at least 40 word pairs."""
        rules = get_builtin_rules("es")
        assert rules is not None
        assert len(rules.word_pairs) >= 40

    def test_minimum_whitelist(self) -> None:
        """Spanish whitelist has at least 10 entries."""
        rules = get_builtin_rules("es")
        assert rules is not None
        assert len(rules.loan_word_whitelist) >= 10

    def test_tilde_coverage(self) -> None:
        """Spanish rules include n-tilde corrections."""
        rules = get_builtin_rules("es")
        assert rules is not None
        tilde_pairs = [p for p in rules.word_pairs if "ñ" in p.correct_form]
        assert len(tilde_pairs) >= 10
