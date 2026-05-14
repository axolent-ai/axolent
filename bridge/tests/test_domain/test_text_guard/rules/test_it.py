"""Tests for Italian (it) Text Guard rules.

Coverage target: 10+ positive cases, 5+ negative cases.
"""

from __future__ import annotations

import pytest

from domain.text_guard import TextGuard, get_builtin_rules


@pytest.fixture
def it_guard() -> TextGuard:
    """Full Italian guard from built-in rules."""
    rules = get_builtin_rules("it")
    assert rules is not None
    return TextGuard(rules, mode="fix")


class TestItPositiveCases:
    """Words that MUST be corrected."""

    def test_citta_to_citta_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("La citta e bella.") == "La città e bella."

    def test_perche_to_perche_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Perche non vieni?") == "Perché non vieni?"

    def test_qualita_to_qualita_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Alta qualita.") == "Alta qualità."

    def test_realta_to_realta_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("La realta dei fatti.") == "La realtà dei fatti."

    def test_societa_to_societa_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("La societa moderna.") == "La società moderna."

    def test_universita_to_universita_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("L'universita e grande.") == "L'università e grande."

    def test_liberta_to_liberta_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("La liberta e importante.") == "La libertà e importante."

    def test_gia_to_gia_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("E' gia fatto.") == "E' già fatto."

    def test_piu_to_piu_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Voglio di piu.") == "Voglio di più."

    def test_cosi_to_cosi_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Cosi va bene.") == "Così va bene."

    def test_pero_to_pero_accent(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Pero non posso.") == "Però non posso."


class TestItNegativeCases:
    """Words that must NOT be corrected."""

    def test_area_unchanged(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("The area is large.") == "The area is large."

    def test_camera_unchanged(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Turn on the camera.") == "Turn on the camera."

    def test_data_unchanged(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Load the data.") == "Load the data."

    def test_opera_unchanged(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Go to the opera.") == "Go to the opera."

    def test_piano_unchanged(self, it_guard: TextGuard) -> None:
        assert it_guard.fix("Play the piano.") == "Play the piano."

    def test_meta_unchanged(self, it_guard: TextGuard) -> None:
        """English 'meta' must not become 'metà'."""
        assert it_guard.fix("Use the meta tag.") == "Use the meta tag."


class TestItRuleSetStats:
    """Verify the Italian rule set has sufficient coverage."""

    def test_minimum_word_pairs(self) -> None:
        """Italian rules have at least 20 word pairs."""
        rules = get_builtin_rules("it")
        assert rules is not None
        assert len(rules.word_pairs) >= 20

    def test_minimum_whitelist(self) -> None:
        """Italian whitelist has at least 5 entries."""
        rules = get_builtin_rules("it")
        assert rules is not None
        assert len(rules.loan_word_whitelist) >= 5
