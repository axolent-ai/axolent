"""Tests fuer domain.personality: Personality-Prompt-Kombination.

Testet build_combined_prompt und build_effective_prompt mit Language-Override.
"""

from domain.personality import PersonalityConfig, build_effective_prompt


class TestPersonalityConfig:
    """PersonalityConfig Kombinations-Logik."""

    def test_personality_combined_prompt_format(self) -> None:
        """System-Prompt und Constitution werden mit Trennlinie kombiniert."""
        config = PersonalityConfig(
            system_prompt="Du bist Jarvis.",
            user_constitution="Antworte immer freundlich.",
        )
        result = config.build_combined_prompt()
        assert "Du bist Jarvis." in result
        assert "Antworte immer freundlich." in result
        assert "---" in result

    def test_only_system_prompt(self) -> None:
        """Wenn nur System-Prompt vorhanden, keine Trennlinie."""
        config = PersonalityConfig(system_prompt="Nur System.", user_constitution="")
        result = config.build_combined_prompt()
        assert result == "Nur System."
        assert "---" not in result

    def test_only_constitution(self) -> None:
        """Wenn nur Constitution vorhanden, wird nur diese zurueckgegeben."""
        config = PersonalityConfig(system_prompt="", user_constitution="Nur Regeln.")
        result = config.build_combined_prompt()
        assert result == "Nur Regeln."

    def test_both_empty_returns_empty(self) -> None:
        """Ohne beides wird ein leerer String zurueckgegeben."""
        config = PersonalityConfig(system_prompt="", user_constitution="")
        result = config.build_combined_prompt()
        assert result == ""


class TestBuildEffectivePrompt:
    """build_effective_prompt mit optionalem Language-Override."""

    def test_no_language_override_for_german(self) -> None:
        """Bei 'de' wird kein Language-Override angehaengt."""
        result = build_effective_prompt("Base prompt.", "de")
        assert "LANGUAGE OVERRIDE" not in result
        assert result == "Base prompt."

    def test_language_override_for_english(self) -> None:
        """Bei 'en' wird ein Language-Override-Block angehaengt."""
        result = build_effective_prompt("Base prompt.", "en")
        assert "LANGUAGE OVERRIDE" in result
        assert "'en'" in result

    def test_language_override_for_spanish(self) -> None:
        """Beliebige Nicht-de-Sprache loest Override aus."""
        result = build_effective_prompt("Base.", "es")
        assert "LANGUAGE OVERRIDE" in result
        assert "'es'" in result

    def test_empty_language_hint_no_override(self) -> None:
        """Leerer Language-Hint fuegt keinen Override an."""
        result = build_effective_prompt("Base.", "")
        assert "LANGUAGE OVERRIDE" not in result
        assert result == "Base."

    def test_empty_base_prompt_with_language(self) -> None:
        """Auch ohne Base-Prompt wird Language-Override gesetzt."""
        result = build_effective_prompt("", "fr")
        assert "LANGUAGE OVERRIDE" in result
        assert "'fr'" in result
