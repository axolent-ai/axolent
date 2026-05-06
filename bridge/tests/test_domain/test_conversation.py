"""Tests fuer domain.conversation: ConversationTurn und Context-Building.

Testet Dataclass-Erstellung, MAX_HISTORY_TURNS-Limit und build_context_block.
"""

from domain.conversation import (
    MAX_HISTORY_TURNS,
    ConversationTurn,
    build_context_block,
)


class TestConversationTurn:
    """ConversationTurn Dataclass-Tests."""

    def test_conversation_turn_dataclass(self) -> None:
        """ConversationTurn wird korrekt erstellt mit role und content."""
        turn = ConversationTurn(role="user", content="Hallo")
        assert turn.role == "user"
        assert turn.content == "Hallo"
        assert turn.timestamp  # auto-generiert

    def test_conversation_turn_assistant(self) -> None:
        """Assistant-Turns funktionieren identisch."""
        turn = ConversationTurn(role="assistant", content="Antwort")
        assert turn.role == "assistant"
        assert turn.content == "Antwort"

    def test_conversation_turn_frozen(self) -> None:
        """ConversationTurn ist immutable."""
        turn = ConversationTurn(role="user", content="Test")
        try:
            turn.content = "Geaendert"  # type: ignore[misc]
            assert False, "FrozenInstanceError erwartet"
        except Exception:
            pass

    def test_conversation_history_max_20_turns(self) -> None:
        """MAX_HISTORY_TURNS ist auf 20 gesetzt."""
        assert MAX_HISTORY_TURNS == 20


class TestBuildContextBlock:
    """build_context_block Formatierungs-Tests."""

    def test_empty_history_returns_current_message(self) -> None:
        """Ohne History wird nur die aktuelle Nachricht zurueckgegeben."""
        result = build_context_block([], "Meine Frage")
        assert result == "Meine Frage"

    def test_with_history_formats_correctly(self) -> None:
        """Mit History wird das korrekte Format mit Labels erzeugt."""
        history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hallo!"),
        ]
        result = build_context_block(history, "Wie gehts?")
        assert "[VERLAUF DER UNTERHALTUNG]" in result
        assert "User: Hi" in result
        assert "Jarvis: Hallo!" in result
        assert "[AKTUELLE NACHRICHT]" in result
        assert "Wie gehts?" in result

    def test_multiple_turns_in_order(self) -> None:
        """Mehrere Turns werden in der richtigen Reihenfolge formatiert."""
        history = [
            ConversationTurn(role="user", content="Eins"),
            ConversationTurn(role="assistant", content="Zwei"),
            ConversationTurn(role="user", content="Drei"),
            ConversationTurn(role="assistant", content="Vier"),
        ]
        result = build_context_block(history, "Fuenf")
        lines = result.split("\n")
        # Reihenfolge pruefen
        eins_idx = next(i for i, line in enumerate(lines) if "Eins" in line)
        vier_idx = next(i for i, line in enumerate(lines) if "Vier" in line)
        assert eins_idx < vier_idx

    def test_current_message_always_last(self) -> None:
        """Die aktuelle Nachricht steht immer am Ende."""
        history = [ConversationTurn(role="user", content="Vorher")]
        result = build_context_block(history, "Aktuell")
        assert result.endswith("Aktuell")
