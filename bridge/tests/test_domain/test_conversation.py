"""Tests for domain.conversation: ConversationTurn and context building.

Tests dataclass creation, MAX_HISTORY_TURNS limit, and build_context_block.
"""

from domain.conversation import (
    MAX_HISTORY_TURNS,
    ConversationTurn,
    build_context_block,
)


class TestConversationTurn:
    """ConversationTurn dataclass tests."""

    def test_conversation_turn_dataclass(self) -> None:
        """ConversationTurn is correctly created with role and content."""
        turn = ConversationTurn(role="user", content="Hallo")
        assert turn.role == "user"
        assert turn.content == "Hallo"
        assert turn.timestamp  # auto-generated

    def test_conversation_turn_assistant(self) -> None:
        """Assistant turns work identically."""
        turn = ConversationTurn(role="assistant", content="Antwort")
        assert turn.role == "assistant"
        assert turn.content == "Antwort"

    def test_conversation_turn_frozen(self) -> None:
        """ConversationTurn is immutable."""
        turn = ConversationTurn(role="user", content="Test")
        try:
            turn.content = "Geändert"  # type: ignore[misc]
            assert False, "FrozenInstanceError expected"
        except Exception:
            pass

    def test_conversation_history_max_20_turns(self) -> None:
        """MAX_HISTORY_TURNS is set to 20."""
        assert MAX_HISTORY_TURNS == 20


class TestBuildContextBlock:
    """build_context_block formatting tests."""

    def test_empty_history_returns_current_message(self) -> None:
        """Without history, only the current message is returned."""
        result = build_context_block([], "Meine Frage")
        assert result == "Meine Frage"

    def test_with_history_formats_correctly(self) -> None:
        """With history the correct format with labels is produced."""
        history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hallo!"),
        ]
        result = build_context_block(history, "Wie gehts?")
        assert "[CONVERSATION HISTORY]" in result
        assert "User: Hi" in result
        assert "Axolent: Hallo!" in result
        assert "[CURRENT MESSAGE]" in result
        assert "Wie gehts?" in result

    def test_multiple_turns_in_order(self) -> None:
        """Multiple turns are formatted in the correct order."""
        history = [
            ConversationTurn(role="user", content="Eins"),
            ConversationTurn(role="assistant", content="Zwei"),
            ConversationTurn(role="user", content="Drei"),
            ConversationTurn(role="assistant", content="Vier"),
        ]
        result = build_context_block(history, "Fuenf")
        lines = result.split("\n")
        # Check order
        eins_idx = next(i for i, line in enumerate(lines) if "Eins" in line)
        vier_idx = next(i for i, line in enumerate(lines) if "Vier" in line)
        assert eins_idx < vier_idx

    def test_current_message_always_last(self) -> None:
        """The current message is always at the end."""
        history = [ConversationTurn(role="user", content="Vorher")]
        result = build_context_block(history, "Aktuell")
        assert result.endswith("Aktuell")
