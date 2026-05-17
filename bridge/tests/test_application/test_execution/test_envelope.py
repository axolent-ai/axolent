"""Tests for RequestEnvelope creation and immutability."""

from __future__ import annotations

from datetime import timezone

import pytest

from application.execution.envelope import RequestEnvelope


class TestEnvelopeCreation:
    """Test RequestEnvelope construction and factory methods."""

    def test_default_construction_has_request_id(self) -> None:
        """Every envelope gets a unique request_id."""
        env = RequestEnvelope()
        assert env.request_id
        assert len(env.request_id) == 12

    def test_two_envelopes_have_different_ids(self) -> None:
        """request_id is unique per envelope."""
        env1 = RequestEnvelope()
        env2 = RequestEnvelope()
        assert env1.request_id != env2.request_id

    def test_from_telegram_basic(self) -> None:
        """from_telegram factory creates correct envelope."""
        env = RequestEnvelope.from_telegram(
            user_id=123,
            chat_id=456,
            text="Hello world",
            username="testuser",
        )
        assert env.user_id == 123
        assert env.chat_id == 456
        assert env.raw_text == "Hello world"
        assert env.channel == "telegram"
        assert env.command is None
        assert env.args == []
        assert env.username == "testuser"

    def test_from_telegram_with_command(self) -> None:
        """from_telegram with command and args."""
        env = RequestEnvelope.from_telegram(
            user_id=1,
            chat_id=2,
            text="/lang en",
            command="lang",
            args=["en"],
        )
        assert env.command == "lang"
        assert env.args == ["en"]

    def test_from_debate_command(self) -> None:
        """from_debate_command creates debate-specific envelope."""
        env = RequestEnvelope.from_debate_command(
            user_id=100,
            chat_id=200,
            question="Is water wet?",
            username="debater",
        )
        assert env.command == "debate"
        assert env.raw_text == "Is water wet?"
        assert env.args == ["Is water wet?"]
        assert env.username == "debater"

    def test_envelope_is_frozen(self) -> None:
        """Envelope is immutable (frozen dataclass)."""
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="hi")
        with pytest.raises(Exception):
            env.raw_text = "modified"  # type: ignore[misc]

    def test_timestamp_is_utc(self) -> None:
        """Timestamp defaults to UTC."""
        env = RequestEnvelope()
        assert env.timestamp_utc.tzinfo == timezone.utc

    def test_from_telegram_with_reply_to(self) -> None:
        """reply_to_text is preserved."""
        env = RequestEnvelope.from_telegram(
            user_id=1,
            chat_id=2,
            text="my reply",
            reply_to_text="original message",
        )
        assert env.reply_to_text == "original message"
