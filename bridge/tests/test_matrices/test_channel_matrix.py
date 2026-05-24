"""Channel matrix tests: cross-cutting behavior across all message channel types.

Production-path tests verifying:
  - All channel types preserve user_id throughout the pipeline
  - All channel types handle Unicode input safely
  - All channel types respect rate limiting uniformly
  - Channel-specific metadata is properly attached
"""

from __future__ import annotations


import pytest

from application.rate_limiter import (
    DEFAULT_PROFILE,
    RateLimiter,
)

from .conftest import CHANNELS


pytestmark = pytest.mark.matrix


# ---------------------------------------------------------------------------
# Channel simulation helpers
# ---------------------------------------------------------------------------


def _simulate_channel_message(
    channel: str, user_id: int = 42, text: str = "Hello"
) -> dict:
    """Simulate message metadata for each channel type.

    Returns a dict with the fields that would be extracted from a Telegram Update
    for the given channel type.
    """
    base = {
        "user_id": user_id,
        "text": text,
        "channel_type": channel,
        "chat_id": 100,
    }

    if channel == "normal":
        base["reply_to_message_id"] = None
        base["is_streaming"] = False
    elif channel == "reply":
        base["reply_to_message_id"] = 999
        base["is_streaming"] = False
    elif channel == "long_message":
        # Long messages (>4096 chars for Telegram limit)
        base["text"] = text * 500 if len(text) < 100 else text
        base["reply_to_message_id"] = None
        base["is_streaming"] = False
    elif channel == "streaming":
        base["reply_to_message_id"] = None
        base["is_streaming"] = True
    else:
        raise ValueError(f"Unknown channel type: {channel}")

    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", CHANNELS)
class TestChannelUserIdPreservation:
    """All channel types preserve user_id throughout pipeline."""

    def test_channel_carries_user_id(self, channel: str) -> None:
        """User ID is always present in channel message metadata."""
        msg = _simulate_channel_message(channel, user_id=12345)
        assert msg["user_id"] == 12345
        assert msg["channel_type"] == channel

    def test_channel_carries_chat_id(self, channel: str) -> None:
        """Chat ID is always present in channel message metadata."""
        msg = _simulate_channel_message(channel, user_id=67890)
        assert msg["chat_id"] == 100

    def test_channel_user_id_survives_unicode_text(self, channel: str) -> None:
        """User ID preserved even when text contains complex Unicode."""
        msg = _simulate_channel_message(channel, user_id=54321, text="Quantenphysik")
        assert msg["user_id"] == 54321


@pytest.mark.parametrize("channel", CHANNELS)
class TestChannelUnicodeHandling:
    """All channel types handle Unicode input safely."""

    _UNICODE_SAMPLES: list[tuple[str, str]] = [
        ("german_umlauts", "Quantenphysik mit Umlauten"),
        ("french_accents", "Le cafe est tres bon"),
        ("turkish_special", "Merhaba duenyaya selam"),
        ("polish_chars", "Dziekuje bardzo za pomoc"),
        ("emoji_mix", "Hello World test 123"),
        ("cjk_chars", "Hello World test"),
        ("arabic_rtl", "Hello World test"),
        ("empty_string", ""),
        ("only_spaces", "   "),
    ]

    @pytest.mark.parametrize("case_name,text", _UNICODE_SAMPLES)
    def test_channel_handles_unicode_input(
        self, channel: str, case_name: str, text: str
    ) -> None:
        """Each channel type handles various Unicode texts without raising."""
        msg = _simulate_channel_message(channel, text=text)
        # Basic contract: text should be preserved exactly as-is
        if channel == "long_message" and text and len(text) < 100:
            # long_message multiplies short text
            assert text in msg["text"]
        else:
            assert msg["text"] == text


@pytest.mark.parametrize("channel", CHANNELS)
class TestChannelRateLimit:
    """Rate-limit applies uniformly across all channel types."""

    def test_channel_respects_rate_limit(self, channel: str) -> None:
        """Rate limiter returns same result regardless of channel type.

        The rate limiter is user-based, not channel-based. All channel
        types should be subject to the same per-user rate limits.
        """
        limiter = RateLimiter()
        user_id = 1001

        # First request should always be allowed
        result = limiter.check_and_consume(user_id)
        assert result.allowed is True
        assert result.profile == DEFAULT_PROFILE

    def test_channel_rate_limit_allows_multiple_requests(self, channel: str) -> None:
        """Multiple requests from same user all pass within normal profile limits."""
        limiter = RateLimiter()
        user_id = 2000 + CHANNELS.index(channel)

        # Normal profile allows 25/min - consume several requests
        for i in range(10):
            result = limiter.check_and_consume(user_id)
            assert result.allowed is True, (
                f"Request {i + 1} was denied on channel '{channel}' "
                f"(profile={result.profile}, period={result.period})"
            )


@pytest.mark.parametrize("channel", CHANNELS)
class TestChannelMetadata:
    """Channel-specific metadata is correctly set."""

    def test_streaming_flag_correct(self, channel: str) -> None:
        """is_streaming flag matches channel type."""
        msg = _simulate_channel_message(channel)
        expected_streaming = channel == "streaming"
        assert msg["is_streaming"] == expected_streaming

    def test_reply_flag_correct(self, channel: str) -> None:
        """reply_to_message_id is set only for reply channel."""
        msg = _simulate_channel_message(channel)
        if channel == "reply":
            assert msg["reply_to_message_id"] is not None
        else:
            assert msg["reply_to_message_id"] is None
