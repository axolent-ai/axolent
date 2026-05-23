"""Tests for individual resolvers."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pytest

from application.execution.context import PartialExecutionContext
from application.execution.envelope import RequestEnvelope
from application.execution.resolvers import (
    ChannelResolver,
    LanguageResolverAdapter,
    TimeResolver,
    _classify_time_of_day,
    _get_weekday_name,
    _resolve_timezone,
)
from application.language_resolver import LanguageContext


class _MockLanguageResolver:
    """Mock LanguageResolver for testing."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self.call_count = 0

    async def resolve(self, user_id, chat_id, text, override=None):
        self.call_count += 1
        code = override or self.lang
        return LanguageContext(
            code=code,
            source="override" if override else "detected",
            confidence=0.99,
            switched_from=None,
            request_id="test-req",
        )


class TestLanguageResolverAdapter:
    """Test LanguageResolverAdapter behavior."""

    @pytest.mark.asyncio
    async def test_resolves_language(self) -> None:
        """Adapter delegates to LanguageResolver."""
        mock = _MockLanguageResolver(lang="es")
        adapter = LanguageResolverAdapter(mock)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="Hola mundo")
        partial = PartialExecutionContext.from_envelope(env)
        result = await adapter.resolve(partial)

        assert result.language is not None
        assert result.language.code == "es"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_if_already_resolved(self) -> None:
        """Adapter is a no-op if language already set."""
        mock = _MockLanguageResolver(lang="es")
        adapter = LanguageResolverAdapter(mock)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        partial = PartialExecutionContext.from_envelope(env)
        partial.language = LanguageContext(
            code="fr",
            source="override",
            confidence=1.0,
            switched_from=None,
            request_id="pre-set",
        )

        result = await adapter.resolve(partial)
        assert result.language.code == "fr"
        assert mock.call_count == 0  # not called

    @pytest.mark.asyncio
    async def test_passes_override(self) -> None:
        """Adapter forwards language_override."""
        mock = _MockLanguageResolver(lang="de")
        adapter = LanguageResolverAdapter(mock)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        partial = PartialExecutionContext.from_envelope(env, language_override="it")
        result = await adapter.resolve(partial)

        assert result.language.code == "it"
        assert result.language.source == "override"

    @pytest.mark.asyncio
    async def test_request_id_preserved(self) -> None:
        """Request ID from partial is used in LanguageContext."""
        mock = _MockLanguageResolver(lang="en")
        adapter = LanguageResolverAdapter(mock)

        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="hi")
        partial = PartialExecutionContext.from_envelope(env)
        result = await adapter.resolve(partial)

        assert result.language.request_id == env.request_id


class TestResolveTimezone:
    """Test _resolve_timezone helper."""

    def test_uses_env_var_when_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AXOLENT_TIMEZONE env var is respected."""
        monkeypatch.setenv("AXOLENT_TIMEZONE", "Europe/Berlin")
        tz_info, tz_name = _resolve_timezone()
        assert tz_name == "Europe/Berlin"
        assert tz_info == ZoneInfo("Europe/Berlin")

    def test_invalid_env_var_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid AXOLENT_TIMEZONE falls back gracefully."""
        monkeypatch.setenv("AXOLENT_TIMEZONE", "Mars/Olympus")
        tz_info, tz_name = _resolve_timezone()
        # Should fall back to system local or Europe/Berlin
        assert tz_name != "Mars/Olympus"
        assert tz_info is not None

    def test_no_env_var_returns_valid_tz(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without env var, returns a valid timezone."""
        monkeypatch.delenv("AXOLENT_TIMEZONE", raising=False)
        tz_info, tz_name = _resolve_timezone()
        assert tz_info is not None
        assert isinstance(tz_name, str)
        assert len(tz_name) > 0


class TestTimeResolver:
    """Test TimeResolver behavior."""

    @pytest.mark.asyncio
    async def test_fills_time_context(self) -> None:
        """TimeResolver fills partial.time."""
        resolver = TimeResolver()
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        partial = PartialExecutionContext.from_envelope(env)
        partial.language = LanguageContext(
            code="en",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="r1",
        )

        result = await resolver.resolve(partial)
        assert result.time is not None
        assert result.time.timezone_name != ""
        assert result.time.weekday >= 0
        assert result.time.weekday <= 6

    @pytest.mark.asyncio
    async def test_german_weekday_name(self) -> None:
        """TimeResolver uses German weekday names for DE language."""
        resolver = TimeResolver()
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        partial = PartialExecutionContext.from_envelope(env)
        partial.language = LanguageContext(
            code="de",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="r2",
        )

        result = await resolver.resolve(partial)
        german_days = {
            "Montag",
            "Dienstag",
            "Mittwoch",
            "Donnerstag",
            "Freitag",
            "Samstag",
            "Sonntag",
        }
        assert result.time.weekday_name in german_days

    @pytest.mark.asyncio
    async def test_resolve_uses_env_timezone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TimeResolver respects AXOLENT_TIMEZONE env var."""
        monkeypatch.setenv("AXOLENT_TIMEZONE", "Europe/Berlin")
        resolver = TimeResolver()
        partial = PartialExecutionContext()
        result = await resolver.resolve(partial)

        assert result.time.timezone_name == "Europe/Berlin"
        assert result.time.now_local.tzinfo == ZoneInfo("Europe/Berlin")
        assert result.time.now_utc.tzinfo.tzname(None) == "UTC"

    @pytest.mark.asyncio
    async def test_now_local_differs_from_utc_in_europe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """now_local in Europe/Berlin has CET (+1) or CEST (+2) offset."""
        monkeypatch.setenv("AXOLENT_TIMEZONE", "Europe/Berlin")
        resolver = TimeResolver()
        partial = PartialExecutionContext()
        result = await resolver.resolve(partial)

        offset_hours = result.time.now_local.utcoffset().total_seconds() / 3600
        assert offset_hours in (1, 2)  # CET=+1, CEST=+2

    @pytest.mark.asyncio
    async def test_now_local_and_utc_represent_same_instant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """now_local and now_utc must represent the same point in time."""
        monkeypatch.setenv("AXOLENT_TIMEZONE", "America/New_York")
        resolver = TimeResolver()
        partial = PartialExecutionContext()
        result = await resolver.resolve(partial)

        # Same instant: difference should be less than 1 second
        delta = abs(
            (
                result.time.now_utc
                - result.time.now_local.astimezone(result.time.now_utc.tzinfo)
            ).total_seconds()
        )
        assert delta < 1.0

    @pytest.mark.asyncio
    async def test_weekday_uses_local_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Weekday and time_of_day are derived from local time, not UTC."""
        monkeypatch.setenv("AXOLENT_TIMEZONE", "Europe/Berlin")
        resolver = TimeResolver()
        partial = PartialExecutionContext()
        result = await resolver.resolve(partial)

        # Verify weekday matches local time, not UTC
        assert result.time.weekday == result.time.now_local.weekday()


class TestTimeOfDayClassification:
    """Test _classify_time_of_day helper."""

    def test_morning(self) -> None:
        assert _classify_time_of_day(5) == "morning"
        assert _classify_time_of_day(11) == "morning"

    def test_afternoon(self) -> None:
        assert _classify_time_of_day(12) == "afternoon"
        assert _classify_time_of_day(16) == "afternoon"

    def test_evening(self) -> None:
        assert _classify_time_of_day(17) == "evening"
        assert _classify_time_of_day(20) == "evening"

    def test_night(self) -> None:
        assert _classify_time_of_day(21) == "night"
        assert _classify_time_of_day(4) == "night"
        assert _classify_time_of_day(0) == "night"


class TestWeekdayNames:
    """Test _get_weekday_name helper."""

    def test_english_monday(self) -> None:
        assert _get_weekday_name(0, "en") == "Monday"

    def test_german_friday(self) -> None:
        assert _get_weekday_name(4, "de") == "Freitag"

    def test_unknown_lang_defaults_to_english(self) -> None:
        assert _get_weekday_name(6, "zh") == "Sunday"


class TestChannelResolver:
    """Test ChannelResolver behavior."""

    @pytest.mark.asyncio
    async def test_telegram_capabilities(self) -> None:
        """Telegram gets correct capability profile."""
        resolver = ChannelResolver()
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        partial = PartialExecutionContext.from_envelope(env)
        result = await resolver.resolve(partial)

        assert result.channel_capabilities is not None
        assert result.channel_capabilities.max_message_length == 4096
        assert result.channel_capabilities.streaming_supported is True

    @pytest.mark.asyncio
    async def test_unknown_channel_gets_defaults(self) -> None:
        """Unknown channel gets default capabilities."""
        resolver = ChannelResolver()
        partial = PartialExecutionContext()
        partial.channel = "unknown_channel"
        result = await resolver.resolve(partial)

        assert result.channel_capabilities is not None
        assert result.channel_capabilities.streaming_supported is True
