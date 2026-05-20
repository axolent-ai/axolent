"""Tests for SettingsService (application layer).

Tests:
  - get_settings returns defaults for new user (no DB row yet)
  - set_model persists to DB
  - toggle_debate_provider persists and toggles correctly
  - toggle_personality persists
  - set_rate_limit persists
  - set_timezone persists
  - toggle_debate_provider raises on planned/unknown provider
  - toggle_personality raises on unknown flag
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.settings_service import (
    DEFAULT_DEBATE_PROVIDERS,
    SettingsService,
    UserSettings,
)
from infrastructure.sqlite_storage import SqliteConnection, SqliteSettingsStorage


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_settings_service.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def storage(conn: SqliteConnection) -> SqliteSettingsStorage:
    return SqliteSettingsStorage(conn)


@pytest.fixture
def service(storage: SqliteSettingsStorage) -> SettingsService:
    return SettingsService(storage=storage)


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────


class TestGetSettingsDefaults:
    """get_settings returns sensible defaults for a user with no DB row."""

    async def test_returns_user_settings_instance(
        self, service: SettingsService
    ) -> None:
        settings = await service.get_settings(user_id=999)
        assert isinstance(settings, UserSettings)

    async def test_default_language_is_none(self, service: SettingsService) -> None:
        settings = await service.get_settings(user_id=999)
        assert settings.language is None

    async def test_default_model_is_none(self, service: SettingsService) -> None:
        settings = await service.get_settings(user_id=999)
        assert settings.model is None

    async def test_default_rate_limit_profile_is_normal(
        self, service: SettingsService
    ) -> None:
        settings = await service.get_settings(user_id=999)
        assert settings.rate_limit_profile == "normal"

    async def test_default_debate_providers(self, service: SettingsService) -> None:
        settings = await service.get_settings(user_id=999)
        assert settings.debate_providers == DEFAULT_DEBATE_PROVIDERS

    async def test_default_personality_p1_is_on(self, service: SettingsService) -> None:
        settings = await service.get_settings(user_id=999)
        assert settings.personality_p1_proactive is True

    async def test_default_personality_p4_is_off(
        self, service: SettingsService
    ) -> None:
        settings = await service.get_settings(user_id=999)
        assert settings.personality_p4_confidence_signal is False

    async def test_default_timezone_is_utc(self, service: SettingsService) -> None:
        settings = await service.get_settings(user_id=999)
        assert settings.timezone == "UTC"

    async def test_user_id_matches(self, service: SettingsService) -> None:
        settings = await service.get_settings(user_id=42)
        assert settings.user_id == 42


class TestSetModel:
    """set_model persists the value and get_settings reflects it."""

    async def test_set_model_persists(self, service: SettingsService) -> None:
        await service.set_model(user_id=1, model="claude-opus-4-7")
        settings = await service.get_settings(user_id=1)
        assert settings.model == "claude-opus-4-7"

    async def test_clear_model_with_none(self, service: SettingsService) -> None:
        await service.set_model(user_id=1, model="claude-sonnet-4-6")
        await service.set_model(user_id=1, model=None)
        settings = await service.get_settings(user_id=1)
        assert settings.model is None


class TestToggleDebateProvider:
    """toggle_debate_provider toggles correctly and rejects planned/unknown."""

    async def test_toggle_adds_provider(self, service: SettingsService) -> None:
        result = await service.toggle_debate_provider(user_id=1, provider="llama")
        assert "llama" in result

    async def test_toggle_removes_provider(self, service: SettingsService) -> None:
        await service.toggle_debate_provider(user_id=1, provider="llama")
        result = await service.toggle_debate_provider(user_id=1, provider="llama")
        assert "llama" not in result

    async def test_toggle_persists(self, service: SettingsService) -> None:
        await service.toggle_debate_provider(user_id=1, provider="llama")
        settings = await service.get_settings(user_id=1)
        assert "llama" in settings.debate_providers

    async def test_toggle_planned_raises(self, service: SettingsService) -> None:
        with pytest.raises(ValueError, match="planned"):
            await service.toggle_debate_provider(user_id=1, provider="gpt4o")

    async def test_toggle_unknown_raises(self, service: SettingsService) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            await service.toggle_debate_provider(user_id=1, provider="nonexistent")


class TestTogglePersonality:
    """toggle_personality toggles flags and rejects unknown flags."""

    async def test_toggle_p4_on(self, service: SettingsService) -> None:
        await service.toggle_personality(user_id=1, feature="personality_p4", on=True)
        settings = await service.get_settings(user_id=1)
        assert settings.personality_p4_confidence_signal is True

    async def test_toggle_p1_off(self, service: SettingsService) -> None:
        await service.toggle_personality(user_id=1, feature="personality_p1", on=False)
        settings = await service.get_settings(user_id=1)
        assert settings.personality_p1_proactive is False

    async def test_toggle_unknown_raises(self, service: SettingsService) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            await service.toggle_personality(
                user_id=1, feature="personality_p99", on=True
            )


class TestSetRateLimit:
    """set_rate_limit persists and validates profiles."""

    async def test_set_valid_profile(self, service: SettingsService) -> None:
        success = await service.set_rate_limit(user_id=1, profile="power")
        assert success is True
        settings = await service.get_settings(user_id=1)
        assert settings.rate_limit_profile == "power"

    async def test_set_invalid_profile_returns_false(
        self, service: SettingsService
    ) -> None:
        success = await service.set_rate_limit(user_id=1, profile="megaboost")
        assert success is False

    async def test_all_valid_profiles(self, service: SettingsService) -> None:
        for profile in ("light", "normal", "power", "unlimited"):
            success = await service.set_rate_limit(user_id=1, profile=profile)
            assert success is True


class TestSetTimezone:
    """set_timezone persists IANA timezone strings."""

    async def test_set_timezone_vienna(self, service: SettingsService) -> None:
        await service.set_timezone(user_id=1, tz="Europe/Vienna")
        settings = await service.get_settings(user_id=1)
        assert settings.timezone == "Europe/Vienna"

    async def test_overwrite_timezone(self, service: SettingsService) -> None:
        await service.set_timezone(user_id=1, tz="Europe/Berlin")
        await service.set_timezone(user_id=1, tz="Asia/Tokyo")
        settings = await service.get_settings(user_id=1)
        assert settings.timezone == "Asia/Tokyo"
