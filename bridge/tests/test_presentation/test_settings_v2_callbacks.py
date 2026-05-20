"""Tests for /settings v2 inline keyboard callbacks.

Tests:
  - test_settings_command_renders_v2_main_menu
  - test_v2_main_menu_has_6_categories_plus_close
  - test_model_submenu_renders_via_v2_cat
  - test_debate_submenu_renders_multi_select
  - test_personality_submenu_renders_toggles
  - test_callback_updates_settings_persisted (rate_limit, personality, timezone, debate)
  - test_back_button_returns_to_v2_main_menu
  - test_close_button_dismisses_keyboard
  - test_settings_command_creates_default_settings_for_new_user
  - test_debate_planned_provider_shows_toast_not_toggle
  - test_language_submenu_renders_all_20_languages
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.model_service import ModelService
from application.settings_service import SettingsService
from infrastructure.sqlite_storage import (
    SqliteConnection,
    SqliteModelStorage,
    SqliteSettingsStorage,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_v2.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def model_service(conn: SqliteConnection) -> ModelService:
    return ModelService(storage=SqliteModelStorage(conn))


@pytest.fixture
def settings_storage(conn: SqliteConnection) -> SqliteSettingsStorage:
    return SqliteSettingsStorage(conn)


@pytest.fixture
def settings_service(settings_storage: SqliteSettingsStorage) -> SettingsService:
    return SettingsService(storage=settings_storage)


def _make_callback_update(
    callback_data: str,
    user_id: int = 1,
    chat_id: int = 10,
    chat_type: str = "private",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()

    query = MagicMock()
    query.data = callback_data
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.username = "testuser"
    query.answer = AsyncMock()
    query.message = MagicMock()
    query.message.reply_text = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update.callback_query = query
    return update


def _make_command_update(user_id: int = 1, chat_id: int = 10) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


def _make_context(
    model_service: ModelService | None = None,
    settings_service: SettingsService | None = None,
) -> MagicMock:
    mock_chat_service = MagicMock()
    mock_chat_service.get_chat_language = AsyncMock(return_value="de")
    mock_chat_service.set_chat_language = AsyncMock()

    context = MagicMock()
    context.args = []
    context.bot = MagicMock()
    context.application = MagicMock()
    context.application.bot_data = {
        "chat_service": mock_chat_service,
        "model_service": model_service,
        "settings_service": settings_service,
        "task_router": None,
        "system_prompt": "test",
        "memory_service": None,
        "persistent_provider": None,
        "process_pool": None,
        "rate_limiter": None,
        "bookmark_service": None,
    }
    return context


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────


class TestSettingsV2MainMenu:
    """Tests for /settings command (v2 main menu)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield

    async def test_settings_command_renders_v2_main_menu(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """/settings sends a message with the v2 inline keyboard."""
        from presentation.handlers import handle_settings_command

        update = _make_command_update()
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_command(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        text = call_args[0][0]
        # v2 title must be present
        assert "Einstellungen" in text

    async def test_v2_main_menu_has_6_categories_plus_close(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """Main menu keyboard has 4 rows: 3x2 categories + 1 close."""
        from presentation.handlers import handle_settings_command

        update = _make_command_update()
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_command(update, context)

        keyboard = update.message.reply_text.call_args[1]["reply_markup"]
        # 3 rows of 2 buttons + 1 row with close = 4 rows
        assert len(keyboard.inline_keyboard) == 4
        # Each of the first 3 rows has 2 buttons
        for row in keyboard.inline_keyboard[:3]:
            assert len(row) == 2

    async def test_settings_command_creates_default_settings_for_new_user(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """New user gets UserSettings with defaults (no crash, no empty state)."""
        from presentation.handlers import handle_settings_command

        update = _make_command_update(user_id=9999)
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_command(update, context)

        # Settings service should be queryable and return defaults
        settings = await settings_service.get_settings(9999)
        assert settings.user_id == 9999
        assert settings.model is None  # no explicit model set
        assert settings.rate_limit_profile == "normal"


class TestSettingsV2CategoryMenus:
    """Tests for category sub-menus opened via settings_v2_cat:<category>."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield

    async def test_model_submenu_renders_via_v2_cat(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_cat:model opens model sub-menu."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_cat:model")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        query = update.callback_query
        query.edit_message_text.assert_called_once()
        text = query.edit_message_text.call_args[0][0]
        assert "Modell" in text or "Model" in text

        keyboard = query.edit_message_text.call_args[1]["reply_markup"]
        # Must have at least a back button
        all_btns = [btn for row in keyboard.inline_keyboard for btn in row]
        back_btns = [b for b in all_btns if "settings_v2_model_back" in b.callback_data]
        assert len(back_btns) == 1

    async def test_debate_submenu_renders_multi_select(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_cat:debate shows checkboxes for providers."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_cat:debate")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        all_btns = [btn for row in keyboard.inline_keyboard for btn in row]
        # Should have ☑ or ☐ in at least one button
        checkbox_btns = [b for b in all_btns if "☑" in b.text or "☐" in b.text]
        assert len(checkbox_btns) >= 2

    async def test_personality_submenu_renders_toggles(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_cat:personality shows 6 toggles (☑/☐) and a back button."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_cat:personality")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        all_btns = [btn for row in keyboard.inline_keyboard for btn in row]
        # 6 feature toggles + 1 back = 7 buttons
        assert len(all_btns) == 7
        checkbox_btns = [b for b in all_btns if "☑" in b.text or "☐" in b.text]
        assert len(checkbox_btns) == 6

    async def test_language_submenu_renders_all_20_languages(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_cat:language shows all 20 language options."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_cat:language")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        all_btns = [btn for row in keyboard.inline_keyboard for btn in row]
        lang_btns = [b for b in all_btns if "settings_v2_lang:" in b.callback_data]
        assert len(lang_btns) == 20


class TestSettingsV2Persistence:
    """Tests that callbacks persist settings correctly."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield

    async def test_debate_toggle_persists(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_debate_toggle:llama adds llama to providers."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_debate_toggle:llama")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        settings = await settings_service.get_settings(user_id=1)
        assert "llama" in settings.debate_providers

    async def test_rate_limit_callback_persists(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_rl:power sets rate limit to power."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_rl:power")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        settings = await settings_service.get_settings(user_id=1)
        assert settings.rate_limit_profile == "power"

    async def test_personality_toggle_persists(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_pf:personality_p4 toggles P4 to ON (default is OFF)."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_pf:personality_p4")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        settings = await settings_service.get_settings(user_id=1)
        assert settings.personality_p4_confidence_signal is True

    async def test_timezone_set_persists(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_tz:Europe/Vienna persists the timezone."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_tz:Europe/Vienna")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        settings = await settings_service.get_settings(user_id=1)
        assert settings.timezone == "Europe/Vienna"


class TestSettingsV2Navigation:
    """Tests for back/close navigation."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield

    async def test_back_button_returns_to_v2_main_menu(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_lang_back renders v2 main menu."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_lang_back")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Einstellungen" in text

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # v2 main: 4 rows
        assert len(keyboard.inline_keyboard) == 4

    async def test_model_back_returns_to_v2_main_menu(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_model_back renders v2 main menu."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_model_back")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        assert len(keyboard.inline_keyboard) == 4

    async def test_close_button_dismisses_keyboard(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """settings_v2_close removes the keyboard (edit_message_reply_markup None)."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_close")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        query = update.callback_query
        query.edit_message_reply_markup.assert_called_once_with(reply_markup=None)
        query.edit_message_text.assert_not_called()


class TestSettingsV2DebatePlanned:
    """Tests that planned providers show toast instead of toggling."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield

    async def test_debate_planned_provider_shows_toast_not_toggle(
        self,
        model_service: ModelService,
        settings_service: SettingsService,
    ) -> None:
        """Tapping a planned provider answers with alert, does not edit message."""
        from presentation.settings_callbacks import handle_settings_v2_callback

        update = _make_callback_update("settings_v2_debate_planned:gpt4o")
        context = _make_context(
            model_service=model_service, settings_service=settings_service
        )
        await handle_settings_v2_callback(update, context)

        query = update.callback_query
        # answer() must have been called with show_alert=True
        query.answer.assert_called()
        call_kwargs = query.answer.call_args[1]
        assert call_kwargs.get("show_alert") is True
        # edit_message_text must NOT have been called (no menu change)
        query.edit_message_text.assert_not_called()
