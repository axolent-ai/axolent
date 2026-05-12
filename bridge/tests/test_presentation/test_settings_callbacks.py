"""Tests für /settings Inline-Keyboard Callbacks (R18 Phase 2b).

Testet:
  - /settings Command öffnet Ebene A (Hauptmenü)
  - Slot-Buttons öffnen Ebene B (Modell-Auswahl)
  - Modell-Klick setzt Override und kehrt zu Ebene A zurück
  - Reset-Slot setzt einen Slot zurück
  - Reset-All zeigt Bestätigungs-Dialog
  - Reset-All-Confirm setzt alles zurück
  - Zurück-Button kehrt zum Hauptmenü zurück
  - Sprachmenü öffnet und Sprachauswahl funktioniert
  - Messages werden ge-editet, nicht neu gesendet
  - i18n: EN-Strings funktionieren
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.model_service import ModelService
from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporärer DB-Pfad für Test-Isolation."""
    return tmp_path / "test_settings.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    """Frische SQLite-Connection für jeden Test."""
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def model_service(conn: SqliteConnection) -> ModelService:
    """ModelService mit echtem SQLite-Backend."""
    return ModelService(storage=SqliteModelStorage(conn))


def _make_callback_update(
    callback_data: str,
    user_id: int = 1,
    chat_id: int = 10,
    chat_type: str = "private",
) -> MagicMock:
    """Erstellt ein gemocktes Telegram-Update für CallbackQuery."""
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
    update.callback_query = query
    return update


def _make_command_update(user_id: int = 1, chat_id: int = 10) -> MagicMock:
    """Erstellt ein gemocktes Telegram-Update für Commands."""
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
    task_router: object | None = None,
) -> MagicMock:
    """Erstellt einen gemockten Telegram-Context mit Services."""
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
        "task_router": task_router,
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


class TestSettingsCommand:
    """Tests für /settings Command (Ebene A)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_settings_command_opens_main_menu(
        self, model_service: ModelService
    ) -> None:
        """/settings sendet Nachricht mit Inline-Keyboard (6 Slots + Sprache + Reset)."""
        from presentation.handlers import handle_settings_command

        update = _make_command_update()
        context = _make_context(model_service=model_service)

        await handle_settings_command(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        text = call_args[0][0]
        assert "Einstellungen" in text

        # Keyboard muss vorhanden sein
        keyboard = call_args[1]["reply_markup"]
        assert keyboard is not None
        # 6 Slots + 1 Sprache + 1 Reset = 8 Rows
        assert len(keyboard.inline_keyboard) == 8

    async def test_settings_main_menu_shows_defaults(
        self, model_service: ModelService
    ) -> None:
        """Hauptmenü zeigt (Default) Suffix wenn kein Override gesetzt ist."""
        from presentation.handlers import handle_settings_command

        update = _make_command_update()
        context = _make_context(model_service=model_service)

        await handle_settings_command(update, context)

        keyboard = update.message.reply_text.call_args[1]["reply_markup"]
        # Erster Button (CHAT) sollte Default zeigen
        first_btn = keyboard.inline_keyboard[0][0]
        assert "(Default)" in first_btn.text
        assert "CHAT:" in first_btn.text

    async def test_settings_main_menu_shows_override(
        self, model_service: ModelService
    ) -> None:
        """Hauptmenü zeigt Override-Modell ohne (Default) Suffix."""
        from presentation.handlers import handle_settings_command

        # Setze Override für CODE Slot
        model_service.set_user_model(1, "haiku", slot="code")

        update = _make_command_update()
        context = _make_context(model_service=model_service)

        await handle_settings_command(update, context)

        keyboard = update.message.reply_text.call_args[1]["reply_markup"]
        # CODE ist der zweite Slot (index 1)
        code_btn = keyboard.inline_keyboard[1][0]
        assert "CODE:" in code_btn.text
        assert "Haiku" in code_btn.text
        assert "(Default)" not in code_btn.text


class TestSettingsSlotCallback:
    """Tests für Slot-Auswahl Callbacks (Ebene B)."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_slot_click_opens_model_selection(
        self, model_service: ModelService
    ) -> None:
        """Klick auf Slot-Button öffnet Ebene B mit Modell-Buttons."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_slot:code")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        query = update.callback_query
        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()

        call_args = query.edit_message_text.call_args
        text = call_args[0][0]
        assert "CODE" in text

        keyboard = call_args[1]["reply_markup"]
        # 3 Modelle + Zurück + Reset = 5 Rows
        assert len(keyboard.inline_keyboard) == 5

    async def test_model_click_sets_override_and_returns_to_main(
        self, model_service: ModelService
    ) -> None:
        """Klick auf Modell setzt Override und kehrt zu Ebene A zurück."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_model:code:haiku")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        # Override muss gesetzt sein
        override = model_service.get_user_model(1, "code")
        assert override is not None
        assert "haiku" in override.lower()

        # Zurück zum Hauptmenü
        query = update.callback_query
        call_args = query.edit_message_text.call_args
        text = call_args[0][0]
        assert "Einstellungen" in text

    async def test_slot_shows_current_model_marker(
        self, model_service: ModelService
    ) -> None:
        """Ebene B markiert das aktive Modell mit ●."""
        from presentation.settings_callbacks import handle_settings_callback

        # Setze Haiku als Override
        model_service.set_user_model(1, "haiku", slot="code")

        update = _make_callback_update("settings_slot:code")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # Finde den Haiku-Button
        haiku_btns = [
            btn
            for row in keyboard.inline_keyboard
            for btn in row
            if "Haiku" in btn.text
        ]
        assert len(haiku_btns) == 1
        assert "●" in haiku_btns[0].text

        # Opus und Sonnet sollten ○ haben
        other_btns = [
            btn
            for row in keyboard.inline_keyboard
            for btn in row
            if ("Opus" in btn.text or "Sonnet" in btn.text)
        ]
        for btn in other_btns:
            assert "○" in btn.text


class TestSettingsResetCallbacks:
    """Tests für Reset-Callbacks."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_slot_removes_override(
        self, model_service: ModelService
    ) -> None:
        """settings_reset:<slot> entfernt Override und zeigt Hauptmenü."""
        from presentation.settings_callbacks import handle_settings_callback

        model_service.set_user_model(1, "haiku", slot="code")
        assert model_service.get_user_model(1, "code") is not None

        update = _make_callback_update("settings_reset:code")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        # Override muss weg sein
        assert model_service.get_user_model(1, "code") is None

        # Hauptmenü muss angezeigt werden
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Einstellungen" in text

    async def test_reset_all_shows_confirmation(
        self, model_service: ModelService
    ) -> None:
        """settings_reset_all zeigt Bestätigungs-Dialog."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_reset_all")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Wirklich" in text or "Really" in text

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # 2 Buttons: Ja + Abbrechen
        assert len(keyboard.inline_keyboard) == 2

    async def test_reset_all_confirm_removes_all_overrides(
        self, model_service: ModelService
    ) -> None:
        """settings_reset_all_confirm entfernt alle Overrides."""
        from presentation.settings_callbacks import handle_settings_callback

        model_service.set_user_model(1, "haiku", slot="code")
        model_service.set_user_model(1, "opus", slot="chat")

        update = _make_callback_update("settings_reset_all_confirm")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        # Alle Overrides müssen weg sein
        assert model_service.get_user_model(1, "code") is None
        assert model_service.get_user_model(1, "chat") is None

        # Hauptmenü mit Default-Werten
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Einstellungen" in text


class TestSettingsBackCallback:
    """Tests für Zurück-Button."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_back_returns_to_main_menu(self, model_service: ModelService) -> None:
        """settings_back kehrt zum Hauptmenü zurück."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Einstellungen" in text

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        assert len(keyboard.inline_keyboard) == 8


class TestSettingsLanguageCallbacks:
    """Tests für Sprachauswahl."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_lang_menu_opens(self, model_service: ModelService) -> None:
        """settings_lang_menu öffnet Sprachauswahl."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_lang_menu")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Sprache" in text or "language" in text.lower()

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # DE + EN + Zurück = 3 Rows
        assert len(keyboard.inline_keyboard) == 3

    async def test_lang_set_changes_language(self, model_service: ModelService) -> None:
        """settings_lang:en setzt Sprache auf Englisch."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_lang:en")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        # ChatService.set_chat_language muss aufgerufen worden sein
        chat_service = context.application.bot_data["chat_service"]
        chat_service.set_chat_language.assert_called_once_with(1, 10, "en")

        # Hauptmenü in neuer Sprache
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Settings" in text


class TestSettingsMessageEditing:
    """Tests: Messages werden ge-editet, nicht neu gesendet."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_callback_edits_message_not_sends_new(
        self, model_service: ModelService
    ) -> None:
        """Callback benutzt edit_message_text, nicht reply_text."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_slot:chat")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        query = update.callback_query
        # edit_message_text MUSS aufgerufen worden sein
        query.edit_message_text.assert_called_once()
        # reply_text auf der query.message darf NICHT aufgerufen worden sein
        query.message.reply_text.assert_not_called()


class TestSettingsI18n:
    """Tests: EN-Strings funktionieren korrekt."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_english_main_menu(self, model_service: ModelService) -> None:
        """Hauptmenü in Englisch zeigt EN-Strings."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        # Sprache auf EN setzen
        chat_service = context.application.bot_data["chat_service"]
        chat_service.get_chat_language = AsyncMock(return_value="en")

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Settings" in text

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # Reset-All-Button in EN
        reset_btn = keyboard.inline_keyboard[-1][0]
        assert "Reset all" in reset_btn.text
