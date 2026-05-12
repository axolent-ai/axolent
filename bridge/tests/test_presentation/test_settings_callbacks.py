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
        # 6 Slots + 1 Sprache = 7 Rows (kein "Alle zurücksetzen" ohne Slot-Overrides)
        assert len(keyboard.inline_keyboard) == 7

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
        # 6 Slots + 1 Sprache = 7 Rows (kein "Alle zurücksetzen" ohne Slot-Overrides)
        assert len(keyboard.inline_keyboard) == 7


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


class TestSettingsImplicitReset:
    """R18 Phase 2 Bug-Fix: Settings-UI zeigt (Default) nach implizitem Reset."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @pytest.fixture
    def model_service_with_defaults(self, conn: SqliteConnection) -> ModelService:
        """ModelService mit Slot-Defaults (Produktions-Config)."""
        slot_defaults = {
            "code": "claude-opus-4-7",
            "chat": "claude-sonnet-4-6",
            "quick": "claude-haiku-4-5-20251001",
            "reason": "claude-opus-4-7",
            "research": "claude-opus-4-7",
            "creative": "claude-sonnet-4-6",
        }
        return ModelService(
            storage=SqliteModelStorage(conn), slot_defaults=slot_defaults
        )

    async def test_settings_shows_default_after_implicit_reset(
        self, model_service_with_defaults: ModelService
    ) -> None:
        """Nach Wahl des Slot-Default-Modells zeigt Hauptmenü (Default)."""
        from presentation.settings_callbacks import handle_settings_callback

        svc = model_service_with_defaults
        uid = 1

        # 1. Haiku als Override fuer CODE setzen
        svc.set_user_model(uid, "haiku", slot="code")
        assert svc.get_user_model(uid, "code") is not None

        # 2. Opus waehlen (= CODE Default) -> impliziter Reset
        svc.set_user_model(uid, "opus", slot="code")
        assert svc.get_user_model(uid, "code") is None

        # 3. Settings-Hauptmenü aufrufen
        update = _make_callback_update("settings_back")
        context = _make_context(model_service=svc)

        await handle_settings_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]

        # CODE ist der zweite Slot (Index 1)
        code_btn = keyboard.inline_keyboard[1][0]
        assert "CODE:" in code_btn.text
        assert "(Default)" in code_btn.text, (
            f"Nach implizitem Reset muss (Default) angezeigt werden, "
            f"Button-Text war: '{code_btn.text}'"
        )

    async def test_model_click_default_triggers_implicit_reset(
        self, model_service_with_defaults: ModelService
    ) -> None:
        """Klick auf Default-Modell in Ebene B loest impliziten Reset aus."""
        from presentation.settings_callbacks import handle_settings_callback

        svc = model_service_with_defaults
        uid = 1

        # Erst Haiku als Override setzen
        svc.set_user_model(uid, "haiku", slot="code")

        # Dann via Callback Opus waehlen (= Default fuer CODE)
        update = _make_callback_update("settings_model:code:opus")
        context = _make_context(model_service=svc)

        await handle_settings_callback(update, context)

        # Override muss weg sein
        assert svc.get_user_model(uid, "code") is None

        # Hauptmenü muss (Default) zeigen
        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        code_btn = keyboard.inline_keyboard[1][0]
        assert "(Default)" in code_btn.text


class TestSettingsI18n:
    """Tests: EN-Strings funktionieren korrekt."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_english_main_menu(self, model_service: ModelService) -> None:
        """Hauptmenü in Englisch zeigt EN-Strings (inkl. Reset all bei Slot-Override)."""
        from presentation.settings_callbacks import handle_settings_callback

        # Slot-Override setzen damit "Reset all" Button erscheint
        model_service.set_user_model(1, "haiku", slot="code")

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        # Sprache auf EN setzen
        chat_service = context.application.bot_data["chat_service"]
        chat_service.get_chat_language = AsyncMock(return_value="en")

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Settings" in text

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # Reset-All-Button in EN (letzter Button, da Slot-Override existiert)
        reset_btn = keyboard.inline_keyboard[-1][0]
        assert "Reset all" in reset_btn.text


# ──────────────────────────────────────────────────────────────
# Global-Override Tests (Fix A + Fix B)
# ──────────────────────────────────────────────────────────────


class TestGlobalOverrideShownInSettings:
    """Settings-UI zeigt globalen Override korrekt an."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_global_override_shows_headline_in_text(
        self, model_service: ModelService
    ) -> None:
        """Nach /setmodel opus zeigt /settings die Override-Headline im Message-Text."""
        from presentation.settings_callbacks import handle_settings_callback

        # Globalen Override setzen (slot="global" ist Default)
        model_service.set_user_model(1, "opus")

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        # Headline muss im Message-Text stehen
        assert "<b>Globaler Override aktiv:" in text
        assert "Opus" in text

        # Keyboard darf KEINEN Button mit "Globaler Override" enthalten
        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        for row in keyboard.inline_keyboard:
            for btn in row:
                assert "Globaler Override" not in btn.text

    async def test_global_override_shows_reset_button(
        self, model_service: ModelService
    ) -> None:
        """Nach globalem Override gibt es einen Reset-Global-Button."""
        from presentation.settings_callbacks import handle_settings_callback

        model_service.set_user_model(1, "opus")

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # Erster Button muss der Reset-Global-Button sein (Headline ist jetzt im Text)
        reset_btn = keyboard.inline_keyboard[0][0]
        assert "settings_reset_global" in reset_btn.callback_data

    async def test_global_override_slots_show_global_suffix(
        self, model_service: ModelService
    ) -> None:
        """Alle Slots zeigen das globale Modell mit (global) Suffix."""
        from presentation.settings_callbacks import handle_settings_callback

        model_service.set_user_model(1, "opus")

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # Slots starten ab Index 1 (nach Reset-Global, Headline ist jetzt im Text)
        for i in range(1, 7):  # 6 Slots
            btn = keyboard.inline_keyboard[i][0]
            assert "(global)" in btn.text, (
                f"Slot-Button '{btn.text}' muss '(global)' enthalten"
            )
            assert "(Default)" not in btn.text

    async def test_no_global_override_no_headline_in_text(
        self, model_service: ModelService
    ) -> None:
        """Ohne globalen Override: keine Headline im Text, 8 Rows (6 Slots + Sprache + Reset)."""
        from presentation.settings_callbacks import handle_settings_callback

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "<b>Globaler Override aktiv:" not in text

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # 6 Slots + 1 Sprache = 7 Rows (kein "Alle zurücksetzen" ohne Slot-Overrides)
        assert len(keyboard.inline_keyboard) == 7

    async def test_global_override_with_slot_override_mixed(
        self, model_service: ModelService
    ) -> None:
        """Slot-Override hat Vorrang vor Global-Override in der Anzeige."""
        from presentation.settings_callbacks import handle_settings_callback

        # Global auf Opus, aber CODE explizit auf Haiku
        model_service.set_user_model(1, "opus")
        model_service.set_user_model(1, "haiku", slot="code")

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        # CODE (Index 2, nach Reset-Global + CHAT) zeigt Haiku ohne "(global)"
        code_btn = keyboard.inline_keyboard[2][0]
        assert "CODE:" in code_btn.text
        assert "Haiku" in code_btn.text
        assert "(global)" not in code_btn.text

        # CHAT (Index 1) zeigt Opus mit "(global)"
        chat_btn = keyboard.inline_keyboard[1][0]
        assert "CHAT:" in chat_btn.text
        assert "Opus" in chat_btn.text
        assert "(global)" in chat_btn.text

    async def test_global_override_en_strings(
        self, model_service: ModelService
    ) -> None:
        """Globaler Override in EN zeigt englische Strings im Message-Text."""
        from presentation.settings_callbacks import handle_settings_callback

        model_service.set_user_model(1, "opus")

        update = _make_callback_update("settings_back")
        context = _make_context(model_service=model_service)
        chat_service = context.application.bot_data["chat_service"]
        chat_service.get_chat_language = AsyncMock(return_value="en")

        await handle_settings_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "<b>Global Override active:" in text
        assert "Opus" in text

        # Keyboard darf KEINEN Headline-Button enthalten
        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        for row in keyboard.inline_keyboard:
            for btn in row:
                assert "Global override" not in btn.text
                assert "Global Override" not in btn.text


class TestSettingsResetGlobalCallback:
    """settings_reset_global löscht den globalen Override."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_global_removes_override(
        self, model_service: ModelService
    ) -> None:
        """settings_reset_global entfernt den globalen Override."""
        from presentation.settings_callbacks import handle_settings_callback

        model_service.set_user_model(1, "opus")
        assert model_service.get_user_model(1, "global") is not None

        update = _make_callback_update("settings_reset_global")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        # Override muss weg sein
        assert model_service.get_user_model(1, "global") is None

        # Hauptmenü ohne Headline, ohne "Alle zurücksetzen" (keine Slot-Overrides)
        keyboard = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        assert len(keyboard.inline_keyboard) == 7

    async def test_reset_global_preserves_slot_overrides(
        self, model_service: ModelService
    ) -> None:
        """settings_reset_global entfernt nur global, nicht Slot-Overrides."""
        from presentation.settings_callbacks import handle_settings_callback

        model_service.set_user_model(1, "opus")
        model_service.set_user_model(1, "haiku", slot="code")

        update = _make_callback_update("settings_reset_global")
        context = _make_context(model_service=model_service)

        await handle_settings_callback(update, context)

        # Global weg, Code bleibt
        assert model_service.get_user_model(1, "global") is None
        assert model_service.get_user_model(1, "code") is not None


class TestResetAllButtonConditional:
    """'Alle zurücksetzen' Button erscheint nur bei pro-Slot Overrides."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_all_hidden_when_nothing_set(
        self, model_service: ModelService
    ) -> None:
        """Ohne Overrides: kein 'Alle zurücksetzen' Button."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        context = _make_context(model_service=model_service)
        _, keyboard = build_main_menu_keyboard(1, model_service, context, "de")

        btn_texts = [btn.text for row in keyboard.inline_keyboard for btn in row]
        assert not any("zurücksetzen" in t.lower() and "Alle" in t for t in btn_texts)
        assert not any(
            "settings_reset_all" in btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
        )

    async def test_reset_all_hidden_when_only_global_active(
        self, model_service: ModelService
    ) -> None:
        """Nur Global-Override aktiv: kein 'Alle zurücksetzen' Button."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        model_service.set_user_model(1, "opus")

        context = _make_context(model_service=model_service)
        _, keyboard = build_main_menu_keyboard(1, model_service, context, "de")

        reset_all_btns = [
            btn
            for row in keyboard.inline_keyboard
            for btn in row
            if btn.callback_data == "settings_reset_all"
        ]
        assert len(reset_all_btns) == 0

    async def test_reset_all_visible_when_slot_overrides_exist(
        self, model_service: ModelService
    ) -> None:
        """Pro-Slot Override existiert: 'Alle zurücksetzen' Button sichtbar."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        model_service.set_user_model(1, "haiku", slot="code")

        context = _make_context(model_service=model_service)
        _, keyboard = build_main_menu_keyboard(1, model_service, context, "de")

        reset_all_btns = [
            btn
            for row in keyboard.inline_keyboard
            for btn in row
            if btn.callback_data == "settings_reset_all"
        ]
        assert len(reset_all_btns) == 1


class TestGlobalOverrideTakesPrecedence:
    """TaskRouter berücksichtigt global-Override korrekt."""

    def test_global_override_used_when_no_slot_override(
        self, model_service: ModelService
    ) -> None:
        """Global-Override wird genutzt wenn kein Slot-Override existiert."""
        from application.task_router import SlotConfig, TaskRouter
        from domain.task_slot import TaskSlot

        configs = [
            SlotConfig(
                slot=TaskSlot.CHAT,
                default_model="haiku",
                fallback=True,
            ),
        ]
        router = TaskRouter(configs, model_service=model_service)

        # Global auf Opus setzen
        model_service.set_user_model(1, "opus")

        result = router.resolve_model(1, TaskSlot.CHAT)
        assert result == "claude-opus-4-7"

    def test_slot_override_beats_global(self, model_service: ModelService) -> None:
        """Slot-Override hat Vorrang vor Global-Override."""
        from application.task_router import SlotConfig, TaskRouter
        from domain.task_slot import TaskSlot

        configs = [
            SlotConfig(
                slot=TaskSlot.CHAT,
                default_model="haiku",
                fallback=True,
            ),
        ]
        router = TaskRouter(configs, model_service=model_service)

        # Global Opus, Chat Sonnet
        model_service.set_user_model(1, "opus")
        model_service.set_user_model(1, "sonnet", slot="chat")

        result = router.resolve_model(1, TaskSlot.CHAT)
        assert result == "claude-sonnet-4-6"


class TestImplicitResetAuditLog:
    """Audit-Log differenziert implicit_reset vs set."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    @pytest.fixture
    def model_service_with_defaults(self, conn: SqliteConnection) -> ModelService:
        """ModelService mit Slot-Defaults."""
        slot_defaults = {
            "code": "claude-opus-4-7",
            "chat": "claude-sonnet-4-6",
        }
        return ModelService(
            storage=SqliteModelStorage(conn), slot_defaults=slot_defaults
        )

    async def test_implicit_reset_audit_action(
        self, model_service_with_defaults: ModelService
    ) -> None:
        """Bei impliziter Reset wird action 'settings_model_implicit_reset' geloggt."""
        from presentation.settings_callbacks import handle_settings_callback

        svc = model_service_with_defaults

        # Erst Override setzen (haiku statt default opus fuer code)
        svc.set_user_model(1, "haiku", slot="code")

        # Dann Default waehlen (opus = code default) -> implicit reset
        update = _make_callback_update("settings_model:code:opus")
        context = _make_context(model_service=svc)

        with patch("presentation.settings_callbacks.log_command_audit") as mock_audit:
            await handle_settings_callback(update, context)

            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["action"] == "settings_model_implicit_reset"
            assert "implicit_reset" in call_kwargs["details"]
            assert "default-equal" in call_kwargs["details"]

    async def test_normal_set_audit_action(
        self, model_service_with_defaults: ModelService
    ) -> None:
        """Bei normalem Set wird action 'settings_model' geloggt."""
        from presentation.settings_callbacks import handle_settings_callback

        svc = model_service_with_defaults

        update = _make_callback_update("settings_model:code:haiku")
        context = _make_context(model_service=svc)

        with patch("presentation.settings_callbacks.log_command_audit") as mock_audit:
            await handle_settings_callback(update, context)

            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["action"] == "settings_model"
            assert "set slot=code" in call_kwargs["details"]
