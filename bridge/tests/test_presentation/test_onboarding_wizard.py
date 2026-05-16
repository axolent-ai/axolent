"""Tests for the setup wizard (onboarding).

Tests the complete wizard flow: start, language selection, completion,
skip behavior, /onboarding command, and the 3-message hint logic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.onboarding import VALID_LANGUAGE_CODES
from infrastructure.onboarding_storage import OnboardingStorage
from infrastructure.sqlite_storage import SqliteConnection


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporärer DB-Pfad."""
    return tmp_path / "test_wizard.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    """Frische SQLite-Connection."""
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def onboarding_storage(conn: SqliteConnection) -> OnboardingStorage:
    """OnboardingStorage mit echtem SQLite-Backend."""
    return OnboardingStorage(conn)


@pytest.fixture(autouse=True)
def _allow_all() -> None:
    """Whitelist-Bypass für alle Tests."""
    with patch("presentation.decorators.ALLOW_ALL_USERS", True):
        yield  # type: ignore[misc]


def _make_update(user_id: int = 1, chat_id: int = 10) -> MagicMock:
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


def _make_callback_update(data: str, user_id: int = 1, chat_id: int = 10) -> MagicMock:
    """Erstellt ein gemocktes Telegram-Update für Callbacks."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.callback_query = MagicMock()
    update.callback_query.data = data
    update.callback_query.from_user = update.effective_user
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.reply_text = AsyncMock()
    return update


def _make_context(
    onboarding_storage: OnboardingStorage | None = None,
    chat_service: object | None = None,
) -> MagicMock:
    """Erstellt einen gemockten Telegram-Context mit bot_data."""
    context = MagicMock()
    context.args = []

    if chat_service is None:
        mock_chat_service = MagicMock()
        mock_chat_service.get_chat_language = AsyncMock(return_value="de")
        mock_chat_service.set_chat_language = AsyncMock()
        mock_chat_service.save_static_response_to_history = AsyncMock()
    else:
        mock_chat_service = chat_service

    context.application = MagicMock()
    context.application.bot_data = {
        "chat_service": mock_chat_service,
        "system_prompt": "Test prompt.",
        "onboarding_storage": onboarding_storage,
    }
    return context


# ──────────────────────────────────────────────────────────────
# Wizard Start Tests
# ──────────────────────────────────────────────────────────────


class TestWizardStart:
    """Tests für den Wizard-Start via /start."""

    async def test_start_shows_wizard_for_new_user(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/start zeigt Wizard für nicht-onboarded User."""
        from presentation.handlers import handle_start_command

        update = _make_update()
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_start_command(update, context)

        # Wizard should send language selection with keyboard
        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        text = call_args[0][0]
        assert "Willkommen" in text or "Welcome" in text
        # Keyboard should be present
        assert "reply_markup" in call_args[1]
        keyboard = call_args[1]["reply_markup"]
        assert keyboard is not None

    async def test_start_shows_welcome_for_onboarded_user(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/start shows normal welcome message for onboarded user."""
        from domain.onboarding import get_start_welcome_text
        from presentation.handlers import handle_start_command

        onboarding_storage.set_onboarded(1)

        update = _make_update()
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_start_command(update, context)

        # Default language is "de" (no sticky language set), so DE welcome text
        expected = get_start_welcome_text("de")
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert text == expected

    async def test_start_after_wizard_respects_sticky_language(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/start nach Wizard mit Sticky-Language zeigt Text in gewaehlter Sprache.

        Regression-Test fuer Bug: /start ignorierte Sticky-Language nach
        /onboarding und fiel auf Englisch zurueck weil nur de/en Uebersetzungen
        in _START_WELCOME_TEXTS existierten.
        """
        from i18n.domain.i18n import t
        from presentation.handlers import handle_start_command

        onboarding_storage.set_onboarded(1)

        # Chat-Service mit Sticky-Language "tr" (Tuerkisch)
        mock_chat_service = MagicMock()
        mock_chat_service.get_chat_language = AsyncMock(return_value="tr")
        mock_chat_service.save_static_response_to_history = AsyncMock()

        update = _make_update()
        context = _make_context(
            onboarding_storage=onboarding_storage,
            chat_service=mock_chat_service,
        )

        await handle_start_command(update, context)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        expected = t("start.welcome", "tr")
        assert text == expected
        assert "Axolent" in text
        # Darf nicht auf Englisch oder Deutsch fallen
        assert "Send me" not in text
        assert "Schick mir" not in text


# ──────────────────────────────────────────────────────────────
# Complete Click Sequence: DE
# ──────────────────────────────────────────────────────────────


class TestWizardFlowDE:
    """Kompletter Wizard-Flow auf Deutsch."""

    async def test_full_flow_deutsch(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Deutsch wählen -> Step 2 -> Done = onboarded."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        # Step 1: Wähle Deutsch
        update = _make_callback_update("wizard_lang:de")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        # Check Step 2 was shown
        call_args = update.callback_query.edit_message_text.call_args
        text = call_args[0][0]
        assert "Deutsch" in text
        assert "startklar" in text
        assert "/settings" in text
        assert "/help" in text

        # Verify lang was saved but not yet onboarded
        state = onboarding_storage.get_state(1)
        assert state is not None
        assert state.wizard_lang == "de"
        assert state.onboarded is False

        # Step 2: Click "Los geht's"
        update2 = _make_callback_update("wizard_done")
        await handle_wizard_callback(update2, context)

        # Now onboarded
        assert onboarding_storage.is_onboarded(1) is True

    async def test_step2_has_completion_keyboard(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Step 2 hat einen 'Los geht's'-Button."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        update = _make_callback_update("wizard_lang:de")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        call_args = update.callback_query.edit_message_text.call_args
        keyboard = call_args[1].get("reply_markup")
        assert keyboard is not None
        # Should have exactly 1 row with 1 button
        assert len(keyboard.inline_keyboard) == 1
        button = keyboard.inline_keyboard[0][0]
        assert "wizard_done" in button.callback_data


# ──────────────────────────────────────────────────────────────
# Complete Click Sequence: EN
# ──────────────────────────────────────────────────────────────


class TestWizardFlowEN:
    """Kompletter Wizard-Flow auf Englisch."""

    async def test_full_flow_english(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """English wählen -> Step 2 -> Done = onboarded."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        # Step 1: Choose English
        update = _make_callback_update("wizard_lang:en")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        # Check Step 2 in English
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "English" in text
        assert "all set" in text
        assert "/settings" in text

        # Step 2: Click done
        update2 = _make_callback_update("wizard_done")
        await handle_wizard_callback(update2, context)

        assert onboarding_storage.is_onboarded(1) is True
        state = onboarding_storage.get_state(1)
        assert state.wizard_lang == "en"


# ──────────────────────────────────────────────────────────────
# Auto-Detect Flow
# ──────────────────────────────────────────────────────────────


class TestWizardFlowAuto:
    """Wizard-Flow mit Auto-Erkennung."""

    async def test_auto_detect_flow(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Auto-Detect -> Step 2 -> Done = onboarded mit lang=auto."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        # Step 1: Auto-Detect
        update = _make_callback_update("wizard_lang_auto")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        # Step 2 should show (in DE as fallback)
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Auto-detect" in text

        # Verify lang=auto saved
        state = onboarding_storage.get_state(1)
        assert state is not None
        assert state.wizard_lang == "auto"

        # Complete
        update2 = _make_callback_update("wizard_done")
        await handle_wizard_callback(update2, context)

        assert onboarding_storage.is_onboarded(1) is True


# ──────────────────────────────────────────────────────────────
# Skip Behavior
# ──────────────────────────────────────────────────────────────


class TestWizardSkip:
    """Tests für Skip-Verhalten."""

    async def test_skip_step1_does_not_onboard(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Skip in Step 1 markiert User NICHT als onboarded."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        update = _make_callback_update("wizard_skip")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        assert onboarding_storage.is_onboarded(1) is False

    async def test_skip_step2_onboards_with_lang(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Skip in Step 2 (nach Sprachwahl) markiert als onboarded."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        # First choose a language
        onboarding_storage.set_wizard_lang(1, "fr")

        # Then skip
        update = _make_callback_update("wizard_skip")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        assert onboarding_storage.is_onboarded(1) is True


# ──────────────────────────────────────────────────────────────
# /onboarding Command
# ──────────────────────────────────────────────────────────────


class TestOnboardingCommand:
    """/onboarding startet Wizard manuell."""

    async def test_onboarding_starts_wizard_for_onboarded_user(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/onboarding startet Wizard auch wenn User schon onboarded ist."""
        from presentation.handlers import handle_onboarding_command

        onboarding_storage.set_onboarded(1)

        update = _make_update()
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_onboarding_command(update, context)

        # Should show language selection
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        # is_restart=True adds a prefix
        assert "Willkommen" in text or "Welcome" in text


# ──────────────────────────────────────────────────────────────
# Onboarded Flag Tests
# ──────────────────────────────────────────────────────────────


class TestOnboardedFlag:
    """Tests für das onboarded-Flag."""

    async def test_flag_set_after_wizard_done(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """wizard_done setzt onboarded=True."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        # Choose lang then complete
        onboarding_storage.set_wizard_lang(1, "de")

        update = _make_callback_update("wizard_done")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        assert onboarding_storage.is_onboarded(1) is True

    async def test_flag_not_set_after_lang_select(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Sprachwahl allein setzt onboarded NICHT."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        update = _make_callback_update("wizard_lang:de")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        assert onboarding_storage.is_onboarded(1) is False


# ──────────────────────────────────────────────────────────────
# Parametrized: All 20 Languages + Auto
# ──────────────────────────────────────────────────────────────


class TestAllLanguagesValid:
    """Parametrized test: alle 20 Sprachen + Auto sind valide Wizard-Optionen."""

    @pytest.mark.parametrize("lang_code", sorted(VALID_LANGUAGE_CODES))
    async def test_language_code_valid(
        self, lang_code: str, onboarding_storage: OnboardingStorage
    ) -> None:
        """Sprach-Code {lang_code} ist gültig und verarbeitbar."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        if lang_code == "auto":
            callback_data = "wizard_lang_auto"
        else:
            callback_data = f"wizard_lang:{lang_code}"

        update = _make_callback_update(callback_data, user_id=hash(lang_code) % 100000)
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        # Step 2 should have been shown (edit_message_text called)
        update.callback_query.edit_message_text.assert_called_once()
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert len(text) > 10  # non-empty Step 2 text


# ──────────────────────────────────────────────────────────────
# 3-Message Hint Logic
# ──────────────────────────────────────────────────────────────


class TestThreeMessageHint:
    """Tests für die 3-Nachrichten-Hint-Logik bei übersprungenen Usern."""

    async def test_hint_after_3_messages(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Nach der 3. Nachricht ohne Onboarding wird Hint angezeigt."""
        # Simulate: user is NOT onboarded, sends 3 messages
        # We can't easily test the full handle_message flow (it calls LLM),
        # so we test the storage logic directly

        # Messages 1 and 2: no hint
        onboarding_storage.increment_skip_count(1)
        onboarding_storage.increment_skip_count(1)
        assert not onboarding_storage.is_hint_shown(1)

        # Message 3: should trigger hint
        count = onboarding_storage.increment_skip_count(1)
        assert count == 3

        # Mark hint as shown
        onboarding_storage.set_hint_shown(1)
        assert onboarding_storage.is_hint_shown(1)

    async def test_hint_not_repeated(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Hint wird nur einmal angezeigt."""
        onboarding_storage.set_hint_shown(1)

        # Further increments don't reset hint_shown
        onboarding_storage.increment_skip_count(1)
        onboarding_storage.increment_skip_count(1)
        assert onboarding_storage.is_hint_shown(1) is True


# ──────────────────────────────────────────────────────────────
# Snapshot Tests: Wizard Screens
# ──────────────────────────────────────────────────────────────


class TestWizardScreenSnapshots:
    """Snapshot-Tests für die Wizard-Screens."""

    def test_step1_keyboard_structure_de(self) -> None:
        """Step 1 Keyboard hat 7 Reihen (5 Sprach + Auto + Skip)."""
        from presentation.onboarding_callbacks import build_language_keyboard

        keyboard = build_language_keyboard("de")
        rows = keyboard.inline_keyboard

        # 5 language rows + 1 auto + 1 skip = 7
        assert len(rows) == 7

        # First 5 rows have 4 buttons each
        for i in range(5):
            assert len(rows[i]) == 4, f"Row {i} should have 4 buttons"

        # Auto-detect row has 1 button
        assert len(rows[5]) == 1
        assert "wizard_lang_auto" in rows[5][0].callback_data

        # Skip row has 1 button
        assert len(rows[6]) == 1
        assert "wizard_skip" in rows[6][0].callback_data

    def test_step1_keyboard_structure_en(self) -> None:
        """Step 1 Keyboard in EN hat gleiche Struktur."""
        from presentation.onboarding_callbacks import build_language_keyboard

        keyboard = build_language_keyboard("en")
        rows = keyboard.inline_keyboard
        assert len(rows) == 7

    def test_step1_first_row_correct_order(self) -> None:
        """Erste Reihe: Deutsch, English, Français, Español."""
        from presentation.onboarding_callbacks import build_language_keyboard

        keyboard = build_language_keyboard("de")
        first_row = keyboard.inline_keyboard[0]

        assert first_row[0].text == "Deutsch"
        assert first_row[1].text == "English"
        assert first_row[2].text == "Français"
        assert first_row[3].text == "Español"

    def test_step2_keyboard_has_one_button(self) -> None:
        """Step 2 Keyboard hat genau einen Button."""
        from presentation.onboarding_callbacks import build_completion_keyboard

        keyboard = build_completion_keyboard("de")
        assert len(keyboard.inline_keyboard) == 1
        assert len(keyboard.inline_keyboard[0]) == 1
        assert "wizard_done" in keyboard.inline_keyboard[0][0].callback_data

    def test_step2_text_de_snapshot(self) -> None:
        """Step 2 DE enthält erwartete Elemente."""
        from domain.onboarding import get_step2_text

        text = get_step2_text("de", "Deutsch")
        assert "startklar" in text
        assert "Deutsch" in text
        assert "/settings" in text
        assert "/help" in text

    def test_step2_text_en_snapshot(self) -> None:
        """Step 2 EN enthält erwartete Elemente."""
        from domain.onboarding import get_step2_text

        text = get_step2_text("en", "English")
        assert "all set" in text
        assert "English" in text
        assert "/settings" in text
        assert "/help" in text
