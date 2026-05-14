"""Deterministic UI-Tests fuer Telegram-UI-Ausgaben.

Inline-Vergleiche statt syrupy-Snapshots (robuster, keine externe Fixture-Dependency).
Bei jeder UI-Aenderung schlaegt der Test fehl. Entwickler prueft ob neue Werte OK sind.

Getestete Szenarien:
  1. /settings Hauptmenü ohne Overrides
  2. /settings Hauptmenü mit Global-Override
  3. /settings Hauptmenü mit Slot-Overrides (zeigt "Alle zurücksetzen")
  4. /models Ausgabe ohne Overrides
  5. /models Ausgabe mit verschiedenen Overrides
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.model_service import ModelService
from application.task_router import SlotConfig, TaskRouter
from domain.task_slot import TaskSlot
from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporaerer DB-Pfad fuer Test-Isolation."""
    return tmp_path / "test_snapshot.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    """Frische SQLite-Connection fuer jeden Test."""
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def model_service(conn: SqliteConnection) -> ModelService:
    """ModelService mit echtem SQLite-Backend."""
    return ModelService(storage=SqliteModelStorage(conn))


@pytest.fixture
def slot_configs() -> list[SlotConfig]:
    """Standard SlotConfigs fuer Tests."""
    return [
        SlotConfig(slot=TaskSlot.CHAT, default_model="sonnet", fallback=True),
        SlotConfig(slot=TaskSlot.CODE, default_model="opus"),
        SlotConfig(slot=TaskSlot.REASON, default_model="opus"),
        SlotConfig(slot=TaskSlot.CREATIVE, default_model="sonnet"),
        SlotConfig(slot=TaskSlot.QUICK, default_model="haiku"),
        SlotConfig(slot=TaskSlot.RESEARCH, default_model="opus"),
    ]


@pytest.fixture
def task_router(
    slot_configs: list[SlotConfig], model_service: ModelService
) -> TaskRouter:
    """TaskRouter mit Standard-Konfiguration."""
    return TaskRouter(slot_configs, model_service=model_service)


def _make_context(
    model_service: ModelService | None = None,
    task_router: object | None = None,
) -> MagicMock:
    """Erstellt einen gemockten Telegram-Context."""
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


def _serialize_keyboard(keyboard) -> list[list[dict[str, str]]]:
    """Serialisiert InlineKeyboardMarkup in eine vergleichbare Struktur."""
    result = []
    for row in keyboard.inline_keyboard:
        row_data = []
        for btn in row:
            row_data.append(
                {
                    "text": btn.text,
                    "callback_data": btn.callback_data or "",
                }
            )
        result.append(row_data)
    return result


def _make_command_update(user_id: int = 1, chat_id: int = 10) -> MagicMock:
    """Erstellt ein gemocktes Telegram-Update fuer Commands."""
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


# ----------------------------------------------------------------
# Settings Menu Tests (deterministic inline assertions)
# ----------------------------------------------------------------


class TestSettingsMenuSnapshots:
    """Deterministic-Tests fuer /settings Inline-Keyboard Menu."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    def test_settings_main_menu_no_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
    ) -> None:
        """Settings-Menu ohne Overrides hat korrekte Slot-Labels."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        context = _make_context(model_service=model_service, task_router=task_router)

        text, keyboard = build_main_menu_keyboard(
            user_id=1, model_service=model_service, context=context, lang="de"
        )

        kb = _serialize_keyboard(keyboard)

        # Text assertions
        assert "Einstellungen" in text
        assert "Modelle pro Slot" in text

        # Keyboard: 6 Slot-Buttons + 1 Sprache-Button = 7 Reihen
        assert len(kb) == 7

        # Slot-Labels muessen Default anzeigen
        assert kb[0][0]["text"] == "CHAT: Sonnet 4.6 (Default)"
        assert kb[1][0]["text"] == "CODE: Opus 4.7 (Default)"
        assert kb[2][0]["text"] == "REASON: Opus 4.7 (Default)"
        assert kb[3][0]["text"] == "CREATIVE: Sonnet 4.6 (Default)"
        assert kb[4][0]["text"] == "QUICK: Haiku 4.5 (Default)"
        assert kb[5][0]["text"] == "RESEARCH: Opus 4.7 (Default)"
        assert "Sprache" in kb[6][0]["text"]

        # Callback-Data korrekt
        assert kb[0][0]["callback_data"] == "settings_slot:chat"
        assert kb[1][0]["callback_data"] == "settings_slot:code"

    def test_settings_main_menu_with_global_override(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
    ) -> None:
        """Settings-Menu mit globalem Override (Opus) zeigt Override-Banner."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        model_service.set_user_model(1, "opus")

        context = _make_context(model_service=model_service, task_router=task_router)

        text, keyboard = build_main_menu_keyboard(
            user_id=1, model_service=model_service, context=context, lang="de"
        )

        kb = _serialize_keyboard(keyboard)

        # Override-Banner im Text
        assert "Globaler Override aktiv" in text
        assert "Opus 4.7" in text

        # 8 Reihen: Reset-Button + 6 Slots + Sprache
        assert len(kb) == 8
        assert "Override aufheben" in kb[0][0]["text"]

        # Alle Slots zeigen "(global)"
        for row in kb[1:7]:
            assert "(global)" in row[0]["text"]
            assert "Opus 4.7" in row[0]["text"]

    def test_settings_main_menu_with_slot_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
    ) -> None:
        """Settings-Menu mit Slot-Overrides zeigt 'Alle zurücksetzen'."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        model_service.set_user_model(1, "haiku", slot="code")
        model_service.set_user_model(1, "opus", slot="chat")

        context = _make_context(model_service=model_service, task_router=task_router)

        text, keyboard = build_main_menu_keyboard(
            user_id=1, model_service=model_service, context=context, lang="de"
        )

        kb = _serialize_keyboard(keyboard)

        # 8 Reihen: 6 Slots + Sprache + "Alle zurücksetzen"
        assert len(kb) == 8

        # Overridden Slots zeigen Modell ohne "(Default)"
        assert kb[0][0]["text"] == "CHAT: Opus 4.7"
        assert kb[1][0]["text"] == "CODE: Haiku 4.5"

        # Non-overridden Slots zeigen "(Default)"
        assert "(Default)" in kb[2][0]["text"]

        # Letzter Button: "Alle zurücksetzen"
        last_row = kb[-1][0]
        assert "zurücksetzen" in last_row["text"]
        assert last_row["callback_data"] == "settings_reset_all"


# ----------------------------------------------------------------
# Models Command Tests
# ----------------------------------------------------------------


class TestModelsCommandSnapshots:
    """Deterministic-Tests fuer /models Ausgabe."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_models_output_no_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
    ) -> None:
        """Models-Output ohne Overrides zeigt alle 6 Slots mit Default."""
        from presentation.handlers import handle_models_command

        update = _make_command_update()
        context = _make_context(model_service=model_service, task_router=task_router)

        await handle_models_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "CHAT: Sonnet 4.6 (Default)" in reply_text
        assert "CODE: Opus 4.7 (Default)" in reply_text
        assert "REASON: Opus 4.7 (Default)" in reply_text
        assert "CREATIVE: Sonnet 4.6 (Default)" in reply_text
        assert "QUICK: Haiku 4.5 (Default)" in reply_text
        assert "RESEARCH: Opus 4.7 (Default)" in reply_text
        assert "/setmodel" in reply_text

    async def test_models_output_with_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
    ) -> None:
        """Models-Output mit Overrides zeigt Override-Markierung."""
        from presentation.handlers import handle_models_command

        model_service.set_user_model(1, "opus")
        model_service.set_user_model(1, "haiku", slot="code")

        update = _make_command_update()
        context = _make_context(model_service=model_service, task_router=task_router)

        await handle_models_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        # Global Override auf Opus: alle Slots zeigen Opus (Override)
        assert "CHAT: Opus 4.7 (Override)" in reply_text
        # CODE hat spezifischen Override auf Haiku
        assert "CODE: Haiku 4.5 (Override)" in reply_text
        # Nicht-overridden Slots zeigen Global-Override
        assert "REASON: Opus 4.7 (Override)" in reply_text
