"""Snapshot-Tests für Telegram-UI-Ausgaben.

Verwendet syrupy für deterministische Snapshot-Vergleiche.
Bei jeder UI-Änderung schlägt der Test fehl. Entwickler entscheidet
ob neuer Snapshot OK ist via: pytest --snapshot-update

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


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporärer DB-Pfad für Test-Isolation."""
    return tmp_path / "test_snapshot.db"


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


@pytest.fixture
def slot_configs() -> list[SlotConfig]:
    """Standard SlotConfigs für Tests."""
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


# ──────────────────────────────────────────────────────────────
# Snapshot Tests
# ──────────────────────────────────────────────────────────────


class TestSettingsMenuSnapshots:
    """Snapshot-Tests für /settings Inline-Keyboard Menü."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    def test_settings_main_menu_no_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
        snapshot,
    ) -> None:
        """Snapshot: /settings Hauptmenü ohne Overrides."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        context = _make_context(model_service=model_service, task_router=task_router)

        text, keyboard = build_main_menu_keyboard(
            user_id=1, model_service=model_service, context=context, lang="de"
        )

        snapshot_data = {
            "text": text,
            "keyboard": _serialize_keyboard(keyboard),
        }
        assert snapshot_data == snapshot

    def test_settings_main_menu_with_global_override(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
        snapshot,
    ) -> None:
        """Snapshot: /settings Hauptmenü mit globalem Override (Opus)."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        model_service.set_user_model(1, "opus")

        context = _make_context(model_service=model_service, task_router=task_router)

        text, keyboard = build_main_menu_keyboard(
            user_id=1, model_service=model_service, context=context, lang="de"
        )

        snapshot_data = {
            "text": text,
            "keyboard": _serialize_keyboard(keyboard),
        }
        assert snapshot_data == snapshot

    def test_settings_main_menu_with_slot_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
        snapshot,
    ) -> None:
        """Snapshot: /settings Hauptmenü mit Slot-Overrides (zeigt 'Alle zurücksetzen')."""
        from presentation.settings_callbacks import build_main_menu_keyboard

        model_service.set_user_model(1, "haiku", slot="code")
        model_service.set_user_model(1, "opus", slot="chat")

        context = _make_context(model_service=model_service, task_router=task_router)

        text, keyboard = build_main_menu_keyboard(
            user_id=1, model_service=model_service, context=context, lang="de"
        )

        snapshot_data = {
            "text": text,
            "keyboard": _serialize_keyboard(keyboard),
        }
        assert snapshot_data == snapshot


class TestModelsCommandSnapshots:
    """Snapshot-Tests für /models Ausgabe."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_models_output_no_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
        snapshot,
    ) -> None:
        """Snapshot: /models ohne Overrides."""
        from presentation.handlers import handle_models_command

        update = _make_command_update()
        context = _make_context(model_service=model_service, task_router=task_router)

        await handle_models_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert reply_text == snapshot

    async def test_models_output_with_overrides(
        self,
        model_service: ModelService,
        task_router: TaskRouter,
        snapshot,
    ) -> None:
        """Snapshot: /models mit Global-Override (Opus) + Slot-Override (CODE=Haiku)."""
        from presentation.handlers import handle_models_command

        model_service.set_user_model(1, "opus")
        model_service.set_user_model(1, "haiku", slot="code")

        update = _make_command_update()
        context = _make_context(model_service=model_service, task_router=task_router)

        await handle_models_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert reply_text == snapshot
