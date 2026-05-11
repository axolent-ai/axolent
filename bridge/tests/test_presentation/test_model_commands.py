"""Tests fuer /setmodel reset + /resetmodel Handler.

Testet:
  - /setmodel reset zeigt Default-Modell-Name in der Antwort
  - /resetmodel funktioniert als eigenstaendiger Command
  - /resetmodel ohne Override zeigt Default-Modell-Name
  - /resetmodel mit Override entfernt ihn und zeigt Default
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.model_service import DEFAULT_MODEL, ModelService
from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporaerer DB-Pfad fuer Test-Isolation."""
    return tmp_path / "test_model_cmd.db"


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


def _make_update(user_id: int = 1, chat_id: int = 10) -> MagicMock:
    """Erstellt ein gemocktes Telegram-Update."""
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
    args: list[str] | None = None,
    model_service: ModelService | None = None,
) -> MagicMock:
    """Erstellt einen gemockten Telegram-Context mit model_service."""
    mock_chat_service = MagicMock()
    mock_chat_service.get_chat_language = AsyncMock(return_value="de")

    context = MagicMock()
    context.args = args or []
    context.bot = MagicMock()
    context.application = MagicMock()
    context.application.bot_data = {
        "chat_service": mock_chat_service,
        "system_prompt": "test",
        "memory_service": None,
        "persistent_provider": None,
        "process_pool": None,
        "rate_limiter": None,
        "bookmark_service": None,
    }
    if model_service is not None:
        context.application.bot_data["model_service"] = model_service
    return context


# ──────────────────────────────────────────────────────────────
# /setmodel reset: Default-Modell-Name in Antwort
# ──────────────────────────────────────────────────────────────


class TestSetmodelResetShowsDefault:
    """Tests dass /setmodel reset den Default-Modell-Name anzeigt."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_reset_with_override_shows_default_name(
        self, model_service: ModelService
    ) -> None:
        """/setmodel reset mit aktivem Override zeigt Default-Modell."""
        from presentation.handlers import handle_setmodel_command

        # Override setzen
        model_service.set_user_model(user_id=1, alias_or_id="opus")

        update = _make_update()
        context = _make_context(args=["reset"], model_service=model_service)

        await handle_setmodel_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        # Muss den Display-Name des Defaults enthalten
        default_display = ModelService.get_model_display_name(DEFAULT_MODEL)
        assert default_display in reply_text
        assert DEFAULT_MODEL in reply_text

    async def test_reset_without_override_shows_default_name(
        self, model_service: ModelService
    ) -> None:
        """/setmodel reset ohne Override zeigt Default-Modell-Name."""
        from presentation.handlers import handle_setmodel_command

        update = _make_update()
        context = _make_context(args=["reset"], model_service=model_service)

        await handle_setmodel_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        default_display = ModelService.get_model_display_name(DEFAULT_MODEL)
        assert default_display in reply_text
        assert DEFAULT_MODEL in reply_text
        assert "Default" in reply_text or "default" in reply_text


# ──────────────────────────────────────────────────────────────
# /resetmodel: Eigenstaendiger Command
# ──────────────────────────────────────────────────────────────


class TestResetmodelCommand:
    """/resetmodel als eigenstaendiger Shortcut fuer /setmodel reset."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_resetmodel_removes_override(
        self, model_service: ModelService
    ) -> None:
        """/resetmodel entfernt ein aktives Override."""
        from presentation.handlers import handle_resetmodel_command

        # Override setzen
        model_service.set_user_model(user_id=1, alias_or_id="opus")
        assert model_service.get_user_model(user_id=1) is not None

        update = _make_update()
        context = _make_context(model_service=model_service)

        await handle_resetmodel_command(update, context)

        # Override muss entfernt sein
        assert model_service.get_user_model(user_id=1) is None

        # Antwort zeigt Default
        reply_text = update.message.reply_text.call_args[0][0]
        default_display = ModelService.get_model_display_name(DEFAULT_MODEL)
        assert default_display in reply_text
        assert "zurückgesetzt" in reply_text.lower() or "reset" in reply_text.lower()

    async def test_resetmodel_without_override_shows_default(
        self, model_service: ModelService
    ) -> None:
        """/resetmodel ohne Override zeigt Default-Name."""
        from presentation.handlers import handle_resetmodel_command

        update = _make_update()
        context = _make_context(model_service=model_service)

        await handle_resetmodel_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        default_display = ModelService.get_model_display_name(DEFAULT_MODEL)
        assert default_display in reply_text
        assert DEFAULT_MODEL in reply_text

    async def test_resetmodel_no_model_service(self) -> None:
        """/resetmodel ohne ModelService gibt Fehlermeldung."""
        from presentation.handlers import handle_resetmodel_command

        update = _make_update()
        context = _make_context(model_service=None)

        await handle_resetmodel_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "nicht initialisiert" in reply_text.lower()

    async def test_resetmodel_english_locale(self, model_service: ModelService) -> None:
        """/resetmodel mit englischer Sprache zeigt englische Meldung."""
        from presentation.handlers import handle_resetmodel_command

        # Override setzen
        model_service.set_user_model(user_id=1, alias_or_id="haiku")

        update = _make_update()
        context = _make_context(model_service=model_service)
        # Sprache auf Englisch umstellen
        context.application.bot_data["chat_service"].get_chat_language = AsyncMock(
            return_value="en"
        )

        await handle_resetmodel_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "reset" in reply_text.lower() or "default" in reply_text.lower()
        default_display = ModelService.get_model_display_name(DEFAULT_MODEL)
        assert default_display in reply_text
