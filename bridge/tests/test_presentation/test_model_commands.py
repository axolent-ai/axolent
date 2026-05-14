"""Tests for /setmodel reset + /resetmodel handler.

Tests:
  - /setmodel reset shows default model name in the response
  - /resetmodel works as a standalone command
  - /resetmodel without override shows default model name
  - /resetmodel with override removes it and shows default
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
    """Temporärer DB-Pfad für Test-Isolation."""
    return tmp_path / "test_model_cmd.db"


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
        """/setmodel reset mit aktivem Override zeigt Erfolg."""
        from presentation.handlers import handle_setmodel_command

        # Override setzen
        model_service.set_user_model(user_id=1, alias_or_id="opus")

        update = _make_update()
        context = _make_context(args=["reset"], model_service=model_service)

        await handle_setmodel_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        # Phase 2a: /setmodel reset entfernt alle Overrides
        assert "zurückgesetzt" in reply_text.lower() or "entfernt" in reply_text.lower()

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
# /resetmodel: Eigenständiger Command
# ──────────────────────────────────────────────────────────────


class TestResetmodelCommand:
    """/resetmodel als eigenständiger Shortcut für /setmodel reset."""

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
        assert "not initialized" in reply_text.lower()

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


# ──────────────────────────────────────────────────────────────
# /setmodel <alias>: Happy-Path und Error-Path
# ──────────────────────────────────────────────────────────────


class TestSetmodelHappyAndErrorPath:
    """Tests für /setmodel mit gültigem und ungültigem Alias."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_setmodel_opus_happy_path(self, model_service: ModelService) -> None:
        """/setmodel opus setzt Modell und bestaetigt mit Display-Name."""
        from presentation.handlers import handle_setmodel_command

        update = _make_update()
        context = _make_context(args=["opus"], model_service=model_service)

        await handle_setmodel_command(update, context)

        # Modell muss gesetzt sein
        assert model_service.get_user_model(user_id=1) == "claude-opus-4-7"

        # Antwort muss den Display-Name enthalten
        reply_text = update.message.reply_text.call_args[0][0]
        assert "Opus 4.7" in reply_text
        assert "claude-opus-4-7" in reply_text

    async def test_setmodel_invalid_model_error(
        self, model_service: ModelService
    ) -> None:
        """/setmodel ungültigeswort zeigt Fehlermeldung mit verfügbaren Aliassen."""
        from presentation.handlers import handle_setmodel_command

        update = _make_update()
        context = _make_context(args=["ungültigeswort"], model_service=model_service)

        await handle_setmodel_command(update, context)

        # Modell darf NICHT gesetzt sein
        assert model_service.get_user_model(user_id=1) is None

        # Antwort muss Fehlermeldung mit verfügbaren Aliassen enthalten
        reply_text = update.message.reply_text.call_args[0][0]
        assert "ungültigeswort" in reply_text.lower()
        assert "opus" in reply_text or "sonnet" in reply_text


# ──────────────────────────────────────────────────────────────
# /models: Zeigt aktives Modell und verfügbare Optionen
# ──────────────────────────────────────────────────────────────


class TestModelsCommand:
    """/models zeigt aktuelles Modell und Optionen."""

    @pytest.fixture(autouse=True)
    def _allow_all(self) -> None:
        """Whitelist-Bypass."""
        with patch("presentation.decorators.ALLOW_ALL_USERS", True):
            yield  # type: ignore[misc]

    async def test_models_shows_default_and_options(
        self, model_service: ModelService
    ) -> None:
        """/models ohne Override zeigt pro-Slot Belegung."""
        from presentation.handlers import handle_models_command

        update = _make_update()
        context = _make_context(model_service=model_service)

        await handle_models_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        # Phase 2a: zeigt pro-Slot Belegung
        assert "CHAT" in reply_text
        assert "CODE" in reply_text
        assert "REASON" in reply_text
        assert "CREATIVE" in reply_text
        assert "QUICK" in reply_text
        assert "RESEARCH" in reply_text
        assert "Sonnet" in reply_text  # Default fuer CHAT
        assert "Default" in reply_text

    async def test_models_shows_active_override(
        self, model_service: ModelService
    ) -> None:
        """/models mit Override zeigt das aktive Override-Modell pro Slot."""
        from presentation.handlers import handle_models_command

        model_service.set_user_model(user_id=1, alias_or_id="opus")

        update = _make_update()
        context = _make_context(model_service=model_service)

        await handle_models_command(update, context)

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Opus 4.7" in reply_text
        assert "Override" in reply_text
