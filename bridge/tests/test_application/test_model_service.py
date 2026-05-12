"""Tests fuer ModelService: Alias-Resolution, CRUD, Default-Fallback.

Testet:
  - resolve_alias: Alias -> Modell-ID Mapping
  - ModelService.set_user_model: Alias setzen, unbekannter Alias
  - ModelService.get_effective_model: Override vs. Default
  - ModelService.reset_user_model: Reset auf Default
  - ModelService.get_model_display_name: Reverse-Lookup
  - ModelService.list_available_aliases: Alle Aliase
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.model_service import (
    DEFAULT_MODEL,
    ModelService,
    resolve_alias,
)
from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporaerer DB-Pfad fuer Test-Isolation."""
    return tmp_path / "test_model.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    """Frische SQLite-Connection fuer jeden Test."""
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def storage(conn: SqliteConnection) -> SqliteModelStorage:
    """Model-Storage-Instanz."""
    return SqliteModelStorage(conn)


@pytest.fixture
def service(storage: SqliteModelStorage) -> ModelService:
    """ModelService-Instanz mit SQLite-Backend."""
    return ModelService(storage=storage)


# ──────────────────────────────────────────────────────────────
# resolve_alias Tests
# ──────────────────────────────────────────────────────────────


class TestResolveAlias:
    """Tests fuer die Alias-Resolution-Funktion."""

    def test_opus_alias(self) -> None:
        """'opus' wird korrekt aufgeloest."""
        result = resolve_alias("opus")
        assert result == "claude-opus-4-7"

    def test_sonnet_alias(self) -> None:
        """'sonnet' wird korrekt aufgeloest."""
        result = resolve_alias("sonnet")
        assert result == "claude-sonnet-4-6"

    def test_haiku_alias(self) -> None:
        """'haiku' wird korrekt aufgeloest."""
        result = resolve_alias("haiku")
        assert result == "claude-haiku-4-5-20251001"

    def test_case_insensitive(self) -> None:
        """Aliase sind case-insensitive."""
        assert resolve_alias("Opus") == resolve_alias("opus")
        assert resolve_alias("SONNET") == resolve_alias("sonnet")
        assert resolve_alias("Haiku") == resolve_alias("haiku")

    def test_full_model_id_accepted(self) -> None:
        """Volle Modell-IDs werden direkt akzeptiert."""
        full_id = "claude-opus-4-7"
        assert resolve_alias(full_id) == full_id

    def test_unknown_returns_none(self) -> None:
        """Unbekannter Alias gibt None zurueck."""
        assert resolve_alias("gpt-4") is None
        assert resolve_alias("nonexistent") is None
        assert resolve_alias("") is None

    def test_whitespace_stripped(self) -> None:
        """Whitespace wird entfernt."""
        assert resolve_alias("  opus  ") == resolve_alias("opus")


# ──────────────────────────────────────────────────────────────
# ModelService CRUD Tests
# ──────────────────────────────────────────────────────────────


class TestModelServiceSet:
    """Tests fuer set_user_model."""

    def test_set_via_alias(self, service: ModelService) -> None:
        """Modell per Alias setzen funktioniert."""
        success, result = service.set_user_model(user_id=1, alias_or_id="opus")
        assert success is True
        assert result == "claude-opus-4-7"

    def test_set_persists(self, service: ModelService) -> None:
        """Gesetztes Modell wird persistent gespeichert."""
        service.set_user_model(user_id=1, alias_or_id="haiku")
        model = service.get_user_model(user_id=1)
        assert model == "claude-haiku-4-5-20251001"

    def test_set_unknown_fails(self, service: ModelService) -> None:
        """Unbekanntes Modell wird abgelehnt."""
        success, error_msg = service.set_user_model(user_id=1, alias_or_id="gpt-4")
        assert success is False
        assert "gpt-4" in error_msg

    def test_set_overwrites_previous(self, service: ModelService) -> None:
        """Neues Modell ueberschreibt vorheriges."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        service.set_user_model(user_id=1, alias_or_id="haiku")
        model = service.get_user_model(user_id=1)
        assert model == "claude-haiku-4-5-20251001"


class TestModelServiceGet:
    """Tests fuer get_user_model und get_effective_model."""

    def test_no_override_returns_none(self, service: ModelService) -> None:
        """Kein Override gibt None zurueck."""
        assert service.get_user_model(user_id=1) is None

    def test_effective_model_no_override(self, service: ModelService) -> None:
        """Ohne Override wird DEFAULT_MODEL zurueckgegeben."""
        effective = service.get_effective_model(user_id=1)
        assert effective == DEFAULT_MODEL

    def test_effective_model_with_override(self, service: ModelService) -> None:
        """Mit Override wird das Override-Modell zurueckgegeben."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        effective = service.get_effective_model(user_id=1)
        assert effective == "claude-opus-4-7"

    def test_user_isolation(self, service: ModelService) -> None:
        """Override eines Users beeinflusst andere User nicht."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        assert service.get_user_model(user_id=2) is None
        assert service.get_effective_model(user_id=2) == DEFAULT_MODEL


class TestModelServiceReset:
    """Tests fuer reset_user_model."""

    def test_reset_removes_override(self, service: ModelService) -> None:
        """Reset entfernt das Override."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        deleted = service.reset_user_model(user_id=1)
        assert deleted is True
        assert service.get_user_model(user_id=1) is None

    def test_reset_no_override_returns_false(self, service: ModelService) -> None:
        """Reset ohne vorheriges Override gibt False zurueck."""
        deleted = service.reset_user_model(user_id=1)
        assert deleted is False

    def test_effective_after_reset(self, service: ModelService) -> None:
        """Nach Reset ist effective_model wieder der Default."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        service.reset_user_model(user_id=1)
        assert service.get_effective_model(user_id=1) == DEFAULT_MODEL


# ──────────────────────────────────────────────────────────────
# Display + Utility Tests
# ──────────────────────────────────────────────────────────────


class TestModelServiceUtilities:
    """Tests fuer Hilfsfunktionen."""

    def test_display_name_for_alias(self) -> None:
        """Display-Name fuer bekannte Modelle."""
        assert ModelService.get_model_display_name("claude-opus-4-7") == "Opus 4.7"
        assert ModelService.get_model_display_name("claude-sonnet-4-6") == "Sonnet 4.6"
        assert (
            ModelService.get_model_display_name("claude-haiku-4-5-20251001")
            == "Haiku 4.5"
        )

    def test_display_name_unknown_returns_id(self) -> None:
        """Unbekannte Modell-ID wird als Display-Name zurueckgegeben."""
        assert (
            ModelService.get_model_display_name("some-unknown-model")
            == "some-unknown-model"
        )

    def test_list_available_aliases(self) -> None:
        """list_available_aliases gibt nur Anthropic-Aliase zurück (Phase 1)."""
        aliases = ModelService.list_available_aliases()
        assert "opus" in aliases
        assert "sonnet" in aliases
        assert "haiku" in aliases
        # Phase 1: nur Anthropic, nicht die volle MODEL_ALIASES-Liste
        assert len(aliases) == 3


# ──────────────────────────────────────────────────────────────
# SqliteModelStorage Direct Tests
# ──────────────────────────────────────────────────────────────


class TestProviderFilter:
    """Tests für den Provider-Filter in set_user_model (V8-R2 Finding 3).

    Nicht-Anthropic-Modelle dürfen im aktuellen Claude-Hauptpfad nicht
    akzeptiert werden. /setmodel gpt55, gemini, llama etc. müssen abgelehnt
    werden mit klarer Fehlermeldung.
    """

    def test_setmodel_gpt55_rejected(self, service: ModelService) -> None:
        """gpt55 (OpenAI) wird abgelehnt: falscher Provider."""
        success, msg = service.set_user_model(user_id=1, alias_or_id="gpt55")
        assert success is False
        assert "openai" in msg.lower()
        assert "anthropic" in msg.lower() or "claude" in msg.lower()

    def test_setmodel_gemini_rejected(self, service: ModelService) -> None:
        """gemini (Google) wird abgelehnt: falscher Provider."""
        success, msg = service.set_user_model(user_id=1, alias_or_id="gemini")
        assert success is False
        assert "google" in msg.lower()

    def test_setmodel_llama_rejected(self, service: ModelService) -> None:
        """llama (Ollama) wird abgelehnt: falscher Provider."""
        success, msg = service.set_user_model(user_id=1, alias_or_id="llama")
        assert success is False
        assert "ollama" in msg.lower()

    def test_setmodel_opus_accepted(self, service: ModelService) -> None:
        """opus (Anthropic) wird akzeptiert."""
        success, result = service.set_user_model(user_id=1, alias_or_id="opus")
        assert success is True
        assert result == "claude-opus-4-7"

    def test_setmodel_sonnet_accepted(self, service: ModelService) -> None:
        """sonnet (Anthropic) wird akzeptiert."""
        success, result = service.set_user_model(user_id=1, alias_or_id="sonnet")
        assert success is True
        assert result == "claude-sonnet-4-6"

    def test_setmodel_haiku_accepted(self, service: ModelService) -> None:
        """haiku (Anthropic) wird akzeptiert."""
        success, result = service.set_user_model(user_id=1, alias_or_id="haiku")
        assert success is True
        assert "haiku" in result

    def test_list_available_aliases_only_anthropic(self) -> None:
        """list_available_aliases gibt nur Anthropic-Modelle zurück."""
        aliases = ModelService.list_available_aliases()
        # Anthropic-Aliase müssen drin sein
        assert "opus" in aliases
        assert "sonnet" in aliases
        assert "haiku" in aliases
        # Nicht-Anthropic-Aliase dürfen NICHT drin sein
        assert "gpt55" not in aliases
        assert "gemini" not in aliases
        assert "llama" not in aliases
        assert "kimi" not in aliases
        assert "qwen" not in aliases

    def test_rejected_model_not_persisted(self, service: ModelService) -> None:
        """Abgelehntes Modell darf nicht gespeichert werden."""
        service.set_user_model(user_id=1, alias_or_id="gpt55")
        assert service.get_user_model(user_id=1) is None


# ──────────────────────────────────────────────────────────────
# SqliteModelStorage Direct Tests
# ──────────────────────────────────────────────────────────────


class TestSqliteModelStorage:
    """Direkte Tests fuer den SQLite-Adapter."""

    def test_set_and_get(self, storage: SqliteModelStorage) -> None:
        """set_model + get_model roundtrip."""
        storage.set_model(user_id=1, model_id="claude-opus-4-7")
        result = storage.get_model(user_id=1)
        assert result == "claude-opus-4-7"

    def test_get_nonexistent(self, storage: SqliteModelStorage) -> None:
        """get_model fuer nicht-existierenden User gibt None zurueck."""
        assert storage.get_model(user_id=999) is None

    def test_delete(self, storage: SqliteModelStorage) -> None:
        """delete_model entfernt den Override."""
        storage.set_model(user_id=1, model_id="claude-opus-4-7")
        assert storage.delete_model(user_id=1) is True
        assert storage.get_model(user_id=1) is None

    def test_delete_nonexistent(self, storage: SqliteModelStorage) -> None:
        """delete_model fuer nicht-existierenden Override gibt False zurueck."""
        assert storage.delete_model(user_id=999) is False

    def test_upsert(self, storage: SqliteModelStorage) -> None:
        """Zweites set_model ueberschreibt das erste."""
        storage.set_model(user_id=1, model_id="model-a")
        storage.set_model(user_id=1, model_id="model-b")
        assert storage.get_model(user_id=1) == "model-b"

    def test_slot_isolation(self, storage: SqliteModelStorage) -> None:
        """Verschiedene Slots sind isoliert (Vorbereitung Phase 2+)."""
        storage.set_model(user_id=1, model_id="model-a", slot="global")
        storage.set_model(user_id=1, model_id="model-b", slot="chat")
        assert storage.get_model(user_id=1, slot="global") == "model-a"
        assert storage.get_model(user_id=1, slot="chat") == "model-b"

    def test_user_isolation(self, storage: SqliteModelStorage) -> None:
        """Verschiedene User sind isoliert."""
        storage.set_model(user_id=1, model_id="model-a")
        storage.set_model(user_id=2, model_id="model-b")
        assert storage.get_model(user_id=1) == "model-a"
        assert storage.get_model(user_id=2) == "model-b"
