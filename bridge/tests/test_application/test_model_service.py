"""Tests für ModelService: Alias-Resolution, CRUD, Default-Fallback.

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
    """Temporärer DB-Pfad für Test-Isolation."""
    return tmp_path / "test_model.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    """Frische SQLite-Connection für jeden Test."""
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
    """Tests für die Alias-Resolution-Funktion."""

    def test_opus_alias(self) -> None:
        """'opus' wird korrekt aufgelöst."""
        result = resolve_alias("opus")
        assert result == "claude-opus-4-7"

    def test_sonnet_alias(self) -> None:
        """'sonnet' wird korrekt aufgelöst."""
        result = resolve_alias("sonnet")
        assert result == "claude-sonnet-4-6"

    def test_haiku_alias(self) -> None:
        """'haiku' wird korrekt aufgelöst."""
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
        """Unbekannter Alias gibt None zurück."""
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
    """Tests für set_user_model."""

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
        """Neues Modell überschreibt vorheriges."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        service.set_user_model(user_id=1, alias_or_id="haiku")
        model = service.get_user_model(user_id=1)
        assert model == "claude-haiku-4-5-20251001"


class TestModelServiceGet:
    """Tests für get_user_model und get_effective_model."""

    def test_no_override_returns_none(self, service: ModelService) -> None:
        """Kein Override gibt None zurück."""
        assert service.get_user_model(user_id=1) is None

    def test_effective_model_no_override(self, service: ModelService) -> None:
        """Ohne Override wird DEFAULT_MODEL zurückgegeben."""
        effective = service.get_effective_model(user_id=1)
        assert effective == DEFAULT_MODEL

    def test_effective_model_with_override(self, service: ModelService) -> None:
        """Mit Override wird das Override-Modell zurückgegeben."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        effective = service.get_effective_model(user_id=1)
        assert effective == "claude-opus-4-7"

    def test_user_isolation(self, service: ModelService) -> None:
        """Override eines Users beeinflusst andere User nicht."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        assert service.get_user_model(user_id=2) is None
        assert service.get_effective_model(user_id=2) == DEFAULT_MODEL


class TestModelServiceReset:
    """Tests für reset_user_model."""

    def test_reset_removes_override(self, service: ModelService) -> None:
        """Reset entfernt das Override."""
        service.set_user_model(user_id=1, alias_or_id="opus")
        deleted = service.reset_user_model(user_id=1)
        assert deleted is True
        assert service.get_user_model(user_id=1) is None

    def test_reset_no_override_returns_false(self, service: ModelService) -> None:
        """Reset ohne vorheriges Override gibt False zurück."""
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
    """Tests für Hilfsfunktionen."""

    def test_display_name_for_alias(self) -> None:
        """Display-Name für bekannte Modelle."""
        assert ModelService.get_model_display_name("claude-opus-4-7") == "Opus 4.7"
        assert ModelService.get_model_display_name("claude-sonnet-4-6") == "Sonnet 4.6"
        assert (
            ModelService.get_model_display_name("claude-haiku-4-5-20251001")
            == "Haiku 4.5"
        )

    def test_display_name_unknown_returns_id(self) -> None:
        """Unbekannte Modell-ID wird als Display-Name zurückgegeben."""
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
    """Direkte Tests für den SQLite-Adapter."""

    def test_set_and_get(self, storage: SqliteModelStorage) -> None:
        """set_model + get_model roundtrip."""
        storage.set_model(user_id=1, model_id="claude-opus-4-7")
        result = storage.get_model(user_id=1)
        assert result == "claude-opus-4-7"

    def test_get_nonexistent(self, storage: SqliteModelStorage) -> None:
        """get_model für nicht-existierenden User gibt None zurück."""
        assert storage.get_model(user_id=999) is None

    def test_delete(self, storage: SqliteModelStorage) -> None:
        """delete_model entfernt den Override."""
        storage.set_model(user_id=1, model_id="claude-opus-4-7")
        assert storage.delete_model(user_id=1) is True
        assert storage.get_model(user_id=1) is None

    def test_delete_nonexistent(self, storage: SqliteModelStorage) -> None:
        """delete_model für nicht-existierenden Override gibt False zurück."""
        assert storage.delete_model(user_id=999) is False

    def test_upsert(self, storage: SqliteModelStorage) -> None:
        """Zweites set_model überschreibt das erste."""
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


# ──────────────────────────────────────────────────────────────
# Stale Storage Revalidation Tests (V8-R3 Finding 2)
# ──────────────────────────────────────────────────────────────


class TestStaleStorageRevalidation:
    """V8-R3 Finding 2: Stale Werte von Nicht-Anthropic-Modellen
    die vor dem Provider-Filter gespeichert wurden, müssen beim
    Lesen bereinigt werden (delete + warning + return None).
    """

    def test_stale_non_anthropic_model_returns_none(
        self, service: ModelService
    ) -> None:
        """Direkt in Storage geschriebenes gpt-5-5 wird bei get_user_model
        als stale erkannt, aufgeräumt und None zurückgegeben."""
        # Direkt in Storage schreiben (umgeht set_user_model Provider-Filter)
        service._storage.set_model(user_id=1, model_id="gpt-5-5")

        # get_user_model muss None zurückgeben (stale Revalidierung)
        result = service.get_user_model(user_id=1)
        assert result is None, (
            f"Stale Nicht-Anthropic-Modell 'gpt-5-5' hätte None ergeben müssen, "
            f"bekam aber: {result}"
        )

    def test_stale_model_cleaned_from_storage(self, service: ModelService) -> None:
        """Nach Revalidierung muss der stale Eintrag aus der DB entfernt sein."""
        service._storage.set_model(user_id=1, model_id="gpt-5-5")

        # Erster Aufruf: erkennt stale, bereinigt
        service.get_user_model(user_id=1)

        # Direkter Storage-Check: Eintrag muss weg sein
        raw = service._storage.get_model(user_id=1)
        assert raw is None, (
            f"Stale Eintrag hätte aus Storage gelöscht werden müssen, "
            f"ist aber noch da: {raw}"
        )

    def test_stale_effective_model_falls_back_to_default(
        self, service: ModelService
    ) -> None:
        """get_effective_model mit stale Storage fällt auf Default zurück."""
        service._storage.set_model(user_id=1, model_id="gemini-3-1-pro")

        effective = service.get_effective_model(user_id=1)
        assert effective == DEFAULT_MODEL, (
            f"Stale Modell hätte auf DEFAULT_MODEL ({DEFAULT_MODEL}) "
            f"fallen müssen, bekam: {effective}"
        )

    def test_valid_anthropic_model_not_cleaned(self, service: ModelService) -> None:
        """Valide Anthropic-Modelle werden NICHT fälschlich bereinigt."""
        service.set_user_model(user_id=1, alias_or_id="opus")

        result = service.get_user_model(user_id=1)
        assert result == "claude-opus-4-7", (
            "Gültiges Anthropic-Modell darf nicht als stale behandelt werden"
        )


# ──────────────────────────────────────────────────────────────
# Implicit Reset Tests (R18 Phase 2 Bug-Fix)
# ──────────────────────────────────────────────────────────────


class TestImplicitReset:
    """R18 Phase 2 Bug-Fix: Wenn ein User das Modell wählt das bereits
    der Slot-Default ist, darf kein Override gespeichert werden.
    Bestehende Overrides werden entfernt (impliziter Reset).
    """

    @pytest.fixture
    def service_with_defaults(self, storage: SqliteModelStorage) -> ModelService:
        """ModelService mit Slot-Defaults (wie in Produktion)."""
        slot_defaults = {
            "code": "claude-opus-4-7",
            "chat": "claude-sonnet-4-6",
            "quick": "claude-haiku-4-5-20251001",
            "reason": "claude-opus-4-7",
            "research": "claude-opus-4-7",
            "creative": "claude-sonnet-4-6",
        }
        return ModelService(storage=storage, slot_defaults=slot_defaults)

    def test_set_default_model_no_override_stored(
        self, service_with_defaults: ModelService
    ) -> None:
        """Opus fuer CODE (= Slot-Default): kein Override gespeichert."""
        success, result = service_with_defaults.set_user_model(
            user_id=1, alias_or_id="opus", slot="code"
        )
        assert success is True
        assert result == "claude-opus-4-7"

        # Kein Override darf existieren
        override = service_with_defaults.get_user_model(user_id=1, slot="code")
        assert override is None, (
            f"Override sollte None sein (impliziter Reset), ist aber: {override}"
        )

    def test_set_default_clears_existing_override(
        self, service_with_defaults: ModelService
    ) -> None:
        """Erst Haiku setzen, dann Opus (= Default): Override wird entfernt."""
        # Erst einen Override setzen
        service_with_defaults.set_user_model(
            user_id=1, alias_or_id="haiku", slot="code"
        )
        assert service_with_defaults.get_user_model(user_id=1, slot="code") is not None

        # Dann den Default wählen
        success, result = service_with_defaults.set_user_model(
            user_id=1, alias_or_id="opus", slot="code"
        )
        assert success is True

        # Override muss jetzt weg sein
        override = service_with_defaults.get_user_model(user_id=1, slot="code")
        assert override is None, (
            f"Bestehender Override hätte entfernt werden müssen, ist aber: {override}"
        )

    def test_non_default_still_creates_override(
        self, service_with_defaults: ModelService
    ) -> None:
        """Haiku fuer CODE (nicht der Default): Override wird gespeichert."""
        success, result = service_with_defaults.set_user_model(
            user_id=1, alias_or_id="haiku", slot="code"
        )
        assert success is True

        override = service_with_defaults.get_user_model(user_id=1, slot="code")
        assert override == "claude-haiku-4-5-20251001"

    def test_global_slot_unaffected(self, service_with_defaults: ModelService) -> None:
        """Global-Slot hat keinen impliziten Reset (kein Slot-Default)."""
        success, result = service_with_defaults.set_user_model(
            user_id=1, alias_or_id="opus", slot="global"
        )
        assert success is True

        override = service_with_defaults.get_user_model(user_id=1, slot="global")
        assert override == "claude-opus-4-7"

    def test_service_without_slot_defaults_unaffected(
        self, service: ModelService
    ) -> None:
        """Service ohne Slot-Defaults (Legacy): kein impliziter Reset."""
        success, result = service.set_user_model(
            user_id=1, alias_or_id="opus", slot="code"
        )
        assert success is True

        override = service.get_user_model(user_id=1, slot="code")
        assert override == "claude-opus-4-7"

    def test_global_override_prevents_implicit_reset(
        self, service_with_defaults: ModelService
    ) -> None:
        """V9 Fix 3: Global=Opus, User wählt Sonnet für CHAT (= Slot-Default).
        Implicit-Reset darf NICHT greifen, da der User den Slot
        explizit vom Global entkoppeln will."""
        # Global-Override auf Opus setzen
        service_with_defaults.set_user_model(
            user_id=1, alias_or_id="opus", slot="global"
        )

        # User wählt Sonnet für CHAT (= Slot-Default)
        success, result = service_with_defaults.set_user_model(
            user_id=1, alias_or_id="sonnet", slot="chat"
        )
        assert success is True
        assert result == "claude-sonnet-4-6"

        # Slot-Override MUSS existieren (nicht gelöscht!)
        override = service_with_defaults.get_user_model(user_id=1, slot="chat")
        assert override == "claude-sonnet-4-6", (
            f"Bei aktivem Global-Override darf Implicit-Reset nicht greifen. "
            f"Slot-Override sollte 'claude-sonnet-4-6' sein, ist aber: {override}"
        )

    def test_no_global_still_does_implicit_reset(
        self, service_with_defaults: ModelService
    ) -> None:
        """V9 Fix 3: Ohne Global-Override greift Implicit-Reset weiterhin."""
        # Kein Global-Override aktiv
        success, result = service_with_defaults.set_user_model(
            user_id=1, alias_or_id="sonnet", slot="chat"
        )
        assert success is True

        # Slot-Override darf NICHT existieren (Implicit-Reset)
        override = service_with_defaults.get_user_model(user_id=1, slot="chat")
        assert override is None, (
            f"Ohne Global-Override muss Implicit-Reset greifen. "
            f"Override sollte None sein, ist aber: {override}"
        )
