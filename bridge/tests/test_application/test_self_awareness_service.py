"""Tests fuer application.self_awareness_service: Self-Awareness-Block-Aufbau.

Isolierte Unit-Tests mit gemockten Dependencies (ModelService, TaskRouter, ModelRegistry).
Testet:
  - Block-Aufbau mit Default-Modell
  - Block-Aufbau mit User-Override
  - Slot-Belegungsliste fuer alle 6 Slots
  - i18n (DE/EN)
  - Graceful Degradation bei Fehlern
  - Edge Cases (None-Werte, unbekannte Modelle)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import tempfile

import pytest

from application.self_awareness_service import SelfAwarenessService
from application.model_registry import ModelRegistry


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def registry() -> ModelRegistry:
    """Echte ModelRegistry mit Produktions-YAML."""
    return ModelRegistry()


@pytest.fixture
def mock_model_service() -> MagicMock:
    """Gemockter ModelService."""
    svc = MagicMock()
    svc.get_all_slot_overrides = MagicMock(return_value={})
    return svc


@pytest.fixture
def mock_task_router() -> MagicMock:
    """Gemockter TaskRouter."""
    router = MagicMock()
    # Default: gibt immer claude-sonnet-4-6 zurueck
    router.get_default_for_slot = MagicMock(return_value="claude-sonnet-4-6")
    return router


@pytest.fixture
def sa_service(
    mock_model_service: MagicMock,
    mock_task_router: MagicMock,
    registry: ModelRegistry,
) -> SelfAwarenessService:
    """Standard-SelfAwarenessService mit Mocks."""
    return SelfAwarenessService(
        model_service=mock_model_service,
        task_router=mock_task_router,
        model_registry=registry,
    )


# ---------------------------------------------------------------
# Test: Block-Aufbau Basics
# ---------------------------------------------------------------


class TestBuildBasics:
    """Grundlegende Block-Aufbau-Tests."""

    def test_build_returns_non_empty_string(
        self, sa_service: SelfAwarenessService
    ) -> None:
        """build() gibt einen nicht-leeren String zurueck."""
        result = sa_service.build(user_id=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_contains_self_awareness_marker(
        self, sa_service: SelfAwarenessService
    ) -> None:
        """Block enthaelt [SELF-AWARENESS] Marker."""
        result = sa_service.build(user_id=1)
        assert "[SELF-AWARENESS]" in result

    def test_build_contains_model_info(self, sa_service: SelfAwarenessService) -> None:
        """Block enthaelt Modell-Info (Display-Name und ID)."""
        result = sa_service.build(user_id=1)
        assert "Modell:" in result or "Current model:" in result

    def test_build_contains_slot_info(self, sa_service: SelfAwarenessService) -> None:
        """Block enthaelt Slot-Info."""
        result = sa_service.build(user_id=1, task_slot_name="code")
        assert "Slot: code" in result

    def test_build_contains_provider_info(
        self, sa_service: SelfAwarenessService
    ) -> None:
        """Block enthaelt Provider-Info."""
        result = sa_service.build(user_id=1)
        assert "Provider:" in result

    def test_build_default_slot_is_chat(self, sa_service: SelfAwarenessService) -> None:
        """Ohne task_slot_name ist der Default 'chat'."""
        result = sa_service.build(user_id=1)
        assert "Slot: chat" in result

    def test_build_with_specific_model(self, sa_service: SelfAwarenessService) -> None:
        """Explizites user_model wird im Block angezeigt."""
        result = sa_service.build(user_id=1, user_model="claude-opus-4-7")
        assert "Opus 4.7" in result
        assert "claude-opus-4-7" in result


# ---------------------------------------------------------------
# Test: i18n
# ---------------------------------------------------------------


class TestI18n:
    """Sprachvarianten des Blocks."""

    def test_build_german_default(self, sa_service: SelfAwarenessService) -> None:
        """Standard-Sprache ist Deutsch."""
        result = sa_service.build(user_id=1)
        assert "Modell:" in result

    def test_build_english(self, sa_service: SelfAwarenessService) -> None:
        """lang='en' erzeugt englischen Block."""
        result = sa_service.build(user_id=1, lang="en")
        assert "Current model:" in result

    def test_build_unknown_language_falls_back_to_english(
        self, sa_service: SelfAwarenessService
    ) -> None:
        """Unbekannte Sprache faellt auf Englisch zurueck."""
        result = sa_service.build(user_id=1, lang="ja")
        assert "Current model:" in result


# ---------------------------------------------------------------
# Test: Slot-Belegungsliste
# ---------------------------------------------------------------


class TestSlotInfos:
    """Tests fuer _build_all_slot_infos."""

    def test_all_six_slots_present(self, sa_service: SelfAwarenessService) -> None:
        """Slot-Belegungsliste enthaelt alle 6 Slots."""
        infos = sa_service._build_all_slot_infos(user_id=1)
        assert len(infos) == 6
        slot_names = {info.slot_name for info in infos}
        assert slot_names == {"chat", "code", "reason", "creative", "quick", "research"}

    def test_all_slots_default_source(self, sa_service: SelfAwarenessService) -> None:
        """Ohne Overrides haben alle Slots source='default'."""
        infos = sa_service._build_all_slot_infos(user_id=1)
        for info in infos:
            assert info.source == "default"

    def test_global_override_affects_all_slots(
        self,
        mock_model_service: MagicMock,
        mock_task_router: MagicMock,
        registry: ModelRegistry,
    ) -> None:
        """Globaler Override setzt alle Slots auf 'global' source."""
        mock_model_service.get_all_slot_overrides = MagicMock(
            return_value={"global": "claude-opus-4-7"}
        )
        sa_svc = SelfAwarenessService(
            model_service=mock_model_service,
            task_router=mock_task_router,
            model_registry=registry,
        )
        infos = sa_svc._build_all_slot_infos(user_id=1)
        for info in infos:
            assert info.source == "global"
            assert "Opus" in info.model_display_name

    def test_slot_override_takes_precedence(
        self,
        mock_model_service: MagicMock,
        mock_task_router: MagicMock,
        registry: ModelRegistry,
    ) -> None:
        """Slot-spezifischer Override hat Vorrang vor Global."""
        mock_model_service.get_all_slot_overrides = MagicMock(
            return_value={
                "global": "claude-sonnet-4-6",
                "code": "claude-opus-4-7",
            }
        )
        sa_svc = SelfAwarenessService(
            model_service=mock_model_service,
            task_router=mock_task_router,
            model_registry=registry,
        )
        infos = sa_svc._build_all_slot_infos(user_id=1)
        code_info = next(i for i in infos if i.slot_name == "code")
        chat_info = next(i for i in infos if i.slot_name == "chat")
        assert code_info.source == "user-override"
        assert "Opus" in code_info.model_display_name
        assert chat_info.source == "global"

    def test_slot_infos_in_block_output(self, sa_service: SelfAwarenessService) -> None:
        """Block-Output enthaelt Slot-Belegungsliste."""
        result = sa_service.build(user_id=1)
        assert "CHAT:" in result
        assert "CODE:" in result
        assert "REASON:" in result


# ---------------------------------------------------------------
# Test: Graceful Degradation
# ---------------------------------------------------------------


class TestGracefulDegradation:
    """Fehler-Robustheit."""

    def test_build_without_user_id(self, sa_service: SelfAwarenessService) -> None:
        """Ohne user_id wird kein Slot-Liste gebaut, Block ist trotzdem da."""
        result = sa_service.build(user_id=None)
        assert "[SELF-AWARENESS]" in result
        # Keine Slot-Liste da user_id=None
        assert "CHAT:" not in result

    def test_build_with_none_model_service(self, registry: ModelRegistry) -> None:
        """Ohne ModelService funktioniert der Block trotzdem."""
        sa_svc = SelfAwarenessService(
            model_service=None,
            task_router=None,
            model_registry=registry,
        )
        result = sa_svc.build(user_id=1)
        assert "[SELF-AWARENESS]" in result

    def test_build_with_unknown_model_id(
        self, sa_service: SelfAwarenessService
    ) -> None:
        """Unbekannte Modell-ID wird als Fallback direkt angezeigt."""
        result = sa_service.build(user_id=1, user_model="totally-unknown-model-xyz")
        assert "[SELF-AWARENESS]" in result
        assert "totally-unknown-model-xyz" in result
        assert "Provider: unknown" in result

    def test_build_with_model_service_exception(
        self,
        mock_task_router: MagicMock,
        registry: ModelRegistry,
    ) -> None:
        """Exception im ModelService wird graceful behandelt."""
        broken_model_service = MagicMock()
        broken_model_service.get_all_slot_overrides = MagicMock(
            side_effect=RuntimeError("DB connection lost")
        )
        sa_svc = SelfAwarenessService(
            model_service=broken_model_service,
            task_router=mock_task_router,
            model_registry=registry,
        )
        # Soll nicht crashen, Slot-Liste wird uebersprungen
        result = sa_svc.build(user_id=1)
        assert "[SELF-AWARENESS]" in result


# ---------------------------------------------------------------
# Test: Integration mit echtem ModelService + TaskRouter
# ---------------------------------------------------------------


class TestIntegrationWithRealServices:
    """Integration-Tests mit echtem ModelService + TaskRouter."""

    def test_with_real_model_service_and_task_router(
        self, registry: ModelRegistry
    ) -> None:
        """Funktioniert mit echtem ModelService und TaskRouter."""
        from application.model_service import ModelService
        from application.task_router import TaskRouter, load_slot_configs
        from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_sa_integration.db"
            conn = SqliteConnection(db_path)
            try:
                storage = SqliteModelStorage(conn)
                model_service = ModelService(storage=storage)
                slot_configs = load_slot_configs()
                task_router = TaskRouter(
                    slot_configs=slot_configs,
                    model_service=model_service,
                )

                sa_svc = SelfAwarenessService(
                    model_service=model_service,
                    task_router=task_router,
                    model_registry=registry,
                )

                result = sa_svc.build(
                    user_id=1,
                    user_model="claude-opus-4-7",
                    task_slot_name="code",
                    lang="de",
                )
                assert "Opus 4.7" in result
                assert "Slot: code" in result
                assert "anthropic" in result
            finally:
                conn.close()

    def test_with_user_override(self, registry: ModelRegistry) -> None:
        """User-Override wird korrekt in Slot-Liste reflektiert."""
        from application.model_service import ModelService
        from application.task_router import TaskRouter, load_slot_configs
        from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_sa_override.db"
            conn = SqliteConnection(db_path)
            try:
                storage = SqliteModelStorage(conn)
                slot_configs = load_slot_configs()
                slot_defaults = {
                    cfg.slot.value: cfg.default_model for cfg in slot_configs
                }
                model_service = ModelService(
                    storage=storage, slot_defaults=slot_defaults
                )
                model_service.set_user_model(user_id=1, alias_or_id="opus")

                task_router = TaskRouter(
                    slot_configs=slot_configs,
                    model_service=model_service,
                )

                sa_svc = SelfAwarenessService(
                    model_service=model_service,
                    task_router=task_router,
                    model_registry=registry,
                )

                result = sa_svc.build(user_id=1)
                # Global Override auf Opus: alle Slots sollten Opus zeigen
                assert "Opus 4.7" in result
            finally:
                conn.close()
