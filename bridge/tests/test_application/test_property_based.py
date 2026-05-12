"""Property-Based Tests mit Hypothesis.

Hypothesis generiert automatisch Edge-Cases (leere Strings, Unicode,
sehr lange Inputs, Grenzwerte) und findet Invarianten-Verletzungen.

Getestete Properties:
  1. TaskRouter.classify() gibt immer einen gültigen TaskSlot zurück
  2. ModelService.set_user_model() validiert Aliases korrekt
  3. SqliteModelStorage Roundtrip: set + get liefert immer den gesetzten Wert
"""

from __future__ import annotations


import pytest
from hypothesis import given, settings, HealthCheck, strategies as st

from application.model_service import ModelService, resolve_alias
from application.task_router import SlotConfig, TaskRouter
from domain.task_slot import TaskSlot
from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def full_router() -> TaskRouter:
    """TaskRouter mit Produktions-nahen SlotConfigs."""
    configs = [
        SlotConfig(
            slot=TaskSlot.CODE,
            default_model="opus",
            patterns=("```", "def ", "function ", "class ", "import "),
            keywords=("debug", "refactor", "implementier", "bug", "error"),
            min_keyword_matches=2,
        ),
        SlotConfig(
            slot=TaskSlot.REASON,
            default_model="opus",
            keywords=("analysier", "vergleich", "schritt fuer schritt", "strategie"),
            min_keyword_matches=2,
            min_word_count=50,
        ),
        SlotConfig(
            slot=TaskSlot.RESEARCH,
            default_model="opus",
            keywords=("recherchier", "research", "zusammenfass", "markt"),
            patterns=("http://", "https://"),
            min_keyword_matches=2,
        ),
        SlotConfig(
            slot=TaskSlot.CREATIVE,
            default_model="sonnet",
            keywords=("brainstorm", "ideen", "vorschlaege", "schreib"),
            min_keyword_matches=2,
        ),
        SlotConfig(
            slot=TaskSlot.QUICK,
            default_model="haiku",
            keywords=("klassifizier", "extrahier", "ja oder nein"),
            min_keyword_matches=1,
            max_word_count=50,
        ),
        SlotConfig(
            slot=TaskSlot.CHAT,
            default_model="sonnet",
            fallback=True,
        ),
    ]
    return TaskRouter(configs)


# ──────────────────────────────────────────────────────────────
# Property 1: TaskRouter gibt immer gültigen Slot zurück
# ──────────────────────────────────────────────────────────────


class TestTaskRouterPropertyBased:
    """TaskRouter.classify() muss für JEDEN Input einen gültigen TaskSlot liefern."""

    @given(text=st.text(min_size=0, max_size=500))
    @settings(
        max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_classify_always_returns_valid_slot(
        self, full_router: TaskRouter, text: str
    ) -> None:
        """Invariante: Ergebnis ist immer ein gültiger TaskSlot mit score >= 0."""
        result = full_router.classify(text)
        assert result.slot in TaskSlot
        assert result.score >= 0

    @given(text=st.text(min_size=0, max_size=500))
    @settings(
        max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_classify_matched_patterns_are_strings(
        self, full_router: TaskRouter, text: str
    ) -> None:
        """Invariante: matched_patterns und matched_keywords sind immer String-Tuples."""
        result = full_router.classify(text)
        assert isinstance(result.matched_patterns, tuple)
        assert isinstance(result.matched_keywords, tuple)
        for p in result.matched_patterns:
            assert isinstance(p, str)
        for k in result.matched_keywords:
            assert isinstance(k, str)

    @given(text=st.text(min_size=1, max_size=500))
    @settings(
        max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_classify_with_prefix_returns_matching_slot(
        self, full_router: TaskRouter, text: str
    ) -> None:
        """Invariante: /code <text> muss immer CODE-Slot zurückgeben."""
        result = full_router.classify(f"/code {text}")
        assert result.slot == TaskSlot.CODE
        assert result.score == 1000  # Expliziter Marker-Score


# ──────────────────────────────────────────────────────────────
# Property 2: ModelService validiert Aliases korrekt
# ──────────────────────────────────────────────────────────────


class TestModelServicePropertyBased:
    """ModelService.set_user_model() muss Aliases korrekt validieren."""

    @given(
        user_id=st.integers(min_value=1, max_value=10**12),
        alias=st.sampled_from(["opus", "sonnet", "haiku"]),
    )
    @settings(max_examples=50)
    def test_valid_alias_always_succeeds(self, user_id: int, alias: str) -> None:
        """Invariante: bekannte Aliases setzen immer erfolgreich."""
        conn = SqliteConnection(":memory:")
        try:
            storage = SqliteModelStorage(conn)
            svc = ModelService(storage=storage)
            success, resolved = svc.set_user_model(user_id, alias)
            assert success is True
            assert resolve_alias(alias) == resolved
        finally:
            conn.close()

    @given(
        user_id=st.integers(min_value=1, max_value=10**12),
        alias=st.text(min_size=1, max_size=50).filter(
            lambda x: x.lower().strip()
            not in {
                "opus",
                "sonnet",
                "haiku",
                "claude-opus-4-7",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            }
        ),
    )
    @settings(max_examples=100)
    def test_unknown_alias_always_fails(self, user_id: int, alias: str) -> None:
        """Invariante: unbekannte Aliases schlagen immer fehl (kein Crash)."""
        conn = SqliteConnection(":memory:")
        try:
            storage = SqliteModelStorage(conn)
            svc = ModelService(storage=storage)
            success, error_msg = svc.set_user_model(user_id, alias)
            assert success is False
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0
        finally:
            conn.close()

    @given(
        user_id=st.integers(min_value=1, max_value=10**12),
        alias=st.sampled_from(["opus", "sonnet", "haiku"]),
        slot=st.sampled_from(
            ["global", "chat", "code", "reason", "creative", "quick", "research"]
        ),
    )
    @settings(max_examples=50)
    def test_set_then_get_roundtrip(self, user_id: int, alias: str, slot: str) -> None:
        """Invariante: nach set_user_model() liefert get_user_model() den Wert zurück."""
        conn = SqliteConnection(":memory:")
        try:
            storage = SqliteModelStorage(conn)
            svc = ModelService(storage=storage)
            success, resolved = svc.set_user_model(user_id, alias, slot=slot)
            assert success is True
            stored = svc.get_user_model(user_id, slot=slot)
            assert stored == resolved
        finally:
            conn.close()


# ──────────────────────────────────────────────────────────────
# Property 3: SQLite Storage Roundtrip
# ──────────────────────────────────────────────────────────────


class TestSqliteStoragePropertyBased:
    """SqliteModelStorage: set + get muss immer den gesetzten Wert liefern."""

    @given(
        user_id=st.integers(min_value=1, max_value=10**12),
        slot=st.sampled_from(
            ["global", "chat", "code", "reason", "creative", "quick", "research"]
        ),
        model_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=200)
    def test_sqlite_storage_roundtrip(
        self, user_id: int, slot: str, model_id: str
    ) -> None:
        """Invariante: set_model + get_model ist identisch."""
        conn = SqliteConnection(":memory:")
        try:
            storage = SqliteModelStorage(conn)
            storage.set_model(user_id, model_id, slot)
            result = storage.get_model(user_id, slot)
            assert result == model_id
        finally:
            conn.close()

    @given(
        user_id=st.integers(min_value=1, max_value=10**12),
        slot=st.sampled_from(
            ["global", "chat", "code", "reason", "creative", "quick", "research"]
        ),
    )
    @settings(max_examples=50)
    def test_delete_then_get_returns_none(self, user_id: int, slot: str) -> None:
        """Invariante: nach delete_model() gibt get_model() None zurück."""
        conn = SqliteConnection(":memory:")
        try:
            storage = SqliteModelStorage(conn)
            storage.set_model(user_id, "test-model", slot)
            storage.delete_model(user_id, slot)
            assert storage.get_model(user_id, slot) is None
        finally:
            conn.close()

    @given(
        user_id=st.integers(min_value=1, max_value=10**12),
        models=st.dictionaries(
            keys=st.sampled_from(
                ["chat", "code", "reason", "creative", "quick", "research"]
            ),
            values=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=1,
            max_size=6,
        ),
    )
    @settings(max_examples=50)
    def test_get_all_models_returns_all_set(
        self, user_id: int, models: dict[str, str]
    ) -> None:
        """Invariante: get_all_models() enthält alle gesetzten Overrides."""
        conn = SqliteConnection(":memory:")
        try:
            storage = SqliteModelStorage(conn)
            for slot, model_id in models.items():
                storage.set_model(user_id, model_id, slot)
            all_models = storage.get_all_models(user_id)
            for slot, model_id in models.items():
                assert all_models[slot] == model_id
        finally:
            conn.close()
