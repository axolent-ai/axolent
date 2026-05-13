"""Tests für infrastructure.onboarding_storage: SQLite-basierter Onboarding-State.

Testet CRUD-Operationen, Migration bestehender User, Skip-Counter
und Hint-Shown-Flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.onboarding_storage import OnboardingStorage
from infrastructure.sqlite_storage import SqliteConnection


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporärer DB-Pfad für Test-Isolation."""
    return tmp_path / "test_onboarding.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    """Frische SQLite-Connection für jeden Test."""
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def storage(conn: SqliteConnection) -> OnboardingStorage:
    """OnboardingStorage mit echtem SQLite-Backend."""
    return OnboardingStorage(conn)


class TestOnboardingStorage:
    """Tests für Onboarding-CRUD-Operationen."""

    def test_new_user_not_onboarded(self, storage: OnboardingStorage) -> None:
        """Neuer User ist nicht onboarded."""
        assert storage.is_onboarded(12345) is False

    def test_get_state_returns_none_for_new_user(
        self, storage: OnboardingStorage
    ) -> None:
        """get_state gibt None für unbekannten User zurück."""
        assert storage.get_state(12345) is None

    def test_set_onboarded(self, storage: OnboardingStorage) -> None:
        """set_onboarded markiert User als onboarded."""
        storage.set_onboarded(12345, lang="de")
        assert storage.is_onboarded(12345) is True

        state = storage.get_state(12345)
        assert state is not None
        assert state.onboarded is True
        assert state.wizard_lang == "de"

    def test_set_wizard_lang_without_onboarding(
        self, storage: OnboardingStorage
    ) -> None:
        """set_wizard_lang speichert Sprache ohne onboarded zu setzen."""
        storage.set_wizard_lang(12345, "fr")
        assert storage.is_onboarded(12345) is False

        state = storage.get_state(12345)
        assert state is not None
        assert state.wizard_lang == "fr"
        assert state.onboarded is False

    def test_set_onboarded_preserves_lang(self, storage: OnboardingStorage) -> None:
        """set_onboarded nach set_wizard_lang behält die Sprache."""
        storage.set_wizard_lang(12345, "es")
        storage.set_onboarded(12345)
        state = storage.get_state(12345)
        assert state is not None
        assert state.onboarded is True
        assert state.wizard_lang == "es"

    def test_set_onboarded_with_lang_overrides(
        self, storage: OnboardingStorage
    ) -> None:
        """set_onboarded mit expliziter Sprache überschreibt vorherige."""
        storage.set_wizard_lang(12345, "es")
        storage.set_onboarded(12345, lang="de")
        state = storage.get_state(12345)
        assert state is not None
        assert state.wizard_lang == "de"

    def test_idempotent_set_onboarded(self, storage: OnboardingStorage) -> None:
        """Doppelter set_onboarded ist idempotent."""
        storage.set_onboarded(12345, lang="en")
        storage.set_onboarded(12345, lang="en")
        assert storage.is_onboarded(12345) is True


class TestSkipCounter:
    """Tests für den Skip-Counter (3-Nachrichten-Logik)."""

    def test_increment_from_zero(self, storage: OnboardingStorage) -> None:
        """Erster Increment gibt 1 zurück."""
        count = storage.increment_skip_count(12345)
        assert count == 1

    def test_increment_multiple(self, storage: OnboardingStorage) -> None:
        """Mehrfacher Increment zählt korrekt hoch."""
        storage.increment_skip_count(12345)
        storage.increment_skip_count(12345)
        count = storage.increment_skip_count(12345)
        assert count == 3

    def test_hint_not_shown_initially(self, storage: OnboardingStorage) -> None:
        """Hint ist initial nicht angezeigt."""
        assert storage.is_hint_shown(12345) is False

    def test_set_hint_shown(self, storage: OnboardingStorage) -> None:
        """set_hint_shown markiert den Hint als angezeigt."""
        storage.set_hint_shown(12345)
        assert storage.is_hint_shown(12345) is True

    def test_hint_shown_idempotent(self, storage: OnboardingStorage) -> None:
        """Doppeltes set_hint_shown ist idempotent."""
        storage.set_hint_shown(12345)
        storage.set_hint_shown(12345)
        assert storage.is_hint_shown(12345) is True

    def test_skip_count_persists_in_state(self, storage: OnboardingStorage) -> None:
        """Skip-Count ist im OnboardingState lesbar."""
        storage.increment_skip_count(12345)
        storage.increment_skip_count(12345)
        state = storage.get_state(12345)
        assert state is not None
        assert state.skip_count == 2


class TestMigration:
    """Tests für die Bestandsuser-Migration."""

    def test_migrate_existing_users_from_bookmarks(
        self, conn: SqliteConnection, storage: OnboardingStorage
    ) -> None:
        """User aus bookmarks-Tabelle wird als onboarded migriert."""
        # Bookmark-Eintrag anlegen
        conn.execute(
            "INSERT INTO bookmarks (user_id, username, chat_id, message_id, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (99999, "testuser", 1, 1, "test", "2026-01-01"),
        )

        migrated = storage.migrate_existing_users(conn)
        assert migrated == 1
        assert storage.is_onboarded(99999) is True

    def test_migrate_idempotent(
        self, conn: SqliteConnection, storage: OnboardingStorage
    ) -> None:
        """Doppelte Migration ändert nichts."""
        conn.execute(
            "INSERT INTO bookmarks (user_id, username, chat_id, message_id, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (99999, "testuser", 1, 1, "test", "2026-01-01"),
        )

        storage.migrate_existing_users(conn)
        migrated = storage.migrate_existing_users(conn)
        assert migrated == 0

    def test_migrate_does_not_overwrite_existing(
        self, conn: SqliteConnection, storage: OnboardingStorage
    ) -> None:
        """Migration überschreibt nicht-onboarded User nicht."""
        # User hat Wizard gestartet aber nicht abgeschlossen
        storage.set_wizard_lang(99999, "fr")
        assert storage.is_onboarded(99999) is False

        conn.execute(
            "INSERT INTO bookmarks (user_id, username, chat_id, message_id, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (99999, "testuser", 1, 1, "test", "2026-01-01"),
        )

        # Migration sollte diesen User nicht nochmal anfassen
        migrated = storage.migrate_existing_users(conn)
        assert migrated == 0
        # Status bleibt unverändert
        assert storage.is_onboarded(99999) is False


class TestResetForTests:
    """Tests für die Test-Hilfsmethode."""

    def test_reset_clears_all(self, storage: OnboardingStorage) -> None:
        """_reset_all_for_tests löscht alle Einträge."""
        storage.set_onboarded(1)
        storage.set_onboarded(2)
        storage._reset_all_for_tests()
        assert storage.get_state(1) is None
        assert storage.get_state(2) is None
