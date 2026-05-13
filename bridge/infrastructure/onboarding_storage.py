"""SQLite-Storage für Onboarding-State.

Neue Tabelle `user_onboarding` in der bestehenden jarvis.db.
Abwärtskompatible Migration: bestehende User gelten als onboarded=True.
"""

from __future__ import annotations

import logging
from typing import Optional

from domain.onboarding import OnboardingState
from infrastructure.sqlite_storage import SqliteConnection

log = logging.getLogger(__name__)

# Schema for user_onboarding table (idempotent CREATE)
_ONBOARDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_onboarding (
    user_id INTEGER PRIMARY KEY,
    onboarded INTEGER NOT NULL DEFAULT 0,
    wizard_lang TEXT,
    skip_count INTEGER NOT NULL DEFAULT 0,
    hint_shown INTEGER NOT NULL DEFAULT 0
);
"""


class OnboardingStorage:
    """SQLite-Adapter für Onboarding-State.

    Speichert pro user_id den Onboarding-Status. Neue User starten
    mit onboarded=False. Bestehende User (die schon in anderen Tabellen
    existieren) werden beim ersten Zugriff als onboarded=True markiert.
    """

    def __init__(self, conn: SqliteConnection) -> None:
        self._conn = conn
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Creates the onboarding table if it doesn't exist."""
        self._conn.execute(_ONBOARDING_SCHEMA, ())
        log.debug("Onboarding-Schema initialisiert")

    def get_state(self, user_id: int) -> Optional[OnboardingState]:
        """Reads the onboarding state for a user.

        Args:
            user_id: Telegram User-ID.

        Returns:
            OnboardingState or None if no record exists.
        """
        row = self._conn.fetchone(
            "SELECT user_id, onboarded, wizard_lang, skip_count, hint_shown "
            "FROM user_onboarding WHERE user_id = ?",
            (user_id,),
        )
        if row is None:
            return None
        return OnboardingState(
            user_id=row["user_id"],
            onboarded=bool(row["onboarded"]),
            wizard_lang=row["wizard_lang"],
            skip_count=row["skip_count"],
            hint_shown=bool(row["hint_shown"]),
        )

    def is_onboarded(self, user_id: int) -> bool:
        """Checks if user is onboarded.

        Returns True if user has an onboarding record with onboarded=True.
        Returns False if no record exists or onboarded=False.
        """
        row = self._conn.fetchone(
            "SELECT onboarded FROM user_onboarding WHERE user_id = ?",
            (user_id,),
        )
        if row is None:
            return False
        return bool(row["onboarded"])

    def set_onboarded(self, user_id: int, lang: Optional[str] = None) -> None:
        """Marks a user as onboarded.

        Args:
            user_id: Telegram User-ID.
            lang: Language chosen during wizard (optional).
        """
        self._conn.execute(
            """INSERT INTO user_onboarding (user_id, onboarded, wizard_lang)
               VALUES (?, 1, ?)
               ON CONFLICT(user_id)
               DO UPDATE SET onboarded = 1, wizard_lang = COALESCE(?, wizard_lang)""",
            (user_id, lang, lang),
        )
        log.info("User %d als onboarded markiert (lang=%s)", user_id, lang)

    def set_wizard_lang(self, user_id: int, lang: str) -> None:
        """Sets the wizard language for step 1 completion.

        Creates the record if it doesn't exist (onboarded stays False).

        Args:
            user_id: Telegram User-ID.
            lang: Chosen language code.
        """
        self._conn.execute(
            """INSERT INTO user_onboarding (user_id, onboarded, wizard_lang)
               VALUES (?, 0, ?)
               ON CONFLICT(user_id)
               DO UPDATE SET wizard_lang = ?""",
            (user_id, lang, lang),
        )
        log.debug("Wizard-Lang gesetzt: user_id=%d lang=%s", user_id, lang)

    def increment_skip_count(self, user_id: int) -> int:
        """Increments the skip counter and returns the new value.

        Creates the record if it doesn't exist.

        Args:
            user_id: Telegram User-ID.

        Returns:
            New skip count value.
        """
        self._conn.execute(
            """INSERT INTO user_onboarding (user_id, onboarded, skip_count)
               VALUES (?, 0, 1)
               ON CONFLICT(user_id)
               DO UPDATE SET skip_count = skip_count + 1""",
            (user_id,),
        )
        row = self._conn.fetchone(
            "SELECT skip_count FROM user_onboarding WHERE user_id = ?",
            (user_id,),
        )
        count = row["skip_count"] if row else 1
        log.debug("Skip-Count inkrementiert: user_id=%d count=%d", user_id, count)
        return count

    def set_hint_shown(self, user_id: int) -> None:
        """Marks the onboarding hint as shown.

        Args:
            user_id: Telegram User-ID.
        """
        self._conn.execute(
            """INSERT INTO user_onboarding (user_id, onboarded, hint_shown)
               VALUES (?, 0, 1)
               ON CONFLICT(user_id)
               DO UPDATE SET hint_shown = 1""",
            (user_id,),
        )
        log.debug("Hint-Shown markiert: user_id=%d", user_id)

    def is_hint_shown(self, user_id: int) -> bool:
        """Checks if the onboarding hint has been shown.

        Args:
            user_id: Telegram User-ID.

        Returns:
            True if hint was already shown.
        """
        row = self._conn.fetchone(
            "SELECT hint_shown FROM user_onboarding WHERE user_id = ?",
            (user_id,),
        )
        if row is None:
            return False
        return bool(row["hint_shown"])

    def migrate_existing_users(self, conn: SqliteConnection) -> int:
        """Marks all existing users (from other tables) as onboarded.

        Idempotent: only inserts for users not already in user_onboarding.
        Finds users in: bookmarks, memory_entries, user_profiles, user_slot_models.

        Args:
            conn: SqliteConnection (same connection).

        Returns:
            Number of users migrated.
        """
        # Collect all known user_ids from existing tables
        migrated = 0
        tables_and_cols = [
            ("bookmarks", "user_id"),
            ("memory_entries", "user_id"),
            ("user_profiles", "user_id"),
            ("user_slot_models", "user_id"),
        ]

        existing_user_ids: set[int] = set()
        for table, col in tables_and_cols:
            try:
                rows = conn.fetchall(f"SELECT DISTINCT {col} FROM {table}")  # nosec B608 hardcoded table/col names, not user input
                for row in rows:
                    uid = row[0]
                    if isinstance(uid, int) and uid > 0:
                        existing_user_ids.add(uid)
            except Exception:  # nosec B112 expected: table may not exist yet
                continue

        for uid in existing_user_ids:
            # Only insert if not already present
            existing = conn.fetchone(
                "SELECT 1 FROM user_onboarding WHERE user_id = ?", (uid,)
            )
            if existing is None:
                conn.execute(
                    "INSERT INTO user_onboarding (user_id, onboarded) VALUES (?, 1)",
                    (uid,),
                )
                migrated += 1

        if migrated > 0:
            log.info(
                "Onboarding-Migration: %d bestehende User als onboarded markiert",
                migrated,
            )
        return migrated

    def _reset_all_for_tests(self) -> None:
        """Deletes all onboarding records (only for tests)."""
        self._conn.execute("DELETE FROM user_onboarding", ())
