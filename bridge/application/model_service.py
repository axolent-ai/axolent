"""Model-Service: Verwaltet User-Modell-Overrides.

Alias-Resolution: 'opus' -> 'claude-opus-4-20250514', etc.
Phase 1: nur globaler Slot (gilt fuer alle Anfragen).
Phase 2+: per-Slot (chat, code, etc.).

Backward-kompatibel: kein Override -> CLAUDE_POOL_MODEL Env-Variable.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteModelStorage

log = logging.getLogger(__name__)

# Alias -> volle Modell-ID. Aktuell nur Anthropic-Modelle.
# Erweiterbar fuer OpenAI, Gemini, etc. in Phase 2+.
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-haiku-3-5-20241022",
}

# Default-Modell aus Environment (Fallback wenn kein Override)
DEFAULT_MODEL: str = os.getenv("CLAUDE_POOL_MODEL", "claude-sonnet-4-20250514")

# Alle akzeptierten Modell-IDs (fuer Validierung)
VALID_MODEL_IDS: set[str] = set(MODEL_ALIASES.values())


def resolve_alias(alias_or_id: str) -> Optional[str]:
    """Loest einen Alias oder eine volle Modell-ID auf.

    Args:
        alias_or_id: Alias ('opus', 'sonnet', 'haiku') oder volle Modell-ID.

    Returns:
        Volle Modell-ID oder None wenn nicht erkannt.
    """
    lower = alias_or_id.lower().strip()

    # Alias-Lookup
    if lower in MODEL_ALIASES:
        return MODEL_ALIASES[lower]

    # Direkte Modell-ID (schon vollqualifiziert)
    if lower in VALID_MODEL_IDS:
        return lower

    return None


class ModelService:
    """Verwaltet User-Modell-Overrides.

    Speichert und liest Modell-Praeferenzen via SqliteModelStorage.
    Bietet Alias-Resolution und Validierung.
    """

    def __init__(self, storage: "SqliteModelStorage") -> None:
        self._storage = storage

    def get_user_model(self, user_id: int, slot: str = "global") -> Optional[str]:
        """Liest das aktive Modell fuer einen User.

        Args:
            user_id: Telegram-User-ID.
            slot: Slot-Name (default: 'global').

        Returns:
            Volle Modell-ID oder None wenn kein Override (= Default nutzen).
        """
        return self._storage.get_model(user_id, slot)

    def get_effective_model(self, user_id: int) -> str:
        """Bestimmt das effektive Modell: User-Override oder Default.

        Args:
            user_id: Telegram-User-ID.

        Returns:
            Volle Modell-ID (garantiert nicht None).
        """
        override = self._storage.get_model(user_id, "global")
        return override if override else DEFAULT_MODEL

    def set_user_model(self, user_id: int, alias_or_id: str) -> tuple[bool, str]:
        """Setzt das globale Modell fuer einen User.

        Args:
            user_id: Telegram-User-ID.
            alias_or_id: Alias oder volle Modell-ID.

        Returns:
            Tuple (success: bool, resolved_model_id_or_error: str).
        """
        resolved = resolve_alias(alias_or_id)
        if resolved is None:
            available = ", ".join(sorted(MODEL_ALIASES.keys()))
            return (
                False,
                f"Unbekanntes Modell: '{alias_or_id}'. Verfuegbar: {available}",
            )

        self._storage.set_model(user_id, resolved)
        log.info(
            "User %d hat Modell auf '%s' gesetzt (Input: '%s')",
            user_id,
            resolved,
            alias_or_id,
        )
        return True, resolved

    def reset_user_model(self, user_id: int) -> bool:
        """Entfernt das Modell-Override (zurueck auf Default).

        Args:
            user_id: Telegram-User-ID.

        Returns:
            True wenn ein Override entfernt wurde.
        """
        deleted = self._storage.delete_model(user_id, "global")
        if deleted:
            log.info("User %d hat Modell auf Default zurueckgesetzt", user_id)
        return deleted

    @staticmethod
    def get_model_display_name(model_id: str) -> str:
        """Gibt einen menschenlesbaren Namen fuer eine Modell-ID zurueck.

        Args:
            model_id: Volle Modell-ID.

        Returns:
            Display-Name (z.B. 'Opus' fuer 'claude-opus-4-20250514').
        """
        # Reverse-Lookup: model_id -> alias
        for alias, full_id in MODEL_ALIASES.items():
            if full_id == model_id:
                return alias.capitalize()
        return model_id

    @staticmethod
    def list_available_aliases() -> dict[str, str]:
        """Gibt alle verfuegbaren Aliase mit ihren Modell-IDs zurueck."""
        return dict(MODEL_ALIASES)
