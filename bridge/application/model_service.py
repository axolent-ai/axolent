"""Model-Service: Verwaltet User-Modell-Overrides.

Alias-Resolution: 'opus' -> 'claude-opus-4-7', etc.
Phase 1: nur globaler Slot (gilt für alle Anfragen).
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
# Erweiterbar für OpenAI, Gemini, etc. in Phase 2+.
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

# Default-Modell aus Environment (Fallback wenn kein Override)
DEFAULT_MODEL: str = os.getenv("CLAUDE_POOL_MODEL", "claude-sonnet-4-6")

# Alle akzeptierten Modell-IDs (für Validierung)
VALID_MODEL_IDS: set[str] = set(MODEL_ALIASES.values())


def resolve_alias(alias_or_id: str) -> Optional[str]:
    """Löst einen Alias oder eine volle Modell-ID auf.

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
        """Liest das aktive Modell für einen User.

        Revalidiert gespeicherte Modell-IDs: falls der gespeicherte Wert
        nicht mehr in VALID_MODEL_IDS enthalten ist (z.B. nach Alias-Update),
        wird er ignoriert und None zurückgegeben.

        Args:
            user_id: Telegram-User-ID.
            slot: Slot-Name (default: 'global').

        Returns:
            Volle Modell-ID oder None wenn kein Override (= Default nutzen).
        """
        stored = self._storage.get_model(user_id, slot)
        if stored is None:
            return None
        if stored not in VALID_MODEL_IDS:
            log.warning(
                "Gespeichertes Modell '%s' für user_id=%d slot='%s' ist nicht mehr "
                "gültig (nicht in VALID_MODEL_IDS). Fallback auf Default.",
                stored,
                user_id,
                slot,
            )
            return None
        return stored

    def get_effective_model(self, user_id: int) -> str:
        """Bestimmt das effektive Modell: User-Override oder Default.

        Args:
            user_id: Telegram-User-ID.

        Returns:
            Volle Modell-ID (garantiert nicht None).
        """
        override = self.get_user_model(user_id, "global")
        return override if override else DEFAULT_MODEL

    def set_user_model(
        self, user_id: int, alias_or_id: str, slot: str = "global"
    ) -> tuple[bool, str]:
        """Setzt das Modell für einen User (Phase 1: nur globaler Slot).

        Args:
            user_id: Telegram-User-ID.
            alias_or_id: Alias oder volle Modell-ID.
            slot: Slot-Name (default: 'global'). Phase 2: 'chat', 'code', etc.

        Returns:
            Tuple (success: bool, resolved_model_id_or_error: str).
        """
        resolved = resolve_alias(alias_or_id)
        if resolved is None:
            available = ", ".join(sorted(MODEL_ALIASES.keys()))
            return (
                False,
                f"Unbekanntes Modell: '{alias_or_id}'. Verfügbar: {available}",
            )

        self._storage.set_model(user_id, resolved, slot=slot)
        log.info(
            "User %d hat Modell auf '%s' gesetzt (Input: '%s', slot: '%s')",
            user_id,
            resolved,
            alias_or_id,
            slot,
        )
        return True, resolved

    def reset_user_model(self, user_id: int, slot: str = "global") -> bool:
        """Entfernt das Modell-Override (zurück auf Default).

        Args:
            user_id: Telegram-User-ID.
            slot: Slot-Name (default: 'global').

        Returns:
            True wenn ein Override entfernt wurde.
        """
        deleted = self._storage.delete_model(user_id, slot)
        if deleted:
            log.info(
                "User %d hat Modell auf Default zurückgesetzt (slot: '%s')",
                user_id,
                slot,
            )
        return deleted

    @staticmethod
    def get_model_display_name(model_id: str) -> str:
        """Gibt einen menschenlesbaren Namen für eine Modell-ID zurück.

        Args:
            model_id: Volle Modell-ID.

        Returns:
            Display-Name (z.B. 'Opus 4.7' für 'claude-opus-4-7').
        """
        # Reverse-Lookup: model_id -> alias mit Versionsnummer
        _DISPLAY_NAMES: dict[str, str] = {
            "opus": "Opus 4.7",
            "sonnet": "Sonnet 4.6",
            "haiku": "Haiku 4.5",
        }
        for alias, full_id in MODEL_ALIASES.items():
            if full_id == model_id:
                return _DISPLAY_NAMES.get(alias, alias.capitalize())
        return model_id

    @staticmethod
    def list_available_aliases() -> dict[str, str]:
        """Gibt alle verfügbaren Aliase mit ihren Modell-IDs zurück."""
        return dict(MODEL_ALIASES)
