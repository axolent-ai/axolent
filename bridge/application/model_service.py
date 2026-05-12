"""Model-Service: Verwaltet User-Modell-Overrides.

Alias-Resolution: 'opus' -> 'claude-opus-4-7', etc.
Phase 1: nur globaler Slot (gilt für alle Anfragen).
Phase 2+: per-Slot (chat, code, etc.).

Backward-kompatibel: kein Override -> CLAUDE_POOL_MODEL Env-Variable.

Phase 2b: MODEL_ALIASES und VALID_MODEL_IDS werden dynamisch aus der
ModelRegistry geladen. Die öffentliche Schnittstelle bleibt identisch.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from application.model_registry import ModelRegistry

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteModelStorage

log = logging.getLogger(__name__)

# ModelRegistry als zentrale Datenquelle (Phase 2b)
_registry = ModelRegistry()

# Backward-compatible module-level exports (populated from Registry)
MODEL_ALIASES: dict[str, str] = _registry.all_aliases()

# Default-Modell aus Environment (Fallback wenn kein Override)
DEFAULT_MODEL: str = os.getenv("CLAUDE_POOL_MODEL", "claude-sonnet-4-6")

# Alle akzeptierten Modell-IDs (für Validierung, dynamisch aus Registry)
VALID_MODEL_IDS: set[str] = _registry.all_ids()


def resolve_alias(alias_or_id: str) -> Optional[str]:
    """Löst einen Alias oder eine volle Modell-ID auf.

    Delegiert an ModelRegistry für zentrales Lookup.

    Args:
        alias_or_id: Alias ('opus', 'sonnet', 'haiku') oder volle Modell-ID.

    Returns:
        Volle Modell-ID oder None wenn nicht erkannt.
    """
    return _registry.resolve_id(alias_or_id)


class ModelService:
    """Verwaltet User-Modell-Overrides.

    Speichert und liest Modell-Präferenzen via SqliteModelStorage.
    Bietet Alias-Resolution und Validierung.
    """

    def __init__(
        self,
        storage: "SqliteModelStorage",
        slot_defaults: dict[str, str] | None = None,
    ) -> None:
        self._storage = storage
        # Slot-Defaults: slot_name -> resolved model_id (z.B. "code" -> "claude-opus-4-7")
        # Wird von main.py aus SlotConfigs extrahiert und uebergeben.
        self._slot_defaults: dict[str, str] = slot_defaults or {}

    def get_user_model(self, user_id: int, slot: str = "global") -> Optional[str]:
        """Liest das aktive Modell für einen User.

        Revalidiert gespeicherte Modell-IDs in zwei Stufen:
        1. Ist die ID überhaupt in VALID_MODEL_IDS? (z.B. nach Alias-Update)
        2. Gehört die ID zum ACTIVE_PROVIDER? (verhindert stale Werte von
           Nicht-Anthropic-Modellen die vor dem Provider-Filter gesetzt wurden)

        Bei ungültigen Werten wird der Storage-Eintrag aufgeräumt (Option B).

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
                "gültig (nicht in VALID_MODEL_IDS). Räume auf und Fallback auf Default.",
                stored,
                user_id,
                slot,
            )
            self._storage.delete_model(user_id, slot)
            return None
        # Provider-Revalidierung: stale Werte von Nicht-Anthropic-Modellen bereinigen
        metadata = _registry.get(stored)
        if metadata is not None and metadata.provider != self.ACTIVE_PROVIDER:
            log.warning(
                "Gespeichertes Modell '%s' (provider=%s) für user_id=%d slot='%s' "
                "gehört nicht zum aktiven Provider '%s'. Räume auf und Fallback auf Default.",
                stored,
                metadata.provider,
                user_id,
                slot,
                self.ACTIVE_PROVIDER,
            )
            self._storage.delete_model(user_id, slot)
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

    def get_all_slot_overrides(self, user_id: int) -> dict[str, str]:
        """Liest alle Slot-Overrides für einen User.

        Args:
            user_id: Telegram-User-ID.

        Returns:
            Dict von slot_name -> model_id für alle gesetzten Overrides.
        """
        return self._storage.get_all_models(user_id)

    def reset_all_slots(self, user_id: int) -> int:
        """Entfernt alle Modell-Overrides für einen User.

        Args:
            user_id: Telegram-User-ID.

        Returns:
            Anzahl geloeschter Overrides.
        """
        return self._storage.delete_all_models(user_id)

    # Provider der aktuell als Default-Backend aktiv ist (Phase 1: nur Claude).
    # Phase 2 (TaskRouter) erweitert das auf dynamisches Provider-Routing.
    ACTIVE_PROVIDER: str = "anthropic"

    def set_user_model(
        self, user_id: int, alias_or_id: str, slot: str = "global"
    ) -> tuple[bool, str]:
        """Setzt das Modell für einen User (Phase 1: nur globaler Slot).

        Validiert zusätzlich, dass das Modell zum aktiven Provider passt.
        Solange der Hauptpfad nur Claude unterstützt, werden Nicht-Anthropic-
        Modelle abgelehnt (Phase 2: TaskRouter erweitert das).

        Args:
            user_id: Telegram-User-ID.
            alias_or_id: Alias oder volle Modell-ID.
            slot: Slot-Name (default: 'global'). Phase 2: 'chat', 'code', etc.

        Returns:
            Tuple (success: bool, resolved_model_id_or_error: str).
        """
        resolved = resolve_alias(alias_or_id)
        if resolved is None:
            available = ", ".join(sorted(self.list_available_aliases().keys()))
            return (
                False,
                f"Unbekanntes Modell: '{alias_or_id}'. Verfügbar: {available}",
            )

        # Provider-Check: nur Modelle des aktiven Providers akzeptieren
        metadata = _registry.get(alias_or_id)
        if metadata is not None and metadata.provider != self.ACTIVE_PROVIDER:
            available = ", ".join(sorted(self.list_available_aliases().keys()))
            return (
                False,
                f"Modell '{metadata.display_name}' nutzt Provider "
                f"'{metadata.provider}'. Aktuell wird nur Anthropic Claude "
                f"unterstützt. Verfügbar: {available}",
            )

        # Impliziter Reset: Wenn das gewaehlte Modell dem Slot-Default entspricht,
        # wird kein Override gespeichert (bzw. ein bestehender entfernt).
        # Damit zeigt die UI korrekt "(Default)" an.
        if slot != "global" and slot in self._slot_defaults:
            slot_default_id = self._slot_defaults[slot]
            if resolved == slot_default_id:
                self._storage.delete_model(user_id, slot)
                log.info(
                    "User %d hat Modell '%s' gewählt das dem Slot-Default "
                    "für '%s' entspricht. Override entfernt (impliziter Reset).",
                    user_id,
                    resolved,
                    slot,
                )
                return True, resolved

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

        Delegiert an ModelRegistry für zentrales Display-Name-Lookup.

        Args:
            model_id: Volle Modell-ID.

        Returns:
            Display-Name (z.B. 'Opus 4.7' für 'claude-opus-4-7').
        """
        return _registry.get_display_name(model_id)

    @staticmethod
    def list_available_aliases() -> dict[str, str]:
        """Gibt verfügbare Aliase für den aktiven Provider zurück.

        Phase 1: nur Anthropic-Modelle. Phase 2 (TaskRouter) erweitert das.
        """
        active_models = _registry.for_provider(ModelService.ACTIVE_PROVIDER)
        active_ids = {m.id for m in active_models}
        return {
            alias: model_id
            for alias, model_id in MODEL_ALIASES.items()
            if model_id in active_ids
        }
