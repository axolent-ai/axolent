"""Self-Awareness-Service: baut den Self-Awareness-Block für den System-Prompt.

Extrahiert aus ChatService (Phase 3 Polish). Verantwortlich für:
  - Modell-Metadaten aus der ModelRegistry resolven
  - Slot-Belegungsliste für alle 6 Task-Slots zusammenstellen
  - Self-Awareness-Block als String bauen (i18n-fähig: DE/EN)

Dependencies (Constructor-Injection):
  - ModelService: User-Override-Lookup
  - TaskRouter: Slot-Default-Resolution
  - ModelRegistry: Modell-Metadaten-Lookup
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from domain.personality import SlotInfo, build_self_awareness_block

if TYPE_CHECKING:
    from application.model_registry import ModelRegistry
    from application.model_service import ModelService
    from application.task_router import TaskRouter

log = logging.getLogger(__name__)


class SelfAwarenessService:
    """Baut den Self-Awareness-Block für den System-Prompt.

    Gibt dem Modell faktische Informationen über sich selbst,
    damit es nicht halluziniert wenn der User fragt welches Modell läuft.
    """

    def __init__(
        self,
        model_service: Optional["ModelService"],
        task_router: Optional["TaskRouter"],
        model_registry: "ModelRegistry",
    ) -> None:
        self._model_service = model_service
        self._task_router = task_router
        self._registry = model_registry

    def build(
        self,
        user_id: int | None = None,
        user_model: str | None = None,
        task_slot_name: str | None = None,
        lang: str = "de",
    ) -> str:
        """Baut den Self-Awareness-Block für den System-Prompt.

        Resolved Modell-Metadaten aus der ModelRegistry und baut den Block.
        Wenn kein Modell resolved wurde, wird der System-Default verwendet.
        Wenn user_id gegeben, werden alle 6 Slot-Belegungen inkludiert.

        Args:
            user_id: Telegram-User-ID für Slot-Belegungsliste (optional).
            user_model: Resolved Modell-ID oder None.
            task_slot_name: Name des Task-Slots oder None.
            lang: Sprach-Code für i18n des Blocks (default: "de").

        Returns:
            Self-Awareness-Block als String, oder leerer String bei Fehler.
        """
        from application.model_service import DEFAULT_MODEL

        model_id = user_model or DEFAULT_MODEL
        slot = task_slot_name or "chat"

        try:
            metadata = self._registry.get(model_id)

            # Alle 6 Slot-Belegungen sammeln (wenn user_id vorhanden)
            all_slots: list[SlotInfo] | None = None
            if user_id is not None:
                try:
                    all_slots = self._build_all_slot_infos(user_id)
                except Exception:
                    log.debug(
                        "Slot-Belegungsliste konnte nicht gebaut werden",
                        exc_info=True,
                    )

            if metadata is not None:
                return build_self_awareness_block(
                    model_display_name=metadata.display_name,
                    model_id=metadata.id,
                    task_slot=slot,
                    provider=metadata.provider,
                    all_slots=all_slots,
                    lang=lang,
                )
            # Fallback: ID direkt verwenden wenn nicht in Registry
            return build_self_awareness_block(
                model_display_name=model_id,
                model_id=model_id,
                task_slot=slot,
                provider="unknown",
                all_slots=all_slots,
                lang=lang,
            )
        except Exception:
            log.debug("Self-Awareness-Block konnte nicht gebaut werden", exc_info=True)
            return ""

    def _build_all_slot_infos(self, user_id: int) -> list[SlotInfo]:
        """Baut die Slot-Belegungsliste für alle 6 Task-Slots.

        Priorität pro Slot:
          1. Slot-spezifischer Override
          2. Globaler Override
          3. Slot-Default (via TaskRouter.get_default_for_slot, Single Source of Truth)

        Args:
            user_id: Telegram-User-ID.

        Returns:
            Liste von SlotInfo für alle 6 Slots.
        """
        from application.model_service import DEFAULT_MODEL
        from domain.task_slot import TaskSlot

        result: list[SlotInfo] = []

        overrides: dict[str, str] = {}
        if self._model_service is not None:
            overrides = self._model_service.get_all_slot_overrides(user_id)

        global_override = overrides.get("global")

        for slot in TaskSlot:
            slot_override = overrides.get(slot.value)
            if slot_override:
                model_id = slot_override
                source = "user-override"
            elif global_override:
                model_id = global_override
                source = "global"
            else:
                # Single Source of Truth: TaskRouter.get_default_for_slot
                if self._task_router is not None:
                    model_id = self._task_router.get_default_for_slot(slot)
                else:
                    model_id = DEFAULT_MODEL
                source = "default"

            meta = self._registry.get(model_id)
            display_name = meta.display_name if meta else model_id
            result.append(
                SlotInfo(
                    slot_name=slot.value,
                    model_display_name=display_name,
                    source=source,
                )
            )

        return result
