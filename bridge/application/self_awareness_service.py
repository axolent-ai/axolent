"""Self-awareness service: builds the self-awareness block for the system prompt.

Extracted from ChatService (Phase 3 Polish). Responsible for:
  * Resolving model metadata from the ModelRegistry
  * Assembling the slot occupancy list for all 6 task slots
  * Building the self-awareness block as a string (i18n-capable: DE/EN)

Dependencies (constructor injection):
  * ModelService: user override lookup
  * TaskRouter: slot default resolution
  * ModelRegistry: model metadata lookup
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
    """Builds the self-awareness block for the system prompt.

    Gives the model factual information about itself so it does not
    hallucinate when the user asks which model is running.
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
        """Build the self-awareness block for the system prompt.

        Resolves model metadata from the ModelRegistry and builds the block.
        If no model was resolved, the system default is used.
        If user_id is given, all 6 slot occupancies are included.

        Args:
            user_id: Telegram user ID for slot occupancy list (optional).
            user_model: Resolved model ID or None.
            task_slot_name: Name of the task slot or None.
            lang: Language code for i18n of the block (default: "de").

        Returns:
            Self-awareness block as string, or empty string on error.
        """
        from application.model_service import DEFAULT_MODEL

        model_id = user_model or DEFAULT_MODEL
        slot = task_slot_name or "chat"

        try:
            metadata = self._registry.get(model_id)

            # Collect all 6 slot occupancies (if user_id is available)
            all_slots: list[SlotInfo] | None = None
            if user_id is not None:
                try:
                    all_slots = self._build_all_slot_infos(user_id)
                except Exception:
                    log.debug(
                        "Could not build slot occupancy list",
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
            # Fallback: use ID directly if not in registry
            return build_self_awareness_block(
                model_display_name=model_id,
                model_id=model_id,
                task_slot=slot,
                provider="unknown",
                all_slots=all_slots,
                lang=lang,
            )
        except Exception:
            log.debug("Could not build self-awareness block", exc_info=True)
            return ""

    def _build_all_slot_infos(self, user_id: int) -> list[SlotInfo]:
        """Build the slot occupancy list for all 6 task slots.

        Priority per slot:
          1. Slot-specific override
          2. Global override
          3. Slot default (via TaskRouter.get_default_for_slot, single source of truth)

        Args:
            user_id: Telegram user ID.

        Returns:
            List of SlotInfo for all 6 slots.
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
                # Single source of truth: TaskRouter.get_default_for_slot
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
