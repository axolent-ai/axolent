"""Model service: manages user model overrides.

Alias resolution: 'opus' -> 'claude-opus-4-7', etc.
Phase 1: global slot only (applies to all requests).
Phase 2+: per-slot (chat, code, etc.).

Backward-compatible: no override -> CLAUDE_POOL_MODEL env variable.

Phase 2b: MODEL_ALIASES and VALID_MODEL_IDS are loaded dynamically from the
ModelRegistry. The public interface remains identical.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from application.model_registry import ModelRegistry

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteModelStorage

log = logging.getLogger(__name__)

# ModelRegistry as central data source (Phase 2b)
_registry = ModelRegistry()

# Backward-compatible module-level exports (populated from Registry)
MODEL_ALIASES: dict[str, str] = _registry.all_aliases()

# Default model from environment (fallback when no override)
DEFAULT_MODEL: str = os.getenv("CLAUDE_POOL_MODEL", "claude-sonnet-4-6")

# All accepted model IDs (for validation, dynamic from Registry)
VALID_MODEL_IDS: set[str] = _registry.all_ids()


def resolve_alias(alias_or_id: str) -> Optional[str]:
    """Resolve an alias or full model ID.

    Delegates to ModelRegistry for central lookup.

    Args:
        alias_or_id: Alias ('opus', 'sonnet', 'haiku') or full model ID.

    Returns:
        Full model ID or None if not recognized.
    """
    return _registry.resolve_id(alias_or_id)


class ModelService:
    """Manages user model overrides.

    Stores and reads model preferences via SqliteModelStorage.
    Provides alias resolution and validation.
    """

    def __init__(
        self,
        storage: "SqliteModelStorage",
        slot_defaults: dict[str, str] | None = None,
    ) -> None:
        self._storage = storage
        # Slot defaults: slot_name -> resolved model_id (e.g. "code" -> "claude-opus-4-7")
        # Extracted from SlotConfigs in main.py and passed here.
        self._slot_defaults: dict[str, str] = slot_defaults or {}

    def get_user_model(self, user_id: int, slot: str = "global") -> Optional[str]:
        """Read the active model for a user.

        Revalidates stored model IDs in two stages:
        1. Is the ID in VALID_MODEL_IDS at all? (e.g. after alias update)
        2. Does the ID belong to ACTIVE_PROVIDER? (prevents stale values from
           non-Anthropic models set before the provider filter)

        Invalid values are cleaned up from storage (Option B).

        Args:
            user_id: Telegram user ID.
            slot: Slot name (default: 'global').

        Returns:
            Full model ID or None if no override (= use default).
        """
        stored = self._storage.get_model(user_id, slot)
        if stored is None:
            return None
        if stored not in VALID_MODEL_IDS:
            log.warning(
                "Stored model '%s' for user_id=%d slot='%s' is no longer "
                "valid (not in VALID_MODEL_IDS). Cleaning up, falling back to default.",
                stored,
                user_id,
                slot,
            )
            self._storage.delete_model(user_id, slot)
            return None
        # Provider revalidation: clean up stale values from non-Anthropic models
        metadata = _registry.get(stored)
        if metadata is not None and metadata.provider != self.ACTIVE_PROVIDER:
            log.warning(
                "Stored model '%s' (provider=%s) for user_id=%d slot='%s' "
                "does not belong to active provider '%s'. Cleaning up, falling back to default.",
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
        """Determine the effective model: user override or default.

        Args:
            user_id: Telegram user ID.

        Returns:
            Full model ID (guaranteed not None).
        """
        override = self.get_user_model(user_id, "global")
        return override if override else DEFAULT_MODEL

    def get_all_slot_overrides(self, user_id: int) -> dict[str, str]:
        """Read all slot overrides for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            Dict of slot_name -> model_id for all active overrides.
        """
        return self._storage.get_all_models(user_id)

    def reset_all_slots(self, user_id: int) -> int:
        """Remove all model overrides for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            Number of deleted overrides.
        """
        return self._storage.delete_all_models(user_id)

    # Provider currently active as default backend (Phase 1: Claude only).
    # Phase 2 (TaskRouter) extends this to dynamic provider routing.
    ACTIVE_PROVIDER: str = "anthropic"

    def set_user_model(
        self, user_id: int, alias_or_id: str, slot: str = "global"
    ) -> tuple[bool, str]:
        """Set the model for a user (Phase 1: global slot only).

        Additionally validates that the model belongs to the active provider.
        As long as the main path only supports Claude, non-Anthropic
        models are rejected (Phase 2: TaskRouter extends this).

        Args:
            user_id: Telegram user ID.
            alias_or_id: Alias or full model ID.
            slot: Slot name (default: 'global'). Phase 2: 'chat', 'code', etc.

        Returns:
            Tuple (success: bool, resolved_model_id_or_error: str).
        """
        resolved = resolve_alias(alias_or_id)
        if resolved is None:
            available = ", ".join(sorted(self.list_available_aliases().keys()))
            return (
                False,
                f"Unknown model: '{alias_or_id}'. Available: {available}",
            )

        # Provider check: only accept models from the active provider
        metadata = _registry.get(alias_or_id)
        if metadata is not None and metadata.provider != self.ACTIVE_PROVIDER:
            available = ", ".join(sorted(self.list_available_aliases().keys()))
            return (
                False,
                f"Model '{metadata.display_name}' uses provider "
                f"'{metadata.provider}'. Currently only Anthropic Claude "
                f"is supported. Available: {available}",
            )

        # Implicit reset: if the chosen model matches the slot default
        # AND no global override is active, no override is stored
        # (or an existing one is removed). This way the UI correctly shows "(Default)".
        #
        # If a global override is active, choosing the slot default is an
        # explicit override (user wants to decouple the slot from global) and
        # MUST be stored as a slot override.
        self._last_was_implicit_reset = False
        if slot != "global" and slot in self._slot_defaults:
            slot_default_id = self._slot_defaults[slot]
            if resolved == slot_default_id:
                # Check global override: only implicit-reset when NO global is active
                global_override = self._storage.get_model(user_id, "global")
                if global_override is None:
                    self._storage.delete_model(user_id, slot)
                    self._last_was_implicit_reset = True
                    log.info(
                        "User %d chose model '%s' which matches the slot default "
                        "for '%s'. Override removed (implicit reset).",
                        user_id,
                        resolved,
                        slot,
                    )
                    return True, resolved
                # Global active: explicit override, proceed to normal save
                log.info(
                    "User %d chose model '%s' (= slot default for '%s'), "
                    "but global override '%s' is active. Saving as slot override.",
                    user_id,
                    resolved,
                    slot,
                    global_override,
                )

        self._storage.set_model(user_id, resolved, slot=slot)
        log.info(
            "User %d set model to '%s' (input: '%s', slot: '%s')",
            user_id,
            resolved,
            alias_or_id,
            slot,
        )
        return True, resolved

    @property
    def last_was_implicit_reset(self) -> bool:
        """True if the last set_user_model() triggered an implicit reset.

        Allows the presentation layer to distinguish between "set" and
        "implicit_reset" in the audit log.
        """
        return getattr(self, "_last_was_implicit_reset", False)

    def reset_user_model(self, user_id: int, slot: str = "global") -> bool:
        """Remove the model override (back to default).

        Args:
            user_id: Telegram user ID.
            slot: Slot name (default: 'global').

        Returns:
            True if an override was removed.
        """
        deleted = self._storage.delete_model(user_id, slot)
        if deleted:
            log.info(
                "User %d reset model to default (slot: '%s')",
                user_id,
                slot,
            )
        return deleted

    @staticmethod
    def get_model_display_name(model_id: str) -> str:
        """Return a human-readable name for a model ID.

        Delegates to ModelRegistry for central display-name lookup.

        Args:
            model_id: Full model ID.

        Returns:
            Display name (e.g. 'Opus 4.7' for 'claude-opus-4-7').
        """
        return _registry.get_display_name(model_id)

    @staticmethod
    def list_available_aliases() -> dict[str, str]:
        """Return available aliases for the active provider.

        Phase 1: Anthropic models only. Phase 2 (TaskRouter) extends this.
        """
        active_models = _registry.for_provider(ModelService.ACTIVE_PROVIDER)
        active_ids = {m.id for m in active_models}
        return {
            alias: model_id
            for alias, model_id in MODEL_ALIASES.items()
            if model_id in active_ids
        }
