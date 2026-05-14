"""InlineKeyboard callbacks for the /settings menu.

Processes settings_* callback queries:
  - settings_slot:<slot>        -> Shows model selection for a slot (level B)
  - settings_model:<slot>:<alias> -> Sets model override for slot
  - settings_reset:<slot>       -> Resets a slot to default
  - settings_reset_all          -> Shows confirmation dialog
  - settings_reset_all_confirm  -> Resets all slots
  - settings_back               -> Back to main menu (level A)
  - settings_lang:<code>        -> Sets language
  - settings_lang_menu          -> Shows language selection (level B)
"""

from __future__ import annotations

import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from application.audit_service import log_command_audit
from application.model_service import DEFAULT_MODEL, ModelService, resolve_alias
from domain.task_slot import TaskSlot
from presentation.decorators import require_private_chat, require_whitelist

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# i18n Strings
# ---------------------------------------------------------------------------

_SETTINGS_STRINGS: dict[str, dict[str, str]] = {
    "de": {
        "main_title": "Einstellungen",
        "models_section": "Modelle pro Slot:",
        "lang_section": "Sprache:",
        "reset_all_btn": "Alle zurücksetzen",
        "default_suffix": "(Default)",
        "global_override_suffix": "(global)",
        "global_override_headline": "Globaler Override: {display_name} (alle Slots)",
        "global_override_text": "⚡ <b>Globaler Override aktiv: {display_name} (alle Slots)</b>",
        "reset_global_btn": "Globalen Override aufheben",
        "slot_select_title": "{slot} — Modell wählen",
        "current_marker": "●",
        "other_marker": "○",
        "back_btn": "← Zurück zum Hauptmenü",
        "reset_slot_btn": "Auf Default zurücksetzen",
        "reset_confirm_title": "Wirklich alle Modell-Overrides zurücksetzen?",
        "reset_confirm_yes": "Ja, alle zurücksetzen",
        "reset_confirm_cancel": "Abbrechen",
        "reset_all_done": "Alle Modell-Overrides zurückgesetzt ({count} entfernt).",
        "reset_all_nothing": "Keine Overrides aktiv, nichts zu tun.",
        "model_set": "Modell für {slot} gesetzt: {display_name}",
        "model_reset": "Modell für {slot} auf Default zurückgesetzt.",
        "model_reset_nothing": "{slot} nutzt bereits den Default.",
        "lang_title": "Sprache wählen",
        "lang_back": "← Zurück",
        "lang_set": "Sprache gewechselt: {name}",
    },
    "en": {
        "main_title": "Settings",
        "models_section": "Models per slot:",
        "lang_section": "Language:",
        "reset_all_btn": "Reset all",
        "default_suffix": "(Default)",
        "global_override_suffix": "(global)",
        "global_override_headline": "Global override: {display_name} (all slots)",
        "global_override_text": "⚡ <b>Global Override active: {display_name} (all slots)</b>",
        "reset_global_btn": "Remove global override",
        "slot_select_title": "{slot} — Choose model",
        "current_marker": "●",
        "other_marker": "○",
        "back_btn": "← Back to main menu",
        "reset_slot_btn": "Reset to default",
        "reset_confirm_title": "Really reset all model overrides?",
        "reset_confirm_yes": "Yes, reset all",
        "reset_confirm_cancel": "Cancel",
        "reset_all_done": "All model overrides reset ({count} removed).",
        "reset_all_nothing": "No overrides active, nothing to do.",
        "model_set": "Model for {slot} set: {display_name}",
        "model_reset": "Model for {slot} reset to default.",
        "model_reset_nothing": "{slot} already uses default.",
        "lang_title": "Choose language",
        "lang_back": "← Back",
        "lang_set": "Language changed: {name}",
    },
}

# Available language options for the settings menu (synced with domain.onboarding.WIZARD_LANGUAGES)
_SETTINGS_LANGUAGES: dict[str, str] = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "nl": "Nederlands",
    "pl": "Polski",
    "sv": "Svenska",
    "tr": "Türkçe",
    "ru": "Русский",
    "uk": "Українська",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "ar": "العربية",
    "hi": "हिन्दी",
    "id": "Bahasa Indo.",
    "th": "ภาษาไทย",
    "vi": "Tiếng Việt",
}

# Available model aliases for Anthropic (active provider)
_AVAILABLE_ALIASES: list[str] = ["opus", "sonnet", "haiku"]


def _get_settings_strings(lang: str = "de") -> dict[str, str]:
    """Returns settings i18n strings for the given language."""
    return _SETTINGS_STRINGS.get(lang, _SETTINGS_STRINGS["de"])


def _get_model_service(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Gets the ModelService from bot_data."""
    return context.application.bot_data.get("model_service")


def _get_chat_service(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Gets the ChatService from bot_data."""
    return context.application.bot_data.get("chat_service")


def _get_task_router(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Gets the TaskRouter from bot_data."""
    return context.application.bot_data.get("task_router")


def _get_slot_default_model(slot: TaskSlot, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Determines the default model for a slot.

    Delegates to TaskRouter.get_default_for_slot (single source of truth).
    Falls back to DEFAULT_MODEL when no TaskRouter is available.
    """
    task_router = _get_task_router(context)
    if task_router is not None and hasattr(task_router, "get_default_for_slot"):
        return task_router.get_default_for_slot(slot)
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Keyboard Builders
# ---------------------------------------------------------------------------


def build_main_menu_keyboard(
    user_id: int,
    model_service: ModelService,
    context: ContextTypes.DEFAULT_TYPE,
    lang: str = "de",
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the main menu (level A) for /settings.

    Shows global override prominently when active.
    Display priority per slot:
      1. Slot-specific override (no suffix)
      2. Global override (with "(global)" suffix)
      3. Slot default (with "(Default)" suffix)

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    s = _get_settings_strings(lang)
    overrides = model_service.get_all_slot_overrides(user_id)

    # Global override checked separately (slot="global" is not a TaskSlot)
    global_override = overrides.get("global")

    buttons: list[list[InlineKeyboardButton]] = []

    # Global override: only reset button in keyboard (headline goes in the text)
    if global_override:
        buttons.append(
            [
                InlineKeyboardButton(
                    f"\U0001f504 {s['reset_global_btn']}",
                    callback_data="settings_reset_global",
                )
            ]
        )

    for slot in TaskSlot:
        slot_override = overrides.get(slot.value)
        if slot_override:
            # Slot-specific override takes priority
            display = model_service.get_model_display_name(slot_override)
            label = f"{slot.value.upper()}: {display}"
        elif global_override:
            # Global override active, slot has no own override
            display = model_service.get_model_display_name(global_override)
            label = f"{slot.value.upper()}: {display} {s['global_override_suffix']}"
        else:
            default_id = _get_slot_default_model(slot, context)
            display = model_service.get_model_display_name(default_id)
            label = f"{slot.value.upper()}: {display} {s['default_suffix']}"

        buttons.append(
            [InlineKeyboardButton(label, callback_data=f"settings_slot:{slot.value}")]
        )

    # Language button
    current_lang_name = _SETTINGS_LANGUAGES.get(lang, lang.upper())
    buttons.append(
        [
            InlineKeyboardButton(
                f"Sprache: {current_lang_name}"
                if lang == "de"
                else f"Language: {current_lang_name}",
                callback_data="settings_lang_menu",
            )
        ]
    )

    # Reset all: only show when at least one per-slot override exists
    slot_overrides_count = sum(1 for key in overrides if key != "global")
    if slot_overrides_count > 0:
        buttons.append(
            [
                InlineKeyboardButton(
                    f"\U0001f504 {s['reset_all_btn']}",
                    callback_data="settings_reset_all",
                )
            ]
        )

    # Message text: headline for global override in text (not in keyboard)
    if global_override:
        global_display = model_service.get_model_display_name(global_override)
        override_line = s["global_override_text"].format(display_name=global_display)
        text = f"⚙️ {s['main_title']}\n\n{override_line}\n\n{s['models_section']}"
    else:
        text = f"⚙️ {s['main_title']}\n\n{s['models_section']}"
    return text, InlineKeyboardMarkup(buttons)


def build_slot_menu_keyboard(
    slot: TaskSlot,
    user_id: int,
    model_service: ModelService,
    context: ContextTypes.DEFAULT_TYPE,
    lang: str = "de",
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the slot model selection menu (level B).

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    s = _get_settings_strings(lang)

    # Determine effective model for this slot
    # Priority: slot override > global override > slot default
    slot_override = model_service.get_user_model(user_id, slot.value)
    global_override = model_service.get_user_model(user_id, "global")
    default_model_id = _get_slot_default_model(slot, context)
    effective_model = slot_override or global_override or default_model_id

    buttons: list[list[InlineKeyboardButton]] = []

    for alias in _AVAILABLE_ALIASES:
        model_id = resolve_alias(alias)
        if model_id is None:
            continue
        display = model_service.get_model_display_name(model_id)

        if model_id == effective_model:
            marker = s["current_marker"]
        else:
            marker = s["other_marker"]

        buttons.append(
            [
                InlineKeyboardButton(
                    f"{marker} {display}",
                    callback_data=f"settings_model:{slot.value}:{alias}",
                )
            ]
        )

    # Back + Reset
    buttons.append([InlineKeyboardButton(s["back_btn"], callback_data="settings_back")])
    buttons.append(
        [
            InlineKeyboardButton(
                f"\U0001f504 {s['reset_slot_btn']}",
                callback_data=f"settings_reset:{slot.value}",
            )
        ]
    )

    text = f"\U0001f527 {s['slot_select_title'].format(slot=slot.value.upper())}"
    return text, InlineKeyboardMarkup(buttons)


def build_reset_confirm_keyboard(
    lang: str = "de",
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the reset confirmation dialog.

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    s = _get_settings_strings(lang)
    buttons = [
        [
            InlineKeyboardButton(
                s["reset_confirm_yes"], callback_data="settings_reset_all_confirm"
            )
        ],
        [
            InlineKeyboardButton(
                s["reset_confirm_cancel"], callback_data="settings_back"
            )
        ],
    ]
    text = f"⚠️ {s['reset_confirm_title']}"
    return text, InlineKeyboardMarkup(buttons)


def build_lang_menu_keyboard(
    current_lang: str = "de",
    lang: str = "de",
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the language selection menu (grid layout, 4 buttons per row).

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    s = _get_settings_strings(lang)
    buttons: list[list[InlineKeyboardButton]] = []

    # Build grid: 4 buttons per row
    row: list[InlineKeyboardButton] = []
    for code, name in _SETTINGS_LANGUAGES.items():
        if code == current_lang:
            marker = s["current_marker"]
        else:
            marker = s["other_marker"]
        row.append(
            InlineKeyboardButton(
                f"{marker} {name}",
                callback_data=f"settings_lang:{code}",
            )
        )
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append(
        [InlineKeyboardButton(s["lang_back"], callback_data="settings_back")]
    )

    text = f"\U0001f310 {s['lang_title']}"
    return text, InlineKeyboardMarkup(buttons)


# ---------------------------------------------------------------------------
# Callback Handlers
# ---------------------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_settings_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Central callback handler for all settings_* patterns.

    Routes by callback data prefix to the right sub-logic.
    Always edits the existing message (no new sends).
    """
    query = update.callback_query
    data: str = query.data or ""

    if not data.startswith("settings_"):
        return

    await query.answer()

    user = query.from_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    model_service = _get_model_service(context)
    if model_service is None or not isinstance(model_service, ModelService):
        await query.edit_message_text("Settings not available.")
        return

    chat_service = _get_chat_service(context)
    lang = "de"
    if chat_service is not None and hasattr(chat_service, "get_chat_language"):
        lang = await chat_service.get_chat_language(user_id, chat_id) or "de"

    # --- Route by callback data ---

    if data == "settings_noop":
        # Headline button without action (informational only)
        return

    if data == "settings_reset_global":
        # Remove global override
        deleted = model_service.reset_user_model(user_id, slot="global")
        log_command_audit(
            action="settings_reset_global",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"reset global (was_active={deleted})",
        )
        # Back to main menu
        text, keyboard = build_main_menu_keyboard(user_id, model_service, context, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data.startswith("settings_slot:"):
        # Level B: slot model selection
        slot_name = data.split(":")[1]
        slot = TaskSlot.from_string(slot_name)
        if slot is None:
            await query.edit_message_text("Unknown slot.")
            return
        text, keyboard = build_slot_menu_keyboard(
            slot, user_id, model_service, context, lang
        )
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data.startswith("settings_model:"):
        # Set model: settings_model:<slot>:<alias>
        parts = data.split(":")
        if len(parts) < 3:
            return
        slot_name = parts[1]
        alias = parts[2]
        slot = TaskSlot.from_string(slot_name)
        if slot is None:
            return

        success, result = model_service.set_user_model(user_id, alias, slot=slot.value)
        if success:
            was_implicit_reset = model_service.last_was_implicit_reset
            audit_action = (
                "settings_model_implicit_reset"
                if was_implicit_reset
                else "settings_model"
            )
            log.info(
                "Settings: User %d set %s -> %s (%s, implicit_reset=%s)",
                user_id,
                slot.value,
                alias,
                result,
                was_implicit_reset,
            )
            if was_implicit_reset:
                details = (
                    f"implicit_reset slot={slot.value}, was default-equal alias={alias}"
                )
            else:
                details = f"set slot={slot.value} alias={alias} -> {result}"
            log_command_audit(
                action=audit_action,
                user_id=user_id,
                chat_id=chat_id,
                username=user.username if user else None,
                details=details,
            )

        # Back to main menu mit aktualisierten Werten
        text, keyboard = build_main_menu_keyboard(user_id, model_service, context, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data.startswith("settings_reset:"):
        # Reset single slot: settings_reset:<slot>
        slot_name = data.split(":")[1]
        slot = TaskSlot.from_string(slot_name)
        if slot is None:
            return
        deleted = model_service.reset_user_model(user_id, slot=slot.value)
        log_command_audit(
            action="settings_reset_slot",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"reset slot={slot.value} (was_active={deleted})",
        )

        # Back to main menu
        text, keyboard = build_main_menu_keyboard(user_id, model_service, context, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data == "settings_reset_all":
        # Show confirmation dialog
        text, keyboard = build_reset_confirm_keyboard(lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data == "settings_reset_all_confirm":
        # Actually reset all slots
        count = model_service.reset_all_slots(user_id)
        log_command_audit(
            action="settings_reset_all",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"reset all (removed={count})",
        )

        # Back to main menu
        text, keyboard = build_main_menu_keyboard(user_id, model_service, context, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data == "settings_back":
        # Back to main menu
        text, keyboard = build_main_menu_keyboard(user_id, model_service, context, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data == "settings_lang_menu":
        # Show language selection
        text, keyboard = build_lang_menu_keyboard(current_lang=lang, lang=lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    if data.startswith("settings_lang:"):
        # Set language
        new_lang = data.split(":")[1]
        if new_lang not in _SETTINGS_LANGUAGES:
            return

        if chat_service is not None and hasattr(chat_service, "set_chat_language"):
            await chat_service.set_chat_language(user_id, chat_id, new_lang)

        log_command_audit(
            action="settings_lang",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"set lang={new_lang}",
        )
        log.info("Settings: User %d set language to '%s'", user_id, new_lang)

        # Back to main menu (in neuer Sprache)
        text, keyboard = build_main_menu_keyboard(
            user_id, model_service, context, new_lang
        )
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return
