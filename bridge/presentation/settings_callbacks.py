"""InlineKeyboard callbacks for the /settings menu.

Two independent menu systems live in this module:

Legacy (slot-based model overrides, /settings v1):
  - settings_slot:<slot>           -> Shows model selection for a slot (level B)
  - settings_model:<slot>:<alias>  -> Sets model override for slot
  - settings_reset:<slot>          -> Resets a slot to default
  - settings_reset_all             -> Shows confirmation dialog
  - settings_reset_all_confirm     -> Resets all slots
  - settings_back                  -> Back to main menu (level A)
  - settings_lang:<code>           -> Sets language
  - settings_lang_menu             -> Shows language selection (level B)

v2 (6 categories, /settings new main menu):
  - settings_v2_main               -> Renders 6-category main menu (level A)
  - settings_v2_close              -> Removes keyboard (close)
  - settings_v2_cat:<category>     -> Opens category sub-menu (level B)
  - settings_v2_model_back         -> Returns from model sub-menu to v2 main
  - settings_v2_lang:<code>        -> Sets language via v2 menu
  - settings_v2_lang_back          -> Returns from lang sub-menu to v2 main
  - settings_v2_debate_toggle:<id> -> Toggles a debate provider
  - settings_v2_rl:<profile>       -> Sets rate limit profile
  - settings_v2_pf:<flag>          -> Toggles a personality flag
  - settings_v2_tz:<iana>          -> Sets timezone
  - settings_v2_tz_other           -> Prompts for manual timezone entry
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from application.audit_service import log_command_audit
from application.model_service import DEFAULT_MODEL, ModelService, resolve_alias
from application.settings_service import (
    COMMON_TIMEZONES,
    DEBATE_PROVIDERS,
    PERSONALITY_FLAGS,
    VALID_RATE_LIMIT_PROFILES,
    SettingsService,
)
from domain.language import DEFAULT_LANGUAGE
from domain.task_slot import TaskSlot
from i18n.domain.i18n import t
from presentation.decorators import require_private_chat, require_whitelist

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# i18n Strings (migrated to JSON-based t() system)
# ---------------------------------------------------------------------------

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
    """Returns settings i18n strings for the given language via t().

    Maps internal keys to the JSON i18n system.
    """
    return {
        "main_title": t("settings.main_title", lang),
        "models_section": t("settings.models_section", lang),
        "lang_section": t("settings.lang_section", lang),
        "reset_all_btn": t("settings.reset_all_btn", lang),
        "default_suffix": t("settings.default_suffix", lang),
        "global_override_suffix": t("settings.global_override_suffix", lang),
        "global_override_text": t(
            "settings.global_override_text", lang, display_name="{display_name}"
        ),
        "reset_global_btn": t("settings.reset_global_btn", lang),
        "slot_select_title": t("settings.slot_select_title", lang, slot="{slot}"),
        "current_marker": "●",
        "other_marker": "○",
        "back_btn": t("settings.back_btn", lang),
        "reset_slot_btn": t("settings.reset_slot_btn", lang),
        "reset_confirm_title": t("settings.reset_confirm_title", lang),
        "reset_confirm_yes": t("settings.reset_confirm_yes", lang),
        "reset_confirm_cancel": t("settings.reset_confirm_cancel", lang),
        "reset_all_done": t("settings.reset_all_done", lang, count="{count}"),
        "reset_all_nothing": t("settings.reset_all_nothing", lang),
        "model_set": t(
            "settings.model_set", lang, slot="{slot}", display_name="{display_name}"
        ),
        "model_reset": t("settings.model_reset", lang, slot="{slot}"),
        "model_reset_nothing": t("settings.model_reset_nothing", lang, slot="{slot}"),
        "lang_title": t("settings.lang_title", lang),
        "lang_back": t("settings.lang_back", lang),
        "lang_set": t("settings.lang_set", lang, name="{name}"),
    }


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
    lang_btn_text = t("settings.lang_button", lang, name=current_lang_name)
    buttons.append(
        [
            InlineKeyboardButton(
                lang_btn_text,
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
        await query.edit_message_text(t("errors.settings_not_available", "en"))
        return

    chat_service = _get_chat_service(context)
    lang = DEFAULT_LANGUAGE
    if chat_service is not None and hasattr(chat_service, "get_chat_language"):
        lang = (
            await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
        )

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
            await query.edit_message_text(t("errors.unknown_slot", "en"))
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

        # Back to main menu with updated values
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

        # Back to main menu (in new language)
        text, keyboard = build_main_menu_keyboard(
            user_id, model_service, context, new_lang
        )
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return


# ===========================================================================
# /settings v2 — 6-category hierarchical menu
# ===========================================================================
# Callback data format: settings_v2_<action>[:<param>]
# Max Telegram callback_data length: 64 bytes — kept well under by compact IDs.
# ===========================================================================

# Personality flag UI config: (column_name, i18n_key, default_on)
_PERSONALITY_UI: list[tuple[str, str, bool]] = [
    ("personality_p1", "settings.personality.p1_label", True),
    ("personality_p2", "settings.personality.p2_label", True),
    ("personality_p3", "settings.personality.p3_label", True),
    ("personality_p4", "settings.personality.p4_label", False),
    ("personality_p5", "settings.personality.p5_label", True),
    ("personality_p6", "settings.personality.p6_label", True),
]


def _get_settings_service(context: ContextTypes.DEFAULT_TYPE) -> SettingsService | None:
    """Gets the SettingsService from bot_data (can be None for backwards compat)."""
    return context.application.bot_data.get("settings_service")


# ---------------------------------------------------------------------------
# v2 Keyboard Builders
# ---------------------------------------------------------------------------


def build_v2_main_menu(lang: str = "de") -> tuple[str, InlineKeyboardMarkup]:
    """Builds the v2 main menu (6 categories, 2-column grid).

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    title = t("settings.title", lang)
    buttons: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                f"🌐 {t('settings.btn_language', lang)}",
                callback_data="settings_v2_cat:language",
            ),
            InlineKeyboardButton(
                f"🤖 {t('settings.btn_model', lang)}",
                callback_data="settings_v2_cat:model",
            ),
        ],
        [
            InlineKeyboardButton(
                f"💬 {t('settings.btn_debate', lang)}",
                callback_data="settings_v2_cat:debate",
            ),
            InlineKeyboardButton(
                f"⏱ {t('settings.btn_rate_limit', lang)}",  # i18n: ok
                callback_data="settings_v2_cat:rate_limit",
            ),
        ],
        [
            InlineKeyboardButton(
                f"🎭 {t('settings.btn_personality', lang)}",
                callback_data="settings_v2_cat:personality",
            ),
            InlineKeyboardButton(
                f"⏰ {t('settings.btn_timezone', lang)}",  # i18n: ok
                callback_data="settings_v2_cat:timezone",
            ),
        ],
        [
            InlineKeyboardButton(
                t("settings.btn_close", lang),
                callback_data="settings_v2_close",
            )
        ],
    ]
    text = f"⚙️ <b>{title}</b>"
    return text, InlineKeyboardMarkup(buttons)


def build_v2_language_menu(
    current_lang: str = "de", lang: str = "de"
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the v2 language selection sub-menu (4 per row).

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for code, name in _SETTINGS_LANGUAGES.items():
        marker = "●" if code == current_lang else "○"
        row.append(
            InlineKeyboardButton(
                f"{marker} {name}", callback_data=f"settings_v2_lang:{code}"
            )
        )
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append(
        [
            InlineKeyboardButton(
                f"< {t('settings.btn_back', lang)}",
                callback_data="settings_v2_lang_back",
            )
        ]
    )
    text = f"🌐 <b>{t('settings.btn_language', lang)}</b>"
    return text, InlineKeyboardMarkup(buttons)


def build_v2_model_menu(
    user_id: int,
    model_service: ModelService,
    context: ContextTypes.DEFAULT_TYPE,
    lang: str = "de",
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the v2 model selection sub-menu (reuses legacy slot menu rows).

    Embeds the existing slot-based model selection, adding only a back button
    that leads to the v2 main menu instead of the old main menu.

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    # Delegate keyboard building to legacy builder for slot rows
    _legacy_text, legacy_keyboard = build_main_menu_keyboard(
        user_id, model_service, context, lang
    )
    # Replace all legacy "settings_back" or "settings_lang_menu" buttons with v2 back
    rows: list[list[InlineKeyboardButton]] = []
    for row in legacy_keyboard.inline_keyboard:
        new_row = []
        for btn in row:
            if btn.callback_data in ("settings_lang_menu",):
                # Skip the old language button (handled via v2 language sub-menu)
                continue
            new_row.append(btn)
        if new_row:
            rows.append(new_row)
    # Add v2 back button at the bottom
    rows.append(
        [
            InlineKeyboardButton(
                f"< {t('settings.btn_back', lang)}",
                callback_data="settings_v2_model_back",
            )
        ]
    )
    current_model = model_service.get_effective_model(user_id)
    display = model_service.get_model_display_name(current_model)
    title = t("settings.model.title", lang, current=display)
    text = f"🤖 <b>{title}</b>"
    return text, InlineKeyboardMarkup(rows)


def build_v2_debate_menu(
    active_providers: tuple[str, ...], lang: str = "de"
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the debate provider multi-select sub-menu.

    Active providers show ☑/☐. Planned providers show as greyed label.
    Tap on planned provider sends a toast (handled via callback).

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for provider in DEBATE_PROVIDERS:
        pid: str = str(provider["id"])
        label: str = str(provider["label"])
        is_active_provider: bool = bool(provider["active"])
        if is_active_provider:
            checked = pid in active_providers
            marker = "☑" if checked else "☐"
            btn = InlineKeyboardButton(
                f"{marker} {label}", callback_data=f"settings_v2_debate_toggle:{pid}"
            )
        else:
            # Planned: show with indicator, tap = toast
            btn = InlineKeyboardButton(
                f"○ {label}", callback_data=f"settings_v2_debate_planned:{pid}"
            )
        row.append(btn)
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append(
        [
            InlineKeyboardButton(
                f"< {t('settings.btn_back', lang)}",
                callback_data="settings_v2_lang_back",  # reuses v2_main route
            )
        ]
    )
    title = t("settings.debate.title", lang)
    help_text = t("settings.debate.help_multi", lang)
    text = f"💬 <b>{title}</b>\n{help_text}"
    return text, InlineKeyboardMarkup(buttons)


def build_v2_rate_limit_menu(
    current_profile: str, lang: str = "de"
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the rate limit profile single-select sub-menu.

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    profile_labels: dict[str, str] = {
        "light": "Light",
        "normal": "Normal",
        "power": "Power",
        "unlimited": "Unlimited",
    }
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for profile_id, profile_label in profile_labels.items():
        marker = "●" if profile_id == current_profile else "○"
        row.append(
            InlineKeyboardButton(
                f"{marker} {profile_label}",
                callback_data=f"settings_v2_rl:{profile_id}",
            )
        )
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append(
        [
            InlineKeyboardButton(
                f"< {t('settings.btn_back', lang)}",
                callback_data="settings_v2_lang_back",
            )
        ]
    )
    title = t("settings.rate_limit.title", lang, current=current_profile)
    text = f"⏱ <b>{title}</b>"
    return text, InlineKeyboardMarkup(buttons)


def build_v2_personality_menu(
    settings_row: dict[str, Any] | None, lang: str = "de"
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the personality feature multi-toggle sub-menu (1 per row).

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    row_data = settings_row or {}
    buttons: list[list[InlineKeyboardButton]] = []
    for flag, i18n_key, default_on in _PERSONALITY_UI:
        current_val = row_data.get(flag)
        if current_val is None:
            is_on = default_on
        else:
            is_on = bool(current_val)
        marker = "☑" if is_on else "☐"
        label = t(i18n_key, lang)
        buttons.append(
            [
                InlineKeyboardButton(
                    f"{marker} {label}",
                    callback_data=f"settings_v2_pf:{flag}",
                )
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton(
                f"< {t('settings.btn_back', lang)}",
                callback_data="settings_v2_lang_back",
            )
        ]
    )
    title = t("settings.personality.title", lang)
    text = f"🎭 <b>{title}</b>"
    return text, InlineKeyboardMarkup(buttons)


def build_v2_timezone_menu(
    current_tz: str, lang: str = "de"
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds the timezone sub-menu (top-20 + other button).

    Returns:
        Tuple (message_text, keyboard_markup).
    """
    buttons: list[list[InlineKeyboardButton]] = []
    for tz in COMMON_TIMEZONES:
        marker = "●" if tz == current_tz else "○"
        # Use compact callback to stay under 64 bytes: tz can be up to ~30 chars
        buttons.append(
            [
                InlineKeyboardButton(
                    f"{marker} {tz}",
                    callback_data=f"settings_v2_tz:{tz}",
                )
            ]
        )
    # "Other" button -> prompts user to type /settz <iana>
    buttons.append(
        [
            InlineKeyboardButton(
                t("settings.timezone.other", lang),
                callback_data="settings_v2_tz_other",
            )
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(
                f"< {t('settings.btn_back', lang)}",
                callback_data="settings_v2_lang_back",
            )
        ]
    )
    title = t("settings.timezone.title", lang, current=current_tz)
    text = f"⏰ <b>{title}</b>"
    return text, InlineKeyboardMarkup(buttons)


# ---------------------------------------------------------------------------
# v2 Central Callback Handler
# ---------------------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_settings_v2_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Central callback handler for all settings_v2_* patterns.

    Routes by callback data prefix. Always edits the existing message.
    """
    query = update.callback_query
    data: str = query.data or ""

    if not data.startswith("settings_v2_"):
        return

    await query.answer()

    user = query.from_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    model_service = _get_model_service(context)
    chat_service = _get_chat_service(context)
    settings_service = _get_settings_service(context)

    # Resolve current language
    lang = DEFAULT_LANGUAGE
    if chat_service is not None and hasattr(chat_service, "get_chat_language"):
        lang = (
            await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
        )

    # --- Close ---
    if data == "settings_v2_close":
        await query.edit_message_reply_markup(reply_markup=None)
        return

    # --- Main menu re-render ---
    if data == "settings_v2_main":
        text, keyboard = build_v2_main_menu(lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Category navigation ---
    if data.startswith("settings_v2_cat:"):
        category = data.split(":")[1]
        if category == "language":
            text, keyboard = build_v2_language_menu(current_lang=lang, lang=lang)
            await query.edit_message_text(
                text, reply_markup=keyboard, parse_mode="HTML"
            )
        elif category == "model":
            if model_service is None or not isinstance(model_service, ModelService):
                await query.answer(
                    t("errors.settings_not_available", lang), show_alert=True
                )
                return
            text, keyboard = build_v2_model_menu(user_id, model_service, context, lang)
            await query.edit_message_text(
                text, reply_markup=keyboard, parse_mode="HTML"
            )
        elif category == "debate":
            settings = (
                await settings_service.get_settings(user_id)
                if settings_service
                else None
            )
            active = settings.debate_providers if settings else ()
            text, keyboard = build_v2_debate_menu(active, lang)
            await query.edit_message_text(
                text, reply_markup=keyboard, parse_mode="HTML"
            )
        elif category == "rate_limit":
            settings = (
                await settings_service.get_settings(user_id)
                if settings_service
                else None
            )
            profile = settings.rate_limit_profile if settings else "normal"
            text, keyboard = build_v2_rate_limit_menu(profile, lang)
            await query.edit_message_text(
                text, reply_markup=keyboard, parse_mode="HTML"
            )
        elif category == "personality":
            raw_row = (
                settings_service._storage.get_settings_row(user_id)
                if settings_service
                else None
            )
            text, keyboard = build_v2_personality_menu(raw_row, lang)
            await query.edit_message_text(
                text, reply_markup=keyboard, parse_mode="HTML"
            )
        elif category == "timezone":
            settings = (
                await settings_service.get_settings(user_id)
                if settings_service
                else None
            )
            current_tz = settings.timezone if settings else "UTC"
            text, keyboard = build_v2_timezone_menu(current_tz, lang)
            await query.edit_message_text(
                text, reply_markup=keyboard, parse_mode="HTML"
            )
        return

    # --- Language set via v2 menu ---
    if data.startswith("settings_v2_lang:"):
        new_lang = data.split(":")[1]
        if new_lang not in _SETTINGS_LANGUAGES:
            return
        if chat_service is not None and hasattr(chat_service, "set_chat_language"):
            await chat_service.set_chat_language(user_id, chat_id, new_lang)
        if settings_service is not None:
            await settings_service.set_language(user_id, new_lang)
        log_command_audit(
            action="settings_v2_lang",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"lang={new_lang}",
        )
        # Show updated language sub-menu in new language
        text, keyboard = build_v2_language_menu(current_lang=new_lang, lang=new_lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Language back (also used by other sub-menus as generic back-to-main) ---
    if data == "settings_v2_lang_back":
        text, keyboard = build_v2_main_menu(lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Model back ---
    if data == "settings_v2_model_back":
        text, keyboard = build_v2_main_menu(lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Debate provider toggle ---
    if data.startswith("settings_v2_debate_toggle:"):
        provider_id = data.split(":")[1]
        if settings_service is None:
            await query.answer(
                t("errors.settings_not_available", lang), show_alert=True
            )
            return
        try:
            updated = await settings_service.toggle_debate_provider(
                user_id, provider_id
            )
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        log_command_audit(
            action="settings_v2_debate_toggle",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"provider={provider_id} active={provider_id in updated}",
        )
        text, keyboard = build_v2_debate_menu(updated, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Planned provider tap (info toast only) ---
    if data.startswith("settings_v2_debate_planned:"):
        toast_msg = t("settings.toast_provider_planned", lang)
        await query.answer(toast_msg, show_alert=True)
        return

    # --- Rate limit profile set ---
    if data.startswith("settings_v2_rl:"):
        profile = data.split(":")[1]
        if profile not in VALID_RATE_LIMIT_PROFILES:
            return
        if settings_service is not None:
            await settings_service.set_rate_limit(user_id, profile)
        log_command_audit(
            action="settings_v2_rate_limit",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"profile={profile}",
        )
        text, keyboard = build_v2_rate_limit_menu(profile, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Personality flag toggle ---
    if data.startswith("settings_v2_pf:"):
        flag = data.split(":")[1]
        if flag not in PERSONALITY_FLAGS:
            return
        if settings_service is None:
            await query.answer(
                t("errors.settings_not_available", lang), show_alert=True
            )
            return
        # Read current state and invert
        row = settings_service._storage.get_settings_row(user_id)
        _PERSONALITY_DEFAULTS_MAP = {
            "personality_p1": True,
            "personality_p2": True,
            "personality_p3": True,
            "personality_p4": False,
            "personality_p5": True,
            "personality_p6": True,
        }
        current_val = (row or {}).get(flag)
        current_bool = (
            current_val if current_val is not None else _PERSONALITY_DEFAULTS_MAP[flag]
        )
        new_bool = not bool(current_bool)
        await settings_service.toggle_personality(user_id, flag, new_bool)
        log_command_audit(
            action="settings_v2_personality",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"flag={flag} on={new_bool}",
        )
        # Refresh personality menu
        updated_row = settings_service._storage.get_settings_row(user_id)
        text, keyboard = build_v2_personality_menu(updated_row, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Timezone set ---
    if data.startswith("settings_v2_tz:"):
        # Extract tz: everything after first "settings_v2_tz:"
        tz = data[len("settings_v2_tz:") :]
        if settings_service is not None:
            await settings_service.set_timezone(user_id, tz)
        log_command_audit(
            action="settings_v2_timezone",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"tz={tz}",
        )
        text, keyboard = build_v2_timezone_menu(tz, lang)
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode="HTML")
        return

    # --- Timezone "other" -> power-user prompt ---
    if data == "settings_v2_tz_other":
        await query.answer(
            t("settings.timezone.search", lang) + " /settz <iana>",
            show_alert=True,
        )
        return
