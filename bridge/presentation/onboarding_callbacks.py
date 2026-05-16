"""InlineKeyboard callbacks for the setup wizard (onboarding).

Processes wizard_* callback queries:
  - wizard_lang:<code>     -> Language chosen, proceed to step 2
  - wizard_lang_auto       -> Auto-detect chosen, proceed to step 2
  - wizard_skip            -> Wizard skipped
  - wizard_done            -> Step 2 completed, user onboarded
"""

from __future__ import annotations

import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from application.audit_service import log_command_audit
from domain.onboarding import (
    LANGUAGE_KEYBOARD_ROWS,
    VALID_LANGUAGE_CODES,
    WIZARD_LANGUAGES,
    get_language_name,
)
from i18n.domain.i18n import t
from presentation.decorators import require_private_chat, require_whitelist

log = logging.getLogger(__name__)


def _get_onboarding_storage(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Gets the OnboardingStorage from bot_data."""
    return context.application.bot_data.get("onboarding_storage")


def _get_chat_service(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Gets the ChatService from bot_data."""
    return context.application.bot_data.get("chat_service")


# ---------------------------------------------------------------------------
# Keyboard Builders
# ---------------------------------------------------------------------------


def build_language_keyboard(ui_lang: str = "de") -> InlineKeyboardMarkup:
    """Builds the Step 1 language selection keyboard.

    Layout: 5 rows of 4 buttons + auto-detect row + skip row.

    Args:
        ui_lang: Language for UI text (auto-detect label, skip label).

    Returns:
        InlineKeyboardMarkup with language buttons.
    """
    buttons: list[list[InlineKeyboardButton]] = []

    for row_codes in LANGUAGE_KEYBOARD_ROWS:
        row: list[InlineKeyboardButton] = []
        for code in row_codes:
            name = WIZARD_LANGUAGES.get(code, code)
            row.append(InlineKeyboardButton(name, callback_data=f"wizard_lang:{code}"))
        buttons.append(row)

    # Auto-detect row (full width)
    auto_text = f"🌐 {t('onboarding.auto_detect', ui_lang)}"
    buttons.append([InlineKeyboardButton(auto_text, callback_data="wizard_lang_auto")])

    # Skip row
    skip_text = t("onboarding.skip", ui_lang)
    buttons.append([InlineKeyboardButton(skip_text, callback_data="wizard_skip")])

    return InlineKeyboardMarkup(buttons)


def build_completion_keyboard(lang: str = "de") -> InlineKeyboardMarkup:
    """Builds the Step 2 completion keyboard.

    Single button: "Let's go" / "Los geht's".

    Args:
        lang: Language for button text.

    Returns:
        InlineKeyboardMarkup with the completion button.
    """
    text = t("onboarding.lets_go", lang)
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"🚀 {text}", callback_data="wizard_done")]]
    )


# ---------------------------------------------------------------------------
# Public: Start wizard (called from handlers.py)
# ---------------------------------------------------------------------------


async def start_wizard(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    is_restart: bool = False,
) -> None:
    """Sends the Step 1 language selection message.

    Args:
        update: Telegram Update.
        context: Telegram Context.
        is_restart: True if triggered by /onboarding (manual restart).
    """
    user = update.effective_user
    user_id: int = user.id if user else 0

    # Determine UI language (try sticky language, fallback to "de")
    chat_service = _get_chat_service(context)
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    ui_lang = "de"
    if chat_service and hasattr(chat_service, "get_chat_language"):
        stored_lang = await chat_service.get_chat_language(user_id, chat_id)
        if stored_lang in ("de", "en"):
            ui_lang = stored_lang

    step1_text = t("onboarding.step1", ui_lang)
    keyboard = build_language_keyboard(ui_lang)

    prefix = ""
    if is_restart:
        prefix = "🔄 "

    await update.message.reply_text(
        f"{prefix}{step1_text}",
        reply_markup=keyboard,
    )

    log_command_audit(
        action="wizard_start",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details=f"is_restart={is_restart}",
    )


# ---------------------------------------------------------------------------
# Callback Handler
# ---------------------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_wizard_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Central callback handler for all wizard_* patterns.

    Routes by callback data prefix to the right sub-logic.
    """
    query = update.callback_query
    data: str = query.data or ""

    if not data.startswith("wizard_"):
        return

    await query.answer()

    user = query.from_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    onboarding_storage = _get_onboarding_storage(context)
    if onboarding_storage is None:
        await query.edit_message_text(t("errors.onboarding_not_available", "en"))
        return

    chat_service = _get_chat_service(context)

    # --- wizard_lang:<code> ---
    if data.startswith("wizard_lang:"):
        lang_code = data.split(":")[1]
        if lang_code not in VALID_LANGUAGE_CODES:
            return

        # Save language preference
        onboarding_storage.set_wizard_lang(user_id, lang_code)

        # Set sticky language for chat
        if chat_service and hasattr(chat_service, "set_chat_language"):
            await chat_service.set_chat_language(user_id, chat_id, lang_code)

        # Show Step 2
        lang_name = get_language_name(lang_code)
        step2_text = t("onboarding.step2", lang_code, lang_name=lang_name)
        keyboard = build_completion_keyboard(lang_code)

        await query.edit_message_text(step2_text, reply_markup=keyboard)

        log_command_audit(
            action="wizard_lang_select",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"lang={lang_code}",
        )
        return

    # --- wizard_lang_auto ---
    if data == "wizard_lang_auto":
        # Save as "auto" (Smart-Language-Detection stays active)
        onboarding_storage.set_wizard_lang(user_id, "auto")

        # Don't set sticky language (keep auto-detection active)

        # Show Step 2 in German (default for auto)
        lang_name = get_language_name("auto")
        step2_text = t("onboarding.step2", "de", lang_name=lang_name)
        keyboard = build_completion_keyboard("de")

        await query.edit_message_text(step2_text, reply_markup=keyboard)

        log_command_audit(
            action="wizard_lang_select",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details="lang=auto",
        )
        return

    # --- wizard_skip ---
    if data == "wizard_skip":
        # Check if user has already chosen a language (step 1 completed)
        state = onboarding_storage.get_state(user_id)
        if state and state.wizard_lang:
            # Skip from Step 2: mark as onboarded, keep language
            skip_lang = state.wizard_lang if state.wizard_lang != "auto" else "de"
            onboarding_storage.set_onboarded(user_id, state.wizard_lang)
            skip_text = t("onboarding.skip_step2", skip_lang)
            await query.edit_message_text(f"✓ {skip_text}")  # i18n: ok
            log_command_audit(
                action="wizard_skip_step2",
                user_id=user_id,
                chat_id=chat_id,
                username=user.username if user else None,
            )
        else:
            # Skip from Step 1: do NOT mark as onboarded
            # UI lang from sticky language or fallback
            skip_lang = "de"
            if chat_service and hasattr(chat_service, "get_chat_language"):
                stored = await chat_service.get_chat_language(user_id, chat_id)
                if stored:
                    skip_lang = stored
            skip_text = t("onboarding.skip_step1", skip_lang)
            await query.edit_message_text(skip_text)
            log_command_audit(
                action="wizard_skip_step1",
                user_id=user_id,
                chat_id=chat_id,
                username=user.username if user else None,
            )
        return

    # --- wizard_done ---
    if data == "wizard_done":
        # Step 2 completed: mark as onboarded
        state = onboarding_storage.get_state(user_id)
        lang = state.wizard_lang if state else None
        onboarding_storage.set_onboarded(user_id, lang)

        done_lang = lang if lang and lang != "auto" else "de"
        done_text = t("onboarding.done", done_lang)
        await query.edit_message_text(f"✓ {done_text}")  # i18n: ok

        log_command_audit(
            action="wizard_done",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"lang={lang}",
        )
        return
