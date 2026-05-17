"""InlineKeyboard callbacks for bookmark buttons.

Processes bm_show and bm_del callback queries
from the /bookmarks view.
"""

from __future__ import annotations

import logging
from datetime import datetime

from telegram import Update
from telegram.ext import ContextTypes

from application.audit_service import log_command_audit
from application.bookmark_service import BookmarkService
from domain.language import DEFAULT_LANGUAGE
from domain.markdown import markdown_to_telegram_html, strip_markdown
from i18n.domain.i18n import t
from presentation.decorators import require_private_chat, require_whitelist
from presentation.render import split_message

log = logging.getLogger(__name__)


@require_whitelist
@require_private_chat
async def handle_bookmark_show_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Shows the full text of a bookmark (button 'Show full text').

    Callback data format: 'bm_show:<chat_id>:<message_id>'.
    Splits plain text first, then converts each chunk to HTML.
    Prevents HTML tags from being split across chunks.
    """
    query = update.callback_query
    data: str = query.data or ""

    if not data.startswith("bm_show:"):
        return

    try:
        parts = data.split(":")
        bm_chat_id = int(parts[1])
        msg_id = int(parts[2])
    except (ValueError, IndexError):
        await query.answer(text=t("errors.invalid_id", "en"), show_alert=False)
        return

    user = query.from_user
    user_id: int = user.id if user else 0

    bookmark_service: BookmarkService = context.application.bot_data.get(
        "bookmark_service"
    )
    if bookmark_service is None:
        await query.answer(
            text=t("errors.bookmark_service_not_initialized", "en"), show_alert=False
        )
        return

    bm = bookmark_service.get_bookmark(user_id, bm_chat_id, msg_id)
    if bm is None:
        await query.answer(text=t("errors.bookmark_not_found", "en"), show_alert=False)
        log_command_audit(
            action="bm_show",
            user_id=user_id,
            chat_id=bm_chat_id,
            username=user.username if user else None,
            entry_id=f"msg_{msg_id}",
            success=False,
            details="not found",
        )
        return

    await query.answer()

    content = bm.get("content", "(no content)")
    # Split plain text first, then convert each chunk to HTML (FIX 7)
    plain_chunks = split_message(content)
    used_fallback = False

    for plain_chunk in plain_chunks:
        if used_fallback:
            await query.message.reply_text(strip_markdown(plain_chunk))
            continue

        html_chunk = markdown_to_telegram_html(plain_chunk)
        try:
            await query.message.reply_text(html_chunk, parse_mode="HTML")
        except Exception:
            used_fallback = True
            await query.message.reply_text(strip_markdown(plain_chunk))

    log_command_audit(
        action="bm_show",
        user_id=user_id,
        chat_id=bm_chat_id,
        username=user.username if user else None,
        entry_id=f"msg_{msg_id}",
    )


@require_whitelist
@require_private_chat
async def handle_bookmark_delete_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Removes a bookmark (button 'Remove').

    Callback data format: 'bm_del:<chat_id>:<message_id>'.
    """
    query = update.callback_query
    data: str = query.data or ""

    if not data.startswith("bm_del:"):
        return

    try:
        parts = data.split(":")
        bm_chat_id = int(parts[1])
        msg_id = int(parts[2])
    except (ValueError, IndexError):
        await query.answer(text=t("errors.invalid_id", "en"), show_alert=False)
        return

    user = query.from_user
    user_id: int = user.id if user else 0

    bookmark_service: BookmarkService = context.application.bot_data.get(
        "bookmark_service"
    )
    if bookmark_service is None:
        await query.answer(
            text=t("errors.bookmark_service_not_initialized", "en"), show_alert=False
        )
        return

    # Get bookmark data BEFORE deleting (for date in confirmation)
    bm_data = bookmark_service.get_bookmark(user_id, bm_chat_id, msg_id)

    deleted: bool = bookmark_service.remove_bookmark(user_id, bm_chat_id, msg_id)
    if deleted:
        await query.answer(text=t("bookmark.removed_toast", "en"), show_alert=False)

        # Format date for chat confirmation
        date_display = ""
        if bm_data:
            ts = bm_data.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    date_display = f" from {dt.strftime('%d.%m.%Y %H:%M')}"
                except (ValueError, TypeError):
                    pass

        from domain.i18n import BOOKMARK_DELETE_CONFIRM_TEXTS, get_text

        # Get user language for i18n
        chat_service = context.application.bot_data.get("chat_service")
        _del_lang = DEFAULT_LANGUAGE
        if chat_service and hasattr(chat_service, "get_chat_language"):
            _del_lang = (
                await chat_service.get_chat_language(user_id, bm_chat_id)
                or DEFAULT_LANGUAGE
            )
        _del_text = get_text(
            BOOKMARK_DELETE_CONFIRM_TEXTS, _del_lang, date_display=date_display
        )
        await query.message.reply_text(f"✓ {_del_text}")  # i18n: ok
        log.info(
            "Bookmark removed via button: user_id=%d message_id=%d", user_id, msg_id
        )
    else:
        await query.answer(text=t("errors.bookmark_not_found", "en"), show_alert=False)
    log_command_audit(
        action="bm_del",
        user_id=user_id,
        chat_id=bm_chat_id,
        username=user.username if user else None,
        entry_id=f"msg_{msg_id}",
        success=deleted,
    )
