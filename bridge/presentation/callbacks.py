"""InlineKeyboard-Callbacks für Bookmark-Buttons.

Verarbeitet bm_show und bm_del Callback-Queries
von der /bookmarks-Ansicht.
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from application.audit_service import log_command_audit
from application.bookmark_service import get_bookmark, remove_bookmark
from domain.markdown import markdown_to_telegram_html, strip_markdown
from presentation.decorators import require_private_chat, require_whitelist
from presentation.render import split_message

log = logging.getLogger(__name__)


@require_whitelist
@require_private_chat
async def handle_bookmark_show_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Zeigt den Volltext eines Bookmarks an (Button 'Volltext anzeigen').

    Callback data format: 'bm_show:<chat_id>:<message_id>'.
    Splittet Plain-Text zuerst, konvertiert dann pro Chunk zu HTML.
    Verhindert dass HTML-Tags zerschnitten werden.
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
        await query.answer(text="Ungültige ID", show_alert=False)
        return

    user = query.from_user
    user_id: int = user.id if user else 0

    bm = get_bookmark(user_id, bm_chat_id, msg_id)
    if bm is None:
        await query.answer(text="Bookmark nicht gefunden", show_alert=False)
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

    content = bm.get("content", "(kein Inhalt)")
    # Split Plain-Text zuerst, dann pro Chunk HTML konvertieren (FIX 7)
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
    """Entfernt einen Bookmark (Button 'Entfernen').

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
        await query.answer(text="Ungültige ID", show_alert=False)
        return

    user = query.from_user
    user_id: int = user.id if user else 0

    deleted: bool = remove_bookmark(user_id, bm_chat_id, msg_id)
    if deleted:
        await query.answer(text="Entfernt", show_alert=False)
        log.info(
            "Bookmark entfernt via Button: user_id=%d message_id=%d", user_id, msg_id
        )
    else:
        await query.answer(text="Bookmark nicht gefunden", show_alert=False)
    log_command_audit(
        action="bm_del",
        user_id=user_id,
        chat_id=bm_chat_id,
        username=user.username if user else None,
        entry_id=f"msg_{msg_id}",
        success=deleted,
    )
