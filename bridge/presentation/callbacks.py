"""InlineKeyboard-Callbacks für Bookmark-Buttons.

Verarbeitet bm_show und bm_del Callback-Queries
von der /bookmarks-Ansicht.
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from application.bookmark_service import get_bookmark, remove_bookmark
from domain.markdown import markdown_to_telegram_html, strip_markdown
from presentation.render import split_message

log = logging.getLogger(__name__)


async def handle_bookmark_show_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Zeigt den Volltext eines Bookmarks an (Button 'Volltext anzeigen').

    Callback data format: 'bm_show:<chat_id>:<message_id>'.
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
        return

    await query.answer()

    content = bm.get("content", "(kein Inhalt)")
    # Volltext in Chunks senden (mit HTML-Konvertierung + Fallback)
    html_content = markdown_to_telegram_html(content)
    for chunk in split_message(html_content):
        try:
            await query.message.reply_text(chunk, parse_mode="HTML")
        except Exception:
            plain_content = strip_markdown(content)
            for plain_chunk in split_message(plain_content):
                await query.message.reply_text(plain_chunk)
            break


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
