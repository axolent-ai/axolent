"""Telegram-Rendering: Chunking + HTML/Plain-Text-Fallback.

Verantwortlich für das sichere Senden von Antworten an Telegram.
Splittet Plain-Text ZUERST, konvertiert dann pro Chunk zu HTML.
Verhindert dass HTML-Tags zerschnitten werden.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock

from telegram import Message, Update

from domain.markdown import markdown_to_telegram_html, strip_markdown

log = logging.getLogger(__name__)

TELEGRAM_CHUNK_SIZE: int = 4000

# In-memory Cache: maps (chat_id, message_id) -> full response text für Bookmark-Saving.
# LRU-bounded via OrderedDict, thread-safe via Lock.
_response_cache: OrderedDict[tuple[int, int], str] = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_MAX = 500


def cache_response(chat_id: int, message_id: int, response: str) -> None:
    """Cached eine Bot-Antwort, entfernt älteste Einträge bei Überschreitung von _CACHE_MAX.

    Args:
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID der gesendeten Antwort.
        response: Volltext der Antwort (Markdown, vor Konvertierung).
    """
    with _CACHE_LOCK:
        _response_cache[(chat_id, message_id)] = response
        while len(_response_cache) > _CACHE_MAX:
            _response_cache.popitem(last=False)


def get_cached_response(chat_id: int, message_id: int) -> str | None:
    """Holt eine gecachte Bot-Antwort per (chat_id, message_id).

    Args:
        chat_id: Telegram Chat-ID.
        message_id: Telegram Message-ID.

    Returns:
        Gecachter Antworttext oder None wenn nicht im Cache.
    """
    with _CACHE_LOCK:
        return _response_cache.get((chat_id, message_id))


def split_message(text: str, chunk_size: int = TELEGRAM_CHUNK_SIZE) -> list[str]:
    """Teilt einen langen Text in Chunks für Telegram (max 4096 Zeichen).

    Args:
        text: Der vollständige Antworttext.
        chunk_size: Maximale Zeichenanzahl pro Chunk.

    Returns:
        Liste von Text-Chunks.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]
    return chunks


async def send_response(update: Update, response: str) -> Message | None:
    """Sendet eine Claude-Antwort als HTML-Chunks an Telegram mit Plain-Text-Fallback.

    Flow:
        1. Plain-Text in Chunks splitten (vor HTML-Konvertierung)
        2. Pro Chunk: Markdown -> HTML konvertieren
        3. HTML senden, bei Fehler auf Plain-Text wechseln
        4. Letzten gesendeten Message cachen für Bookmark-Zugriff

    Args:
        update: Telegram Update (enthält die User-Nachricht).
        response: Vollständige Claude-Antwort (Markdown).

    Returns:
        Die letzte gesendete Telegram-Message (für Cache), oder None.
    """
    plain_chunks = split_message(response, chunk_size=TELEGRAM_CHUNK_SIZE - 200)
    last_sent_message: Message | None = None
    sent_ids: list[int] = []
    used_fallback = False
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    for i, plain_chunk in enumerate(plain_chunks):
        if used_fallback:
            # Im Fallback-Modus: nur Plain-Text senden
            last_sent_message = await update.message.reply_text(
                strip_markdown(plain_chunk)
            )
            if last_sent_message is not None:
                sent_ids.append(last_sent_message.message_id)
            continue

        html_chunk = markdown_to_telegram_html(plain_chunk)
        try:
            last_sent_message = await update.message.reply_text(
                html_chunk, parse_mode="HTML"
            )
            if last_sent_message is not None:
                sent_ids.append(last_sent_message.message_id)
        except Exception as html_err:
            log.warning(
                "HTML-Send fehlgeschlagen für Chunk %d/%d, wechsle auf Plain-Text: %s",
                i + 1,
                len(plain_chunks),
                html_err,
            )
            used_fallback = True
            # Diesen Chunk als Plain-Text senden
            last_sent_message = await update.message.reply_text(
                strip_markdown(plain_chunk)
            )
            if last_sent_message is not None:
                sent_ids.append(last_sent_message.message_id)

    # Cache ALL sent message IDs -> full response for bookmark retrieval via /save
    if sent_ids:
        with _CACHE_LOCK:
            for msg_id in sent_ids:
                _response_cache[(chat_id, msg_id)] = response
            while len(_response_cache) > _CACHE_MAX:
                _response_cache.popitem(last=False)

    if used_fallback:
        log.info("Antwort via Plain-Text-Fallback gesendet")

    return last_sent_message
