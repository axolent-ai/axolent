"""Telegram rendering: chunking + HTML/plain-text fallback.

Responsible for safely sending responses to Telegram.
Splits plain text FIRST, then converts each chunk to HTML.
Prevents HTML tags from being split across chunks.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock

from telegram import Message, Update

import re

from domain.markdown import markdown_to_telegram_html, strip_markdown

log = logging.getLogger(__name__)

_SLASH_BEFORE_LETTER = re.compile(r"/(?=[a-zA-Z])")


def sanitize_telegram_slashes(text: str) -> str:
    """Replace ``/`` directly before a letter with U+2044 fraction slash.

    Telegram interprets ``/word`` as a clickable bot command link.
    This function replaces the slash only when followed by a letter,
    leaving standalone slashes untouched.
    """
    return _SLASH_BEFORE_LETTER.sub("⁄", text)


TELEGRAM_CHUNK_SIZE: int = 4000

# In-memory cache: maps (chat_id, message_id) -> full response text for bookmark saving.
# LRU-bounded via OrderedDict, thread-safe via Lock.
_response_cache: OrderedDict[tuple[int, int], str] = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_MAX = 500


def cache_response(chat_id: int, message_id: int, response: str) -> None:
    """Caches a bot response, evicts oldest entries when exceeding _CACHE_MAX.

    Args:
        chat_id: Telegram chat ID.
        message_id: Telegram message ID of the sent response.
        response: Full response text (Markdown, before conversion).
    """
    with _CACHE_LOCK:
        _response_cache[(chat_id, message_id)] = response
        while len(_response_cache) > _CACHE_MAX:
            _response_cache.popitem(last=False)


def get_cached_response(chat_id: int, message_id: int) -> str | None:
    """Retrieves a cached bot response by (chat_id, message_id).

    Args:
        chat_id: Telegram chat ID.
        message_id: Telegram message ID.

    Returns:
        Cached response text or None if not in cache.
    """
    with _CACHE_LOCK:
        return _response_cache.get((chat_id, message_id))


def split_message(text: str, chunk_size: int = TELEGRAM_CHUNK_SIZE) -> list[str]:
    """Splits a long text into chunks for Telegram (max 4096 characters).

    Args:
        text: The full response text.
        chunk_size: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]
    return chunks


async def send_response(update: Update, response: str) -> Message | None:
    """Sends a Claude response as HTML chunks to Telegram with plain-text fallback.

    Flow:
        1. Split plain text into chunks (before HTML conversion)
        2. Per chunk: convert Markdown -> HTML
        3. Send HTML, fall back to plain text on error
        4. Cache last sent message for bookmark access

    Args:
        update: Telegram Update (contains the user message).
        response: Full Claude response (Markdown).

    Returns:
        The last sent Telegram Message (for cache), or None.
    """
    response = sanitize_telegram_slashes(response)
    plain_chunks = split_message(response, chunk_size=TELEGRAM_CHUNK_SIZE - 200)
    last_sent_message: Message | None = None
    sent_ids: list[int] = []
    used_fallback = False
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    for i, plain_chunk in enumerate(plain_chunks):
        if used_fallback:
            # In fallback mode: send plain text only
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
                "HTML send failed for chunk %d/%d, switching to plain text: %s",
                i + 1,
                len(plain_chunks),
                html_err,
            )
            used_fallback = True
            # Send this chunk as plain text
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
        log.info("Response sent via plain-text fallback")

    return last_sent_message
