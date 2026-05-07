"""Telegram-Handler: Message-, Command- und Callback-Handler.

Alle Telegram-spezifischen Handler die auf User-Input reagieren.
Nutzt application-Layer für Business-Logik, presentation/render für Output.
"""

from __future__ import annotations

import asyncio
import logging
import time
from threading import Lock
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from application.audit_service import log_command_audit
from application.bookmark_service import (
    list_bookmarks,
    save_or_toggle_bookmark,
    search,
)
from application.chat_service import ChatService
from domain.bookmark import format_bookmark_preview
from presentation.decorators import require_private_chat, require_whitelist
from presentation.render import (
    get_cached_response,
    send_response,
    split_message,
)

if TYPE_CHECKING:
    from application.memory_service import MemoryService

log = logging.getLogger(__name__)

# Typing-Keepalive: Telegram zeigt Typing ca. 5s, wir triggern alle 4s neu
TYPING_KEEPALIVE_INTERVAL_SECONDS: float = 4.0


# Concurrency Controls: max 4 Claude-Prozesse global, max 1 pro User
GLOBAL_CLAUDE_SEMAPHORE = asyncio.Semaphore(4)
_user_locks: dict[int, tuple[asyncio.Lock, float]] = {}
_user_locks_meta_lock = Lock()
_USER_LOCK_TTL_SECONDS = 3600  # 1h ohne Aktivität -> entfernt

# Supported languages for /lang command
_SUPPORTED_LANGUAGES: set[str] = {
    "de",
    "en",
    "es",
    "fr",
    "it",
    "pt",
    "nl",
    "pl",
    "ru",
    "ja",
    "ko",
    "zh",
}


async def _typing_keepalive(
    chat: Any, interval: float = TYPING_KEEPALIVE_INTERVAL_SECONDS
) -> None:
    """Sendet Typing-Indicator periodisch bis der Task gecancelled wird.

    Läuft als Background-Task parallel zum LLM-Call. Telegram zeigt den
    Typing-Indicator nur ca. 5 Sekunden, daher triggern wir alle 4s neu.
    Bei Telegram-API-Fehlern (Network-Hickup etc.) wird leise weitergemacht
    oder beendet, niemals raised.

    Args:
        chat: Telegram Chat-Objekt mit send_chat_action-Methode.
        interval: Sekunden zwischen Re-Triggers (Default: 4.0).
    """
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await chat.send_chat_action(ChatAction.TYPING)
            except Exception as exc:
                log.debug("Typing-Keepalive ignoriert Fehler: %s", exc)
    except asyncio.CancelledError:
        pass


def _get_user_lock(user_id: int) -> asyncio.Lock:
    """Gibt den pro-User Lock zurück (lazy init mit TTL-Cleanup)."""
    now = time.monotonic()
    with _user_locks_meta_lock:
        # Stale Locks entfernen (nur wenn nicht gehalten)
        stale = [
            uid
            for uid, (lock, ts) in _user_locks.items()
            if now - ts > _USER_LOCK_TTL_SECONDS and not lock.locked()
        ]
        for uid in stale:
            del _user_locks[uid]
        if user_id not in _user_locks:
            _user_locks[user_id] = (asyncio.Lock(), now)
        else:
            lock, _ = _user_locks[user_id]
            _user_locks[user_id] = (lock, now)
        return _user_locks[user_id][0]


def _get_chat_service(context: ContextTypes.DEFAULT_TYPE) -> ChatService:
    """Holt den ChatService aus bot_data.

    Args:
        context: Telegram-Handler-Context.

    Returns:
        ChatService-Instanz.

    Raises:
        RuntimeError: Wenn ChatService nicht in bot_data ist.
    """
    svc = context.application.bot_data.get("chat_service")
    if svc is None:
        raise RuntimeError(
            "ChatService nicht in bot_data. main.py muss ChatService initialisieren."
        )
    return svc


def _get_system_prompt(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Holt den System-Prompt aus bot_data.

    Args:
        context: Telegram-Handler-Context.

    Returns:
        System-Prompt-String.

    Raises:
        RuntimeError: Wenn system_prompt nicht in bot_data ist.
    """
    prompt = context.application.bot_data.get("system_prompt")
    if prompt is None:
        raise RuntimeError(
            "system_prompt nicht in bot_data. main.py muss system_prompt setzen."
        )
    return prompt


def _get_memory_service(
    context: ContextTypes.DEFAULT_TYPE,
) -> "MemoryService | None":
    """Holt den MemoryService aus bot_data (kann None sein).

    Args:
        context: Telegram-Handler-Context.

    Returns:
        MemoryService-Instanz oder None.
    """
    return context.application.bot_data.get("memory_service")


def build_bookmarks_keyboard(bookmarks: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    """Baut ein InlineKeyboard für die /bookmarks-Auflistung.

    Jeder Bookmark bekommt zwei Buttons: 'Volltext' und 'Entfernen'.

    Args:
        bookmarks: Liste von Bookmark-Dicts mit 'message_id' und 'chat_id'.

    Returns:
        InlineKeyboardMarkup mit zwei Buttons pro Bookmark-Zeile.
    """
    rows: list[list[InlineKeyboardButton]] = []
    for i, bm in enumerate(bookmarks, 1):
        msg_id: int = bm.get("message_id", 0)
        bm_chat_id: int = bm.get("chat_id", 0)
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"#{i} Volltext",
                    callback_data=f"bm_show:{bm_chat_id}:{msg_id}",
                ),
                InlineKeyboardButton(
                    text=f"#{i} Entfernen",
                    callback_data=f"bm_del:{bm_chat_id}:{msg_id}",
                ),
            ]
        )
    return InlineKeyboardMarkup(rows)


HELP_TEXT: str = (
    "Verfügbare Commands:\n\n"
    "/save  (als Antwort auf Bot-Nachricht)  Bookmark speichern/entfernen\n"
    "/bookmarks  Gespeicherte Bookmarks anzeigen\n"
    "/bookmarks search <Begriff>  Bookmarks durchsuchen\n"
    "/remember <text>  Etwas im Langzeitgedächtnis speichern\n"
    "/memory  Letzte Memory-Einträge anzeigen\n"
    "/memory search <Begriff>  Memory durchsuchen\n"
    "/forget <id>  Memory-Eintrag löschen\n"
    "/reset  Konversation zurücksetzen (neuer Chat)\n"
    "/lang <code>  Sprache wechseln (de, en, es, fr, ...)\n"
    "/help  Diese Hilfe anzeigen"
)

START_TEXT: str = (
    "Jarvis-LITE Bridge ist bereit.\n\n"
    "Schick mir eine Frage und ich leite sie an Claude weiter.\n\n"
    "Tipp: Du kannst Bot-Nachrichten als Bookmark speichern. "
    "Antworte einfach mit /save."
)


@require_whitelist
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Verarbeitet eingehende Telegram-Nachrichten via Claude Code Subprozess.

    Flow:
        1. Whitelist-Check (via Decorator)
        2. Typing-Indicator senden
        3. Pro-User Lock + globale Semaphore
        4. Claude CLI aufrufen (async, non-blocking) with conversation history
        5. Antwort in Chunks zurücksenden (HTML mit Fallback)
        6. Response cachen für Bookmark-Zugriff
    """
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    text: str = update.message.text or ""

    # Fix B: Reply-To-Kontext extrahieren (wenn User auf eine Bot-Nachricht antwortet)
    reply_to_text: str | None = None
    if update.message.reply_to_message and update.message.reply_to_message.text:
        reply_to_text = update.message.reply_to_message.text

    log.info(
        "Eingehende Nachricht von %s (%s): %d Zeichen%s",
        username,
        user_id,
        len(text),
        " (reply-to)" if reply_to_text else "",
    )

    # Typing-Indicator: User sieht dass der Bot arbeitet
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    # Pro-User Lock + globale Semaphore
    user_lock = _get_user_lock(user_id)
    async with user_lock:  # max 1 Request pro User gleichzeitig
        async with GLOBAL_CLAUDE_SEMAPHORE:  # max 4 global
            keepalive = asyncio.create_task(
                _typing_keepalive(
                    update.effective_chat,
                    interval=TYPING_KEEPALIVE_INTERVAL_SECONDS,
                )
            )
            try:
                result = await chat_service.process_user_message(
                    text=text,
                    user_id=user_id,
                    chat_id=chat_id,
                    username=username,
                    system_prompt=_get_system_prompt(context),
                    reply_to_text=reply_to_text,
                )
            finally:
                keepalive.cancel()
                try:
                    await keepalive
                except asyncio.CancelledError:
                    pass

    if not result.success:
        await update.message.reply_text(result.error_message)
        return

    # Antwort senden (HTML-Chunks + Fallback)
    await send_response(update, result.response)

    log.info(
        "Antwort gesendet: %d Zeichen in %.1fs",
        len(result.response),
        result.duration,
    )


@require_whitelist
async def handle_reset_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /reset. Löscht Conversation-History und Sticky-Language für diesen Chat."""
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    await chat_service.reset(user_id, chat_id)
    reset_msg = "Konversation zurückgesetzt. Wir starten frisch!"
    await update.message.reply_text(reset_msg)
    await chat_service.save_static_response_to_history(user_id, chat_id, reset_msg)
    log.info("User %d reset conversation in chat %d", user_id, chat_id)
    log_command_audit(
        action="reset",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
    )


@require_whitelist
async def handle_lang_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /lang <code>. Setzt die Sticky-Language für diesen Chat.

    Benutzung: /lang de, /lang en, /lang es, /lang fr, etc.
    """
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    args: list[str] = context.args or []
    if not args:
        supported = ", ".join(sorted(_SUPPORTED_LANGUAGES))
        await update.message.reply_text(
            f"Benutzung: /lang <code>\n\n"
            f"Unterstützte Sprachen: {supported}\n\n"
            f"Beispiel: /lang en"
        )
        return

    lang_code = args[0].lower().strip()
    if lang_code not in _SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(_SUPPORTED_LANGUAGES))
        await update.message.reply_text(
            f"Unbekannte Sprache: '{lang_code}'\n\nUnterstützte Sprachen: {supported}"
        )
        return

    # Alte Sprache merken für Audit-Details
    old_lang = await chat_service.get_chat_language(user_id, chat_id) or "auto"

    await chat_service.set_chat_language(user_id, chat_id, lang_code)

    lang_names: dict[str, str] = {
        "de": "Deutsch",
        "en": "English",
        "es": "Español",
        "fr": "Français",
        "it": "Italiano",
        "pt": "Português",
        "nl": "Nederlands",
        "pl": "Polski",
        "ru": "Русский",
        "ja": "日本語",
        "ko": "한국어",
        "zh": "中文",
    }
    name = lang_names.get(lang_code, lang_code)
    lang_msg = f"Sprache gewechselt: {name} ({lang_code})"
    await update.message.reply_text(lang_msg)
    await chat_service.save_static_response_to_history(user_id, chat_id, lang_msg)
    log.info("User %d set language to '%s' in chat %d", user_id, lang_code, chat_id)
    log_command_audit(
        action="lang_change",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details=f"{old_lang} -> {lang_code}",
    )


@require_whitelist
async def handle_new_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /new. Alias für /reset."""
    await handle_reset_command(update, context)


@require_whitelist
@require_private_chat
async def handle_save_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /save als Reply auf eine Bot-Nachricht (Toggle-Bookmark).

    Benutzung: Reply auf eine Bot-Nachricht mit /save zum Speichern/Entfernen.
    """
    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None

    # Muss eine Antwort auf eine andere Nachricht sein
    reply_msg = update.message.reply_to_message
    if reply_msg is None:
        await update.message.reply_text(
            "Antworte auf eine Bot-Nachricht mit /save um sie als Bookmark zu speichern."
        )
        return

    msg_id: int = reply_msg.message_id
    chat_id: int = update.effective_chat.id

    # Inhalt ermitteln: Cache zuerst, dann Message-Text
    content: str | None = get_cached_response(chat_id, msg_id)
    if content is None:
        content = reply_msg.text or ""
    if not content:
        content = "(Inhalt nicht verfügbar)"

    was_saved, user_message = save_or_toggle_bookmark(
        user_id=user_id,
        username=username,
        chat_id=chat_id,
        message_id=msg_id,
        content=content,
    )
    await update.message.reply_text(f"✓ {user_message}")
    log.info(
        "Bookmark %s via /save: user=%s message_id=%d",
        "gespeichert" if was_saved else "entfernt",
        username,
        msg_id,
    )
    log_command_audit(
        action="save_bookmark",
        user_id=user_id,
        chat_id=chat_id,
        username=username,
        entry_id=f"msg_{msg_id}",
        details="saved" if was_saved else "toggled_off",
    )


@require_whitelist
@require_private_chat
async def handle_bookmarks_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /bookmarks und /bookmarks search <query>.

    Benutzung:
        /bookmarks              -> Letzte 10 Bookmarks anzeigen
        /bookmarks search term  -> Bookmarks nach Inhalt durchsuchen
    """
    user = update.effective_user
    user_id: int = user.id if user else 0

    args: list[str] = context.args or []

    username: str | None = user.username if user else None

    # /bookmarks search <query>
    if len(args) >= 2 and args[0].lower() == "search":
        query_term = " ".join(args[1:])
        results = search(user_id, query_term, limit=20)

        if not results:
            await update.message.reply_text(
                f"Keine Bookmarks mit '{query_term}' gefunden."
            )
            log_command_audit(
                action="list_bookmarks",
                user_id=user_id,
                chat_id=update.effective_chat.id if update.effective_chat else 0,
                username=username,
                details=f"search '{query_term}': 0 results",
            )
            return

        header = f"Suchergebnisse für '{query_term}' ({len(results)} Treffer):\n\n"
        lines: list[str] = [header]

        for i, bm in enumerate(results, 1):
            lines.append(format_bookmark_preview(bm, i))
            lines.append("")

        keyboard = build_bookmarks_keyboard(results)
        text_body = "\n".join(lines)
        chunks = split_message(text_body)
        last_idx = len(chunks) - 1
        for ci, chunk in enumerate(chunks):
            if ci == last_idx:
                await update.message.reply_text(chunk, reply_markup=keyboard)
            else:
                await update.message.reply_text(chunk)
        log_command_audit(
            action="list_bookmarks",
            user_id=user_id,
            chat_id=update.effective_chat.id if update.effective_chat else 0,
            username=username,
            details=f"search '{query_term}': {len(results)} results",
        )
        return

    # /bookmarks (keine Argumente) -> letzte anzeigen
    bookmarks = list_bookmarks(user_id, limit=10)
    bm_chat_id = update.effective_chat.id if update.effective_chat else 0

    if not bookmarks:
        await update.message.reply_text(
            "Du hast noch keine Bookmarks. "
            "Antworte auf eine Bot-Nachricht mit /save um sie zu speichern."
        )
        log_command_audit(
            action="list_bookmarks",
            user_id=user_id,
            chat_id=bm_chat_id,
            username=username,
            details="0 bookmarks",
        )
        return

    header = f"Deine letzten {len(bookmarks)} Bookmarks:\n\n"
    lines: list[str] = [header]

    for i, bm in enumerate(bookmarks, 1):
        lines.append(format_bookmark_preview(bm, i))
        lines.append("")

    keyboard = build_bookmarks_keyboard(bookmarks)
    text_body = "\n".join(lines)
    chunks = split_message(text_body)
    last_idx = len(chunks) - 1
    for ci, chunk in enumerate(chunks):
        if ci == last_idx:
            await update.message.reply_text(chunk, reply_markup=keyboard)
        else:
            await update.message.reply_text(chunk)
    log_command_audit(
        action="list_bookmarks",
        user_id=user_id,
        chat_id=bm_chat_id,
        username=username,
        details=f"{len(bookmarks)} bookmarks",
    )


@require_whitelist
async def handle_help_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /help. Zeigt verfügbare Commands an."""
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    await update.message.reply_text(HELP_TEXT)
    await chat_service.save_static_response_to_history(user_id, chat_id, HELP_TEXT)


@require_whitelist
async def handle_start_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /start. Zeigt Willkommensnachricht an."""
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    await update.message.reply_text(START_TEXT)
    await chat_service.save_static_response_to_history(user_id, chat_id, START_TEXT)


@require_whitelist
@require_private_chat
async def handle_remember_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /remember <text>.

    Speichert Text als Episodic Memory.
    Als Reply auf Bot-Nachricht: speichert die Bot-Antwort.
    Ohne Reply: speichert den mitgegebenen Text.
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text("Memory-System nicht initialisiert.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    args: list[str] = context.args or []

    # Inhalt bestimmen
    content: str = ""
    reply_msg = update.message.reply_to_message

    if reply_msg and reply_msg.text:
        # Reply auf Bot-Nachricht: Bot-Antwort speichern
        content = reply_msg.text
        # Wenn zusätzlich Text angegeben: als Kontext-Label verwenden
        if args:
            label = " ".join(args)
            content = f"[{label}] {content}"
    elif args:
        # Kein Reply: Text direkt speichern
        content = " ".join(args)
    else:
        await update.message.reply_text(
            "Benutzung:\n"
            "/remember <text>  Text speichern\n"
            "/remember <label>  (als Reply)  Bot-Antwort mit Label speichern"
        )
        return

    entry_id = memory_service.remember_episodic(user_id=user_id, content=content)
    await update.message.reply_text(f"Gespeichert. [{entry_id}]")
    log.info("User %d remembered: %s (id=%s)", user_id, content[:50], entry_id)
    log_command_audit(
        action="remember",
        user_id=user_id,
        chat_id=update.effective_chat.id if update.effective_chat else 0,
        username=user.username if user else None,
        entry_id=entry_id,
    )


@require_whitelist
@require_private_chat
async def handle_memory_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /memory und /memory search <query>.

    /memory              Letzte 10 episodische Einträge anzeigen
    /memory search <q>   Memory durchsuchen
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text("Memory-System nicht initialisiert.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    args: list[str] = context.args or []

    # /memory search <query>
    if len(args) >= 2 and args[0].lower() == "search":
        query_term = " ".join(args[1:])
        results = memory_service.recall(user_id, query_term, layer="episodic")

        if not results:
            await update.message.reply_text(
                f"Keine Erinnerungen mit '{query_term}' gefunden."
            )
            return

        lines: list[str] = [
            f"Suchergebnisse für '{query_term}' ({len(results)} Treffer):\n"
        ]
        for entry in results[:10]:
            lines.append(f"  [{entry['id']}] {entry['content'][:80]}")
        await update.message.reply_text("\n".join(lines))
        return

    # /memory (keine Argumente): letzte 10 anzeigen
    entries = memory_service.list_recent(user_id, layer="episodic", limit=10)

    if not entries:
        await update.message.reply_text(
            "Noch keine Erinnerungen gespeichert. Nutze /remember <text> um etwas zu merken."
        )
        return

    lines: list[str] = [f"Letzte {len(entries)} Erinnerungen:\n"]
    for entry in entries:
        lines.append(f"  [{entry['id']}] {entry['content'][:80]}")
    await update.message.reply_text("\n".join(lines))


@require_whitelist
@require_private_chat
async def handle_forget_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /forget <entry_id>.

    Löscht einen Memory-Eintrag anhand seiner ID.
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text("Memory-System nicht initialisiert.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    args: list[str] = context.args or []

    if not args:
        await update.message.reply_text(
            "Benutzung: /forget <entry_id>\n\nIDs findest du via /memory"
        )
        return

    entry_id = args[0].strip()
    deleted = memory_service.forget(user_id, entry_id)

    if deleted:
        await update.message.reply_text(f"Vergessen: {entry_id}")
        log.info("User %d forgot memory: %s", user_id, entry_id)
    else:
        await update.message.reply_text(
            f"Eintrag '{entry_id}' nicht gefunden oder gehört dir nicht."
        )
    log_command_audit(
        action="forget",
        user_id=user_id,
        chat_id=update.effective_chat.id if update.effective_chat else 0,
        username=user.username if user else None,
        entry_id=entry_id,
        success=deleted,
    )
