"""Telegram-Handler: Message-, Command- und Callback-Handler.

Alle Telegram-spezifischen Handler die auf User-Input reagieren.
Nutzt application-Layer für Business-Logik, presentation/render für Output.

Seit R04: Streaming-Handler für Echtzeit-Token-Updates via Telegram-Edits.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from threading import Lock
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from application.audit_service import log_command_audit, write_raw_audit
from application.bookmark_service import BookmarkService
from application.chat_service import ChatService
from application.rate_limiter import PROFILES, RateLimiter, RateLimitResult
from application.streaming_handler import (
    StreamingSession,
    abort_streaming,
    create_streaming_message,
    finalize_streaming,
    process_streaming_edit,
)
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


def _get_bookmark_service(
    context: ContextTypes.DEFAULT_TYPE,
) -> BookmarkService:
    """Holt den BookmarkService aus bot_data.

    Args:
        context: Telegram-Handler-Context.

    Returns:
        BookmarkService-Instanz.

    Raises:
        RuntimeError: Wenn BookmarkService nicht in bot_data ist.
    """
    svc = context.application.bot_data.get("bookmark_service")
    if svc is None:
        raise RuntimeError(
            "BookmarkService nicht in bot_data. main.py muss BookmarkService initialisieren."
        )
    return svc


def _get_rate_limiter(
    context: ContextTypes.DEFAULT_TYPE,
) -> "RateLimiter | None":
    """Holt den RateLimiter aus bot_data (kann None sein).

    Args:
        context: Telegram-Handler-Context.

    Returns:
        RateLimiter-Instanz oder None.
    """
    return context.application.bot_data.get("rate_limiter")


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
    "\U0001f916 Jarvis-LITE Befehlsübersicht\n\n"
    "Multi-AI:\n"
    "• /debate <Frage> fragt mehrere KIs parallel und "
    "vergleicht Antworten (Multi-AI-Debate)\n\n"
    "Bookmarks (Bot-Antworten speichern):\n"
    "• /save als Reply auf eine Bot-Nachricht speichert sie als Bookmark\n"
    "• /bookmarks zeigt deine gespeicherten Bookmarks "
    "(mit Inline-Buttons zum Anzeigen und Entfernen)\n\n"
    "Memory (eigene Notizen):\n"
    "• /remember <Text> speichert eine Notiz die der Bot "
    "in zukünftigen Antworten berücksichtigt\n"
    "• /forget <id> löscht eine Notiz "
    "(id steht in der Bestätigung von /remember)\n"
    "• /memory zeigt deine aktiven Notizen\n\n"
    "Limits & Profile:\n"
    "• /usage zeigt aktuellen Verbrauch und Profil\n"
    "• /setlimit <profil> wechselt Profil "
    "(light, normal, power, unlimited)\n\n"
    "Konversation:\n"
    "• /reset löscht den aktuellen Konversationsverlauf\n"
    "• /lang [code] wechselt Sprache (z.B. de, en, oder leer = automatisch)\n"
    "• /start zeigt die Begrüßung\n"
    "• /help diese Übersicht\n\n"
    "Ohne Slash:\n"
    "Schreibe einfach deine Frage, der Bot leitet sie an Claude weiter."
)

START_TEXT: str = (
    "Jarvis-LITE Bridge ist bereit.\n\n"
    "Schick mir eine Frage und ich leite sie an Claude weiter.\n\n"
    "Tipp: Du kannst Bot-Nachrichten als Bookmark speichern. "
    "Antworte einfach mit /save."
)


def _get_persistent_provider(
    context: ContextTypes.DEFAULT_TYPE,
) -> Any:
    """Holt den PersistentProvider aus bot_data (kann None sein).

    Typ: ClaudePersistentProvider | None
    (Typ-Annotation als Any wegen Hexagonal-Layer-Contract:
    presentation darf infrastructure nicht direkt importieren.)

    Args:
        context: Telegram-Handler-Context.

    Returns:
        ClaudePersistentProvider-Instanz oder None.
    """
    return context.application.bot_data.get("persistent_provider")


@require_whitelist
@require_private_chat
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Verarbeitet eingehende Telegram-Nachrichten via Claude Code Subprozess.

    R04-Flow (Streaming):
        1. Whitelist-Check (via Decorator)
        2. Privacy-Check: nur private Chats (via Decorator)
        3. Typing-Indicator senden
        4. Pro-User Lock + globale Semaphore
        5. Streaming-Nachricht erstellen ("...")
        6. Token-Stream lesen, periodisch Telegram-Edits senden
        7. Finale Edit mit vollständigem Text
        8. History + Audit speichern

    Fallback auf Legacy-Flow wenn PersistentProvider nicht verfügbar.
    """
    chat_service = _get_chat_service(context)
    persistent_provider = _get_persistent_provider(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    text: str = update.message.text or ""

    # Reply-To-Kontext extrahieren
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

    # C-2: Rate-Limit pruefen (vor LLM-Call, vor Lock)
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is not None:
        result: RateLimitResult = rate_limiter.check_and_consume(user_id)
        if not result.allowed:
            from datetime import datetime, timezone

            # Menschliche Fehlermeldung mit aktiver Lösung
            period_labels = {"minute": "Minute", "hour": "Stunde", "day": "Tag"}
            period_label = period_labels.get(result.period or "", "")
            retry_display = int(result.retry_after) if result.retry_after else 0

            if result.period == "minute":
                reset_info = f"Reset in {retry_display}s"
            elif result.period == "hour":
                reset_info = f"Reset in {retry_display // 60} Minuten"
            else:
                reset_info = f"Reset in {retry_display // 3600}h"

            # Profil-spezifische Upgrade-Optionen
            if result.profile == "light":
                options = (
                    "Du kannst dein Limit jederzeit kostenlos ändern:\n"
                    "• /usage — aktuelle Übersicht\n"
                    "• /setlimit normal — mehr Spielraum "
                    "(350/h, 1500/Tag)\n"
                    "• /setlimit power — viel mehr "
                    "(900/h, 10.000/Tag)"
                )
            elif result.profile == "normal":
                options = (
                    "Du kannst dein Limit jederzeit kostenlos ändern:\n"
                    "• /usage — aktuelle Übersicht\n"
                    "• /setlimit power — viel mehr Spielraum "
                    "(900/h, 10.000/Tag)"
                )
            else:
                options = (
                    "• /usage — aktuelle Übersicht\n"
                    "• /setlimit unlimited — alle Limits deaktivieren"
                )

            limit_msg = (
                f"Du hast dein "
                f"{'Minuten' if result.period == 'minute' else period_label + 'n' if result.period == 'hour' else 'Tages'}"
                f"-Limit erreicht "
                f"({result.current_count}/{result.limit_value} "
                f"{'in dieser ' + period_label if result.period != 'day' else 'heute'}"
                f", {result.profile.capitalize()}-Profil).\n\n"
                f"{reset_info}.\n\n"
                f"{options}"
            )
            await update.message.reply_text(limit_msg)

            write_raw_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "rate_limit_exceeded",
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "username": username,
                    "profile": result.profile,
                    "period": result.period,
                    "count": result.current_count,
                    "limit": result.limit_value,
                    "retry_after_seconds": result.retry_after,
                }
            )
            log.info(
                "Rate-Limit für User %s (%s): %s-Limit, retry_after=%.1fs",
                username,
                user_id,
                result.period,
                result.retry_after or 0,
            )
            return

        # 70%-Warnung (einmalig pro Window)
        if result.warning_70 and result.warning_period:
            usage = rate_limiter.get_usage(user_id)
            if result.warning_period == "minute":
                warn_used = usage.minute_used
                warn_limit = usage.minute_limit
                warn_reset = f"Reset in {int(usage.minute_reset_seconds)}s"
                warn_period_label = "Minute"
            elif result.warning_period == "hour":
                warn_used = usage.hour_used
                warn_limit = usage.hour_limit
                warn_reset = f"Reset in {int(usage.hour_reset_seconds) // 60} Minuten"
                warn_period_label = "Stunde"
            else:
                warn_used = usage.day_used
                warn_limit = usage.day_limit
                warn_reset = f"Reset in {int(usage.day_reset_seconds) // 3600}h"
                warn_period_label = "Tag"

            # Nächsthöheres Profil als Upgrade-Vorschlag
            user_profile = result.profile
            if user_profile == "light":
                upgrade_hint = (
                    "Falls du gerne noch mehr machen willst: "
                    "/setlimit normal hebt das Limit auf 350/h."
                )
            elif user_profile == "normal":
                upgrade_hint = (
                    "Falls du gerne noch mehr machen willst: "
                    "/setlimit power hebt das Limit auf 900/h."
                )
            else:
                upgrade_hint = "Profil ändern: /setlimit"

            warn_msg = (
                f"\U0001f4a1 Du nutzt Jarvis fleißig "
                f"— schon {warn_used}/{warn_limit} Anfragen "
                f"diese {warn_period_label}.\n"
                f"{warn_reset}.\n\n"
                f"{upgrade_hint}"
            )
            await update.message.reply_text(warn_msg)

        # Unlimited-Reminder
        if result.unlimited_reminder:
            reminder_msg = (
                "\U0001f513 Hinweis: Du bist im Unlimited-Modus. "
                "Keine Limits aktiv.\n"
                "Falls du wieder strukturierter arbeiten willst: "
                "/setlimit normal"
            )
            await update.message.reply_text(reminder_msg)
            # Audit für Unlimited-Reminder
            from datetime import datetime, timezone

            write_raw_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "unlimited_mode_warning",
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "username": username,
                    "profile": "unlimited",
                }
            )

    # Typing-Indicator
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    # Pro-User Lock + globale Semaphore
    user_lock = _get_user_lock(user_id)
    async with user_lock:
        async with GLOBAL_CLAUDE_SEMAPHORE:
            # R04: Streaming-Pfad wenn PersistentProvider verfügbar
            # Type-Safety: hasattr statt isinstance wegen Layer-Contract
            # (presentation darf infrastructure.providers.base nicht importieren)
            if (
                persistent_provider is not None
                and hasattr(persistent_provider, "query_streaming")
                and persistent_provider.is_available()
            ):
                await _handle_message_streaming(
                    update=update,
                    context=context,
                    chat_service=chat_service,
                    persistent_provider=persistent_provider,
                    user_id=user_id,
                    chat_id=chat_id,
                    username=username,
                    text=text,
                    reply_to_text=reply_to_text,
                )
            else:
                # Legacy-Fallback: non-streaming
                await _handle_message_legacy(
                    update=update,
                    context=context,
                    chat_service=chat_service,
                    user_id=user_id,
                    chat_id=chat_id,
                    username=username,
                    text=text,
                    reply_to_text=reply_to_text,
                )


async def _handle_message_streaming(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_service: ChatService,
    persistent_provider: Any,
    user_id: int,
    chat_id: int,
    username: str | None,
    text: str,
    reply_to_text: str | None,
) -> None:
    """Streaming-Message-Handler (R04).

    Erstellt eine Placeholder-Nachricht und editiert sie
    inkrementell mit eingehenden Tokens.

    Fehler-Handling:
        - Error-Events: generische Meldung mit error_id an User,
          Originaltext ins Audit-Log + Application-Log
        - RuntimeError: analog
        - Outer Exceptions (z.B. create_streaming_message fehlschlägt):
          generische Fehlermeldung, Audit-Eintrag
        - Audit: immer 2 Einträge (started + completed/crashed)
    """
    from datetime import datetime, timezone

    t_start = time.monotonic()
    streaming_chunks = 0
    final_text = ""
    had_error = False
    error_id = ""
    session: StreamingSession | None = None
    memory_entries_loaded = 0

    # Audit "started" Eintrag
    audit_started: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "stream_started",
        "user_id": user_id,
        "chat_id": chat_id,
        "username": username,
        "prompt_length": len(text),
        "provider": "claude_persistent",
    }
    write_raw_audit(audit_started)

    # Process-Info vorab holen (für was_cold)
    pool = context.application.bot_data.get("process_pool")
    was_cold = False
    subprocess_pid = 0

    try:
        # Streaming-Nachricht erstellen
        streaming_msg = await create_streaming_message(update.effective_chat)
        session = StreamingSession(
            message=streaming_msg,
            started_at=time.monotonic(),
        )

        # Status-Session erstellen (R02-B)
        from application.status_manager import SHOW_STATUS_UPDATES, StatusSession

        status_session: StatusSession | None = None
        if SHOW_STATUS_UPDATES:
            # Sprache fuer Status-Texte bestimmen
            chat_lang = await chat_service.get_chat_language(user_id, chat_id) or "de"

            async def _status_callback(status_text: str) -> None:
                """Editiert die Placeholder-Nachricht mit Status-Text."""
                try:
                    await streaming_msg.edit_text(status_text)
                except Exception as e:
                    log.debug("Status-Edit fehlgeschlagen: %s", e)

            status_session = StatusSession(
                callback=_status_callback,
                language=chat_lang,
            )

        # was_cold und subprocess_pid aus dem Pool holen
        if pool is not None:
            managed, was_cold = await pool.get_or_create(user_id, chat_id)
            subprocess_pid = managed.pid

        # Typing-Keepalive parallel zum Stream
        keepalive = asyncio.create_task(
            _typing_keepalive(
                update.effective_chat,
                interval=TYPING_KEEPALIVE_INTERVAL_SECONDS,
            )
        )

        try:
            (
                stream_iter,
                memory_entries_loaded,
            ) = await chat_service.process_user_message_streaming(
                text=text,
                user_id=user_id,
                chat_id=chat_id,
                username=username,
                system_prompt=_get_system_prompt(context),
                persistent_provider=persistent_provider,
                reply_to_text=reply_to_text,
                status_session=status_session,
            )
            async for event in stream_iter:
                if event.event_type == "content_delta":
                    streaming_chunks += 1
                    await process_streaming_edit(session, event.text)

                elif event.event_type == "result":
                    final_text = event.full_text
                    await finalize_streaming(session, final_text)

                elif event.event_type == "error":
                    had_error = True
                    error_id = uuid.uuid4().hex[:8]
                    # Originaltext ins Log (nicht zum User)
                    log.error(
                        "Streaming error event (ref: %s): %s | raw: %s",
                        error_id,
                        event.text,
                        event.raw,
                    )
                    # Generische Meldung an User
                    await abort_streaming(
                        session,
                        "Der Sprachmodell-Anbieter meldet ein Problem "
                        f"(ref: {error_id}). Versuch es gleich noch mal.",
                    )
                    break

        except RuntimeError as e:
            had_error = True
            error_id = uuid.uuid4().hex[:8]
            log.error("Streaming RuntimeError (ref: %s): %s", error_id, e)
            await abort_streaming(
                session,
                f"Interner Fehler (ref: {error_id}).",
            )

        finally:
            keepalive.cancel()
            try:
                await keepalive
            except asyncio.CancelledError:
                pass

        duration = time.monotonic() - t_start

        # Fallback: wenn kein finaler Text aber akkumulierter Text vorhanden
        if not final_text and session.accumulated_text and not had_error:
            final_text = session.accumulated_text
            await finalize_streaming(session, final_text)

        # History + Audit speichern (+ C-3 Leakage-Check)
        if final_text and not had_error:
            checked_text = await chat_service.save_streaming_result(
                user_id=user_id,
                chat_id=chat_id,
                user_text=text,
                response_text=final_text,
                duration_seconds=duration,
                username=username,
                was_cold=was_cold,
                streaming_chunks=streaming_chunks,
                subprocess_pid=subprocess_pid,
                memory_entries_loaded=memory_entries_loaded,
                system_prompt=_get_system_prompt(context),
            )
            # C-3: Wenn Leakage erkannt, finales Edit mit Refusal
            if checked_text != final_text:
                await finalize_streaming(session, checked_text)
                final_text = checked_text
            log.info(
                "Streaming-Antwort: %d Zeichen, %d Chunks, %.1fs",
                len(final_text),
                streaming_chunks,
                duration,
            )
        elif had_error:
            # Audit für Fehler-Fall
            audit_error: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "stream_error",
                "user_id": user_id,
                "chat_id": chat_id,
                "username": username,
                "error_id": error_id,
                "duration_seconds": round(duration, 2),
                "streaming_chunks": streaming_chunks,
                "was_cold": was_cold,
                "subprocess_pid": subprocess_pid,
            }
            write_raw_audit(audit_error)
            log.warning(
                "Streaming fehlgeschlagen nach %.1fs (ref: %s)",
                duration,
                error_id,
            )

    except Exception as outer_exc:
        # P1-8: Outer Exception Coverage (z.B. create_streaming_message wirft)
        duration = time.monotonic() - t_start
        error_id = uuid.uuid4().hex[:8]
        log.exception("Outer streaming exception (ref: %s): %s", error_id, outer_exc)
        # Audit-Eintrag für Crash
        audit_crash: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_error",
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "error_id": error_id,
            "duration_seconds": round(duration, 2),
            "error": "outer_exception",
        }
        write_raw_audit(audit_crash)

        # User-facing Fehlermeldung
        error_msg = f"Interner Fehler (ref: {error_id})."
        try:
            if session is not None:
                await abort_streaming(session, error_msg)
            elif update.message:
                await update.message.reply_text(error_msg)
        except Exception as notify_exc:
            log.warning(
                "Konnte User nicht über Fehler benachrichtigen: %s",
                notify_exc,
            )


async def _handle_message_legacy(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_service: ChatService,
    user_id: int,
    chat_id: int,
    username: str | None,
    text: str,
    reply_to_text: str | None,
) -> None:
    """Legacy-Message-Handler (pre-R04, non-streaming Fallback)."""
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

    await send_response(update, result.response)

    log.info(
        "Legacy-Antwort gesendet: %d Zeichen in %.1fs",
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

    bookmark_service = _get_bookmark_service(context)
    was_saved, user_message = bookmark_service.save_or_toggle_bookmark(
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

    bookmark_service = _get_bookmark_service(context)

    # /bookmarks search <query>
    if len(args) >= 2 and args[0].lower() == "search":
        query_term = " ".join(args[1:])
        results = bookmark_service.search(user_id, query_term, limit=20)

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
    bookmarks = bookmark_service.list_bookmarks(user_id, limit=10)
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
        forget_msg = (
            f"Vergessen: {entry_id}\n\n"
            "Hinweis: Falls der Inhalt in der laufenden Konversation steht, "
            "nutze /reset für vollständigen Neustart."
        )
        await update.message.reply_text(forget_msg)
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


@require_whitelist
@require_private_chat
async def handle_usage_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /usage. Zeigt aktuellen Verbrauch und Limits an."""
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is None:
        await update.message.reply_text("Rate-Limiter nicht initialisiert.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0

    usage = rate_limiter.get_usage(user_id)

    if usage.profile == "unlimited":
        msg = (
            "\U0001f4ca Deine Nutzung & dein Profil:\n\n"
            "Profil: Unlimited\n\n"
            "\U0001f513 Keine Limits aktiv.\n\n"
            "Profil ändern: /setlimit normal"
        )
    else:
        profile_display = usage.profile.capitalize()

        # Reset-Zeiten formatieren
        min_reset = f"{int(usage.minute_reset_seconds)}s"
        hour_reset_min = int(usage.hour_reset_seconds) // 60
        hour_reset = f"{hour_reset_min} Min"
        day_reset = "00:00"

        # Progress-Bars (10 Zeichen breit)
        def _bar(used: int, limit: int) -> str:
            if limit == 0:
                return "[██████████]"
            ratio = min(1.0, used / limit)
            filled = int(ratio * 10)
            empty = 10 - filled
            return f"[{'▓' * filled}{'░' * empty}]"

        min_bar = _bar(usage.minute_used, usage.minute_limit)
        hour_bar = _bar(usage.hour_used, usage.hour_limit)
        day_bar = _bar(usage.day_used, usage.day_limit)

        msg = (
            f"\U0001f4ca Deine Nutzung & dein Profil:\n\n"
            f"Profil: {profile_display}\n\n"
            f"Diese Minute: {usage.minute_used}/{usage.minute_limit} "
            f"{min_bar} (Reset in {min_reset})\n"
            f"Diese Stunde: {usage.hour_used}/{usage.hour_limit} "
            f"{hour_bar} (Reset in {hour_reset})\n"
            f"Heute: {usage.day_used}/{usage.day_limit} "
            f"{day_bar} (Reset um {day_reset})\n\n"
            f"Profil ändern: /setlimit <light|normal|power|unlimited>"
        )

    await update.message.reply_text(msg)
    log_command_audit(
        action="usage",
        user_id=user_id,
        chat_id=update.effective_chat.id if update.effective_chat else 0,
        username=user.username if user else None,
        details=f"profile={usage.profile}",
    )


@require_whitelist
@require_private_chat
async def handle_setlimit_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /setlimit <profil>. Wechselt das Rate-Limit-Profil.

    Akzeptiert: light, normal, power, unlimited.
    Bei unlimited: Zwei-Stufen-Bestätigung erforderlich.
    """
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is None:
        await update.message.reply_text("Rate-Limiter nicht initialisiert.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    args: list[str] = context.args or []

    if not args:
        current = rate_limiter.get_user_profile(user_id)
        available = ", ".join(PROFILES.keys())
        await update.message.reply_text(
            f"Aktuelles Profil: {current.capitalize()}\n\n"
            f"Benutzung: /setlimit <profil>\n"
            f"Verfügbar: {available}"
        )
        return

    target_profile = args[0].lower().strip()

    # Unlimited: Zwei-Stufen-Bestätigung
    if target_profile == "unlimited":
        if len(args) < 2 or args[1].lower() != "confirm":
            await update.message.reply_text(
                "⚠️ Du willst alle Limits deaktivieren.\n\n"
                "Risiko:\n"
                "• Telegram kann den Bot zeitweise sperren bei zu vielen Edits\n"
                "• Deine Subscription wird schneller leer\n"
                "• Du bekommst alle 100 Anfragen einen Reminder\n\n"
                "Falls du sicher bist: /setlimit unlimited confirm"
            )
            return

    if target_profile not in PROFILES:
        available = ", ".join(PROFILES.keys())
        await update.message.reply_text(
            f"Unbekanntes Profil: '{target_profile}'\n\nVerfügbar: {available}"
        )
        return

    old_profile = rate_limiter.get_user_profile(user_id)
    success = rate_limiter.set_user_profile(user_id, chat_id, target_profile)

    if success:
        limits = PROFILES[target_profile]
        if target_profile == "unlimited":
            confirm_msg = (
                f"\U0001f513 Profil gewechselt: {old_profile.capitalize()} → "
                f"Unlimited\n\n"
                f"Keine Limits aktiv. Reminder alle 100 Anfragen.\n"
                f"Zurück: /setlimit normal"
            )
        else:
            confirm_msg = (
                f"✓ Profil gewechselt: {old_profile.capitalize()} → "
                f"{target_profile.capitalize()}\n\n"
                f"Neue Limits:\n"
                f"• {limits['per_minute']}/Min\n"
                f"• {limits['per_hour']}/Stunde\n"
                f"• {limits['per_day']}/Tag"
            )
        await update.message.reply_text(confirm_msg)
    else:
        await update.message.reply_text("Fehler beim Profilwechsel.")

    log_command_audit(
        action="setlimit",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details=f"{old_profile} -> {target_profile}",
    )


# ---------------------------------------------------------------------------
# /debate Command (R10: Multi-AI-Debate)
# ---------------------------------------------------------------------------

# Provider display names for formatted output
_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "claude_persistent": "\U0001f916 Claude",
    "claude": "\U0001f916 Claude",
    "ollama_local": "\U0001f999 Llama (lokal)",
    "openai": "\U0001f4a1 OpenAI",
    "gemini": "✨ Gemini",
    "mistral": "\U0001f32c️ Mistral",
}

DEBATE_HELP_TEXT: str = (
    "Nutze /debate <Frage> um mehrere KIs parallel zu befragen.\n\n"
    "Beispiel: /debate Was ist Bitcoin?"
)


def _format_debate_result(result: Any) -> str:
    """Formatiert ein DebateResult als Telegram-Text.

    Args:
        result: DebateResult-Instanz.

    Returns:
        Formatierter Text fuer Telegram.
    """
    lines: list[str] = []
    lines.append("\U0001f3af Multi-AI-Debate\n")
    lines.append(f"\U0001f4cc Frage: {result.question}\n")

    if not result.responses:
        lines.append("Keine Provider konnten antworten.")
        if result.errors:
            lines.append(f"\nFehler: {', '.join(result.errors.keys())}")
        return "\n".join(lines)

    for provider_name, response_text in result.responses.items():
        display_name = _PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
        lines.append("━" * 20)
        lines.append(f"{display_name}:")
        lines.append(response_text.strip())
        lines.append("")

    # Fehler anzeigen (falls einige Provider crashed sind)
    if result.errors:
        lines.append("━" * 20)
        lines.append("⚠️ Fehler:")
        for provider_name, error_msg in result.errors.items():
            display_name = _PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
            lines.append(f"  {display_name}: {error_msg}")
        lines.append("")

    # Konsens-Analyse
    if result.consensus_analysis:
        lines.append("━" * 20)
        lines.append(f"✨ Konsens / Dissens:\n{result.consensus_analysis}")

    # Nur 1 Provider Hinweis
    if len(result.responses) == 1 and not result.errors:
        lines.append(
            "\n\U0001f4a1 Nur 1 Provider verfuegbar. "
            "Fuer echtes Multi-AI-Debate: weitere Provider konfigurieren "
            "(z.B. Ollama installieren)."
        )

    # Dauer
    lines.append(f"\n⏱ {result.duration_seconds:.1f}s")

    return "\n".join(lines)


@require_whitelist
@require_private_chat
async def handle_debate_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Verarbeitet /debate <Frage>. Multi-AI-Debate Feature (R10).

    Fragt mehrere Provider parallel und zeigt Antworten side-by-side.
    """
    from datetime import datetime, timezone

    from application.debate_orchestrator import DebateOrchestrator

    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    # Frage aus Command-Argumenten extrahieren
    args: list[str] = context.args or []
    if not args:
        await update.message.reply_text(DEBATE_HELP_TEXT)
        return

    question = " ".join(args)

    # Rate-Limit pruefen (gleiche Logik wie handle_message)
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is not None:
        result_rl: RateLimitResult = rate_limiter.check_and_consume(user_id)
        if not result_rl.allowed:
            await update.message.reply_text(
                "Du hast dein Limit erreicht. Warte einen Moment oder "
                "erhoehe dein Profil mit /setlimit."
            )
            write_raw_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "rate_limit_exceeded",
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "username": username,
                    "command": "debate",
                    "profile": result_rl.profile,
                    "period": result_rl.period,
                }
            )
            return

    # Status-Nachricht senden
    status_msg = await update.message.reply_text(
        "\U0001f3af Frage KIs parallel... kann 30-60 Sekunden dauern."
    )

    # Typing-Keepalive waehrend der Debate
    keepalive = asyncio.create_task(
        _typing_keepalive(
            update.effective_chat,
            interval=TYPING_KEEPALIVE_INTERVAL_SECONDS,
        )
    )

    try:
        # DebateOrchestrator aus bot_data oder neu erstellen
        chat_service = _get_chat_service(context)
        orchestrator = DebateOrchestrator(
            provider_router=chat_service.provider_router,
        )

        debate_result = await orchestrator.debate(
            question=question,
            user_id=user_id,
            chat_id=chat_id,
        )
    finally:
        keepalive.cancel()
        try:
            await keepalive
        except asyncio.CancelledError:
            pass

    # Status-Nachricht loeschen (best-effort, unkritisch wenn fehlschlaegt)
    try:
        await status_msg.delete()
    except Exception:  # nosec B110
        pass

    # Ergebnis formatieren und senden
    formatted = _format_debate_result(debate_result)
    chunks = split_message(formatted)
    for chunk in chunks:
        await update.message.reply_text(chunk)

    # Audit-Log
    write_raw_audit(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "debate",
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "question_length": len(question),
            "providers_queried": debate_result.providers_queried,
            "providers_responded": list(debate_result.responses.keys()),
            "providers_errored": list(debate_result.errors.keys()),
            "duration_seconds": round(debate_result.duration_seconds, 2),
        }
    )

    log.info(
        "Debate abgeschlossen fuer User %s: %d Provider, %.1fs",
        username,
        len(debate_result.responses),
        debate_result.duration_seconds,
    )
