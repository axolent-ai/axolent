"""Telegram handlers: message, command, and callback handlers.

All Telegram-specific handlers that react to user input.
Uses the application layer for business logic, presentation/render for output.

Since R04: streaming handler for real-time token updates via Telegram edits.
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
from application.model_service import DEFAULT_MODEL, ModelService
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

# Typing keepalive: Telegram shows typing for ~5s, we re-trigger every 4s
TYPING_KEEPALIVE_INTERVAL_SECONDS: float = 4.0


# Concurrency controls: max 4 Claude processes global, max 1 per user
GLOBAL_CLAUDE_SEMAPHORE = asyncio.Semaphore(4)
_user_locks: dict[int, tuple[asyncio.Lock, float]] = {}
_user_locks_meta_lock = Lock()
_USER_LOCK_TTL_SECONDS = 3600  # 1h without activity -> removed

# Supported languages for /lang command (synced with domain.onboarding.WIZARD_LANGUAGES)
_SUPPORTED_LANGUAGES: set[str] = {
    "de",
    "en",
    "es",
    "fr",
    "it",
    "pt",
    "nl",
    "pl",
    "sv",
    "tr",
    "ru",
    "uk",
    "zh",
    "ja",
    "ko",
    "ar",
    "hi",
    "id",
    "th",
    "vi",
}


async def _typing_keepalive(
    chat: Any, interval: float = TYPING_KEEPALIVE_INTERVAL_SECONDS
) -> None:
    """Sends typing indicator periodically until the task is cancelled.

    Runs as a background task parallel to the LLM call. Telegram shows the
    typing indicator for only ~5 seconds, so we re-trigger every 4s.
    On Telegram API errors (network hickups etc.) it silently continues
    or terminates, never raises.

    Args:
        chat: Telegram Chat object with send_chat_action method.
        interval: Seconds between re-triggers (default: 4.0).
    """
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await chat.send_chat_action(ChatAction.TYPING)
            except Exception as exc:
                log.debug("Typing keepalive ignoring error: %s", exc)
    except asyncio.CancelledError:
        pass


def _get_user_lock(user_id: int) -> asyncio.Lock:
    """Returns the per-user lock (lazy init with TTL cleanup)."""
    now = time.monotonic()
    with _user_locks_meta_lock:
        # Remove stale locks (only when not held)
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
    """Gets the ChatService from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        ChatService instance.

    Raises:
        RuntimeError: If ChatService is not in bot_data.
    """
    svc = context.application.bot_data.get("chat_service")
    if svc is None:
        raise RuntimeError(
            "ChatService not in bot_data. main.py must initialize ChatService."
        )
    return svc


def _get_system_prompt(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Gets the system prompt from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        System prompt string.

    Raises:
        RuntimeError: If system_prompt is not in bot_data.
    """
    prompt = context.application.bot_data.get("system_prompt")
    if prompt is None:
        raise RuntimeError(
            "system_prompt not in bot_data. main.py must set system_prompt."
        )
    return prompt


def _get_memory_service(
    context: ContextTypes.DEFAULT_TYPE,
) -> "MemoryService | None":
    """Gets the MemoryService from bot_data (can be None).

    Args:
        context: Telegram handler context.

    Returns:
        MemoryService instance or None.
    """
    return context.application.bot_data.get("memory_service")


def _get_bookmark_service(
    context: ContextTypes.DEFAULT_TYPE,
) -> BookmarkService:
    """Gets the BookmarkService from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        BookmarkService instance.

    Raises:
        RuntimeError: If BookmarkService is not in bot_data.
    """
    svc = context.application.bot_data.get("bookmark_service")
    if svc is None:
        raise RuntimeError(
            "BookmarkService not in bot_data. main.py must initialize BookmarkService."
        )
    return svc


def _get_rate_limiter(
    context: ContextTypes.DEFAULT_TYPE,
) -> "RateLimiter | None":
    """Gets the RateLimiter from bot_data (can be None).

    Args:
        context: Telegram handler context.

    Returns:
        RateLimiter instance or None.
    """
    return context.application.bot_data.get("rate_limiter")


def build_bookmarks_keyboard(bookmarks: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    """Builds an InlineKeyboard for the /bookmarks listing.

    Each bookmark gets two buttons: 'Full text' and 'Remove'.

    Args:
        bookmarks: List of bookmark dicts with 'message_id' and 'chat_id'.

    Returns:
        InlineKeyboardMarkup with two buttons per bookmark row.
    """
    rows: list[list[InlineKeyboardButton]] = []
    for i, bm in enumerate(bookmarks, 1):
        msg_id: int = bm.get("message_id", 0)
        bm_chat_id: int = bm.get("chat_id", 0)
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"#{i} Full text",
                    callback_data=f"bm_show:{bm_chat_id}:{msg_id}",
                ),
                InlineKeyboardButton(
                    text=f"#{i} Remove",
                    callback_data=f"bm_del:{bm_chat_id}:{msg_id}",
                ),
            ]
        )
    return InlineKeyboardMarkup(rows)


HELP_TEXT_DE: str = (
    "\U0001f916 <b>Axolent Befehlsübersicht</b>\n\n"
    "<b>Chat</b>\n"
    "• Schreibe einfach eine Nachricht und der Bot antwortet\n"
    "• /new neuer Chat (löscht Verlauf)\n"
    "• /reset löscht den Konversationsverlauf\n\n"
    "<b>Memory</b>\n"
    "• /remember &lt;Text&gt; speichert eine Notiz\n"
    "• /memory zeigt deine Notizen\n"
    "• /forget &lt;id&gt; löscht eine Notiz\n\n"
    "<b>Bookmarks</b>\n"
    "• /save (als Reply) speichert eine Bot-Antwort\n"
    "• /bookmarks zeigt gespeicherte Bookmarks\n\n"
    "<b>Multi-AI</b>\n"
    "• /debate &lt;Frage&gt; fragt mehrere KIs parallel\n\n"
    "<b>Konfiguration</b>\n"
    "• /settings visuelle Einstellungen\n"
    "• /setmodel &lt;modell&gt; wechselt KI-Modell (opus, sonnet, haiku)\n"
    "• /resetmodel setzt Modell auf Default zurück\n"
    "• /models zeigt aktuelle Slot-Belegung\n"
    "• /setlimit &lt;profil&gt; wechselt Profil (light, normal, power, unlimited)\n"
    "• /usage zeigt Verbrauch und Profil\n\n"
    "<b>Setup &amp; Sprache</b>\n"
    "• /start Begrüßung (Setup-Wizard für neue User)\n"
    "• /onboarding Setup-Wizard manuell starten\n"
    "• /lang &lt;code&gt; Sprache wechseln (de, en, fr, ...)\n\n"
    "<b>Hilfe</b>\n"
    "• /help diese Übersicht"
)

HELP_TEXT_EN: str = (
    "\U0001f916 <b>Axolent Command Overview</b>\n\n"
    "<b>Chat</b>\n"
    "• Just send a message and the bot will answer\n"
    "• /new new chat (clears history)\n"
    "• /reset clears conversation history\n\n"
    "<b>Memory</b>\n"
    "• /remember &lt;text&gt; saves a note\n"
    "• /memory shows your notes\n"
    "• /forget &lt;id&gt; deletes a note\n\n"
    "<b>Bookmarks</b>\n"
    "• /save (as reply) bookmarks a bot response\n"
    "• /bookmarks shows saved bookmarks\n\n"
    "<b>Multi-AI</b>\n"
    "• /debate &lt;question&gt; asks multiple AIs in parallel\n\n"
    "<b>Configuration</b>\n"
    "• /settings visual settings menu\n"
    "• /setmodel &lt;model&gt; changes AI model (opus, sonnet, haiku)\n"
    "• /resetmodel resets model to default\n"
    "• /models shows current slot assignments\n"
    "• /setlimit &lt;profile&gt; changes profile (light, normal, power, unlimited)\n"
    "• /usage shows usage and profile\n\n"
    "<b>Setup &amp; Language</b>\n"
    "• /start welcome (setup wizard for new users)\n"
    "• /onboarding start setup wizard manually\n"
    "• /lang &lt;code&gt; change language (de, en, fr, ...)\n\n"
    "<b>Help</b>\n"
    "• /help this overview"
)

# Legacy alias for backwards compatibility
HELP_TEXT: str = HELP_TEXT_DE

START_TEXT: str = (
    "Axolent is ready.\n\n"
    "Send me a question and I will answer it.\n\n"
    "Tip: You can bookmark bot messages. "
    "Just reply with /save."
)


def _get_persistent_provider(
    context: ContextTypes.DEFAULT_TYPE,
) -> Any:
    """Gets the PersistentProvider from bot_data (can be None).

    Type: ClaudePersistentProvider | None
    (Type annotation as Any due to hexagonal layer contract:
    presentation must not import infrastructure directly.)

    Args:
        context: Telegram handler context.

    Returns:
        ClaudePersistentProvider instance or None.
    """
    return context.application.bot_data.get("persistent_provider")


@require_whitelist
@require_private_chat
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes incoming Telegram messages via Claude Code subprocess.

    R04 flow (streaming):
        1. Whitelist check (via decorator)
        2. Privacy check: private chats only (via decorator)
        3. Send typing indicator
        4. Per-user lock + global semaphore
        5. Create streaming message ("...")
        6. Read token stream, periodically send Telegram edits
        7. Final edit with complete text
        8. Save history + audit

    Falls back to legacy flow when PersistentProvider is not available.
    """
    chat_service = _get_chat_service(context)
    persistent_provider = _get_persistent_provider(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    text: str = update.message.text or ""

    # Extract reply-to context
    reply_to_text: str | None = None
    if update.message.reply_to_message and update.message.reply_to_message.text:
        reply_to_text = update.message.reply_to_message.text

    log.info(
        "Incoming message from %s (%s): %d chars%s",
        username,
        user_id,
        len(text),
        " (reply-to)" if reply_to_text else "",
    )

    # Onboarding hint: if user skipped wizard, show hint after 3rd message
    onboarding_storage = context.application.bot_data.get("onboarding_storage")
    if onboarding_storage is not None and not onboarding_storage.is_onboarded(user_id):
        if not onboarding_storage.is_hint_shown(user_id):
            skip_count = onboarding_storage.increment_skip_count(user_id)
            if skip_count == 3:
                from domain.onboarding import get_onboarding_hint_text

                hint_lang = (
                    await chat_service.get_chat_language(user_id, chat_id) or "de"
                )
                hint = get_onboarding_hint_text(hint_lang)
                await update.message.reply_text(hint)
                onboarding_storage.set_hint_shown(user_id)

    # C-2: Check rate limit (before LLM call, before lock)
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is not None:
        result: RateLimitResult = rate_limiter.check_and_consume(user_id)
        if not result.allowed:
            from datetime import datetime, timezone

            # Human-readable error message with actionable solution
            period_labels = {"minute": "minute", "hour": "hour", "day": "day"}
            period_label = period_labels.get(result.period or "", "")
            retry_display = int(result.retry_after) if result.retry_after else 0

            if result.period == "minute":
                reset_info = f"Reset in {retry_display}s"
            elif result.period == "hour":
                reset_info = f"Reset in {retry_display // 60} minutes"
            else:
                reset_info = f"Reset in {retry_display // 3600}h"

            # Profile-specific upgrade options
            if result.profile == "light":
                options = (
                    "You can change your limit anytime for free:\n"
                    "• /usage — current overview\n"
                    "• /setlimit normal — more headroom "
                    "(350/h, 1,500/day)\n"
                    "• /setlimit power — much more "
                    "(900/h, 10,000/day)"
                )
            elif result.profile == "normal":
                options = (
                    "You can change your limit anytime for free:\n"
                    "• /usage — current overview\n"
                    "• /setlimit power — much more headroom "
                    "(900/h, 10,000/day)"
                )
            else:
                options = (
                    "• /usage — current overview\n"
                    "• /setlimit unlimited — disable all limits"
                )

            limit_msg = (
                f"You have reached your "
                f"{period_label} limit "
                f"({result.current_count}/{result.limit_value} "
                f"{'this ' + period_label if result.period != 'day' else 'today'}"
                f", {result.profile.capitalize()} profile).\n\n"
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
                "Rate limit for user %s (%s): %s limit, retry_after=%.1fs",
                username,
                user_id,
                result.period,
                result.retry_after or 0,
            )
            return

        # 70% warning (once per window)
        if result.warning_70 and result.warning_period:
            usage = rate_limiter.get_usage(user_id)
            if result.warning_period == "minute":
                warn_used = usage.minute_used
                warn_limit = usage.minute_limit
                warn_reset = f"Reset in {int(usage.minute_reset_seconds)}s"
                warn_period_label = "minute"
            elif result.warning_period == "hour":
                warn_used = usage.hour_used
                warn_limit = usage.hour_limit
                warn_reset = f"Reset in {int(usage.hour_reset_seconds) // 60} minutes"
                warn_period_label = "hour"
            else:
                warn_used = usage.day_used
                warn_limit = usage.day_limit
                warn_reset = f"Reset in {int(usage.day_reset_seconds) // 3600}h"
                warn_period_label = "day"

            # Next higher profile as upgrade suggestion
            user_profile = result.profile
            if user_profile == "light":
                upgrade_hint = (
                    "Want to do more? /setlimit normal raises the limit to 350/h."
                )
            elif user_profile == "normal":
                upgrade_hint = (
                    "Want to do more? /setlimit power raises the limit to 900/h."
                )
            else:
                upgrade_hint = "Change profile: /setlimit"

            warn_msg = (
                f"\U0001f4a1 You are using Axolent actively, "
                f"already {warn_used}/{warn_limit} requests "
                f"this {warn_period_label}.\n"
                f"{warn_reset}.\n\n"
                f"{upgrade_hint}"
            )
            await update.message.reply_text(warn_msg)

        # Unlimited reminder
        if result.unlimited_reminder:
            reminder_msg = (
                "\U0001f513 Note: You are in unlimited mode. "
                "No limits active.\n"
                "If you want more structure: "
                "/setlimit normal"
            )
            await update.message.reply_text(reminder_msg)
            # Audit for unlimited reminder
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

    # Typing indicator
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    # Per-user lock + global semaphore
    user_lock = _get_user_lock(user_id)
    async with user_lock:
        async with GLOBAL_CLAUDE_SEMAPHORE:
            # R04: Streaming path when PersistentProvider available
            # Type safety: hasattr instead of isinstance due to layer contract
            # (presentation must not import infrastructure.providers.base)
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
    """Streaming message handler (R04).

    Creates a placeholder message and incrementally edits it
    with incoming tokens.

    Error handling:
        - Error events: generic message with error_id to user,
          original text to audit log + application log
        - RuntimeError: same approach
        - Outer exceptions (e.g. create_streaming_message throws):
          generic error message, audit entry
        - Audit: always 2 entries (started + completed/crashed)
    """
    from datetime import datetime, timezone

    t_start = time.monotonic()
    streaming_chunks = 0
    final_text = ""
    had_error = False
    error_id = ""
    session: StreamingSession | None = None
    memory_entries_loaded = 0
    task_meta: dict[str, Any] = {}

    # Audit "started" entry
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

    # Process info is extracted from the stream's init event (not upfront,
    # because an upfront get_or_create without model argument would not
    # detect a model switch and would reuse the old subprocess).
    was_cold = False
    subprocess_pid = 0

    try:
        # Create streaming placeholder message
        streaming_msg = await create_streaming_message(update.effective_chat)
        session = StreamingSession(
            message=streaming_msg,
            started_at=time.monotonic(),
        )

        # Determine chat language once (used for text guard + status session)
        chat_lang = await chat_service.get_chat_language(user_id, chat_id) or "de"

        # Text Guard: streaming diacritic filter
        from application.text_guard_service import TextGuardService

        _tg_service = TextGuardService()
        _stream_guard = _tg_service.get_streaming_guard(chat_lang)
        _text_guard = _tg_service.get_guard(chat_lang, mode="fix")

        # Create status session (R02-B)
        from application.status_manager import SHOW_STATUS_UPDATES, StatusSession

        status_session: StatusSession | None = None
        if SHOW_STATUS_UPDATES:

            async def _status_callback(status_text: str) -> None:
                """Edits the placeholder message with status text."""
                try:
                    await streaming_msg.edit_text(status_text)
                except Exception as e:
                    log.debug("Status edit failed: %s", e)

            status_session = StatusSession(
                callback=_status_callback,
                language=chat_lang,
            )

        # Typing keepalive parallel to stream
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
                task_meta,
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
                if event.event_type == "init":
                    # Process metadata from the pool's init event
                    was_cold = event.was_cold
                    subprocess_pid = event.subprocess_pid
                    continue

                elif event.event_type == "content_delta":
                    streaming_chunks += 1
                    # Text Guard: filter token through streaming guard
                    _token = event.text
                    if _stream_guard is not None:
                        _filtered = _stream_guard.process_token(_token)
                        if _filtered is not None:
                            await process_streaming_edit(session, _filtered)
                        # else: buffered, waiting for word boundary
                    else:
                        await process_streaming_edit(session, _token)

                elif event.event_type == "result":
                    # Text Guard: flush streaming buffer + fix final text
                    final_text = event.full_text
                    if _stream_guard is not None:
                        _remaining = _stream_guard.flush()
                        if _remaining:
                            await process_streaming_edit(session, _remaining)
                    if _text_guard is not None:
                        final_text = _text_guard.fix(final_text)
                    await finalize_streaming(session, final_text)

                elif event.event_type == "error":
                    had_error = True
                    error_id = uuid.uuid4().hex[:8]
                    # Original text to log (not to user)
                    log.error(
                        "Streaming error event (ref: %s): %s | raw: %s",
                        error_id,
                        event.text,
                        event.raw,
                    )
                    # Generic message to user
                    await abort_streaming(
                        session,
                        "The language model provider reports a problem "
                        f"(ref: {error_id}). Please try again shortly.",
                    )
                    break

        except RuntimeError as e:
            had_error = True
            error_id = uuid.uuid4().hex[:8]
            log.error("Streaming RuntimeError (ref: %s): %s", error_id, e)
            await abort_streaming(
                session,
                f"Internal error (ref: {error_id}).",
            )

        finally:
            keepalive.cancel()
            try:
                await keepalive
            except asyncio.CancelledError:
                pass

        duration = time.monotonic() - t_start

        # Fallback: no final text but accumulated text available
        if not final_text and session.accumulated_text and not had_error:
            final_text = session.accumulated_text
            if _text_guard is not None:
                final_text = _text_guard.fix(final_text)
            await finalize_streaming(session, final_text)

        # Save history + audit (+ C-3 leakage check)
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
                task_meta=task_meta,
            )
            # C-3: If leakage detected, final edit with refusal
            if checked_text != final_text:
                await finalize_streaming(session, checked_text)
                final_text = checked_text
            log.info(
                "Streaming response: %d chars, %d chunks, %.1fs",
                len(final_text),
                streaming_chunks,
                duration,
            )
        elif had_error:
            # Audit for error case
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
                **task_meta,
            }
            write_raw_audit(audit_error)
            log.warning(
                "Streaming failed after %.1fs (ref: %s)",
                duration,
                error_id,
            )

    except Exception as outer_exc:
        # P1-8: Outer exception coverage (e.g. create_streaming_message throws)
        duration = time.monotonic() - t_start
        error_id = uuid.uuid4().hex[:8]
        log.exception("Outer streaming exception (ref: %s): %s", error_id, outer_exc)
        # Audit entry for crash
        audit_crash: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_error",
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "error_id": error_id,
            "duration_seconds": round(duration, 2),
            "error": "outer_exception",
            **task_meta,
        }
        write_raw_audit(audit_crash)

        # User-facing error message
        error_msg = f"Internal error (ref: {error_id})."
        try:
            if session is not None:
                await abort_streaming(session, error_msg)
            elif update.message:
                await update.message.reply_text(error_msg)
        except Exception as notify_exc:
            log.warning(
                "Could not notify user about error: %s",
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
    """Legacy message handler (pre-R04, non-streaming fallback)."""
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

    # Text Guard: fix diacritics in legacy (non-streaming) responses
    from application.text_guard_service import TextGuardService

    _legacy_tg = TextGuardService()
    _legacy_lang = await chat_service.get_chat_language(user_id, chat_id) or "de"
    _legacy_guard = _legacy_tg.get_guard(_legacy_lang, mode="fix")
    _response = result.response
    if _legacy_guard is not None:
        _response = _legacy_guard.fix(_response)

    await send_response(update, _response)

    log.info(
        "Legacy response sent: %d chars in %.1fs",
        len(result.response),
        result.duration,
    )


_RESET_TEXTS: dict[str, str] = {
    "de": "Konversation zurückgesetzt. Wir starten frisch!",
    "en": "Conversation reset. Let's start fresh!",
}


@require_whitelist
async def handle_reset_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /reset. Clears conversation history and sticky language for this chat."""
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    # Read language BEFORE reset (reset clears sticky language)
    lang = await chat_service.get_chat_language(user_id, chat_id) or "de"

    await chat_service.reset(user_id, chat_id)
    reset_msg = (
        _RESET_TEXTS.get(lang, _RESET_TEXTS["en"])
        if lang != "de"
        else _RESET_TEXTS["de"]
    )
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
    """Handles /lang <code>. Sets the sticky language for this chat.

    Usage: /lang de, /lang en, /lang es, /lang fr, etc.
    """
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    args: list[str] = context.args or []
    if not args:
        supported = ", ".join(sorted(_SUPPORTED_LANGUAGES))
        await update.message.reply_text(
            f"Usage: /lang <code>\n\n"
            f"Supported languages: {supported}\n\n"
            f"Example: /lang en"
        )
        return

    lang_code = args[0].lower().strip()
    if lang_code not in _SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(_SUPPORTED_LANGUAGES))
        await update.message.reply_text(
            f"Unknown language: '{lang_code}'\n\nSupported languages: {supported}"
        )
        return

    # Remember old language for audit details
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
        "sv": "Svenska",
        "tr": "Türkçe",
        "ru": "Русский",
        "uk": "Українська",
        "zh": "中文",
        "ja": "日本語",
        "ko": "한국어",
        "ar": "العربية",
        "hi": "हिन्दी",
        "id": "Bahasa Indonesia",
        "th": "ภาษาไทย",
        "vi": "Tiếng Việt",
    }
    name = lang_names.get(lang_code, lang_code)
    lang_msg = (
        f"Language changed: {name} ({lang_code})"
        if lang_code != "de"
        else f"Sprache gewechselt: {name} ({lang_code})"
    )
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
    """Handles /new. Alias for /reset."""
    await handle_reset_command(update, context)


@require_whitelist
@require_private_chat
async def handle_save_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /save as reply to a bot message (toggle bookmark).

    Usage: Reply to a bot message with /save to save/remove.
    """
    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None

    # Must be a reply to another message
    reply_msg = update.message.reply_to_message
    if reply_msg is None:
        await update.message.reply_text(
            "Reply to a bot message with /save to bookmark it."
        )
        return

    msg_id: int = reply_msg.message_id
    chat_id: int = update.effective_chat.id

    # Determine content: cache first, then message text
    content: str | None = get_cached_response(chat_id, msg_id)
    if content is None:
        content = reply_msg.text or ""
    if not content:
        content = "(content not available)"

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
        "saved" if was_saved else "removed",
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
    """Handles /bookmarks and /bookmarks search <query>.

    Usage:
        /bookmarks              -> Show last 10 bookmarks
        /bookmarks search term  -> Search bookmarks by content
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
                f"No bookmarks matching '{query_term}' found."
            )
            log_command_audit(
                action="list_bookmarks",
                user_id=user_id,
                chat_id=update.effective_chat.id if update.effective_chat else 0,
                username=username,
                details=f"search '{query_term}': 0 results",
            )
            return

        header = f"Search results for '{query_term}' ({len(results)} matches):\n\n"
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
            "You have no bookmarks yet. "
            "Reply to a bot message with /save to bookmark it."
        )
        log_command_audit(
            action="list_bookmarks",
            user_id=user_id,
            chat_id=bm_chat_id,
            username=username,
            details="0 bookmarks",
        )
        return

    header = f"Your last {len(bookmarks)} bookmarks:\n\n"
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
    """Handles /help. Shows available commands (DE/EN depending on language)."""
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    # Choose language-specific help text (DE for "de", EN for all other languages)
    lang = await chat_service.get_chat_language(user_id, chat_id) or "de"

    help_text = HELP_TEXT_DE if lang == "de" else HELP_TEXT_EN
    try:
        await update.message.reply_text(help_text, parse_mode="HTML")
    except Exception:
        # Fallback: plain text without HTML
        from domain.markdown import strip_markdown

        await update.message.reply_text(strip_markdown(help_text))
    await chat_service.save_static_response_to_history(user_id, chat_id, help_text)


@require_whitelist
async def handle_start_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /start. Shows setup wizard for new users, welcome for onboarded users."""
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    # Check onboarding state
    onboarding_storage = context.application.bot_data.get("onboarding_storage")
    if onboarding_storage is not None and not onboarding_storage.is_onboarded(user_id):
        # New user: start wizard
        from presentation.onboarding_callbacks import start_wizard

        await start_wizard(update, context, is_restart=False)
        return

    # Already onboarded: show welcome in sticky language
    from domain.onboarding import get_start_welcome_text

    lang = await chat_service.get_chat_language(user_id, chat_id) or "de"
    welcome_text = get_start_welcome_text(lang)
    await update.message.reply_text(welcome_text)
    await chat_service.save_static_response_to_history(user_id, chat_id, welcome_text)


@require_whitelist
@require_private_chat
async def handle_onboarding_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /onboarding. Starts the setup wizard manually (also for onboarded users)."""
    from presentation.onboarding_callbacks import start_wizard

    await start_wizard(update, context, is_restart=True)


@require_whitelist
@require_private_chat
async def handle_remember_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /remember <text>.

    Saves text as episodic memory.
    As reply to bot message: saves the bot response.
    Without reply: saves the provided text.
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text("Memory system not initialized.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    args: list[str] = context.args or []

    # Determine content
    content: str = ""
    reply_msg = update.message.reply_to_message

    if reply_msg and reply_msg.text:
        # Reply to bot message: save bot response
        content = reply_msg.text
        # If additional text provided: use as context label
        if args:
            label = " ".join(args)
            content = f"[{label}] {content}"
    elif args:
        # No reply: save text directly
        content = " ".join(args)
    else:
        await update.message.reply_text(
            "Usage:\n"
            "/remember <text>  save text\n"
            "/remember <label>  (as reply)  save bot response with label"
        )
        return

    entry_id = memory_service.remember_episodic(user_id=user_id, content=content)
    await update.message.reply_text(f"Saved. [{entry_id}]")
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
    """Handles /memory and /memory search <query>.

    /memory              Show last 10 episodic entries
    /memory search <q>   Search memory
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text("Memory system not initialized.")
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
                f"No memories matching '{query_term}' found."
            )
            return

        lines: list[str] = [
            f"Search results for '{query_term}' ({len(results)} matches):\n"
        ]
        for entry in results[:10]:
            lines.append(f"  [{entry['id']}] {entry['content'][:80]}")
        await update.message.reply_text("\n".join(lines))
        return

    # /memory (no arguments): show last 10
    entries = memory_service.list_recent(user_id, layer="episodic", limit=10)

    if not entries:
        await update.message.reply_text(
            "No memories saved yet. Use /remember <text> to save something."
        )
        return

    lines: list[str] = [f"Last {len(entries)} memories:\n"]
    for entry in entries:
        lines.append(f"  [{entry['id']}] {entry['content'][:80]}")
    await update.message.reply_text("\n".join(lines))


@require_whitelist
@require_private_chat
async def handle_forget_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /forget <entry_id>.

    Deletes a memory entry by its ID.
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text("Memory system not initialized.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    args: list[str] = context.args or []

    if not args:
        await update.message.reply_text(
            "Usage: /forget <entry_id>\n\nFind IDs via /memory"
        )
        return

    entry_id = args[0].strip()
    deleted = memory_service.forget(user_id, entry_id)

    if deleted:
        forget_msg = (
            f"Forgotten: {entry_id}\n\n"
            "Note: If the content is in the current conversation, "
            "use /reset for a full restart."
        )
        await update.message.reply_text(forget_msg)
        log.info("User %d forgot memory: %s", user_id, entry_id)
    else:
        await update.message.reply_text(
            f"Entry '{entry_id}' not found or does not belong to you."
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
    """Handles /usage. Shows current usage and limits."""
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is None:
        await update.message.reply_text("Rate limiter not initialized.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0

    usage = rate_limiter.get_usage(user_id)

    if usage.profile == "unlimited":
        msg = (
            "\U0001f4ca Your usage & profile:\n\n"
            "Profile: Unlimited\n\n"
            "\U0001f513 No limits active.\n\n"
            "Change profile: /setlimit normal"
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
            f"\U0001f4ca Your usage & profile:\n\n"
            f"Profile: {profile_display}\n\n"
            f"This minute: {usage.minute_used}/{usage.minute_limit} "
            f"{min_bar} (Reset in {min_reset})\n"
            f"This hour: {usage.hour_used}/{usage.hour_limit} "
            f"{hour_bar} (Reset in {hour_reset})\n"
            f"Today: {usage.day_used}/{usage.day_limit} "
            f"{day_bar} (Reset at {day_reset})\n\n"
            f"Change profile: /setlimit <light|normal|power|unlimited>"
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
    """Handles /setlimit <profile>. Changes the rate limit profile.

    Accepts: light, normal, power, unlimited.
    For unlimited: two-step confirmation required.
    """
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is None:
        await update.message.reply_text("Rate limiter not initialized.")
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    args: list[str] = context.args or []

    if not args:
        current = rate_limiter.get_user_profile(user_id)
        available = ", ".join(PROFILES.keys())
        await update.message.reply_text(
            f"Current profile: {current.capitalize()}\n\n"
            f"Usage: /setlimit <profile>\n"
            f"Available: {available}"
        )
        return

    target_profile = args[0].lower().strip()

    # Unlimited: two-step confirmation
    if target_profile == "unlimited":
        if len(args) < 2 or args[1].lower() != "confirm":
            await update.message.reply_text(
                "⚠️ You want to disable all limits.\n\n"
                "Risk:\n"
                "• Telegram may temporarily block the bot with too many edits\n"
                "• Your subscription will be used up faster\n"
                "• You will get a reminder every 100 requests\n\n"
                "If you are sure: /setlimit unlimited confirm"
            )
            return

    if target_profile not in PROFILES:
        available = ", ".join(PROFILES.keys())
        await update.message.reply_text(
            f"Unknown profile: '{target_profile}'\n\nAvailable: {available}"
        )
        return

    old_profile = rate_limiter.get_user_profile(user_id)
    success = rate_limiter.set_user_profile(user_id, chat_id, target_profile)

    if success:
        limits = PROFILES[target_profile]
        if target_profile == "unlimited":
            confirm_msg = (
                f"\U0001f513 Profile changed: {old_profile.capitalize()} → "
                f"Unlimited\n\n"
                f"No limits active. Reminder every 100 requests.\n"
                f"Revert: /setlimit normal"
            )
        else:
            confirm_msg = (
                f"✓ Profile changed: {old_profile.capitalize()} → "
                f"{target_profile.capitalize()}\n\n"
                f"New limits:\n"
                f"• {limits['per_minute']}/min\n"
                f"• {limits['per_hour']}/hour\n"
                f"• {limits['per_day']}/day"
            )
        await update.message.reply_text(confirm_msg)
    else:
        await update.message.reply_text("Error changing profile.")

    log_command_audit(
        action="setlimit",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details=f"{old_profile} -> {target_profile}",
    )


# ---------------------------------------------------------------------------
# /setmodel + /models Commands (R18 Phase 1: User-Model-Override)
# ---------------------------------------------------------------------------

# i18n strings for setmodel/models
_MODEL_STRINGS: dict[str, dict[str, str]] = {
    "de": {
        "set_success": "Modell gewechselt: {display_name} ({model_id})",
        "set_success_note": "Gilt ab der nächsten Nachricht.",
        "set_slot_success": "Modell für {slot} gewechselt: {display_name} ({model_id})",
        "set_slot_note": "Gilt nur für {slot}-Anfragen.",
        "reset_success": "Modell auf Default zurückgesetzt ({default_model}).",
        "reset_slot_success": "Modell für {slot} auf Default zurückgesetzt.",
        "reset_nothing": "Kein Modell-Override aktiv. Du nutzt bereits den Default ({default_model}).",
        "reset_all_success": "Alle Modell-Overrides zurückgesetzt ({count} entfernt).",
        "unknown_model": "Unbekanntes Modell: '{input}'",
        "unknown_slot": "Unbekannter Slot: '{input}'. Verfügbar: {slots}",
        "available_aliases": "Verfügbar: {aliases}",
        "usage_hint": (
            "Benutzung:\n"
            "  /setmodel <modell> (global)\n"
            "  /setmodel <slot> <modell> (pro Slot)\n"
            "  /setmodel reset (alles zurück)\n"
            "  /setmodel reset <slot> (Slot zurück)\n\n"
            "Slots: {slots}\n"
            "Beispiel: /setmodel code opus"
        ),
        "models_header": "Aktuelle Modell-Belegung:",
        "models_slot_line_override": "{slot}: {display_name} (Override)",
        "models_slot_line_default": "{slot}: {display_name} (Default)",
        "models_active": "Aktiv: {display_name} ({model_id})",
        "models_default": "Default: {display_name} ({model_id})",
        "models_available": "Verfügbare Modelle:",
        "models_change_hint": (
            "Wechseln: /setmodel <slot> <modell>\nBeispiel: /setmodel code opus"
        ),
    },
    "en": {
        "set_success": "Model changed: {display_name} ({model_id})",
        "set_success_note": "Takes effect from your next message.",
        "set_slot_success": "Model for {slot} changed: {display_name} ({model_id})",
        "set_slot_note": "Applies only to {slot} requests.",
        "reset_success": "Model reset to default ({default_model}).",
        "reset_slot_success": "Model for {slot} reset to default.",
        "reset_nothing": "No model override active. You are already using the default ({default_model}).",
        "reset_all_success": "All model overrides reset ({count} removed).",
        "unknown_model": "Unknown model: '{input}'",
        "unknown_slot": "Unknown slot: '{input}'. Available: {slots}",
        "available_aliases": "Available: {aliases}",
        "usage_hint": (
            "Usage:\n"
            "  /setmodel <model> (global)\n"
            "  /setmodel <slot> <model> (per slot)\n"
            "  /setmodel reset (reset all)\n"
            "  /setmodel reset <slot> (reset slot)\n\n"
            "Slots: {slots}\n"
            "Example: /setmodel code opus"
        ),
        "models_header": "Current model assignment:",
        "models_slot_line_override": "{slot}: {display_name} (Override)",
        "models_slot_line_default": "{slot}: {display_name} (Default)",
        "models_active": "Active: {display_name} ({model_id})",
        "models_default": "Default: {display_name} ({model_id})",
        "models_available": "Available models:",
        "models_change_hint": (
            "Switch: /setmodel <slot> <model>\nExample: /setmodel code opus"
        ),
    },
}


def _get_model_strings(lang: str = "de") -> dict[str, str]:
    """Returns model i18n strings for the given language."""
    return _MODEL_STRINGS.get(lang, _MODEL_STRINGS["de"])


def _get_model_service(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Gets the ModelService from bot_data (can be None)."""
    return context.application.bot_data.get("model_service")


@require_whitelist
@require_private_chat
async def handle_setmodel_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /setmodel in multiple variants.

    Phase 2a syntax:
      /setmodel <model>           set globally
      /setmodel <slot> <model>    set per slot
      /setmodel reset             reset all
      /setmodel reset <slot>      reset one slot
    """
    from domain.task_slot import TaskSlot

    model_service = _get_model_service(context)
    if model_service is None or not isinstance(model_service, ModelService):
        await update.message.reply_text("Model system not initialized.")
        return

    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or "de"
    s = _get_model_strings(lang)
    slot_names = ", ".join(TaskSlot.all_names())

    args: list[str] = context.args or []
    if not args:
        aliases = model_service.list_available_aliases()
        alias_list = ", ".join(sorted(aliases.keys()))
        msg = (
            f"{s['usage_hint'].format(slots=slot_names)}\n\n"
            f"{s['available_aliases'].format(aliases=alias_list)}"
        )
        await update.message.reply_text(msg)
        return

    first = args[0].lower().strip()

    # /setmodel reset [slot]
    if first == "reset":
        if len(args) >= 2:
            # /setmodel reset <slot>
            slot_input = args[1].lower().strip()
            slot = TaskSlot.from_string(slot_input)
            if slot is None:
                await update.message.reply_text(
                    s["unknown_slot"].format(input=slot_input, slots=slot_names)
                )
                return
            deleted = model_service.reset_user_model(user_id, slot=slot.value)
            if deleted:
                msg = s["reset_slot_success"].format(slot=slot.value.upper())
            else:
                default_display = model_service.get_model_display_name(DEFAULT_MODEL)
                msg = s["reset_nothing"].format(
                    default_model=f"{default_display} ({DEFAULT_MODEL})"
                )
            await update.message.reply_text(msg)
            log_command_audit(
                action="setmodel",
                user_id=user_id,
                chat_id=chat_id,
                username=user.username if user else None,
                details=f"reset slot={slot.value} (was_active={deleted})",
            )
        else:
            # /setmodel reset (alles)
            count = model_service.reset_all_slots(user_id)
            if count > 0:
                msg = s["reset_all_success"].format(count=count)
            else:
                default_display = model_service.get_model_display_name(DEFAULT_MODEL)
                msg = s["reset_nothing"].format(
                    default_model=f"{default_display} ({DEFAULT_MODEL})"
                )
            await update.message.reply_text(msg)
            log_command_audit(
                action="setmodel",
                user_id=user_id,
                chat_id=chat_id,
                username=user.username if user else None,
                details=f"reset all (removed={count})",
            )
        return

    # Check if first argument is a slot
    slot = TaskSlot.from_string(first)

    if slot is not None and len(args) >= 2:
        # /setmodel <slot> <model>
        model_input = args[1].lower().strip()
        success, result = model_service.set_user_model(
            user_id, model_input, slot=slot.value
        )
        if success:
            was_implicit_reset = model_service.last_was_implicit_reset
            display_name = model_service.get_model_display_name(result)
            msg = (
                f"{s['set_slot_success'].format(slot=slot.value.upper(), display_name=display_name, model_id=result)}\n"
                f"{s['set_slot_note'].format(slot=slot.value.upper())}"
            )
            await update.message.reply_text(msg)
            if was_implicit_reset:
                audit_action = "setmodel_implicit_reset"
                audit_details = (
                    f"implicit_reset slot={slot.value}, "
                    f"was default-equal alias={model_input}"
                )
            else:
                audit_action = "setmodel"
                audit_details = f"set slot={slot.value} alias={model_input} -> {result}"
            log_command_audit(
                action=audit_action,
                user_id=user_id,
                chat_id=chat_id,
                username=user.username if user else None,
                details=audit_details,
            )
        else:
            aliases = model_service.list_available_aliases()
            alias_list = ", ".join(sorted(aliases.keys()))
            msg = (
                f"{s['unknown_model'].format(input=model_input)}\n"
                f"{s['available_aliases'].format(aliases=alias_list)}"
            )
            await update.message.reply_text(msg)
        return

    # /setmodel <model> (global)
    success, result = model_service.set_user_model(user_id, first)
    if success:
        display_name = model_service.get_model_display_name(result)
        msg = (
            f"{s['set_success'].format(display_name=display_name, model_id=result)}\n"
            f"{s['set_success_note']}"
        )
        await update.message.reply_text(msg)
        log_command_audit(
            action="setmodel",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details=f"set global {first} -> {result}",
        )
    else:
        aliases = model_service.list_available_aliases()
        alias_list = ", ".join(sorted(aliases.keys()))
        msg = (
            f"{s['unknown_model'].format(input=first)}\n"
            f"{s['available_aliases'].format(aliases=alias_list)}"
        )
        await update.message.reply_text(msg)


@require_whitelist
@require_private_chat
async def handle_resetmodel_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /resetmodel. Shortcut for /setmodel reset.

    Standalone command so Telegram shows it as a clickable
    blue link in the help text (commands with arguments
    are not linked).
    """
    model_service = _get_model_service(context)
    if model_service is None or not isinstance(model_service, ModelService):
        await update.message.reply_text("Model system not initialized.")
        return

    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or "de"
    s = _get_model_strings(lang)

    deleted = model_service.reset_user_model(user_id)
    default_display = model_service.get_model_display_name(DEFAULT_MODEL)
    default_label = f"{default_display} ({DEFAULT_MODEL})"
    if deleted:
        msg = s["reset_success"].format(default_model=default_label)
    else:
        msg = s["reset_nothing"].format(default_model=default_label)
    await update.message.reply_text(msg)
    log_command_audit(
        action="resetmodel",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details=f"reset (was_active={deleted})",
    )


@require_whitelist
@require_private_chat
async def handle_models_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /models. Shows per-slot model assignment."""
    from domain.task_slot import TaskSlot

    model_service = _get_model_service(context)
    if model_service is None or not isinstance(model_service, ModelService):
        await update.message.reply_text("Model system not initialized.")
        return

    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or "de"
    s = _get_model_strings(lang)

    # Get TaskRouter for slot defaults
    task_router = context.application.bot_data.get("task_router")

    # Determine slot defaults
    slot_defaults: dict[str, str] = {}
    if task_router is not None and hasattr(task_router, "get_slot_defaults"):
        for slot, alias in task_router.get_slot_defaults().items():
            # Alias -> resolve to full model ID
            from application.model_service import resolve_alias

            resolved = resolve_alias(alias)
            slot_defaults[slot.value] = resolved if resolved else alias

    # Load user overrides
    overrides = model_service.get_all_slot_overrides(user_id)
    global_override = overrides.get("global")

    # Build per-slot lines
    slot_lines: list[str] = []
    for slot in TaskSlot:
        slot_name = slot.value.upper()

        # Determine effective model: slot override > global > slot default > system default
        slot_override = overrides.get(slot.value)
        if slot_override:
            display = model_service.get_model_display_name(slot_override)
            line = f"  {s['models_slot_line_override'].format(slot=slot_name, display_name=display)}"
        elif global_override:
            display = model_service.get_model_display_name(global_override)
            line = f"  {s['models_slot_line_override'].format(slot=slot_name, display_name=display)}"
        elif slot.value in slot_defaults:
            display = model_service.get_model_display_name(slot_defaults[slot.value])
            line = f"  {s['models_slot_line_default'].format(slot=slot_name, display_name=display)}"
        else:
            display = model_service.get_model_display_name(DEFAULT_MODEL)
            line = f"  {s['models_slot_line_default'].format(slot=slot_name, display_name=display)}"

        slot_lines.append(line)

    msg = (
        f"{s['models_header']}\n"
        + "\n".join(slot_lines)
        + f"\n\n{s['models_change_hint']}"
    )
    await update.message.reply_text(msg)
    log_command_audit(
        action="models",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details=f"overrides={overrides}",
    )


# ---------------------------------------------------------------------------
# /settings Command (R18 Phase 2b: Inline-Keyboard Settings)
# ---------------------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_settings_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /settings. Opens inline keyboard settings menu.

    Shows level A: slot assignment, language, reset all.
    All interactions are handled by callback handlers in settings_callbacks.py.
    """
    model_service = _get_model_service(context)
    if model_service is None or not isinstance(model_service, ModelService):
        await update.message.reply_text("Model system not initialized.")
        return

    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or "de"

    from presentation.settings_callbacks import build_main_menu_keyboard

    text, keyboard = build_main_menu_keyboard(user_id, model_service, context, lang)
    await update.message.reply_text(text, reply_markup=keyboard, parse_mode="HTML")
    log_command_audit(
        action="settings",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details="opened main menu",
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
    "Use /debate <question> to query multiple AIs in parallel.\n\n"
    "Example: /debate What is Bitcoin?"
)


# i18n strings for Debate output (DE default, EN prepared for future activation)
_DEBATE_STRINGS: dict[str, dict[str, str]] = {
    "de": {
        "header": "\U0001f3af Multi-AI-Debate",
        "question_label": "\U0001f4cc Frage",
        "no_providers": "Keine Provider konnten antworten.",
        "errors_label": "Fehler",
        "recommendation_label": "Kernaussage",
        "strongest_contribution": "Stärkster Beitrag",
        "tie_result": "Ergebnis: Gleichstand",
        "synthesis_header": "\U0001f3af Synthese",
        "consensus_header": "✨ Konsens / Dissens",
        "detail_header": "\U0001f4dd Detail-Antworten",
        "single_provider_hint": (
            "\U0001f4a1 Nur 1 Provider verfügbar. "
            "Für echtes Multi-AI-Debate: weitere Provider konfigurieren "
            "(z.B. Ollama installieren)."
        ),
        "quality_warning_prefix": "⚠️",
        "errors_section": "⚠️ Fehler:",
    },
    "en": {
        "header": "\U0001f3af Multi-AI Debate",
        "question_label": "\U0001f4cc Question",
        "no_providers": "No providers could respond.",
        "errors_label": "Errors",
        "recommendation_label": "Key Takeaway",
        "strongest_contribution": "Strongest Contribution",
        "tie_result": "Result: Tie",
        "synthesis_header": "\U0001f3af Synthesis",
        "consensus_header": "✨ Consensus / Dissent",
        "detail_header": "\U0001f4dd Detail Responses",
        "single_provider_hint": (
            "\U0001f4a1 Only 1 provider available. "
            "For a real Multi-AI Debate: configure more providers "
            "(e.g. install Ollama)."
        ),
        "quality_warning_prefix": "⚠️",
        "errors_section": "⚠️ Errors:",
    },
}


def _get_debate_strings(lang: str = "de") -> dict[str, str]:
    """Returns debate i18n strings for the given language.

    Falls back to German if language not available.

    Args:
        lang: ISO-639-1 language code.

    Returns:
        Dict of string keys to localized values.
    """
    return _DEBATE_STRINGS.get(lang, _DEBATE_STRINGS["de"])


def _format_debate_result(result: Any, lang: str = "de") -> str:
    """Formats a DebateResult as Telegram text (BLUF order).

    Block order (Bottom Line Up Front):
    1. Question
    2. Key takeaway (compact answer to the question)
    3. Strongest contribution (best single provider)
    4. Synthesis (combined core answer)
    5. Detail responses from AIs (Claude + Llama as originals)
    6. Pro/con per provider (analysis of originals)
    7. Timer

    Args:
        result: DebateResult instance.
        lang: Language for labels (default: "de").

    Returns:
        Formatted text for Telegram.
    """
    s = _get_debate_strings(lang)
    lines: list[str] = []
    lines.append(f"{s['header']}\n")
    lines.append(f"{s['question_label']}: {result.question}\n")

    if not result.responses:
        lines.append(s["no_providers"])
        if result.errors:
            lines.append(f"\n{s['errors_label']}: {', '.join(result.errors.keys())}")
        return "\n".join(lines)

    # --- Block 2: Kernaussage (BLUF) ---
    if result.final_verdict is not None:
        if result.final_verdict.recommendation:
            lines.append("━" * 20)
            lines.append(
                f"{s['recommendation_label']}: {result.final_verdict.recommendation}"
            )
            lines.append("")

        # --- Block 3: Strongest contribution ---
        winner_display = _PROVIDER_DISPLAY_NAMES.get(
            result.final_verdict.winner, result.final_verdict.winner
        )
        if result.final_verdict.winner == "tie":
            lines.append(s["tie_result"])
        else:
            lines.append(f"{s['strongest_contribution']}: {winner_display}")
        lines.append("")

        # --- Block 4: Synthese ---
        if result.final_verdict.synthesis:
            lines.append("━" * 20)
            lines.append(f"{s['synthesis_header']}\n")
            lines.append(result.final_verdict.synthesis)
            lines.append("")

        if result.final_verdict.judge_quality_warning:
            lines.append(
                f"\n{s['quality_warning_prefix']} "
                f"{result.final_verdict.judge_quality_warning}"
            )

    elif result.consensus_analysis:
        # Fallback: old consensus heuristic when judge fails
        lines.append("━" * 20)
        lines.append(f"{s['consensus_header']}:\n{result.consensus_analysis}")

    # --- Block 5: Detail-Antworten der KIs ---
    lines.append("━" * 20)
    lines.append(f"{s['detail_header']}\n")
    for provider_name, response_text in result.responses.items():
        display_name = _PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
        lines.append(f"{display_name}:")
        lines.append(response_text.strip())
        lines.append("")

    # --- Block 6: Pro/Contra je Provider (Analyse der Originale) ---
    if result.final_verdict is not None and result.final_verdict.evaluations:
        lines.append("━" * 20)
        for evaluation in result.final_verdict.evaluations:
            eval_display = _PROVIDER_DISPLAY_NAMES.get(
                evaluation.provider, evaluation.provider
            )
            pros_str = ", ".join(evaluation.pros) if evaluation.pros else ""
            cons_str = ", ".join(evaluation.cons) if evaluation.cons else ""
            if pros_str:
                lines.append(f"✅ {eval_display}: {pros_str}")
            if cons_str:
                lines.append(f"❌ {eval_display}: {cons_str}")
        lines.append("")

    # Show errors (if some providers crashed)
    if result.errors:
        lines.append("━" * 20)
        lines.append(s["errors_section"])
        for provider_name, error_msg in result.errors.items():
            display_name = _PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
            lines.append(f"  {display_name}: {error_msg}")
        lines.append("")

    # Single provider hint
    if len(result.responses) == 1 and not result.errors:
        lines.append(f"\n{s['single_provider_hint']}")

    # --- Block 7: Timer ---
    lines.append(f"\n⏱ {result.duration_seconds:.1f}s")

    return "\n".join(lines)


@require_whitelist
@require_private_chat
async def handle_debate_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /debate <question>. Multi-AI debate feature (R10).

    Queries multiple providers in parallel and shows answers side-by-side.
    """
    from datetime import datetime, timezone

    from application.debate_orchestrator import DebateOrchestrator

    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    # Extract question from command arguments
    args: list[str] = context.args or []
    if not args:
        await update.message.reply_text(DEBATE_HELP_TEXT)
        return

    question = " ".join(args)

    # Check rate limit (same logic as handle_message)
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is not None:
        result_rl: RateLimitResult = rate_limiter.check_and_consume(user_id)
        if not result_rl.allowed:
            await update.message.reply_text(
                "You have reached your limit. Wait a moment or "
                "increase your profile with /setlimit."
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

    # Send status message
    status_msg = await update.message.reply_text(
        "\U0001f3af Querying AIs in parallel... may take 30-60 seconds."
    )

    # Typing keepalive during debate
    keepalive = asyncio.create_task(
        _typing_keepalive(
            update.effective_chat,
            interval=TYPING_KEEPALIVE_INTERVAL_SECONDS,
        )
    )

    try:
        # Get or create DebateOrchestrator
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

    # Delete status message (best-effort, non-critical if it fails)
    try:
        await status_msg.delete()
    except Exception:  # nosec B110
        pass

    # Determine language for debate output
    debate_lang = await chat_service.get_chat_language(user_id, chat_id) or "de"

    # Format and send result
    formatted = _format_debate_result(debate_result, lang=debate_lang)

    # Text Guard: fix diacritics in debate output before sending
    from application.text_guard_service import TextGuardService

    _debate_tg = TextGuardService()
    _debate_guard = _debate_tg.get_guard(debate_lang, mode="fix")
    if _debate_guard is not None:
        formatted = _debate_guard.fix(formatted)

    chunks = split_message(formatted)
    for chunk in chunks:
        await update.message.reply_text(chunk)

    # Audit log
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
        "Debate completed for user %s: %d providers, %.1fs",
        username,
        len(debate_result.responses),
        debate_result.duration_seconds,
    )
