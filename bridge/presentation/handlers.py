"""Telegram handlers: message, command, and callback handlers.

All Telegram-specific handlers that react to user input.
Uses the application layer for business logic, presentation/render for output.

Since R04: streaming handler for real-time token updates via Telegram edits.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from typeguard import typechecked

from application.audit_service import (
    filter_task_meta,
    log_command_audit,
    write_raw_audit,
)
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
from domain.language import DEFAULT_LANGUAGE
from i18n.domain.i18n import t
from presentation.decorators import lcp_aware, require_private_chat, require_whitelist
from presentation.render import (
    get_cached_response,
    sanitize_telegram_slashes,
    send_response,
    split_message,
)

from application.execution import (
    ContextKernel,
    ExecutionPlanner,
    RequestEnvelope,
)
from application.security.injection_detector import InjectionDetector
from application.security.secret_scanner import SecretBlockedError

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

# T25: Active streaming sessions per (user_id, chat_id).
# Used by /reset to cancel a running stream before clearing state.
_active_streaming_sessions: dict[tuple[int, int], StreamingSession] = {}
_active_sessions_lock = Lock()

# T25: Background task registry. Prevents GC of streaming tasks and enables
# cleanup on shutdown. Key: (user_id, chat_id), Value: asyncio.Task.
_background_streaming_tasks: dict[tuple[int, int], asyncio.Task] = {}  # type: ignore[type-arg]


def _register_background_task(
    task: "asyncio.Task[None]", user_id: int, chat_id: int
) -> None:
    """Register a background streaming task and set up cleanup callback."""
    key = (user_id, chat_id)
    _background_streaming_tasks[key] = task

    def _on_task_done(t: "asyncio.Task[None]") -> None:
        _background_streaming_tasks.pop(key, None)
        if t.cancelled():
            log.debug(
                "Background stream task cancelled: user=%d chat=%d", user_id, chat_id
            )
        elif t.exception():
            log.error(
                "Background stream task crashed: user=%d chat=%d: %s",
                user_id,
                chat_id,
                t.exception(),
            )

    task.add_done_callback(_on_task_done)


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


def _get_context_kernel(context: ContextTypes.DEFAULT_TYPE) -> ContextKernel:
    """Gets the ContextKernel from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        ContextKernel instance.

    Raises:
        RuntimeError: If ContextKernel is not in bot_data.
    """
    kernel = context.application.bot_data.get("context_kernel")
    if kernel is None:
        raise RuntimeError(
            "context_kernel not in bot_data. main.py must initialize ContextKernel."
        )
    return kernel


def _get_execution_planner(context: ContextTypes.DEFAULT_TYPE) -> ExecutionPlanner:
    """Gets the ExecutionPlanner from bot_data.

    If not explicitly registered, creates a default instance.
    Phase 0 Commit 3: the planner is a lightweight object,
    safe to create on demand.

    Args:
        context: Telegram handler context.

    Returns:
        ExecutionPlanner instance.
    """
    planner = context.application.bot_data.get("execution_planner")
    if planner is None:
        planner = ExecutionPlanner()
        context.application.bot_data["execution_planner"] = planner
    return planner


def build_bookmarks_keyboard(
    bookmarks: list[dict[str, Any]], lang: str = "en"
) -> InlineKeyboardMarkup:
    """Builds an InlineKeyboard for the /bookmarks listing.

    Each bookmark gets two buttons: 'Full text' and 'Remove'.

    Args:
        bookmarks: List of bookmark dicts with 'message_id' and 'chat_id'.
        lang: ISO-639-1 language code for button labels.

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
                    text=f"#{i} {t('bookmark.full_text_btn', lang)}",
                    callback_data=f"bm_show:{bm_chat_id}:{msg_id}",
                ),
                InlineKeyboardButton(
                    text=f"#{i} {t('bookmark.remove_btn', lang)}",
                    callback_data=f"bm_del:{bm_chat_id}:{msg_id}",
                ),
            ]
        )
    return InlineKeyboardMarkup(rows)


HELP_TEXT_DE: str = t("help.title", "de") + "\n\n" + t("help.body", "de")

HELP_TEXT_EN: str = t("help.title", "en") + "\n\n" + t("help.body", "en")

# Legacy alias for backwards compatibility
HELP_TEXT: str = HELP_TEXT_DE

START_TEXT: str = t("start.welcome", "en")


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
@typechecked
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

    # Detect inline commands embedded in regular messages (T19 fix)
    # Users sometimes append /setmodel, /reset etc. at the end of a message.
    # Telegram only triggers CommandHandler when / is the very first character.
    _inline_cmd_match = re.search(
        r"(?:^|\n)\s*/(?:setmodel|reset|resetmodel|lang|new)\b",
        text,
        re.IGNORECASE,
    )
    if _inline_cmd_match and not text.strip().startswith("/"):
        _cmd_lang = (
            await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
        )
        await update.message.reply_text(t("inline_command.warning", _cmd_lang))
        # Strip the command from the text before sending to LLM
        text = text[: _inline_cmd_match.start()].rstrip()
        if not text:
            return

    # Extract reply-to context
    reply_to_text: str | None = None
    if update.message.reply_to_message and update.message.reply_to_message.text:
        reply_to_text = update.message.reply_to_message.text

    # EK-01: Build RequestEnvelope BEFORE rate-limit check so that all
    # audit events (including rejections) carry a stable request_id.
    _msg_envelope = RequestEnvelope.from_telegram(
        user_id=user_id,
        chat_id=chat_id,
        text=text,
        username=username,
        reply_to_text=reply_to_text,
    )

    # Universal audit anchor: request_received
    write_raw_audit(
        {
            "timestamp": _msg_envelope.timestamp_utc.isoformat(),
            "event_type": "request_received",
            "request_id": _msg_envelope.request_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "channel": _msg_envelope.channel,
            "command": _msg_envelope.command,
        }
    )

    log.info(
        "Incoming message from %s (%s): %d chars%s [req=%s]",
        username,
        user_id,
        len(text),
        " (reply-to)" if reply_to_text else "",
        _msg_envelope.request_id,
    )

    # Onboarding hint: if user skipped wizard, show hint after 3rd message
    onboarding_storage = context.application.bot_data.get("onboarding_storage")
    if onboarding_storage is not None and not onboarding_storage.is_onboarded(user_id):
        if not onboarding_storage.is_hint_shown(user_id):
            skip_count = onboarding_storage.increment_skip_count(user_id)
            if skip_count == 3:
                hint_lang = (
                    await chat_service.get_chat_language(user_id, chat_id)
                    or DEFAULT_LANGUAGE
                )
                hint = t("onboarding.hint", hint_lang)
                await update.message.reply_text(hint)
                onboarding_storage.set_hint_shown(user_id)

    # C-2: Check rate limit (before LLM call, before lock)
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is not None:
        result: RateLimitResult = rate_limiter.check_and_consume(user_id)
        if not result.allowed:
            _rl_lang = (
                await chat_service.get_chat_language(user_id, chat_id)
                or DEFAULT_LANGUAGE
            )

            # Period label and reset info via i18n
            retry_display = int(result.retry_after) if result.retry_after else 0
            period_key = f"rate_limit.period_{result.period or 'minute'}"
            period_label = t(period_key, _rl_lang)

            if result.period == "minute":
                reset_info = t(
                    "rate_limit.reset_minute", _rl_lang, seconds=retry_display
                )
            elif result.period == "hour":
                reset_info = t(
                    "rate_limit.reset_hour", _rl_lang, minutes=retry_display // 60
                )
            else:
                reset_info = t(
                    "rate_limit.reset_day", _rl_lang, hours=retry_display // 3600
                )

            # Profile-specific options via i18n
            options_key = f"rate_limit.options_{result.profile}"
            options = t(options_key, _rl_lang)

            # Window display via i18n
            window_key = f"rate_limit.window_{result.period or 'minute'}"
            window = t(window_key, _rl_lang)

            limit_msg = t(
                "rate_limit.exceeded",
                _rl_lang,
                period=period_label,
                current=result.current_count,
                limit=result.limit_value,
                window=f"{window}, {result.profile.capitalize()}",
                profile=result.profile.capitalize(),
                reset_info=reset_info,
                options=options,
            )
            await update.message.reply_text(limit_msg)

            write_raw_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "rate_limit_exceeded",
                    "request_id": _msg_envelope.request_id,
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
            _warn_lang = (
                await chat_service.get_chat_language(user_id, chat_id)
                or DEFAULT_LANGUAGE
            )
            usage = rate_limiter.get_usage(user_id)
            if result.warning_period == "minute":
                warn_used = usage.minute_used
                warn_limit = usage.minute_limit
                warn_reset = t(
                    "rate_limit.reset_minute",
                    _warn_lang,
                    seconds=int(usage.minute_reset_seconds),
                )
            elif result.warning_period == "hour":
                warn_used = usage.hour_used
                warn_limit = usage.hour_limit
                warn_reset = t(
                    "rate_limit.reset_hour",
                    _warn_lang,
                    minutes=int(usage.hour_reset_seconds) // 60,
                )
            else:
                warn_used = usage.day_used
                warn_limit = usage.day_limit
                warn_reset = t(
                    "rate_limit.reset_day",
                    _warn_lang,
                    hours=int(usage.day_reset_seconds) // 3600,
                )

            warn_period_label = t(
                f"rate_limit.period_{result.warning_period}", _warn_lang
            )

            # Upgrade hint via i18n
            upgrade_key = f"rate_limit.upgrade_{result.profile}"
            upgrade_hint = t(upgrade_key, _warn_lang)

            warn_msg = t(
                "rate_limit.warning_70",
                _warn_lang,
                used=warn_used,
                limit=warn_limit,
                period=warn_period_label,
                reset_info=warn_reset,
                upgrade_hint=upgrade_hint,
            )
            await update.message.reply_text(warn_msg)

        # Unlimited reminder
        if result.unlimited_reminder:
            _remind_lang = (
                await chat_service.get_chat_language(user_id, chat_id)
                or DEFAULT_LANGUAGE
            )
            reminder_msg = t("rate_limit.unlimited_reminder", _remind_lang)
            await update.message.reply_text(reminder_msg)
            # Audit for unlimited reminder
            write_raw_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "unlimited_mode_warning",
                    "request_id": _msg_envelope.request_id,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "username": username,
                    "profile": "unlimited",
                }
            )

    # Typing indicator
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    # RISK-3: Ask-Before-Apply pre-flight check.
    # If a skill match requires user confirmation, show inline keyboard
    # and store pending state instead of streaming immediately.
    try:
        _pre_match = chat_service.pre_match_skill(
            user_id=user_id,
            text=text,
            lang=(
                await chat_service.get_chat_language(user_id, chat_id)
                or DEFAULT_LANGUAGE
            ),
        )
        if _pre_match is not None:
            from application.skill_compression.skill_matcher import should_ask_user

            if should_ask_user(_pre_match):
                from presentation.skill_commands import (
                    build_skill_confirm_keyboard,
                    get_pending_skill_confirmations,
                )

                _confirm_lang = (
                    await chat_service.get_chat_language(user_id, chat_id)
                    or DEFAULT_LANGUAGE
                )
                # Store pending confirmation (C3-SC-01: composite key)
                _pending_store = get_pending_skill_confirmations(context)
                _hyp_id = _pre_match.hypothesis.hypothesis_id
                _pending_store[(user_id, chat_id, _hyp_id)] = {
                    "skill_match": _pre_match,
                    "original_text": text,
                    "timestamp": time.time(),
                    "envelope": _msg_envelope,
                }
                # Show confirmation keyboard
                _hyp = _pre_match.hypothesis
                _safe_claim = html_mod.escape(_hyp.claim)
                _confirm_text = (
                    f"{t('skill.confirm_apply_question', _confirm_lang)}\n\n"
                    f"<i>{_safe_claim}</i>"
                )
                _keyboard = build_skill_confirm_keyboard(
                    _hyp.hypothesis_id, _confirm_lang
                )
                try:
                    await update.message.reply_text(
                        _confirm_text,
                        reply_markup=_keyboard,
                        parse_mode="HTML",
                    )
                except Exception:
                    # R2-SC-02 fail-safe: HTML send failed, try plain-text
                    log.warning(
                        "Ask-before-apply HTML send failed, trying plain-text",
                        exc_info=True,
                    )
                    try:
                        _plain_text = (
                            f"{t('skill.confirm_apply_question', _confirm_lang)}"
                            f"\n\n{_hyp.claim}"
                        )
                        await update.message.reply_text(
                            _plain_text,
                            reply_markup=_keyboard,
                        )
                    except Exception:
                        # Both HTML and plain-text failed: abort confirmation
                        # CRITICAL: must NOT fall through to streaming path
                        log.error(
                            "Ask-before-apply confirmation send failed completely "
                            "for hyp=%s user=%d. Aborting skill application.",
                            _hyp.hypothesis_id,
                            user_id,
                            exc_info=True,
                        )
                        # Clean up pending state so it doesn't linger
                        _pending_store.pop((user_id, chat_id, _hyp_id), None)
                        return
                log.info(
                    "Ask-before-apply: showing confirmation for hyp=%s user=%d",
                    _hyp.hypothesis_id,
                    user_id,
                )
                return
    except Exception:
        # If pre-check fails (matching logic, not send), proceed normally
        log.debug("Ask-before-apply pre-check failed, proceeding", exc_info=True)

    # T25: Background-Task pattern for streaming.
    # Instead of blocking the entire handler (which blocks the Update queue),
    # the streaming call runs as a background task. This allows /reset and other
    # commands to be processed while a long stream is running.
    # The per-user lock + global semaphore are acquired INSIDE the task to
    # serialize messages from the same user without blocking other updates.
    if (
        persistent_provider is not None
        and hasattr(persistent_provider, "query_streaming")
        and await persistent_provider.is_available()
    ):
        task = asyncio.create_task(
            _streaming_background_task(
                update=update,
                context=context,
                chat_service=chat_service,
                persistent_provider=persistent_provider,
                user_id=user_id,
                chat_id=chat_id,
                username=username,
                text=text,
                reply_to_text=reply_to_text,
                envelope=_msg_envelope,
            ),
            name=f"stream_{user_id}_{chat_id}_{_msg_envelope.request_id[:8]}",
        )
        # Register task to prevent garbage collection and enable cleanup
        _register_background_task(task, user_id, chat_id)
    else:
        # Legacy-Fallback: non-streaming (runs synchronously, short-lived)
        user_lock = _get_user_lock(user_id)
        async with user_lock:
            async with GLOBAL_CLAUDE_SEMAPHORE:
                await _handle_message_legacy(
                    update=update,
                    context=context,
                    chat_service=chat_service,
                    user_id=user_id,
                    chat_id=chat_id,
                    username=username,
                    text=text,
                    reply_to_text=reply_to_text,
                    envelope=_msg_envelope,
                )


async def _streaming_background_task(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_service: ChatService,
    persistent_provider: Any,
    user_id: int,
    chat_id: int,
    username: str | None,
    text: str,
    reply_to_text: str | None,
    *,
    envelope: RequestEnvelope,
) -> None:
    """T25: Background wrapper that acquires locks then runs streaming.

    This runs as an asyncio.Task so handle_message can return immediately,
    allowing /reset and other commands to be processed in parallel.
    The per-user lock ensures messages from the same user are serialized.
    """
    user_lock = _get_user_lock(user_id)
    async with user_lock:
        async with GLOBAL_CLAUDE_SEMAPHORE:
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
                envelope=envelope,
            )


async def reprocess_after_skill_confirmation(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    username: str | None,
    text: str,
    envelope: RequestEnvelope,
) -> None:
    """Re-process a user message after skill confirmation (Round-5 fix).

    Called from skill_commands._handle_skill_confirm_inline when the user
    clicks "Ja" on the ask-before-apply dialog. At this point the hypothesis
    has been promoted to 'active', so the skill matcher will find it and
    should_ask_user() returns False, allowing the skill instruction block
    to be injected into the LLM prompt.

    This function creates a background streaming task identical to what
    handle_message does, but without the ask-before-apply pre-flight
    (which would be redundant since the user just confirmed).

    Args:
        context: Telegram handler context.
        chat_id: Telegram chat ID.
        user_id: Telegram user ID.
        username: Telegram username.
        text: Original user message text.
        envelope: Original RequestEnvelope.
    """
    chat_service = _get_chat_service(context)
    persistent_provider = _get_persistent_provider(context)

    # Send typing indicator
    try:
        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    except Exception:
        log.debug("Typing indicator failed in reprocess", exc_info=True)

    if (
        persistent_provider is not None
        and hasattr(persistent_provider, "query_streaming")
        and await persistent_provider.is_available()
    ):
        # Build a minimal Update-like object for the streaming handler.
        # The streaming handler needs update.effective_chat for message
        # creation and update.message for legacy fallback error handling.
        # We use the bot's get_chat to get the Chat object.
        try:
            chat_obj = await context.bot.get_chat(chat_id)
        except Exception:
            log.error("Cannot get chat %d for skill reprocess", chat_id, exc_info=True)
            return

        # Create a FakeUpdate that provides the minimal interface needed
        # by _handle_message_streaming (effective_chat, message for error fallback).
        fake_update = _SkillReprocessUpdate(chat_obj)

        task = asyncio.create_task(
            _streaming_background_task(
                update=fake_update,
                context=context,
                chat_service=chat_service,
                persistent_provider=persistent_provider,
                user_id=user_id,
                chat_id=chat_id,
                username=username,
                text=text,
                reply_to_text=None,
                envelope=envelope,
            ),
            name=f"skill_reprocess_{user_id}_{chat_id}_{envelope.request_id[:8]}",
        )
        _register_background_task(task, user_id, chat_id)
    else:
        # Legacy fallback: non-streaming.
        # Cannot easily run without a full Update object; log and skip.
        log.warning(
            "Skill reprocess: no streaming provider available, "
            "user=%d chat=%d. Skill confirmed but response not sent.",
            user_id,
            chat_id,
        )


class _SkillReprocessUpdate:
    """Minimal Update-like object for re-processing after skill confirmation.

    Provides the interface that _handle_message_streaming needs:
      - effective_chat: Chat object (for send_chat_action, create_streaming_message)
      - message: a stub with reply_text for error fallback

    This avoids importing or constructing a full telegram.Update for a
    synthetic re-process. The streaming handler only accesses
    update.effective_chat and update.message.reply_text (for error fallback).
    """

    def __init__(self, chat):
        self.effective_chat = chat
        self.effective_user = None
        self.message = _StubMessage(chat)


class _StubMessage:
    """Stub message for error-fallback reply_text calls."""

    def __init__(self, chat):
        self._chat = chat

    async def reply_text(self, text, **kwargs):
        """Send error text as a regular message to the chat."""
        try:
            await self._chat.send_message(text)
        except Exception:
            log.debug("Stub reply_text failed", exc_info=True)


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
    *,
    envelope: RequestEnvelope,
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
    t_start = time.monotonic()
    streaming_chunks = 0
    final_text = ""
    had_error = False
    error_id = ""
    session: StreamingSession | None = None
    memory_entries_loaded = 0
    task_meta: dict[str, Any] = {}

    # EK-01: envelope is now passed in from handle_message (built before rate-limit)

    # Audit "started" entry (plan details added after plan creation below)
    audit_started: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "stream_started",
        "request_id": envelope.request_id,
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

        # T25: Register active session so /reset can cancel it
        session_key = (user_id, chat_id)
        with _active_sessions_lock:
            _active_streaming_sessions[session_key] = session

        # Phase 0 Commit 2: Resolve context via Execution Kernel
        # (replaces inline LanguageResolver instantiation)
        _kernel = _get_context_kernel(context)
        _exec_ctx = await _kernel.build(envelope)

        # Phase 0 Commit 3: Create ExecutionPlan before ChatService call
        _planner = _get_execution_planner(context)
        _exec_plan = _planner.plan_chat(_exec_ctx)

        # Phase 0 Commit 6: Structured audit event for execution plan
        write_raw_audit(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "execution_plan_created",
                "request_id": envelope.request_id,
                "user_id": user_id,
                "chat_id": chat_id,
                "channel": envelope.channel,
                "task_type": _exec_plan.task_type,
                "language": _exec_ctx.language.code,
                "language_source": _exec_ctx.language.source,
                "language_confidence": _exec_ctx.language.confidence,
                "provider_chain": _exec_plan.provider_chain,
                "memory_refs": list(_exec_plan.memory_used),
                "verifier_profile": _exec_plan.verifier_profile,
                "audit_required": _exec_plan.audit_required,
            }
        )

        # Text Guard: streaming diacritic filter (uses exec_ctx.language.code)
        from application.text_guard_service import TextGuardService

        _tg_service = TextGuardService()
        _stream_guard = _tg_service.get_streaming_guard(_exec_ctx.language.code)
        _text_guard = _tg_service.get_guard(_exec_ctx.language.code, mode="fix")

        # Create status session (R02-B, Phase 0 Commit 5: context-based)
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
                context=_exec_ctx,
            )

        # Typing keepalive parallel to stream
        keepalive = asyncio.create_task(
            _typing_keepalive(
                update.effective_chat,
                interval=TYPING_KEEPALIVE_INTERVAL_SECONDS,
            )
        )

        stream_iter = None
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
                context=_exec_ctx,
                plan=_exec_plan,
                cancel_event=session.cancel_event,  # T25: propagate for readline interrupt
            )
            async for event in stream_iter:
                # T25: Check cancellation before processing each event
                if session.is_cancelled:
                    log.info(
                        "Stream cancelled for user=%d chat=%d (after %d chunks)",
                        user_id,
                        chat_id,
                        streaming_chunks,
                    )
                    break

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
            # Explicitly close the async generator to release managed.lock
            # and avoid orphaned generators holding resources until GC.
            if stream_iter is not None:
                try:
                    await stream_iter.aclose()
                except Exception:  # nosec B110 - best-effort generator cleanup
                    pass  # generator may already be exhausted

        duration = time.monotonic() - t_start

        # EK-03: If cancelled, decide between user /stop (hard discard)
        # and StreamGuard abort (trigger non-streaming repair call).
        if session.is_cancelled:
            is_guard = task_meta.get("_guard_abort", False)

            if is_guard:
                # StreamGuard detected language drift. Trigger a
                # non-streaming repair call to produce a correct response.
                log.info(
                    "StreamGuard abort: triggering repair call "
                    "(user=%d chat=%d, partial=%d chars)",
                    user_id,
                    chat_id,
                    len(task_meta.get("_guard_abort_text", "")),
                )
                write_raw_audit(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event_type": "stream_guard_repair_triggered",
                        "request_id": envelope.request_id,
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "streaming_chunks": streaming_chunks,
                        "partial_chars": len(task_meta.get("_guard_abort_text", "")),
                        "duration_seconds": round(duration, 2),
                    }
                )
                # Non-streaming repair: re-send the user message through
                # the synchronous path which includes LCP enforcement.
                try:
                    repair_result = await chat_service.process_user_message(
                        text=text,
                        user_id=user_id,
                        chat_id=chat_id,
                        username=username,
                        system_prompt=_get_system_prompt(context),
                        provider_name=task_meta.get("_provider_name"),
                        reply_to_text=reply_to_text,
                        context=_exec_ctx,
                        plan=_exec_plan,
                    )
                    final_text = repair_result.response
                    await finalize_streaming(session, final_text)
                    log.info(
                        "StreamGuard repair succeeded: %d chars (user=%d)",
                        len(final_text),
                        user_id,
                    )
                except Exception as repair_err:
                    log.error(
                        "StreamGuard repair failed: %s (user=%d)",
                        repair_err,
                        user_id,
                    )
                    await abort_streaming(
                        session,
                        "Language correction failed. Please try again.",
                    )
                # process_user_message already saves history; do not
                # fall through to save_streaming_result (would double-save).
                return
            else:
                # User /stop or /reset: hard discard (original behavior).
                log.info(
                    "Stream cancelled (terminal): user=%d chat=%d, "
                    "discarding %d accumulated chars",
                    user_id,
                    chat_id,
                    len(session.accumulated_text or ""),
                )
                write_raw_audit(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event_type": "stream_cancelled",
                        "request_id": envelope.request_id,
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "streaming_chunks": streaming_chunks,
                        "accumulated_chars": len(session.accumulated_text or ""),
                        "duration_seconds": round(duration, 2),
                        **filter_task_meta(task_meta),
                    }
                )
                return  # hard exit: no fallback, no save, no finalize

        # Fallback: no final text but accumulated text available
        if not final_text and session.accumulated_text and not had_error:
            final_text = session.accumulated_text
            if _text_guard is not None:
                final_text = _text_guard.fix(final_text)
            await finalize_streaming(session, final_text)

        # Save history + audit (+ C-3 leakage check + LCP enforcement)
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
                request_id=envelope.request_id,
                language_code=task_meta.get("_language_code"),
                language_ctx=task_meta.get("_language_ctx"),
                user_model=task_meta.get("_user_model"),
                provider_name=task_meta.get("_provider_name"),
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
                "request_id": envelope.request_id,
                "user_id": user_id,
                "chat_id": chat_id,
                "username": username,
                "error_id": error_id,
                "duration_seconds": round(duration, 2),
                "streaming_chunks": streaming_chunks,
                "was_cold": was_cold,
                "subprocess_pid": subprocess_pid,
                **filter_task_meta(task_meta),
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
            "request_id": envelope.request_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "error_id": error_id,
            "duration_seconds": round(duration, 2),
            "error": "outer_exception",
            **filter_task_meta(task_meta),
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
    finally:
        # T25: Always deregister the active session
        with _active_sessions_lock:
            _active_streaming_sessions.pop((user_id, chat_id), None)


async def _handle_message_legacy(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_service: ChatService,
    user_id: int,
    chat_id: int,
    username: str | None,
    text: str,
    reply_to_text: str | None,
    *,
    envelope: RequestEnvelope,
) -> None:
    """Legacy message handler (pre-R04, non-streaming fallback).

    Phase 0 Commit 5: uses ExecutionContext for language, TextGuard,
    and ChatService call (no separate language resolution).
    """
    # EK-01: envelope passed in from handle_message (built before rate-limit)
    _kernel = _get_context_kernel(context)
    _exec_ctx = await _kernel.build(envelope)
    _planner = _get_execution_planner(context)
    _exec_plan = _planner.plan_chat(_exec_ctx)

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
            context=_exec_ctx,
            plan=_exec_plan,
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

    # Text Guard: fix diacritics (uses exec_ctx.language.code)
    from application.text_guard_service import TextGuardService

    _legacy_tg = TextGuardService()
    _legacy_guard = _legacy_tg.get_guard(_exec_ctx.language.code, mode="fix")
    _response = result.response
    if _legacy_guard is not None:
        _response = _legacy_guard.fix(_response)

    await send_response(update, _response)

    log.info(
        "Legacy response sent: %d chars in %.1fs",
        len(result.response),
        result.duration,
    )


@require_whitelist
async def handle_reset_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /reset. Clears conversation history and sticky language for this chat.

    Bug-Fix Round 3 (2026-05-27): /reset now waits for in-flight LLM
    responses to complete before clearing history. Previously /reset
    would silently kill the running response, leaving the user without
    an answer. Adds 30s timeout fallback so /reset cannot hang forever.
    """
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    # Round 3: Wait for active streaming session to COMPLETE (not cancel).
    # User spec: "Bot soll Antwort trotzdem fertig liefern, dann reset."
    # We wait up to 30s for the stream to finish naturally. If it does not
    # finish in time, we cancel as fallback to prevent /reset from hanging.
    session_key = (user_id, chat_id)
    with _active_sessions_lock:
        active_session = _active_streaming_sessions.get(session_key)
    if active_session is not None:
        # Wait for the stream to finish (up to 30s in 200ms increments)
        stream_completed = False
        for _ in range(150):  # 150 * 200ms = 30s
            await asyncio.sleep(0.2)
            with _active_sessions_lock:
                if _active_streaming_sessions.get(session_key) is None:
                    stream_completed = True
                    break
        if not stream_completed:
            # Timeout fallback: cancel the stream so /reset does not hang
            active_session.cancel()
            for _ in range(10):  # 10 * 100ms = 1s grace for cancellation
                await asyncio.sleep(0.1)
                with _active_sessions_lock:
                    if _active_streaming_sessions.get(session_key) is None:
                        break
            log.warning(
                "Reset: stream timed out after 30s, cancelled for user=%d chat=%d",
                user_id,
                chat_id,
            )
        else:
            log.info(
                "Reset: stream completed naturally for user=%d chat=%d",
                user_id,
                chat_id,
            )

    # Read language BEFORE reset (reset clears sticky language)
    raw_lang = await chat_service.get_chat_language(user_id, chat_id)
    lang = raw_lang or DEFAULT_LANGUAGE

    await chat_service.reset(user_id, chat_id)

    # Restore sticky language after reset (language preference survives /reset)
    if raw_lang is not None:
        await chat_service.set_chat_language(user_id, chat_id, raw_lang)

    reset_msg = t("reset.confirmation", lang)
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
async def handle_stop_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /stop. Cancels active stream but keeps conversation history.

    Unlike /reset, this does NOT clear conversation history or sticky language.
    """
    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    raw_lang = await chat_service.get_chat_language(user_id, chat_id)
    lang = raw_lang or DEFAULT_LANGUAGE

    session_key = (user_id, chat_id)
    with _active_sessions_lock:
        active_session = _active_streaming_sessions.get(session_key)

    if active_session is None:
        await update.message.reply_text(t("stop.no_active_stream", lang))
        log.info("Stop: no active stream for user=%d chat=%d", user_id, chat_id)
        log_command_audit(
            action="stop",
            user_id=user_id,
            chat_id=chat_id,
            username=user.username if user else None,
            details="no_active_stream",
        )
        return

    active_session.cancel()
    for _ in range(20):
        await asyncio.sleep(0.1)
        with _active_sessions_lock:
            if _active_streaming_sessions.get(session_key) is None:
                break

    await update.message.reply_text(t("stop.confirmation", lang))
    log.info("Stop: cancelled active stream for user=%d chat=%d", user_id, chat_id)
    log_command_audit(
        action="stop",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
    )


@require_whitelist
@typechecked
async def handle_lang_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /lang [code]. Sets the sticky language for this chat.

    Without argument: shows inline keyboard with all supported languages.
    With argument: sets language directly (e.g. /lang de).
    """
    chat_service = _get_chat_service(context)

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    args: list[str] = context.args or []
    if not args:
        # Show inline keyboard with all languages
        _usage_lang = (
            await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
        )
        keyboard = _build_lang_keyboard()
        await update.message.reply_text(
            t("lang.list_header", _usage_lang),
            reply_markup=keyboard,
        )
        return

    lang_code = args[0].lower().strip()
    if lang_code not in _SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(_SUPPORTED_LANGUAGES))
        _err_lang = (
            await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
        )
        await update.message.reply_text(
            t("lang.unknown", _err_lang, code=lang_code, supported=supported)
        )
        return

    await _apply_lang_change(update, chat_service, user_id, chat_id, lang_code, user)


def _build_lang_keyboard() -> InlineKeyboardMarkup:
    """Build an InlineKeyboardMarkup with all supported languages (3 columns)."""
    from i18n.domain.i18n import get_supported_languages

    languages = get_supported_languages()
    # Only show languages that are in _SUPPORTED_LANGUAGES, sorted by code
    languages = sorted(
        [lang for lang in languages if lang["code"] in _SUPPORTED_LANGUAGES],
        key=lambda x: x["code"],
    )

    buttons: list[InlineKeyboardButton] = []
    for lang in languages:
        label = f"{lang['native']} ({lang['code']})"
        buttons.append(
            InlineKeyboardButton(text=label, callback_data=f"lang_set:{lang['code']}")
        )

    # Arrange in rows of 3
    rows: list[list[InlineKeyboardButton]] = []
    for i in range(0, len(buttons), 3):
        rows.append(buttons[i : i + 3])

    return InlineKeyboardMarkup(rows)


@require_whitelist
@require_private_chat
async def handle_lang_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles lang_set:<code> callback from inline keyboard."""
    query = update.callback_query
    data: str = query.data or ""

    if not data.startswith("lang_set:"):
        return

    lang_code = data.split(":", 1)[1].strip()
    if lang_code not in _SUPPORTED_LANGUAGES:
        await query.answer(text="Unknown language", show_alert=False)  # i18n: ok
        return

    chat_service = _get_chat_service(context)
    user = query.from_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    await query.answer()
    await _apply_lang_change(update, chat_service, user_id, chat_id, lang_code, user)


async def _apply_lang_change(
    update: Update,
    chat_service: ChatService,
    user_id: int,
    chat_id: int,
    lang_code: str,
    user: Any,
) -> None:
    """Apply language change and send confirmation. Used by both /lang and callback."""
    from i18n.domain.i18n import get_supported_languages

    lang_meta = {lg["code"]: lg for lg in get_supported_languages()}

    # Remember old language for audit details
    old_lang = await chat_service.get_chat_language(user_id, chat_id) or "auto"

    await chat_service.set_chat_language(user_id, chat_id, lang_code)

    name = lang_meta.get(lang_code, {}).get("native", lang_code)
    lang_msg = t("lang.changed", lang_code, name=name, code=lang_code)

    # Determine reply target: regular message or callback query
    cb = update.callback_query
    if cb is not None and getattr(cb, "message", None) is not None:
        await cb.message.edit_text(lang_msg)
    else:
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
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    chat_service = _get_chat_service(context)
    _save_lang = (
        await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    )

    # Must be a reply to another message
    reply_msg = update.message.reply_to_message
    if reply_msg is None:
        await update.message.reply_text(t("bookmark.save_hint", _save_lang))
        return

    msg_id: int = reply_msg.message_id

    # Determine content: cache first, then message text
    content: str | None = get_cached_response(chat_id, msg_id)
    if content is None:
        content = reply_msg.text or ""
    if not content:
        content = "(content not available)"

    bookmark_service = _get_bookmark_service(context)
    was_saved, _user_message = bookmark_service.save_or_toggle_bookmark(
        user_id=user_id,
        username=username,
        chat_id=chat_id,
        message_id=msg_id,
        content=content,
    )
    _bm_text = t(
        "bookmark.saved" if was_saved else "bookmark.removed",
        _save_lang,
    )
    await update.message.reply_text(f"✓ {_bm_text}")  # i18n: ok
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
    _bm_chat_id: int = update.effective_chat.id if update.effective_chat else 0

    args: list[str] = context.args or []

    username: str | None = user.username if user else None

    chat_service = _get_chat_service(context)
    _bm_lang = (
        await chat_service.get_chat_language(user_id, _bm_chat_id) or DEFAULT_LANGUAGE
    )

    bookmark_service = _get_bookmark_service(context)

    # /bookmarks search <query>
    if len(args) >= 2 and args[0].lower() == "search":
        query_term = " ".join(args[1:])
        results = bookmark_service.search(user_id, query_term, limit=20)

        if not results:
            await update.message.reply_text(
                t("memory.search_no_results", _bm_lang, query=query_term)
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

        keyboard = build_bookmarks_keyboard(results, lang=_bm_lang)
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
        await update.message.reply_text(t("bookmark.list_empty", _bm_lang))
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

    keyboard = build_bookmarks_keyboard(bookmarks, lang=_bm_lang)
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
    lang = await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE

    _help_title = t("help.title", lang)
    _help_body = t("help.body", lang)
    help_text = f"{_help_title}\n\n{_help_body}"
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
    lang = await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    welcome_text = t("start.welcome", lang)
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
@lcp_aware
@typechecked
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
        await update.message.reply_text(t("errors.memory_not_initialized", "en"))
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    args: list[str] = context.args or []

    # Determine content
    content: str = ""
    reply_msg = update.message.reply_to_message

    chat_service = _get_chat_service(context)
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    _remember_lang = (
        await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    )

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
        await update.message.reply_text(t("remember.usage", _remember_lang))
        return

    # GAP-05 FIX: Check for prompt injection patterns before storing.
    # Memory entries are injected into every future system prompt, so
    # a malicious entry would achieve persistent prompt injection.
    _injection_detector = InjectionDetector()
    injection_match = _injection_detector.check(content)
    if injection_match is not None:
        # R7-BLOCKER-02: Do NOT log matched_text (user PII / injection payload).
        log.warning(
            "remember rejected for injection pattern=%s severity=%s user=%s chat=%s",
            injection_match.pattern_name,
            injection_match.severity,
            user_id,
            chat_id,
        )
        # R7-BLOCKER-02: write_raw_audit expects a single dict, not kwargs.
        # Do NOT include content_preview or matched_text (PII / payload).
        write_raw_audit(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "remember_injection_blocked",
                "user_id": user_id,
                "chat_id": chat_id,
                "pattern": injection_match.pattern_name,
                "severity": injection_match.severity,
            }
        )
        await update.message.reply_text(  # i18n: ok (security message, intentionally English-only for audit clarity)
            "This memory contains a pattern that looks like a prompt "
            "injection attempt and was rejected. If this is a false "
            "positive, please rephrase your memory entry."
        )
        return

    # BL-3: MemoryService.remember_episodic() contains SecretScanner gate.
    # Handler catches SecretBlockedError for i18n reply + audit.
    try:
        entry_id = memory_service.remember_episodic(user_id=user_id, content=content)
    except SecretBlockedError as exc:
        _first_secret = exc.matches[0]
        log.warning(  # nosemgrep: python-logger-credential-disclosure
            "remember blocked by secret scanner: user=%d pattern=%s layer=%d",
            user_id,
            _first_secret.pattern_name,
            _first_secret.layer,
        )
        write_raw_audit(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "remember_secret_blocked",
                "user_id": user_id,
                "chat_id": chat_id,
                "pattern": _first_secret.pattern_name,
                "layer": _first_secret.layer,
            }
        )
        secret_label = t(_first_secret.pattern_label_key, _remember_lang)
        await update.message.reply_text(
            t("remember.secret_blocked", _remember_lang, secret_type=secret_label)
        )
        return

    await update.message.reply_text(
        t("remember.saved", _remember_lang, entry_id=entry_id)
    )
    # A3.2: Log only content length, never cleartext content.
    log.info(
        "[remember] user=%d remembered: content_len=%d id=%s",
        user_id,
        len(content),
        entry_id,
    )
    log_command_audit(
        action="remember",
        user_id=user_id,
        chat_id=update.effective_chat.id if update.effective_chat else 0,
        username=user.username if user else None,
        entry_id=entry_id,
    )


async def _translate_memory_entries(
    entries: list[dict],
    target_lang: str,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int = 0,
    chat_id: int = 0,
) -> list[dict]:
    """Translate memory entries to the user's language for /memory display.

    T26: On-the-fly translation. Original entries in DB are never modified.
    Falls back to originals if translation service is unavailable or fails.

    Args:
        entries: Memory entry dicts (must have 'id' and 'content').
        target_lang: ISO 639-1 target language code.
        context: Telegram handler context (for ProviderRouter access).
        user_id: Telegram user ID (needed by claude_persistent provider).
        chat_id: Telegram chat ID (needed by claude_persistent provider).

    Returns:
        List of entry dicts with translated content (or originals on failure).
    """
    try:
        from application.memory_translation_service import translate_entries

        chat_service = _get_chat_service(context)
        router = chat_service.provider_router
        if router is None:
            return entries
        return await translate_entries(
            entries=entries,
            target_lang=target_lang,
            provider_router=router,
            user_id=user_id or None,
            chat_id=chat_id or None,
        )
    except Exception as exc:
        log.warning("Memory translation failed, showing originals: %s", exc)
        return entries


@require_whitelist
@require_private_chat
async def handle_memory_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /memory and /memory search <query>.

    /memory              Show last 10 episodic entries
    /memory search <q>   Search memory

    T26: Memory entries are translated on-the-fly to the user's
    current language before display. Originals in DB are never modified.
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text(t("errors.memory_not_initialized", "en"))
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    args: list[str] = context.args or []

    chat_service = _get_chat_service(context)
    _mem_lang = (
        await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    )

    # /memory search <query>
    if len(args) >= 2 and args[0].lower() == "search":
        query_term = " ".join(args[1:])
        results = memory_service.recall(user_id, query_term, layer="episodic")

        if not results:
            await update.message.reply_text(
                t("memory.search_no_results", _mem_lang, query=query_term)
            )
            return

        # T26: translate search results to user language
        results = await _translate_memory_entries(
            results[:10], _mem_lang, context, user_id=user_id, chat_id=chat_id
        )

        lines: list[str] = [
            t(
                "memory.search_header",
                _mem_lang,
                query=query_term,
                count=len(results),
            )
        ]
        for entry in results:
            lines.append(f"  [{entry['id']}] {entry['content'][:80]}")
        await update.message.reply_text("\n".join(lines))
        return

    # /memory (no arguments): show last 10
    entries = memory_service.list_recent(user_id, layer="episodic", limit=10)

    if not entries:
        await update.message.reply_text(t("memory.empty", _mem_lang))
        return

    # T26: translate entries to user language before display
    entries = await _translate_memory_entries(
        entries, _mem_lang, context, user_id=user_id, chat_id=chat_id
    )

    lines: list[str] = [t("memory.list_header", _mem_lang, count=len(entries))]
    for entry in entries:
        lines.append(f"  [{entry['id']}] {entry['content'][:80]}")
    await update.message.reply_text("\n".join(lines))


@require_whitelist
@require_private_chat
@lcp_aware
async def handle_forget_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /forget <entry_id>.

    Deletes a memory entry by its ID.
    """
    memory_service = _get_memory_service(context)
    if memory_service is None:
        await update.message.reply_text(t("errors.memory_not_initialized", "en"))
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    args: list[str] = context.args or []

    chat_service = _get_chat_service(context)
    _forget_lang = (
        await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    )

    if not args:
        await update.message.reply_text(t("forget.usage", _forget_lang))
        return

    entry_id = args[0].strip().strip("[]")
    deleted = memory_service.forget(user_id, entry_id)

    if deleted:
        forget_msg = t("forget.success", _forget_lang, entry_id=entry_id)
        await update.message.reply_text(forget_msg)
        log.info("User %d forgot memory: %s", user_id, entry_id)
    else:
        await update.message.reply_text(
            t("forget.not_found", _forget_lang, entry_id=entry_id)
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
        await update.message.reply_text(t("errors.rate_limiter_not_initialized", "en"))
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    chat_service = _get_chat_service(context)
    _usage_lang = (
        await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    )

    usage = rate_limiter.get_usage(user_id)

    if usage.profile == "unlimited":
        msg = (
            f"{t('usage.header', _usage_lang)}\n\n"
            f"{t('usage.profile_unlimited', _usage_lang)}"
        )
    else:
        profile_display = usage.profile.capitalize()

        # Reset times
        min_reset = f"{int(usage.minute_reset_seconds)}s"
        hour_reset_min = int(usage.hour_reset_seconds) // 60
        hour_reset = f"{hour_reset_min} Min"
        day_reset = "00:00"

        # Progress bars (10 chars wide)
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
            f"{t('usage.header', _usage_lang)}\n\n"
            f"{t('usage.body', _usage_lang, profile=profile_display, min_used=usage.minute_used, min_limit=usage.minute_limit, min_bar=min_bar, min_reset=min_reset, hour_used=usage.hour_used, hour_limit=usage.hour_limit, hour_bar=hour_bar, hour_reset=hour_reset, day_used=usage.day_used, day_limit=usage.day_limit, day_bar=day_bar, day_reset=day_reset)}"
        )

    await update.message.reply_text(msg)
    log_command_audit(
        action="usage",
        user_id=user_id,
        chat_id=chat_id,
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
        await update.message.reply_text(t("errors.rate_limiter_not_initialized", "en"))
        return

    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0
    args: list[str] = context.args or []

    chat_service = _get_chat_service(context)
    _sl_lang = (
        await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    )

    if not args:
        current = rate_limiter.get_user_profile(user_id)
        available = ", ".join(PROFILES.keys())
        await update.message.reply_text(
            t(
                "setlimit.current",
                _sl_lang,
                profile=current.capitalize(),
                available=available,
            )
        )
        return

    target_profile = args[0].lower().strip()

    # Unlimited: two-step confirmation
    if target_profile == "unlimited":
        if len(args) < 2 or args[1].lower() != "confirm":
            await update.message.reply_text(t("setlimit.confirm_unlimited", _sl_lang))
            return

    if target_profile not in PROFILES:
        available = ", ".join(PROFILES.keys())
        await update.message.reply_text(
            t(
                "setlimit.unknown_profile",
                _sl_lang,
                profile=target_profile,
                available=available,
            )
        )
        return

    old_profile = rate_limiter.get_user_profile(user_id)
    success = rate_limiter.set_user_profile(user_id, chat_id, target_profile)

    if success:
        limits = PROFILES[target_profile]
        if target_profile == "unlimited":
            confirm_msg = t(
                "setlimit.changed_unlimited",
                _sl_lang,
                old=old_profile.capitalize(),
            )
        else:
            confirm_msg = t(
                "setlimit.changed",
                _sl_lang,
                old=old_profile.capitalize(),
                new=target_profile.capitalize(),
                per_minute=str(limits["per_minute"]),
                per_hour=str(limits["per_hour"]),
                per_day=str(limits["per_day"]),
            )
        await update.message.reply_text(confirm_msg)
    else:
        await update.message.reply_text(t("errors.profile_change_failed", _sl_lang))

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

# _MODEL_STRINGS legacy dict removed: migrated to t("setmodel.*", lang) i18n system.


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
        await update.message.reply_text(t("errors.model_not_initialized", "en"))
        return

    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    slot_names = ", ".join(TaskSlot.all_names())

    args: list[str] = context.args or []
    if not args:
        aliases = model_service.list_available_aliases()
        alias_list = ", ".join(sorted(aliases.keys()))
        msg = (
            f"{t('setmodel.usage_hint', lang, slots=slot_names)}\n\n"
            f"{t('setmodel.available_aliases', lang, aliases=alias_list)}"
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
                    t("setmodel.unknown_slot", lang, input=slot_input, slots=slot_names)
                )
                return
            deleted = model_service.reset_user_model(user_id, slot=slot.value)
            if deleted:
                msg = t("setmodel.reset_slot_success", lang, slot=slot.value.upper())
            else:
                default_display = model_service.get_model_display_name(DEFAULT_MODEL)
                msg = t(
                    "setmodel.reset_nothing",
                    lang,
                    default_model=f"{default_display} ({DEFAULT_MODEL})",
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
                msg = t("setmodel.reset_all_success", lang, count=count)
            else:
                default_display = model_service.get_model_display_name(DEFAULT_MODEL)
                msg = t(
                    "setmodel.reset_nothing",
                    lang,
                    default_model=f"{default_display} ({DEFAULT_MODEL})",
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
                f"{t('setmodel.set_slot_success', lang, slot=slot.value.upper(), display_name=display_name, model_id=result)}\n"
                f"{t('setmodel.set_slot_note', lang, slot=slot.value.upper())}"
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
                f"{t('setmodel.unknown_model', lang, input=model_input)}\n"
                f"{t('setmodel.available_aliases', lang, aliases=alias_list)}"
            )
            await update.message.reply_text(msg)
        return

    # /setmodel <model> (global)
    success, result = model_service.set_user_model(user_id, first)
    if success:
        display_name = model_service.get_model_display_name(result)
        msg = (
            f"{t('setmodel.set_success', lang, display_name=display_name, model_id=result)}\n"
            f"{t('setmodel.set_success_note', lang)}"
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
            f"{t('setmodel.unknown_model', lang, input=first)}\n"
            f"{t('setmodel.available_aliases', lang, aliases=alias_list)}"
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
        await update.message.reply_text(t("errors.model_not_initialized", "en"))
        return

    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE

    deleted = model_service.reset_user_model(user_id)
    default_display = model_service.get_model_display_name(DEFAULT_MODEL)
    default_label = f"{default_display} ({DEFAULT_MODEL})"
    if deleted:
        msg = t("setmodel.reset_success", lang, default_model=default_label)
    else:
        msg = t("setmodel.reset_nothing", lang, default_model=default_label)
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
        await update.message.reply_text(t("errors.model_not_initialized", "en"))
        return

    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE

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
            line = f"  {t('setmodel.models_slot_line_override', lang, slot=slot_name, display_name=display)}"
        elif global_override:
            display = model_service.get_model_display_name(global_override)
            line = f"  {t('setmodel.models_slot_line_override', lang, slot=slot_name, display_name=display)}"
        elif slot.value in slot_defaults:
            display = model_service.get_model_display_name(slot_defaults[slot.value])
            line = f"  {t('setmodel.models_slot_line_default', lang, slot=slot_name, display_name=display)}"
        else:
            display = model_service.get_model_display_name(DEFAULT_MODEL)
            line = f"  {t('setmodel.models_slot_line_default', lang, slot=slot_name, display_name=display)}"

        slot_lines.append(line)

    msg = (
        f"{t('setmodel.models_header', lang)}\n"
        + "\n".join(slot_lines)
        + f"\n\n{t('setmodel.models_change_hint', lang)}"
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
    """Handles /settings. Opens the v2 hierarchical inline keyboard settings menu.

    Shows the 6-category main menu (Language, Model, Debate, Rate-Limit,
    Personality, Timezone). All interactions handled by
    settings_callbacks.handle_settings_v2_callback and
    settings_callbacks.handle_settings_callback.

    Legacy power-user shortcuts (/setmodel, /lang, /setlimit, /resetmodel)
    remain fully functional in parallel.
    """
    chat_service = _get_chat_service(context)
    user = update.effective_user
    user_id: int = user.id if user else 0
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    lang = await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE

    from presentation.settings_callbacks import build_v2_main_menu

    text, keyboard = build_v2_main_menu(lang)
    await update.message.reply_text(text, reply_markup=keyboard, parse_mode="HTML")
    log_command_audit(
        action="settings_v2",
        user_id=user_id,
        chat_id=chat_id,
        username=user.username if user else None,
        details="opened v2 main menu",
    )


# ---------------------------------------------------------------------------
# /debate Command (R10: Multi-AI-Debate)
# ---------------------------------------------------------------------------

# Provider display names for formatted output (generic fallback)
_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "claude_persistent": "\U0001f916 Claude",
    "claude": "\U0001f916 Claude",
    "ollama_local": "\U0001f999 Llama (lokal)",
    "openai": "\U0001f4a1 OpenAI",
    "gemini": "✨ Gemini",
    "mistral": "\U0001f32c️ Mistral",
}

# Model ID -> specific display name (more informative than generic provider name)
_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "claude-opus-4-7": "\U0001f916 Claude Opus 4.7",
    "claude-sonnet-4-6": "\U0001f916 Claude Sonnet 4.6",
    "claude-haiku-4-5-20251001": "\U0001f916 Claude Haiku 4.5",
}


# i18n strings for Debate output (DE default, EN prepared for future activation)
def _get_debate_strings(lang: str = "en") -> dict[str, str]:
    """Returns debate i18n strings for the given language via t().

    Args:
        lang: ISO-639-1 language code.

    Returns:
        Dict of string keys to localized values.
    """
    return {
        "header": t("debate.header", lang),
        "question_label": t("debate.question_label", lang),
        "no_providers": t("debate.no_providers", lang),
        "errors_label": t("debate.errors_label", lang),
        "recommendation_label": t("debate.recommendation_label", lang),
        "strongest_contribution": t("debate.strongest_contribution", lang),
        "tie_result": t("debate.tie_result", lang),
        "synthesis_header": t("debate.synthesis_header", lang),
        "consensus_header": t("debate.consensus_header", lang),
        "detail_header": t("debate.detail_header", lang),
        "single_provider_hint": t("debate.single_provider_hint", lang),
        "quality_warning_prefix": "⚠️",
        "errors_section": f"⚠️ {t('debate.errors_label', lang)}:",
    }


def _format_debate_result(result: Any, lang: str = "en") -> str:
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
        lang: Language for labels (default: "en").

    Returns:
        Formatted text for Telegram.
    """
    s = _get_debate_strings(lang)
    lines: list[str] = []

    # Build provider -> display name mapping (prefer specific model names)
    provider_models: dict[str, str] = getattr(result, "provider_models", {})

    def _display_name(provider_name: str) -> str:
        """Get the most specific display name for a provider."""
        model_id = provider_models.get(provider_name)
        if model_id and model_id in _MODEL_DISPLAY_NAMES:
            return _MODEL_DISPLAY_NAMES[model_id]
        return _PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)

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
        winner_display = _display_name(result.final_verdict.winner)
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
        display_name = _display_name(provider_name)
        lines.append(f"{display_name}:")
        lines.append(response_text.strip())
        lines.append("")

    # --- Block 6: Pro/Contra je Provider (Analyse der Originale) ---
    if result.final_verdict is not None and result.final_verdict.evaluations:
        lines.append("━" * 20)
        for evaluation in result.final_verdict.evaluations:
            eval_display = _display_name(evaluation.provider)
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
            display_name = _display_name(provider_name)
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
@lcp_aware
@typechecked
async def handle_debate_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles /debate <question>. Multi-AI debate feature (R10).

    Phase 0 Commit 4: migrated to Execution Kernel.
    Language is resolved from the actual question text via ContextKernel,
    not just from sticky/default. Provider and judge prompts come from
    the InstructionCompiler for consistent language handling.

    Queries multiple providers in parallel and shows answers side-by-side.
    """
    from application.debate_orchestrator import DebateOrchestrator
    from application.execution import InstructionCompiler

    user = update.effective_user
    user_id: int = user.id if user else 0
    username: str | None = user.username if user else None
    chat_id: int = update.effective_chat.id if update.effective_chat else 0

    # Extract question from command arguments (needed early for usage check)
    args: list[str] = context.args or []

    # For the usage-hint fallback, get sticky language (fast, no detection)
    chat_service = _get_chat_service(context)
    _fallback_lang = (
        await chat_service.get_chat_language(user_id, chat_id) or DEFAULT_LANGUAGE
    )

    if not args:
        await update.message.reply_text(t("debate.usage", _fallback_lang))
        return

    question = " ".join(args)

    # Build RequestEnvelope from the debate question (not from /debate command text)
    envelope = RequestEnvelope.from_debate_command(
        user_id=user_id,
        chat_id=chat_id,
        question=question,
        username=username,
    )

    # EK-01: Universal audit anchor for debate path
    write_raw_audit(
        {
            "timestamp": envelope.timestamp_utc.isoformat(),
            "event_type": "request_received",
            "request_id": envelope.request_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "channel": envelope.channel,
            "command": envelope.command,
        }
    )

    # EK-02 FIX: Rate-limit check BEFORE ContextKernel.build to prevent
    # rejected requests from mutating sticky language via LanguageResolver.
    rate_limiter = _get_rate_limiter(context)
    if rate_limiter is not None:
        result_rl: RateLimitResult = rate_limiter.check_and_consume(user_id)
        if not result_rl.allowed:
            # Use sticky language (read-only, already fetched above) for reject message
            await update.message.reply_text(
                t("debate.rate_limit_short", _fallback_lang)
            )
            write_raw_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "rate_limit_exceeded",
                    "request_id": envelope.request_id,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "username": username,
                    "command": "debate",
                    "profile": result_rl.profile,
                    "period": result_rl.period,
                }
            )
            return

    # Resolve ExecutionContext: language comes from the question text
    # (only reached if rate limit allows the request)
    context_kernel = _get_context_kernel(context)
    exec_ctx = await context_kernel.build(envelope)

    # Build ExecutionPlan for debate
    execution_planner = _get_execution_planner(context)
    exec_plan = execution_planner.plan_debate(exec_ctx)

    # Phase 0 Commit 6: Structured audit event for execution plan
    write_raw_audit(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "execution_plan_created",
            "request_id": envelope.request_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "channel": envelope.channel,
            "task_type": exec_plan.task_type,
            "language": exec_ctx.language.code,
            "language_source": exec_ctx.language.source,
            "language_confidence": exec_ctx.language.confidence,
            "provider_chain": exec_plan.provider_chain,
            "memory_refs": list(exec_plan.memory_used),
            "verifier_profile": exec_plan.verifier_profile,
            "audit_required": exec_plan.audit_required,
        }
    )

    # Effective language from the resolved context
    resolved_lang = exec_ctx.language.code

    # Send status message with resolved language
    status_msg = await update.message.reply_text(
        f"\U0001f3af {t('debate.querying', resolved_lang)}"
    )

    # Typing keepalive during debate
    keepalive = asyncio.create_task(
        _typing_keepalive(
            update.effective_chat,
            interval=TYPING_KEEPALIVE_INTERVAL_SECONDS,
        )
    )

    try:
        # Create InstructionCompiler for the orchestrator
        instruction_compiler = InstructionCompiler()

        # Get or create DebateOrchestrator with InstructionCompiler + LCP enforcement
        _debate_enforcement = context.bot_data.get("language_enforcement")
        orchestrator = DebateOrchestrator(
            provider_router=chat_service.provider_router,
            instruction_compiler=instruction_compiler,
            language_enforcement=_debate_enforcement,
        )

        # Resolve user model: debate should respect /setmodel preference
        user_model: str | None = None
        if chat_service.model_service is not None:
            user_model = chat_service.model_service.get_user_model(user_id)

        debate_result = await orchestrator.debate(
            question=question,
            user_id=user_id,
            chat_id=chat_id,
            user_lang=resolved_lang,
            envelope=envelope,
            context=exec_ctx,
            plan=exec_plan,
            model=user_model,
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

    # Format and send result (use resolved language)
    formatted = _format_debate_result(debate_result, lang=resolved_lang)

    # Text Guard: fix diacritics in debate output before sending
    from application.text_guard_service import TextGuardService

    _debate_tg = TextGuardService()
    _debate_guard = _debate_tg.get_guard(resolved_lang, mode="fix")
    if _debate_guard is not None:
        formatted = _debate_guard.fix(formatted)

    formatted = sanitize_telegram_slashes(formatted)
    await send_response(update, formatted)

    # Save debate turns to conversation history so follow-up messages
    # have context. Only the synthesis is stored (not per-provider details)
    # to keep the context window lean.
    _synthesis_text = ""
    if debate_result.final_verdict and debate_result.final_verdict.synthesis:
        _synthesis_text = debate_result.final_verdict.synthesis
    elif debate_result.consensus_analysis:
        _synthesis_text = debate_result.consensus_analysis
    else:
        # Fallback: first provider response (better than nothing)
        _first_resp = next(iter(debate_result.responses.values()), "")
        _synthesis_text = _first_resp
    await chat_service.save_debate_turns(user_id, chat_id, question, _synthesis_text)

    # Audit log (enriched with kernel data)
    write_raw_audit(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "debate",
            "request_id": exec_ctx.request_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "question_length": len(question),
            "language": resolved_lang,
            "language_source": exec_ctx.language.source,
            "providers_queried": debate_result.providers_queried,
            "providers_responded": list(debate_result.responses.keys()),
            "providers_errored": list(debate_result.errors.keys()),
            "duration_seconds": round(debate_result.duration_seconds, 2),
            "plan": exec_plan.to_audit_dict(),
        }
    )

    log.info(
        "Debate completed for user %s: %d providers, %.1fs (lang=%s, source=%s)",
        username,
        len(debate_result.responses),
        debate_result.duration_seconds,
        resolved_lang,
        exec_ctx.language.source,
    )
