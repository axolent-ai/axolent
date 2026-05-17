"""Streaming handler: coordinates token streaming to Telegram edits.

Converts StreamEvents from ClaudePersistentProvider into
Telegram message edits. Rate-limited with adaptive throttle.

Features:
    * Aggregates tokens until the next edit time
    * Burst-mode: first 5 edits are fast (0.2s), then gradually slows
    * Subsequent edits follow a graduated throttle curve up to 1.5s
    * On final result: last edit with complete text + HTML formatting
    * Intermediate edits: markdown is rendered live, incomplete tokens
      at the end are safely trimmed (Option A: smart-trim)
    * Final edit converts markdown to Telegram HTML via domain.markdown
    * For responses >4096 chars: multi-message split at sensible boundaries
    * Telegram API errors are silently swallowed (UX > crash)
    * Adaptive flood control: on Telegram 429 (RetryAfter), the session
      pauses, intermediate edits are skipped, throttle is increased
      exponentially and recovers after successful edits.
    * Final edits have highest priority and are retried on 429.
    * AXOLENT_STREAMING_MODE=local disables all throttling (desktop app).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from domain.markdown import markdown_to_telegram_html, strip_markdown

if TYPE_CHECKING:
    from telegram import Message

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streaming mode (env-driven)
# ---------------------------------------------------------------------------
# "telegram" (default): burst-then-throttle for Telegram rate limits
# "local": no throttling at all (desktop app, direct socket)
STREAMING_MODE: str = os.environ.get("AXOLENT_STREAMING_MODE", "telegram").lower()

# ---------------------------------------------------------------------------
# Burst-mode throttle curve constants
# ---------------------------------------------------------------------------
# Edits 1-5:  fast burst (user sees immediate response)
BURST_PHASE_END: int = 5
BURST_THROTTLE: float = 0.2

# Edits 6-10: linear ramp from MID_THROTTLE_START to MID_THROTTLE_END
MID_PHASE_START: int = 6
MID_PHASE_END: int = 10
MID_THROTTLE_START: float = 0.4
MID_THROTTLE_END: float = 1.0

# Edits 11-20: linear ramp from RAMP_THROTTLE_START to RAMP_THROTTLE_END
RAMP_PHASE_START: int = 11
RAMP_PHASE_END: int = 20
RAMP_THROTTLE_START: float = 1.0
RAMP_THROTTLE_END: float = 1.5

# Edits 21+: stable at DEFAULT_THROTTLE
STABLE_THROTTLE: float = 1.5

# ---------------------------------------------------------------------------
# Adaptive throttle / flood control constants
# ---------------------------------------------------------------------------
# R04 Round 4: Adaptive flood control. Live stress test with 308 streaming chunks
# triggered cascading Telegram 429s. Backoff factor 2.0, recovery 0.7
# after 5 successful edits. Final edits have priority and are retried on
# 429. Empirically validated with 0 errors on a 15,678-char response.

# Default throttle for intermediate edits (seconds)
DEFAULT_THROTTLE: float = 1.5

# Maximum throttle after repeated 429s (seconds)
MAX_THROTTLE: float = 10.0

# Factor for throttle increase on 429
THROTTLE_BACKOFF_FACTOR: float = 2.0

# After N successful edits, throttle is gradually reduced
THROTTLE_RECOVERY_AFTER: int = 5

# Factor for throttle reduction on recovery
THROTTLE_RECOVERY_FACTOR: float = 0.7

# Maximum retries for final edits on 429
FINAL_EDIT_MAX_RETRIES: int = 2

# Minimal throttle for local mode (prevents asyncio starvation)
LOCAL_MODE_THROTTLE: float = 0.0

# ---------------------------------------------------------------------------
# Existing constants
# ---------------------------------------------------------------------------

# Rate limiting: Telegram allows max ~30 edits/min per chat
# Default throttle is now adaptively controlled (see above)
EDIT_INTERVAL_SECONDS: float = DEFAULT_THROTTLE

# Legacy: first-edit delay (superseded by burst-mode; kept for reference)
FIRST_EDIT_DELAY_SECONDS: float = 1.5

# Maximum message length for Telegram (4096 chars)
TELEGRAM_MAX_LENGTH: int = 4096

# Buffer for part markers ("(2/3)") and safety margin
_SPLIT_SAFETY_MARGIN: int = 30


@dataclass
class StreamingSession:
    """State of a running streaming session.

    Attributes:
        message: The Telegram message being edited (current active part).
        accumulated_text: Text collected so far (CURRENT PART only).
        last_edit_time: Timestamp of the last edit.
        edit_count: Number of edits so far.
        started_at: Session start time.
        is_first_edit: Whether no edit has been sent yet.
        _last_edit_html: Last sent edit text (for duplicate detection).
        _paused_until: Monotonic timestamp until which edits are paused (flood control).
        _current_throttle: Current adaptive edit interval in seconds.
        _consecutive_success: Counter of successful edits since last 429.
        _edits_sent: Number of edits successfully sent (for burst-mode curve).
        _backoff_active: Whether backoff has raised throttle above the curve.
        cancel_event: When set, the streaming loop should stop immediately.
            Used by /reset to cancel an active stream before clearing state.
        part_count: Number of message parts sent so far (1 = first/only).
        previous_parts: Finalized text for each completed part.
        current_part_offset: Character offset into the full accumulated text
            where the current part starts.
        _full_accumulated_text: Complete text across ALL parts (for history/audit).
    """

    message: "Message"
    accumulated_text: str = ""
    last_edit_time: float = 0.0
    edit_count: int = 0
    started_at: float = 0.0
    is_first_edit: bool = True
    _last_edit_html: str = ""
    _paused_until: float = 0.0
    _current_throttle: float = field(default_factory=lambda: DEFAULT_THROTTLE)
    _consecutive_success: int = 0
    _edits_sent: int = 0
    _backoff_active: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    # T23: Live multi-message rollover state
    part_count: int = 1
    previous_parts: list[str] = field(default_factory=list)
    current_part_offset: int = 0
    _full_accumulated_text: str = ""

    @property
    def is_cancelled(self) -> bool:
        """Check whether cancellation has been requested."""
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        """Request cancellation of this streaming session."""
        self.cancel_event.set()

    @property
    def full_text(self) -> str:
        """Return the complete accumulated text across all parts."""
        return self._full_accumulated_text


def _compute_base_throttle(edits_sent: int) -> float:
    """Compute the base throttle for the burst-mode curve.

    The curve is:
        edit 1-5:   0.2s (burst)
        edit 6-10:  linear 0.4s -> 1.0s
        edit 11-20: linear 1.0s -> 1.5s
        edit 21+:   stable 1.5s

    For local mode: always returns LOCAL_MODE_THROTTLE (0.0).

    Args:
        edits_sent: Number of edits already sent in this session.

    Returns:
        Base throttle interval in seconds.
    """
    if STREAMING_MODE == "local":
        return LOCAL_MODE_THROTTLE

    # Next edit number (1-indexed)
    next_edit = edits_sent + 1

    if next_edit <= BURST_PHASE_END:
        return BURST_THROTTLE

    if next_edit <= MID_PHASE_END:
        # Linear interpolation from MID_THROTTLE_START to MID_THROTTLE_END
        progress = (next_edit - MID_PHASE_START) / (MID_PHASE_END - MID_PHASE_START)
        return MID_THROTTLE_START + progress * (MID_THROTTLE_END - MID_THROTTLE_START)

    if next_edit <= RAMP_PHASE_END:
        # Linear interpolation from RAMP_THROTTLE_START to RAMP_THROTTLE_END
        progress = (next_edit - RAMP_PHASE_START) / (RAMP_PHASE_END - RAMP_PHASE_START)
        return RAMP_THROTTLE_START + progress * (
            RAMP_THROTTLE_END - RAMP_THROTTLE_START
        )

    return STABLE_THROTTLE


def _get_effective_throttle(session: StreamingSession) -> float:
    """Get the effective throttle for the next edit.

    If backoff is active (429 occurred), use _current_throttle (which was
    doubled by backoff). Otherwise, use the burst-mode curve.

    Args:
        session: The current StreamingSession.

    Returns:
        Effective throttle interval in seconds.
    """
    if STREAMING_MODE == "local":
        return LOCAL_MODE_THROTTLE

    base = _compute_base_throttle(session._edits_sent)

    # If backoff raised the throttle above curve, use backoff value
    if session._backoff_active and session._current_throttle > base:
        return session._current_throttle

    return base


async def create_streaming_message(chat: Any) -> "Message":
    """Create the initial placeholder message for streaming.

    Args:
        chat: Telegram chat object.

    Returns:
        The sent message (will be edited later).
    """
    return await chat.send_message("...")


async def process_streaming_edit(
    session: StreamingSession,
    new_text: str,
) -> None:
    """Add new text and edit the message if needed.

    Rate-limited with burst-mode curve: fast at start, gradually slower.
    First edit only after a short delay (burst-mode: 0.2s, not 1.5s).
    During flood control pause, intermediate edits are skipped.
    In local mode: no throttling at all.

    T23: Live multi-message rollover. When the current part's HTML
    approaches 4096 chars, the current message is finalized and a new
    message is sent as the next part. Subsequent edits go to the new message.

    Args:
        session: The current StreamingSession.
        new_text: New incremental text.
    """
    session.accumulated_text += new_text
    session._full_accumulated_text += new_text
    now = time.monotonic()

    # Local mode: skip first-edit delay entirely
    if STREAMING_MODE == "local":
        session.is_first_edit = False
        await _do_edit(session)
        # T23: Check rollover after edit
        await _check_live_rollover(session)
        return

    # First edit: wait until enough text has accumulated (burst-mode delay)
    if session.is_first_edit:
        elapsed = now - session.started_at
        # In burst mode, first edit fires after just BURST_THROTTLE
        first_delay = BURST_THROTTLE if session._edits_sent == 0 else BURST_THROTTLE
        if elapsed < first_delay:
            return
        session.is_first_edit = False

    # Flood control pause: intermediate edits are skipped
    if session._paused_until and now < session._paused_until:
        return

    # Rate limiting: use burst-mode curve (or backoff if active)
    effective_throttle = _get_effective_throttle(session)
    time_since_edit = now - session.last_edit_time
    if time_since_edit < effective_throttle:
        return

    await _do_edit(session)

    # T23: Check if we need to rollover to a new message part
    await _check_live_rollover(session)


# ---------------------------------------------------------------------------
# T23: Live Multi-Message Rollover
# ---------------------------------------------------------------------------
# Threshold at which we trigger a rollover to a new message.
# Below TELEGRAM_MAX_LENGTH to leave room for part markers and HTML expansion.
_ROLLOVER_THRESHOLD: int = 3900


def _find_rollover_boundary(text: str, threshold: int = _ROLLOVER_THRESHOLD) -> int:
    """Find a safe split point near the threshold for live rollover.

    Searches backward from threshold for a safe boundary:
    1. Paragraph break (double newline)
    2. Single newline
    3. Sentence end
    4. Word boundary (space)

    Returns 0 if no safe boundary found (should not trigger rollover).

    Args:
        text: The accumulated markdown text for the current part.
        threshold: The target threshold to search near.

    Returns:
        Split position, or 0 if no safe boundary found.
    """
    if len(text) < threshold:
        return 0

    search_text = text[:threshold]

    # Priority 1: paragraph break (double newline)
    pos = search_text.rfind("\n\n")
    if pos > threshold // 2:
        candidate = pos + 2
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Priority 2: single newline
    pos = search_text.rfind("\n")
    if pos > threshold // 2:
        candidate = pos + 1
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Priority 3: sentence end
    sentence_end = None
    for m in re.finditer(r"[.!?]\s", search_text):
        if m.end() > threshold // 2:
            sentence_end = m.end()
    if sentence_end and _is_safe_markdown_position(text, sentence_end):
        return sentence_end

    # Priority 4: word boundary
    pos = search_text.rfind(" ")
    if pos > threshold // 2:
        candidate = pos + 1
        if _is_safe_markdown_position(text, candidate):
            return candidate

    return 0


async def _check_live_rollover(session: StreamingSession) -> None:
    """Check if the current part needs a live rollover to a new message.

    Triggered after each successful edit. If the current part's text
    exceeds _ROLLOVER_THRESHOLD and a safe boundary exists, the current
    message is finalized with the text up to the boundary, and a new
    message is sent to continue streaming.

    Args:
        session: The current StreamingSession.
    """
    current_text = session.accumulated_text
    if len(current_text) < _ROLLOVER_THRESHOLD:
        return

    # Check HTML length (which is what Telegram actually limits)
    safe_end = find_safe_markdown_end(current_text)
    if safe_end <= 0:
        return

    html_check = markdown_to_telegram_html(current_text[:safe_end])
    if len(html_check) < _ROLLOVER_THRESHOLD:
        return

    # Find a safe boundary to split at
    split_pos = _find_rollover_boundary(current_text)
    if split_pos == 0:
        return

    # Finalize current part: split text at boundary
    part_text = current_text[:split_pos]
    remaining_text = current_text[split_pos:]

    # First: try to send a new message for the continuation.
    # If this fails, we abort the rollover without modifying the current message.
    try:
        new_msg = await session.message.chat.send_message("...")
    except Exception as e:
        log.warning("T23: Failed to send rollover message, skipping: %s", e)
        return  # Abort rollover, current message continues growing

    # New message created successfully. Now finalize the current message.
    part_html = markdown_to_telegram_html(part_text)
    try:
        await _send_html_with_fallback(session.message, part_html, part_text)
        session._last_edit_html = part_html
        session.last_edit_time = time.monotonic()
        _record_edit_success(session)
    except Exception as e:
        retry_after = _is_retry_after(e)
        if retry_after is not None:
            _apply_flood_backoff(session, retry_after)
        else:
            _handle_edit_error(e)
        # Even if edit fails, continue with rollover (new msg already exists)

    # Record the completed part and switch to new message
    session.previous_parts.append(part_text)
    session.part_count += 1
    session.message = new_msg
    session.accumulated_text = remaining_text
    session.current_part_offset += split_pos
    session._last_edit_html = ""
    session.is_first_edit = False
    session.last_edit_time = time.monotonic()
    log.info(
        "T23: Live rollover to part %d (split at %d chars)",
        session.part_count,
        split_pos,
    )


async def finalize_streaming(session: StreamingSession, final_text: str) -> str:
    """Finalize the streaming session with the complete text.

    T23: If live rollover already happened, only finalizes the LAST
    (current) part. Previous parts are already finalized in-place.

    For short responses (<= 4096 chars) without rollover: one edit with HTML.
    For long responses without rollover: multi-message split in finalize.

    Args:
        session: The current StreamingSession.
        final_text: The complete response text.

    Returns:
        The final text (untruncated, for history storage).
    """
    # T23: If rollover already happened, we only need to finalize the last part.
    if session.part_count > 1:
        # The final_text is the FULL text. Extract only the current part's portion.
        # current_part_offset marks where the current part starts in the full text.
        current_part_text = final_text[session.current_part_offset :]
        session.accumulated_text = current_part_text

        html_text = markdown_to_telegram_html(current_part_text)
        if len(html_text) <= TELEGRAM_MAX_LENGTH:
            await _do_edit_html(session)
        else:
            # Current part itself is still too long (edge case): split it
            await _finalize_multi_message(session, current_part_text)

        # Previous parts were sent without markers during stream.
        # We don't re-edit them with markers to avoid rate-limits — the
        # live-streaming UX is more important than cosmetic markers.
        return final_text

    # Standard path: no rollover happened
    session.accumulated_text = final_text

    # HTML-converted text determines whether split is needed
    html_text = markdown_to_telegram_html(final_text)

    if len(html_text) <= TELEGRAM_MAX_LENGTH:
        await _do_edit_html(session)
        return final_text

    # Long response: multi-message split
    await _finalize_multi_message(session, final_text)
    return final_text


async def _finalize_multi_message(
    session: StreamingSession,
    full_text: str,
) -> None:
    """Split a long response into multiple Telegram messages.

    Strategy:
        1. Split plain text (markdown) at sensible boundaries
        2. Convert each part to HTML separately (so tags close correctly)
        3. Part 1 as edit of the existing streaming message
        4. Parts 2+ as new messages in the chat

    Multi-message parts belong to the final phase and receive
    the same RetryAfter handling as final edits.

    Args:
        session: The current StreamingSession.
        full_text: The complete markdown text.
    """
    parts = split_text_for_telegram(full_text)
    total = len(parts)

    for i, part in enumerate(parts):
        part_num = i + 1
        if total > 1:
            marker = f"\n\n({part_num}/{total})"
        else:
            marker = ""

        html_part = markdown_to_telegram_html(part + marker)

        if i == 0:
            await _send_final_edit_with_retry(
                session, html_part, part + marker, part_num, total
            )
        else:
            await _send_final_message_with_retry(
                session, html_part, part + marker, part_num, total
            )


async def _send_final_edit_with_retry(
    session: StreamingSession,
    html_text: str,
    plain_source: str,
    part_num: int,
    total: int,
) -> None:
    """Edit the first message in a multi-message split with retry on 429.

    After exhausted retries: fallback to send_message (like _do_edit_html),
    so part 1 does not stay stuck in the placeholder.

    Args:
        session: The current StreamingSession.
        html_text: Fully converted HTML text.
        plain_source: Original markdown for strip_markdown fallback.
        part_num: Part number (for logging).
        total: Total number of parts (for logging).
    """
    for attempt in range(1 + FINAL_EDIT_MAX_RETRIES):
        try:
            await _send_html_with_fallback(session.message, html_text, plain_source)
            session.last_edit_time = time.monotonic()
            session.edit_count += 1
            _record_edit_success(session)
            return
        except Exception as e:
            retry_after = _is_retry_after(e)
            if retry_after is not None and attempt < FINAL_EDIT_MAX_RETRIES:
                _apply_flood_backoff(session, retry_after)
                log.info(
                    "Multi-message edit part %d/%d: 429, waiting %ds",
                    part_num,
                    total,
                    retry_after,
                )
                await asyncio.sleep(retry_after)
                continue
            if retry_after is not None:
                log.error(
                    "Multi-message edit part %d/%d: 429 after %d retries, "
                    "falling back to send_message",
                    part_num,
                    total,
                    FINAL_EDIT_MAX_RETRIES,
                )
                plain_text = strip_markdown(plain_source)
                try:
                    await session.message.chat.send_message(plain_text)
                except Exception as fb_e:
                    log.error(
                        "Multi-message edit fallback send_message "
                        "part %d/%d failed: %s",
                        part_num,
                        total,
                        fb_e,
                    )
                return
            _handle_edit_error(e)
            return


async def _send_final_message_with_retry(
    session: StreamingSession,
    html_text: str,
    plain_source: str,
    part_num: int,
    total: int,
) -> None:
    """Send a follow-up message in a multi-message split with retry on 429.

    Args:
        session: The current StreamingSession.
        html_text: Fully converted HTML text.
        plain_source: Original markdown for strip_markdown fallback.
        part_num: Part number (for logging).
        total: Total number of parts (for logging).
    """
    for attempt in range(1 + FINAL_EDIT_MAX_RETRIES):
        try:
            await session.message.chat.send_message(html_text, parse_mode="HTML")
            _record_edit_success(session)
            return
        except Exception as e:
            retry_after = _is_retry_after(e)
            if retry_after is not None and attempt < FINAL_EDIT_MAX_RETRIES:
                _apply_flood_backoff(session, retry_after)
                log.info(
                    "Multi-message send part %d/%d: 429, waiting %ds",
                    part_num,
                    total,
                    retry_after,
                )
                await asyncio.sleep(retry_after)
                continue

            error_str = str(e).lower()
            if "can't parse entities" in error_str or "bad request" in error_str:
                log.warning(
                    "HTML send part %d/%d failed, falling back to plain: %s",
                    part_num,
                    total,
                    e,
                )
                plain = strip_markdown(plain_source)
                try:
                    await session.message.chat.send_message(plain)
                except Exception as fb_e:
                    log.warning(
                        "Plain send part %d/%d failed: %s",
                        part_num,
                        total,
                        fb_e,
                    )
                return
            log.warning("Send part %d/%d error: %s", part_num, total, e)
            return


async def _send_html_with_fallback(
    message: "Message",
    html_text: str,
    plain_source: str,
) -> None:
    """Edit a message with HTML, fallback to plain text.

    RetryAfter exceptions are NOT caught but re-raised,
    so the calling code (e.g. _send_final_edit_with_retry) can
    control the retry loop.

    Args:
        message: The Telegram message being edited.
        html_text: Fully converted HTML text.
        plain_source: Original markdown for strip_markdown fallback.

    Raises:
        Exception: RetryAfter/flood control exceptions are re-raised.
    """
    try:
        await message.edit_text(html_text, parse_mode="HTML")
    except Exception as e:
        # Flood control: re-raise for retry logic in caller
        if _is_retry_after(e) is not None:
            raise

        error_str = str(e).lower()
        if "message is not modified" in error_str:
            pass
        elif "can't parse entities" in error_str or "bad request" in error_str:
            log.warning("HTML edit failed, falling back to plain text: %s", e)
            plain_text = strip_markdown(plain_source)
            try:
                await message.edit_text(plain_text)
            except Exception as fallback_e:
                _handle_edit_error(fallback_e)
        else:
            _handle_edit_error(e)


def split_text_for_telegram(
    text: str,
    max_length: int = TELEGRAM_MAX_LENGTH,
) -> list[str]:
    """Split text intelligently for Telegram messages.

    Splitting priority:
        1. Double newline (paragraph end)
        2. Single newline (line end)
        3. Sentence end (. ! ?)
        4. Word boundary (space)
        5. Hard cut (fallback)

    Markdown-aware: does not cut in the middle of **bold**, *italic*,
    `code`, ```code blocks```, or [links](url).

    Args:
        text: The text to split.
        max_length: Maximum length per part (including part-marker buffer).

    Returns:
        List of text parts, each <= max_length chars.
    """
    if len(text) <= max_length - _SPLIT_SAFETY_MARGIN:
        return [text]

    effective_max = max_length - _SPLIT_SAFETY_MARGIN
    parts: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= effective_max:
            parts.append(remaining)
            break

        split_pos = _find_split_position(remaining, effective_max)
        part = remaining[:split_pos].rstrip()
        remaining = remaining[split_pos:].lstrip("\n")

        if part:
            parts.append(part)

    return parts if parts else [text[:effective_max]]


def _find_split_position(text: str, max_pos: int) -> int:
    """Find the best position to split.

    Searches backward from max_pos for the best break point.
    Respects markdown token boundaries.

    Args:
        text: The text to search in.
        max_pos: Maximum position (exclusive).

    Returns:
        The best split position.
    """
    search_text = text[:max_pos]

    # Priority 1: double newline (paragraph)
    pos = search_text.rfind("\n\n")
    if pos > max_pos // 3:
        candidate = pos + 2
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Priority 2: single newline
    pos = search_text.rfind("\n")
    if pos > max_pos // 3:
        candidate = pos + 1
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Priority 3: sentence end (. ! ? followed by space or line end)
    sentence_end = None
    for m in re.finditer(r"[.!?]\s", search_text):
        if m.end() > max_pos // 3:
            sentence_end = m.end()
    if sentence_end and _is_safe_markdown_position(text, sentence_end):
        return sentence_end

    # Priority 4: word boundary (space)
    pos = search_text.rfind(" ")
    if pos > max_pos // 3:
        candidate = pos + 1
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Fallback: hard cut
    return max_pos


def _is_safe_markdown_position(text: str, pos: int) -> bool:
    """Check whether a position is safe for splitting.

    Unsafe if we are in the middle of a markdown token:
    * Odd number of ** before position (open bold marker)
    * Open backtick block (```)
    * Open inline backticks (`)
    * Open link [text]( without closing )

    Args:
        text: The full text.
        pos: The position to check.

    Returns:
        True if splitting at this position is safe.
    """
    before = text[:pos]

    # Check: open fenced code block (odd count of ```)
    fence_count = before.count("```")
    if fence_count % 2 != 0:
        return False

    # Check: open bold marker (odd count of **)
    bold_count = before.count("**")
    if bold_count % 2 != 0:
        return False

    # Check: open inline code (odd count of single `)
    # First remove ```, then count single `
    cleaned = before.replace("```", "")
    backtick_count = cleaned.count("`")
    if backtick_count % 2 != 0:
        return False

    # Check: open link [text]( (no closing ))
    last_open_bracket = before.rfind("[")
    if last_open_bracket >= 0:
        after_bracket = before[last_open_bracket:]
        if "(" in after_bracket and ")" not in after_bracket.split("(", 1)[1:]:
            close_paren = text.find(")", pos)
            if close_paren >= 0:
                return False

    return True


# R04 Round 2: Markdown smart-trim prevents the user from seeing
# raw ** or ` tokens during streaming. Trims incomplete markdown tokens
# at the end so the visible part renders cleanly as HTML.
def find_safe_markdown_end(text: str) -> int:
    """Find the last safe position for markdown rendering.

    Used for intermediate edits (Option A: smart-trim).
    Trims incomplete markdown tokens at the end so the visible
    part can be rendered cleanly as HTML.

    Searches backward for the last position where all markdown
    tokens are closed.

    Args:
        text: The accumulated text so far.

    Returns:
        Position up to which rendering is safe.
    """
    if not text:
        return 0

    # Quick check: if everything is closed, use the full text
    if _is_safe_markdown_position(text, len(text)):
        return len(text)

    # Search backward: last position where everything is safe
    # Start 50 chars before end (typical token length)
    search_start = max(0, len(text) - 50)
    best = search_start

    for i in range(len(text), search_start, -1):
        if _is_safe_markdown_position(text, i):
            best = i
            break

    return best


async def abort_streaming(session: StreamingSession, error_text: str) -> None:
    """Abort streaming and display an error message.

    Args:
        session: The current StreamingSession.
        error_text: Error message for the user.
    """
    session.accumulated_text = error_text
    await _do_edit(session)


# R04 Round 3: HTML truncation bug fix. Instead of blindly truncating HTML
# (which breaks <b> tags and triggers Telegram 400 Bad Request "Can't parse entities"),
# the markdown text is shortened via binary search so that
# markdown_to_telegram_html(result) <= max_html_length.
def _truncate_markdown_for_html_limit(
    text: str,
    max_html_length: int = TELEGRAM_MAX_LENGTH,
) -> str:
    """Truncate markdown text so the HTML conversion stays under the limit.

    Instead of blindly truncating HTML (which breaks tags), the markdown text
    is shortened via binary search so that
    markdown_to_telegram_html(result) <= max_html_length.

    Args:
        text: Markdown text.
        max_html_length: Maximum HTML length (default: 4096).

    Returns:
        Truncated markdown text whose HTML version is under the limit.
    """
    html = markdown_to_telegram_html(text)
    if len(html) <= max_html_length:
        return text

    # Estimate: markdown is shorter than HTML (tags take space).
    # Start with proportional estimate.
    ratio = max_html_length / max(len(html), 1)
    estimate = int(len(text) * ratio * 0.9)  # 10% safety margin

    # Binary search for exact boundary
    lo = max(0, estimate - 200)
    hi = min(len(text), estimate + 200)

    # Ensure lo is actually under the limit
    while lo > 0 and len(markdown_to_telegram_html(text[:lo])) > max_html_length - 3:
        lo = lo // 2

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        safe_pos = find_safe_markdown_end(text[:mid])
        if safe_pos == 0:
            safe_pos = mid

        candidate = text[:safe_pos]
        candidate_html = markdown_to_telegram_html(candidate)

        if len(candidate_html) <= max_html_length - 3:  # room for "..."
            best = safe_pos
            lo = mid + 1
        else:
            hi = mid - 1

    return text[:best]


def _is_retry_after(exc: Exception) -> int | None:
    """Check whether an exception is a Telegram RetryAfter (429).

    Recognizes both the python-telegram-bot RetryAfter exception and
    generic exceptions whose message contains 'flood control'.

    Returns:
        retry_after in seconds, or None if not a 429.
    """
    # python-telegram-bot >= 20: telegram.error.RetryAfter
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return int(retry_after)

    # Fallback: string match for generic exceptions
    msg = str(exc).lower()
    if "flood control" in msg or "429" in msg:
        import re as _re

        m = _re.search(r"retry in (\d+)", msg)
        if m:
            return int(m.group(1))
        return 30  # Conservative default if not parsable

    return None


def _apply_flood_backoff(session: StreamingSession, retry_after: int) -> None:
    """Apply flood control backoff to the session.

    Sets pause timestamp, doubles the current effective throttle (base from
    burst-mode curve), resets success counter, marks backoff as active.
    """
    now = time.monotonic()
    session._paused_until = now + retry_after

    # Double the current effective throttle (burst curve or existing backoff)
    current_base = _compute_base_throttle(session._edits_sent)
    effective = max(current_base, session._current_throttle)
    session._current_throttle = min(
        effective * THROTTLE_BACKOFF_FACTOR,
        MAX_THROTTLE,
    )
    session._backoff_active = True
    session._consecutive_success = 0
    log.info(
        "Flood control: pausing for %ds, throttle adaptively raised to %.1fs",
        retry_after,
        session._current_throttle,
    )


def _record_edit_success(session: StreamingSession) -> None:
    """Record a successful edit, advance burst counter, reduce backoff on recovery."""
    session._edits_sent += 1
    session._consecutive_success += 1

    if (
        session._backoff_active
        and session._consecutive_success >= THROTTLE_RECOVERY_AFTER
    ):
        old_throttle = session._current_throttle
        session._current_throttle = max(
            session._current_throttle * THROTTLE_RECOVERY_FACTOR,
            _compute_base_throttle(session._edits_sent),
        )
        session._consecutive_success = 0

        # If throttle has recovered to the curve value, deactivate backoff
        base = _compute_base_throttle(session._edits_sent)
        if session._current_throttle <= base:
            session._backoff_active = False

        if old_throttle > session._current_throttle:
            log.debug(
                "Throttle reduced to %.1fs (was %.1fs)",
                session._current_throttle,
                old_throttle,
            )


async def _do_edit(session: StreamingSession) -> None:
    """Perform a Telegram message edit (intermediate edits).

    Uses Option A (smart-trim): markdown is converted to HTML live.
    Incomplete markdown tokens at the end are trimmed so the visible
    part is cleanly formatted.

    Handles Telegram API errors silently (but logs them).
    On RetryAfter (429): session is paused, intermediate edit skipped.
    On Telegram length overflow, the markdown text is intelligently
    truncated (not the HTML, as that would break tags).
    """
    raw = session.accumulated_text
    if not raw.strip():
        raw = "..."

    # Smart-trim: find safe markdown end position
    safe_end = find_safe_markdown_end(raw)

    if safe_end > 0 and safe_end >= len(raw) // 2:
        safe_text = raw[:safe_end]

        # Truncate markdown if HTML is too long (instead of blindly truncating HTML)
        html_text = markdown_to_telegram_html(safe_text)
        if len(html_text) > TELEGRAM_MAX_LENGTH:
            safe_text = _truncate_markdown_for_html_limit(safe_text)
            html_text = markdown_to_telegram_html(safe_text)
            if len(html_text) > TELEGRAM_MAX_LENGTH:
                # Absolute fallback: plain text
                html_text = strip_markdown(safe_text)
                if len(html_text) > TELEGRAM_MAX_LENGTH:
                    html_text = html_text[: TELEGRAM_MAX_LENGTH - 3] + "..."

        # Duplicate check: no API call if text is identical to last edit
        if html_text == session._last_edit_html:
            return

        try:
            await session.message.edit_text(html_text, parse_mode="HTML")
            session._last_edit_html = html_text
            session.last_edit_time = time.monotonic()
            session.edit_count += 1
            _record_edit_success(session)
            return
        except Exception as e:
            # Flood control: pause + skip (intermediate edit, no retry)
            retry_after = _is_retry_after(e)
            if retry_after is not None:
                _apply_flood_backoff(session, retry_after)
                return

            error_str = str(e).lower()
            if "message is not modified" in error_str:
                session._last_edit_html = html_text
                session.last_edit_time = time.monotonic()
                return
            if "can't parse entities" in error_str or "bad request" in error_str:
                log.debug("HTML intermediate edit failed, falling back to plain: %s", e)
                # Fall through to plain text
            else:
                _handle_edit_error(e)
                return

    # Fallback: plain text (if safe_end is too short or HTML failed)
    text = raw
    if len(text) > TELEGRAM_MAX_LENGTH:
        text = text[: TELEGRAM_MAX_LENGTH - 3] + "..."

    # Duplicate check for plain text
    if text == session._last_edit_html:
        return

    try:
        await session.message.edit_text(text)
        session._last_edit_html = text
        session.last_edit_time = time.monotonic()
        session.edit_count += 1
        _record_edit_success(session)
    except Exception as e:
        # Flood control: pause + skip
        retry_after = _is_retry_after(e)
        if retry_after is not None:
            _apply_flood_backoff(session, retry_after)
            return
        _handle_edit_error(e)


async def _do_edit_html(session: StreamingSession) -> None:
    """Perform the final Telegram message edit with HTML formatting.

    Converts the complete markdown text to Telegram HTML
    via markdown_to_telegram_html(). Falls back to strip_markdown()
    if the HTML version is rejected by Telegram.

    FINAL EDIT: has highest priority. On RetryAfter, waits and
    retries (max FINAL_EDIT_MAX_RETRIES attempts).
    The user MUST see the finished response.

    Note: for long texts this function is no longer called directly;
    instead the path goes through _finalize_multi_message().
    """
    raw_text = session.accumulated_text
    if not raw_text.strip():
        raw_text = "..."

    html_text = markdown_to_telegram_html(raw_text)

    for attempt in range(1 + FINAL_EDIT_MAX_RETRIES):
        try:
            await session.message.edit_text(html_text, parse_mode="HTML")
            session.last_edit_time = time.monotonic()
            session.edit_count += 1
            _record_edit_success(session)
            return
        except Exception as e:
            # Flood control on final edit: wait and retry
            retry_after = _is_retry_after(e)
            if retry_after is not None:
                _apply_flood_backoff(session, retry_after)
                if attempt < FINAL_EDIT_MAX_RETRIES:
                    log.info(
                        "Final edit 429, waiting %ds (attempt %d/%d)",
                        retry_after,
                        attempt + 1,
                        FINAL_EDIT_MAX_RETRIES,
                    )
                    await asyncio.sleep(retry_after)
                    continue
                # All retries exhausted: fall back to new message
                log.error(
                    "Final edit still 429 after %d retries, "
                    "falling back to send_message",
                    FINAL_EDIT_MAX_RETRIES,
                )
                plain_text = strip_markdown(raw_text)
                try:
                    await session.message.chat.send_message(plain_text)
                except Exception as fb_e:
                    log.error("Final edit fallback send_message failed: %s", fb_e)
                return

            error_str = str(e).lower()
            if "message is not modified" in error_str:
                return
            if "can't parse entities" in error_str or "bad request" in error_str:
                log.warning("HTML edit failed, falling back to plain text: %s", e)
                plain_text = strip_markdown(raw_text)
                try:
                    await session.message.edit_text(plain_text)
                    session.last_edit_time = time.monotonic()
                    session.edit_count += 1
                    _record_edit_success(session)
                except Exception as fallback_e:
                    _handle_edit_error(fallback_e)
                return
            _handle_edit_error(e)
            return


def _handle_edit_error(e: Exception) -> None:
    """Handle Telegram API errors silently (but log them)."""
    error_str = str(e).lower()
    if "message is not modified" in error_str:
        pass
    elif "message to edit not found" in error_str:
        log.warning("Streaming edit failed: message deleted")
    else:
        log.warning("Streaming edit error: %s", e)
