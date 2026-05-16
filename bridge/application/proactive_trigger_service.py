"""Proactive Trigger Service: drives P1 (Proactive Memory) and P5 (Time/Pattern Awareness).

Determines when the bot should proactively inject a memory nudge or
time-based observation into the conversation. Respects:
  - Frequency cap (max 1/day proactive, max 1/conversation reactive)
  - Quiet hours (derived from user activity patterns)
  - Reminder-style tonality (never to-do style)

Architecture:
  - Called on every incoming user message
  - Checks trigger conditions
  - Returns an optional nudge string that the chat service prepends/appends
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# Frequency cap constants
MAX_PROACTIVE_PER_DAY = 1
MAX_REACTIVE_PER_CONVERSATION = 1

# Quiet hours: default if no user pattern detected yet
DEFAULT_QUIET_START_HOUR = 23  # 23:00
DEFAULT_QUIET_END_HOUR = 7  # 07:00

# Minimum pause before "welcome back" trigger (hours)
PAUSE_THRESHOLD_HOURS = 72  # 3 days

# Minimum days of activity data for pattern detection
MIN_ACTIVITY_DAYS = 3

# Time between messages to count as "session start" (seconds)
SESSION_GAP_SECONDS = 8 * 3600  # 8 hours = new session (morning start)


@dataclass
class UserActivityRecord:
    """Tracks user activity timestamps for pattern detection.

    Attributes:
        message_timestamps: List of recent message timestamps (unix).
        last_proactive_push_ts: Timestamp of last proactive push (unix).
        last_reactive_push_ts: Timestamp of last reactive push per conversation.
        proactive_pushes_today: Count of proactive pushes today.
        proactive_push_date: Date string of last proactive push count reset.
    """

    message_timestamps: list[float] = field(default_factory=list)
    last_proactive_push_ts: float = 0.0
    last_reactive_push_ts: dict[int, float] = field(default_factory=dict)
    proactive_pushes_today: int = 0
    proactive_push_date: str = ""


@dataclass
class TriggerResult:
    """Result of a trigger check.

    Attributes:
        should_fire: Whether a nudge should be injected.
        nudge_text: The text to inject (empty if should_fire is False).
        trigger_type: Type of trigger that fired.
        reason: Human-readable reason for the trigger.
    """

    should_fire: bool = False
    nudge_text: str = ""
    trigger_type: str = ""
    reason: str = ""


class ProactiveTriggerService:
    """Manages proactive trigger logic for memory nudges and time awareness.

    In-memory state (resets on bot restart). Activity records track when
    users are active to derive quiet hours and trigger conditions.
    """

    def __init__(self) -> None:
        self._activity: dict[int, UserActivityRecord] = {}
        self._max_timestamps = 200  # Keep last 200 message timestamps per user

    def record_activity(self, user_id: int, timestamp: Optional[float] = None) -> None:
        """Record a user activity event (message received).

        Args:
            user_id: Telegram user ID.
            timestamp: Unix timestamp (defaults to now).
        """
        ts = timestamp or time.time()
        record = self._activity.setdefault(user_id, UserActivityRecord())
        record.message_timestamps.append(ts)

        # Trim
        if len(record.message_timestamps) > self._max_timestamps:
            record.message_timestamps = record.message_timestamps[
                -self._max_timestamps :
            ]

    def check_triggers(
        self,
        user_id: int,
        chat_id: int,
        current_message: str,
        memory_entries: Optional[list[dict]] = None,
        current_time: Optional[datetime] = None,
    ) -> TriggerResult:
        """Check if any proactive trigger should fire.

        Called on every incoming message. Evaluates all trigger conditions
        and returns the highest-priority one that passes.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            current_message: The user's current message text.
            memory_entries: Relevant memory entries for this user (optional).
            current_time: Override for current time (for testing).

        Returns:
            TriggerResult with nudge text if a trigger fired.
        """
        now = current_time or datetime.now(timezone.utc)
        now_ts = now.timestamp()
        record = self._activity.get(user_id)

        if record is None:
            return TriggerResult()

        # Check quiet hours first
        if self._is_quiet_hours(user_id, now):
            return TriggerResult()

        # Check frequency cap
        if not self._can_push_proactive(user_id, now):
            return TriggerResult()

        # Trigger 1: Session start after long pause
        result = self._check_session_start_trigger(record, now_ts, now)
        if result.should_fire:
            self._mark_proactive_push(user_id, now)
            return result

        # Trigger 2: Pattern awareness (weekday/time observations)
        result = self._check_pattern_trigger(record, now, current_message)
        if result.should_fire:
            self._mark_proactive_push(user_id, now)
            return result

        return TriggerResult()

    def check_reactive_trigger(
        self,
        user_id: int,
        chat_id: int,
        current_message: str,
        memory_entries: list[dict],
    ) -> TriggerResult:
        """Check if a reactive memory nudge should fire within a conversation.

        Reactive nudges are injected when a memory entry is highly relevant
        to the current conversation topic but was not explicitly queried.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            current_message: Current message text.
            memory_entries: Memory entries loaded for this query.

        Returns:
            TriggerResult (fires at most once per conversation).
        """
        record = self._activity.get(user_id)
        if record is None:
            return TriggerResult()

        # Check if we already fired a reactive nudge in this conversation
        last_reactive = record.last_reactive_push_ts.get(chat_id, 0.0)
        now_ts = time.time()
        # "Same conversation" = within last 2 hours
        if now_ts - last_reactive < 7200:
            return TriggerResult()

        # Reactive nudge: find memory entries with time references
        for entry in memory_entries:
            content = entry.get("content", "")
            # Simple heuristic: entries that mention dates, deadlines, or plans
            if self._has_time_reference(content):
                nudge = self._build_memory_nudge(content, "time_reference")
                if nudge:
                    record.last_reactive_push_ts[chat_id] = now_ts
                    return TriggerResult(
                        should_fire=True,
                        nudge_text=nudge,
                        trigger_type="reactive_memory",
                        reason=f"Memory entry has time reference: {content[:50]}...",
                    )

        return TriggerResult()

    def get_active_hours(self, user_id: int) -> tuple[int, int]:
        """Derive user's active hours from activity patterns.

        Args:
            user_id: Telegram user ID.

        Returns:
            Tuple of (start_hour, end_hour) in UTC.
            Defaults to (7, 23) if insufficient data.
        """
        record = self._activity.get(user_id)
        if record is None or len(record.message_timestamps) < 20:
            return (DEFAULT_QUIET_END_HOUR, DEFAULT_QUIET_START_HOUR)

        # Count messages per hour
        hour_counts: dict[int, int] = {}
        for ts in record.message_timestamps:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            hour = dt.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        if not hour_counts:
            return (DEFAULT_QUIET_END_HOUR, DEFAULT_QUIET_START_HOUR)

        # Find the earliest and latest active hours (with at least 2 messages)
        active_hours = sorted(h for h, c in hour_counts.items() if c >= 2)
        if not active_hours:
            return (DEFAULT_QUIET_END_HOUR, DEFAULT_QUIET_START_HOUR)

        return (active_hours[0], active_hours[-1])

    def get_time_context_block(
        self, user_id: int, now: Optional[datetime] = None
    ) -> str:
        """Build a time/pattern awareness block for the system prompt.

        Gives the LLM awareness of current time and user patterns.

        Args:
            user_id: Telegram user ID.
            now: Override current time (for testing).

        Returns:
            Prompt block string (may be empty if no patterns detected).
        """
        current = now or datetime.now(timezone.utc)
        weekday_names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }

        lines = [
            "[TIME CONTEXT]",
            f"Current time (UTC): {current.strftime('%Y-%m-%d %H:%M')}",
            f"Day: {weekday_names[current.weekday()]}",
        ]

        # Add user pattern observations
        record = self._activity.get(user_id)
        if record and len(record.message_timestamps) >= 20:
            start_h, end_h = self.get_active_hours(user_id)
            lines.append(
                f"User typically active: {start_h:02d}:00 to {end_h:02d}:00 UTC"
            )

            # Detect if this is an unusual time for the user
            current_hour = current.hour
            if current_hour < start_h or current_hour > end_h:
                lines.append(
                    "Note: User is active outside their normal hours. "
                    "Consider a gentle acknowledgment if appropriate."
                )

        # Weekend awareness
        if current.weekday() >= 5:
            lines.append("It is the weekend.")

        return "\n".join(lines)

    def _is_quiet_hours(self, user_id: int, now: datetime) -> bool:
        """Check if current time falls in user's quiet hours.

        Args:
            user_id: Telegram user ID.
            now: Current datetime.

        Returns:
            True if it is quiet hours (no proactive pushes).
        """
        start_h, end_h = self.get_active_hours(user_id)
        current_hour = now.hour

        # If active window is e.g. 7-23, quiet is 23-7
        quiet_start = end_h
        quiet_end = start_h

        if quiet_start > quiet_end:
            # Normal case: quiet from 23 to 7
            return current_hour >= quiet_start or current_hour < quiet_end
        else:
            # Edge case: quiet window does not wrap midnight
            return quiet_start <= current_hour < quiet_end

    def _can_push_proactive(self, user_id: int, now: datetime) -> bool:
        """Check if the frequency cap allows a proactive push.

        Args:
            user_id: Telegram user ID.
            now: Current datetime.

        Returns:
            True if a push is allowed.
        """
        record = self._activity.get(user_id)
        if record is None:
            return True

        today_str = now.strftime("%Y-%m-%d")
        if record.proactive_push_date != today_str:
            # New day, reset counter
            record.proactive_pushes_today = 0
            record.proactive_push_date = today_str

        return record.proactive_pushes_today < MAX_PROACTIVE_PER_DAY

    def _mark_proactive_push(self, user_id: int, now: datetime) -> None:
        """Mark that a proactive push was sent.

        Args:
            user_id: Telegram user ID.
            now: Current datetime.
        """
        record = self._activity.setdefault(user_id, UserActivityRecord())
        record.last_proactive_push_ts = now.timestamp()
        today_str = now.strftime("%Y-%m-%d")
        if record.proactive_push_date != today_str:
            record.proactive_pushes_today = 1
            record.proactive_push_date = today_str
        else:
            record.proactive_pushes_today += 1

    def _check_session_start_trigger(
        self, record: UserActivityRecord, now_ts: float, now: datetime
    ) -> TriggerResult:
        """Check if this is a new session after a long pause.

        Args:
            record: User's activity record.
            now_ts: Current unix timestamp.
            now: Current datetime.

        Returns:
            TriggerResult.
        """
        if len(record.message_timestamps) < 2:
            return TriggerResult()

        # Time since previous message (before the current one which was just recorded)
        prev_ts = (
            record.message_timestamps[-2] if len(record.message_timestamps) >= 2 else 0
        )
        gap_seconds = now_ts - prev_ts

        if gap_seconds < SESSION_GAP_SECONDS:
            return TriggerResult()

        gap_hours = gap_seconds / 3600
        gap_days = gap_hours / 24

        # Build appropriate nudge based on gap length
        if gap_days >= 3:
            # Long absence
            return TriggerResult(
                should_fire=True,
                nudge_text=self._build_welcome_back_nudge(gap_days, now),
                trigger_type="long_pause",
                reason=f"User returned after {gap_days:.1f} days absence",
            )
        elif gap_hours >= 8:
            # New session (e.g., morning start)
            hour = now.hour
            if 5 <= hour <= 10:
                # Morning greeting context
                return TriggerResult(
                    should_fire=True,
                    nudge_text=self._build_morning_nudge(now),
                    trigger_type="morning_start",
                    reason=f"Morning session start at {hour}:00",
                )

        return TriggerResult()

    def _check_pattern_trigger(
        self, record: UserActivityRecord, now: datetime, message: str
    ) -> TriggerResult:
        """Check for time/pattern based triggers.

        Detects: late night activity, weekend work, recurring patterns.

        Args:
            record: User's activity record.
            now: Current datetime.
            message: Current message text.

        Returns:
            TriggerResult.
        """
        hour = now.hour
        weekday = now.weekday()

        # Late night awareness (after 23:00 or before 5:00)
        if hour >= 23 or hour < 5:
            # Only trigger if we have enough history to know this is unusual
            if len(record.message_timestamps) >= 30:
                late_night_count = sum(
                    1
                    for ts in record.message_timestamps[:-1]  # exclude current
                    if datetime.fromtimestamp(ts, tz=timezone.utc).hour >= 23
                    or datetime.fromtimestamp(ts, tz=timezone.utc).hour < 5
                )
                total = len(record.message_timestamps) - 1
                # If less than 10% of messages are usually late night
                if total > 0 and late_night_count / total < 0.1:
                    return TriggerResult(
                        should_fire=True,
                        nudge_text=(
                            "\n\n_Nebenbei: Es ist ziemlich spät. "
                            "Kein Urteil, nur eine Beobachtung._"
                        ),
                        trigger_type="late_night",
                        reason=f"Unusual late activity at {hour}:00",
                    )

        # Sunday work awareness
        if weekday == 6 and 9 <= hour <= 18:
            if len(record.message_timestamps) >= 30:
                sunday_count = sum(
                    1
                    for ts in record.message_timestamps[:-1]
                    if datetime.fromtimestamp(ts, tz=timezone.utc).weekday() == 6
                )
                total = len(record.message_timestamps) - 1
                if total > 0 and sunday_count / total < 0.05:
                    return TriggerResult(
                        should_fire=True,
                        nudge_text=(
                            "\n\n_Heute ist Sonntag. Arbeitest du bewusst, "
                            "oder hat es dich einfach hierher getrieben?_"
                        ),
                        trigger_type="weekend_work",
                        reason="Working on Sunday, unusual for this user",
                    )

        return TriggerResult()

    def _build_welcome_back_nudge(self, gap_days: float, now: datetime) -> str:
        """Build a welcome-back nudge after a long pause.

        Uses reminder-style, not to-do-style.

        Args:
            gap_days: Number of days since last activity.
            now: Current datetime.

        Returns:
            Nudge text string.
        """
        if gap_days >= 7:
            return (
                "\n\n_Schön dass du wieder da bist. "
                "War eine Weile still hier. "
                "Gibt es etwas wobei ich dir helfen kann?_"
            )
        else:
            return "\n\n_Hey, ein paar Tage her. Wo waren wir stehen geblieben?_"

    def _build_morning_nudge(self, now: datetime) -> str:
        """Build a morning session nudge.

        Args:
            now: Current datetime.

        Returns:
            Nudge text string.
        """
        weekday_names_de = {
            0: "Montag",
            1: "Dienstag",
            2: "Mittwoch",
            3: "Donnerstag",
            4: "Freitag",
            5: "Samstag",
            6: "Sonntag",
        }
        day_name = weekday_names_de.get(now.weekday(), "")
        return f"\n\n_Guten Morgen. {day_name} also. Was steht an?_"

    def _build_memory_nudge(self, content: str, trigger_reason: str) -> str:
        """Build a reactive memory nudge in reminder style.

        Args:
            content: Memory entry content.
            trigger_reason: Why this nudge was triggered.

        Returns:
            Nudge text string.
        """
        # Truncate long memory content
        short_content = content[:80] + "..." if len(content) > 80 else content
        return (
            f"\n\n_Mir fällt gerade ein: Du hattest mal erwähnt: "
            f'"{short_content}" '
            f"Ist das noch relevant?_"
        )

    @staticmethod
    def _has_time_reference(text: str) -> bool:
        """Check if a text contains time references (dates, deadlines, plans).

        Args:
            text: Text to analyze.

        Returns:
            True if time references detected.
        """
        import re as _re

        time_patterns = [
            r"\d{1,2}\.\d{1,2}\.\d{2,4}",  # 16.03.2026
            r"\d{4}-\d{2}-\d{2}",  # 2026-03-16
            r"(?:montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)",
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"(?:morgen|übermorgen|nächste woche|nächsten monat)",
            r"(?:tomorrow|next week|next month|deadline)",
            r"(?:in \d+ tagen|in \d+ wochen|in \d+ monaten)",
            r"(?:bis zum|spätestens|fällig|due)",
        ]
        text_lower = text.lower()
        return any(_re.search(p, text_lower) for p in time_patterns)
