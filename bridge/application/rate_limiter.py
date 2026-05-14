"""Rate limiter: profile-based per-user rate limiting.

Business rule: each user has limited requests per time window,
defined by a profile (light, normal, power, unlimited).

Profiles:
    * Light:     17/min,  100/h,    400/day
    * Normal:    25/min,  350/h,  1,500/day  (default)
    * Power:     60/min,  900/h, 10,000/day
    * Unlimited: no limits (with reminder every 100 requests)

Architecture: application layer (business rule, no Telegram code).
In-memory storage for buckets (session-based), profiles persistent via JSONL.
Eviction after 1h of inactivity.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from infrastructure.encoding import append_jsonl_utf8, open_utf8

# TYPE_CHECKING guard for SQLite backend (optional dependency)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteProfileStorage

log = logging.getLogger(__name__)

# Eviction: remove user buckets after 1h of inactivity
_EVICTION_TTL_SECONDS: float = 3600.0

# 70% warning: once per window
_WARNING_THRESHOLD: float = 0.7

# Unlimited mode: reminder every N requests
_UNLIMITED_REMINDER_INTERVAL: int = 100


# --- Profile definitions ---

PROFILES: dict[str, dict[str, int]] = {
    "light": {"per_minute": 17, "per_hour": 100, "per_day": 400},
    "normal": {"per_minute": 25, "per_hour": 350, "per_day": 1500},
    "power": {"per_minute": 60, "per_hour": 900, "per_day": 10000},
    "unlimited": {"per_minute": 0, "per_hour": 0, "per_day": 0},
}

DEFAULT_PROFILE: str = "normal"

# Persistent profile store (JSONL)
_PROFILES_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "user_profiles.jsonl"
)


def _load_user_profiles() -> dict[int, str]:
    """Load user profiles from the JSONL file.

    Reads all lines and takes the last entry per user
    (append-only log, last entry wins).

    Returns:
        Dict: user_id -> profile_name.
    """
    profiles: dict[int, str] = {}
    if not _PROFILES_PATH.exists():
        return profiles

    try:
        with open_utf8(_PROFILES_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    uid = entry.get("user_id")
                    profile = entry.get("profile", DEFAULT_PROFILE)
                    if uid is not None and profile in PROFILES:
                        profiles[int(uid)] = profile
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except OSError as e:
        log.warning("Could not load user profiles: %s", e)

    return profiles


def _save_user_profile(user_id: int, chat_id: int, profile: str) -> None:
    """Persist a user profile as a JSONL entry.

    Args:
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        profile: Profile name (light, normal, power, unlimited).
    """
    from datetime import datetime, timezone

    entry = {
        "user_id": user_id,
        "chat_id": chat_id,
        "profile": profile,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl_utf8(entry, _PROFILES_PATH)


class TokenBucket:
    """Fixed-window counter for a single time window.

    Counts requests per window and blocks when capacity is reached.
    The window resets once window_seconds have elapsed.

    Previously a token-bucket algorithm with continuous refill was used.
    Problem: tokens trickled in between requests, so with capacity=17
    and 30s distribution, effectively ~25 requests were possible instead of 17.
    Fix (2026-05-09): switched to fixed-window counter. Exactly capacity
    requests per window, no refill, no drift.

    Attributes:
        capacity: Maximum number of requests per window.
        request_count: Actual number of consumed requests in the current window.
        window_seconds: Length of the time window in seconds.
        window_start: Timestamp of window start (for counter reset).
    """

    __slots__ = (
        "capacity",
        "request_count",
        "window_seconds",
        "window_start",
    )

    def __init__(self, capacity: int, window_seconds: float) -> None:
        """Initialize the counter.

        Args:
            capacity: Maximum requests per window.
            window_seconds: Length of the time window in seconds.
        """
        self.capacity = capacity
        self.request_count: int = 0
        self.window_seconds = window_seconds
        self.window_start: float = time.monotonic()

    def _maybe_reset_window(self, now: float) -> None:
        """Reset the request counter if the time window has expired.

        Args:
            now: Current timestamp (time.monotonic).
        """
        if now - self.window_start >= self.window_seconds:
            self.request_count = 0
            self.window_start = now

    def try_consume(self) -> tuple[bool, float]:
        """Attempt to consume a request.

        Checks whether the window has expired (reset), then whether
        request_count < capacity. No token refill, no drift.

        Returns:
            Tuple of (allowed, retry_after_seconds).
            allowed=True if request was permitted.
            retry_after_seconds > 0 if not permitted (wait time until window reset).
        """
        now = time.monotonic()

        self._maybe_reset_window(now)

        if self.request_count < self.capacity:
            self.request_count += 1
            return True, 0.0

        elapsed_in_window = now - self.window_start
        retry_after = self.window_seconds - elapsed_in_window
        return False, max(0.0, retry_after)

    def usage_fraction(self) -> float:
        """Return the current usage fraction (0.0 to 1.0).

        0.0 = nothing consumed, 1.0 = limit reached.
        """
        if self.capacity == 0:
            return 0.0
        now = time.monotonic()
        self._maybe_reset_window(now)
        return min(1.0, self.request_count / self.capacity)

    def consumed_count(self) -> int:
        """Return the number of consumed requests in the current window."""
        now = time.monotonic()
        self._maybe_reset_window(now)
        return self.request_count

    def seconds_until_reset(self) -> float:
        """Return seconds until the next window reset."""
        now = time.monotonic()
        elapsed_in_window = now - self.window_start
        remaining = self.window_seconds - elapsed_in_window
        return max(0.0, remaining)

    def rollback(self) -> None:
        """Undo the last consume operation.

        Used when an outer bucket (hour/day) rejects the request
        after an inner bucket (minute) already consumed.
        """
        if self.request_count > 0:
            self.request_count -= 1


class _UserBuckets:
    """Three token buckets for a single user.

    Attributes:
        minute_bucket: Burst protection.
        hour_bucket: Sustained load protection.
        day_bucket: Daily budget.
        last_activity: Timestamp of last activity (for eviction).
        profile: Active profile.
        warning_sent_minute: Whether 70% warning for minute was sent.
        warning_sent_hour: Whether 70% warning for hour was sent.
        warning_sent_day: Whether 70% warning for day was sent.
        unlimited_counter: Counter for unlimited reminders.
    """

    __slots__ = (
        "minute_bucket",
        "hour_bucket",
        "day_bucket",
        "last_activity",
        "profile",
        "warning_sent_minute",
        "warning_sent_hour",
        "warning_sent_day",
        "unlimited_counter",
    )

    def __init__(self, profile: str = DEFAULT_PROFILE) -> None:
        limits = PROFILES.get(profile, PROFILES[DEFAULT_PROFILE])
        self.profile = profile
        self.minute_bucket = TokenBucket(
            capacity=limits["per_minute"], window_seconds=60.0
        )
        self.hour_bucket = TokenBucket(
            capacity=limits["per_hour"], window_seconds=3600.0
        )
        self.day_bucket = TokenBucket(
            capacity=limits["per_day"], window_seconds=86400.0
        )
        self.last_activity = time.monotonic()
        self.warning_sent_minute = False
        self.warning_sent_hour = False
        self.warning_sent_day = False
        self.unlimited_counter = 0


class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether the request is permitted.
        retry_after: Wait time in seconds (None if allowed).
        period: Which limit was hit (minute/hour/day/None).
        limit_value: Maximum value of the limit.
        current_count: Current consumption.
        profile: Active profile for the user.
        warning_70: Whether the 70% warning should be triggered.
        warning_period: Which window the 70% warning applies to.
        unlimited_reminder: Whether an unlimited reminder should be sent.
    """

    __slots__ = (
        "allowed",
        "retry_after",
        "period",
        "limit_value",
        "current_count",
        "profile",
        "warning_70",
        "warning_period",
        "unlimited_reminder",
    )

    def __init__(
        self,
        allowed: bool = True,
        retry_after: Optional[float] = None,
        period: Optional[str] = None,
        limit_value: int = 0,
        current_count: int = 0,
        profile: str = DEFAULT_PROFILE,
        warning_70: bool = False,
        warning_period: Optional[str] = None,
        unlimited_reminder: bool = False,
    ) -> None:
        self.allowed = allowed
        self.retry_after = retry_after
        self.period = period
        self.limit_value = limit_value
        self.current_count = current_count
        self.profile = profile
        self.warning_70 = warning_70
        self.warning_period = warning_period
        self.unlimited_reminder = unlimited_reminder


class UsageInfo:
    """Usage information for /usage.

    Attributes:
        profile: Active profile.
        minute_used: Consumption this minute.
        minute_limit: Limit per minute.
        minute_reset_seconds: Seconds until reset.
        hour_used: Consumption this hour.
        hour_limit: Limit per hour.
        hour_reset_seconds: Seconds until reset.
        day_used: Consumption today.
        day_limit: Limit per day.
        day_reset_seconds: Seconds until reset.
    """

    __slots__ = (
        "profile",
        "minute_used",
        "minute_limit",
        "minute_reset_seconds",
        "hour_used",
        "hour_limit",
        "hour_reset_seconds",
        "day_used",
        "day_limit",
        "day_reset_seconds",
    )

    def __init__(
        self,
        profile: str = DEFAULT_PROFILE,
        minute_used: int = 0,
        minute_limit: int = 0,
        minute_reset_seconds: float = 0.0,
        hour_used: int = 0,
        hour_limit: int = 0,
        hour_reset_seconds: float = 0.0,
        day_used: int = 0,
        day_limit: int = 0,
        day_reset_seconds: float = 0.0,
    ) -> None:
        self.profile = profile
        self.minute_used = minute_used
        self.minute_limit = minute_limit
        self.minute_reset_seconds = minute_reset_seconds
        self.hour_used = hour_used
        self.hour_limit = hour_limit
        self.hour_reset_seconds = hour_reset_seconds
        self.day_used = day_used
        self.day_limit = day_limit
        self.day_reset_seconds = day_reset_seconds


class RateLimiter:
    """Per-user rate limiter with three time windows and profile system.

    Thread-safe via Lock. Eviction of inactive users after 1h.
    Profiles are persisted (JSONL).

    Usage:
        limiter = RateLimiter()
        result = limiter.check_and_consume(user_id=12345)
        if not result.allowed:
            # User has hit the limit
            ...
    """

    def __init__(
        self,
        profile_storage: "SqliteProfileStorage | None" = None,
    ) -> None:
        self._users: dict[int, _UserBuckets] = {}
        self._lock = Lock()
        self._profile_storage = profile_storage
        if profile_storage is not None:
            self._profiles = profile_storage.load_all()
        else:
            self._profiles = _load_user_profiles()

    def get_user_profile(self, user_id: int) -> str:
        """Return the active profile for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            Profile name (light, normal, power, unlimited).
        """
        return self._profiles.get(user_id, DEFAULT_PROFILE)

    def set_user_profile(self, user_id: int, chat_id: int, profile: str) -> bool:
        """Set the profile for a user.

        Persists the change and creates new buckets with the new limits.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            profile: Profile name (light, normal, power, unlimited).

        Returns:
            True if successful, False if profile is invalid.
        """
        if profile not in PROFILES:
            return False

        with self._lock:
            self._profiles[user_id] = profile
            if user_id in self._users:
                self._users[user_id] = _UserBuckets(profile=profile)

        # Persist
        if self._profile_storage is not None:
            self._profile_storage.save(user_id, chat_id, profile)
        else:
            _save_user_profile(user_id, chat_id, profile)
        log.info("User %d: profile changed to '%s'", user_id, profile)
        return True

    def check_and_consume(self, user_id: int) -> RateLimitResult:
        """Check whether the user may send a request and consume a token.

        Checks all three buckets (minute, hour, day). If any of them
        has no token, the request is denied. Only when all three
        allow, one token is consumed from each.

        In unlimited mode no tokens are consumed, but a reminder
        counter is incremented.

        Args:
            user_id: Telegram user ID.

        Returns:
            RateLimitResult with all relevant information.
        """
        with self._lock:
            self._evict_stale()
            profile = self._profiles.get(user_id, DEFAULT_PROFILE)

            if user_id not in self._users:
                self._users[user_id] = _UserBuckets(profile=profile)
            elif self._users[user_id].profile != profile:
                self._users[user_id] = _UserBuckets(profile=profile)

            buckets = self._users[user_id]
            buckets.last_activity = time.monotonic()

            # Unlimited mode: no limit, but reminder counter
            if profile == "unlimited":
                buckets.unlimited_counter += 1
                show_reminder = (
                    buckets.unlimited_counter % _UNLIMITED_REMINDER_INTERVAL == 0
                )
                return RateLimitResult(
                    allowed=True,
                    profile=profile,
                    unlimited_reminder=show_reminder,
                )

            # Check all three buckets
            min_ok, min_retry = buckets.minute_bucket.try_consume()
            if not min_ok:
                return RateLimitResult(
                    allowed=False,
                    retry_after=round(min_retry, 1),
                    period="minute",
                    limit_value=buckets.minute_bucket.capacity,
                    current_count=buckets.minute_bucket.capacity,
                    profile=profile,
                )

            hour_ok, hour_retry = buckets.hour_bucket.try_consume()
            if not hour_ok:
                buckets.minute_bucket.rollback()
                return RateLimitResult(
                    allowed=False,
                    retry_after=round(hour_retry, 1),
                    period="hour",
                    limit_value=buckets.hour_bucket.capacity,
                    current_count=buckets.hour_bucket.capacity,
                    profile=profile,
                )

            day_ok, day_retry = buckets.day_bucket.try_consume()
            if not day_ok:
                buckets.minute_bucket.rollback()
                buckets.hour_bucket.rollback()
                return RateLimitResult(
                    allowed=False,
                    retry_after=round(day_retry, 1),
                    period="day",
                    limit_value=buckets.day_bucket.capacity,
                    current_count=buckets.day_bucket.capacity,
                    profile=profile,
                )

            # Success: check 70% warning
            warning_70 = False
            warning_period: Optional[str] = None

            # Minute warning
            min_consumed = buckets.minute_bucket.consumed_count()
            min_cap = buckets.minute_bucket.capacity
            if (
                not buckets.warning_sent_minute
                and min_cap > 0
                and min_consumed >= int(min_cap * _WARNING_THRESHOLD)
            ):
                warning_70 = True
                warning_period = "minute"
                buckets.warning_sent_minute = True

            # Hour warning (takes priority if both fire simultaneously)
            hour_consumed = buckets.hour_bucket.consumed_count()
            hour_cap = buckets.hour_bucket.capacity
            if (
                not buckets.warning_sent_hour
                and hour_cap > 0
                and hour_consumed >= int(hour_cap * _WARNING_THRESHOLD)
            ):
                warning_70 = True
                warning_period = "hour"
                buckets.warning_sent_hour = True

            # Day warning
            day_consumed = buckets.day_bucket.consumed_count()
            day_cap = buckets.day_bucket.capacity
            if (
                not buckets.warning_sent_day
                and day_cap > 0
                and day_consumed >= int(day_cap * _WARNING_THRESHOLD)
            ):
                warning_70 = True
                warning_period = "day"
                buckets.warning_sent_day = True

            # Warning reset: when bucket has refilled (< 50%)
            if min_cap > 0 and min_consumed < int(min_cap * 0.5):
                buckets.warning_sent_minute = False
            if hour_cap > 0 and hour_consumed < int(hour_cap * 0.5):
                buckets.warning_sent_hour = False
            if day_cap > 0 and day_consumed < int(day_cap * 0.5):
                buckets.warning_sent_day = False

            return RateLimitResult(
                allowed=True,
                profile=profile,
                warning_70=warning_70,
                warning_period=warning_period,
            )

    def get_usage(self, user_id: int) -> UsageInfo:
        """Return current usage information for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            UsageInfo with consumption, limits, and reset times.
        """
        with self._lock:
            profile = self._profiles.get(user_id, DEFAULT_PROFILE)
            limits = PROFILES.get(profile, PROFILES[DEFAULT_PROFILE])

            if user_id not in self._users:
                return UsageInfo(
                    profile=profile,
                    minute_used=0,
                    minute_limit=limits["per_minute"],
                    minute_reset_seconds=0.0,
                    hour_used=0,
                    hour_limit=limits["per_hour"],
                    hour_reset_seconds=0.0,
                    day_used=0,
                    day_limit=limits["per_day"],
                    day_reset_seconds=0.0,
                )

            buckets = self._users[user_id]

            return UsageInfo(
                profile=profile,
                minute_used=buckets.minute_bucket.consumed_count(),
                minute_limit=buckets.minute_bucket.capacity,
                minute_reset_seconds=round(
                    buckets.minute_bucket.seconds_until_reset(), 0
                ),
                hour_used=buckets.hour_bucket.consumed_count(),
                hour_limit=buckets.hour_bucket.capacity,
                hour_reset_seconds=round(buckets.hour_bucket.seconds_until_reset(), 0),
                day_used=buckets.day_bucket.consumed_count(),
                day_limit=buckets.day_bucket.capacity,
                day_reset_seconds=round(buckets.day_bucket.seconds_until_reset(), 0),
            )

    def _evict_stale(self) -> None:
        """Remove buckets of users who have been inactive longer than TTL.

        Must be called within self._lock.
        """
        now = time.monotonic()
        stale_ids = [
            uid
            for uid, buckets in self._users.items()
            if now - buckets.last_activity > _EVICTION_TTL_SECONDS
        ]
        for uid in stale_ids:
            del self._users[uid]
        if stale_ids:
            log.debug("Rate limiter: %d inactive users evicted", len(stale_ids))

    def _reset_all_for_tests(self) -> None:
        """Reset all buckets and profiles. ONLY for tests."""
        with self._lock:
            self._users.clear()
            self._profiles.clear()
