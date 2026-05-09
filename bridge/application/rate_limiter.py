"""Rate-Limiter: Profile-basiertes Per-User Rate-Limiting.

Business-Regel: Jeder User hat begrenzte Anfragen pro Zeitfenster,
definiert durch ein Profil (light, normal, power, unlimited).

Profile:
    - Light:     17/min,  100/h,    400/day
    - Normal:    25/min,  350/h,  1.500/day  (Default)
    - Power:     60/min,  900/h, 10.000/day
    - Unlimited: keine Limits (mit Reminder alle 100 Anfragen)

Architektur: Application-Layer (Business-Regel, kein Telegram-Code).
In-Memory Storage fuer Buckets (Session-basiert), Profile persistent via JSONL.
Eviction nach 1h Inaktivitaet.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from infrastructure.encoding import append_jsonl_utf8, open_utf8

log = logging.getLogger(__name__)

# Eviction: entferne User-Buckets nach 1h Inaktivitaet
_EVICTION_TTL_SECONDS: float = 3600.0

# 70% Warnung: einmalig pro Window
_WARNING_THRESHOLD: float = 0.7

# Unlimited-Mode: Reminder alle N Anfragen
_UNLIMITED_REMINDER_INTERVAL: int = 100


# --- Profile-Definitionen ---

PROFILES: dict[str, dict[str, int]] = {
    "light": {"per_minute": 17, "per_hour": 100, "per_day": 400},
    "normal": {"per_minute": 25, "per_hour": 350, "per_day": 1500},
    "power": {"per_minute": 60, "per_hour": 900, "per_day": 10000},
    "unlimited": {"per_minute": 0, "per_hour": 0, "per_day": 0},
}

DEFAULT_PROFILE: str = "normal"

# Persistenter Profil-Speicher (JSONL)
_PROFILES_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "user_profiles.jsonl"
)


def _load_user_profiles() -> dict[int, str]:
    """Laedt User-Profile aus der JSONL-Datei.

    Liest alle Zeilen und nimmt den jeweils letzten Eintrag pro User
    (append-only Log, letzter Eintrag gewinnt).

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
        log.warning("Konnte User-Profile nicht laden: %s", e)

    return profiles


def _save_user_profile(user_id: int, chat_id: int, profile: str) -> None:
    """Persistiert ein User-Profil als JSONL-Eintrag.

    Args:
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        profile: Profilname (light, normal, power, unlimited).
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
    """Fixed-Window-Counter fuer ein einzelnes Zeitfenster.

    Zaehlt Anfragen pro Fenster und blockiert bei Erreichen der Kapazitaet.
    Das Fenster wird zurueckgesetzt sobald window_seconds vergangen sind.

    Frueher war hier ein Token-Bucket-Algorithmus mit kontinuierlichem Refill.
    Problem: Tokens flossen zwischen Anfragen nach, sodass z.B. bei capacity=17
    und 30s Verteilung effektiv ~25 Anfragen moeglich waren statt 17.
    Fix (2026-05-09): Umstellung auf Fixed-Window-Counter. Exakt capacity
    Anfragen pro Fenster, kein Refill, kein Drift.

    Attributes:
        capacity: Maximale Anzahl Anfragen pro Fenster.
        request_count: Tatsaechliche Anzahl konsumierter Anfragen im aktuellen Fenster.
        window_seconds: Laenge des Zeitfensters in Sekunden.
        window_start: Zeitstempel des Fenster-Starts (fuer Counter-Reset).
    """

    __slots__ = (
        "capacity",
        "request_count",
        "window_seconds",
        "window_start",
    )

    def __init__(self, capacity: int, window_seconds: float) -> None:
        """Initialisiert den Counter.

        Args:
            capacity: Maximale Anfragen pro Fenster.
            window_seconds: Laenge des Zeitfensters in Sekunden.
        """
        self.capacity = capacity
        self.request_count: int = 0
        self.window_seconds = window_seconds
        self.window_start: float = time.monotonic()

    def _maybe_reset_window(self, now: float) -> None:
        """Setzt den Request-Counter zurueck wenn das Zeitfenster abgelaufen ist.

        Args:
            now: Aktueller Zeitstempel (time.monotonic).
        """
        if now - self.window_start >= self.window_seconds:
            self.request_count = 0
            self.window_start = now

    def try_consume(self) -> tuple[bool, float]:
        """Versucht eine Anfrage zu konsumieren.

        Prueft ob das Fenster abgelaufen ist (Reset), dann ob
        request_count < capacity. Kein Token-Refill, kein Drift.

        Returns:
            Tuple von (allowed, retry_after_seconds).
            allowed=True wenn Anfrage erlaubt war.
            retry_after_seconds > 0 wenn nicht erlaubt (Wartezeit bis Window-Reset).
        """
        now = time.monotonic()

        # Window-Reset pruefen
        self._maybe_reset_window(now)

        if self.request_count < self.capacity:
            self.request_count += 1
            return True, 0.0

        # Berechne Wartezeit bis Window-Reset
        elapsed_in_window = now - self.window_start
        retry_after = self.window_seconds - elapsed_in_window
        return False, max(0.0, retry_after)

    def usage_fraction(self) -> float:
        """Gibt den aktuellen Verbrauchsanteil zurueck (0.0 bis 1.0).

        0.0 = nichts verbraucht, 1.0 = Limit erreicht.
        """
        if self.capacity == 0:
            return 0.0
        now = time.monotonic()
        self._maybe_reset_window(now)
        return min(1.0, self.request_count / self.capacity)

    def consumed_count(self) -> int:
        """Gibt die Anzahl konsumierter Anfragen im aktuellen Fenster zurueck."""
        now = time.monotonic()
        self._maybe_reset_window(now)
        return self.request_count

    def seconds_until_reset(self) -> float:
        """Gibt Sekunden bis zum naechsten Window-Reset zurueck."""
        now = time.monotonic()
        elapsed_in_window = now - self.window_start
        remaining = self.window_seconds - elapsed_in_window
        return max(0.0, remaining)

    def rollback(self) -> None:
        """Macht die letzte consume-Operation rueckgaengig.

        Wird verwendet wenn ein aeusserer Bucket (Hour/Day) die Anfrage
        ablehnt, nachdem ein innerer Bucket (Minute) schon konsumiert hat.
        """
        if self.request_count > 0:
            self.request_count -= 1


class _UserBuckets:
    """Drei Token-Buckets fuer einen einzelnen User.

    Attributes:
        minute_bucket: Burst-Schutz.
        hour_bucket: Sustained-Load-Schutz.
        day_bucket: Tages-Budget.
        last_activity: Zeitstempel der letzten Aktivitaet (fuer Eviction).
        profile: Aktives Profil.
        warning_sent_minute: Ob 70%-Warnung fuer Minute gesendet wurde.
        warning_sent_hour: Ob 70%-Warnung fuer Stunde gesendet wurde.
        warning_sent_day: Ob 70%-Warnung fuer Tag gesendet wurde.
        unlimited_counter: Zaehler fuer Unlimited-Reminder.
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
    """Ergebnis einer Rate-Limit-Pruefung.

    Attributes:
        allowed: Ob die Anfrage erlaubt ist.
        retry_after: Wartezeit in Sekunden (None wenn erlaubt).
        period: Welches Limit gegriffen hat (minute/hour/day/None).
        limit_value: Maximaler Wert des Limits.
        current_count: Aktueller Verbrauch.
        profile: Aktives Profil des Users.
        warning_70: Ob die 70%-Warnung ausgeloest werden soll.
        warning_period: Welches Window die 70%-Warnung betrifft.
        unlimited_reminder: Ob ein Unlimited-Reminder gesendet werden soll.
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
    """Verbrauchsinformationen fuer /usage.

    Attributes:
        profile: Aktives Profil.
        minute_used: Verbrauch diese Minute.
        minute_limit: Limit pro Minute.
        minute_reset_seconds: Sekunden bis Reset.
        hour_used: Verbrauch diese Stunde.
        hour_limit: Limit pro Stunde.
        hour_reset_seconds: Sekunden bis Reset.
        day_used: Verbrauch heute.
        day_limit: Limit pro Tag.
        day_reset_seconds: Sekunden bis Reset.
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
    """Per-User Rate-Limiter mit drei Zeitfenstern und Profil-System.

    Thread-safe via Lock. Eviction von inaktiven Usern nach 1h.
    Profile werden persistent gespeichert (JSONL).

    Usage:
        limiter = RateLimiter()
        result = limiter.check_and_consume(user_id=12345)
        if not result.allowed:
            # User hat Limit erreicht
            ...
    """

    def __init__(self) -> None:
        self._users: dict[int, _UserBuckets] = {}
        self._lock = Lock()
        self._profiles: dict[int, str] = _load_user_profiles()

    def get_user_profile(self, user_id: int) -> str:
        """Gibt das aktive Profil eines Users zurueck.

        Args:
            user_id: Telegram User-ID.

        Returns:
            Profilname (light, normal, power, unlimited).
        """
        return self._profiles.get(user_id, DEFAULT_PROFILE)

    def set_user_profile(self, user_id: int, chat_id: int, profile: str) -> bool:
        """Setzt das Profil eines Users.

        Persistiert die Aenderung und erstellt neue Buckets mit den
        neuen Limits.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            profile: Profilname (light, normal, power, unlimited).

        Returns:
            True wenn erfolgreich, False wenn Profil ungueltig.
        """
        if profile not in PROFILES:
            return False

        with self._lock:
            self._profiles[user_id] = profile
            # Buckets zuruecksetzen mit neuem Profil
            if user_id in self._users:
                self._users[user_id] = _UserBuckets(profile=profile)

        # Persistent speichern
        _save_user_profile(user_id, chat_id, profile)
        log.info("User %d: Profil gewechselt zu '%s'", user_id, profile)
        return True

    def check_and_consume(self, user_id: int) -> RateLimitResult:
        """Prueft ob der User eine Anfrage senden darf und konsumiert ein Token.

        Prueft alle drei Buckets (Minute, Stunde, Tag). Wenn einer davon
        kein Token hat, wird die Anfrage abgelehnt. Nur wenn alle drei
        erlauben, wird je ein Token konsumiert.

        Im Unlimited-Modus werden keine Tokens konsumiert, aber ein
        Reminder-Counter hochgezaehlt.

        Args:
            user_id: Telegram User-ID.

        Returns:
            RateLimitResult mit allen relevanten Informationen.
        """
        with self._lock:
            self._evict_stale()
            profile = self._profiles.get(user_id, DEFAULT_PROFILE)

            if user_id not in self._users:
                self._users[user_id] = _UserBuckets(profile=profile)
            elif self._users[user_id].profile != profile:
                # Profil hat sich geaendert -> Buckets neu erstellen
                self._users[user_id] = _UserBuckets(profile=profile)

            buckets = self._users[user_id]
            buckets.last_activity = time.monotonic()

            # Unlimited-Modus: kein Limit, aber Reminder-Counter
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

            # Alle drei Buckets pruefen
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
                # Minute-Zaehler zurueckgeben
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
                # Minute und Hour zurueckgeben
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

            # Erfolgreich: 70%-Warnung pruefen
            warning_70 = False
            warning_period: Optional[str] = None

            # Minute-Warnung
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

            # Stunden-Warnung (hat Prioritaet wenn beide gleichzeitig feuern)
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

            # Tages-Warnung
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

            # Warnung-Reset: wenn Bucket wieder aufgefuellt hat (< 50%)
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
        """Gibt aktuelle Verbrauchsinformationen fuer einen User zurueck.

        Args:
            user_id: Telegram User-ID.

        Returns:
            UsageInfo mit Verbrauch, Limits und Reset-Zeiten.
        """
        with self._lock:
            profile = self._profiles.get(user_id, DEFAULT_PROFILE)
            limits = PROFILES.get(profile, PROFILES[DEFAULT_PROFILE])

            if user_id not in self._users:
                # Kein Verbrauch
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
        """Entfernt Buckets von Usern die laenger als TTL inaktiv waren.

        Muss innerhalb von self._lock aufgerufen werden.
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
            log.debug("Rate-Limiter: %d inaktive User evicted", len(stale_ids))

    def _reset_all_for_tests(self) -> None:
        """Setzt alle Buckets und Profile zurueck. NUR fuer Tests."""
        with self._lock:
            self._users.clear()
            self._profiles.clear()
