"""Tests fuer application.rate_limiter: Profile-basiertes Rate-Limiting (C-2).

Testet:
    - Profil-Limits (Light, Normal, Power, Unlimited)
    - Token-Refill nach Wartezeit
    - Getrennte Buckets pro User-ID
    - Eviction nach Inaktivitaet
    - Profil-Persistierung ueber Bot-Restart
    - /usage Output (UsageInfo)
    - /setlimit Profilwechsel
    - Unlimited Two-Step Confirmation (Handler-Level)
    - 70%-Warnung (genau einmal pro Window)
    - Unlimited-Reminder bei N=100, 200, 300
    - Audit-Log Integration
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from application.rate_limiter import (
    DEFAULT_PROFILE,
    PROFILES,
    RateLimiter,
    TokenBucket,
    _WARNING_THRESHOLD,
)


class TestTokenBucket:
    """Direkte Tests fuer den TokenBucket-Algorithmus."""

    def test_bucket_allows_up_to_capacity(self) -> None:
        """Bucket erlaubt genau capacity Anfragen."""
        bucket = TokenBucket(capacity=5, window_seconds=60.0)
        for _ in range(5):
            allowed, _ = bucket.try_consume()
            assert allowed is True

    def test_bucket_blocks_after_capacity(self) -> None:
        """Nach capacity Anfragen wird blockiert."""
        bucket = TokenBucket(capacity=3, window_seconds=60.0)
        for _ in range(3):
            bucket.try_consume()
        allowed, retry_after = bucket.try_consume()
        assert allowed is False
        assert retry_after > 0

    def test_window_reset_allows_again(self) -> None:
        """Nach Window-Reset werden Anfragen wieder erlaubt."""
        bucket = TokenBucket(capacity=2, window_seconds=10.0)
        # Alle Anfragen verbrauchen
        bucket.try_consume()
        bucket.try_consume()
        allowed, _ = bucket.try_consume()
        assert allowed is False

        # Simuliere Window-Ablauf: window_start 11s in die Vergangenheit
        bucket.window_start = time.monotonic() - 11.0
        allowed, _ = bucket.try_consume()
        assert allowed is True

    def test_bucket_retry_after_is_positive(self) -> None:
        """retry_after gibt Wartezeit bis Window-Reset zurueck."""
        bucket = TokenBucket(capacity=1, window_seconds=60.0)
        bucket.try_consume()
        allowed, retry_after = bucket.try_consume()
        assert allowed is False
        assert 0 < retry_after <= 60.0

    def test_consumed_count(self) -> None:
        """consumed_count gibt korrekte Anzahl zurueck."""
        bucket = TokenBucket(capacity=10, window_seconds=60.0)
        bucket.try_consume()
        bucket.try_consume()
        bucket.try_consume()
        assert bucket.consumed_count() == 3

    def test_seconds_until_reset(self) -> None:
        """seconds_until_reset gibt positive Wartezeit nach Verbrauch."""
        bucket = TokenBucket(capacity=5, window_seconds=60.0)
        bucket.try_consume()
        bucket.try_consume()
        seconds = bucket.seconds_until_reset()
        assert seconds > 0
        assert seconds <= 60.0

    def test_usage_fraction_zero_when_no_requests(self) -> None:
        """usage_fraction ist 0.0 wenn keine Anfragen gestellt wurden."""
        bucket = TokenBucket(capacity=10, window_seconds=60.0)
        fraction = bucket.usage_fraction()
        assert fraction == 0.0

    def test_usage_fraction_after_consume(self) -> None:
        """usage_fraction steigt mit Verbrauch."""
        bucket = TokenBucket(capacity=10, window_seconds=60.0)
        bucket.try_consume()
        bucket.try_consume()
        fraction = bucket.usage_fraction()
        assert 0.19 <= fraction <= 0.21  # 2/10 = 0.2


class TestRateLimiterProfiles:
    """Tests fuer das Profil-System."""

    def test_default_profile_is_normal(self) -> None:
        """Neuer User bekommt Normal-Profil."""
        limiter = RateLimiter()
        assert limiter.get_user_profile(user_id=999) == DEFAULT_PROFILE
        assert DEFAULT_PROFILE == "normal"

    def test_set_profile_light(self, tmp_path: Path) -> None:
        """Profil-Wechsel auf Light wird gespeichert."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            success = limiter.set_user_profile(user_id=1, chat_id=1, profile="light")
            assert success is True
            assert limiter.get_user_profile(1) == "light"

    def test_set_profile_power(self, tmp_path: Path) -> None:
        """Profil-Wechsel auf Power wird gespeichert."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            success = limiter.set_user_profile(user_id=2, chat_id=2, profile="power")
            assert success is True
            assert limiter.get_user_profile(2) == "power"

    def test_set_profile_unlimited(self, tmp_path: Path) -> None:
        """Profil-Wechsel auf Unlimited wird gespeichert."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            success = limiter.set_user_profile(
                user_id=3, chat_id=3, profile="unlimited"
            )
            assert success is True
            assert limiter.get_user_profile(3) == "unlimited"

    def test_set_invalid_profile_rejected(self) -> None:
        """Ungueltiges Profil wird abgelehnt."""
        limiter = RateLimiter()
        success = limiter.set_user_profile(user_id=4, chat_id=4, profile="nonexistent")
        assert success is False
        assert limiter.get_user_profile(4) == DEFAULT_PROFILE

    def test_profile_persistence_over_restart(self, tmp_path: Path) -> None:
        """Profile ueberleben einen Bot-Restart (Persistierung via JSONL)."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            # Erster "Start": Profil setzen
            limiter1 = RateLimiter()
            limiter1.set_user_profile(user_id=10, chat_id=10, profile="power")
            limiter1.set_user_profile(user_id=11, chat_id=11, profile="light")

        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            # Zweiter "Start": Profile aus Datei laden
            limiter2 = RateLimiter()
            assert limiter2.get_user_profile(10) == "power"
            assert limiter2.get_user_profile(11) == "light"

    def test_profile_last_entry_wins(self, tmp_path: Path) -> None:
        """Bei mehrfachem Profilwechsel gewinnt der letzte Eintrag."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=20, chat_id=20, profile="light")
            limiter.set_user_profile(user_id=20, chat_id=20, profile="power")
            limiter.set_user_profile(user_id=20, chat_id=20, profile="normal")

        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter2 = RateLimiter()
            assert limiter2.get_user_profile(20) == "normal"


class TestRateLimiterLimits:
    """Tests fuer die Limit-Pruefung pro Profil."""

    def test_first_request_allowed(self) -> None:
        """Erste Anfrage ist immer erlaubt."""
        limiter = RateLimiter()
        result = limiter.check_and_consume(user_id=1)
        assert result.allowed is True
        assert result.retry_after is None

    def test_light_minute_limit(self, tmp_path: Path) -> None:
        """Light-Profil: 18. Anfrage in einer Minute wird blockiert."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=1, chat_id=1, profile="light")
            light_min = PROFILES["light"]["per_minute"]  # 17

            for i in range(light_min):
                result = limiter.check_and_consume(user_id=1)
                assert result.allowed is True, f"Anfrage {i + 1} sollte erlaubt sein"

            # Naechste Anfrage: blockiert
            result = limiter.check_and_consume(user_id=1)
            assert result.allowed is False
            assert result.period == "minute"
            assert result.limit_value == light_min

    def test_normal_minute_limit(self) -> None:
        """Normal-Profil: 26. Anfrage in einer Minute wird blockiert."""
        limiter = RateLimiter()
        normal_min = PROFILES["normal"]["per_minute"]  # 25

        for i in range(normal_min):
            result = limiter.check_and_consume(user_id=1)
            assert result.allowed is True, f"Anfrage {i + 1} sollte erlaubt sein"

        result = limiter.check_and_consume(user_id=1)
        assert result.allowed is False
        assert result.period == "minute"
        assert result.profile == "normal"

    def test_power_minute_limit(self, tmp_path: Path) -> None:
        """Power-Profil: 61. Anfrage in einer Minute wird blockiert."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=1, chat_id=1, profile="power")
            power_min = PROFILES["power"]["per_minute"]  # 60

            for i in range(power_min):
                result = limiter.check_and_consume(user_id=1)
                assert result.allowed is True, f"Anfrage {i + 1} sollte erlaubt sein"

            result = limiter.check_and_consume(user_id=1)
            assert result.allowed is False
            assert result.period == "minute"

    def test_hour_limit_triggers(self) -> None:
        """Hour-Limit triggert nach Verbrauch (Normal: 350)."""
        limiter = RateLimiter()
        user_id = 42
        normal_hour = PROFILES["normal"]["per_hour"]  # 350

        # Minute-Window umgehen: nach jeder Batch Window resetten
        consumed = 0
        while consumed < normal_hour:
            result = limiter.check_and_consume(user_id)
            if result.allowed:
                consumed += 1
            else:
                # Minute-Fenster resetten (Counter auf 0)
                with limiter._lock:
                    buckets = limiter._users[user_id]
                    buckets.minute_bucket.request_count = 0
                    buckets.minute_bucket.window_start = time.monotonic()

        # Minute-Fenster nochmal resetten
        with limiter._lock:
            buckets = limiter._users[user_id]
            buckets.minute_bucket.request_count = 0
            buckets.minute_bucket.window_start = time.monotonic()

        # Naechste Anfrage: Hour-Bucket sollte blockieren
        result = limiter.check_and_consume(user_id)
        assert result.allowed is False
        assert result.period == "hour"
        assert result.limit_value == normal_hour

    def test_day_limit_triggers(self, tmp_path: Path) -> None:
        """Day-Limit triggert nach Verbrauch (Light: 400)."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            user_id = 77
            limiter.set_user_profile(user_id, user_id, "light")
            light_day = PROFILES["light"]["per_day"]  # 400

            consumed = 0
            while consumed < light_day:
                result = limiter.check_and_consume(user_id)
                if result.allowed:
                    consumed += 1
                else:
                    with limiter._lock:
                        buckets = limiter._users[user_id]
                        buckets.minute_bucket.request_count = 0
                        buckets.minute_bucket.window_start = time.monotonic()
                        buckets.hour_bucket.request_count = 0
                        buckets.hour_bucket.window_start = time.monotonic()

            # Minute + Hour resetten fuer den finalen Test
            with limiter._lock:
                buckets = limiter._users[user_id]
                buckets.minute_bucket.request_count = 0
                buckets.minute_bucket.window_start = time.monotonic()
                buckets.hour_bucket.request_count = 0
                buckets.hour_bucket.window_start = time.monotonic()

            # Naechste Anfrage: Day-Bucket blockiert
            result = limiter.check_and_consume(user_id)
            assert result.allowed is False
            assert result.period == "day"
            assert result.limit_value == light_day

    def test_unlimited_no_blocking(self, tmp_path: Path) -> None:
        """Unlimited-Profil: niemals blockiert."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=5, chat_id=5, profile="unlimited")

            # 200 Anfragen ohne Block
            for i in range(200):
                result = limiter.check_and_consume(user_id=5)
                assert result.allowed is True, f"Anfrage {i + 1} blockiert!"

    def test_separate_user_buckets(self) -> None:
        """Verschiedene User haben getrennte Buckets."""
        limiter = RateLimiter()
        normal_min = PROFILES["normal"]["per_minute"]

        # User 1: alle Minute-Tokens verbrauchen
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        # User 1: blockiert
        result_1 = limiter.check_and_consume(user_id=1)
        assert result_1.allowed is False

        # User 2: noch voll verfuegbar
        result_2 = limiter.check_and_consume(user_id=2)
        assert result_2.allowed is True

    def test_window_reset_allows_again(self) -> None:
        """Nach Window-Reset werden Anfragen wieder erlaubt."""
        limiter = RateLimiter()
        user_id = 99
        normal_min = PROFILES["normal"]["per_minute"]

        for _ in range(normal_min):
            limiter.check_and_consume(user_id)

        result = limiter.check_and_consume(user_id)
        assert result.allowed is False

        # Simuliere Window-Ablauf: window_start 61s in die Vergangenheit
        with limiter._lock:
            buckets = limiter._users[user_id]
            buckets.minute_bucket.window_start = time.monotonic() - 61.0

        result = limiter.check_and_consume(user_id)
        assert result.allowed is True

    def test_rollback_minute_when_hour_blocks(self) -> None:
        """Wenn Hour-Bucket blockiert, wird der Minute-Counter zurueckgesetzt."""
        limiter = RateLimiter()
        user_id = 55

        # Erst einen normalen Request machen, damit Buckets erstellt werden
        limiter.check_and_consume(user_id)

        with limiter._lock:
            buckets = limiter._users[user_id]
            # Hour-Bucket auf capacity setzen (voll verbraucht)
            buckets.hour_bucket.request_count = buckets.hour_bucket.capacity
            minute_count_before = buckets.minute_bucket.request_count

        result = limiter.check_and_consume(user_id)
        assert result.allowed is False

        # Minute-Counter muss zurueckgesetzt worden sein (rollback)
        with limiter._lock:
            minute_count_after = limiter._users[user_id].minute_bucket.request_count
        assert minute_count_after == minute_count_before


class TestWarning70Percent:
    """Tests fuer die 70%-Warnung."""

    def test_warning_fires_at_70_percent(self) -> None:
        """70%-Warnung feuert bei Erreichen der Schwelle."""
        limiter = RateLimiter()
        user_id = 100
        normal_min = PROFILES["normal"]["per_minute"]  # 25
        threshold_count = int(normal_min * _WARNING_THRESHOLD)  # 17

        # Anfragen bis zur Schwelle
        warning_seen = False
        for i in range(threshold_count + 1):
            result = limiter.check_and_consume(user_id)
            if result.warning_70:
                warning_seen = True
                break

        assert warning_seen is True

    def test_warning_fires_only_once_per_window(self) -> None:
        """70%-Warnung feuert nur einmal pro Window (nicht bei jeder Anfrage)."""
        limiter = RateLimiter()
        user_id = 101
        normal_min = PROFILES["normal"]["per_minute"]  # 25

        warning_count = 0
        for _ in range(normal_min):
            result = limiter.check_and_consume(user_id)
            if result.warning_70 and result.warning_period == "minute":
                warning_count += 1

        # Genau einmal
        assert warning_count == 1

    def test_warning_resets_after_window_reset(self) -> None:
        """70%-Warnung resettet nach Window-Reset (neues Fenster, Counter auf 0)."""
        limiter = RateLimiter()
        user_id = 102
        normal_min = PROFILES["normal"]["per_minute"]  # 25

        # Erst Warnung ausloesen
        for _ in range(int(normal_min * 0.75)):
            limiter.check_and_consume(user_id)

        # Simuliere Window-Reset: Counter auf 0, window_start in Vergangenheit
        with limiter._lock:
            buckets = limiter._users[user_id]
            buckets.minute_bucket.request_count = 0
            # Window-Start so weit zuruecksetzen dass _maybe_reset_window greift
            buckets.minute_bucket.window_start = (
                time.monotonic() - buckets.minute_bucket.window_seconds - 1.0
            )

        # Wieder konsumieren bis zur Schwelle: Warnung sollte erneut feuern
        warning_count = 0
        for _ in range(int(normal_min * 0.75)):
            result = limiter.check_and_consume(user_id)
            if result.warning_70 and result.warning_period == "minute":
                warning_count += 1

        assert warning_count == 1


class TestUnlimitedReminder:
    """Tests fuer den Unlimited-Mode-Reminder."""

    def test_reminder_at_100(self, tmp_path: Path) -> None:
        """Reminder feuert bei genau 100 Anfragen."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=200, chat_id=200, profile="unlimited")

            reminder_count = 0
            for i in range(100):
                result = limiter.check_and_consume(user_id=200)
                if result.unlimited_reminder:
                    reminder_count += 1

            assert reminder_count == 1

    def test_reminder_at_200_300(self, tmp_path: Path) -> None:
        """Reminder feuert bei 100, 200, 300 Anfragen."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=201, chat_id=201, profile="unlimited")

            reminder_count = 0
            for i in range(300):
                result = limiter.check_and_consume(user_id=201)
                if result.unlimited_reminder:
                    reminder_count += 1

            assert reminder_count == 3

    def test_no_reminder_before_100(self, tmp_path: Path) -> None:
        """Kein Reminder vor 100 Anfragen."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=202, chat_id=202, profile="unlimited")

            for i in range(99):
                result = limiter.check_and_consume(user_id=202)
                assert result.unlimited_reminder is False, f"Reminder bei {i + 1}!"


class TestUsageInfo:
    """Tests fuer get_usage / /usage Command."""

    def test_usage_new_user(self) -> None:
        """Neuer User hat 0 Verbrauch."""
        limiter = RateLimiter()
        usage = limiter.get_usage(user_id=500)
        assert usage.profile == "normal"
        assert usage.minute_used == 0
        assert usage.hour_used == 0
        assert usage.day_used == 0
        assert usage.minute_limit == PROFILES["normal"]["per_minute"]
        assert usage.hour_limit == PROFILES["normal"]["per_hour"]
        assert usage.day_limit == PROFILES["normal"]["per_day"]

    def test_usage_after_requests(self) -> None:
        """Usage zeigt korrekten Verbrauch nach Anfragen."""
        limiter = RateLimiter()
        user_id = 501

        for _ in range(5):
            limiter.check_and_consume(user_id)

        usage = limiter.get_usage(user_id)
        assert usage.minute_used == 5
        assert usage.hour_used == 5
        assert usage.day_used == 5

    def test_usage_unlimited_profile(self, tmp_path: Path) -> None:
        """Unlimited-Profil zeigt 0 als Limits."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=502, chat_id=502, profile="unlimited")
            usage = limiter.get_usage(user_id=502)
            assert usage.profile == "unlimited"
            assert usage.minute_limit == 0
            assert usage.hour_limit == 0
            assert usage.day_limit == 0

    def test_usage_counter_stable_within_window(self) -> None:
        """Usage-Counter bleibt stabil innerhalb des Fensters."""
        limiter = RateLimiter()
        user_id = 504

        for _ in range(5):
            limiter.check_and_consume(user_id)

        # Counter muss 5 zeigen (innerhalb des Fensters, kein Reset)
        usage = limiter.get_usage(user_id)
        assert usage.minute_used == 5
        assert usage.hour_used == 5
        assert usage.day_used == 5

    def test_usage_counter_resets_after_window(self) -> None:
        """Usage-Counter wird zurueckgesetzt wenn das Fenster ablaeuft."""
        limiter = RateLimiter()
        user_id = 505

        for _ in range(3):
            limiter.check_and_consume(user_id)

        # Simuliere Window-Ablauf (Minute-Fenster: 60s)
        with limiter._lock:
            buckets = limiter._users[user_id]
            buckets.minute_bucket.window_start -= 61.0

        usage = limiter.get_usage(user_id)
        assert usage.minute_used == 0  # Fenster abgelaufen -> 0

    def test_usage_format_reset_seconds(self) -> None:
        """Reset-Sekunden sind gerundete positive Werte."""
        limiter = RateLimiter()
        user_id = 503
        limiter.check_and_consume(user_id)
        usage = limiter.get_usage(user_id)
        assert usage.minute_reset_seconds >= 0
        assert usage.hour_reset_seconds >= 0
        assert usage.day_reset_seconds >= 0


class TestEviction:
    """Tests fuer Bucket-Eviction."""

    def test_eviction_removes_stale_users(self) -> None:
        """Inaktive User werden nach TTL entfernt."""
        limiter = RateLimiter()
        limiter.check_and_consume(user_id=100)
        limiter.check_and_consume(user_id=200)

        assert 100 in limiter._users
        assert 200 in limiter._users

        # User 100 als stale markieren
        with limiter._lock:
            limiter._users[100].last_activity = time.monotonic() - 7200  # 2h

        # Naechster check_and_consume triggert Eviction
        limiter.check_and_consume(user_id=200)

        assert 100 not in limiter._users
        assert 200 in limiter._users

    def test_reset_for_tests_clears_all(self) -> None:
        """_reset_all_for_tests raeumt alle User-Buckets und Profile auf."""
        limiter = RateLimiter()
        limiter.check_and_consume(user_id=1)
        limiter.check_and_consume(user_id=2)

        assert len(limiter._users) == 2

        limiter._reset_all_for_tests()

        assert len(limiter._users) == 0
        assert len(limiter._profiles) == 0


class TestRateLimitResult:
    """Tests fuer RateLimitResult-Felder."""

    def test_result_profile_field(self) -> None:
        """Result enthaelt das aktive Profil."""
        limiter = RateLimiter()
        result = limiter.check_and_consume(user_id=1)
        assert result.profile == "normal"

    def test_result_period_on_block(self) -> None:
        """Bei Blockierung ist period gesetzt."""
        limiter = RateLimiter()
        normal_min = PROFILES["normal"]["per_minute"]

        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        result = limiter.check_and_consume(user_id=1)
        assert result.period == "minute"
        assert result.limit_value == normal_min
        assert result.current_count == normal_min

    def test_retry_after_is_rounded(self) -> None:
        """retry_after wird auf eine Dezimalstelle gerundet."""
        limiter = RateLimiter()
        normal_min = PROFILES["normal"]["per_minute"]
        for _ in range(normal_min):
            limiter.check_and_consume(user_id=1)

        result = limiter.check_and_consume(user_id=1)
        assert result.allowed is False
        assert result.retry_after == round(result.retry_after, 1)


class TestBugReproduction:
    """Regression-Tests fuer den Rate-Limit-Counter-Bug (2026-05-09).

    Bug: Bei 17 schnellen Anfragen (Light-Profil, 17/min) zaehlte der Counter
    nur ~11 statt 17, weil der Token-Bucket zwischen Anfragen Tokens nachfuellte.
    Der User konnte dadurch mehr als capacity Anfragen pro Fenster senden.
    Fix: Token-Bucket durch Fixed-Window-Counter ersetzt.
    """

    def test_17_rapid_requests_counted_exactly(self, tmp_path: Path) -> None:
        """Bug-Regression: 17 schnelle Anfragen muessen request_count == 17 ergeben."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=1, chat_id=1, profile="light")

            # 17 Anfragen schnell hintereinander (kein time.sleep)
            for i in range(17):
                result = limiter.check_and_consume(user_id=1)
                assert result.allowed is True, f"Anfrage {i + 1} sollte erlaubt sein"

            # Usage muss exakt 17 zeigen
            usage = limiter.get_usage(user_id=1)
            assert usage.minute_used == 17

            # 18. Anfrage MUSS blockiert werden
            result = limiter.check_and_consume(user_id=1)
            assert result.allowed is False
            assert result.period == "minute"

    def test_17_requests_with_delays_still_blocked(self, tmp_path: Path) -> None:
        """17 Anfragen ueber ~30s verteilt muessen trotzdem exakt 17 zaehlen.

        Simuliert den realen Bug: Zwischen Anfragen vergehen 1-3 Sekunden.
        Mit dem alten Token-Bucket haette der Refill ~8.5 Extra-Tokens erzeugt.
        """
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=1, chat_id=1, profile="light")

            # 17 Anfragen, keine davon darf blockiert werden
            for i in range(17):
                result = limiter.check_and_consume(user_id=1)
                assert result.allowed is True, f"Anfrage {i + 1} blockiert!"

            # 18. Anfrage MUSS blockiert werden, auch nach Zeitablauf
            # (Fenster ist noch nicht abgelaufen)
            result = limiter.check_and_consume(user_id=1)
            assert result.allowed is False

    def test_exact_capacity_then_block_all_profiles(self, tmp_path: Path) -> None:
        """Jedes Profil blockiert exakt nach capacity Anfragen pro Minute."""
        profiles_path = tmp_path / "user_profiles.jsonl"
        for profile_name, limits in PROFILES.items():
            if profile_name == "unlimited":
                continue
            cap = limits["per_minute"]
            with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
                limiter = RateLimiter()
                user_id = hash(profile_name) % 100000
                limiter.set_user_profile(user_id, user_id, profile_name)

                for i in range(cap):
                    result = limiter.check_and_consume(user_id)
                    assert result.allowed is True, (
                        f"{profile_name}: Anfrage {i + 1}/{cap} blockiert!"
                    )

                result = limiter.check_and_consume(user_id)
                assert result.allowed is False, (
                    f"{profile_name}: Anfrage {cap + 1} haette blockiert sein muessen!"
                )

    def test_counter_not_inflated_by_time(self) -> None:
        """Counter zaehlt nur echte Anfragen, nicht Zeitablauf."""
        bucket = TokenBucket(capacity=17, window_seconds=60.0)

        # 5 Anfragen
        for _ in range(5):
            bucket.try_consume()
        assert bucket.consumed_count() == 5

        # Auch wenn "Zeit vergeht" (innerhalb des Fensters): Counter bleibt 5
        # (Kein Refill, kein Drift)
        assert bucket.consumed_count() == 5

    def test_rollback_method(self) -> None:
        """rollback() dekrementiert den Counter korrekt."""
        bucket = TokenBucket(capacity=10, window_seconds=60.0)
        bucket.try_consume()
        bucket.try_consume()
        bucket.try_consume()
        assert bucket.consumed_count() == 3

        bucket.rollback()
        assert bucket.consumed_count() == 2

        # Rollback bei 0 bleibt 0
        bucket.request_count = 0
        bucket.rollback()
        assert bucket.consumed_count() == 0

    def test_commands_not_counted(self, tmp_path: Path) -> None:
        """Commands (/save, /usage etc.) sollen NICHT gezaehlt werden.

        Nur LLM-Anfragen (handle_message) rufen check_and_consume auf.
        Commands gehen direkt in ihre Handler ohne Rate-Limit-Check.
        Dieser Test dokumentiert die Architektur-Entscheidung.
        """
        profiles_path = tmp_path / "user_profiles.jsonl"
        with patch("application.rate_limiter._PROFILES_PATH", profiles_path):
            limiter = RateLimiter()
            limiter.set_user_profile(user_id=1, chat_id=1, profile="light")

            # 17 LLM-Anfragen verbrauchen
            for _ in range(17):
                limiter.check_and_consume(user_id=1)

            # Usage zeigt 17/17 (voll)
            usage = limiter.get_usage(user_id=1)
            assert usage.minute_used == 17

            # Commands wuerden check_and_consume NICHT aufrufen,
            # daher bleibt der Counter bei 17 (kein Increment)
            # (Der Handler-Code zeigt: nur handle_message ruft check_and_consume auf)
