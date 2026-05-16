"""Tests for application.proactive_trigger_service.

Tests trigger logic, frequency caps, quiet hours, and nudge generation.
Part of P1 (Proactive Memory) and P5 (Time/Pattern Awareness).
"""

import time
from datetime import datetime, timezone

from application.proactive_trigger_service import (
    ProactiveTriggerService,
    TriggerResult,
)


class TestTriggerResult:
    """TriggerResult data structure."""

    def test_default_no_fire(self) -> None:
        """Default TriggerResult does not fire."""
        result = TriggerResult()
        assert result.should_fire is False
        assert result.nudge_text == ""
        assert result.trigger_type == ""

    def test_fire_with_data(self) -> None:
        """TriggerResult with fire data."""
        result = TriggerResult(
            should_fire=True,
            nudge_text="Hey!",
            trigger_type="test",
            reason="Testing",
        )
        assert result.should_fire is True
        assert result.nudge_text == "Hey!"


class TestRecordActivity:
    """Activity recording."""

    def test_record_creates_entry(self) -> None:
        """Recording activity creates an activity record."""
        service = ProactiveTriggerService()
        service.record_activity(123)
        record = service._activity.get(123)
        assert record is not None
        assert len(record.message_timestamps) == 1

    def test_record_accumulates(self) -> None:
        """Multiple recordings accumulate timestamps."""
        service = ProactiveTriggerService()
        service.record_activity(123, timestamp=1000.0)
        service.record_activity(123, timestamp=2000.0)
        service.record_activity(123, timestamp=3000.0)
        record = service._activity[123]
        assert len(record.message_timestamps) == 3

    def test_record_trims_old(self) -> None:
        """Buffer is trimmed when exceeding max size."""
        service = ProactiveTriggerService()
        service._max_timestamps = 5
        for i in range(10):
            service.record_activity(123, timestamp=float(i * 1000))
        record = service._activity[123]
        assert len(record.message_timestamps) == 5


class TestFrequencyCap:
    """Frequency cap enforcement."""

    def test_can_push_first_time(self) -> None:
        """First push of the day is allowed."""
        service = ProactiveTriggerService()
        service.record_activity(123)
        now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        assert service._can_push_proactive(123, now) is True

    def test_cannot_push_after_max(self) -> None:
        """After MAX_PROACTIVE_PER_DAY pushes, no more allowed today."""
        service = ProactiveTriggerService()
        service.record_activity(123)
        now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        # Mark one push
        service._mark_proactive_push(123, now)
        assert service._can_push_proactive(123, now) is False

    def test_new_day_resets_cap(self) -> None:
        """New day resets the push counter."""
        service = ProactiveTriggerService()
        service.record_activity(123)
        day1 = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        service._mark_proactive_push(123, day1)
        # Next day
        day2 = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        assert service._can_push_proactive(123, day2) is True


class TestQuietHours:
    """Quiet hours detection."""

    def test_default_quiet_hours(self) -> None:
        """With no activity data, uses default quiet hours."""
        service = ProactiveTriggerService()
        service.record_activity(123)
        # Default: active 7-23, quiet 23-7
        midnight = datetime(2026, 5, 17, 2, 0, tzinfo=timezone.utc)
        assert service._is_quiet_hours(123, midnight) is True

    def test_active_hours_not_quiet(self) -> None:
        """During active hours, not quiet."""
        service = ProactiveTriggerService()
        service.record_activity(123)
        noon = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        assert service._is_quiet_hours(123, noon) is False

    def test_derived_active_hours(self) -> None:
        """With enough activity data, active hours are derived."""
        service = ProactiveTriggerService()
        # Simulate messages between 9:00 and 20:00 UTC
        for day in range(14):
            for hour in range(9, 21):
                ts = datetime(2026, 5, 1 + day, hour, 30, tzinfo=timezone.utc)
                service.record_activity(123, timestamp=ts.timestamp())
        start, end = service.get_active_hours(123)
        assert start >= 9
        assert end <= 20


class TestSessionStartTrigger:
    """Session start after pause."""

    def test_long_pause_triggers(self) -> None:
        """Pause longer than 3 days triggers welcome-back."""
        service = ProactiveTriggerService()
        # First message 4 days ago
        old_ts = time.time() - (4 * 24 * 3600)
        service.record_activity(123, timestamp=old_ts)
        # New message now
        now_ts = time.time()
        service.record_activity(123, timestamp=now_ts)

        now = datetime.now(timezone.utc)
        result = service.check_triggers(123, 1, "Hallo", current_time=now)
        assert result.should_fire is True
        assert result.trigger_type == "long_pause"
        assert "wieder da" in result.nudge_text or "Tage" in result.nudge_text

    def test_short_pause_no_trigger(self) -> None:
        """Pause shorter than session gap does not trigger."""
        service = ProactiveTriggerService()
        now_ts = time.time()
        service.record_activity(123, timestamp=now_ts - 3600)  # 1 hour ago
        service.record_activity(123, timestamp=now_ts)

        now = datetime.now(timezone.utc)
        result = service.check_triggers(123, 1, "Hallo", current_time=now)
        assert result.should_fire is False

    def test_morning_start_trigger(self) -> None:
        """Morning session start after overnight gap triggers greeting."""
        service = ProactiveTriggerService()
        # Last message yesterday evening
        yesterday_evening = datetime(2026, 5, 16, 21, 0, tzinfo=timezone.utc)
        service.record_activity(123, timestamp=yesterday_evening.timestamp())
        # New message this morning
        this_morning = datetime(2026, 5, 17, 7, 30, tzinfo=timezone.utc)
        service.record_activity(123, timestamp=this_morning.timestamp())

        result = service.check_triggers(
            123, 1, "Guten Morgen", current_time=this_morning
        )
        assert result.should_fire is True
        assert result.trigger_type == "morning_start"
        assert "Morgen" in result.nudge_text


class TestReactiveTrigger:
    """Reactive memory triggers within conversations."""

    def test_memory_with_time_reference_triggers(self) -> None:
        """Memory entry with date reference triggers reactive nudge."""
        service = ProactiveTriggerService()
        service.record_activity(123)

        entries = [
            {"content": "Trademark-Anmeldung bis 16.03.2026 erledigen", "id": "ep_1"},
        ]
        result = service.check_reactive_trigger(123, 1, "Was steht an?", entries)
        assert result.should_fire is True
        assert result.trigger_type == "reactive_memory"

    def test_memory_without_time_no_trigger(self) -> None:
        """Memory entry without time reference does not trigger."""
        service = ProactiveTriggerService()
        service.record_activity(123)

        entries = [
            {"content": "User mag Delfine", "id": "ep_2"},
        ]
        result = service.check_reactive_trigger(123, 1, "Was magst du?", entries)
        assert result.should_fire is False

    def test_reactive_fires_only_once_per_conversation(self) -> None:
        """Reactive trigger fires at most once per conversation window."""
        service = ProactiveTriggerService()
        service.record_activity(123)

        entries = [
            {"content": "Meeting nächste Woche Montag", "id": "ep_3"},
        ]
        # First trigger fires
        result1 = service.check_reactive_trigger(123, 1, "Was steht an?", entries)
        assert result1.should_fire is True

        # Second attempt in same conversation does not fire
        result2 = service.check_reactive_trigger(123, 1, "Und sonst?", entries)
        assert result2.should_fire is False


class TestTimeContextBlock:
    """Time context block for system prompt."""

    def test_contains_current_time(self) -> None:
        """Block contains current time."""
        service = ProactiveTriggerService()
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        block = service.get_time_context_block(123, now=now)
        assert "[TIME CONTEXT]" in block
        assert "2026-05-17 14:30" in block
        assert "Sunday" in block

    def test_weekend_indicator(self) -> None:
        """Block indicates weekend days."""
        service = ProactiveTriggerService()
        # Saturday
        saturday = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)
        block = service.get_time_context_block(123, now=saturday)
        assert "weekend" in block.lower()

    def test_weekday_no_weekend_indicator(self) -> None:
        """Block does not indicate weekend on weekdays."""
        service = ProactiveTriggerService()
        # Wednesday
        wednesday = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        block = service.get_time_context_block(123, now=wednesday)
        assert "weekend" not in block.lower()

    def test_user_active_hours_shown(self) -> None:
        """Block shows user active hours when enough data exists."""
        service = ProactiveTriggerService()
        # Simulate activity pattern
        for day in range(14):
            for hour in range(9, 18):
                ts = datetime(2026, 5, 1 + day, hour, 0, tzinfo=timezone.utc)
                service.record_activity(123, timestamp=ts.timestamp())

        now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        block = service.get_time_context_block(123, now=now)
        assert "typically active" in block.lower()


class TestHasTimeReference:
    """Time reference detection in memory content."""

    def test_date_format_dd_mm_yyyy(self) -> None:
        """Detects DD.MM.YYYY date format."""
        assert ProactiveTriggerService._has_time_reference("Termin am 16.03.2026")

    def test_date_format_iso(self) -> None:
        """Detects ISO date format."""
        assert ProactiveTriggerService._has_time_reference("Deadline: 2026-03-16")

    def test_weekday_german(self) -> None:
        """Detects German weekday names."""
        assert ProactiveTriggerService._has_time_reference("Meeting Montag")

    def test_relative_time_german(self) -> None:
        """Detects relative time references in German."""
        assert ProactiveTriggerService._has_time_reference("Das mache ich morgen")

    def test_no_time_reference(self) -> None:
        """Returns False for text without time references."""
        assert not ProactiveTriggerService._has_time_reference("Ich mag Delfine")


class TestNudgeTonality:
    """Verify nudge texts use reminder style, not to-do style."""

    def test_welcome_back_not_imperative(self) -> None:
        """Welcome-back nudge uses friendly, not commanding tone."""
        service = ProactiveTriggerService()
        now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        nudge = service._build_welcome_back_nudge(5.0, now)
        # Should not contain imperative/commanding language
        assert "musst" not in nudge.lower()
        assert "vergiss nicht" not in nudge.lower()
        assert "du musst" not in nudge.lower()

    def test_morning_nudge_not_imperative(self) -> None:
        """Morning nudge uses greeting style, not task-list."""
        service = ProactiveTriggerService()
        now = datetime(2026, 5, 17, 8, 0, tzinfo=timezone.utc)
        nudge = service._build_morning_nudge(now)
        assert "musst" not in nudge.lower()
        assert "Morgen" in nudge

    def test_memory_nudge_uses_reminder_style(self) -> None:
        """Memory nudge uses 'mir faellt ein' style."""
        service = ProactiveTriggerService()
        nudge = service._build_memory_nudge("ODS Report Montag", "time_reference")
        assert "fällt" in nudge.lower() or "erwähnt" in nudge.lower()
        assert "musst" not in nudge.lower()
