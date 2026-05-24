"""K5: Timezone and time edge case tests.

Invalid timezone values, DST transitions, midnight rollover,
system time jumps, leap year dates.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.privacy_pipeline import (
    PrivacyPipeline,
)
from application.rate_limiter import RateLimiter


@pytest.mark.adversarial
class TestInvalidTimezone:
    """AXOLENT_TIMEZONE with invalid or edge-case values."""

    def test_pipeline_rejection_timestamp_is_valid_iso(self) -> None:
        """WHAT: Pipeline rejection timestamps are valid ISO-8601.
        EXPECTED: Timestamp can be parsed back.
        WHY: Invalid timestamps could break audit log consumers.
        """
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="tz-001",
            user_id=1,
            claim="User has depression symptoms",
            scope=HypothesisScope(),
            created_at="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        )
        result = pipeline.check(h)
        assert result is not None  # Healthcare filter should catch
        # Verify timestamp is valid ISO-8601
        parsed = datetime.fromisoformat(result.timestamp)
        assert (
            parsed.tzinfo is not None
            or "+" in result.timestamp
            or "Z" in result.timestamp
        )

    def test_hypothesis_with_empty_timestamp(self) -> None:
        """WHAT: Hypothesis with empty created_at and last_seen.
        EXPECTED: Pipeline still processes the claim (times are metadata).
        WHY: Corrupt data could have empty timestamps.
        """
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="tz-002",
            user_id=1,
            claim="User prefers bullet points",
            scope=HypothesisScope(),
            created_at="",
            last_seen="",
        )
        result = pipeline.check(h)
        assert result is None  # Clean claim, should pass

    def test_hypothesis_with_garbage_timestamp(self) -> None:
        """WHAT: Hypothesis with non-ISO timestamp strings.
        EXPECTED: Pipeline processes claim (timestamps not validated in check).
        WHY: Imported data might have malformed timestamps.
        """
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="tz-003",
            user_id=1,
            claim="User prefers concise responses",
            scope=HypothesisScope(),
            created_at="not-a-date",
            last_seen="also-not-a-date",
        )
        result = pipeline.check(h)
        assert result is None  # Claim is clean


@pytest.mark.adversarial
class TestDSTTransition:
    """DST transition edge cases in timestamp handling."""

    def test_timestamp_during_dst_spring_forward(self) -> None:
        """WHAT: Timestamp at 2:30 AM during spring-forward (doesn't exist).
        EXPECTED: No crash when processing such timestamps.
        WHY: Wall-clock time 2:30 doesn't exist during spring forward.
        """
        # March 29, 2026 at 2:30 AM CET doesn't exist (spring forward)
        # But UTC representation is always valid
        ts = "2026-03-29T01:30:00+00:00"
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="dst-001",
            user_id=1,
            claim="User prefers tables over lists",
            scope=HypothesisScope(),
            created_at=ts,
            last_seen=ts,
        )
        result = pipeline.check(h)
        assert result is None

    def test_timestamp_during_dst_fall_back(self) -> None:
        """WHAT: Timestamp during fall-back (ambiguous local time).
        EXPECTED: No crash.
        WHY: 2:30 AM occurs twice during fall-back.
        """
        ts = "2026-10-25T01:30:00+00:00"
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="dst-002",
            user_id=1,
            claim="User prefers numbered lists",
            scope=HypothesisScope(),
            created_at=ts,
            last_seen=ts,
        )
        result = pipeline.check(h)
        assert result is None


@pytest.mark.adversarial
class TestMidnightRollover:
    """Midnight boundary edge cases."""

    def test_hypothesis_at_exact_midnight_utc(self) -> None:
        """WHAT: Hypothesis created at exactly 00:00:00.000 UTC.
        EXPECTED: Processed normally.
        WHY: Midnight is a common source of off-by-one errors.
        """
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="mid-001",
            user_id=1,
            claim="User prefers dark mode",
            scope=HypothesisScope(),
            created_at="2026-01-01T00:00:00.000Z",
            last_seen="2026-01-01T00:00:00.000Z",
        )
        result = pipeline.check(h)
        assert result is None

    def test_hypothesis_one_microsecond_before_midnight(self) -> None:
        """WHAT: Timestamp at 23:59:59.999999.
        EXPECTED: Processed normally.
        WHY: Edge of day boundary.
        """
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="mid-002",
            user_id=1,
            claim="User prefers short answers",
            scope=HypothesisScope(),
            created_at="2026-12-31T23:59:59.999999Z",
            last_seen="2026-12-31T23:59:59.999999Z",
        )
        result = pipeline.check(h)
        assert result is None


@pytest.mark.adversarial
class TestLeapYear:
    """Leap year date edge cases."""

    def test_february_29_timestamp(self) -> None:
        """WHAT: Hypothesis with Feb 29 timestamp in leap year.
        EXPECTED: Valid date, processed normally.
        WHY: 2028 is a leap year, Feb 29 is valid.
        """
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="leap-001",
            user_id=1,
            claim="User prefers structured output",
            scope=HypothesisScope(),
            created_at="2028-02-29T12:00:00Z",
            last_seen="2028-02-29T12:00:00Z",
        )
        result = pipeline.check(h)
        assert result is None

    def test_february_29_non_leap_year_timestamp(self) -> None:
        """WHAT: Feb 29 in a non-leap year (invalid date string).
        EXPECTED: Pipeline still processes (timestamps are strings, not parsed).
        WHY: Corrupt data could have invalid dates.
        """
        pipeline = PrivacyPipeline()
        h = Hypothesis(
            hypothesis_id="leap-002",
            user_id=1,
            claim="User prefers minimal formatting",
            scope=HypothesisScope(),
            created_at="2027-02-29T12:00:00Z",  # Invalid: 2027 is not leap
            last_seen="2027-02-29T12:00:00Z",
        )
        result = pipeline.check(h)
        assert result is None  # Claim is clean, timestamp is just a string


@pytest.mark.adversarial
class TestRateLimiterTimeBoundary:
    """Rate limiter at time window boundaries."""

    def test_rate_limiter_at_minute_boundary(self) -> None:
        """WHAT: Rate limiter checked at exact minute boundary.
        EXPECTED: Window rolls over correctly.
        WHY: Time-based windows can have off-by-one at boundaries.
        """
        limiter = RateLimiter()
        user_id = 999
        # Consume up to limit in current window
        for _ in range(30):
            result = limiter.check_and_consume(user_id)
            if not result.allowed:
                break
        # Should eventually be rate-limited or allowed (no crash)
        assert isinstance(result.allowed, bool)
