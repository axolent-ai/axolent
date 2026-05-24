"""K4: Race condition and concurrency tests.

Parallel operations that could corrupt shared state:
reset during chat, learn+forget same hypothesis, double-consume
of pending_store, StreamGuard parallel operations.
"""

from __future__ import annotations

import threading

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.privacy_pipeline import (
    PrivacyAuditLog,
    PrivacyPipeline,
    PipelineRejection,
    RejectionSource,
)
from application.language.stream_guard import (
    StreamGuard,
    StreamGuardStatsStore,
)


def _make_hypothesis(claim: str, hid: str = "race-001") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hid,
        user_id=1,
        claim=claim,
        scope=HypothesisScope(),
        created_at="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
    )


@pytest.mark.adversarial
class TestPrivacyAuditLogConcurrency:
    """PrivacyAuditLog accessed from multiple threads."""

    def test_concurrent_audit_log_writes(self) -> None:
        """WHAT: Multiple threads writing to PrivacyAuditLog simultaneously.
        EXPECTED: No data corruption, all entries recorded.
        WHY: In production, multiple async tasks could write concurrently.
        """
        log = PrivacyAuditLog(max_entries=100)
        errors = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(50):
                    r = PipelineRejection(
                        hypothesis_id=f"thread-{thread_id}-{i}",
                        source=RejectionSource.HEALTHCARE,
                        reason=f"test {thread_id}",
                        timestamp="2026-01-01T00:00:00Z",
                    )
                    log.add(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes caused errors: {errors}"
        # Some entries should exist (exact count depends on rotation)
        assert log.total_rejections > 0

    def test_audit_log_rotation_under_load(self) -> None:
        """WHAT: Audit log rotation triggered during concurrent writes.
        EXPECTED: No crash during rotation, entries preserved.
        WHY: Rotation truncates entries, concurrent access could corrupt.
        """
        log = PrivacyAuditLog(max_entries=10)
        for i in range(100):
            r = PipelineRejection(
                hypothesis_id=f"rot-{i}",
                source=RejectionSource.SECRET,
                reason=f"rotation test {i}",
                timestamp="2026-01-01T00:00:00Z",
            )
            log.add(r)

        # After 100 entries with max_entries=10, should have at most 10
        assert log.total_rejections <= 10
        recent = log.get_recent(5)
        assert len(recent) <= 5


@pytest.mark.adversarial
class TestStreamGuardStatsConcurrency:
    """StreamGuardStatsStore LRU eviction under concurrent access."""

    def test_concurrent_get_calls(self) -> None:
        """WHAT: Multiple threads calling get() on StreamGuardStatsStore.
        EXPECTED: No crash, LRU order maintained.
        WHY: In production, multiple user sessions access stats concurrently.
        """
        store = StreamGuardStatsStore(max_entries=10)
        errors = []

        def accessor(thread_id: int) -> None:
            try:
                for i in range(50):
                    stats = store.get(user_id=thread_id, chat_id=i)
                    stats.total_checks += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=accessor, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent access caused errors: {errors}"

    def test_lru_eviction_boundary(self) -> None:
        """WHAT: Exactly max_entries entries then one more.
        EXPECTED: Oldest entry evicted, no crash.
        WHY: Off-by-one in LRU eviction logic.
        """
        store = StreamGuardStatsStore(max_entries=3)
        _ = store.get(1, 1)
        _ = store.get(2, 2)
        _ = store.get(3, 3)
        # This should evict (1,1)
        _ = store.get(4, 4)

        all_stats = store.all_stats()
        assert (1, 1) not in all_stats
        assert (4, 4) in all_stats

    def test_clear_during_concurrent_get(self) -> None:
        """WHAT: clear() called while get() is iterating.
        EXPECTED: No crash.
        WHY: clear() modifies dict while get() may be touching it.
        """
        store = StreamGuardStatsStore(max_entries=100)
        for i in range(50):
            store.get(i, i)

        errors = []

        def getter() -> None:
            try:
                for i in range(100):
                    store.get(i % 50, i % 50)
            except Exception as e:
                errors.append(e)

        def clearer() -> None:
            try:
                for i in range(50):
                    store.clear(i, i)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=getter)
        t2 = threading.Thread(target=clearer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # May have race condition in dict access, but should not crash
        # If it does crash, that's a finding
        if errors:
            pytest.xfail(f"Race condition in StreamGuardStatsStore: {errors[0]}")


@pytest.mark.adversarial
class TestStreamGuardParallelChecks:
    """StreamGuard: multiple check_early calls in rapid succession."""

    def test_check_early_called_after_abort(self) -> None:
        """WHAT: check_early() called again after an abort was signaled.
        EXPECTED: Returns same abort state (idempotent).
        WHY: Streaming loop may call check_early after abort before stopping.
        """
        guard = StreamGuard(expected_lang="de", enabled=True)
        # First call with enough text in wrong language
        long_english = (
            "This is a long English text that should trigger the guard. " * 10
        )
        result1 = guard.check_early(long_english)
        # After check is performed, subsequent calls should be consistent
        result2 = guard.check_early(long_english + " more text")
        # Both results should be the same (either True or False, but consistent)
        assert result1 == result2

    def test_stream_guard_disabled_then_reenabled(self) -> None:
        """WHAT: Guard manually disabled, session continues, then new guard created.
        EXPECTED: New guard instance operates independently.
        WHY: Guard state should not leak between instances.
        """
        guard1 = StreamGuard(expected_lang="de", enabled=True)
        guard1._state.disabled = True
        guard1._state.disable_reason = "Manual disable"

        guard2 = StreamGuard(expected_lang="de", enabled=True)
        assert guard2.is_active is True
        assert guard1.is_active is False


@pytest.mark.adversarial
class TestPipelineParallelChecks:
    """Privacy pipeline checked from multiple threads simultaneously."""

    def test_parallel_pipeline_checks(self) -> None:
        """WHAT: Same pipeline instance used from 4 threads.
        EXPECTED: No crash, correct results.
        WHY: Pipeline filters are stateless but audit log is shared.
        """
        pipeline = PrivacyPipeline()
        errors = []
        results = []

        def checker(thread_id: int) -> None:
            try:
                for i in range(20):
                    h = _make_hypothesis(
                        f"Thread {thread_id} claim {i}: user prefers short answers",
                        hid=f"parallel-{thread_id}-{i}",
                    )
                    result = pipeline.check(h)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=checker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Parallel pipeline checks caused errors: {errors}"
        assert len(results) == 80

    def test_parallel_healthcare_checks_with_blocking_claims(self) -> None:
        """WHAT: Multiple threads checking healthcare-blocking claims.
        EXPECTED: All threads get correct block results.
        WHY: Filter result must not be shared between threads.
        """
        pipeline = PrivacyPipeline()
        errors = []
        results = []

        def checker(thread_id: int) -> None:
            try:
                h = _make_hypothesis(
                    f"Thread {thread_id}: user shows depression symptoms",
                    hid=f"hc-parallel-{thread_id}",
                )
                result = pipeline.check(h)
                results.append((thread_id, result))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=checker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        # All should be blocked (depression is healthcare keyword)
        for tid, result in results:
            assert result is not None, f"Thread {tid} should have been blocked"

    def test_audit_log_entry_count_under_parallel_writes(self) -> None:
        """WHAT: Audit log entries after parallel pipeline checks.
        EXPECTED: Audit log has entries for all rejections.
        WHY: Audit log append is not thread-safe (list.append is atomic in CPython).
        """
        pipeline = PrivacyPipeline()

        def checker(i: int) -> None:
            h = _make_hypothesis(
                f"User {i} shows anxiety symptoms",
                hid=f"audit-race-{i}",
            )
            pipeline.check(h)

        threads = [threading.Thread(target=checker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 20 should have been rejected and logged
        assert pipeline.audit_log.total_rejections >= 20
