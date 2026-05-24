"""Performance budget tests for AXOLENT Bridge.

Each test verifies that a critical operation completes within its
allocated performance budget (defined in budgets.yaml).

Design:
    - Warmup run before measurement (avoids cold-start JIT/import cost)
    - 100 iterations averaged for statistical robustness
    - Marked @pytest.mark.performance (not in default test run)
    - Budget violations cause immediate test failure with timing details

Run locally:
    pytest tests/test_performance/ -v -m performance

These tests are sensitive to system load. They are NOT run in pr-check
(shared CI is unreliable for perf). See .github/workflows/performance.yml.
"""

from __future__ import annotations

import asyncio
import copy

import pytest


# ===================================================================
# LCP (Language Control Plane)
# ===================================================================


@pytest.mark.performance
class TestLanguageDetection:
    """Performance budgets for language detection subsystem."""

    def test_language_detection_short_text_under_budget(self, perf_timer):
        """detect_language() for short text must complete within budget per call."""
        from domain.language import detect_language

        # Warmup (import + first call cost)
        detect_language("hello world")
        detect_language("Hallo Welt")

        # Measure: 100 calls, average must be under budget
        with perf_timer("language_detection_short_text"):
            for _ in range(100):
                detect_language("hello world")

    def test_language_detection_german_short(self, perf_timer):
        """detect_language() for German short text within budget."""
        from domain.language import detect_language

        # Warmup
        detect_language("Guten Morgen")

        with perf_timer("language_detection_short_text"):
            for _ in range(100):
                detect_language("Guten Morgen")

    def test_language_resolver_resolve_readonly_under_budget(self, perf_timer):
        """LanguageResolver.resolve_readonly() within budget per call."""
        from application.language.resolver import LanguageResolver
        from infrastructure.conversation_storage import _reset_all_for_tests

        _reset_all_for_tests()
        resolver = LanguageResolver()

        loop = asyncio.new_event_loop()

        # Warmup
        loop.run_until_complete(
            resolver.resolve_readonly(user_id=1, chat_id=1, text="hello there")
        )

        with perf_timer("language_resolver_resolve_readonly"):
            for _ in range(100):
                loop.run_until_complete(
                    resolver.resolve_readonly(user_id=1, chat_id=1, text="hello there")
                )

        loop.close()

    def test_language_resolver_resolve_under_budget(self, perf_timer):
        """LanguageResolver.resolve() with sticky-update within budget per call."""
        from application.language.resolver import LanguageResolver
        from infrastructure.conversation_storage import _reset_all_for_tests

        _reset_all_for_tests()
        resolver = LanguageResolver()

        loop = asyncio.new_event_loop()

        # Warmup
        loop.run_until_complete(
            resolver.resolve(user_id=99, chat_id=99, text="testing speed")
        )

        with perf_timer("language_resolver_resolve"):
            for _ in range(100):
                loop.run_until_complete(
                    resolver.resolve(user_id=99, chat_id=99, text="testing speed")
                )

        loop.close()


# ===================================================================
# Skill Compression Privacy Pipeline
# ===================================================================


@pytest.mark.performance
class TestPrivacyPipeline:
    """Performance budgets for the privacy pipeline filters."""

    def _make_clean_hypothesis(self):
        """Create a hypothesis that passes all filters (clean)."""
        from application.skill_compression.hypothesis_storage import Hypothesis

        return Hypothesis(
            hypothesis_id="perf-test-001",
            user_id=12345,
            type="preference",
            claim="User prefers bullet points in responses",
            status="candidate",
        )

    def test_privacy_pipeline_check_clean_under_budget(self, perf_timer):
        """Full privacy pipeline (3 filters) on clean hypothesis within budget."""
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )

        pipeline = PrivacyPipeline()
        hypothesis = self._make_clean_hypothesis()

        # Warmup
        pipeline.check(hypothesis)

        with perf_timer("privacy_pipeline_check"):
            for _ in range(100):
                pipeline.check(hypothesis)

    def test_healthcare_filter_under_budget(self, perf_timer):
        """HealthcareFilter.filter_hypothesis() within budget per call."""
        from application.skill_compression.privacy.healthcare_filter import (
            HealthcareFilter,
        )

        hf = HealthcareFilter()
        hypothesis = self._make_clean_hypothesis()

        # Warmup
        hf.filter_hypothesis(hypothesis)

        with perf_timer("healthcare_filter_check"):
            for _ in range(100):
                hf.filter_hypothesis(hypothesis)

    def test_secret_scanner_under_budget(self, perf_timer):
        """SecretScanner.block_if_secrets() within budget per call."""
        from application.skill_compression.privacy.secret_scanner import (
            SecretScanner,
        )

        scanner = SecretScanner()
        hypothesis = self._make_clean_hypothesis()

        # Warmup
        scanner.block_if_secrets(hypothesis)

        with perf_timer("secret_scanner_check"):
            for _ in range(100):
                scanner.block_if_secrets(hypothesis)

    def test_nudge_filter_under_budget(self, perf_timer):
        """NudgeFilter.violates_nudge_policy() within budget per call."""
        from application.skill_compression.privacy.nudge_filter import (
            NudgeFilter,
        )

        nf = NudgeFilter()
        hypothesis = self._make_clean_hypothesis()

        # Warmup
        nf.violates_nudge_policy(hypothesis)

        with perf_timer("nudge_filter_check"):
            for _ in range(100):
                nf.violates_nudge_policy(hypothesis)


# ===================================================================
# Sentry Filter
# ===================================================================


@pytest.mark.performance
class TestSentryFilter:
    """Performance budget for the Sentry before_send filter."""

    def test_sentry_before_send_under_budget(self, perf_timer):
        """_sentry_before_send() must complete in under 1ms per event."""
        from main import _sentry_before_send

        event = {
            "extra": {
                "user_id": 123,
                "request_id": "abc123",
                "model_id": "claude-4",
                "message_text": "this should be stripped",
            },
            "request": {
                "url": "https://api.telegram.org/bot123:ABCdef/sendMessage",
                "data": '{"chat_id": 123, "text": "hi"}',
            },
            "exception": {
                "values": [
                    {
                        "type": "ValueError",
                        "value": "some user text leaked here",
                        "stacktrace": {
                            "frames": [{"filename": "main.py", "vars": {"x": "secret"}}]
                        },
                    }
                ]
            },
            "breadcrumbs": {
                "values": [
                    {"data": {"url": "https://api.telegram.org/bot999:XYZ/getUpdates"}}
                ]
            },
        }

        # Warmup
        for _ in range(10):
            _sentry_before_send(copy.deepcopy(event), {})

        # Measure: 100 calls, average must be under budget
        with perf_timer("sentry_before_send"):
            for _ in range(100):
                _sentry_before_send(copy.deepcopy(event), {})


# ===================================================================
# Rate Limiter
# ===================================================================


@pytest.mark.performance
class TestRateLimiter:
    """Performance budget for rate limiter check."""

    def test_rate_limiter_check_under_budget(self, perf_timer):
        """RateLimiter.check_and_consume() within budget per call."""
        from application.rate_limiter import RateLimiter

        limiter = RateLimiter()

        # Warmup
        limiter.check_and_consume(user_id=1)

        with perf_timer("rate_limiter_check"):
            for _ in range(100):
                limiter.check_and_consume(user_id=1)


# ===================================================================
# Stream Guard
# ===================================================================


@pytest.mark.performance
class TestStreamGuard:
    """Performance budget for StreamGuard classification."""

    def test_stream_guard_classify_under_budget(self, perf_timer):
        """StreamGuard.classify_and_report_abort() within budget per call."""
        from application.language.stream_guard import (
            StreamGuard,
            StreamGuardStats,
        )

        english_text = "This is completely in English. " * 20

        # Warmup: create guard, trigger abort, classify
        g = StreamGuard(expected_lang="de", enabled=True)
        g.check_early(english_text)
        g.classify_and_report_abort(english_text, StreamGuardStats())

        # Measure: the expensive part is langdetect on 620-char text.
        # Each iteration creates a fresh guard because classify is one-shot.
        with perf_timer("stream_guard_classify"):
            for _ in range(100):
                g = StreamGuard(expected_lang="de", enabled=True)
                g.check_early(english_text)
                if g.state.aborted:
                    g.classify_and_report_abort(english_text, StreamGuardStats())


# ===================================================================
# Storage
# ===================================================================


@pytest.mark.performance
class TestStorage:
    """Performance budgets for storage operations."""

    def test_memory_storage_retrieve_under_budget(self, perf_timer, tmp_path):
        """Memory retrieval for one user (up to 100 entries) within budget."""
        from infrastructure.memory_storage import MemoryStorage

        storage = MemoryStorage(tmp_path)

        # Seed 100 entries for user 42
        for i in range(100):
            storage.append(
                {
                    "id": f"entry-{i}",
                    "user_id": 42,
                    "timestamp": f"2026-05-24T{i:02d}:00:00Z",
                    "content": f"Memory entry number {i} with some text padding",
                    "layer": "episodic",
                },
                layer="episodic",
            )

        # Warmup
        storage.list_entries(user_id=42, layer="episodic", limit=100)

        with perf_timer("memory_storage_retrieve_user"):
            for _ in range(100):
                storage.list_entries(user_id=42, layer="episodic", limit=100)

    def test_sqlite_simple_query_under_budget(self, perf_timer):
        """Simple SELECT on indexed column within budget per call."""
        from infrastructure.conversation_storage import (
            get_language,
            set_language,
            _reset_all_for_tests,
        )

        _reset_all_for_tests()

        loop = asyncio.new_event_loop()

        # Seed a language entry
        loop.run_until_complete(set_language(user_id=100, chat_id=100, lang="en"))

        # Warmup
        loop.run_until_complete(get_language(user_id=100, chat_id=100))

        with perf_timer("sqlite_connection_query_simple"):
            for _ in range(100):
                loop.run_until_complete(get_language(user_id=100, chat_id=100))

        loop.close()
