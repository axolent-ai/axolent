"""Integration test: full pipeline from message to DB and back.

Tests the complete flow:
  Bot message -> Event normalization -> DB insert -> Retrieve -> Similarity check

This test uses CryptoConnection in non-encrypted mode (test environment).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from application.skill_compression.event_normalizer import normalize_event
from application.skill_compression.fingerprint_matcher import (
    compute_similarity,
    find_matches,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from infrastructure.crypto_storage import CryptoConnection


@pytest.fixture
def crypto_conn(tmp_path):
    """Create a CryptoConnection in non-encrypted mode for testing."""
    db_path = tmp_path / "test_pipeline.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    return conn


@pytest.fixture
def storage(crypto_conn):
    """Create HypothesisStorage with schema on CryptoConnection."""
    s = HypothesisStorage(crypto_conn)
    s.init_schema()
    return s


class TestFullPipeline:
    """Integration test: message -> normalize -> store -> retrieve -> match."""

    def test_message_to_event_to_db(self, storage, crypto_conn):
        """A user message should be normalizable and storable."""
        # Step 1: Normalize
        event = normalize_event(
            "Schreib eine 30 Sekunden Ad Copy fuer Retargeting",
            user_id=42,
            scope={"project": "client_ads", "client": "honey-brand"},
        )

        assert event.intent != "general"  # Should detect ad-related intent
        assert event.fingerprint_hash  # Must have a hash

        # Step 2: Create a hypothesis from this event
        h = Hypothesis(
            hypothesis_id=f"hyp_{event.fingerprint_hash[:12]}",
            user_id=event.user_id,
            type="request",
            scope=HypothesisScope(
                project=event.scope.get("project", ""),
                client=event.scope.get("client", ""),
            ),
            claim=f"User requests {event.intent} in {event.domain}",
            status="candidate",
            version=1,
            created_at=event.timestamp,
            last_seen=event.timestamp,
            pattern_hash=event.fingerprint_hash,
            scope_hash=event.fingerprint_hash[:16],
        )
        storage.insert_hypothesis(h)

        # Step 3: Retrieve
        retrieved = storage.get_hypothesis(h.hypothesis_id)
        assert retrieved is not None
        assert retrieved.pattern_hash == event.fingerprint_hash

    def test_multiple_events_similarity_matching(self, storage):
        """Multiple similar events should be identified as candidates."""
        events = [
            normalize_event(
                "Erstelle eine Ad Copy fuer Retargeting Kampagne mit Hook",
                user_id=42,
                scope={"project": "ads"},
            ),
            normalize_event(
                "Schreib eine Retargeting Ad Copy mit CTA",
                user_id=42,
                scope={"project": "ads"},
            ),
            normalize_event(
                "Write a Python function to parse CSV files",
                user_id=42,
                scope={"project": "dev"},
            ),
        ]

        # Ad copy events should be similar to each other
        match_01 = compute_similarity(events[0], events[1])
        assert match_01.similarity_score > 0.5, (
            f"Ad copy events should be similar, got {match_01.similarity_score}"
        )

        # Ad copy vs code should be dissimilar
        match_02 = compute_similarity(events[0], events[2])
        assert match_02.similarity_score < match_01.similarity_score, (
            "Ad copy vs code should be less similar than ad copy vs ad copy"
        )

    def test_event_store_and_evidence_chain(self, storage):
        """Event -> Hypothesis -> Evidence should form a complete chain."""
        event = normalize_event("Review this Python code", user_id=42)
        ts = datetime.now(timezone.utc).isoformat()

        # Create hypothesis
        h = Hypothesis(
            hypothesis_id="hyp_chain_test",
            user_id=42,
            type="preference",
            claim="User prefers code review with root cause first",
            status="candidate",
            version=1,
            created_at=ts,
            last_seen=ts,
            pattern_hash=event.fingerprint_hash,
        )
        storage.insert_hypothesis(h)

        # Add evidence
        storage.insert_evidence(
            evidence_id="evi_chain_1",
            hypothesis_id="hyp_chain_test",
            hypothesis_version=1,
            signal_type="no_correction",
            signal_strength=0.8,
            created_at=ts,
        )
        storage.insert_evidence(
            evidence_id="evi_chain_2",
            hypothesis_id="hyp_chain_test",
            hypothesis_version=1,
            signal_type="explicit_confirm",
            signal_strength=1.0,
            created_at=ts,
        )

        # Retrieve and verify chain
        evidence = storage.get_evidence_for_hypothesis("hyp_chain_test")
        assert len(evidence) == 2
        assert {e["signal_type"] for e in evidence} == {
            "no_correction",
            "explicit_confirm",
        }

    def test_tombstone_blocks_pattern(self, storage):
        """A tombstoned fingerprint should block new hypothesis creation."""
        normalize_event("Never do this again", user_id=42)
        ts = datetime.now(timezone.utc).isoformat()

        # Create and tombstone a hypothesis
        h = Hypothesis(
            hypothesis_id="hyp_tombstone_test",
            user_id=42,
            type="negative",
            claim="Never use emojis",
            status="candidate",
            created_at=ts,
            last_seen=ts,
            pattern_hash="blocked_fp_hash",
        )
        storage.insert_hypothesis(h)

        storage.insert_tombstone(
            tombstone_id="tomb_test",
            hypothesis_id="hyp_tombstone_test",
            fingerprint="blocked_fp_hash",
            deleted_at=ts,
            expires_at="2099-12-31T23:59:59+00:00",
        )

        # Check should block re-learning
        assert storage.check_tombstone("blocked_fp_hash") is True

        # Unrelated fingerprint should pass
        assert storage.check_tombstone("different_fp") is False

    def test_crypto_connection_stores_data(self, crypto_conn, storage):
        """Data stored via CryptoConnection should persist within session."""
        ts = datetime.now(timezone.utc).isoformat()
        h = Hypothesis(
            hypothesis_id="hyp_persist_test",
            user_id=99,
            type="request",
            claim="Test persistence",
            created_at=ts,
            last_seen=ts,
        )
        storage.insert_hypothesis(h)

        # Retrieve via same connection
        result = storage.get_hypothesis("hyp_persist_test")
        assert result is not None
        assert result.user_id == 99

    def test_find_matches_across_events(self):
        """find_matches should rank similar events correctly."""
        target = normalize_event(
            "Create a 30 second retargeting ad copy",
            user_id=1,
        )
        candidates = [
            normalize_event("Write a retargeting ad copy with hook", user_id=1),
            normalize_event("Build a Python REST API with FastAPI", user_id=1),
            normalize_event("Create ad copy for awareness campaign", user_id=1),
        ]

        matches = find_matches(target, candidates, threshold=0.0)
        assert len(matches) > 0
        # First match should be more similar than last
        if len(matches) > 1:
            assert matches[0].similarity_score >= matches[-1].similarity_score
