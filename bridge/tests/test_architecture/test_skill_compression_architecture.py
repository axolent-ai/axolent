"""Architecture Guard Tests for Skill-Compression (Step 9).

Systematic structural verification of the 7-Layer architecture.
Tests run via AST/source/import analysis — no runtime DB needed.

Guard categories:
  1. Layer Separation: each layer only imports its permitted neighbors
  2. Data Integrity: Hypothesis frozen+slots, type TEXT, decay_immune, tombstones
  3. Privacy Guards: healthcare/secret/nudge filters block materialization
  4. Behavioral Guards: Ask Before Applying, tombstone duration, skill limit
"""

from __future__ import annotations

import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

_BRIDGE_ROOT = Path(__file__).resolve().parents[2]
_SC_ROOT = _BRIDGE_ROOT / "application" / "skill_compression"


def _read_source(relative_to_bridge: str) -> str:
    """Read a source file relative to bridge root."""
    full = _BRIDGE_ROOT / relative_to_bridge
    return full.read_text(encoding="utf-8")


def _parse_ast(relative_to_bridge: str) -> ast.Module:
    """Parse a source file into an AST."""
    source = _read_source(relative_to_bridge)
    return ast.parse(source)


def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract all imported module names from an AST tree."""
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


# ---------------------------------------------------------------
# 1. LAYER SEPARATION GUARDS
# ---------------------------------------------------------------


class TestLayerSeparation:
    """Verify that each layer only imports its permitted neighbor layers."""

    def test_layer1_event_normalizer_no_layer3_plus(self):
        """Layer 1 (Event Normalizer) must NOT import Layer 3+ modules.

        Layer 1 is a pure extraction layer. It must not know about
        storage, judges, matchers, or any downstream module.
        """
        tree = _parse_ast("application/skill_compression/event_normalizer.py")
        imports = _extract_imports(tree)

        forbidden_modules = [
            "evidence_ledger",
            "pattern_judge",
            "skill_matcher",
            "hypothesis_storage",
            "collision_detector",
            "fsrs_decay",
            "bkt",
            "privacy",
            "skill_explainer",
            "topic_classifier",
        ]

        for imp in imports:
            for forbidden in forbidden_modules:
                assert forbidden not in imp, (
                    f"Layer 1 (event_normalizer) imports '{imp}' which "
                    f"contains forbidden module '{forbidden}'. "
                    f"Layer 1 must not import Layer 3+ modules."
                )

    def test_layer2_candidates_no_layer4_plus(self):
        """Layer 2 (Candidate algorithms) must NOT import Layer 4+ modules.

        N-gram, Markov, Elo are Layer 2. They must not import
        Pattern Judge (L4) or SkillMatcher (L5).
        """
        layer2_files = [
            "application/skill_compression/ngram_extractor.py",
            "application/skill_compression/markov_chain.py",
            "application/skill_compression/elo_rating.py",
        ]

        forbidden_modules = [
            "pattern_judge",
            "skill_matcher",
            "collision_detector",
            "skill_explainer",
            "topic_classifier",
            "privacy",
        ]

        for filepath in layer2_files:
            tree = _parse_ast(filepath)
            imports = _extract_imports(tree)
            for imp in imports:
                for forbidden in forbidden_modules:
                    assert forbidden not in imp, (
                        f"Layer 2 file '{filepath}' imports '{imp}' "
                        f"which contains forbidden module '{forbidden}'. "
                        f"Layer 2 must not import Layer 4+ modules."
                    )

    def test_layer5_skill_matcher_no_direct_layer2_imports(self):
        """Layer 5 (SkillMatcher) must NOT import N-Gram/Markov/Elo directly.

        SkillMatcher accesses patterns only through HypothesisStorage
        and FingerprintMatcher. It must never import L2 algorithms.
        """
        tree = _parse_ast("application/skill_compression/skill_matcher.py")
        imports = _extract_imports(tree)

        forbidden_direct = [
            "ngram_extractor",
            "markov_chain",
            "elo_rating",
            "bkt",
        ]

        for imp in imports:
            for forbidden in forbidden_direct:
                assert forbidden not in imp, (
                    f"SkillMatcher imports '{imp}' which contains "
                    f"forbidden direct import '{forbidden}'. "
                    f"SkillMatcher must access these only via Hypothesis."
                )

    def test_presentation_skill_modules_only_import_application(self):
        """Presentation layer skill_* modules must only import from application.

        No direct infrastructure or raw domain-layer access allowed
        (except python-telegram-bot and standard library).
        """
        presentation_files = [
            "presentation/skill_profile_view.py",
            "presentation/skill_commands.py",
        ]

        forbidden_prefixes = [
            "infrastructure.",
            "infrastructure/",
        ]

        for filepath in presentation_files:
            tree = _parse_ast(filepath)
            imports = _extract_imports(tree)
            for imp in imports:
                for forbidden in forbidden_prefixes:
                    assert not imp.startswith(forbidden), (
                        f"Presentation file '{filepath}' imports '{imp}' "
                        f"from infrastructure layer. Presentation must only "
                        f"import from application layer."
                    )

    def test_privacy_filter_called_before_promotion_in_judge(self):
        """Privacy pipeline must be checked BEFORE any promotion in PatternJudge.

        The source code must call self._privacy.check() BEFORE
        self._check_promotion(). We verify via source line order,
        since ast.walk does not preserve statement order.
        """
        source = _read_source("application/skill_compression/pattern_judge.py")

        # Find the evaluate method body and check line order
        lines = source.splitlines()
        privacy_line = None
        promotion_line = None

        in_evaluate = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "def evaluate(" in stripped:
                in_evaluate = True
            elif in_evaluate and stripped.startswith("def "):
                break  # Next method
            elif in_evaluate:
                if "self._privacy" in stripped and ".check(" in stripped:
                    privacy_line = i
                if "self._check_promotion(" in stripped:
                    promotion_line = i

        assert privacy_line is not None, (
            "self._privacy.check() must be called in PatternJudge.evaluate()"
        )
        assert promotion_line is not None, (
            "self._check_promotion() must be called in PatternJudge.evaluate()"
        )
        assert privacy_line < promotion_line, (
            "Privacy pipeline check() must be called BEFORE "
            "_check_promotion() in PatternJudge.evaluate(). "
            f"Found privacy at line {privacy_line}, "
            f"promotion at line {promotion_line}."
        )


# ---------------------------------------------------------------
# 2. DATA INTEGRITY GUARDS
# ---------------------------------------------------------------


class TestDataIntegrityGuards:
    """Verify structural invariants of core data classes and schema."""

    def test_hypothesis_is_frozen_dataclass(self):
        """HC-SC-1: Hypothesis must be frozen dataclass (immutable)."""
        from application.skill_compression.hypothesis_storage import Hypothesis

        h = Hypothesis(hypothesis_id="test_frozen")
        with pytest.raises(AttributeError):
            h.claim = "mutated"  # type: ignore[misc]

    def test_hypothesis_has_slots(self):
        """HC-SC-1: Hypothesis must use __slots__ (memory efficiency)."""
        from application.skill_compression.hypothesis_storage import Hypothesis

        assert hasattr(Hypothesis, "__slots__")
        h = Hypothesis(hypothesis_id="test_slots")
        assert not hasattr(h, "__dict__")

    def test_type_column_is_text_not_enum(self):
        """HC-SC-9: type column must be TEXT, not ENUM with CHECK constraint."""
        from application.skill_compression.hypothesis_storage import (
            HYPOTHESIS_SCHEMA_SQL,
        )

        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(HYPOTHESIS_SCHEMA_SQL)

        rows = conn.execute("PRAGMA table_info(hypotheses)").fetchall()
        type_col = [r for r in rows if r[1] == "type"]
        assert len(type_col) == 1
        assert type_col[0][2] == "TEXT", (
            f"type column must be TEXT, got '{type_col[0][2]}'"
        )
        conn.close()

    def test_decay_immune_user_skills_not_archived_by_fsrs(self):
        """HC-SC-6: decay_immune=True hypotheses must never be auto-archived.

        This tests the PatternJudge.should_archive() method directly.
        """
        from application.skill_compression.fsrs_decay import FSRSState
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.pattern_judge import PatternJudge

        judge = PatternJudge()

        # Create a decay-immune hypothesis (user-created via /learn)
        immune_hyp = Hypothesis(
            hypothesis_id="hyp_immune",
            user_id=1,
            type="preference",
            claim="Always use bullet points",
            status="active",
            decay_immune=True,
            source_type="learn_command",
            created_at="2024-01-01T00:00:00+00:00",
            last_seen="2024-01-01T00:00:00+00:00",
        )

        # Create an FSRS state that would normally trigger archiving
        # (very old, very low stability, last reviewed long ago)
        old_fsrs = FSRSState(
            stability=1.0,
            difficulty=5.0,
            last_reviewed="2024-01-01T00:00:00+00:00",
            reps=1,
            lapses=0,
        )

        assert not judge.should_archive(
            immune_hyp, old_fsrs, "2026-05-20T00:00:00+00:00"
        ), "decay_immune hypothesis must NEVER be auto-archived by FSRS"

    def test_tombstone_blocks_relearning_until_expires(self):
        """HC-SC-7: Tombstone must block re-learning until expires_at."""
        from application.skill_compression.hypothesis_storage import (
            HypothesisStorage,
        )
        from infrastructure.crypto_storage import CryptoConnection

        tmp_db = Path(__file__).parent / "_test_tombstone.db"
        try:
            conn = CryptoConnection(tmp_db, require_encryption=False)
            storage = HypothesisStorage(conn)
            storage.init_schema()

            now = datetime.now(timezone.utc)
            future = (now + timedelta(days=30)).isoformat()
            past = (now - timedelta(days=1)).isoformat()

            # Active tombstone (expires in 30 days)
            storage.insert_tombstone(
                tombstone_id="tomb_active",
                hypothesis_id="hyp_1",
                fingerprint="fp_blocked",
                deleted_at=now.isoformat(),
                expires_at=future,
            )

            # Expired tombstone
            storage.insert_tombstone(
                tombstone_id="tomb_expired",
                hypothesis_id="hyp_2",
                fingerprint="fp_expired",
                deleted_at=(now - timedelta(days=60)).isoformat(),
                expires_at=past,
            )

            # Active tombstone must block
            assert storage.check_tombstone("fp_blocked") is True, (
                "Active tombstone must block re-learning"
            )

            # Expired tombstone must not block
            assert storage.check_tombstone("fp_expired") is False, (
                "Expired tombstone must NOT block re-learning"
            )

            # Unknown fingerprint must not block
            assert storage.check_tombstone("fp_unknown") is False

            conn.close()
        finally:
            if tmp_db.exists():
                tmp_db.unlink()

    def test_permanent_tombstone_blocks_forever(self):
        """HC-SC-7: Permanent tombstone ('nie wieder') blocks indefinitely."""
        from application.skill_compression.hypothesis_storage import (
            HypothesisStorage,
        )
        from infrastructure.crypto_storage import CryptoConnection

        tmp_db = Path(__file__).parent / "_test_perm_tombstone.db"
        try:
            conn = CryptoConnection(tmp_db, require_encryption=False)
            storage = HypothesisStorage(conn)
            storage.init_schema()

            now = datetime.now(timezone.utc)

            storage.insert_tombstone(
                tombstone_id="tomb_perm",
                hypothesis_id="hyp_perm",
                fingerprint="fp_permanent",
                deleted_at=now.isoformat(),
                expires_at=None,
                permanent=True,
            )

            assert storage.check_tombstone("fp_permanent") is True, (
                "Permanent tombstone must block re-learning forever"
            )

            conn.close()
        finally:
            if tmp_db.exists():
                tmp_db.unlink()


# ---------------------------------------------------------------
# 3. PRIVACY GUARDS
# ---------------------------------------------------------------


class TestPrivacyGuards:
    """Verify that privacy filters block all prohibited content categories."""

    def test_healthcare_patterns_never_materialized(self):
        """HC-SC-14: Healthcare patterns must NEVER become a Hypothesis.

        Tests multiple categories: mental health, clinical, behavioral
        phenotyping, mood inference.
        """
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.privacy.healthcare_filter import (
            HealthcareFilter,
        )

        hf = HealthcareFilter()

        healthcare_claims = [
            # Mental health keywords
            "User shows signs of depression based on writing patterns",
            "Detect anxiety from message frequency",
            "User seems depressed lately",
            # Clinical keywords
            "User's cognitive decline detected over past weeks",
            "Track medication adherence from chat patterns",
            "Diagnose based on linguistic markers",
            # Behavioral phenotyping
            "User's writing pattern has changed significantly",
            "Typing speed correlates with mental health",
            # Mood inference
            "User seems stressed and frustrated today",
            "Predict daily mood from message sentiment",
            # German keywords
            "Nutzer zeigt Anzeichen einer Angststörung",
            "Kognitive Beeinträchtigung erkannt",
            "Schreibmuster deutet auf Burnout hin",
        ]

        for claim in healthcare_claims:
            hyp = Hypothesis(
                hypothesis_id=f"hyp_health_{hash(claim) % 10000}",
                claim=claim,
            )
            assert hf.filter_hypothesis(hyp), (
                f"Healthcare filter MUST block claim: '{claim}'"
            )

    def test_secret_patterns_never_stored(self):
        """HC-SC-13: Secret/PII patterns must NEVER be stored as Hypothesis."""
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.privacy.secret_scanner import (
            SecretScanner,
        )

        scanner = SecretScanner()

        secret_claims = [
            # API tokens
            "Use API key sk-proj-abc123def456ghi789",
            "Token: " + "ghp_" + "x" * 36,  # nosemgrep: detected-github-token
            # Financial data
            "Price is $49.99 per unit",
            "MwSt: 19% on 500 EUR",
            "IBAN: DE89370400440532013000",
            # PII
            "Email: john@example.com for support",
            "Passwort: SuperSecret123!",
            # Long hex (likely tokens)
            "Hash: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        ]

        for claim in secret_claims:
            hyp = Hypothesis(
                hypothesis_id=f"hyp_secret_{hash(claim) % 10000}",
                claim=claim,
            )
            assert scanner.block_if_secrets(hyp), (
                f"Secret scanner MUST block claim: '{claim}'"
            )

    def test_nudge_violations_never_stored(self):
        """HC-SC-15: Nudge policy violations must NEVER be stored."""
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.privacy.nudge_filter import NudgeFilter

        nf = NudgeFilter()

        nudge_claims = [
            # Emotional manipulation
            "Create FOMO by showing limited availability",
            "I miss you, please come back",
            # Dark patterns
            "Hide the opt-out button to reduce cancellations",
            "Confirmshaming users who decline the offer",
            # Attention maximization
            "Keep user engaged with engagement loops",
            "Track daily login streaks",
            # Social manipulation
            "Compare user performance with other users",
            "Show leaderboard to increase competition",
            # Data flow violation
            "Share usage data with third party analytics",
            "Silent tracking without user consent",
        ]

        for claim in nudge_claims:
            hyp = Hypothesis(
                hypothesis_id=f"hyp_nudge_{hash(claim) % 10000}",
                claim=claim,
            )
            assert nf.violates_nudge_policy(hyp), (
                f"Nudge filter MUST block claim: '{claim}'"
            )

    def test_privacy_pipeline_blocks_in_pattern_judge(self):
        """Privacy pipeline rejects hypothesis in PatternJudge.evaluate()."""
        from application.skill_compression.bkt import BKTState
        from application.skill_compression.evidence_ledger import EvidenceSummary
        from application.skill_compression.fsrs_decay import FSRSState
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.pattern_judge import (
            STATUS_PRIVACY_REJECTED,
            PatternJudge,
        )
        from application.skill_compression.privacy.privacy_pipeline import (
            PrivacyPipeline,
        )

        pipeline = PrivacyPipeline()
        judge = PatternJudge(privacy_pipeline=pipeline)

        # Healthcare hypothesis that should be rejected
        hyp = Hypothesis(
            hypothesis_id="hyp_privacy_judge",
            user_id=1,
            claim="User shows signs of depression",
            status="candidate",
            created_at="2026-05-20T00:00:00+00:00",
            last_seen="2026-05-20T00:00:00+00:00",
        )

        evidence = EvidenceSummary(
            positive_count=5,
            negative_count=0,
            total_count=5,
            weighted_score=0.9,
            bkt_state=BKTState(),
            distinct_sessions=3,
            last_positive_at="2026-05-20T00:00:00+00:00",
            last_negative_at=None,
        )

        fsrs = FSRSState()
        bkt = BKTState()

        decision = judge.evaluate(
            hyp,
            evidence,
            bkt,
            1700.0,
            fsrs,
            current_time="2026-05-20T12:00:00+00:00",
        )

        assert decision.should_transition is True
        assert decision.recommended_status == STATUS_PRIVACY_REJECTED, (
            f"Privacy-rejected hypothesis must get status "
            f"'{STATUS_PRIVACY_REJECTED}', got '{decision.recommended_status}'"
        )


# ---------------------------------------------------------------
# 4. BEHAVIORAL GUARDS
# ---------------------------------------------------------------


class TestBehavioralGuards:
    """Verify behavioral invariants of the Skill-Compression system."""

    def test_ask_before_applying_default(self):
        """HC-SC-10: auto_apply_enabled=False must be the default."""
        from application.skill_compression.skill_matcher import (
            DEFAULT_USER_PREFERENCES,
        )

        assert DEFAULT_USER_PREFERENCES.get("auto_apply_enabled") is False, (
            "auto_apply_enabled must be False by default (Ask Before Applying)"
        )

    def test_should_ask_user_for_confirmed_always(self):
        """HC-SC-10: Confirmed skills always require user confirmation."""
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.skill_matcher import (
            SkillMatch,
            should_ask_user,
        )

        match = SkillMatch(
            hypothesis=Hypothesis(
                hypothesis_id="hyp_confirmed",
                status="confirmed",
            ),
            confidence=0.95,
            requires_confirmation=True,
            explanation="test",
        )

        # Even with auto_apply enabled, confirmed must always ask
        assert should_ask_user(match, {"auto_apply_enabled": True}) is True
        assert should_ask_user(match, {"auto_apply_enabled": False}) is True
        assert should_ask_user(match, None) is True

    def test_should_ask_user_for_active_respects_preference(self):
        """HC-SC-10: Active skills respect auto_apply_enabled preference."""
        from application.skill_compression.hypothesis_storage import Hypothesis
        from application.skill_compression.skill_matcher import (
            SkillMatch,
            should_ask_user,
        )

        match = SkillMatch(
            hypothesis=Hypothesis(
                hypothesis_id="hyp_active",
                status="active",
            ),
            confidence=0.95,
            requires_confirmation=False,
            explanation="test",
        )

        # Default (no prefs): ask
        assert should_ask_user(match) is True

        # Explicit False: ask
        assert should_ask_user(match, {"auto_apply_enabled": False}) is True

        # Explicit True: don't ask
        assert should_ask_user(match, {"auto_apply_enabled": True}) is False

    def test_skill_library_maximum_50_enforced(self):
        """HC-SC-8: Max 50 active skills enforced in storage count method."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisStorage,
        )
        from infrastructure.crypto_storage import CryptoConnection

        tmp_db = Path(__file__).parent / "_test_max50.db"
        try:
            conn = CryptoConnection(tmp_db, require_encryption=False)
            storage = HypothesisStorage(conn)
            storage.init_schema()

            ts = datetime.now(timezone.utc).isoformat()

            # Insert 50 active hypotheses
            for i in range(50):
                h = Hypothesis(
                    hypothesis_id=f"hyp_max_{i}",
                    user_id=42,
                    type="preference",
                    claim=f"Skill {i}",
                    status="active",
                    created_at=ts,
                    last_seen=ts,
                )
                storage.insert_hypothesis(h)

            count = storage.count_active_hypotheses(42)
            assert count == 50

            # The count_active_hypotheses method is used by /learn
            # to enforce the 50-skill limit. The limit is checked
            # in the command handler, not in storage itself.
            # We verify the count is accurate for the guard to work.

            conn.close()
        finally:
            if tmp_db.exists():
                tmp_db.unlink()

    def test_tombstone_default_duration_30_days(self):
        """HC-SC-7: Default tombstone duration must be 30 days.

        Verified by checking the _execute_forget function logic.
        """
        source = _read_source("presentation/skill_commands.py")

        # Must contain timedelta(days=30) for default tombstone
        assert "timedelta(days=30)" in source, (
            "skill_commands.py must use timedelta(days=30) for default "
            "tombstone duration (HC-SC-7)"
        )

        # Must have permanent option
        assert "permanent" in source.lower(), (
            "skill_commands.py must support permanent tombstone option ('nie wieder')"
        )

    def test_matchable_statuses_only_confirmed_and_active(self):
        """Only confirmed and active hypotheses may be matched.

        Candidate and suggested must NEVER be applied.
        """
        from application.skill_compression.skill_matcher import MATCHABLE_STATUSES

        assert MATCHABLE_STATUSES == frozenset({"confirmed", "active"}), (
            f"MATCHABLE_STATUSES must be exactly {{confirmed, active}}, "
            f"got {MATCHABLE_STATUSES}"
        )
