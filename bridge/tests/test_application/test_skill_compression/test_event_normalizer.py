"""Tests for the Event Normalizer (Layer 1, Step 1.2/10).

Covers:
  - Intent extraction from various message types
  - Domain classification
  - Format type classification
  - Constraint extraction (duration, length, funnel, audience)
  - Correction keyword detection
  - Re-formulation detection
  - Fingerprint hash determinism
  - Graceful handling of unclear/empty input
"""

from __future__ import annotations

import pytest

from application.skill_compression.event_normalizer import (
    compute_fingerprint,
    normalize_event,
)


class TestIntentExtraction:
    """Tests for intent classification from user messages."""

    def test_code_intent(self):
        """Code-related keywords should produce a code intent."""
        event = normalize_event("Write a Python function to parse JSON", user_id=1)
        assert event.intent == "create_code"

    def test_ad_copy_intent(self):
        """Ad copy keywords should produce an ad copy intent."""
        event = normalize_event(
            "Schreib eine Ad Copy mit Hook und CTA fuer Retargeting",
            user_id=1,
        )
        assert event.intent == "create_ad_copy"

    def test_video_concept_intent(self):
        """Video/reel keywords should produce a video concept intent."""
        event = normalize_event(
            "Erstell ein Drehkonzept fuer ein 30s TikTok Reel",
            user_id=1,
        )
        assert event.intent == "create_video_concept"

    def test_analysis_intent(self):
        """Analysis keywords should produce an analyze intent."""
        event = normalize_event(
            "Analysiere die Performance der letzten Kampagne",
            user_id=1,
        )
        assert event.intent == "analyze"

    def test_explain_intent(self):
        """Explanation keywords should produce an explain intent."""
        event = normalize_event("Erklaere mir wie Transformer funktionieren", user_id=1)
        assert event.intent == "explain"

    def test_summarize_intent(self):
        """Summary keywords should produce a summarize intent."""
        event = normalize_event("Fasse den Artikel zusammen", user_id=1)
        assert event.intent == "summarize"

    def test_translate_intent(self):
        """Translation keywords should produce a translate intent."""
        event = normalize_event("Uebersetze den Text ins Englische", user_id=1)
        assert event.intent == "translate"

    def test_unclear_intent_defaults_to_general(self):
        """Unclear or ambiguous input should default to 'general'."""
        event = normalize_event("Hallo, wie geht es dir?", user_id=1)
        assert event.intent == "general"

    def test_empty_text_defaults_to_general(self):
        """Empty text should produce a general intent."""
        event = normalize_event("", user_id=1)
        assert event.intent == "general"


class TestDomainClassification:
    """Tests for domain classification."""

    def test_marketing_domain(self):
        """Marketing keywords should be classified correctly."""
        event = normalize_event(
            "Erstelle eine Retargeting Campaign mit gutem ROAS",
            user_id=1,
        )
        assert event.domain == "marketing"

    def test_development_domain(self):
        """Development keywords should be classified correctly."""
        event = normalize_event(
            "Implement a Python API endpoint with database connection",
            user_id=1,
        )
        assert event.domain == "development"

    def test_finance_domain(self):
        """Finance keywords should be classified correctly."""
        event = normalize_event(
            "Berechne die Steuer fuer die Quartalsrechnung",
            user_id=1,
        )
        assert event.domain == "finance"


class TestConstraintExtraction:
    """Tests for constraint extraction from user messages."""

    def test_duration_seconds(self):
        """Duration in seconds should be extracted."""
        event = normalize_event("Erstelle ein 30 Sekunden Video", user_id=1)
        assert event.constraints.get("duration") == "30s"

    def test_duration_minutes(self):
        """Duration in minutes should be extracted."""
        event = normalize_event("Create a 5 minute tutorial", user_id=1)
        assert event.constraints.get("duration") == "5min"

    def test_word_length(self):
        """Word count constraints should be extracted."""
        event = normalize_event("Write a 500 word article", user_id=1)
        assert event.constraints.get("length") == "500 words"

    def test_funnel_stage(self):
        """Funnel stage should be extracted."""
        event = normalize_event("Create TOFU content for awareness", user_id=1)
        assert event.constraints.get("funnel") == "awareness"

    def test_retargeting_funnel(self):
        """Retargeting should be classified correctly."""
        event = normalize_event(
            "Ad Copy fuer Retargeting Kampagne",
            user_id=1,
        )
        assert event.constraints.get("funnel") == "retargeting"

    def test_no_constraints(self):
        """Messages without constraints should have empty dict."""
        event = normalize_event("Hello world", user_id=1)
        assert event.constraints == {}


class TestCorrectionDetection:
    """Tests for correction keyword detection."""

    def test_correction_nein(self):
        """German 'nein' should be detected as correction."""
        event = normalize_event("Nein, das ist falsch", user_id=1)
        assert event.correction_keywords_present is True

    def test_correction_anders(self):
        """'anders' should be detected as correction."""
        event = normalize_event("Mach das bitte anders", user_id=1)
        assert event.correction_keywords_present is True

    def test_no_correction(self):
        """Normal message should not be flagged as correction."""
        event = normalize_event("Write a Python script", user_id=1)
        assert event.correction_keywords_present is False


class TestReformulationDetection:
    """Tests for re-formulation detection."""

    def test_reformulation_detected(self):
        """Similar messages should be detected as re-formulations."""
        event = normalize_event(
            "Schreib eine Ad Copy fuer Retargeting",
            user_id=1,
            previous_text="Erstelle eine Ad Copy fuer das Retargeting",
        )
        assert event.is_re_formulation is True

    def test_no_reformulation_different_topic(self):
        """Completely different messages should not be re-formulations."""
        event = normalize_event(
            "Write a Python function to parse JSON",
            user_id=1,
            previous_text="Wie ist das Wetter heute?",
        )
        assert event.is_re_formulation is False

    def test_no_previous_message(self):
        """Without previous text, no re-formulation should be detected."""
        event = normalize_event(
            "Write a Python function",
            user_id=1,
            previous_text=None,
        )
        assert event.is_re_formulation is False


class TestFingerprintDeterminism:
    """Tests for fingerprint hash determinism."""

    def test_same_inputs_same_hash(self):
        """Identical structured fields must produce the same hash."""
        hash1 = compute_fingerprint(
            intent="create_code",
            domain="development",
            format_type="code",
            constraints={"length": "100 lines"},
            scope={"project": "axolent"},
            language="en",
        )
        hash2 = compute_fingerprint(
            intent="create_code",
            domain="development",
            format_type="code",
            constraints={"length": "100 lines"},
            scope={"project": "axolent"},
            language="en",
        )
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        """Different structured fields must produce different hashes."""
        hash1 = compute_fingerprint(
            intent="create_code",
            domain="development",
            format_type="code",
            constraints={},
            scope={},
            language="en",
        )
        hash2 = compute_fingerprint(
            intent="analyze",
            domain="marketing",
            format_type="report",
            constraints={},
            scope={},
            language="en",
        )
        assert hash1 != hash2

    def test_hash_is_hex_sha256(self):
        """Hash should be a valid 64-character hex string (SHA-256)."""
        h = compute_fingerprint(
            intent="test",
            domain="test",
            format_type="test",
            constraints={},
            scope={},
            language="en",
        )
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestNormalizedEventStructure:
    """Tests for the NormalizedEvent data structure."""

    def test_event_is_frozen(self):
        """NormalizedEvent should be immutable (frozen=True)."""
        event = normalize_event("test", user_id=1)
        with pytest.raises(AttributeError):
            event.intent = "changed"  # type: ignore[misc]

    def test_event_has_event_id(self):
        """Each event should have a unique ID."""
        event = normalize_event("test", user_id=1)
        assert event.event_id.startswith("evt_")
        assert len(event.event_id) > 4

    def test_event_has_timestamp(self):
        """Each event should have a timestamp."""
        event = normalize_event("test", user_id=1)
        assert event.timestamp != ""

    def test_event_to_dict(self):
        """to_dict should produce a complete dictionary."""
        event = normalize_event("Write Python code", user_id=42)
        d = event.to_dict()
        assert d["user_id"] == 42
        assert "intent" in d
        assert "fingerprint_hash" in d

    def test_custom_scope(self):
        """Custom scope should be preserved in the event."""
        event = normalize_event(
            "test",
            user_id=1,
            scope={"project": "axolent", "client": "test-client"},
        )
        assert event.scope["project"] == "axolent"
        assert event.scope["client"] == "test-client"

    def test_language_detection_german(self):
        """German text should be detected as 'de'."""
        event = normalize_event(
            "Bitte erstelle eine Tabelle mit den Ergebnissen der letzten Woche",
            user_id=1,
        )
        assert event.language == "de"

    def test_language_detection_english(self):
        """English text should be detected as 'en'."""
        event = normalize_event(
            "Please create a table with the results from last week",
            user_id=1,
        )
        assert event.language == "en"
