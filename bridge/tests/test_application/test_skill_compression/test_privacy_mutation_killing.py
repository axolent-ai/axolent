"""Mutation-killing tests for privacy/ module.

These tests target specific decision points, thresholds, boundary values,
and branching logic that mutmut mutants typically survive. Each test would
FAIL if a single line/operator in the source were mutated.

Target: raise privacy/ mutation score from 38% toward 80%.
"""

from __future__ import annotations


from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.healthcare_filter import (
    BLOCKED_HEALTH_DOMAINS,
    HealthcareFilter,
)
from application.skill_compression.privacy.nudge_filter import (
    CATEGORY_DESCRIPTIONS,
    NudgeCategory,
    NudgeFilter,
    NudgeViolation,
    _CATEGORY_PATTERNS,
)
from application.skill_compression.privacy.privacy_pipeline import (
    PipelineRejection,
    PrivacyAuditLog,
    PrivacyPipeline,
    RejectionSource,
)
from application.security.secret_scanner import (
    MATCHED_TEXT_TRUNCATE,
    SECRET_PATTERNS,
    SecretBlockedError,
    SecretMatch,
    SecretScanner,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _hyp(
    claim: str,
    *,
    context: tuple[str, ...] = (),
    hypothesis_id: str = "hyp-mut-test",
) -> Hypothesis:
    """Create a test hypothesis."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=42,
        type="preference",
        scope=HypothesisScope(context=context),
        claim=claim,
        status="candidate",
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T10:00:00+00:00",
    )


# ===============================================================
# HEALTHCARE FILTER: mutation-killing tests
# ===============================================================


class TestHealthcareFilterMutationKilling:
    """Tests that kill specific mutants in healthcare_filter.py."""

    def test_layer1_returns_blocked_true(self) -> None:
        """Mutant: _evaluate returns HealthcareFilterResult(blocked=False) for Layer 1."""
        hf = HealthcareFilter()
        h = _hyp("User prefers bullet points", context=("health",))
        result = hf._evaluate(h)
        assert result.blocked is True
        assert result.layer == 1
        assert "health" in result.matched_term.lower()

    def test_layer1_reason_contains_domain(self) -> None:
        """Mutant: reason string does not include the matched domain."""
        hf = HealthcareFilter()
        h = _hyp("Anything", context=("psychiatry",))
        result = hf._evaluate(h)
        assert "psychiatry" in result.reason

    def test_layer2_returns_exact_matched_term(self) -> None:
        """Mutant: matched_term is empty or wrong keyword."""
        hf = HealthcareFilter()
        h = _hyp("User might have bipolar episodes")
        result = hf._evaluate(h)
        assert result.blocked is True
        assert result.layer == 2
        assert result.matched_term.lower() == "bipolar"

    def test_layer2_case_insensitive(self) -> None:
        """Mutant: re.IGNORECASE removed from _HEALTHCARE_PATTERN."""
        hf = HealthcareFilter()
        h = _hyp("Shows signs of DEPRESSION and ANXIETY")
        assert hf.filter_hypothesis(h) is True

    def test_layer3_behavioral_patterns_block(self) -> None:
        """Mutant: _BEHAVIORAL_CHANGE_PATTERNS list empty or never checked."""
        hf = HealthcareFilter()
        h = _hyp("User's tone shifted over the past months")
        result = hf._evaluate(h)
        assert result.blocked is True
        assert result.layer == 3

    def test_layer3_correlates_pattern(self) -> None:
        """Mutant: correlation pattern removed."""
        hf = HealthcareFilter()
        h = _hyp("This pattern is associated with mental disorders")
        result = hf._evaluate(h)
        assert result.blocked is True
        assert result.layer == 3

    def test_layer4_mood_inference_blocks(self) -> None:
        """Mutant: _MOOD_INFERENCE_PATTERNS never checked."""
        hf = HealthcareFilter()
        h = _hyp("User seems frustrated and overwhelmed")
        result = hf._evaluate(h)
        assert result.blocked is True
        assert result.layer == 4

    def test_layer4_sentiment_analysis(self) -> None:
        """Mutant: sentiment regex removed."""
        hf = HealthcareFilter()
        h = _hyp("User's sentiment shifted dramatically")
        result = hf._evaluate(h)
        assert result.blocked is True
        assert result.layer == 4

    def test_clean_claim_returns_blocked_false(self) -> None:
        """Mutant: _evaluate always returns blocked=True."""
        hf = HealthcareFilter()
        h = _hyp("User prefers Python type hints")
        result = hf._evaluate(h)
        assert result.blocked is False
        assert result.layer == 0
        assert result.reason == ""
        assert result.matched_term == ""

    def test_filter_hypothesis_returns_bool_true(self) -> None:
        """Mutant: filter_hypothesis returns False when _evaluate.blocked is True."""
        hf = HealthcareFilter()
        h = _hyp("Detect depression patterns")
        assert hf.filter_hypothesis(h) is True

    def test_filter_hypothesis_returns_bool_false(self) -> None:
        """Mutant: filter_hypothesis returns True when _evaluate.blocked is False."""
        hf = HealthcareFilter()
        h = _hyp("User prefers bullet points")
        assert hf.filter_hypothesis(h) is False

    def test_get_block_reason_returns_none_for_clean(self) -> None:
        """Mutant: get_block_reason returns a reason for clean claims."""
        hf = HealthcareFilter()
        h = _hyp("Bullet points are preferred")
        assert hf.get_block_reason(h) is None

    def test_get_block_reason_returns_string_for_blocked(self) -> None:
        """Mutant: get_block_reason returns None for blocked claims."""
        hf = HealthcareFilter()
        h = _hyp("Typing speed has declined over time")
        reason = hf.get_block_reason(h)
        assert reason is not None
        assert len(reason) > 10

    def test_is_health_related_event_case_insensitive(self) -> None:
        """Mutant: .lower() removed from is_health_related_event."""
        from application.skill_compression.event_normalizer import NormalizedEvent

        hf = HealthcareFilter()
        event = NormalizedEvent(domain="Health")
        assert hf.is_health_related_event(event) is True

    def test_domain_case_sensitivity_in_scope(self) -> None:
        """Mutant: tag.lower() removed in Layer 1 scope check."""
        hf = HealthcareFilter()
        h = _hyp("Anything", context=("MEDICAL",))
        assert hf.filter_hypothesis(h) is True

    def test_all_blocked_domains_are_frozenset(self) -> None:
        """Mutant: BLOCKED_HEALTH_DOMAINS type changed."""
        assert isinstance(BLOCKED_HEALTH_DOMAINS, frozenset)
        assert "health" in BLOCKED_HEALTH_DOMAINS
        assert "clinical" in BLOCKED_HEALTH_DOMAINS

    def test_layer_order_preserved(self) -> None:
        """Mutant: Layer check order changed (1,2,3,4 must be sequential)."""
        hf = HealthcareFilter()
        # This triggers Layer 1 (domain) AND Layer 2 (keyword "depression")
        h = _hyp("depression pattern", context=("health",))
        result = hf._evaluate(h)
        # Layer 1 should win (checked first)
        assert result.layer == 1

    def test_layer2_word_boundary_enforcement(self) -> None:
        """Mutant: \\b word boundaries removed from regex."""
        hf = HealthcareFilter()
        # "depress" is NOT "depression" (word boundary test)
        # But "therapy" IS a keyword
        h = _hyp("This is about therapy for the user")
        assert hf.filter_hypothesis(h) is True
        # Non-word-boundary match should NOT trigger
        h_clean = _hyp("User prefers compressed output formatting")
        # "compressed" contains "stress" but word boundary prevents match
        assert hf.filter_hypothesis(h_clean) is False


# ===============================================================
# NUDGE FILTER: mutation-killing tests
# ===============================================================


class TestNudgeFilterMutationKilling:
    """Tests that kill specific mutants in nudge_filter.py."""

    def test_evaluate_returns_none_for_clean(self) -> None:
        """Mutant: _evaluate always returns a NudgeViolation."""
        nf = NudgeFilter()
        h = _hyp("User prefers tables in Markdown format")
        result = nf._evaluate(h)
        assert result is None

    def test_evaluate_returns_violation_for_match(self) -> None:
        """Mutant: _evaluate always returns None."""
        nf = NudgeFilter()
        h = _hyp("Create engagement loops for user retention")
        result = nf._evaluate(h)
        assert result is not None
        assert isinstance(result, NudgeViolation)

    def test_violation_has_correct_category(self) -> None:
        """Mutant: wrong category assigned in NudgeViolation."""
        nf = NudgeFilter()
        h = _hyp("Show leaderboard rankings for all users")
        detail = nf.get_violation_detail(h)
        assert detail is not None
        assert detail.category == NudgeCategory.SOCIAL_MANIPULATION

    def test_violation_matched_text_not_empty(self) -> None:
        """Mutant: matched_text set to empty string."""
        nf = NudgeFilter()
        h = _hyp("Political personalization of user recommendations")
        detail = nf.get_violation_detail(h)
        assert detail is not None
        assert len(detail.matched_text) > 0

    def test_violation_matched_text_truncated_at_50(self) -> None:
        """Mutant: [:50] truncation removed or changed."""
        nf = NudgeFilter()
        # Create a claim that would produce a very long match
        long_claim = "Political personalization targeting recommend " + "x" * 100
        h = _hyp(long_claim)
        detail = nf.get_violation_detail(h)
        if detail is not None:
            assert len(detail.matched_text) <= 50

    def test_violates_nudge_policy_returns_true_on_violation(self) -> None:
        """Mutant: violates_nudge_policy returns False when violation exists."""
        nf = NudgeFilter()
        h = _hyp("Silently collect and track all user data")
        assert nf.violates_nudge_policy(h) is True

    def test_violates_nudge_policy_returns_false_on_clean(self) -> None:
        """Mutant: violates_nudge_policy returns True for clean claims."""
        nf = NudgeFilter()
        h = _hyp("Format output as markdown table")
        assert nf.violates_nudge_policy(h) is False

    def test_get_violation_category_returns_value_string(self) -> None:
        """Mutant: get_violation_category returns wrong type."""
        nf = NudgeFilter()
        h = _hyp("Detect user mood from typing patterns")
        result = nf.get_violation_category(h)
        assert result == NudgeCategory.BEHAVIORAL_INFERENCE.value
        assert isinstance(result, str)

    def test_get_violation_category_none_for_clean(self) -> None:
        """Mutant: get_violation_category returns a value for clean claims."""
        nf = NudgeFilter()
        h = _hyp("Use bullet points always")
        assert nf.get_violation_category(h) is None

    def test_description_from_category_descriptions(self) -> None:
        """Mutant: description not pulled from CATEGORY_DESCRIPTIONS."""
        nf = NudgeFilter()
        h = _hyp("Create FOMO to bring users back")
        detail = nf.get_violation_detail(h)
        assert detail is not None
        expected_desc = CATEGORY_DESCRIPTIONS[NudgeCategory.EMOTIONAL_MANIPULATION]
        assert detail.description == expected_desc

    def test_all_seven_categories_detected(self) -> None:
        """Mutant: a category is missing from _CATEGORY_PATTERNS."""
        nf = NudgeFilter()
        test_claims = {
            NudgeCategory.POLITICAL_MANIPULATION: "Political personalization of content",
            NudgeCategory.EMOTIONAL_MANIPULATION: "Fear of missing out on features",
            NudgeCategory.DARK_PATTERNS: "Hide the opt-out cancel button",
            NudgeCategory.ATTENTION_MAXIMIZATION: "Create engagement loops for sessions",
            NudgeCategory.SOCIAL_MANIPULATION: "Compare user with other users",
            NudgeCategory.BEHAVIORAL_INFERENCE: "Predict user mood from patterns",
            NudgeCategory.DATA_FLOW_VIOLATION: "Share data with third party vendor",
        }
        for expected_cat, claim in test_claims.items():
            h = _hyp(claim)
            detail = nf.get_violation_detail(h)
            assert detail is not None, f"No violation for {expected_cat}: {claim}"
            assert detail.category == expected_cat, (
                f"Expected {expected_cat}, got {detail.category} for: {claim}"
            )

    def test_first_matching_category_wins(self) -> None:
        """Mutant: iteration order not respected (fail-first not respected)."""
        nf = NudgeFilter()
        # This claim matches BOTH political AND emotional; first match wins
        h = _hyp("Political personalization and FOMO together")
        detail = nf.get_violation_detail(h)
        assert detail is not None
        # _CATEGORY_PATTERNS is a dict; the first category key that matches wins
        assert detail.category == NudgeCategory.POLITICAL_MANIPULATION

    def test_category_patterns_all_have_entries(self) -> None:
        """Mutant: a NudgeCategory has empty pattern list."""
        for cat in NudgeCategory:
            assert cat in _CATEGORY_PATTERNS
            assert len(_CATEGORY_PATTERNS[cat]) > 0


# ===============================================================
# SECRET SCANNER: mutation-killing tests
# ===============================================================


class TestSecretScannerMutationKilling:
    """Tests that kill specific mutants in secret_scanner.py."""

    def test_scan_returns_empty_for_clean_text(self) -> None:
        """Mutant: scan always returns non-empty list."""
        scanner = SecretScanner()
        matches = scanner.scan("User prefers bullet points in answers")
        assert matches == []

    def test_scan_returns_matches_for_secret(self) -> None:
        """Mutant: scan always returns empty list."""
        scanner = SecretScanner()
        matches = scanner.scan("sk-1234567890abcdef1234567890abcdef")
        assert len(matches) >= 1

    def test_match_layer_is_2_for_regex(self) -> None:
        """Mutant: layer hardcoded to wrong value."""
        scanner = SecretScanner()
        matches = scanner.scan("AKIAIOSFODNN7EXAMPLE1")
        assert any(m.layer == 2 for m in matches)

    def test_match_layer_is_3_for_heuristic_digits(self) -> None:
        """Mutant: heuristic layer value wrong."""
        scanner = SecretScanner()
        matches = scanner.scan("account 12345678901234")
        long_digit_matches = [
            m for m in matches if m.pattern_name == "long_digit_sequence"
        ]
        assert len(long_digit_matches) >= 1
        assert long_digit_matches[0].layer == 3

    def test_match_layer_is_3_for_user_path(self) -> None:
        """Mutant: user_file_path layer wrong."""
        scanner = SecretScanner()
        matches = scanner.scan("/home/user/documents/file.txt")
        path_matches = [m for m in matches if m.pattern_name == "user_file_path"]
        assert len(path_matches) >= 1
        assert path_matches[0].layer == 3

    def test_match_layer_is_3_for_third_party_pii(self) -> None:
        """Mutant: third_party_pii layer wrong."""
        scanner = SecretScanner()
        matches = scanner.scan("Herr Mueller has specific preferences")
        pii_matches = [m for m in matches if m.pattern_name == "third_party_pii"]
        assert len(pii_matches) >= 1
        assert pii_matches[0].layer == 3

    def test_matched_text_truncated_to_limit(self) -> None:
        """Mutant: MATCHED_TEXT_TRUNCATE not applied or wrong value."""
        scanner = SecretScanner()
        long_token = "sk-" + "a" * 100
        matches = scanner.scan(long_token)
        for m in matches:
            if m.matched_text != "[redacted]":
                assert len(m.matched_text) <= MATCHED_TEXT_TRUNCATE

    def test_heuristic_redacted_text(self) -> None:
        """Mutant: heuristic matches use actual text instead of [redacted]."""
        scanner = SecretScanner()
        matches = scanner.scan("account 12345678901234")
        digit_matches = [m for m in matches if m.pattern_name == "long_digit_sequence"]
        assert digit_matches[0].matched_text == "[redacted]"

    def test_third_party_pii_breaks_after_first(self) -> None:
        """Mutant: break removed from _THIRD_PARTY_PII_PATTERNS loop."""
        scanner = SecretScanner()
        # Two PII patterns in same text
        matches = scanner.scan("Herr Schmidt and Mrs. Johnson discussed")
        pii_matches = [m for m in matches if m.pattern_name == "third_party_pii"]
        # Should have exactly 1 due to break
        assert len(pii_matches) == 1

    def test_block_if_secrets_returns_true_for_secret(self) -> None:
        """Mutant: block_if_secrets returns False when matches exist."""
        scanner = SecretScanner()
        h = _hyp("Use token sk-abc123def456ghi789jkl012345678901")
        assert scanner.block_if_secrets(h) is True

    def test_block_if_secrets_returns_false_for_clean(self) -> None:
        """Mutant: block_if_secrets returns True for clean."""
        scanner = SecretScanner()
        h = _hyp("User prefers markdown tables")
        assert scanner.block_if_secrets(h) is False

    def test_block_if_secrets_reads_claim_attribute(self) -> None:
        """Mutant: getattr uses wrong attribute name."""
        scanner = SecretScanner()

        class FakeHypothesis:
            claim = "sk-abcdef1234567890abcdef1234567890"
            hypothesis_id = "fake"

        assert scanner.block_if_secrets(FakeHypothesis()) is True

    def test_get_block_reason_format(self) -> None:
        """Mutant: get_block_reason returns wrong format."""
        scanner = SecretScanner()
        h = _hyp("AKIAIOSFODNN7EXAMPLE1")
        reason = scanner.get_block_reason(h)
        assert reason is not None
        assert "Secret/PII detected" in reason
        assert "Layer" in reason

    def test_get_block_reason_none_for_clean(self) -> None:
        """Mutant: get_block_reason returns reason for clean."""
        scanner = SecretScanner()
        h = _hyp("Clean text about preferences")
        assert scanner.get_block_reason(h) is None

    def test_secret_blocked_error_has_matches(self) -> None:
        """Mutant: SecretBlockedError does not store matches."""
        match = SecretMatch(
            pattern_name="test",
            matched_text="xxx",
            description_de="Test",
            layer=2,
            pattern_label_key="secret.test",
        )
        err = SecretBlockedError([match])
        assert err.matches == [match]
        assert "1 match" in str(err)

    def test_each_secret_pattern_has_required_fields(self) -> None:
        """Mutant: a SECRET_PATTERNS entry missing fields."""
        for sp in SECRET_PATTERNS:
            assert sp.name
            assert sp.pattern is not None
            assert sp.description_de
            assert sp.pattern_label_key
            assert sp.pattern_label_key.startswith("secret.")

    def test_stripe_key_detected(self) -> None:
        """Mutant: stripe_key pattern removed."""
        scanner = SecretScanner()
        matches = scanner.scan("sk_live_" + "a" * 30)
        assert any(m.pattern_name == "stripe_key" for m in matches)

    def test_google_api_key_detected(self) -> None:
        """Mutant: google_api_key pattern removed."""
        scanner = SecretScanner()
        matches = scanner.scan("AIza" + "A" * 35)
        assert any(m.pattern_name == "google_api_key" for m in matches)

    def test_jwt_detected(self) -> None:
        """Mutant: jwt pattern removed."""
        scanner = SecretScanner()
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123xyz"  # nosemgrep: detected-jwt-token
        matches = scanner.scan(jwt)
        assert any(m.pattern_name == "jwt" for m in matches)

    def test_credit_card_detected(self) -> None:
        """Mutant: credit_card pattern removed."""
        scanner = SecretScanner()
        matches = scanner.scan("4111 1111 1111 1111")
        assert any(m.pattern_name == "credit_card" for m in matches)

    def test_ip_address_detected(self) -> None:
        """Mutant: ip_address pattern removed."""
        scanner = SecretScanner()
        matches = scanner.scan("Server at 192.168.1.100")
        assert any(m.pattern_name == "ip_address" for m in matches)

    def test_windows_path_detected(self) -> None:
        """Mutant: Windows path regex variation not matching."""
        scanner = SecretScanner()
        matches = scanner.scan(r"Located at C:\Users\admin\Desktop\file.txt")
        assert any(m.pattern_name == "user_file_path" for m in matches)


# ===============================================================
# PRIVACY PIPELINE: mutation-killing tests
# ===============================================================


class TestPrivacyPipelineMutationKilling:
    """Tests that kill specific mutants in privacy_pipeline.py."""

    def test_pipeline_check_returns_none_for_clean(self) -> None:
        """Mutant: check always returns a PipelineRejection."""
        pipeline = PrivacyPipeline()
        h = _hyp("User prefers short answers")
        assert pipeline.check(h) is None

    def test_pipeline_check_returns_rejection_for_healthcare(self) -> None:
        """Mutant: check always returns None."""
        pipeline = PrivacyPipeline()
        h = _hyp("User has anxiety based on patterns")
        result = pipeline.check(h)
        assert result is not None
        assert isinstance(result, PipelineRejection)

    def test_rejection_source_healthcare(self) -> None:
        """Mutant: wrong RejectionSource assigned."""
        pipeline = PrivacyPipeline()
        h = _hyp("User shows signs of depression")
        result = pipeline.check(h)
        assert result is not None
        assert result.source == RejectionSource.HEALTHCARE

    def test_rejection_source_secret(self) -> None:
        """Mutant: SECRET source not used for secret matches."""
        pipeline = PrivacyPipeline()
        h = _hyp("Token is sk-abc123def456ghi789jkl0123456789ab")
        result = pipeline.check(h)
        assert result is not None
        assert result.source == RejectionSource.SECRET

    def test_rejection_source_nudge(self) -> None:
        """Mutant: NUDGE source not used for nudge matches."""
        pipeline = PrivacyPipeline()
        h = _hyp("Create engagement loops for maximum time")
        result = pipeline.check(h)
        assert result is not None
        assert result.source == RejectionSource.NUDGE

    def test_rejection_has_hypothesis_id(self) -> None:
        """Mutant: hypothesis_id not propagated to rejection."""
        pipeline = PrivacyPipeline()
        h = _hyp("User shows depression", hypothesis_id="test-id-123")
        result = pipeline.check(h)
        assert result is not None
        assert result.hypothesis_id == "test-id-123"

    def test_rejection_has_timestamp(self) -> None:
        """Mutant: timestamp not set."""
        pipeline = PrivacyPipeline()
        h = _hyp("ADHD based on patterns")
        result = pipeline.check(h)
        assert result is not None
        assert result.timestamp is not None
        assert len(result.timestamp) > 10  # ISO format

    def test_rejection_has_reason(self) -> None:
        """Mutant: reason is empty string."""
        pipeline = PrivacyPipeline()
        h = _hyp("User shows signs of bipolar disorder")
        result = pipeline.check(h)
        assert result is not None
        assert len(result.reason) > 5

    def test_fail_fast_healthcare_before_secret(self) -> None:
        """Mutant: filter order changed (secret before healthcare)."""
        pipeline = PrivacyPipeline()
        # "depression" triggers healthcare, "sk-..." triggers secret
        h = _hyp("User has depression and uses sk-abc123def456ghi789jkl0123456")
        result = pipeline.check(h)
        assert result is not None
        assert result.source == RejectionSource.HEALTHCARE

    def test_fail_fast_secret_before_nudge(self) -> None:
        """Mutant: nudge checked before secret."""
        pipeline = PrivacyPipeline()
        # Secret + nudge trigger, but no healthcare
        h = _hyp("Create engagement loops with key sk-abc123def456ghi789jkl012345")
        result = pipeline.check(h)
        assert result is not None
        assert result.source == RejectionSource.SECRET

    def test_is_blocked_true_for_blocked(self) -> None:
        """Mutant: is_blocked returns False for blocked hypothesis."""
        pipeline = PrivacyPipeline()
        h = _hyp("User has PTSD from behavioral analysis")
        assert pipeline.is_blocked(h) is True

    def test_is_blocked_false_for_clean(self) -> None:
        """Mutant: is_blocked returns True for clean hypothesis."""
        pipeline = PrivacyPipeline()
        h = _hyp("User likes tables")
        assert pipeline.is_blocked(h) is False

    def test_audit_log_increments_on_rejection(self) -> None:
        """Mutant: _audit.add not called."""
        pipeline = PrivacyPipeline()
        h = _hyp("User shows signs of schizophrenia")
        pipeline.check(h)
        assert pipeline.audit_log.total_rejections == 1

    def test_audit_log_not_incremented_for_clean(self) -> None:
        """Mutant: audit always incremented."""
        pipeline = PrivacyPipeline()
        h = _hyp("User prefers markdown")
        pipeline.check(h)
        assert pipeline.audit_log.total_rejections == 0


# ===============================================================
# AUDIT LOG: mutation-killing tests
# ===============================================================


class TestPrivacyAuditLogMutationKilling:
    """Tests that kill mutants in PrivacyAuditLog."""

    def test_add_appends_entry(self) -> None:
        """Mutant: add does not append to entries."""
        audit = PrivacyAuditLog()
        rejection = PipelineRejection(
            hypothesis_id="h1",
            source=RejectionSource.HEALTHCARE,
            reason="test",
            timestamp="2026-05-20T10:00:00+00:00",
        )
        audit.add(rejection)
        assert len(audit.entries) == 1
        assert audit.entries[0] is rejection

    def test_rotation_at_max_entries(self) -> None:
        """Mutant: rotation logic wrong (keeps wrong half)."""
        audit = PrivacyAuditLog(max_entries=10)
        for i in range(11):
            rejection = PipelineRejection(
                hypothesis_id=f"h{i}",
                source=RejectionSource.HEALTHCARE,
                reason=f"reason-{i}",
                timestamp="2026-05-20T10:00:00+00:00",
            )
            audit.add(rejection)
        # After overflow: keeps newest half (max_entries // 2 = 5) + the new one
        assert len(audit.entries) <= 10
        # The newest entry should still be there
        assert audit.entries[-1].hypothesis_id == "h10"

    def test_rotation_keeps_newest_half(self) -> None:
        """Mutant: slicing direction reversed (keeps oldest instead of newest)."""
        audit = PrivacyAuditLog(max_entries=4)
        for i in range(5):
            rejection = PipelineRejection(
                hypothesis_id=f"h{i}",
                source=RejectionSource.HEALTHCARE,
                reason=f"reason-{i}",
                timestamp="2026-05-20T10:00:00+00:00",
            )
            audit.add(rejection)
        # After adding 5th (overflow): keeps entries[2:] (newest half) + 5th
        # Oldest (h0, h1) should be gone
        ids = [e.hypothesis_id for e in audit.entries]
        assert "h0" not in ids
        assert "h4" in ids

    def test_get_recent_returns_newest_first(self) -> None:
        """Mutant: reversed() removed from get_recent."""
        audit = PrivacyAuditLog()
        for i in range(3):
            rejection = PipelineRejection(
                hypothesis_id=f"h{i}",
                source=RejectionSource.HEALTHCARE,
                reason=f"reason-{i}",
                timestamp="2026-05-20T10:00:00+00:00",
            )
            audit.add(rejection)
        recent = audit.get_recent(10)
        assert recent[0].hypothesis_id == "h2"  # newest first
        assert recent[-1].hypothesis_id == "h0"  # oldest last

    def test_get_recent_respects_count(self) -> None:
        """Mutant: count parameter ignored."""
        audit = PrivacyAuditLog()
        for i in range(10):
            rejection = PipelineRejection(
                hypothesis_id=f"h{i}",
                source=RejectionSource.HEALTHCARE,
                reason=f"reason-{i}",
                timestamp="2026-05-20T10:00:00+00:00",
            )
            audit.add(rejection)
        recent = audit.get_recent(3)
        assert len(recent) == 3

    def test_total_rejections_property(self) -> None:
        """Mutant: total_rejections returns wrong value."""
        audit = PrivacyAuditLog()
        assert audit.total_rejections == 0
        rejection = PipelineRejection(
            hypothesis_id="h1",
            source=RejectionSource.SECRET,
            reason="test",
            timestamp="2026-05-20T10:00:00+00:00",
        )
        audit.add(rejection)
        assert audit.total_rejections == 1
