"""T2 Tests: ContractValidator rules V1-V17.

Coverage:
  - Production-path tests for all 17 validation rules
  - 4-path tests for user-input validation (Happy/Malicious/Rejection/Privacy)
  - Each rule tested with both valid and invalid input

Test naming: test_v{N}_{description}
"""

from __future__ import annotations

import base64
from dataclasses import replace

import pytest

from application.skill_compression.contract_validator import (
    Severity,
    validate,
)
from application.skill_compression.skill_contract import (
    ActivationConfig,
    ExecutionConfig,
    IntentConfig,
    LifecycleConfig,
    MemoryPolicyConfig,
    MemoryReadConfig,
    MemoryWriteConfig,
    PermissionsConfig,
    SafetyConfig,
    SkillContract,
    StoreMetaConfig,
    TrustConfig,
    compute_risk_level,
    create_minimal_contract,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def valid_contract() -> SkillContract:
    """A contract that passes all validation rules."""
    c = create_minimal_contract(
        name="Valid Skill",
        phrases=("hello",),
        instruction="Say hello back",
    )
    # Finalize risk_level to match permissions (V14)
    return replace(c, risk_level=compute_risk_level(c.permissions))


# ──────────────────────────────────────────────────────────────
# V1: Schema compliance
# ──────────────────────────────────────────────────────────────


class TestV1SchemaCompliance:
    def test_v1_valid_passes(self, valid_contract):
        result = validate(valid_contract)
        v1_errors = [i for i in result.errors if i.rule == "V1"]
        assert len(v1_errors) == 0

    def test_v1_missing_id(self, valid_contract):
        c = replace(valid_contract, id="")
        result = validate(c)
        assert any(i.rule == "V1" and "id" in i.field_path for i in result.errors)

    def test_v1_bad_id_prefix(self, valid_contract):
        c = replace(valid_contract, id="bad_prefix_123")
        result = validate(c)
        assert any(i.rule == "V1" and "skill_" in i.message for i in result.errors)

    def test_v1_missing_name(self, valid_contract):
        c = replace(valid_contract, name="")
        result = validate(c)
        assert any(i.rule == "V1" and "name" in i.field_path for i in result.errors)

    def test_v1_missing_created_at(self, valid_contract):
        c = replace(valid_contract, created_at="")
        result = validate(c)
        assert any(
            i.rule == "V1" and "created_at" in i.field_path for i in result.errors
        )

    def test_v1_invalid_created_by(self, valid_contract):
        c = replace(valid_contract, created_by="admin")
        result = validate(c)
        assert any(
            i.rule == "V1" and "created_by" in i.field_path for i in result.errors
        )

    def test_v1_invalid_origin(self, valid_contract):
        c = replace(valid_contract, origin="unknown_origin")
        result = validate(c)
        assert any(i.rule == "V1" and "origin" in i.field_path for i in result.errors)

    def test_v1_invalid_lifecycle_status(self, valid_contract):
        c = replace(valid_contract, lifecycle=LifecycleConfig(status="invalid_status"))
        result = validate(c)
        assert any(
            i.rule == "V1" and "lifecycle.status" in i.field_path for i in result.errors
        )

    def test_v1_invalid_review_status(self, valid_contract):
        c = replace(valid_contract, review_status="invalid")
        result = validate(c)
        assert any(
            i.rule == "V1" and "review_status" in i.field_path for i in result.errors
        )

    def test_v1_invalid_risk_level(self, valid_contract):
        c = replace(valid_contract, risk_level="critical")
        result = validate(c)
        assert any(
            i.rule == "V1" and "risk_level" in i.field_path for i in result.errors
        )


# ──────────────────────────────────────────────────────────────
# V2: schema_version
# ──────────────────────────────────────────────────────────────


class TestV2SchemaVersion:
    def test_v2_version_2_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V2" for i in result.errors)

    def test_v2_version_1_rejected(self, valid_contract):
        c = replace(valid_contract, schema_version=1)
        result = validate(c)
        assert any(i.rule == "V2" for i in result.errors)

    def test_v2_version_3_rejected(self, valid_contract):
        c = replace(valid_contract, schema_version=3)
        result = validate(c)
        assert any(i.rule == "V2" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V3: contract_version monotonic
# ──────────────────────────────────────────────────────────────


class TestV3ContractVersion:
    def test_v3_no_old_version_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V3" for i in result.errors)

    def test_v3_higher_version_passes(self, valid_contract):
        c = replace(valid_contract, contract_version=5)
        result = validate(c, old_version=3)
        assert not any(i.rule == "V3" for i in result.errors)

    def test_v3_equal_version_passes(self, valid_contract):
        c = replace(valid_contract, contract_version=3)
        result = validate(c, old_version=3)
        assert not any(i.rule == "V3" for i in result.errors)

    def test_v3_lower_version_rejected(self, valid_contract):
        c = replace(valid_contract, contract_version=2)
        result = validate(c, old_version=5)
        assert any(i.rule == "V3" for i in result.errors)

    def test_v3_version_zero_rejected(self, valid_contract):
        c = replace(valid_contract, contract_version=0)
        result = validate(c)
        assert any(i.rule == "V3" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V5: Activation validity
# ──────────────────────────────────────────────────────────────


class TestV5ActivationValidity:
    def test_v5_valid_exact_phrase(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V5" for i in result.errors)

    def test_v5_empty_phrases_for_exact_phrase_rejected(self, valid_contract):
        c = replace(
            valid_contract, activation=ActivationConfig(mode="exact_phrase", phrases=())
        )
        result = validate(c)
        assert any(i.rule == "V5" and "phrases" in i.message for i in result.errors)

    def test_v5_invalid_kind_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            activation=ActivationConfig(kind="invalid_kind", phrases=("test",)),
        )
        result = validate(c)
        assert any(i.rule == "V5" and "kind" in i.field_path for i in result.errors)

    def test_v5_invalid_mode_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            activation=ActivationConfig(mode="invalid_mode", phrases=("test",)),
        )
        result = validate(c)
        assert any(i.rule == "V5" and "mode" in i.field_path for i in result.errors)

    def test_v5_intent_kind_without_label_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            activation=ActivationConfig(
                kind="intent", mode="intent_match", phrases=("test",)
            ),
            intent=IntentConfig(label=""),
        )
        result = validate(c)
        assert any(
            i.rule == "V5" and "intent.label" in i.field_path for i in result.errors
        )

    def test_v5_intent_kind_with_label_passes(self, valid_contract):
        c = replace(
            valid_contract,
            activation=ActivationConfig(
                kind="intent", mode="intent_match", phrases=("test",)
            ),
            intent=IntentConfig(label="my_intent"),
        )
        result = validate(c)
        assert not any(
            i.rule == "V5" and "intent.label" in i.field_path for i in result.errors
        )


# ──────────────────────────────────────────────────────────────
# V6: Safety consistency
# ──────────────────────────────────────────────────────────────


class TestV6SafetyConsistency:
    def test_v6_no_tools_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V6" for i in result.errors)

    def test_v6_permissions_tools_without_allow_tools_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            permissions=PermissionsConfig(tools=("web_search",)),
            safety=SafetyConfig(allow_tools=False),
            risk_level="medium",  # V14 consistency
        )
        result = validate(c)
        assert any(i.rule == "V6" for i in result.errors)

    def test_v6_allowed_tools_without_allow_tools_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            safety=SafetyConfig(allow_tools=False, allowed_tools=("web_search",)),
        )
        result = validate(c)
        assert any(i.rule == "V6" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V7: Memory policy consistency
# ──────────────────────────────────────────────────────────────


class TestV7MemoryPolicy:
    def test_v7_valid_store_passes(self, valid_contract):
        c = replace(
            valid_contract,
            memory_policy=MemoryPolicyConfig(
                write=MemoryWriteConfig(enabled=True, target_store="past_outputs"),
            ),
        )
        result = validate(c)
        assert not any(
            i.rule == "V7" and i.severity == Severity.ERROR for i in result.issues
        )

    def test_v7_unknown_write_store_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            memory_policy=MemoryPolicyConfig(
                write=MemoryWriteConfig(enabled=True, target_store="unknown_store"),
            ),
        )
        result = validate(c)
        assert any(
            i.rule == "V7" and i.severity == Severity.ERROR for i in result.issues
        )

    def test_v7_overlapping_allowed_blocked_warning(self, valid_contract):
        c = replace(
            valid_contract,
            memory_policy=MemoryPolicyConfig(
                read=MemoryReadConfig(
                    enabled=True,
                    allowed_stores=("secrets",),
                    blocked_stores=("secrets",),
                ),
            ),
        )
        result = validate(c)
        assert any(
            i.rule == "V7" and i.severity == Severity.WARNING for i in result.issues
        )


# ──────────────────────────────────────────────────────────────
# V9: Instruction sanitized
# ──────────────────────────────────────────────────────────────


class TestV9InstructionSanitized:
    def test_v9_with_instruction_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V9" for i in result.errors)

    def test_v9_empty_instruction_for_confirmed_rejected(self, valid_contract):
        c = replace(valid_contract, execution=ExecutionConfig(instruction=""))
        result = validate(c)
        assert any(i.rule == "V9" for i in result.errors)

    def test_v9_empty_instruction_for_draft_passes(self, valid_contract):
        c = replace(
            valid_contract,
            execution=ExecutionConfig(instruction=""),
            lifecycle=LifecycleConfig(status="draft"),
            risk_level="unknown",  # Drafts can have unknown
        )
        result = validate(c)
        assert not any(i.rule == "V9" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V10: Trust field types
# ──────────────────────────────────────────────────────────────


class TestV10TrustFieldTypes:
    def test_v10_null_trust_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V10" for i in result.errors)

    def test_v10_valid_checksum_passes(self, valid_contract):
        c = replace(valid_contract, trust=TrustConfig(checksum="a" * 64))
        result = validate(c)
        assert not any(
            i.rule == "V10" and "checksum" in i.field_path for i in result.errors
        )

    def test_v10_invalid_checksum_rejected(self, valid_contract):
        c = replace(valid_contract, trust=TrustConfig(checksum="not_hex"))
        result = validate(c)
        assert any(
            i.rule == "V10" and "checksum" in i.field_path for i in result.errors
        )

    def test_v10_short_checksum_rejected(self, valid_contract):
        c = replace(valid_contract, trust=TrustConfig(checksum="abcdef"))
        result = validate(c)
        assert any(
            i.rule == "V10" and "checksum" in i.field_path for i in result.errors
        )

    def test_v10_valid_signature_passes(self, valid_contract):
        sig = base64.b64encode(b"test_signature_data").decode()
        c = replace(
            valid_contract,
            trust=TrustConfig(signature=sig, signature_algorithm="ed25519"),
        )
        result = validate(c)
        assert not any(
            i.rule == "V10" and "signature" in i.field_path for i in result.errors
        )

    def test_v10_invalid_signature_rejected(self, valid_contract):
        c = replace(valid_contract, trust=TrustConfig(signature="not_valid_base64!!!"))
        result = validate(c)
        assert any(
            i.rule == "V10" and "signature" in i.field_path for i in result.errors
        )

    def test_v10_wrong_algorithm_rejected(self, valid_contract):
        c = replace(valid_contract, trust=TrustConfig(signature_algorithm="rsa256"))
        result = validate(c)
        assert any(
            i.rule == "V10" and "algorithm" in i.field_path for i in result.errors
        )


# ──────────────────────────────────────────────────────────────
# V12: Permission vs safety consistency
# ──────────────────────────────────────────────────────────────


class TestV12PermissionVsSafety:
    def test_v12_tools_with_allow_tools_passes(self, valid_contract):
        c = replace(
            valid_contract,
            permissions=PermissionsConfig(tools=("web_search",)),
            safety=SafetyConfig(allow_tools=True, allowed_tools=("web_search",)),
            risk_level="medium",
        )
        result = validate(c)
        assert not any(i.rule == "V12" for i in result.errors)

    def test_v12_tools_without_allow_tools_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            permissions=PermissionsConfig(tools=("web_search",)),
            safety=SafetyConfig(allow_tools=False),
            risk_level="medium",
        )
        result = validate(c)
        assert any(i.rule == "V12" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V13: License validity
# ──────────────────────────────────────────────────────────────


class TestV13LicenseValidity:
    def test_v13_personal_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V13" for i in result.issues)

    def test_v13_known_spdx_passes(self, valid_contract):
        c = replace(valid_contract, store_meta=StoreMetaConfig(license="MIT"))
        result = validate(c)
        assert not any(i.rule == "V13" for i in result.issues)

    def test_v13_unknown_license_warning(self, valid_contract):
        c = replace(
            valid_contract, store_meta=StoreMetaConfig(license="UnknownLicense-42")
        )
        result = validate(c)
        v13_issues = [i for i in result.issues if i.rule == "V13"]
        assert len(v13_issues) == 1
        assert v13_issues[0].severity == Severity.WARNING

    def test_v13_unknown_license_is_warning_not_error(self, valid_contract):
        c = replace(valid_contract, store_meta=StoreMetaConfig(license="WTFPL"))
        result = validate(c)
        assert not any(
            i.rule == "V13" and i.severity == Severity.ERROR for i in result.issues
        )


# ──────────────────────────────────────────────────────────────
# V14: risk_level consistency
# ──────────────────────────────────────────────────────────────


class TestV14RiskLevelConsistency:
    def test_v14_matching_risk_passes(self, valid_contract):
        """risk_level matches compute_risk_level for no-permission contract."""
        result = validate(valid_contract)
        assert not any(i.rule == "V14" for i in result.errors)

    def test_v14_mismatch_rejected(self, valid_contract):
        c = replace(
            valid_contract, risk_level="high"
        )  # No permissions, should be "low"
        result = validate(c)
        assert any(i.rule == "V14" for i in result.errors)

    def test_v14_draft_guard_clause_allows_unknown(self, valid_contract):
        """V14 guard: drafts may have risk_level='unknown'."""
        c = replace(
            valid_contract,
            lifecycle=LifecycleConfig(status="draft"),
            risk_level="unknown",
        )
        result = validate(c)
        assert not any(i.rule == "V14" for i in result.errors)

    def test_v14_needs_input_guard_allows_unknown(self, valid_contract):
        """V14 guard: needs_input status allows any risk_level."""
        c = replace(
            valid_contract,
            lifecycle=LifecycleConfig(status="needs_input"),
            risk_level="unknown",
        )
        result = validate(c)
        assert not any(i.rule == "V14" for i in result.errors)

    def test_v14_confirmed_with_unknown_rejected(self, valid_contract):
        """Confirmed contracts must not have risk_level='unknown'."""
        c = replace(valid_contract, risk_level="unknown")
        result = validate(c)
        assert any(i.rule == "V14" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V15: Execution type feature flag
# ──────────────────────────────────────────────────────────────


class TestV15ExecutionTypeFeatureFlag:
    def test_v15_llm_instruction_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(i.rule == "V15" for i in result.errors)

    def test_v15_workflow_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            execution=ExecutionConfig(type="workflow", instruction="step1"),
        )
        result = validate(c)
        assert any(i.rule == "V15" and "reserved" in i.message for i in result.errors)

    def test_v15_tool_rejected(self, valid_contract):
        c = replace(
            valid_contract,
            execution=ExecutionConfig(type="tool", instruction="run_tool"),
        )
        result = validate(c)
        assert any(i.rule == "V15" and "reserved" in i.message for i in result.errors)

    def test_v15_invalid_type_rejected(self, valid_contract):
        c = replace(
            valid_contract, execution=ExecutionConfig(type="script", instruction="run")
        )
        result = validate(c)
        assert any(i.rule == "V15" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V16: DB==JSON invariant
# ──────────────────────────────────────────────────────────────


class TestV16DbJsonInvariant:
    def test_v16_matching_values_passes(self, valid_contract):
        result = validate(
            valid_contract,
            db_schema_version=2,
            db_contract_version=1,
        )
        assert not any(i.rule == "V16" for i in result.errors)

    def test_v16_schema_version_mismatch_rejected(self, valid_contract):
        result = validate(
            valid_contract,
            db_schema_version=1,  # Mismatch: DB says 1, JSON says 2
            db_contract_version=1,
        )
        assert any(
            i.rule == "V16" and "schema_version" in i.field_path for i in result.errors
        )

    def test_v16_contract_version_mismatch_rejected(self, valid_contract):
        result = validate(
            valid_contract,
            db_schema_version=2,
            db_contract_version=99,  # Mismatch
        )
        assert any(
            i.rule == "V16" and "contract_version" in i.field_path
            for i in result.errors
        )

    def test_v16_no_db_values_skips_check(self, valid_contract):
        """When no DB values provided, V16 is not checked."""
        result = validate(valid_contract)
        assert not any(i.rule == "V16" for i in result.errors)


# ──────────────────────────────────────────────────────────────
# V17: Origin-signature consistency
# ──────────────────────────────────────────────────────────────


class TestV17OriginSignatureConsistency:
    def test_v17_local_learn_no_signature_passes(self, valid_contract):
        result = validate(valid_contract)
        assert not any(
            i.rule == "V17" and i.severity == Severity.ERROR for i in result.issues
        )

    def test_v17_store_without_signature_rejected(self, valid_contract):
        c = replace(valid_contract, origin="store")
        result = validate(c)
        assert any(
            i.rule == "V17" and i.severity == Severity.ERROR for i in result.issues
        )

    def test_v17_store_with_signature_passes(self, valid_contract):
        sig = base64.b64encode(b"store_sig").decode()
        c = replace(
            valid_contract,
            origin="store",
            trust=TrustConfig(signature=sig, signature_algorithm="ed25519"),
        )
        result = validate(c)
        assert not any(
            i.rule == "V17" and i.severity == Severity.ERROR for i in result.issues
        )

    def test_v17_manual_install_without_signature_warning(self, valid_contract):
        c = replace(valid_contract, origin="manual_install")
        result = validate(c)
        v17_issues = [i for i in result.issues if i.rule == "V17"]
        assert len(v17_issues) == 1
        assert v17_issues[0].severity == Severity.WARNING

    def test_v17_manual_install_with_signature_no_warning(self, valid_contract):
        sig = base64.b64encode(b"install_sig").decode()
        c = replace(
            valid_contract,
            origin="manual_install",
            trust=TrustConfig(signature=sig, signature_algorithm="ed25519"),
        )
        result = validate(c)
        assert not any(i.rule == "V17" for i in result.issues)


# ──────────────────────────────────────────────────────────────
# 4-Path Security Tests (S1-S4)
# ──────────────────────────────────────────────────────────────


class TestFourPathSecurity:
    """4-path tests for user-input validation."""

    def test_s1_happy_path_valid_contract_passes(self, valid_contract):
        """S1 Happy: Normal contract passes all rules."""
        result = validate(valid_contract)
        assert result.is_valid

    def test_s2_malicious_workflow_execution_rejected(self, valid_contract):
        """S2 Malicious: Attempt to use reserved execution type."""
        c = replace(
            valid_contract,
            execution=ExecutionConfig(type="workflow", instruction="steal_data"),
        )
        result = validate(c)
        assert not result.is_valid
        assert any(i.rule == "V15" for i in result.errors)

    def test_s2_malicious_bad_origin_rejected(self, valid_contract):
        """S2 Malicious: Attempt to set invalid origin."""
        c = replace(valid_contract, origin="admin_bypass")
        result = validate(c)
        assert not result.is_valid

    def test_s2_malicious_fake_store_without_signature(self, valid_contract):
        """S2 Malicious: Claim store origin without signature."""
        c = replace(valid_contract, origin="store")
        result = validate(c)
        assert not result.is_valid
        assert any(i.rule == "V17" for i in result.errors)

    def test_s3_rejection_unknown_store_blocked(self, valid_contract):
        """S3 Rejection: Unknown memory store is rejected."""
        c = replace(
            valid_contract,
            memory_policy=MemoryPolicyConfig(
                write=MemoryWriteConfig(enabled=True, target_store="stolen_data"),
            ),
        )
        result = validate(c)
        assert any(i.rule == "V7" for i in result.errors)

    def test_s3_rejection_invalid_checksum_format(self, valid_contract):
        """S3 Rejection: Invalid checksum format is caught."""
        c = replace(
            valid_contract, trust=TrustConfig(checksum="<script>alert(1)</script>")
        )
        result = validate(c)
        assert any(i.rule == "V10" for i in result.errors)

    def test_s4_privacy_no_cleartext_in_validation_errors(self, valid_contract):
        """S4 Privacy: Validation error messages do not leak sensitive data."""
        c = replace(
            valid_contract, trust=TrustConfig(checksum="sk-FAKE-secret-key-12345")
        )
        result = validate(c)
        # The error message should describe the format issue, not echo the full value
        for issue in result.errors:
            assert (
                "sk-FAKE-secret-key" not in issue.message or "checksum" in issue.message
            )

    def test_s4_privacy_permissions_tools_not_in_error_message(self, valid_contract):
        """S4 Privacy: Tool names in permissions are not echoed verbatim in errors."""
        c = replace(
            valid_contract,
            permissions=PermissionsConfig(tools=("secret_internal_tool",)),
            safety=SafetyConfig(allow_tools=False),
            risk_level="medium",
        )
        result = validate(c)
        # V6 or V12 should fire, but should not echo "secret_internal_tool" in detail
        assert not result.is_valid


# ──────────────────────────────────────────────────────────────
# Integration: Full validation pass
# ──────────────────────────────────────────────────────────────


class TestFullValidationPass:
    """Verify that a well-formed contract passes all 17 rules."""

    def test_all_rules_pass_for_valid_contract(self, valid_contract):
        result = validate(
            valid_contract,
            db_schema_version=2,
            db_contract_version=1,
        )
        assert result.is_valid, f"Unexpected errors: {result.errors}"

    def test_multiple_errors_collected(self):
        """A very broken contract should collect multiple errors."""
        c = SkillContract()  # Totally empty/default
        result = validate(c)
        assert not result.is_valid
        assert len(result.errors) >= 2  # At least V1 (id, name, timestamps)
