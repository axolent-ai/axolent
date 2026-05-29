"""Contract Validator: 17 validation rules (V1-V17) for SkillContract.

Each rule returns a ValidationResult. The full validate() function runs
all applicable rules and collects errors/warnings.

Rules:
  V1:  Schema compliance (required fields, correct types)
  V2:  schema_version must be 2
  V3:  contract_version monotonically increasing (requires old_version param)
  V4:  Referential integrity (hypothesis_id exists) - deferred, needs DB
  V5:  Activation validity (kind + phrases + mode consistent)
  V6:  Safety consistency (allow_tools / permissions.tools sync)
  V7:  Memory policy consistency (write target must be known store)
  V8:  Name uniqueness - deferred, needs DB
  V9:  Instruction sanitized flag consistency
  V10: Trust field types (checksum is hex or None, signature is base64 or None)
  V11: Permission deny defaults (auto-fix)
  V12: Permission vs safety consistency (tools -> allow_tools)
  V13: License validity (SPDX or "personal")
  V14: risk_level consistency (with draft guard clause per Addendum K1)
  V15: Execution type feature flag (workflow/tool reserved)
  V16: DB==JSON invariant (external, checked at persist time)
  V17: Origin-signature consistency (Addendum K2)

Dependencies: Python stdlib only.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from application.skill_compression.skill_contract import (
    SkillContract,
    compute_risk_level,
)


# ──────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single validation finding."""

    rule: str  # e.g. "V1", "V15"
    severity: Severity
    message: str
    field_path: str = ""  # e.g. "execution.type"


@dataclass(frozen=True)
class ValidationResult:
    """Aggregated validation outcome."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """True if no ERROR-level issues found."""
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


# ──────────────────────────────────────────────────────────────
# Known stores and licenses
# ──────────────────────────────────────────────────────────────

KNOWN_MEMORY_STORES = frozenset(
    {
        "long_term_facts",
        "history_summaries",
        "project_files",
        "past_outputs",
    }
)

KNOWN_SPDX_LICENSES = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "AGPL-3.0-only",
        "AGPL-3.0-or-later",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "MPL-2.0",
        "LGPL-3.0-only",
        "ISC",
        "Unlicense",
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "personal",
    }
)

VALID_ORIGINS = frozenset({"local_learn", "manual_install", "store", "migrated"})
VALID_ACTIVATION_KINDS = frozenset(
    {"shortcut", "intent", "workflow", "conditional", "conversation_flow"}
)
VALID_ACTIVATION_MODES = frozenset({"exact_phrase", "intent_match", "regex"})
VALID_MATCH_SCOPES = frozenset({"whole_message", "contains", "starts_with"})
VALID_LIFECYCLE_STATUSES = frozenset(
    {
        "confirmed",
        "active",
        "paused",
        "needs_review",
        "draft",
        "needs_input",
    }
)
VALID_REVIEW_STATUSES = frozenset(
    {
        "unreviewed",
        "reviewed",
        "verified",
        "flagged",
        "blocked",
    }
)
VALID_RISK_LEVELS = frozenset({"unknown", "low", "medium", "high"})
VALID_EXECUTION_TYPES = frozenset({"llm_instruction", "workflow", "tool"})
RESERVED_EXECUTION_TYPES = frozenset({"workflow", "tool"})

_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


# ──────────────────────────────────────────────────────────────
# Individual validation rules
# ──────────────────────────────────────────────────────────────


def _v1_schema_compliance(c: SkillContract) -> list[ValidationIssue]:
    """V1: Required fields present and correct types."""
    issues: list[ValidationIssue] = []

    if not c.id:
        issues.append(ValidationIssue("V1", Severity.ERROR, "id is required", "id"))
    elif not c.id.startswith("skill_"):
        issues.append(
            ValidationIssue("V1", Severity.ERROR, "id must start with 'skill_'", "id")
        )

    if not c.name:
        issues.append(ValidationIssue("V1", Severity.ERROR, "name is required", "name"))

    if not c.created_at:
        issues.append(
            ValidationIssue(
                "V1", Severity.ERROR, "created_at is required", "created_at"
            )
        )

    if not c.updated_at:
        issues.append(
            ValidationIssue(
                "V1", Severity.ERROR, "updated_at is required", "updated_at"
            )
        )

    if c.created_by not in ("user", "system"):
        issues.append(
            ValidationIssue(
                "V1",
                Severity.ERROR,
                f"created_by must be 'user' or 'system', got '{c.created_by}'",
                "created_by",
            )
        )

    if c.migration_status not in ("current", "needs_migration"):
        issues.append(
            ValidationIssue(
                "V1",
                Severity.ERROR,
                f"migration_status must be 'current' or 'needs_migration', got '{c.migration_status}'",
                "migration_status",
            )
        )

    if c.origin not in VALID_ORIGINS:
        issues.append(
            ValidationIssue(
                "V1",
                Severity.ERROR,
                f"origin must be one of {sorted(VALID_ORIGINS)}, got '{c.origin}'",
                "origin",
            )
        )

    if c.lifecycle.status not in VALID_LIFECYCLE_STATUSES:
        issues.append(
            ValidationIssue(
                "V1",
                Severity.ERROR,
                f"lifecycle.status must be one of {sorted(VALID_LIFECYCLE_STATUSES)}, got '{c.lifecycle.status}'",
                "lifecycle.status",
            )
        )

    if c.review_status not in VALID_REVIEW_STATUSES:
        issues.append(
            ValidationIssue(
                "V1",
                Severity.ERROR,
                f"review_status must be one of {sorted(VALID_REVIEW_STATUSES)}, got '{c.review_status}'",
                "review_status",
            )
        )

    if c.risk_level not in VALID_RISK_LEVELS:
        issues.append(
            ValidationIssue(
                "V1",
                Severity.ERROR,
                f"risk_level must be one of {sorted(VALID_RISK_LEVELS)}, got '{c.risk_level}'",
                "risk_level",
            )
        )

    return issues


def _v2_schema_version(c: SkillContract) -> list[ValidationIssue]:
    """V2: schema_version must be exactly 2."""
    if c.schema_version != 2:
        return [
            ValidationIssue(
                "V2",
                Severity.ERROR,
                f"schema_version must be 2, got {c.schema_version}",
                "schema_version",
            )
        ]
    return []


def _v3_contract_version_monotonic(
    c: SkillContract, old_version: Optional[int] = None
) -> list[ValidationIssue]:
    """V3: contract_version must be >= old version (if updating)."""
    if c.contract_version < 1:
        return [
            ValidationIssue(
                "V3",
                Severity.ERROR,
                f"contract_version must be >= 1, got {c.contract_version}",
                "contract_version",
            )
        ]
    if old_version is not None and c.contract_version < old_version:
        return [
            ValidationIssue(
                "V3",
                Severity.ERROR,
                f"contract_version {c.contract_version} < old version {old_version}",
                "contract_version",
            )
        ]
    return []


def _v5_activation_validity(c: SkillContract) -> list[ValidationIssue]:
    """V5: Activation kind + phrases + mode must be consistent."""
    issues: list[ValidationIssue] = []
    act = c.activation

    if act.kind not in VALID_ACTIVATION_KINDS:
        issues.append(
            ValidationIssue(
                "V5",
                Severity.ERROR,
                f"activation.kind must be one of {sorted(VALID_ACTIVATION_KINDS)}, got '{act.kind}'",
                "activation.kind",
            )
        )

    if act.mode not in VALID_ACTIVATION_MODES:
        issues.append(
            ValidationIssue(
                "V5",
                Severity.ERROR,
                f"activation.mode must be one of {sorted(VALID_ACTIVATION_MODES)}, got '{act.mode}'",
                "activation.mode",
            )
        )

    if act.match_scope not in VALID_MATCH_SCOPES:
        issues.append(
            ValidationIssue(
                "V5",
                Severity.ERROR,
                f"activation.match_scope must be one of {sorted(VALID_MATCH_SCOPES)}, got '{act.match_scope}'",
                "activation.match_scope",
            )
        )

    # exact_phrase mode requires non-empty phrases
    if act.mode == "exact_phrase" and not act.phrases:
        issues.append(
            ValidationIssue(
                "V5",
                Severity.ERROR,
                "activation.phrases must not be empty for exact_phrase mode",
                "activation.phrases",
            )
        )

    # intent mode requires intent label
    if act.kind == "intent" and not c.intent.label:
        issues.append(
            ValidationIssue(
                "V5",
                Severity.ERROR,
                "intent.label is required when activation.kind is 'intent'",
                "intent.label",
            )
        )

    return issues


def _v6_safety_consistency(c: SkillContract) -> list[ValidationIssue]:
    """V6: allow_tools / allowed_tools / permissions.tools must be in sync."""
    issues: list[ValidationIssue] = []

    # If safety says no tools but permissions declares tools
    if not c.safety.allow_tools and c.permissions.tools:
        issues.append(
            ValidationIssue(
                "V6",
                Severity.ERROR,
                "permissions.tools is non-empty but safety.allow_tools is False",
                "safety.allow_tools",
            )
        )

    # If safety declares allowed_tools but allow_tools is False
    if not c.safety.allow_tools and c.safety.allowed_tools:
        issues.append(
            ValidationIssue(
                "V6",
                Severity.ERROR,
                "safety.allowed_tools is non-empty but safety.allow_tools is False",
                "safety.allowed_tools",
            )
        )

    return issues


def _v7_memory_policy_consistency(c: SkillContract) -> list[ValidationIssue]:
    """V7: Memory write target must be a known store."""
    issues: list[ValidationIssue] = []

    if c.memory_policy.write.enabled and c.memory_policy.write.target_store:
        if c.memory_policy.write.target_store not in KNOWN_MEMORY_STORES:
            issues.append(
                ValidationIssue(
                    "V7",
                    Severity.ERROR,
                    f"memory_policy.write.target_store '{c.memory_policy.write.target_store}' "
                    f"is not a known store. Known: {sorted(KNOWN_MEMORY_STORES)}",
                    "memory_policy.write.target_store",
                )
            )

    # Blocked stores in read should not overlap with allowed stores
    if c.memory_policy.read.enabled:
        overlap = set(c.memory_policy.read.allowed_stores) & set(
            c.memory_policy.read.blocked_stores
        )
        if overlap:
            issues.append(
                ValidationIssue(
                    "V7",
                    Severity.WARNING,
                    f"memory_policy.read: stores {sorted(overlap)} are both allowed and blocked",
                    "memory_policy.read",
                )
            )

    return issues


def _v9_instruction_sanitized(c: SkillContract) -> list[ValidationIssue]:
    """V9: instruction_sanitized flag must match actual execution.instruction state."""
    if not c.execution.instruction and c.lifecycle.status not in (
        "draft",
        "needs_input",
    ):
        return [
            ValidationIssue(
                "V9",
                Severity.ERROR,
                "execution.instruction is empty for a non-draft contract",
                "execution.instruction",
            )
        ]
    return []


def _v10_trust_field_types(c: SkillContract) -> list[ValidationIssue]:
    """V10: Trust field type validation."""
    issues: list[ValidationIssue] = []

    if c.trust.checksum is not None and not _HEX_PATTERN.match(c.trust.checksum):
        issues.append(
            ValidationIssue(
                "V10",
                Severity.ERROR,
                "trust.checksum must be a 64-character lowercase hex string or null",
                "trust.checksum",
            )
        )

    if c.trust.signature is not None:
        try:
            base64.b64decode(c.trust.signature, validate=True)
        except Exception:
            issues.append(
                ValidationIssue(
                    "V10",
                    Severity.ERROR,
                    "trust.signature must be valid base64 or null",
                    "trust.signature",
                )
            )

    if (
        c.trust.signature_algorithm is not None
        and c.trust.signature_algorithm != "ed25519"
    ):
        issues.append(
            ValidationIssue(
                "V10",
                Severity.ERROR,
                f"trust.signature_algorithm must be 'ed25519' or null, got '{c.trust.signature_algorithm}'",
                "trust.signature_algorithm",
            )
        )

    return issues


def _v12_permission_vs_safety(c: SkillContract) -> list[ValidationIssue]:
    """V12: permissions.tools non-empty requires safety.allow_tools == True."""
    if c.permissions.tools and not c.safety.allow_tools:
        return [
            ValidationIssue(
                "V12",
                Severity.ERROR,
                "permissions.tools is non-empty but safety.allow_tools is False; "
                "either clear permissions.tools or set safety.allow_tools=True",
                "permissions.tools",
            )
        ]
    return []


def _v13_license_validity(c: SkillContract) -> list[ValidationIssue]:
    """V13: License must be known SPDX or 'personal'. Unknown is warning only."""
    license_val = c.store_meta.license
    if license_val and license_val not in KNOWN_SPDX_LICENSES:
        return [
            ValidationIssue(
                "V13",
                Severity.WARNING,
                f"store_meta.license '{license_val}' is not a known SPDX license or 'personal'",
                "store_meta.license",
            )
        ]
    return []


def _v14_risk_level_consistency(c: SkillContract) -> list[ValidationIssue]:
    """V14: risk_level must match compute_risk_level(permissions).

    Guard clause (Addendum K1): Only enforced when lifecycle.status
    is NOT in {draft, needs_input}. Drafts may have risk_level='unknown'.
    """
    if c.lifecycle.status in ("draft", "needs_input"):
        return []

    expected = compute_risk_level(c.permissions)
    if c.risk_level != expected:
        return [
            ValidationIssue(
                "V14",
                Severity.ERROR,
                f"risk_level '{c.risk_level}' does not match computed value '{expected}' "
                f"based on permissions. Use _finalize_security_metadata() before validation.",
                "risk_level",
            )
        ]
    return []


def _v15_execution_type_feature_flag(c: SkillContract) -> list[ValidationIssue]:
    """V15: execution.type 'workflow' and 'tool' are reserved, not yet implemented."""
    if c.execution.type in RESERVED_EXECUTION_TYPES:
        return [
            ValidationIssue(
                "V15",
                Severity.ERROR,
                f"execution.type '{c.execution.type}' is reserved and not yet implemented. "
                f"Only 'llm_instruction' is allowed in Phase 1.",
                "execution.type",
            )
        ]
    if c.execution.type not in VALID_EXECUTION_TYPES:
        return [
            ValidationIssue(
                "V15",
                Severity.ERROR,
                f"execution.type '{c.execution.type}' is not a valid type. "
                f"Must be one of {sorted(VALID_EXECUTION_TYPES)}.",
                "execution.type",
            )
        ]
    return []


def _v16_db_json_invariant(
    c: SkillContract,
    *,
    db_schema_version: Optional[int] = None,
    db_contract_version: Optional[int] = None,
) -> list[ValidationIssue]:
    """V16: DB index columns must match JSON values.

    This is called at persist time with the DB column values.
    If db_* params are None, the check is skipped (not applicable).
    """
    issues: list[ValidationIssue] = []

    if db_schema_version is not None and db_schema_version != c.schema_version:
        issues.append(
            ValidationIssue(
                "V16",
                Severity.ERROR,
                f"DB schema_version ({db_schema_version}) != JSON schema_version ({c.schema_version})",
                "schema_version",
            )
        )

    if db_contract_version is not None and db_contract_version != c.contract_version:
        issues.append(
            ValidationIssue(
                "V16",
                Severity.ERROR,
                f"DB contract_version ({db_contract_version}) != JSON contract_version ({c.contract_version})",
                "contract_version",
            )
        )

    return issues


def _v17_origin_signature_consistency(c: SkillContract) -> list[ValidationIssue]:
    """V17: Store-origin skills require a signature. (Addendum K2)

    origin == 'store' and trust.signature is None => Reject.
    origin == 'manual_install' without signature => Warning (not reject).
    """
    if c.origin == "store" and c.trust.signature is None:
        return [
            ValidationIssue(
                "V17",
                Severity.ERROR,
                "origin is 'store' but trust.signature is null. "
                "Store-distributed skills must be signed.",
                "trust.signature",
            )
        ]

    if c.origin == "manual_install" and c.trust.signature is None:
        return [
            ValidationIssue(
                "V17",
                Severity.WARNING,
                "origin is 'manual_install' but trust.signature is null. "
                "Consider signing for integrity verification.",
                "trust.signature",
            )
        ]

    return []


# ──────────────────────────────────────────────────────────────
# Main validation entry point
# ──────────────────────────────────────────────────────────────


def validate(
    contract: SkillContract,
    *,
    old_version: Optional[int] = None,
    db_schema_version: Optional[int] = None,
    db_contract_version: Optional[int] = None,
) -> ValidationResult:
    """Run all applicable validation rules on a SkillContract.

    Args:
        contract: The contract to validate.
        old_version: Previous contract_version for V3 monotonicity check.
        db_schema_version: DB column value for V16 invariant check.
        db_contract_version: DB column value for V16 invariant check.

    Returns:
        ValidationResult with all collected issues.
    """
    all_issues: list[ValidationIssue] = []

    # Core rules (always run)
    all_issues.extend(_v1_schema_compliance(contract))
    all_issues.extend(_v2_schema_version(contract))
    all_issues.extend(_v3_contract_version_monotonic(contract, old_version))
    # V4 (referential integrity) is deferred to ContractStore (needs DB)
    all_issues.extend(_v5_activation_validity(contract))
    all_issues.extend(_v6_safety_consistency(contract))
    all_issues.extend(_v7_memory_policy_consistency(contract))
    # V8 (name uniqueness) is deferred to ContractStore (needs DB)
    all_issues.extend(_v9_instruction_sanitized(contract))
    all_issues.extend(_v10_trust_field_types(contract))
    # V11 (permission deny defaults) is handled by dataclass defaults (auto-fix)
    all_issues.extend(_v12_permission_vs_safety(contract))
    all_issues.extend(_v13_license_validity(contract))
    all_issues.extend(_v14_risk_level_consistency(contract))
    all_issues.extend(_v15_execution_type_feature_flag(contract))
    all_issues.extend(
        _v16_db_json_invariant(
            contract,
            db_schema_version=db_schema_version,
            db_contract_version=db_contract_version,
        )
    )
    all_issues.extend(_v17_origin_signature_consistency(contract))

    return ValidationResult(issues=tuple(all_issues))
