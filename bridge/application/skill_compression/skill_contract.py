"""Skill Contract v2: Typed dataclasses for the full contract schema.

Schema version 2 defines the canonical representation of a learned skill.
JSON is the canonical storage format (DB + IPC). YAML is import/export only.

All dataclasses are frozen (immutable) and use slots for memory efficiency.
Tuple is used instead of list for frozen-hashable collections.

Fields added in v3 plan:
  - trust (TrustConfig): signature, checksum, author_id
  - permissions (PermissionsConfig): deny-by-default capability model
  - store_meta (StoreMetaConfig): package_type, license, store listing
  - audit (AuditConfig): runtime enforcement flag
  - origin: local_learn | manual_install | store
  - review_status, risk_level: security classification

Dependencies: Python stdlib only (json, hashlib, unicodedata, uuid, datetime).
No external libraries.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


# ──────────────────────────────────────────────────────────────
# Deserialization errors
# ──────────────────────────────────────────────────────────────


class ContractDeserializationError(Exception):
    """Raised when strict deserialization detects invalid types or values."""


# ──────────────────────────────────────────────────────────────
# Sub-config dataclasses (alphabetical by section name)
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Trigger normalization rules."""

    case_fold: bool = True
    german_ss_equivalence: bool = True
    ignore_trailing_punctuation: bool = True


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    """Skill trigger definition."""

    kind: str = (
        "shortcut"  # shortcut | intent | workflow | conditional | conversation_flow
    )
    mode: str = "exact_phrase"  # exact_phrase | intent_match | regex
    phrases: tuple[str, ...] = ()
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    match_scope: str = "whole_message"  # whole_message | contains | starts_with
    conditions: tuple[str, ...] = ()
    cooldown_seconds: int = 0


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Runtime enforcement flags."""

    capability_broker_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ConfirmationThresholds:
    """Confidence thresholds for execution gating."""

    auto_execute: float = 0.95
    ask_confirm: float = 0.75
    reject: float = 0.50


@dataclass(frozen=True, slots=True)
class ConfirmationConfig:
    """Skill confirmation policy."""

    mode: str = "confidence_gated"
    thresholds: ConfirmationThresholds = field(default_factory=ConfirmationThresholds)
    destructive_always_confirm: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Skill execution definition."""

    type: str = (
        "llm_instruction"  # llm_instruction | workflow (reserved) | tool (reserved)
    )
    instruction: str = ""
    timeout_seconds: int = 30
    max_tool_calls: int = 0


@dataclass(frozen=True, slots=True)
class FileAccessConfig:
    """File system access permissions."""

    enabled: bool = False
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HistoryAccessConfig:
    """Chat history access permissions."""

    enabled: bool = False
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntentConfig:
    """Intent classification config (for kind=intent skills)."""

    label: str = ""
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    """Skill lifecycle state."""

    status: str = (
        "confirmed"  # confirmed | active | paused | needs_review | draft | needs_input
    )
    editable: bool = True
    decay: str = "immune"
    last_schema_migration: Optional[str] = None


@dataclass(frozen=True, slots=True)
class MemoryReadConfig:
    """Memory read policy."""

    enabled: bool = False
    allowed_stores: tuple[str, ...] = ()
    blocked_stores: tuple[str, ...] = ("secrets", "health", "raw_finance")
    max_items: int = 10


@dataclass(frozen=True, slots=True)
class MemoryWriteConfig:
    """Memory write policy."""

    enabled: bool = False
    target_store: Optional[str] = None
    ttl_days: Optional[int] = None


@dataclass(frozen=True, slots=True)
class MemoryPolicyConfig:
    """Full memory access policy."""

    read: MemoryReadConfig = field(default_factory=MemoryReadConfig)
    write: MemoryWriteConfig = field(default_factory=MemoryWriteConfig)
    conflict_handling: str = "ignore"  # ignore | inject_if_relevant | always_inject


@dataclass(frozen=True, slots=True)
class NetworkAccessConfig:
    """Network access permissions."""

    enabled: bool = False
    domains: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Skill usage tracking."""

    track_success: bool = True
    track_declines: bool = True
    last_triggered_at: Optional[str] = None
    success_count: int = 0
    decline_count: int = 0
    average_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class PermissionsConfig:
    """Deny-by-default permission object.

    Every field defaults to empty/False. A skill must explicitly
    declare what it needs. The CapabilityBroker enforces these
    at runtime.
    """

    tools: tuple[str, ...] = ()
    memory_read: tuple[str, ...] = ()
    memory_write: tuple[str, ...] = ()
    network_access: NetworkAccessConfig = field(default_factory=NetworkAccessConfig)
    file_access: FileAccessConfig = field(default_factory=FileAccessConfig)
    history_access: HistoryAccessConfig = field(default_factory=HistoryAccessConfig)
    secrets_access: bool = False


@dataclass(frozen=True, slots=True)
class PriorityConfig:
    """Skill priority and conflict resolution."""

    rank: int = 80
    beats: tuple[str, ...] = ("memory_conflict", "style_preference")
    loses_to: tuple[str, ...] = ("safety_policy", "explicit_user_command")


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    """Skill safety constraints."""

    requires_secret_scan: bool = True
    pii_risk: str = "low"  # low | medium | high
    allow_tools: bool = False
    allowed_tools: tuple[str, ...] = ()
    max_tool_calls_per_execution: int = 5
    instruction_sanitized: bool = True


@dataclass(frozen=True, slots=True)
class ScopeConfig:
    """Skill scope definition."""

    user_scope: str = "single_user"
    workspace: str = "global"
    channels: tuple[str, ...] = ("telegram",)
    providers: tuple[str, ...] = ("all",)


@dataclass(frozen=True, slots=True)
class StoreMetaConfig:
    """Store metadata. package_type is computed automatically from permissions."""

    package_type: str = "local_skill"  # local_skill | declarative_skill | tool_workflow | code_plugin | privileged_plugin
    license: str = "personal"  # SPDX value or "personal"
    store_listing_id: Optional[str] = None
    required_axolent_version: Optional[str] = None
    manifest_version: Optional[str] = None
    permissions_version: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TrustConfig:
    """Trust/integrity fields. Signature required only for store skills."""

    signature: Optional[str] = None
    signature_algorithm: Optional[str] = None
    checksum: Optional[str] = None
    signed_at: Optional[str] = None
    author_id: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Main SkillContract dataclass
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillContract:
    """Complete Skill Contract v2. JSON is the canonical storage format.

    All fields have safe defaults. A minimal valid contract needs at least:
      - id (generated)
      - name
      - activation.phrases (non-empty for exact_phrase mode)
      - execution.instruction (non-empty)
      - created_at, updated_at (ISO 8601)
    """

    # Identity
    schema_version: int = 2
    contract_version: int = 1
    id: str = ""
    name: str = ""
    hypothesis_id: Optional[str] = None
    created_by: str = "user"  # user | system
    created_at: str = ""
    updated_at: str = ""
    migration_status: str = "current"  # current | needs_migration

    # Core sections
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    slots: tuple = ()
    """Reserved for future workflow structures (Phase 2+). Not executed in
    Phase 1, not security-relevant. A dedicated schema or validator will be
    defined before workflow activation."""
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    provider_hints: tuple = ()
    """Advisory-only routing hints. MUST NEVER force a security-relevant
    routing decision. A dedicated schema or validator will be defined before
    store activation."""
    memory_policy: MemoryPolicyConfig = field(default_factory=MemoryPolicyConfig)
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    tags: tuple[str, ...] = ()

    # Trust & Safety Foundation (v3)
    trust: TrustConfig = field(default_factory=TrustConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    review_status: str = (
        "unreviewed"  # unreviewed | reviewed | verified | flagged | blocked
    )
    risk_level: str = "unknown"  # unknown | low | medium | high
    store_meta: StoreMetaConfig = field(default_factory=StoreMetaConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    # Origin (Addendum K2)
    origin: str = "local_learn"  # local_learn | manual_install | store

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-compatible).

        Uses a deterministic key order matching the dataclass field order.
        Nested dataclasses are recursively converted.
        """
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SkillContract:
        """Deserialize from a plain dict.

        Unknown keys are silently ignored for forward compatibility.
        Missing keys use dataclass defaults.
        """
        return _dict_to_contract(data)

    def to_json(self, *, canonical: bool = False) -> str:
        """Serialize to JSON string.

        Args:
            canonical: If True, produce canonical JSON for checksumming
                       (sort_keys, compact separators, NFC normalized).
        """
        d = self.to_dict()
        if canonical:
            raw = json.dumps(
                d, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            return unicodedata.normalize("NFC", raw)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> SkillContract:
        """Deserialize from JSON string."""
        data = json.loads(raw)
        return cls.from_dict(data)


# ──────────────────────────────────────────────────────────────
# Canonical JSON + Checksum (Addendum K3)
# ──────────────────────────────────────────────────────────────

_CHECKSUM_EXCLUDE_KEYS = frozenset({"checksum", "signature", "signed_at"})


def canonical_json(contract: SkillContract) -> str:
    """Produce canonical JSON for checksum computation.

    Excludes self-referencing trust fields (checksum, signature, signed_at)
    to avoid circular dependency. Result is NFC-normalized for Unicode stability.
    """
    raw = contract.to_dict()
    trust = dict(raw.get("trust", {}))
    for key in _CHECKSUM_EXCLUDE_KEYS:
        trust.pop(key, None)
    raw["trust"] = trust
    canonical = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return unicodedata.normalize("NFC", canonical)


def compute_checksum(contract: SkillContract) -> str:
    """SHA-256 of canonical contract JSON (excluding self-referencing trust fields).

    Returns lowercase hex digest string.
    """
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────
# Permission-derived computations (v3 plan section 7.5)
# ──────────────────────────────────────────────────────────────


def compute_package_type(permissions: PermissionsConfig) -> str:
    """Determine package_type automatically from permissions.

    NOT user self-declaration. System computes based on what the skill requests.

    Hierarchy (highest wins, no early-return masking):
      privileged_plugin > code_plugin > tool_workflow > declarative_skill > local_skill

    Policy: history_access is always at least declarative_skill (never local_skill).
    Policy: tools with wildcard ("*") escalate to code_plugin.
    Policy: memory_read with wildcard ("*") escalates to declarative_skill (minimum).

    See Access/Risk/Package consistency matrix in security docs.
    """
    # --- privileged_plugin (highest) ---
    if permissions.secrets_access:
        return "privileged_plugin"
    # --- code_plugin ---
    if permissions.network_access.enabled or permissions.file_access.enabled:
        return "code_plugin"
    if permissions.tools and "*" in permissions.tools:
        return "code_plugin"
    # --- tool_workflow ---
    if permissions.tools:
        return "tool_workflow"
    # --- declarative_skill ---
    if (
        permissions.memory_read
        or permissions.memory_write
        or permissions.history_access.enabled
    ):
        return "declarative_skill"
    # --- local_skill (lowest) ---
    return "local_skill"


def compute_risk_level(permissions: PermissionsConfig) -> str:
    """Determine risk_level automatically from permissions.

    Default 'unknown' is resolved to a concrete level based on
    what the skill declares it needs.

    MONOTONIC: High checks first, Medium checks second, Low last.
    A low-risk permission can never mask a higher-risk permission.
    Adding a permission to a skill can never downgrade its risk level.

    Power-permission rule: any wildcard ("*"), "all_chats", "all",
    "root", "home" scope MUST be high-risk. These scopes grant
    broad access that requires explicit user approval.

    See Access/Risk/Package consistency matrix in security docs.
    """
    # --- HIGH first (all high-risk checks before any return) ---
    if permissions.secrets_access:
        return "high"
    if permissions.network_access.enabled or permissions.file_access.enabled:
        return "high"
    if permissions.tools and "*" in permissions.tools:
        return "high"
    if permissions.history_access.enabled and _has_power_scope(
        permissions.history_access.scopes
    ):
        return "high"
    # --- MEDIUM next ---
    if permissions.tools:
        return "medium"
    if permissions.memory_write:
        return "medium"
    if permissions.memory_read and "*" in permissions.memory_read:
        return "medium"
    # --- LOW last ---
    if permissions.memory_read:
        return "low"
    if permissions.history_access.enabled:
        return "low"
    return "low"  # No permissions = safe = low


# Power-permission scope detection
_POWER_SCOPES = frozenset({"*", "all", "all_chats", "root", "home"})


def _has_power_scope(scopes: tuple[str, ...]) -> bool:
    """Return True if any scope in the tuple is a power-permission scope."""
    return bool(_POWER_SCOPES.intersection(scopes))


# ──────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────


def new_skill_id() -> str:
    """Generate a new unique skill ID."""
    return f"skill_{uuid4().hex}"


def now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def iter_user_text_fields(contract: SkillContract) -> list[tuple[str, str]]:
    """Yield all user-controlled text fields from a SkillContract.

    Returns (field_label, value) pairs for every field that could contain
    user-authored (or attacker-authored on JSON install) text content.
    Shared by LearnFlowService and SkillInstaller for the One Safety Gate:
    the PrivacyPipeline MUST scan every field returned here.

    The canonical claim ("when I say <trigger>, <instruction>") is built
    by the caller from phrases[0] + instruction. This function returns the
    ADDITIONAL fields that must also be scanned.

    Fields covered:
      - name
      - activation.phrases (ALL, not just [0])
      - execution.instruction
      - tags (each tag)
      - intent.label
      - intent.positive_examples (each)
      - intent.negative_examples (each)

    Args:
        contract: The SkillContract to extract fields from.

    Returns:
        List of (field_label, text_value) tuples. Empty strings are excluded.
    """
    fields: list[tuple[str, str]] = []

    if contract.name and contract.name.strip():
        fields.append(("name", contract.name))

    for i, phrase in enumerate(contract.activation.phrases):
        if phrase and phrase.strip():
            fields.append((f"phrases[{i}]", phrase))

    if contract.execution.instruction and contract.execution.instruction.strip():
        fields.append(("instruction", contract.execution.instruction))

    for i, tag in enumerate(contract.tags):
        if tag and tag.strip():
            fields.append((f"tags[{i}]", tag))

    if contract.intent.label and contract.intent.label.strip():
        fields.append(("intent.label", contract.intent.label))

    for i, ex in enumerate(contract.intent.positive_examples):
        if ex and ex.strip():
            fields.append((f"intent.positive_examples[{i}]", ex))

    for i, ex in enumerate(contract.intent.negative_examples):
        if ex and ex.strip():
            fields.append((f"intent.negative_examples[{i}]", ex))

    return fields


def create_minimal_contract(
    *,
    name: str,
    phrases: tuple[str, ...],
    instruction: str,
    origin: str = "local_learn",
    hypothesis_id: Optional[str] = None,
) -> SkillContract:
    """Create a minimal valid contract with sane defaults.

    Useful for /learn flow. Sets id, timestamps, activation, execution.
    """
    ts = now_iso()
    return SkillContract(
        id=new_skill_id(),
        name=name,
        hypothesis_id=hypothesis_id,
        created_at=ts,
        updated_at=ts,
        activation=ActivationConfig(phrases=phrases),
        execution=ExecutionConfig(instruction=instruction),
        origin=origin,
    )


# ──────────────────────────────────────────────────────────────
# Serialization internals
# ──────────────────────────────────────────────────────────────


def _dataclass_to_dict(obj) -> dict | list | str | int | float | bool | None:
    """Recursively convert a frozen dataclass to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for fname in obj.__dataclass_fields__:
            val = getattr(obj, fname)
            result[fname] = _dataclass_to_dict(val)
        return result
    if isinstance(obj, tuple):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


def _dict_to_contract(data: dict) -> SkillContract:
    """Recursively build a SkillContract from a plain dict."""
    if not isinstance(data, dict):
        raise ContractDeserializationError(
            f"SkillContract root must be object, got {type(data).__name__}"
        )

    # Build nested configs (all sections go through _ensure_dict to reject non-dict types)
    activation = _build_activation(_ensure_dict(data.get("activation"), "activation"))
    intent = _build_intent(_ensure_dict(data.get("intent"), "intent"))
    execution = _build_execution(_ensure_dict(data.get("execution"), "execution"))
    memory_policy = _build_memory_policy(
        _ensure_dict(data.get("memory_policy"), "memory_policy")
    )
    confirmation = _build_confirmation(
        _ensure_dict(data.get("confirmation"), "confirmation")
    )
    safety = _build_safety(_ensure_dict(data.get("safety"), "safety"))
    priority = _build_priority(_ensure_dict(data.get("priority"), "priority"))
    scope = _build_scope(_ensure_dict(data.get("scope"), "scope"))
    observability = _build_observability(
        _ensure_dict(data.get("observability"), "observability")
    )
    lifecycle = _build_lifecycle(_ensure_dict(data.get("lifecycle"), "lifecycle"))
    trust = _build_trust(_ensure_dict(data.get("trust"), "trust"))
    permissions = _build_permissions(
        _ensure_dict(data.get("permissions"), "permissions")
    )
    store_meta = _build_store_meta(_ensure_dict(data.get("store_meta"), "store_meta"))
    audit = _build_audit(_ensure_dict(data.get("audit"), "audit"))

    slots = _ensure_list(data.get("slots", []), "slots")
    provider_hints = _ensure_list(data.get("provider_hints", []), "provider_hints")
    tags = _ensure_str_list(data.get("tags", []), "tags")

    # Top-level version fields: strict int, no bool, no string
    schema_version = _ensure_int_min(
        data.get("schema_version", 2), "schema_version", minimum=1
    )
    contract_version = _ensure_int_min(
        data.get("contract_version", 1), "contract_version", minimum=1
    )

    # Top-level string fields: strict str, no int/bool/list
    id_ = _ensure_str(data.get("id", ""), "id")
    name = _ensure_str(data.get("name", ""), "name")
    hypothesis_id = _ensure_optional_str(data.get("hypothesis_id"), "hypothesis_id")
    created_by = _ensure_str(data.get("created_by", "user"), "created_by")
    created_at = _ensure_str(data.get("created_at", ""), "created_at")
    updated_at = _ensure_str(data.get("updated_at", ""), "updated_at")
    migration_status = _ensure_str(
        data.get("migration_status", "current"), "migration_status"
    )
    review_status = _ensure_str(
        data.get("review_status", "unreviewed"), "review_status"
    )
    risk_level = _ensure_str(data.get("risk_level", "unknown"), "risk_level")
    origin = _ensure_str(data.get("origin", "local_learn"), "origin")

    return SkillContract(
        schema_version=schema_version,
        contract_version=contract_version,
        id=id_,
        name=name,
        hypothesis_id=hypothesis_id,
        created_by=created_by,
        created_at=created_at,
        updated_at=updated_at,
        migration_status=migration_status,
        activation=activation,
        intent=intent,
        slots=tuple(slots),
        execution=execution,
        provider_hints=tuple(provider_hints),
        memory_policy=memory_policy,
        confirmation=confirmation,
        safety=safety,
        priority=priority,
        scope=scope,
        observability=observability,
        lifecycle=lifecycle,
        tags=tuple(tags),
        trust=trust,
        permissions=permissions,
        review_status=review_status,
        risk_level=risk_level,
        store_meta=store_meta,
        audit=audit,
        origin=origin,
    )


# ──────────────────────────────────────────────────────────────
# Strict deserialization validators
# ──────────────────────────────────────────────────────────────


def _ensure_bool(value, field_name: str) -> bool:
    """Ensure value is a real Python bool. Rejects str, int, None, etc.

    This is critical for security: ``"false"`` is truthy in Python,
    so accepting strings would silently bypass safety/permission logic.
    Only ``True`` and ``False`` are accepted.
    """
    if not isinstance(value, bool):
        raise ContractDeserializationError(
            f"{field_name} must be bool, got {type(value).__name__}"
        )
    return value


def _ensure_str(value, field_name: str) -> str:
    """Ensure value is a real Python str. Rejects int, bool, list, etc."""
    if not isinstance(value, str):
        raise ContractDeserializationError(
            f"{field_name} must be str, got {type(value).__name__}"
        )
    return value


def _ensure_optional_str(value, field_name: str) -> str | None:
    """Ensure value is str or None. Rejects int, bool, list, etc."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractDeserializationError(
            f"{field_name} must be str or null, got {type(value).__name__}"
        )
    return value


def _ensure_list(value, field_name: str) -> list:
    """Ensure value is a list/tuple, never a string or other iterable.

    Raises ContractDeserializationError if value is a string or wrong type.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raise ContractDeserializationError(
            f"{field_name} must be list, got str: '{value[:50]}'"
        )
    if not isinstance(value, (list, tuple)):
        raise ContractDeserializationError(
            f"{field_name} must be list, got {type(value).__name__}"
        )
    return list(value)


def _ensure_int_min(value, field_name: str, *, minimum: int) -> int:
    """Ensure value is an int >= minimum. Rejects bool (bool is subclass of int)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractDeserializationError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < minimum:
        raise ContractDeserializationError(
            f"{field_name} must be >= {minimum}, got {value}"
        )
    return value


def _ensure_float_range(
    value, field_name: str, *, low: float = 0.0, high: float = 1.0
) -> float:
    """Ensure value is a float in [low, high]. Rejects bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractDeserializationError(
            f"{field_name} must be number, got {type(value).__name__}"
        )
    fval = float(value)
    if fval < low or fval > high:
        raise ContractDeserializationError(
            f"{field_name} must be in [{low}, {high}], got {fval}"
        )
    return fval


def _ensure_str_list(value, field_name: str) -> list[str]:
    """Ensure value is a list of strings. Rejects non-str elements.

    Raises ContractDeserializationError if any element is not a string.
    """
    items = _ensure_list(value, field_name)
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise ContractDeserializationError(
                f"{field_name}[{i}] must be str, got {type(item).__name__}"
            )
    return items


def _ensure_dict(value, field_name: str) -> dict:
    """Ensure value is a dict (JSON object). None is treated as missing section (returns {}).

    Non-dict types (str, int, list, bool) raise ContractDeserializationError.
    This prevents malformed JSON sections from being silently dropped to defaults.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ContractDeserializationError(
            f"{field_name} must be object, got {type(value).__name__}"
        )
    return value


# ──────────────────────────────────────────────────────────────
# Nested config builders (from dict)
# ──────────────────────────────────────────────────────────────


def _build_activation(d: dict) -> ActivationConfig:
    norm_d = _ensure_dict(d.get("normalization"), "activation.normalization")
    normalization = NormalizationConfig(
        case_fold=_ensure_bool(
            norm_d.get("case_fold", True), "normalization.case_fold"
        ),
        german_ss_equivalence=_ensure_bool(
            norm_d.get("german_ss_equivalence", True),
            "normalization.german_ss_equivalence",
        ),
        ignore_trailing_punctuation=_ensure_bool(
            norm_d.get("ignore_trailing_punctuation", True),
            "normalization.ignore_trailing_punctuation",
        ),
    )
    phrases = _ensure_str_list(d.get("phrases", []), "activation.phrases")
    conditions = _ensure_str_list(d.get("conditions", []), "activation.conditions")
    cooldown = _ensure_int_min(
        d.get("cooldown_seconds", 0), "activation.cooldown_seconds", minimum=0
    )
    return ActivationConfig(
        kind=_ensure_str(d.get("kind", "shortcut"), "activation.kind"),
        mode=_ensure_str(d.get("mode", "exact_phrase"), "activation.mode"),
        phrases=tuple(phrases),
        normalization=normalization,
        match_scope=_ensure_str(
            d.get("match_scope", "whole_message"), "activation.match_scope"
        ),
        conditions=tuple(conditions),
        cooldown_seconds=cooldown,
    )


def _build_intent(d: dict) -> IntentConfig:
    pos = _ensure_str_list(d.get("positive_examples", []), "intent.positive_examples")
    neg = _ensure_str_list(d.get("negative_examples", []), "intent.negative_examples")
    return IntentConfig(
        label=_ensure_str(d.get("label", ""), "intent.label"),
        positive_examples=tuple(pos),
        negative_examples=tuple(neg),
    )


def _build_execution(d: dict) -> ExecutionConfig:
    timeout = _ensure_int_min(
        d.get("timeout_seconds", 30), "execution.timeout_seconds", minimum=1
    )
    max_tools = _ensure_int_min(
        d.get("max_tool_calls", 0), "execution.max_tool_calls", minimum=0
    )
    return ExecutionConfig(
        type=_ensure_str(d.get("type", "llm_instruction"), "execution.type"),
        instruction=_ensure_str(d.get("instruction", ""), "execution.instruction"),
        timeout_seconds=timeout,
        max_tool_calls=max_tools,
    )


def _build_memory_policy(d: dict) -> MemoryPolicyConfig:
    read_d = _ensure_dict(d.get("read"), "memory_policy.read")
    write_d = _ensure_dict(d.get("write"), "memory_policy.write")
    allowed_stores = _ensure_str_list(
        read_d.get("allowed_stores", []), "memory_policy.read.allowed_stores"
    )
    blocked_stores = _ensure_str_list(
        read_d.get("blocked_stores", ["secrets", "health", "raw_finance"]),
        "memory_policy.read.blocked_stores",
    )
    max_items = _ensure_int_min(
        read_d.get("max_items", 10), "memory_policy.read.max_items", minimum=1
    )
    read = MemoryReadConfig(
        enabled=_ensure_bool(
            read_d.get("enabled", False), "memory_policy.read.enabled"
        ),
        allowed_stores=tuple(allowed_stores),
        blocked_stores=tuple(blocked_stores),
        max_items=max_items,
    )
    raw_ttl = write_d.get("ttl_days")
    if raw_ttl is not None:
        ttl_days = _ensure_int_min(raw_ttl, "memory_policy.write.ttl_days", minimum=0)
    else:
        ttl_days = None
    write = MemoryWriteConfig(
        enabled=_ensure_bool(
            write_d.get("enabled", False), "memory_policy.write.enabled"
        ),
        target_store=_ensure_optional_str(
            write_d.get("target_store"), "memory_policy.write.target_store"
        ),
        ttl_days=ttl_days,
    )
    return MemoryPolicyConfig(
        read=read,
        write=write,
        conflict_handling=_ensure_str(
            d.get("conflict_handling", "ignore"), "memory_policy.conflict_handling"
        ),
    )


def _build_confirmation(d: dict) -> ConfirmationConfig:
    thresh_d = _ensure_dict(d.get("thresholds"), "confirmation.thresholds")
    auto_exec = _ensure_float_range(
        thresh_d.get("auto_execute", 0.95), "confirmation.thresholds.auto_execute"
    )
    ask_conf = _ensure_float_range(
        thresh_d.get("ask_confirm", 0.75), "confirmation.thresholds.ask_confirm"
    )
    reject_val = _ensure_float_range(
        thresh_d.get("reject", 0.50), "confirmation.thresholds.reject"
    )
    thresholds = ConfirmationThresholds(
        auto_execute=auto_exec,
        ask_confirm=ask_conf,
        reject=reject_val,
    )
    return ConfirmationConfig(
        mode=_ensure_str(d.get("mode", "confidence_gated"), "confirmation.mode"),
        thresholds=thresholds,
        destructive_always_confirm=_ensure_bool(
            d.get("destructive_always_confirm", True),
            "confirmation.destructive_always_confirm",
        ),
    )


def _build_safety(d: dict) -> SafetyConfig:
    allowed_tools = _ensure_str_list(d.get("allowed_tools", []), "safety.allowed_tools")
    max_tool_calls_per_exec = _ensure_int_min(
        d.get("max_tool_calls_per_execution", 5),
        "safety.max_tool_calls_per_execution",
        minimum=0,
    )
    return SafetyConfig(
        requires_secret_scan=_ensure_bool(
            d.get("requires_secret_scan", True), "safety.requires_secret_scan"
        ),
        pii_risk=_ensure_str(d.get("pii_risk", "low"), "safety.pii_risk"),
        allow_tools=_ensure_bool(d.get("allow_tools", False), "safety.allow_tools"),
        allowed_tools=tuple(allowed_tools),
        max_tool_calls_per_execution=max_tool_calls_per_exec,
        instruction_sanitized=_ensure_bool(
            d.get("instruction_sanitized", True), "safety.instruction_sanitized"
        ),
    )


def _build_priority(d: dict) -> PriorityConfig:
    rank = _ensure_int_min(d.get("rank", 80), "priority.rank", minimum=0)
    beats = _ensure_str_list(
        d.get("beats", ["memory_conflict", "style_preference"]), "priority.beats"
    )
    loses_to = _ensure_str_list(
        d.get("loses_to", ["safety_policy", "explicit_user_command"]),
        "priority.loses_to",
    )
    return PriorityConfig(
        rank=rank,
        beats=tuple(beats),
        loses_to=tuple(loses_to),
    )


def _build_scope(d: dict) -> ScopeConfig:
    channels = _ensure_str_list(d.get("channels", ["telegram"]), "scope.channels")
    providers = _ensure_str_list(d.get("providers", ["all"]), "scope.providers")
    return ScopeConfig(
        user_scope=_ensure_str(d.get("user_scope", "single_user"), "scope.user_scope"),
        workspace=_ensure_str(d.get("workspace", "global"), "scope.workspace"),
        channels=tuple(channels),
        providers=tuple(providers),
    )


def _build_observability(d: dict) -> ObservabilityConfig:
    success_count = _ensure_int_min(
        d.get("success_count", 0), "observability.success_count", minimum=0
    )
    decline_count = _ensure_int_min(
        d.get("decline_count", 0), "observability.decline_count", minimum=0
    )
    average_confidence = _ensure_float_range(
        d.get("average_confidence", 0.0),
        "observability.average_confidence",
        low=0.0,
        high=1.0,
    )
    return ObservabilityConfig(
        track_success=_ensure_bool(
            d.get("track_success", True), "observability.track_success"
        ),
        track_declines=_ensure_bool(
            d.get("track_declines", True), "observability.track_declines"
        ),
        last_triggered_at=_ensure_optional_str(
            d.get("last_triggered_at"), "observability.last_triggered_at"
        ),
        success_count=success_count,
        decline_count=decline_count,
        average_confidence=average_confidence,
    )


def _build_lifecycle(d: dict) -> LifecycleConfig:
    return LifecycleConfig(
        status=_ensure_str(d.get("status", "confirmed"), "lifecycle.status"),
        editable=_ensure_bool(d.get("editable", True), "lifecycle.editable"),
        decay=_ensure_str(d.get("decay", "immune"), "lifecycle.decay"),
        last_schema_migration=_ensure_optional_str(
            d.get("last_schema_migration"), "lifecycle.last_schema_migration"
        ),
    )


def _build_trust(d: dict) -> TrustConfig:
    return TrustConfig(
        signature=_ensure_optional_str(d.get("signature"), "trust.signature"),
        signature_algorithm=_ensure_optional_str(
            d.get("signature_algorithm"), "trust.signature_algorithm"
        ),
        checksum=_ensure_optional_str(d.get("checksum"), "trust.checksum"),
        signed_at=_ensure_optional_str(d.get("signed_at"), "trust.signed_at"),
        author_id=_ensure_optional_str(d.get("author_id"), "trust.author_id"),
    )


def _build_permissions(d: dict) -> PermissionsConfig:
    net_d = _ensure_dict(d.get("network_access"), "permissions.network_access")
    file_d = _ensure_dict(d.get("file_access"), "permissions.file_access")
    hist_d = _ensure_dict(d.get("history_access"), "permissions.history_access")
    tools = _ensure_str_list(d.get("tools", []), "permissions.tools")
    mem_read = _ensure_str_list(d.get("memory_read", []), "permissions.memory_read")
    mem_write = _ensure_str_list(d.get("memory_write", []), "permissions.memory_write")
    net_domains = _ensure_str_list(
        net_d.get("domains", []), "permissions.network_access.domains"
    )
    file_scopes = _ensure_str_list(
        file_d.get("scopes", []), "permissions.file_access.scopes"
    )
    hist_scopes = _ensure_str_list(
        hist_d.get("scopes", []), "permissions.history_access.scopes"
    )
    return PermissionsConfig(
        tools=tuple(tools),
        memory_read=tuple(mem_read),
        memory_write=tuple(mem_write),
        network_access=NetworkAccessConfig(
            enabled=_ensure_bool(
                net_d.get("enabled", False), "permissions.network_access.enabled"
            ),
            domains=tuple(net_domains),
        ),
        file_access=FileAccessConfig(
            enabled=_ensure_bool(
                file_d.get("enabled", False), "permissions.file_access.enabled"
            ),
            scopes=tuple(file_scopes),
        ),
        history_access=HistoryAccessConfig(
            enabled=_ensure_bool(
                hist_d.get("enabled", False), "permissions.history_access.enabled"
            ),
            scopes=tuple(hist_scopes),
        ),
        secrets_access=_ensure_bool(
            d.get("secrets_access", False), "permissions.secrets_access"
        ),
    )


def _build_store_meta(d: dict) -> StoreMetaConfig:
    return StoreMetaConfig(
        package_type=_ensure_str(
            d.get("package_type", "local_skill"), "store_meta.package_type"
        ),
        license=_ensure_str(d.get("license", "personal"), "store_meta.license"),
        store_listing_id=_ensure_optional_str(
            d.get("store_listing_id"), "store_meta.store_listing_id"
        ),
        required_axolent_version=_ensure_optional_str(
            d.get("required_axolent_version"), "store_meta.required_axolent_version"
        ),
        manifest_version=_ensure_optional_str(
            d.get("manifest_version"), "store_meta.manifest_version"
        ),
        permissions_version=_ensure_optional_str(
            d.get("permissions_version"), "store_meta.permissions_version"
        ),
    )


def _build_audit(d: dict) -> AuditConfig:
    return AuditConfig(
        capability_broker_enabled=_ensure_bool(
            d.get("capability_broker_enabled", True), "audit.capability_broker_enabled"
        ),
    )
