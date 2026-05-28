"""T1 Tests: SkillContract dataclass, serialization, canonical JSON, checksum.

Coverage:
  U1:  Roundtrip serialization (to_dict/from_dict, to_json/from_json)
  U16: risk_level computation from permissions
  U17: package_type computation from permissions
  U18: Checksum determinism (same input = same hash)
  U19: Checksum detects tampering (changed instruction = new hash)
       Checksum excludes self-referencing trust fields (K3)
       Checksum NFC normalization stability
       canonical_json idempotency
"""

from __future__ import annotations

import json

import pytest

from application.skill_compression.skill_contract import (
    ActivationConfig,
    AuditConfig,
    ConfirmationConfig,
    ConfirmationThresholds,
    ContractDeserializationError,
    ExecutionConfig,
    FileAccessConfig,
    HistoryAccessConfig,
    IntentConfig,
    LifecycleConfig,
    MemoryPolicyConfig,
    MemoryReadConfig,
    MemoryWriteConfig,
    NetworkAccessConfig,
    NormalizationConfig,
    ObservabilityConfig,
    PermissionsConfig,
    PriorityConfig,
    SafetyConfig,
    ScopeConfig,
    SkillContract,
    StoreMetaConfig,
    TrustConfig,
    canonical_json,
    compute_checksum,
    compute_package_type,
    compute_risk_level,
    create_minimal_contract,
    new_skill_id,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_contract() -> SkillContract:
    """A minimal valid contract for testing."""
    return create_minimal_contract(
        name="Test Skill",
        phrases=("test", "testing"),
        instruction="Reply with a test message",
    )


@pytest.fixture
def full_contract() -> SkillContract:
    """A fully populated contract with all fields set."""
    return SkillContract(
        schema_version=2,
        contract_version=3,
        id="skill_abc123",
        name="Full Skill",
        hypothesis_id="hyp_xyz",
        created_by="user",
        created_at="2026-05-28T10:00:00+00:00",
        updated_at="2026-05-28T12:00:00+00:00",
        migration_status="current",
        activation=ActivationConfig(
            kind="shortcut",
            mode="exact_phrase",
            phrases=("weiss", "weiß"),
            normalization=NormalizationConfig(
                case_fold=True,
                german_ss_equivalence=True,
                ignore_trailing_punctuation=True,
            ),
            match_scope="whole_message",
            conditions=("has_context",),
            cooldown_seconds=5,
        ),
        intent=IntentConfig(
            label="color_query",
            positive_examples=("what color",),
            negative_examples=("what time",),
        ),
        slots=(),
        execution=ExecutionConfig(
            type="llm_instruction",
            instruction="Reply with 3 colors",
            timeout_seconds=60,
            max_tool_calls=2,
        ),
        provider_hints=(),
        memory_policy=MemoryPolicyConfig(
            read=MemoryReadConfig(
                enabled=True,
                allowed_stores=("long_term_facts",),
                blocked_stores=("secrets", "health"),
                max_items=5,
            ),
            write=MemoryWriteConfig(
                enabled=True,
                target_store="past_outputs",
                ttl_days=30,
            ),
            conflict_handling="inject_if_relevant",
        ),
        confirmation=ConfirmationConfig(
            mode="confidence_gated",
            thresholds=ConfirmationThresholds(
                auto_execute=0.90,
                ask_confirm=0.70,
                reject=0.40,
            ),
            destructive_always_confirm=False,
        ),
        safety=SafetyConfig(
            requires_secret_scan=True,
            pii_risk="medium",
            allow_tools=True,
            allowed_tools=("web_search",),
            max_tool_calls_per_execution=3,
            instruction_sanitized=True,
        ),
        priority=PriorityConfig(
            rank=90,
            beats=("style_preference",),
            loses_to=("safety_policy",),
        ),
        scope=ScopeConfig(
            user_scope="single_user",
            workspace="project_a",
            channels=("telegram", "web"),
            providers=("claude", "openai"),
        ),
        observability=ObservabilityConfig(
            track_success=True,
            track_declines=True,
            last_triggered_at="2026-05-28T11:00:00+00:00",
            success_count=10,
            decline_count=2,
            average_confidence=0.85,
        ),
        lifecycle=LifecycleConfig(
            status="active",
            editable=True,
            decay="immune",
            last_schema_migration="2026-05-28",
        ),
        tags=("productivity", "colors"),
        trust=TrustConfig(
            signature=None,
            signature_algorithm=None,
            checksum=None,
            signed_at=None,
            author_id="author_123",
        ),
        permissions=PermissionsConfig(
            tools=("web_search",),
            memory_read=("long_term_facts",),
            memory_write=("past_outputs",),
            network_access=NetworkAccessConfig(enabled=False),
            file_access=FileAccessConfig(enabled=False),
            history_access=HistoryAccessConfig(enabled=False),
            secrets_access=False,
        ),
        review_status="reviewed",
        risk_level="medium",
        store_meta=StoreMetaConfig(
            package_type="tool_workflow",
            license="MIT",
            store_listing_id=None,
            required_axolent_version="0.9.0",
            manifest_version="1",
            permissions_version="1",
        ),
        audit=AuditConfig(capability_broker_enabled=True),
        origin="local_learn",
    )


# ──────────────────────────────────────────────────────────────
# U1: Roundtrip serialization
# ──────────────────────────────────────────────────────────────


class TestRoundtrip:
    """U1: SkillContract serialization/deserialization with all v3 fields."""

    def test_to_dict_from_dict_roundtrip_minimal(self, minimal_contract):
        d = minimal_contract.to_dict()
        restored = SkillContract.from_dict(d)
        assert restored.name == minimal_contract.name
        assert restored.activation.phrases == minimal_contract.activation.phrases
        assert restored.execution.instruction == minimal_contract.execution.instruction
        assert restored.origin == "local_learn"
        assert restored.risk_level == "unknown"

    def test_to_dict_from_dict_roundtrip_full(self, full_contract):
        d = full_contract.to_dict()
        restored = SkillContract.from_dict(d)
        # Identity fields
        assert restored.id == full_contract.id
        assert restored.name == full_contract.name
        assert restored.schema_version == full_contract.schema_version
        assert restored.contract_version == full_contract.contract_version
        assert restored.hypothesis_id == full_contract.hypothesis_id
        # Activation
        assert restored.activation.phrases == ("weiss", "weiß")
        assert restored.activation.normalization.german_ss_equivalence is True
        assert restored.activation.cooldown_seconds == 5
        # Execution
        assert restored.execution.instruction == "Reply with 3 colors"
        assert restored.execution.timeout_seconds == 60
        # Memory policy
        assert restored.memory_policy.read.enabled is True
        assert restored.memory_policy.read.allowed_stores == ("long_term_facts",)
        assert restored.memory_policy.write.target_store == "past_outputs"
        # Trust/Permissions/StoreMeta
        assert restored.trust.author_id == "author_123"
        assert restored.permissions.tools == ("web_search",)
        assert restored.permissions.memory_read == ("long_term_facts",)
        assert restored.store_meta.license == "MIT"
        assert restored.store_meta.required_axolent_version == "0.9.0"
        # Audit
        assert restored.audit.capability_broker_enabled is True
        # Origin
        assert restored.origin == "local_learn"

    def test_to_json_from_json_roundtrip(self, full_contract):
        json_str = full_contract.to_json()
        restored = SkillContract.from_json(json_str)
        assert restored.id == full_contract.id
        assert restored.activation.phrases == full_contract.activation.phrases
        assert restored.permissions.tools == full_contract.permissions.tools

    def test_to_dict_produces_lists_not_tuples(self, minimal_contract):
        """JSON serialization should use lists, not tuples."""
        d = minimal_contract.to_dict()
        assert isinstance(d["activation"]["phrases"], list)
        assert isinstance(d["tags"], list)
        assert isinstance(d["permissions"]["tools"], list)

    def test_from_dict_with_unknown_keys_ignored(self):
        """Forward compatibility: unknown keys are silently ignored."""
        data = {
            "id": "skill_test",
            "name": "Test",
            "future_field": "ignored",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        c = SkillContract.from_dict(data)
        assert c.id == "skill_test"
        assert c.name == "Test"

    def test_from_dict_with_missing_keys_uses_defaults(self):
        """Missing keys use dataclass defaults."""
        c = SkillContract.from_dict({})
        assert c.schema_version == 2
        assert c.origin == "local_learn"
        assert c.risk_level == "unknown"
        assert c.permissions.secrets_access is False

    def test_frozen_immutability(self, minimal_contract):
        """SkillContract must be frozen (immutable)."""
        with pytest.raises(AttributeError):
            minimal_contract.name = "changed"  # type: ignore[misc]

    def test_nested_frozen(self, minimal_contract):
        """Nested config objects must also be frozen."""
        with pytest.raises(AttributeError):
            minimal_contract.activation.kind = "intent"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────
# U16: risk_level computation
# ──────────────────────────────────────────────────────────────


class TestRiskLevelComputation:
    """U16: compute_risk_level returns correct values based on permissions."""

    def test_no_permissions_is_low(self):
        p = PermissionsConfig()
        assert compute_risk_level(p) == "low"

    def test_memory_read_only_is_low(self):
        p = PermissionsConfig(memory_read=("long_term_facts",))
        assert compute_risk_level(p) == "low"

    def test_history_access_is_low(self):
        p = PermissionsConfig(history_access=HistoryAccessConfig(enabled=True))
        assert compute_risk_level(p) == "low"

    def test_tools_is_medium(self):
        p = PermissionsConfig(tools=("web_search",))
        assert compute_risk_level(p) == "medium"

    def test_memory_write_is_medium(self):
        p = PermissionsConfig(memory_write=("past_outputs",))
        assert compute_risk_level(p) == "medium"

    def test_network_access_is_high(self):
        p = PermissionsConfig(
            network_access=NetworkAccessConfig(
                enabled=True, domains=("api.example.com",)
            )
        )
        assert compute_risk_level(p) == "high"

    def test_file_access_is_high(self):
        p = PermissionsConfig(file_access=FileAccessConfig(enabled=True))
        assert compute_risk_level(p) == "high"

    def test_secrets_access_is_high(self):
        p = PermissionsConfig(secrets_access=True)
        assert compute_risk_level(p) == "high"


# ──────────────────────────────────────────────────────────────
# U17: package_type computation
# ──────────────────────────────────────────────────────────────


class TestPackageTypeComputation:
    """U17: compute_package_type returns correct values."""

    def test_no_permissions_is_local_skill(self):
        p = PermissionsConfig()
        assert compute_package_type(p) == "local_skill"

    def test_memory_read_is_declarative(self):
        p = PermissionsConfig(memory_read=("long_term_facts",))
        assert compute_package_type(p) == "declarative_skill"

    def test_tools_is_tool_workflow(self):
        p = PermissionsConfig(tools=("web_search",))
        assert compute_package_type(p) == "tool_workflow"

    def test_network_access_is_code_plugin(self):
        p = PermissionsConfig(network_access=NetworkAccessConfig(enabled=True))
        assert compute_package_type(p) == "code_plugin"

    def test_file_access_is_code_plugin(self):
        p = PermissionsConfig(file_access=FileAccessConfig(enabled=True))
        assert compute_package_type(p) == "code_plugin"

    def test_secrets_access_is_privileged(self):
        p = PermissionsConfig(secrets_access=True)
        assert compute_package_type(p) == "privileged_plugin"

    def test_hierarchy_secrets_wins_over_network(self):
        """secrets_access trumps network_access in tier hierarchy."""
        p = PermissionsConfig(
            secrets_access=True,
            network_access=NetworkAccessConfig(enabled=True),
        )
        assert compute_package_type(p) == "privileged_plugin"

    def test_hierarchy_network_wins_over_tools(self):
        """network_access trumps tools in tier hierarchy."""
        p = PermissionsConfig(
            network_access=NetworkAccessConfig(enabled=True),
            tools=("web_search",),
        )
        assert compute_package_type(p) == "code_plugin"


# ──────────────────────────────────────────────────────────────
# U18: Checksum determinism
# ──────────────────────────────────────────────────────────────


class TestChecksumDeterminism:
    """U18: Same contract produces same checksum, every time."""

    def test_same_contract_same_checksum(self, minimal_contract):
        c1 = compute_checksum(minimal_contract)
        c2 = compute_checksum(minimal_contract)
        assert c1 == c2

    def test_checksum_is_64_char_hex(self, minimal_contract):
        checksum = compute_checksum(minimal_contract)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_canonical_json_idempotent(self, minimal_contract):
        """Two calls to canonical_json produce identical output."""
        j1 = canonical_json(minimal_contract)
        j2 = canonical_json(minimal_contract)
        assert j1 == j2

    def test_canonical_json_is_valid_json(self, minimal_contract):
        """canonical_json output is parseable JSON."""
        cj = canonical_json(minimal_contract)
        parsed = json.loads(cj)
        assert isinstance(parsed, dict)

    def test_canonical_json_sorted_keys(self, minimal_contract):
        """canonical_json uses sorted keys."""
        cj = canonical_json(minimal_contract)
        parsed = json.loads(cj)
        keys = list(parsed.keys())
        assert keys == sorted(keys)


# ──────────────────────────────────────────────────────────────
# U19: Checksum detects tampering
# ──────────────────────────────────────────────────────────────


class TestChecksumTampering:
    """U19: Changed content produces different checksum."""

    def test_changed_instruction_changes_checksum(self, minimal_contract):
        from dataclasses import replace

        modified = replace(
            minimal_contract,
            execution=ExecutionConfig(instruction="TAMPERED instruction"),
        )
        assert compute_checksum(minimal_contract) != compute_checksum(modified)

    def test_changed_name_changes_checksum(self, minimal_contract):
        from dataclasses import replace

        modified = replace(minimal_contract, name="Different Name")
        assert compute_checksum(minimal_contract) != compute_checksum(modified)

    def test_changed_phrases_changes_checksum(self, minimal_contract):
        from dataclasses import replace

        modified = replace(
            minimal_contract,
            activation=ActivationConfig(phrases=("different",)),
        )
        assert compute_checksum(minimal_contract) != compute_checksum(modified)


# ──────────────────────────────────────────────────────────────
# K3: Checksum excludes self-referencing trust fields
# ──────────────────────────────────────────────────────────────


class TestChecksumExcludesSelfRef:
    """K3: checksum/signature/signed_at do not affect checksum value."""

    def test_checksum_field_excluded(self, minimal_contract):
        from dataclasses import replace

        with_checksum = replace(
            minimal_contract,
            trust=TrustConfig(checksum="a" * 64),
        )
        without_checksum = replace(
            minimal_contract,
            trust=TrustConfig(checksum=None),
        )
        assert compute_checksum(with_checksum) == compute_checksum(without_checksum)

    def test_signature_field_excluded(self, minimal_contract):
        """signature is excluded, but signature_algorithm is NOT excluded
        (it describes the algorithm, not the per-signing value).
        So we test: changing only signature does not change checksum."""
        from dataclasses import replace
        import base64

        sig = base64.b64encode(b"test_sig").decode()
        with_sig = replace(
            minimal_contract,
            trust=TrustConfig(signature=sig),
        )
        without_sig = replace(
            minimal_contract,
            trust=TrustConfig(signature=None),
        )
        assert compute_checksum(with_sig) == compute_checksum(without_sig)

    def test_signed_at_excluded(self, minimal_contract):
        from dataclasses import replace

        with_ts = replace(
            minimal_contract,
            trust=TrustConfig(signed_at="2026-05-28T10:00:00Z"),
        )
        without_ts = replace(
            minimal_contract,
            trust=TrustConfig(signed_at=None),
        )
        assert compute_checksum(with_ts) == compute_checksum(without_ts)

    def test_author_id_IS_included(self, minimal_contract):
        """author_id is NOT in the exclusion set, so it affects checksum."""
        from dataclasses import replace

        with_author = replace(
            minimal_contract,
            trust=TrustConfig(author_id="author_123"),
        )
        without_author = replace(
            minimal_contract,
            trust=TrustConfig(author_id=None),
        )
        assert compute_checksum(with_author) != compute_checksum(without_author)

    def test_nfc_normalization(self, minimal_contract):
        """Unicode NFC normalization ensures stable checksums."""
        from dataclasses import replace

        # Use a name with potential NFC/NFD difference
        c1 = replace(minimal_contract, name="Grüße")  # NFC: single codepoint
        c2 = replace(minimal_contract, name="Grüße")  # NFD: base + combining
        # After NFC normalization in canonical_json, these should produce the same checksum
        # because NFC("ü") == "ü"
        assert compute_checksum(c1) == compute_checksum(c2)


# ──────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────


class TestFactoryHelpers:
    """Test new_skill_id and create_minimal_contract."""

    def test_new_skill_id_format(self):
        sid = new_skill_id()
        assert sid.startswith("skill_")
        assert len(sid) > 10

    def test_new_skill_id_unique(self):
        ids = {new_skill_id() for _ in range(100)}
        assert len(ids) == 100

    def test_create_minimal_contract_has_required_fields(self):
        c = create_minimal_contract(
            name="My Skill",
            phrases=("trigger",),
            instruction="Do something",
        )
        assert c.id.startswith("skill_")
        assert c.name == "My Skill"
        assert c.activation.phrases == ("trigger",)
        assert c.execution.instruction == "Do something"
        assert c.created_at != ""
        assert c.updated_at != ""
        assert c.origin == "local_learn"
        assert c.schema_version == 2

    def test_create_minimal_contract_with_origin(self):
        c = create_minimal_contract(
            name="Installed Skill",
            phrases=("install",),
            instruction="Installed action",
            origin="manual_install",
        )
        assert c.origin == "manual_install"

    def test_create_minimal_contract_with_hypothesis_id(self):
        c = create_minimal_contract(
            name="Migrated Skill",
            phrases=("migrate",),
            instruction="Migrated action",
            hypothesis_id="hyp_legacy_123",
        )
        assert c.hypothesis_id == "hyp_legacy_123"


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: numeric fields reject wrong types
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerNumericFields:
    """Strict-Deserializer: numerische Felder muessen int sein, Strings werden abgelehnt."""

    def test_cooldown_seconds_as_string_rejected(self):
        data = {"activation": {"cooldown_seconds": "fast"}}
        with pytest.raises(
            ContractDeserializationError, match="cooldown_seconds.*must be int"
        ):
            SkillContract.from_dict(data)

    def test_timeout_seconds_as_string_rejected(self):
        data = {"execution": {"timeout_seconds": "fast"}}
        with pytest.raises(
            ContractDeserializationError, match="timeout_seconds.*must be int"
        ):
            SkillContract.from_dict(data)

    def test_max_tool_calls_as_string_rejected(self):
        data = {"execution": {"max_tool_calls": "many"}}
        with pytest.raises(
            ContractDeserializationError, match="max_tool_calls.*must be int"
        ):
            SkillContract.from_dict(data)

    def test_max_tool_calls_per_execution_as_string_rejected(self):
        data = {"safety": {"max_tool_calls_per_execution": "many"}}
        with pytest.raises(
            ContractDeserializationError,
            match="max_tool_calls_per_execution.*must be int",
        ):
            SkillContract.from_dict(data)

    def test_priority_rank_as_string_rejected(self):
        data = {"priority": {"rank": "high"}}
        with pytest.raises(
            ContractDeserializationError, match="priority.rank.*must be int"
        ):
            SkillContract.from_dict(data)

    def test_cooldown_seconds_as_float_rejected(self):
        data = {"activation": {"cooldown_seconds": 3.5}}
        with pytest.raises(
            ContractDeserializationError, match="cooldown_seconds.*must be int"
        ):
            SkillContract.from_dict(data)

    def test_timeout_seconds_as_none_rejected(self):
        data = {"execution": {"timeout_seconds": None}}
        with pytest.raises(
            ContractDeserializationError, match="timeout_seconds.*must be int"
        ):
            SkillContract.from_dict(data)

    def test_rank_as_list_rejected(self):
        data = {"priority": {"rank": [1, 2]}}
        with pytest.raises(
            ContractDeserializationError, match="priority.rank.*must be int"
        ):
            SkillContract.from_dict(data)


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: string-list fields reject wrong element types
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerStringListFields:
    """Strict-Deserializer: String-Listen muessen ausschliesslich str-Elemente enthalten."""

    def test_allowed_tools_with_int_element_rejected(self):
        data = {"safety": {"allowed_tools": [123]}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.allowed_tools.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_phrases_with_int_element_rejected(self):
        data = {"activation": {"phrases": [123, "valid"]}}
        with pytest.raises(
            ContractDeserializationError,
            match="activation.phrases.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_tags_with_dict_element_rejected(self):
        data = {"tags": [{"key": "val"}]}
        with pytest.raises(
            ContractDeserializationError, match=r"tags\[0\].*must be str.*got dict"
        ):
            SkillContract.from_dict(data)

    def test_conditions_with_int_element_rejected(self):
        data = {"activation": {"conditions": [42]}}
        with pytest.raises(
            ContractDeserializationError,
            match="activation.conditions.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_positive_examples_with_bool_rejected(self):
        data = {"intent": {"positive_examples": [True]}}
        with pytest.raises(
            ContractDeserializationError,
            match="intent.positive_examples.*must be str.*got bool",
        ):
            SkillContract.from_dict(data)

    def test_negative_examples_with_list_rejected(self):
        data = {"intent": {"negative_examples": [["nested"]]}}
        with pytest.raises(
            ContractDeserializationError,
            match="intent.negative_examples.*must be str.*got list",
        ):
            SkillContract.from_dict(data)

    def test_beats_with_int_element_rejected(self):
        data = {"priority": {"beats": [99]}}
        with pytest.raises(
            ContractDeserializationError, match="priority.beats.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_loses_to_with_dict_element_rejected(self):
        data = {"priority": {"loses_to": [{"x": 1}]}}
        with pytest.raises(
            ContractDeserializationError,
            match="priority.loses_to.*must be str.*got dict",
        ):
            SkillContract.from_dict(data)

    def test_scope_channels_with_int_rejected(self):
        data = {"scope": {"channels": [1]}}
        with pytest.raises(
            ContractDeserializationError, match="scope.channels.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_scope_providers_with_float_rejected(self):
        data = {"scope": {"providers": [3.14]}}
        with pytest.raises(
            ContractDeserializationError,
            match="scope.providers.*must be str.*got float",
        ):
            SkillContract.from_dict(data)

    def test_permissions_tools_with_int_rejected(self):
        data = {"permissions": {"tools": [42]}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.tools.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_permissions_memory_read_with_int_rejected(self):
        data = {"permissions": {"memory_read": [1]}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.memory_read.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_permissions_memory_write_with_none_rejected(self):
        data = {"permissions": {"memory_write": [None]}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.memory_write.*must be str.*got NoneType",
        ):
            SkillContract.from_dict(data)

    def test_permissions_network_domains_with_int_rejected(self):
        data = {"permissions": {"network_access": {"domains": [8080]}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.network_access.domains.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_permissions_file_scopes_with_dict_rejected(self):
        data = {"permissions": {"file_access": {"scopes": [{"path": "/tmp"}]}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.file_access.scopes.*must be str.*got dict",
        ):
            SkillContract.from_dict(data)

    def test_permissions_history_scopes_with_int_rejected(self):
        data = {"permissions": {"history_access": {"scopes": [99]}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.history_access.scopes.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_allowed_stores_with_int_rejected(self):
        data = {"memory_policy": {"read": {"allowed_stores": [1]}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read.allowed_stores.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_blocked_stores_with_bool_rejected(self):
        data = {"memory_policy": {"read": {"blocked_stores": [False]}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read.blocked_stores.*must be str.*got bool",
        ):
            SkillContract.from_dict(data)

    def test_valid_string_lists_still_accepted(self):
        """Sicherheitstest: korrekte String-Listen funktionieren weiterhin."""
        data = {
            "activation": {"phrases": ["hello", "hi"], "conditions": ["has_context"]},
            "intent": {
                "positive_examples": ["what time"],
                "negative_examples": ["nope"],
            },
            "safety": {"allowed_tools": ["web_search"]},
            "priority": {"beats": ["x"], "loses_to": ["y"]},
            "scope": {"channels": ["telegram"], "providers": ["claude"]},
            "permissions": {
                "tools": ["calc"],
                "memory_read": ["facts"],
                "memory_write": ["output"],
                "network_access": {"domains": ["example.com"]},
                "file_access": {"scopes": ["/tmp"]},
                "history_access": {"scopes": ["recent"]},
            },
            "memory_policy": {
                "read": {"allowed_stores": ["store1"], "blocked_stores": ["secrets"]},
            },
            "tags": ["tag1", "tag2"],
        }
        c = SkillContract.from_dict(data)
        assert c.activation.phrases == ("hello", "hi")
        assert c.tags == ("tag1", "tag2")
        assert c.permissions.tools == ("calc",)
        assert c.safety.allowed_tools == ("web_search",)


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: memory_policy numeric fields
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerMemoryPolicyNumeric:
    """Strict-Deserializer: memory_policy numerische Felder muessen korrekt typisiert sein."""

    def test_memory_max_items_as_string_rejected(self):
        data = {"memory_policy": {"read": {"max_items": "many"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read.max_items.*must be int",
        ):
            SkillContract.from_dict(data)

    def test_memory_max_items_negative_rejected(self):
        data = {"memory_policy": {"read": {"max_items": -1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read.max_items.*must be >= 1",
        ):
            SkillContract.from_dict(data)

    def test_memory_max_items_zero_rejected(self):
        data = {"memory_policy": {"read": {"max_items": 0}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read.max_items.*must be >= 1",
        ):
            SkillContract.from_dict(data)

    def test_memory_max_items_valid_accepted(self):
        data = {"memory_policy": {"read": {"max_items": 5}}}
        c = SkillContract.from_dict(data)
        assert c.memory_policy.read.max_items == 5

    def test_memory_ttl_days_as_string_rejected(self):
        data = {"memory_policy": {"write": {"ttl_days": "many"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.write.ttl_days.*must be int",
        ):
            SkillContract.from_dict(data)

    def test_memory_ttl_days_negative_rejected(self):
        data = {"memory_policy": {"write": {"ttl_days": -1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.write.ttl_days.*must be >= 0",
        ):
            SkillContract.from_dict(data)

    def test_memory_ttl_days_none_accepted(self):
        """None ist ein valider Wert fuer ttl_days (kein Ablauf)."""
        data = {"memory_policy": {"write": {"ttl_days": None}}}
        c = SkillContract.from_dict(data)
        assert c.memory_policy.write.ttl_days is None

    def test_memory_ttl_days_zero_accepted(self):
        data = {"memory_policy": {"write": {"ttl_days": 0}}}
        c = SkillContract.from_dict(data)
        assert c.memory_policy.write.ttl_days == 0


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: observability numeric fields
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerObservabilityNumeric:
    """Strict-Deserializer: observability numerische Felder muessen korrekt typisiert sein."""

    def test_observability_success_count_as_string_rejected(self):
        data = {"observability": {"success_count": "many"}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.success_count.*must be int",
        ):
            SkillContract.from_dict(data)

    def test_observability_success_count_negative_rejected(self):
        data = {"observability": {"success_count": -1}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.success_count.*must be >= 0",
        ):
            SkillContract.from_dict(data)

    def test_observability_decline_count_as_string_rejected(self):
        data = {"observability": {"decline_count": "many"}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.decline_count.*must be int",
        ):
            SkillContract.from_dict(data)

    def test_observability_decline_count_negative_rejected(self):
        data = {"observability": {"decline_count": -1}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.decline_count.*must be >= 0",
        ):
            SkillContract.from_dict(data)

    def test_observability_average_confidence_as_string_rejected(self):
        data = {"observability": {"average_confidence": "high"}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.average_confidence.*must be number",
        ):
            SkillContract.from_dict(data)

    def test_observability_average_confidence_above_one_rejected(self):
        data = {"observability": {"average_confidence": 1.5}}
        with pytest.raises(
            ContractDeserializationError,
            match=r"observability.average_confidence.*must be in \[0.0, 1.0\]",
        ):
            SkillContract.from_dict(data)

    def test_observability_average_confidence_below_zero_rejected(self):
        data = {"observability": {"average_confidence": -0.1}}
        with pytest.raises(
            ContractDeserializationError,
            match=r"observability.average_confidence.*must be in \[0.0, 1.0\]",
        ):
            SkillContract.from_dict(data)

    def test_observability_valid_values_accepted(self):
        data = {
            "observability": {
                "success_count": 10,
                "decline_count": 2,
                "average_confidence": 0.85,
            }
        }
        c = SkillContract.from_dict(data)
        assert c.observability.success_count == 10
        assert c.observability.decline_count == 2
        assert c.observability.average_confidence == 0.85


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: bool-as-int explicitly rejected
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerBoolAsIntRejected:
    """Strict-Deserializer: bool wird explizit als ungueltiger int/float-Typ abgelehnt."""

    def test_timeout_seconds_bool_rejected(self):
        data = {"execution": {"timeout_seconds": True}}
        with pytest.raises(
            ContractDeserializationError, match="timeout_seconds.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_priority_rank_bool_rejected(self):
        data = {"priority": {"rank": True}}
        with pytest.raises(
            ContractDeserializationError, match="priority.rank.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_threshold_auto_execute_bool_rejected(self):
        data = {"confirmation": {"thresholds": {"auto_execute": True}}}
        with pytest.raises(
            ContractDeserializationError, match="auto_execute.*must be number.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_max_items_bool_rejected(self):
        data = {"memory_policy": {"read": {"max_items": True}}}
        with pytest.raises(
            ContractDeserializationError, match="max_items.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_average_confidence_bool_rejected(self):
        data = {"observability": {"average_confidence": False}}
        with pytest.raises(
            ContractDeserializationError,
            match="average_confidence.*must be number.*got bool",
        ):
            SkillContract.from_dict(data)

    def test_cooldown_seconds_bool_rejected(self):
        data = {"activation": {"cooldown_seconds": False}}
        with pytest.raises(
            ContractDeserializationError,
            match="cooldown_seconds.*must be int.*got bool",
        ):
            SkillContract.from_dict(data)

    def test_max_tool_calls_bool_rejected(self):
        data = {"execution": {"max_tool_calls": True}}
        with pytest.raises(
            ContractDeserializationError, match="max_tool_calls.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_success_count_bool_rejected(self):
        data = {"observability": {"success_count": True}}
        with pytest.raises(
            ContractDeserializationError, match="success_count.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_ttl_days_bool_rejected(self):
        data = {"memory_policy": {"write": {"ttl_days": True}}}
        with pytest.raises(
            ContractDeserializationError, match="ttl_days.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: Boolean fields reject string/int types
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerBooleanFieldsRejectStrings:
    """Boolean-Felder muessen echte Python bool sein. Strings wie "false" sind truthy
    und wuerden Safety-/Permission-Logik unterwandern."""

    def test_safety_allow_tools_string_false_rejected(self):
        data = {"safety": {"allow_tools": "false"}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.allow_tools.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_safety_requires_secret_scan_string_true_rejected(self):
        data = {"safety": {"requires_secret_scan": "true"}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.requires_secret_scan.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_safety_instruction_sanitized_string_true_rejected(self):
        data = {"safety": {"instruction_sanitized": "true"}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.instruction_sanitized.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_permissions_network_access_enabled_string_false_rejected(self):
        data = {"permissions": {"network_access": {"enabled": "false"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.network_access.enabled.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_permissions_file_access_enabled_string_false_rejected(self):
        data = {"permissions": {"file_access": {"enabled": "false"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.file_access.enabled.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_permissions_history_access_enabled_string_false_rejected(self):
        data = {"permissions": {"history_access": {"enabled": "false"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.history_access.enabled.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_permissions_secrets_access_string_false_rejected(self):
        data = {"permissions": {"secrets_access": "false"}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.secrets_access.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_read_enabled_string_false_rejected(self):
        data = {"memory_policy": {"read": {"enabled": "false"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read.enabled.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_write_enabled_string_false_rejected(self):
        data = {"memory_policy": {"write": {"enabled": "false"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.write.enabled.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_confirmation_destructive_always_confirm_string_false_rejected(self):
        data = {"confirmation": {"destructive_always_confirm": "false"}}
        with pytest.raises(
            ContractDeserializationError,
            match="confirmation.destructive_always_confirm.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_observability_track_success_string_true_rejected(self):
        data = {"observability": {"track_success": "true"}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.track_success.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_observability_track_declines_string_true_rejected(self):
        data = {"observability": {"track_declines": "true"}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.track_declines.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_lifecycle_editable_string_false_rejected(self):
        data = {"lifecycle": {"editable": "false"}}
        with pytest.raises(
            ContractDeserializationError,
            match="lifecycle.editable.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_normalization_case_fold_string_true_rejected(self):
        data = {"activation": {"normalization": {"case_fold": "true"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="normalization.case_fold.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)

    def test_audit_capability_broker_enabled_string_true_rejected(self):
        data = {"audit": {"capability_broker_enabled": "true"}}
        with pytest.raises(
            ContractDeserializationError,
            match="audit.capability_broker_enabled.*must be bool.*got str",
        ):
            SkillContract.from_dict(data)


class TestStrictDeserializerBooleanFieldsRejectInts:
    """Boolean-Felder muessen auch int-Werte wie 0 und 1 ablehnen."""

    def test_safety_allow_tools_int_zero_rejected(self):
        data = {"safety": {"allow_tools": 0}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.allow_tools.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_read_enabled_int_one_rejected(self):
        data = {"memory_policy": {"read": {"enabled": 1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read.enabled.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_permissions_secrets_access_int_rejected(self):
        data = {"permissions": {"secrets_access": 1}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.secrets_access.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_lifecycle_editable_int_rejected(self):
        data = {"lifecycle": {"editable": 0}}
        with pytest.raises(
            ContractDeserializationError,
            match="lifecycle.editable.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_normalization_german_ss_equivalence_int_rejected(self):
        data = {"activation": {"normalization": {"german_ss_equivalence": 1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="normalization.german_ss_equivalence.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_confirmation_destructive_always_confirm_int_rejected(self):
        data = {"confirmation": {"destructive_always_confirm": 0}}
        with pytest.raises(
            ContractDeserializationError,
            match="confirmation.destructive_always_confirm.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_observability_track_success_int_rejected(self):
        data = {"observability": {"track_success": 1}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.track_success.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_safety_instruction_sanitized_int_rejected(self):
        data = {"safety": {"instruction_sanitized": 1}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.instruction_sanitized.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_safety_requires_secret_scan_int_rejected(self):
        data = {"safety": {"requires_secret_scan": 0}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.requires_secret_scan.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_permissions_network_access_enabled_int_rejected(self):
        data = {"permissions": {"network_access": {"enabled": 1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.network_access.enabled.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_permissions_file_access_enabled_int_rejected(self):
        data = {"permissions": {"file_access": {"enabled": 1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.file_access.enabled.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_permissions_history_access_enabled_int_rejected(self):
        data = {"permissions": {"history_access": {"enabled": 0}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.history_access.enabled.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_audit_capability_broker_enabled_int_rejected(self):
        data = {"audit": {"capability_broker_enabled": 0}}
        with pytest.raises(
            ContractDeserializationError,
            match="audit.capability_broker_enabled.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_write_enabled_int_rejected(self):
        data = {"memory_policy": {"write": {"enabled": 1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.write.enabled.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_normalization_ignore_trailing_punctuation_int_rejected(self):
        data = {"activation": {"normalization": {"ignore_trailing_punctuation": 0}}}
        with pytest.raises(
            ContractDeserializationError,
            match="normalization.ignore_trailing_punctuation.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)

    def test_observability_track_declines_int_rejected(self):
        data = {"observability": {"track_declines": 0}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.track_declines.*must be bool.*got int",
        ):
            SkillContract.from_dict(data)


class TestStrictDeserializerBooleanFieldsRejectNone:
    """Boolean-Felder muessen None ablehnen."""

    def test_safety_allow_tools_none_rejected(self):
        data = {"safety": {"allow_tools": None}}
        with pytest.raises(
            ContractDeserializationError,
            match="safety.allow_tools.*must be bool.*got NoneType",
        ):
            SkillContract.from_dict(data)

    def test_permissions_secrets_access_none_rejected(self):
        data = {"permissions": {"secrets_access": None}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.secrets_access.*must be bool.*got NoneType",
        ):
            SkillContract.from_dict(data)


class TestStrictDeserializerBooleanFieldsAcceptReal:
    """Sicherheitstest: echte bool-Werte muessen weiterhin funktionieren."""

    def test_safety_allow_tools_true_accepted(self):
        data = {"safety": {"allow_tools": True}}
        c = SkillContract.from_dict(data)
        assert c.safety.allow_tools is True

    def test_safety_allow_tools_false_accepted(self):
        data = {"safety": {"allow_tools": False}}
        c = SkillContract.from_dict(data)
        assert c.safety.allow_tools is False

    def test_all_bool_fields_real_bools_accepted(self):
        """All 17 boolean fields with real bools still work in a full roundtrip."""
        data = {
            "activation": {
                "normalization": {
                    "case_fold": False,
                    "german_ss_equivalence": False,
                    "ignore_trailing_punctuation": False,
                }
            },
            "audit": {"capability_broker_enabled": False},
            "confirmation": {"destructive_always_confirm": False},
            "memory_policy": {
                "read": {"enabled": True},
                "write": {"enabled": True},
            },
            "permissions": {
                "network_access": {"enabled": True},
                "file_access": {"enabled": True},
                "history_access": {"enabled": True},
                "secrets_access": True,
            },
            "observability": {
                "track_success": False,
                "track_declines": False,
            },
            "lifecycle": {"editable": False},
            "safety": {
                "requires_secret_scan": False,
                "allow_tools": True,
                "instruction_sanitized": False,
            },
        }
        c = SkillContract.from_dict(data)
        assert c.activation.normalization.case_fold is False
        assert c.activation.normalization.german_ss_equivalence is False
        assert c.activation.normalization.ignore_trailing_punctuation is False
        assert c.audit.capability_broker_enabled is False
        assert c.confirmation.destructive_always_confirm is False
        assert c.memory_policy.read.enabled is True
        assert c.memory_policy.write.enabled is True
        assert c.permissions.network_access.enabled is True
        assert c.permissions.file_access.enabled is True
        assert c.permissions.history_access.enabled is True
        assert c.permissions.secrets_access is True
        assert c.observability.track_success is False
        assert c.observability.track_declines is False
        assert c.lifecycle.editable is False
        assert c.safety.requires_secret_scan is False
        assert c.safety.allow_tools is True
        assert c.safety.instruction_sanitized is False


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: schema_version and contract_version
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerVersionFields:
    """schema_version und contract_version muessen echte ints sein."""

    def test_schema_version_as_string_rejected(self):
        data = {"schema_version": "2"}
        with pytest.raises(
            ContractDeserializationError, match="schema_version.*must be int.*got str"
        ):
            SkillContract.from_dict(data)

    def test_schema_version_as_bool_rejected(self):
        data = {"schema_version": True}
        with pytest.raises(
            ContractDeserializationError, match="schema_version.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_contract_version_as_string_rejected(self):
        data = {"contract_version": "1"}
        with pytest.raises(
            ContractDeserializationError, match="contract_version.*must be int.*got str"
        ):
            SkillContract.from_dict(data)

    def test_contract_version_as_bool_rejected(self):
        data = {"contract_version": True}
        with pytest.raises(
            ContractDeserializationError,
            match="contract_version.*must be int.*got bool",
        ):
            SkillContract.from_dict(data)

    def test_contract_version_zero_rejected(self):
        data = {"contract_version": 0}
        with pytest.raises(
            ContractDeserializationError, match="contract_version.*must be >= 1.*got 0"
        ):
            SkillContract.from_dict(data)

    def test_schema_version_zero_rejected(self):
        data = {"schema_version": 0}
        with pytest.raises(
            ContractDeserializationError, match="schema_version.*must be >= 1.*got 0"
        ):
            SkillContract.from_dict(data)

    def test_schema_version_valid_int_accepted(self):
        data = {"schema_version": 2}
        c = SkillContract.from_dict(data)
        assert c.schema_version == 2

    def test_contract_version_valid_int_accepted(self):
        data = {"contract_version": 3}
        c = SkillContract.from_dict(data)
        assert c.contract_version == 3

    def test_schema_version_as_float_rejected(self):
        data = {"schema_version": 2.0}
        with pytest.raises(
            ContractDeserializationError, match="schema_version.*must be int.*got float"
        ):
            SkillContract.from_dict(data)

    def test_contract_version_as_none_rejected(self):
        data = {"contract_version": None}
        with pytest.raises(
            ContractDeserializationError,
            match="contract_version.*must be int.*got NoneType",
        ):
            SkillContract.from_dict(data)


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: string fields reject wrong types
# ──────────────────────────────────────────────────────────────


class TestStrictDeserializerStringFields:
    """String-Felder muessen echte Python str sein."""

    def test_top_level_id_int_rejected(self):
        data = {"id": 123}
        with pytest.raises(
            ContractDeserializationError, match="id.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_top_level_name_int_rejected(self):
        data = {"name": 42}
        with pytest.raises(
            ContractDeserializationError, match="name.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_top_level_origin_int_rejected(self):
        data = {"origin": 1}
        with pytest.raises(
            ContractDeserializationError, match="origin.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_top_level_created_by_bool_rejected(self):
        data = {"created_by": True}
        with pytest.raises(
            ContractDeserializationError, match="created_by.*must be str.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_activation_kind_int_rejected(self):
        data = {"activation": {"kind": 1}}
        with pytest.raises(
            ContractDeserializationError, match="activation.kind.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_execution_type_bool_rejected(self):
        data = {"execution": {"type": True}}
        with pytest.raises(
            ContractDeserializationError, match="execution.type.*must be str.*got bool"
        ):
            SkillContract.from_dict(data)

    def test_lifecycle_status_int_rejected(self):
        data = {"lifecycle": {"status": 1}}
        with pytest.raises(
            ContractDeserializationError, match="lifecycle.status.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_safety_pii_risk_int_rejected(self):
        data = {"safety": {"pii_risk": 1}}
        with pytest.raises(
            ContractDeserializationError, match="safety.pii_risk.*must be str.*got int"
        ):
            SkillContract.from_dict(data)

    def test_scope_user_scope_bool_rejected(self):
        data = {"scope": {"user_scope": False}}
        with pytest.raises(
            ContractDeserializationError,
            match="scope.user_scope.*must be str.*got bool",
        ):
            SkillContract.from_dict(data)

    def test_store_meta_package_type_int_rejected(self):
        data = {"store_meta": {"package_type": 1}}
        with pytest.raises(
            ContractDeserializationError,
            match="store_meta.package_type.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_conflict_handling_int_rejected(self):
        data = {"memory_policy": {"conflict_handling": 1}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.conflict_handling.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_confirmation_mode_int_rejected(self):
        data = {"confirmation": {"mode": 1}}
        with pytest.raises(
            ContractDeserializationError,
            match="confirmation.mode.*must be str.*got int",
        ):
            SkillContract.from_dict(data)

    def test_hypothesis_id_int_rejected(self):
        data = {"hypothesis_id": 123}
        with pytest.raises(
            ContractDeserializationError,
            match="hypothesis_id.*must be str or null.*got int",
        ):
            SkillContract.from_dict(data)

    def test_hypothesis_id_none_accepted(self):
        data = {"hypothesis_id": None}
        c = SkillContract.from_dict(data)
        assert c.hypothesis_id is None

    def test_trust_author_id_int_rejected(self):
        data = {"trust": {"author_id": 123}}
        with pytest.raises(
            ContractDeserializationError,
            match="trust.author_id.*must be str or null.*got int",
        ):
            SkillContract.from_dict(data)

    def test_observability_last_triggered_at_int_rejected(self):
        data = {"observability": {"last_triggered_at": 12345}}
        with pytest.raises(
            ContractDeserializationError,
            match="observability.last_triggered_at.*must be str or null.*got int",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_write_target_store_int_rejected(self):
        data = {"memory_policy": {"write": {"target_store": 1}}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.write.target_store.*must be str or null.*got int",
        ):
            SkillContract.from_dict(data)


# ──────────────────────────────────────────────────────────────
# Codex Bypass Reproduktionen (Round 4 verification)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# Strict Deserializer: Section-type rejection (Round 5)
# Nested sections with wrong type must raise, not silently default
# ──────────────────────────────────────────────────────────────


class TestSectionTypeRejection:
    """Sections and subsections must be dict/object.
    String, int, list, bool values must raise ContractDeserializationError,
    NOT silently fall back to defaults."""

    def test_activation_string_rejected(self):
        data = {"activation": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="activation must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_intent_string_rejected(self):
        data = {"intent": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="intent must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_execution_string_rejected(self):
        data = {"execution": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="execution must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_string_rejected(self):
        data = {"memory_policy": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="memory_policy must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_read_string_rejected(self):
        data = {"memory_policy": {"read": "bad"}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read must be object, got str",
        ):
            SkillContract.from_dict(data)

    def test_memory_policy_write_string_rejected(self):
        data = {"memory_policy": {"write": "bad"}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.write must be object, got str",
        ):
            SkillContract.from_dict(data)

    def test_confirmation_string_rejected(self):
        data = {"confirmation": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="confirmation must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_confirmation_thresholds_string_rejected(self):
        data = {"confirmation": {"thresholds": "bad"}}
        with pytest.raises(
            ContractDeserializationError,
            match="confirmation.thresholds must be object, got str",
        ):
            SkillContract.from_dict(data)

    def test_safety_string_rejected(self):
        data = {"safety": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="safety must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_priority_string_rejected(self):
        data = {"priority": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="priority must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_scope_string_rejected(self):
        data = {"scope": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="scope must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_observability_string_rejected(self):
        data = {"observability": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="observability must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_lifecycle_string_rejected(self):
        data = {"lifecycle": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="lifecycle must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_trust_string_rejected(self):
        data = {"trust": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="trust must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_permissions_string_rejected(self):
        data = {"permissions": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="permissions must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_permissions_network_access_string_rejected(self):
        data = {"permissions": {"network_access": "bad"}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.network_access must be object, got str",
        ):
            SkillContract.from_dict(data)

    def test_permissions_file_access_string_rejected(self):
        data = {"permissions": {"file_access": "bad"}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.file_access must be object, got str",
        ):
            SkillContract.from_dict(data)

    def test_permissions_history_access_string_rejected(self):
        data = {"permissions": {"history_access": "bad"}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.history_access must be object, got str",
        ):
            SkillContract.from_dict(data)

    def test_store_meta_string_rejected(self):
        data = {"store_meta": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="store_meta must be object, got str"
        ):
            SkillContract.from_dict(data)

    def test_audit_string_rejected(self):
        data = {"audit": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="audit must be object, got str"
        ):
            SkillContract.from_dict(data)


class TestSectionTypeEdgeCases:
    """Edge cases: int, list, bool as section values must also be rejected."""

    def test_section_as_int_rejected(self):
        data = {"permissions": 123}
        with pytest.raises(
            ContractDeserializationError, match="permissions must be object, got int"
        ):
            SkillContract.from_dict(data)

    def test_section_as_list_rejected(self):
        data = {"permissions": []}
        with pytest.raises(
            ContractDeserializationError, match="permissions must be object, got list"
        ):
            SkillContract.from_dict(data)

    def test_section_as_bool_rejected(self):
        data = {"safety": True}
        with pytest.raises(
            ContractDeserializationError, match="safety must be object, got bool"
        ):
            SkillContract.from_dict(data)

    def test_subsection_as_int_rejected(self):
        data = {"permissions": {"network_access": 42}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.network_access must be object, got int",
        ):
            SkillContract.from_dict(data)

    def test_subsection_as_list_rejected(self):
        data = {"memory_policy": {"read": ["bad"]}}
        with pytest.raises(
            ContractDeserializationError,
            match="memory_policy.read must be object, got list",
        ):
            SkillContract.from_dict(data)

    def test_subsection_as_bool_rejected(self):
        data = {"confirmation": {"thresholds": False}}
        with pytest.raises(
            ContractDeserializationError,
            match="confirmation.thresholds must be object, got bool",
        ):
            SkillContract.from_dict(data)

    def test_activation_normalization_as_int_rejected(self):
        data = {"activation": {"normalization": 99}}
        with pytest.raises(
            ContractDeserializationError,
            match="activation.normalization must be object, got int",
        ):
            SkillContract.from_dict(data)


class TestSectionMissingUsesDefaults:
    """Missing sections (absent or None) must still use defaults, not error."""

    def test_missing_sections_use_defaults(self):
        """Contract with no optional sections at all is valid with defaults."""
        data = {}
        c = SkillContract.from_dict(data)
        assert c.safety == SafetyConfig()
        assert c.permissions == PermissionsConfig()
        assert c.trust == TrustConfig()
        assert c.audit == AuditConfig()
        assert c.store_meta == StoreMetaConfig()
        assert c.memory_policy == MemoryPolicyConfig()
        assert c.confirmation == ConfirmationConfig()
        assert c.priority == PriorityConfig()
        assert c.scope == ScopeConfig()
        assert c.observability == ObservabilityConfig()
        assert c.lifecycle == LifecycleConfig()
        assert c.activation == ActivationConfig()
        assert c.intent == IntentConfig()
        assert c.execution == ExecutionConfig()

    def test_explicit_none_section_uses_default(self):
        """Explicitly passing None for a section is treated as missing (= defaults)."""
        data = {
            "permissions": None,
            "safety": None,
            "trust": None,
            "audit": None,
            "store_meta": None,
            "memory_policy": None,
            "confirmation": None,
            "activation": None,
            "execution": None,
        }
        c = SkillContract.from_dict(data)
        assert c.permissions == PermissionsConfig()
        assert c.safety == SafetyConfig()
        assert c.trust == TrustConfig()
        assert c.audit == AuditConfig()
        assert c.store_meta == StoreMetaConfig()
        assert c.memory_policy == MemoryPolicyConfig()
        assert c.confirmation == ConfirmationConfig()
        assert c.activation == ActivationConfig()
        assert c.execution == ExecutionConfig()

    def test_explicit_none_subsection_uses_default(self):
        """None for nested subsections = defaults."""
        data = {
            "permissions": {
                "network_access": None,
                "file_access": None,
                "history_access": None,
            },
            "memory_policy": {
                "read": None,
                "write": None,
            },
            "confirmation": {
                "thresholds": None,
            },
            "activation": {
                "normalization": None,
            },
        }
        c = SkillContract.from_dict(data)
        assert c.permissions.network_access == NetworkAccessConfig()
        assert c.permissions.file_access == FileAccessConfig()
        assert c.permissions.history_access == HistoryAccessConfig()
        assert c.memory_policy.read == MemoryReadConfig()
        assert c.memory_policy.write == MemoryWriteConfig()
        assert c.confirmation.thresholds == ConfirmationThresholds()
        assert c.activation.normalization == NormalizationConfig()


class TestCodexSectionBypassReproductions:
    """Reproduces the 6 section-type bypasses from Codex Round 4 review.
    All 6 were ACCEPTED_BY_FROM_DICT before the fix. Now they must REJECT."""

    def test_safety_string_bypass(self):
        """Codex: safety_string ACCEPTED_BY_FROM_DICT valid=True"""
        data = {"safety": "bad"}
        with pytest.raises(ContractDeserializationError, match="safety must be object"):
            SkillContract.from_dict(data)

    def test_permissions_string_bypass(self):
        """Codex: permissions_string ACCEPTED_BY_FROM_DICT valid=True"""
        data = {"permissions": "bad"}
        with pytest.raises(
            ContractDeserializationError, match="permissions must be object"
        ):
            SkillContract.from_dict(data)

    def test_trust_string_bypass(self):
        """Codex: trust_string ACCEPTED_BY_FROM_DICT valid=True"""
        data = {"trust": "bad"}
        with pytest.raises(ContractDeserializationError, match="trust must be object"):
            SkillContract.from_dict(data)

    def test_memory_read_string_bypass(self):
        """Codex: memory_read_string ACCEPTED_BY_FROM_DICT valid=True"""
        data = {"memory_policy": {"read": "bad"}}
        with pytest.raises(
            ContractDeserializationError, match="memory_policy.read must be object"
        ):
            SkillContract.from_dict(data)

    def test_permission_network_string_bypass(self):
        """Codex: permission_network_string ACCEPTED_BY_FROM_DICT valid=True"""
        data = {"permissions": {"network_access": "bad"}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.network_access must be object",
        ):
            SkillContract.from_dict(data)

    def test_confirmation_thresholds_string_bypass(self):
        """Codex: confirmation_thresholds_string ACCEPTED_BY_FROM_DICT valid=True"""
        data = {"confirmation": {"thresholds": "bad"}}
        with pytest.raises(
            ContractDeserializationError, match="confirmation.thresholds must be object"
        ):
            SkillContract.from_dict(data)


class TestCodexBypassReproductions:
    """Reproduziert ALLE 7 Codex-Bypasses aus dem Recheck-Report."""

    def test_safety_allow_tools_string_false_with_tool_perm_rejected(self):
        """Codex: safety_allow_tools_string_false_with_tool_perm valid=True"""
        data = {
            "safety": {"allow_tools": "false"},
            "permissions": {"tools": ["web_search"]},
        }
        with pytest.raises(
            ContractDeserializationError, match="safety.allow_tools.*must be bool"
        ):
            SkillContract.from_dict(data)

    def test_network_enabled_string_false_rejected(self):
        """Codex: network_enabled_string_false valid=True"""
        data = {"permissions": {"network_access": {"enabled": "false"}}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.network_access.enabled.*must be bool",
        ):
            SkillContract.from_dict(data)

    def test_secrets_access_string_false_rejected(self):
        """Codex: secrets_access_string_false valid=True"""
        data = {"permissions": {"secrets_access": "false"}}
        with pytest.raises(
            ContractDeserializationError,
            match="permissions.secrets_access.*must be bool",
        ):
            SkillContract.from_dict(data)

    def test_contract_version_str_crashes_rejected(self):
        """Codex: contract_version_str="1" CRASHED TypeError"""
        data = {"contract_version": "1"}
        with pytest.raises(
            ContractDeserializationError, match="contract_version.*must be int.*got str"
        ):
            SkillContract.from_dict(data)

    def test_contract_version_bool_true_rejected(self):
        """Codex: contract_version_bool=True ACCEPTED"""
        data = {"contract_version": True}
        with pytest.raises(
            ContractDeserializationError,
            match="contract_version.*must be int.*got bool",
        ):
            SkillContract.from_dict(data)

    def test_schema_version_str_rejected(self):
        """Codex: schema_version_str="2" from_dict=ACCEPTED validate=False"""
        data = {"schema_version": "2"}
        with pytest.raises(
            ContractDeserializationError, match="schema_version.*must be int.*got str"
        ):
            SkillContract.from_dict(data)

    def test_schema_version_bool_rejected(self):
        """Codex: schema_version_bool=True from_dict=ACCEPTED validate=False"""
        data = {"schema_version": True}
        with pytest.raises(
            ContractDeserializationError, match="schema_version.*must be int.*got bool"
        ):
            SkillContract.from_dict(data)


class TestStrictDeserializerRootObjectEnforcement:
    """Root-level type enforcement: from_dict/from_json must reject non-object roots.

    Codex Round 5 Finding 1: _dict_to_contract() silently returned non-dict
    values (str, list, None, int) instead of raising ContractDeserializationError.
    Empty object {} must still produce a valid SkillContract with defaults."""

    def test_from_dict_string_root_rejected(self):
        with pytest.raises(
            ContractDeserializationError, match="root must be object, got str"
        ):
            SkillContract.from_dict("bad")

    def test_from_dict_list_root_rejected(self):
        with pytest.raises(
            ContractDeserializationError, match="root must be object, got list"
        ):
            SkillContract.from_dict([])

    def test_from_dict_none_root_rejected(self):
        with pytest.raises(
            ContractDeserializationError, match="root must be object, got NoneType"
        ):
            SkillContract.from_dict(None)

    def test_from_dict_int_root_rejected(self):
        with pytest.raises(
            ContractDeserializationError, match="root must be object, got int"
        ):
            SkillContract.from_dict(123)

    def test_from_json_string_root_rejected(self):
        with pytest.raises(
            ContractDeserializationError, match="root must be object, got str"
        ):
            SkillContract.from_json('"bad"')

    def test_from_json_list_root_rejected(self):
        with pytest.raises(
            ContractDeserializationError, match="root must be object, got list"
        ):
            SkillContract.from_json("[]")

    def test_from_json_null_root_rejected(self):
        with pytest.raises(
            ContractDeserializationError, match="root must be object, got NoneType"
        ):
            SkillContract.from_json("null")

    def test_from_dict_empty_object_uses_defaults(self):
        contract = SkillContract.from_dict({})
        assert isinstance(contract, SkillContract)
        assert contract.schema_version == 2
        assert contract.contract_version == 1

    def test_from_json_empty_object_uses_defaults(self):
        contract = SkillContract.from_json("{}")
        assert isinstance(contract, SkillContract)
        assert contract.schema_version == 2
        assert contract.contract_version == 1
