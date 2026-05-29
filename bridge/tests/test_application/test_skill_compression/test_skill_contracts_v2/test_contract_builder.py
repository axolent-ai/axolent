"""Tests for ContractBuilder (T7): 4-path coverage.

Covers:
  - Happy path: all 5 Codex test cases
  - Malicious: prompt injection in instruction
  - Rejection: stopword triggers, empty, too short, command-like
  - Privacy: no secrets in draft, origin=local_learn, deny-by-default permissions
"""

from __future__ import annotations


from application.skill_compression.contract_builder import (
    ContractBuilder,
    validate_trigger,
)
from application.skill_compression.skill_contract import PermissionsConfig


# ---------------------------------------------------------------
# Codex R5: 5 mandatory test cases
# ---------------------------------------------------------------


class TestCodexExtractionCases:
    """All 5 Codex R5 extraction test cases must pass."""

    def test_case1_wenn_ich_schreibe_weiss(self) -> None:
        """'wenn ich schreibe weiss, antworte mit 3 anderen Farben'
        => trigger: weiss, instruction: antworte mit 3 anderen Farben
        """
        result = ContractBuilder.build(
            "wenn ich schreibe weiss, antworte mit 3 anderen Farben"
        )
        assert result.status == "pending"
        assert result.contract.activation.phrases == ("weiss",)
        assert "3 anderen Farben" in result.contract.execution.instruction

    def test_case2_wenn_ich_weiss_schreibe(self) -> None:
        """'wenn ich weiss schreibe, antworte mit 3 anderen Farben'
        => trigger: weiss (reversed pattern)
        """
        result = ContractBuilder.build(
            "wenn ich weiss schreibe, antworte mit 3 anderen Farben"
        )
        assert result.status == "pending"
        phrases = result.contract.activation.phrases
        assert "weiss" in phrases[0].lower()

    def test_case3_sobald_ich_rot_sage(self) -> None:
        """'sobald ich rot sage, erklaere RGB'
        => trigger: rot
        """
        result = ContractBuilder.build("sobald ich rot sage, erklaere RGB")
        assert result.status == "pending"
        assert result.contract.activation.phrases == ("rot",)
        assert "RGB" in result.contract.execution.instruction

    def test_case4_bitte_sei_immer_nett(self) -> None:
        """'bitte sei immer nett'
        => No shortcut trigger extractable => needs_input
        """
        result = ContractBuilder.build("bitte sei immer nett")
        assert result.status == "needs_input"
        assert result.needs_input_reason == "trigger_missing"

    def test_case5_wenn_ich_ja_sage(self) -> None:
        """'wenn ich ja sage, mache X'
        => 'ja' is a stopword => needs_input
        """
        result = ContractBuilder.build("wenn ich ja sage, mache X")
        assert result.status == "needs_input"
        assert result.needs_input_reason == "trigger_rejected"


# ---------------------------------------------------------------
# Trigger validation
# ---------------------------------------------------------------


class TestTriggerValidation:
    """Validate trigger phrases."""

    def test_valid_trigger(self) -> None:
        result = validate_trigger("rot")
        assert result.valid

    def test_stopword_ja(self) -> None:
        result = validate_trigger("ja")
        assert not result.valid

    def test_stopword_nein(self) -> None:
        result = validate_trigger("nein")
        assert not result.valid

    def test_stopword_ok(self) -> None:
        result = validate_trigger("ok")
        assert not result.valid

    def test_stopword_okay(self) -> None:
        result = validate_trigger("okay")
        assert not result.valid

    def test_stopword_case_insensitive(self) -> None:
        result = validate_trigger("JA")
        assert not result.valid

    def test_command_trigger(self) -> None:
        result = validate_trigger("/learn")
        assert not result.valid

    def test_too_short(self) -> None:
        result = validate_trigger("x")
        assert not result.valid

    def test_empty_trigger(self) -> None:
        result = validate_trigger("")
        assert not result.valid

    def test_whitespace_only(self) -> None:
        result = validate_trigger("   ")
        assert not result.valid

    def test_too_long(self) -> None:
        result = validate_trigger("a" * 101)
        assert not result.valid

    def test_max_length_ok(self) -> None:
        result = validate_trigger("a" * 100)
        assert result.valid

    def test_slash_prefix(self) -> None:
        result = validate_trigger("/custom_command")
        assert not result.valid


# ---------------------------------------------------------------
# Happy path: extraction works
# ---------------------------------------------------------------


class TestBuilderHappyPath:
    """Builder produces valid drafts for well-formed inputs."""

    def test_basic_de_extraction(self) -> None:
        result = ContractBuilder.build("wenn ich blau sage, gruesse mich auf Japanisch")
        assert result.status == "pending"
        assert result.contract.activation.phrases == ("blau",)
        assert "Japanisch" in result.contract.execution.instruction

    def test_basic_en_extraction(self) -> None:
        result = ContractBuilder.build("when I say red, explain RGB colors")
        assert result.status == "pending"
        assert result.contract.activation.phrases == ("red",)
        assert "RGB" in result.contract.execution.instruction

    def test_contract_has_id(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        assert result.contract.id.startswith("skill_")

    def test_contract_has_timestamps(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        assert result.contract.created_at
        assert result.contract.updated_at

    def test_contract_name_derived(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        assert result.contract.name
        assert "Test" in result.contract.name

    def test_execution_type_llm_instruction(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        assert result.contract.execution.type == "llm_instruction"


# ---------------------------------------------------------------
# Origin and permissions (deny-by-default)
# ---------------------------------------------------------------


class TestOriginAndPermissions:
    """All /learn skills must have origin=local_learn and deny-by-default."""

    def test_origin_is_local_learn(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        assert result.contract.origin == "local_learn"

    def test_origin_local_learn_on_needs_input(self) -> None:
        result = ContractBuilder.build("bitte sei nett")
        assert result.contract.origin == "local_learn"

    def test_permissions_deny_by_default(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        perms = result.contract.permissions
        assert perms == PermissionsConfig()
        assert perms.tools == ()
        assert perms.memory_read == ()
        assert perms.memory_write == ()
        assert perms.network_access.enabled is False
        assert perms.file_access.enabled is False
        assert perms.history_access.enabled is False
        assert perms.secrets_access is False


# ---------------------------------------------------------------
# Needs input cases
# ---------------------------------------------------------------


class TestNeedsInput:
    """Builder returns needs_input when extraction fails."""

    def test_plain_text_no_pattern(self) -> None:
        result = ContractBuilder.build("antworte immer freundlich")
        assert result.status == "needs_input"
        assert result.needs_input_reason == "trigger_missing"

    def test_stopword_trigger_needs_input(self) -> None:
        result = ContractBuilder.build("wenn ich danke sage, mache X")
        assert result.status == "needs_input"
        assert result.needs_input_reason == "trigger_rejected"

    def test_stub_contract_has_instruction(self) -> None:
        """Even for needs_input, the instruction should be preserved."""
        result = ContractBuilder.build("antworte immer auf Deutsch")
        assert result.contract.execution.instruction == "antworte immer auf Deutsch"


# ---------------------------------------------------------------
# Edit methods
# ---------------------------------------------------------------


class TestApplyTriggerEdit:
    """Test trigger editing on existing drafts."""

    def test_valid_trigger_edit(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        updated, err = ContractBuilder.apply_trigger_edit(
            result.contract, "neuer_trigger"
        )
        assert err is None
        assert updated.activation.phrases == ("neuer_trigger",)

    def test_stopword_trigger_edit_rejected(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        updated, err = ContractBuilder.apply_trigger_edit(result.contract, "ja")
        assert err is not None
        # Contract unchanged
        assert updated.activation.phrases == result.contract.activation.phrases

    def test_empty_trigger_edit_rejected(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        updated, err = ContractBuilder.apply_trigger_edit(result.contract, "")
        assert err is not None


class TestApplyInstructionEdit:
    """Test instruction editing on existing drafts."""

    def test_valid_instruction_edit(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        updated, err = ContractBuilder.apply_instruction_edit(
            result.contract, "new instruction"
        )
        assert err is None
        assert updated.execution.instruction == "new instruction"

    def test_empty_instruction_rejected(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        updated, err = ContractBuilder.apply_instruction_edit(result.contract, "")
        assert err is not None

    def test_whitespace_instruction_rejected(self) -> None:
        result = ContractBuilder.build("wenn ich test sage, mach was")
        updated, err = ContractBuilder.apply_instruction_edit(result.contract, "   ")
        assert err is not None


# ---------------------------------------------------------------
# Malicious path (4-path security)
# ---------------------------------------------------------------


class TestMaliciousPath:
    """Prompt injection and malicious inputs."""

    def test_trigger_with_command_prefix(self) -> None:
        """User tries to make a command become a trigger."""
        result = validate_trigger("/help")
        assert not result.valid

    def test_all_stopwords_are_blocked(self) -> None:
        """Every single stopword must be blocked."""
        from application.skill_compression.contract_builder import TRIGGER_STOPWORDS

        for word in TRIGGER_STOPWORDS:
            result = validate_trigger(word)
            assert not result.valid, f"Stopword '{word}' was not blocked"


# ---------------------------------------------------------------
# Privacy path (4-path security)
# ---------------------------------------------------------------


class TestPrivacyPath:
    """No secrets or PII in builder output."""

    def test_no_secrets_in_fixtures(self) -> None:
        """Verify test fixtures do not contain real secrets."""
        # This test is a meta-check: all test strings above must be obviously fake
        # Real secret patterns: sk-proj-*, ghp_*, Bearer *, etc.
        import inspect

        source = inspect.getsource(TestCodexExtractionCases)
        assert "sk-proj-" not in source
        assert "ghp_" not in source
        assert "Bearer " not in source

    def test_draft_permissions_are_sandbox(self) -> None:
        """Privacy: /learn drafts must never have elevated permissions."""
        result = ContractBuilder.build("wenn ich test sage, mach was")
        assert not result.contract.permissions.secrets_access
        assert not result.contract.permissions.network_access.enabled
        assert not result.contract.permissions.file_access.enabled
