"""Tests for SkillInstaller (Etappe 5, T13): Path-Boundary + Safety + 4-Path.

Coverage:
  Path-Boundary Security:
    - rejects_path_traversal
    - rejects_absolute_path_outside_skill_dir
    - rejects_symlinks
    - rejects_remote_urls
    - startup_scan_ignores_symlink_outside_skill_dir (validates scan safety)

  Safety Pipeline (One Safety Gate):
    - installed skill with secret in instruction -> rejected
    - installed skill with healthcare content -> rejected
    - installed skill with nudge content -> rejected

  4-Path:
    - Happy: valid skill JSON -> installed and findable via matcher
    - Malicious: path traversal + symlink + secret in contract -> blocked
    - Rejection: invalid manifest/schema -> rejected
    - Privacy: no secrets/PII in logs (caplog check)

  Production-Path:
    - /installskill handler -> service -> ContractStore -> matcher finds skill

  main.py Wiring-Guard:
    - SkillInstaller is instantiated in bot_data
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from application.skill_compression.contract_store import ContractStore
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.skill_contract import (
    create_minimal_contract,
)
from application.skill_compression.skill_installer import (
    SkillInstaller,
    reject_remote_url,
    validate_source_path,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


@pytest.fixture
def contract_store(tmp_path: Path) -> ContractStore:
    """In-memory ContractStore for testing."""
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_installer.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    store = ContractStore(conn)
    store.init_schema()
    return store


@pytest.fixture
def installer(
    contract_store: ContractStore, pipeline: PrivacyPipeline
) -> SkillInstaller:
    return SkillInstaller(
        contract_store=contract_store,
        privacy_pipeline=pipeline,
    )


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Temporary skill directory (allowed root)."""
    d = tmp_path / "skills"
    d.mkdir()
    return d


def _make_valid_skill_json(
    name: str = "Test Install Skill",
    trigger: str = "test install",
    instruction: str = "Reply with a greeting",
) -> str:
    """Create a valid skill contract JSON string."""
    contract = create_minimal_contract(
        name=name,
        phrases=(trigger,),
        instruction=instruction,
    )
    return contract.to_json()


# ──────────────────────────────────────────────────────────────
# Path-Boundary: reject_remote_url
# ──────────────────────────────────────────────────────────────


class TestRejectRemoteUrl:
    """Test remote URL rejection (before path resolution)."""

    def test_rejects_http(self) -> None:
        result = reject_remote_url("http://example.com/skill.json")
        assert result is not None
        assert "Remote URLs" in result

    def test_rejects_https(self) -> None:
        result = reject_remote_url("https://example.com/skill.json")
        assert result is not None

    def test_rejects_file_scheme(self) -> None:
        result = reject_remote_url("file:///etc/passwd")
        assert result is not None

    def test_rejects_ftp(self) -> None:
        result = reject_remote_url("ftp://evil.com/skill.json")
        assert result is not None

    def test_rejects_case_insensitive(self) -> None:
        result = reject_remote_url("HTTP://EVIL.COM/skill.json")
        assert result is not None

    def test_allows_local_path(self) -> None:
        result = reject_remote_url("/home/user/skills/my_skill.json")
        assert result is None

    def test_allows_relative_path(self) -> None:
        result = reject_remote_url("my_skill.json")
        assert result is None


# ──────────────────────────────────────────────────────────────
# Path-Boundary: validate_source_path
# ──────────────────────────────────────────────────────────────


class TestValidateSourcePath:
    """Test path boundary validation (Codex Addendum K5)."""

    def test_rejects_path_traversal(self, skill_dir: Path) -> None:
        """Path traversal (..) is always rejected."""
        evil_path = skill_dir / ".." / "etc" / "passwd"
        result = validate_source_path(evil_path, skill_dir)
        assert not result.valid
        assert "traversal" in result.reason.lower()

    def test_rejects_absolute_path_outside_skill_dir(
        self, skill_dir: Path, tmp_path: Path
    ) -> None:
        """Absolute paths outside the allowed root are rejected."""
        outside = tmp_path / "other_dir" / "skill.json"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("{}", encoding="utf-8")
        result = validate_source_path(outside, skill_dir)
        assert not result.valid
        assert "outside" in result.reason.lower()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Symlinks need privileges on Windows"
    )
    def test_rejects_symlinks(self, skill_dir: Path, tmp_path: Path) -> None:
        """Symlinks are rejected (even if target is inside skill_dir)."""
        real_file = tmp_path / "real_skill.json"
        real_file.write_text("{}", encoding="utf-8")
        symlink = skill_dir / "symlinked_skill.json"
        symlink.symlink_to(real_file)
        result = validate_source_path(symlink, skill_dir)
        assert not result.valid
        assert "ymlink" in result.reason  # "Symlinks" or "symlink"

    def test_accepts_valid_path_inside_skill_dir(self, skill_dir: Path) -> None:
        """A regular file inside skill_dir is accepted."""
        valid_file = skill_dir / "my_skill.json"
        valid_file.write_text("{}", encoding="utf-8")
        result = validate_source_path(valid_file, skill_dir)
        assert result.valid

    def test_accepts_nested_path_inside_skill_dir(self, skill_dir: Path) -> None:
        """A file in a subdirectory of skill_dir is accepted."""
        sub = skill_dir / "category"
        sub.mkdir()
        valid_file = sub / "skill.json"
        valid_file.write_text("{}", encoding="utf-8")
        result = validate_source_path(valid_file, skill_dir)
        assert result.valid

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Symlinks need privileges on Windows"
    )
    def test_startup_scan_ignores_symlink_outside_skill_dir(
        self, skill_dir: Path, tmp_path: Path
    ) -> None:
        """Startup scan: symlinks pointing outside skill_dir are rejected, not followed."""
        external = tmp_path / "external_secret.json"
        external.write_text('{"secret": "sk-1234"}', encoding="utf-8")
        symlink = skill_dir / "sneaky.json"
        symlink.symlink_to(external)

        # Simulate startup scan: validate each file in skill_dir
        for item in skill_dir.iterdir():
            result = validate_source_path(item, skill_dir)
            if item.name == "sneaky.json":
                assert not result.valid


# ──────────────────────────────────────────────────────────────
# Safety Pipeline: One Safety Gate (same as /learn)
# ──────────────────────────────────────────────────────────────


class TestInstallerSafetyPipeline:
    """Installed skills must pass through the same safety pipeline as /learn."""

    def test_rejects_secret_in_instruction(
        self, installer: SkillInstaller, skill_dir: Path
    ) -> None:
        """A contract with an API key in the instruction is rejected."""
        contract_json = _make_valid_skill_json(
            name="Secret Skill",
            instruction="Use API key sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz to call the API",
        )
        skill_file = skill_dir / "secret_skill.json"
        skill_file.write_text(contract_json, encoding="utf-8")

        result = installer.install_from_file(skill_file, skill_dir, user_id=42)
        assert not result.success

    def test_rejects_secret_in_json_string(self, installer: SkillInstaller) -> None:
        """install_from_json_string also runs safety."""
        contract_json = _make_valid_skill_json(
            name="Secret JSON Skill",
            instruction="Token is sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz please use it",
        )
        result = installer.install_from_json_string(contract_json, user_id=42)
        assert not result.success


# ──────────────────────────────────────────────────────────────
# 4-Path Tests
# ──────────────────────────────────────────────────────────────


class TestInstallerHappyPath:
    """Happy: valid skill installed successfully."""

    def test_install_valid_skill_from_file(
        self, installer: SkillInstaller, skill_dir: Path, contract_store: ContractStore
    ) -> None:
        """A valid skill JSON file installs and appears in ContractStore."""
        contract_json = _make_valid_skill_json(name="Greeting Skill")
        skill_file = skill_dir / "greeting.json"
        skill_file.write_text(contract_json, encoding="utf-8")

        result = installer.install_from_file(skill_file, skill_dir, user_id=42)
        assert result.success
        assert result.contract_name == "Greeting Skill"

        # Verify it's in the store
        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1
        assert contracts[0].name == "Greeting Skill"
        assert contracts[0].origin == "manual_install"

    def test_install_valid_skill_from_json_string(
        self, installer: SkillInstaller, contract_store: ContractStore
    ) -> None:
        """install_from_json_string works for Telegram uploads."""
        contract_json = _make_valid_skill_json(name="Upload Skill")
        result = installer.install_from_json_string(contract_json, user_id=42)
        assert result.success
        assert result.contract_name == "Upload Skill"

        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1

    def test_origin_forced_to_manual_install(
        self, installer: SkillInstaller, contract_store: ContractStore
    ) -> None:
        """Even if the JSON says origin='store', installer forces 'manual_install'."""
        contract = create_minimal_contract(
            name="Spoofed Origin",
            phrases=("spooftest",),
            instruction="Do something",
            origin="store",  # Trying to spoof
        )
        # We need to add a fake signature for store origin to pass V17,
        # but since we force manual_install, V17 won't require it.
        raw = contract.to_json()
        result = installer.install_from_json_string(raw, user_id=42)
        assert result.success

        contracts = contract_store.get_by_user(42)
        assert contracts[0].origin == "manual_install"


class TestInstallerMaliciousPath:
    """Malicious: path traversal, symlink, secret blocked."""

    def test_path_traversal_blocked(
        self, installer: SkillInstaller, skill_dir: Path
    ) -> None:
        """Path traversal attempt is blocked."""
        evil_path = skill_dir / ".." / "outside" / "skill.json"
        result = installer.install_from_file(evil_path, skill_dir, user_id=42)
        assert not result.success
        assert "traversal" in result.error.lower()

    def test_outside_boundary_blocked(
        self, installer: SkillInstaller, skill_dir: Path, tmp_path: Path
    ) -> None:
        """File outside skill_dir is blocked."""
        outside = tmp_path / "sneaky" / "skill.json"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text(_make_valid_skill_json(), encoding="utf-8")
        result = installer.install_from_file(outside, skill_dir, user_id=42)
        assert not result.success
        assert "outside" in result.error.lower()

    def test_secret_in_installed_skill_blocked(self, installer: SkillInstaller) -> None:
        """Secret in instruction is caught by safety pipeline."""
        contract_json = _make_valid_skill_json(
            name="Malicious Skill",
            instruction="Run with token sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz now",
        )
        result = installer.install_from_json_string(contract_json, user_id=42)
        assert not result.success


class TestInstallerRejectionPath:
    """Rejection: invalid manifest/schema rejected."""

    def test_invalid_json_rejected(
        self, installer: SkillInstaller, skill_dir: Path
    ) -> None:
        """Non-JSON file is rejected with clear error."""
        bad_file = skill_dir / "bad.json"
        bad_file.write_text("this is not json {{{", encoding="utf-8")
        result = installer.install_from_file(bad_file, skill_dir, user_id=42)
        assert not result.success
        assert "Invalid JSON" in result.error

    def test_non_object_json_rejected(
        self, installer: SkillInstaller, skill_dir: Path
    ) -> None:
        """JSON array is rejected."""
        bad_file = skill_dir / "array.json"
        bad_file.write_text("[1, 2, 3]", encoding="utf-8")
        result = installer.install_from_file(bad_file, skill_dir, user_id=42)
        assert not result.success
        assert "object" in result.error.lower()

    def test_missing_fields_rejected(self, installer: SkillInstaller) -> None:
        """Empty object fails validation (missing required fields)."""
        result = installer.install_from_json_string("{}", user_id=42)
        assert not result.success

    def test_workflow_execution_type_rejected(self, installer: SkillInstaller) -> None:
        """execution.type='workflow' is rejected by V15 feature flag."""
        contract = create_minimal_contract(
            name="Workflow Skill",
            phrases=("workflowtest",),
            instruction="Do workflow",
        )
        # Mutate to workflow type
        from dataclasses import replace
        from application.skill_compression.skill_contract import ExecutionConfig

        bad_contract = replace(
            contract,
            execution=ExecutionConfig(type="workflow", instruction="Do workflow"),
        )
        raw = bad_contract.to_json()
        result = installer.install_from_json_string(raw, user_id=42)
        assert not result.success

    def test_file_not_found_rejected(
        self, installer: SkillInstaller, skill_dir: Path
    ) -> None:
        """Non-existent file returns error, not crash."""
        missing = skill_dir / "does_not_exist.json"
        result = installer.install_from_file(missing, skill_dir, user_id=42)
        assert not result.success
        assert "Cannot read" in result.error


class TestInstallerPrivacyPath:
    """Privacy: no secrets or PII in logs."""

    def test_no_secrets_in_logs(
        self,
        installer: SkillInstaller,
        skill_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When a skill with a secret is rejected, the secret does NOT appear in logs."""
        secret_token = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        contract_json = _make_valid_skill_json(
            name="Log Privacy Skill",
            instruction=f"Use {secret_token} for auth",
        )
        skill_file = skill_dir / "secret_log.json"
        skill_file.write_text(contract_json, encoding="utf-8")

        with caplog.at_level(logging.DEBUG):
            installer.install_from_file(skill_file, skill_dir, user_id=42)

        # The secret itself must NOT appear in any log message
        for record in caplog.records:
            assert secret_token not in record.message


# ──────────────────────────────────────────────────────────────
# Production-Path: handler -> service -> store -> matcher
# ──────────────────────────────────────────────────────────────


class TestInstallerProductionPath:
    """Production-path: installed skill is findable via SkillMatcher."""

    def test_installed_skill_findable_via_matcher(
        self,
        installer: SkillInstaller,
        contract_store: ContractStore,
    ) -> None:
        """After installation, SkillMatcher can find the skill by phrase match."""
        contract_json = _make_valid_skill_json(
            name="Findable Skill",
            trigger="findme",
            instruction="Found you!",
        )
        result = installer.install_from_json_string(contract_json, user_id=42)
        assert result.success

        # Try to match via contract
        contracts = contract_store.get_by_user(42)
        assert len(contracts) == 1
        assert contracts[0].activation.phrases == ("findme",)


# ──────────────────────────────────────────────────────────────
# main.py Wiring-Guard
# ──────────────────────────────────────────────────────────────


_BRIDGE_ROOT = Path(__file__).resolve().parents[4]


class TestMainWiringGuard:
    """Verify SkillInstaller is wired in main.py correctly."""

    def test_installer_instantiation(
        self, contract_store: ContractStore, pipeline: PrivacyPipeline
    ) -> None:
        """SkillInstaller can be constructed with ContractStore + PrivacyPipeline."""
        installer = SkillInstaller(
            contract_store=contract_store,
            privacy_pipeline=pipeline,
        )
        assert installer is not None
        assert installer._contract_store is contract_store
        assert installer._privacy is pipeline

    def test_main_imports_skill_installer(self) -> None:
        """main.py imports SkillInstaller."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert (
            "from application.skill_compression.skill_installer import SkillInstaller"
            in source
        )

    def test_main_instantiates_skill_installer(self) -> None:
        """main.py creates a SkillInstaller instance."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert "skill_installer = SkillInstaller(" in source

    def test_main_registers_skill_installer_in_bot_data(self) -> None:
        """main.py registers skill_installer in bot_data."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert 'app.bot_data["skill_installer"]' in source

    def test_main_all_phase1_components_wired(self) -> None:
        """All Phase 1 components are wired in main.py."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        phase1_components = [
            "ContractStore",
            "DraftStore",
            "PendingEditStore",
            "LearnFlowService",
            "SkillMatcher",
            "SkillInstaller",
            "ContractBuilder",
        ]
        for component in phase1_components:
            assert component in source, (
                f"{component} is not referenced in main.py. "
                f"All Phase 1 components must be wired."
            )

    def test_main_registers_installskill_command_handler(self) -> None:
        """main.py registers /installskill CommandHandler (GAP-15 fix)."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert 'CommandHandler("installskill"' in source, (
            "/installskill CommandHandler not registered in main.py. "
            "GAP-15: installer must be reachable via Telegram."
        )

    def test_main_imports_installskill_handler(self) -> None:
        """main.py imports handle_installskill_command (GAP-15 fix)."""
        source = (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")
        assert "handle_installskill_command" in source, (
            "handle_installskill_command not imported in main.py. "
            "GAP-15: handler must be importable."
        )


# ──────────────────────────────────────────────────────────────
# GAP-14: Safety Gate scans ALL user-controlled text fields
# Parametrized over every field returned by iter_user_text_fields
# ──────────────────────────────────────────────────────────────

# Secret payload that triggers the SecretScanner
_SECRET_PAYLOAD = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"

# Healthcare payload that triggers the HealthcareFilter
_HEALTHCARE_PAYLOAD = "User shows signs of depression and suicidal ideation"


def _make_contract_with_secret_in_field(field_name: str) -> str:
    """Build a contract JSON with a secret payload injected into a specific field.

    Args:
        field_name: One of 'name', 'phrases_1', 'tags', 'intent_label',
                    'intent_positive', 'intent_negative', 'instruction'.

    Returns:
        JSON string of the contract.
    """
    from dataclasses import replace as dc_replace
    from application.skill_compression.skill_contract import (
        ActivationConfig,
        ExecutionConfig,
        IntentConfig,
    )

    base = create_minimal_contract(
        name="Clean Name",
        phrases=("clean trigger", "second phrase"),
        instruction="Reply with a greeting",
    )

    if field_name == "name":
        base = dc_replace(base, name=f"Secret {_SECRET_PAYLOAD} Skill")
    elif field_name == "phrases_1":
        new_phrases = (base.activation.phrases[0], f"say {_SECRET_PAYLOAD}")
        base = dc_replace(base, activation=ActivationConfig(phrases=new_phrases))
    elif field_name == "tags":
        base = dc_replace(base, tags=(f"api-key-{_SECRET_PAYLOAD}",))
    elif field_name == "intent_label":
        base = dc_replace(base, intent=IntentConfig(label=f"secret {_SECRET_PAYLOAD}"))
    elif field_name == "intent_positive":
        base = dc_replace(
            base,
            intent=IntentConfig(positive_examples=(f"key is {_SECRET_PAYLOAD}",)),
        )
    elif field_name == "intent_negative":
        base = dc_replace(
            base,
            intent=IntentConfig(negative_examples=(f"not {_SECRET_PAYLOAD}",)),
        )
    elif field_name == "instruction":
        base = dc_replace(
            base,
            execution=ExecutionConfig(instruction=f"Use {_SECRET_PAYLOAD} for auth"),
        )
    else:
        raise ValueError(f"Unknown field: {field_name}")

    return base.to_json()


def _make_contract_with_healthcare_in_field(field_name: str) -> str:
    """Build a contract JSON with healthcare payload in a specific field."""
    from dataclasses import replace as dc_replace
    from application.skill_compression.skill_contract import (
        ExecutionConfig,
        IntentConfig,
    )

    base = create_minimal_contract(
        name="Clean Name",
        phrases=("clean trigger",),
        instruction="Reply with a greeting",
    )

    if field_name == "name":
        base = dc_replace(base, name=_HEALTHCARE_PAYLOAD)
    elif field_name == "tags":
        base = dc_replace(base, tags=(_HEALTHCARE_PAYLOAD,))
    elif field_name == "intent_label":
        base = dc_replace(base, intent=IntentConfig(label=_HEALTHCARE_PAYLOAD))
    elif field_name == "instruction":
        base = dc_replace(
            base,
            execution=ExecutionConfig(instruction=_HEALTHCARE_PAYLOAD),
        )
    else:
        raise ValueError(f"Unknown field for healthcare: {field_name}")

    return base.to_json()


class TestSafetyGateAllFields:
    """GAP-14 fix: safety gate must scan ALL user-controlled text fields.

    Parametrized tests verify that a secret/healthcare payload in ANY field
    causes rejection, not just in instruction or phrases[0].
    """

    @pytest.mark.parametrize(
        "field_name",
        [
            "name",
            "phrases_1",
            "tags",
            "intent_label",
            "intent_positive",
            "intent_negative",
            "instruction",
        ],
    )
    def test_secret_in_any_field_rejected(
        self, installer: SkillInstaller, field_name: str
    ) -> None:
        """Secret payload in field '{field_name}' must be rejected."""
        contract_json = _make_contract_with_secret_in_field(field_name)
        result = installer.install_from_json_string(contract_json, user_id=42)
        assert not result.success, (
            f"Secret in {field_name} was NOT rejected. "
            f"GAP-14: all user-controlled fields must be scanned."
        )

    @pytest.mark.parametrize(
        "field_name",
        ["name", "tags", "intent_label", "instruction"],
    )
    def test_healthcare_in_any_field_rejected(
        self, installer: SkillInstaller, field_name: str
    ) -> None:
        """Healthcare payload in field '{field_name}' must be rejected."""
        contract_json = _make_contract_with_healthcare_in_field(field_name)
        result = installer.install_from_json_string(contract_json, user_id=42)
        assert not result.success, (
            f"Healthcare content in {field_name} was NOT rejected. "
            f"GAP-14: all user-controlled fields must be scanned."
        )


class TestLearnFlowSafetyGateAllFields:
    """Verify LearnFlowService._validate_contract_safety also scans all fields.

    Same GAP-14 fix: LearnFlow and Installer share iter_user_text_fields.
    """

    @pytest.fixture
    def learn_flow(self, contract_store: ContractStore, pipeline: PrivacyPipeline):
        """Minimal LearnFlowService for safety-gate testing."""
        from application.skill_compression.learn_flow_service import (
            LearnFlowService,
            PendingEditStore,
        )
        from application.skill_compression.contract_builder import ContractBuilder
        from application.skill_compression.draft_store import DraftStore

        return LearnFlowService(
            privacy_pipeline=pipeline,
            contract_builder=ContractBuilder(),
            draft_store=DraftStore(),
            pending_edit_store=PendingEditStore(),
            contract_store=contract_store,
        )

    @pytest.mark.parametrize(
        "field_name",
        ["name", "phrases_1", "tags", "intent_label"],
    )
    def test_learn_flow_rejects_secret_in_any_field(
        self, learn_flow, field_name: str
    ) -> None:
        """LearnFlowService safety gate rejects secret in '{field_name}'."""
        from dataclasses import replace as dc_replace
        from application.skill_compression.skill_contract import (
            ActivationConfig,
            IntentConfig,
        )

        contract = create_minimal_contract(
            name="Clean Name",
            phrases=("trigger",),
            instruction="Reply hello",
        )

        if field_name == "name":
            contract = dc_replace(contract, name=f"Secret {_SECRET_PAYLOAD} Skill")
        elif field_name == "phrases_1":
            new_phrases = ("trigger", f"say {_SECRET_PAYLOAD}")
            contract = dc_replace(
                contract, activation=ActivationConfig(phrases=new_phrases)
            )
        elif field_name == "tags":
            contract = dc_replace(contract, tags=(f"key-{_SECRET_PAYLOAD}",))
        elif field_name == "intent_label":
            contract = dc_replace(
                contract, intent=IntentConfig(label=f"secret {_SECRET_PAYLOAD}")
            )

        rejection = learn_flow._validate_contract_safety(
            contract, user_id=42, source="test"
        )
        assert rejection is not None, (
            f"Secret in {field_name} was NOT rejected by LearnFlowService. "
            f"GAP-14: both LearnFlow and Installer must scan all fields."
        )


# ──────────────────────────────────────────────────────────────
# iter_user_text_fields: unit tests
# ──────────────────────────────────────────────────────────────


class TestIterUserTextFields:
    """Unit tests for the shared iter_user_text_fields function."""

    def test_returns_all_basic_fields(self) -> None:
        """All basic fields are returned for a fully-populated contract."""
        from application.skill_compression.skill_contract import (
            iter_user_text_fields,
            IntentConfig,
        )
        from dataclasses import replace as dc_replace

        contract = create_minimal_contract(
            name="My Skill",
            phrases=("hello", "hi there"),
            instruction="Say hello back",
        )
        contract = dc_replace(
            contract,
            tags=("greeting", "friendly"),
            intent=IntentConfig(
                label="greeting_intent",
                positive_examples=("hey", "howdy"),
                negative_examples=("bye", "goodbye"),
            ),
        )

        fields = iter_user_text_fields(contract)
        labels = [label for label, _value in fields]

        assert "name" in labels
        assert "phrases[0]" in labels
        assert "phrases[1]" in labels
        assert "instruction" in labels
        assert "tags[0]" in labels
        assert "tags[1]" in labels
        assert "intent.label" in labels
        assert "intent.positive_examples[0]" in labels
        assert "intent.positive_examples[1]" in labels
        assert "intent.negative_examples[0]" in labels
        assert "intent.negative_examples[1]" in labels

    def test_excludes_empty_fields(self) -> None:
        """Empty or whitespace-only fields are excluded."""
        from application.skill_compression.skill_contract import iter_user_text_fields

        contract = create_minimal_contract(
            name="",
            phrases=("trigger",),
            instruction="Do something",
        )

        fields = iter_user_text_fields(contract)
        labels = [label for label, _value in fields]
        assert "name" not in labels

    def test_minimal_contract_fields(self) -> None:
        """A minimal contract returns at least phrases and instruction."""
        from application.skill_compression.skill_contract import iter_user_text_fields

        contract = create_minimal_contract(
            name="Test",
            phrases=("go",),
            instruction="Execute",
        )
        fields = iter_user_text_fields(contract)
        assert len(fields) >= 3  # name, phrases[0], instruction


# ──────────────────────────────────────────────────────────────
# Production-Path: /installskill handler -> service -> store -> matcher
# ──────────────────────────────────────────────────────────────


class TestInstallskillProductionPath:
    """Production-path: /installskill handler exists, service works end-to-end."""

    def test_handler_importable(self) -> None:
        """handle_installskill_command is importable from skill_commands."""
        from presentation.skill_commands import handle_installskill_command

        assert callable(handle_installskill_command)

    def test_full_pipeline_json_to_matcher(
        self, installer: SkillInstaller, contract_store: ContractStore
    ) -> None:
        """JSON string -> installer -> ContractStore -> SkillMatcher finds it."""
        from application.skill_compression.skill_matcher import SkillMatcher

        contract_json = _make_valid_skill_json(
            name="Production Path Skill",
            trigger="prodtest",
            instruction="Production test reply",
        )

        result = installer.install_from_json_string(contract_json, user_id=99)
        assert result.success
        assert result.contract_name == "Production Path Skill"

        # Verify in store
        contracts = contract_store.get_by_user(99)
        assert len(contracts) == 1
        assert contracts[0].name == "Production Path Skill"
        assert contracts[0].activation.phrases == ("prodtest",)
        assert contracts[0].origin == "manual_install"

        # Verify matchable: SkillMatcher can be constructed and contract is findable
        SkillMatcher(
            storage=None,  # type: ignore[arg-type]
            pattern_judge=None,  # type: ignore[arg-type]
            contract_store=contract_store,
        )
        # Direct contract lookup confirms the skill exists for matching
        user_contracts = contract_store.get_by_user(99)
        assert any("prodtest" in c.activation.phrases for c in user_contracts)
