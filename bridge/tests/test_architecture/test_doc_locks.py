"""Doc-Lock tests: assert docstring claims against actual code behavior.

Prevents docstring drift in security-critical modules. See
docs/CONVENTIONS.md for the full convention.

Phase 1.5 Item 2: 5 new Doc-Lock tests.
Phase 1.5.2-Polish: privacy-filter-normalize + PermissionGate + risk-level locks.
"""

from __future__ import annotations

import importlib
import inspect


# ---------------------------------------------------------------
# Doc-Lock 1: iter_user_text_fields covers exactly 7 field groups
# ---------------------------------------------------------------


def test_iter_user_text_fields_covers_7_field_groups() -> None:
    """Doc-Lock: docstring claims 7 field groups are iterated.

    Fields (from docstring):
      1. name
      2. activation.phrases (ALL)
      3. execution.instruction
      4. tags (each)
      5. intent.label
      6. intent.positive_examples (each)
      7. intent.negative_examples (each)
    """
    from application.skill_compression.skill_contract import (
        ActivationConfig,
        ExecutionConfig,
        IntentConfig,
        SkillContract,
        iter_user_text_fields,
    )

    contract = SkillContract(
        name="test-skill",
        activation=ActivationConfig(phrases=("trigger one", "trigger two")),
        execution=ExecutionConfig(
            type="llm_instruction",
            instruction="do the thing",
        ),
        tags=("tag-a", "tag-b"),
        intent=IntentConfig(
            label="test-intent",
            positive_examples=("pos-example",),
            negative_examples=("neg-example",),
        ),
    )

    fields = iter_user_text_fields(contract)
    labels = [label for label, _ in fields]

    # All 7 groups must be present
    assert "name" in labels
    assert any(lbl.startswith("phrases[") for lbl in labels)
    assert "instruction" in labels
    assert any(lbl.startswith("tags[") for lbl in labels)
    assert "intent.label" in labels
    assert any(lbl.startswith("intent.positive_examples[") for lbl in labels)
    assert any(lbl.startswith("intent.negative_examples[") for lbl in labels)

    # Multi-valued fields: phrases has 2 entries, tags has 2 entries
    phrase_labels = [lbl for lbl in labels if lbl.startswith("phrases[")]
    assert len(phrase_labels) == 2
    tag_labels = [lbl for lbl in labels if lbl.startswith("tags[")]
    assert len(tag_labels) == 2


# ---------------------------------------------------------------
# Doc-Lock 2: safe_json_load default max_depth is 64
# ---------------------------------------------------------------


def test_safe_json_load_default_max_depth_is_64() -> None:
    """Doc-Lock: safe_json_load docstring says 'max_depth=64 default'."""
    from infrastructure.safe_json import DEFAULT_MAX_DEPTH, safe_json_load

    sig = inspect.signature(safe_json_load)
    assert sig.parameters["max_depth"].default == 64
    assert DEFAULT_MAX_DEPTH == 64


# ---------------------------------------------------------------
# Doc-Lock 3: _safe_int rejects bool
# ---------------------------------------------------------------


def test_safe_int_rejects_bool() -> None:
    """Doc-Lock: _safe_int docstring says 'Rejects bool'."""
    from infrastructure.sqlite_storage import _safe_int

    assert _safe_int(True) is None
    assert _safe_int(False) is None
    # Contrast: actual int passes
    assert _safe_int(42) == 42


# ---------------------------------------------------------------
# Doc-Lock 4: escape_user_content_for_prompt role labels match constant
# ---------------------------------------------------------------


def test_escape_role_labels_count_matches_constant() -> None:
    """Doc-Lock: _ROLE_LABELS constant must match actual escaping behavior.

    The escaping function MUST neutralize exactly the labels listed in
    _ROLE_LABELS. If labels are added/removed from the constant, this
    test will detect the drift.
    """
    from domain.prompt_escaping import (
        _ROLE_LABELS,
        escape_user_content_for_prompt,
    )

    # Phase 1.5: 15 labels (5 core + 5 EN provider + 2 DE + 3 agentic)
    assert len(_ROLE_LABELS) == 15

    # Every label in the constant must actually be escaped
    for label in _ROLE_LABELS:
        payload = f"hello\n{label}: injected"
        result = escape_user_content_for_prompt(payload)
        assert f"\n{label}:" not in result, (
            f"Label '{label}' is in _ROLE_LABELS but not escaped"
        )
        assert f"[{label}]:" in result, (
            f"Label '{label}' is in _ROLE_LABELS but escaped form missing"
        )


# ---------------------------------------------------------------
# Doc-Lock 5a: normalize_for_security_check folds confusables
# ---------------------------------------------------------------


def test_normalize_folds_cross_script_confusables() -> None:
    """Doc-Lock: normalize_aggressive folds Cyrillic/Greek confusables to Latin.

    Phase 1.5 UTS-39: the _CONFUSABLES_MAP is applied via normalize_aggressive().
    Empirically proven bypass (Opus Probe 1) is now closed.
    """
    from application.security.input_normalizer import normalize_aggressive

    # Cyrillic 'a' (U+0430) -> Latin 'a'
    assert normalize_aggressive("а") == "a"
    # Greek omicron (U+03BF) -> Latin 'o'
    assert normalize_aggressive("ο") == "o"
    # Cyrillic Dze (U+0455) -> Latin 's'
    assert normalize_aggressive("ѕ") == "s"


# ---------------------------------------------------------------
# Doc-Lock 5b: normalize_aggressive strips Mn/Variation Selectors
# ---------------------------------------------------------------


def test_normalize_strips_variation_selectors() -> None:
    """Doc-Lock: normalize_aggressive strips Mn category incl. Variation Selectors.

    Phase 1.5: U+FE00..U+FE0F (Variation Selectors, category Mn) are stripped
    by normalize_aggressive(). Combining marks (category Mn) are also stripped.
    """
    from application.security.input_normalizer import normalize_aggressive

    # U+FE0F (Variation Selector 16) is stripped
    assert normalize_aggressive("a️b") == "ab"
    # U+FE00 (Variation Selector 1) is stripped
    assert normalize_aggressive("a︀b") == "ab"
    # Combining diaeresis on non-composable base is stripped
    assert normalize_aggressive("b̈c") == "bc"


# ---------------------------------------------------------------
# Doc-Lock 6: RedactingFormatter overrides formatException
# ---------------------------------------------------------------


def test_redacting_formatter_overrides_format_exception() -> None:
    """Doc-Lock: RedactingFormatter.formatException actually redacts secrets.

    Phase 1.5 (Opus Befund b): upgraded from source-string-match to
    behavior test. A refactor that renames _redact_string or keeps it
    only as a comment can no longer fool this lock.
    """
    import logging
    import sys

    from infrastructure.log_redaction import RedactingFormatter

    # Part 1: formatException must be defined on RedactingFormatter itself
    assert "formatException" in RedactingFormatter.__dict__, (
        "RedactingFormatter must override formatException (not just inherit it)"
    )

    # Part 2: Behavior probe -- actually format an exception with a secret
    # and verify the secret is redacted in the output.
    formatter = RedactingFormatter("%(message)s")
    try:
        raise RuntimeError("Bot token: 123456789:AAFakeTokenThatIsLongEnough1234")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_lock5",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )
    formatted = formatter.formatException(record.exc_info)
    assert "AAFakeToken" not in formatted, (
        "formatException must redact secrets from exception tracebacks"
    )
    assert "REDACTED" in formatted, (
        "formatException must replace secrets with a REDACTED marker"
    )


# ---------------------------------------------------------------
# Doc-Lock 7: All privacy filters use normalize_aggressive
# ---------------------------------------------------------------


class TestPrivacyFiltersNormalizeAggressive:
    """Doc-lock: every privacy/security filter MUST behaviorally block
    a Cyrillic-homoglyph variant of a known dangerous term.

    Phase 1.5 Polish-Polish upgrade: source-string check is insufficient
    (Codex proved a bug passed while source-check was green). This lock
    now uses BEHAVIORAL probes: feed each filter a homoglyph-obfuscated
    payload and assert it is blocked.

    InjectionDetector is excluded: separate two-pass architecture validation.
    """

    PRIVACY_FILTER_MODULES = [
        "application.skill_compression.privacy.healthcare_filter",
        "application.skill_compression.privacy.nudge_filter",
        "application.security.secret_scanner",
        "application.leakage_filter",
    ]

    def test_each_privacy_filter_calls_normalize_aggressive(self) -> None:
        """Each privacy filter module imports and uses normalize_aggressive.

        Retained as first-line structural check (fast, catches accidental
        removal of the import). Behavioral test below is the authoritative lock.
        """
        for module_path in self.PRIVACY_FILTER_MODULES:
            mod = importlib.import_module(module_path)  # nosemgrep
            source = inspect.getsource(mod)
            assert "normalize_aggressive" in source, (
                f"{module_path} does NOT use normalize_aggressive. "
                "Pattern matching on raw input is a known bypass vector "
                "(see Phase 1 Pattern-C class, GAP-08)."
            )

    def test_healthcare_filter_blocks_homoglyph_obfuscation(self) -> None:
        """HealthcareFilter MUST block Cyrillic-homoglyph variant of DE term."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )
        from application.skill_compression.privacy.healthcare_filter import (
            HealthcareFilter,
        )

        hf = HealthcareFilter()
        # Cyrillic 'e' (U+0435) in 'depression'
        hyp = Hypothesis(
            hypothesis_id="doclock7-hc",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="Ich habe dеpression",
            status="candidate",
            created_at="2026-05-30T00:00:00Z",
            last_seen="2026-05-30T00:00:00Z",
        )
        assert hf.filter_hypothesis(hyp), (
            "Doc-Lock 7 behavioral: HealthcareFilter did NOT block "
            "Cyrillic-homoglyph variant 'd\\u0435pression'. "
            "Normalization is missing or pattern-side not normalized."
        )

    def test_healthcare_filter_blocks_de_zero_width(self) -> None:
        """HealthcareFilter MUST block Zero-Width obfuscation in DE term."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )
        from application.skill_compression.privacy.healthcare_filter import (
            HealthcareFilter,
        )

        hf = HealthcareFilter()
        # ZWSP (U+200B) between 'st' and umlaut in Angststoerung
        hyp = Hypothesis(
            hypothesis_id="doclock7-hc-zw",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="Ich habe Angstst​örung",
            status="candidate",
            created_at="2026-05-30T00:00:00Z",
            last_seen="2026-05-30T00:00:00Z",
        )
        assert hf.filter_hypothesis(hyp), (
            "Doc-Lock 7 behavioral: HealthcareFilter did NOT block "
            "Zero-Width-obfuscated 'Angstst\\u200boerung'. "
            "Pass 1 must use normalize_for_security_check (basic)."
        )

    def test_nudge_filter_blocks_homoglyph_obfuscation(self) -> None:
        """NudgeFilter MUST block Cyrillic-homoglyph variant of dark pattern."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )
        from application.skill_compression.privacy.nudge_filter import NudgeFilter

        nf = NudgeFilter()
        # Cyrillic 'i' (U+0456) in 'hide'
        hyp = Hypothesis(
            hypothesis_id="doclock7-nudge",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="hіde the opt-out button to make it hard to cancel",
            status="candidate",
            created_at="2026-05-30T00:00:00Z",
            last_seen="2026-05-30T00:00:00Z",
        )
        assert nf.violates_nudge_policy(hyp), (
            "Doc-Lock 7 behavioral: NudgeFilter did NOT block "
            "Cyrillic-homoglyph variant 'h\\u0456de...opt-out...cancel'. "
            "Normalization or aggressive patterns missing."
        )

    def test_secret_scanner_blocks_homoglyph_obfuscation(self) -> None:
        """SecretScanner MUST block Cyrillic-homoglyph variant of secret."""
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )
        from application.security.secret_scanner import SecretScanner

        scanner = SecretScanner()
        # Cyrillic 'a' (U+0430) in 'sk-ant-api03'
        hyp = Hypothesis(
            hypothesis_id="doclock7-secret",
            user_id=1,
            type="preference",
            scope=HypothesisScope(),
            claim="Key: sk-аnt-api03-abcdefghijklmnopqrstuvwxyz",
            status="candidate",
            created_at="2026-05-30T00:00:00Z",
            last_seen="2026-05-30T00:00:00Z",
        )
        reason = scanner.get_block_reason(hyp)
        assert reason is not None, (
            "Doc-Lock 7 behavioral: SecretScanner did NOT block "
            "Cyrillic-homoglyph variant 'sk-\\u0430nt-api03-...'. "
            "Normalization is missing."
        )

    def test_leakage_filter_blocks_homoglyph_obfuscation(self) -> None:
        """LeakageFilter MUST block Cyrillic-homoglyph in forbidden pattern."""
        from application.leakage_filter import check_for_forbidden_patterns

        # Cyrillic 'a' (U+0430) in 'language lock' (a known forbidden pattern)
        result = check_for_forbidden_patterns(
            "The response contains а lаnguage lock directive"
        )
        assert result is not None, (
            "Doc-Lock 7 behavioral: LeakageFilter did NOT block "
            "Cyrillic-homoglyph variant 'l\\u0430nguage lock'. "
            "Normalization is missing."
        )


# ---------------------------------------------------------------
# Doc-Lock 8: PermissionGate default-deny (enabled + empty allowlist)
# ---------------------------------------------------------------


class TestPermissionGateDefaultDeny:
    """Doc-lock: PermissionGate.check_network_access with enabled=True +
    empty domains list must DENY.

    Phase 1 critical security pattern: empty allowlist means nothing is
    allowed, not everything is allowed. Protects against the
    'enabled=True means all-access' anti-pattern.
    """

    def test_permission_gate_empty_allowlist_denies(self) -> None:
        """enabled=True + empty domains = Deny (not Allow)."""
        from application.skill_compression.permission_gate import (
            PermissionDecision,
            PermissionGate,
        )
        from application.skill_compression.skill_contract import (
            ActivationConfig,
            ExecutionConfig,
            NetworkAccessConfig,
            PermissionsConfig,
            SkillContract,
        )

        contract = SkillContract(
            name="test-empty-allowlist",
            activation=ActivationConfig(phrases=("test",)),
            execution=ExecutionConfig(
                type="llm_instruction",
                instruction="do nothing",
            ),
            permissions=PermissionsConfig(
                network_access=NetworkAccessConfig(enabled=True, domains=()),
            ),
        )
        result = PermissionGate.check_network_access(contract, domain="example.com")
        assert result.decision == PermissionDecision.DENY, (
            "PermissionGate must DENY when enabled=True but allowlist is empty. "
            "Empty allowlist means nothing is allowed (default-deny invariant)."
        )
        assert result.rule == "network_empty_allowlist"

    def test_permission_gate_file_empty_scopes_denies(self) -> None:
        """File access: enabled=True + empty scopes = Deny."""
        from application.skill_compression.permission_gate import (
            PermissionDecision,
            PermissionGate,
        )
        from application.skill_compression.skill_contract import (
            ActivationConfig,
            ExecutionConfig,
            FileAccessConfig,
            PermissionsConfig,
            SkillContract,
        )

        contract = SkillContract(
            name="test-file-empty",
            activation=ActivationConfig(phrases=("test",)),
            execution=ExecutionConfig(
                type="llm_instruction",
                instruction="do nothing",
            ),
            permissions=PermissionsConfig(
                file_access=FileAccessConfig(enabled=True, scopes=()),
            ),
        )
        result = PermissionGate.check_file_access(contract, scope="workspace:read")
        assert result.decision == PermissionDecision.DENY, (
            "PermissionGate must DENY file access when scopes list is empty."
        )

    def test_permission_gate_history_empty_scopes_denies(self) -> None:
        """History access: enabled=True + empty scopes = Deny."""
        from application.skill_compression.permission_gate import (
            PermissionDecision,
            PermissionGate,
        )
        from application.skill_compression.skill_contract import (
            ActivationConfig,
            ExecutionConfig,
            HistoryAccessConfig,
            PermissionsConfig,
            SkillContract,
        )

        contract = SkillContract(
            name="test-history-empty",
            activation=ActivationConfig(phrases=("test",)),
            execution=ExecutionConfig(
                type="llm_instruction",
                instruction="do nothing",
            ),
            permissions=PermissionsConfig(
                history_access=HistoryAccessConfig(enabled=True, scopes=()),
            ),
        )
        result = PermissionGate.check_history_access(contract, scope="current_chat")
        assert result.decision == PermissionDecision.DENY, (
            "PermissionGate must DENY history access when scopes list is empty."
        )


# ---------------------------------------------------------------
# Doc-Lock 9: compute_risk_level monotonicity
# ---------------------------------------------------------------


class TestComputeRiskLevelMonotonicity:
    """Doc-lock: compute_risk_level(permissions) MUST be monotonically
    non-decreasing as permissions expand.

    Invariants:
      - Wildcard (*) in tools => high
      - Adding permissions never decreases risk level
      - Dangerous permissions (secrets, network, file) => high
    """

    _LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}

    def test_wildcard_tools_is_high(self) -> None:
        """Wildcard '*' in tools must produce 'high' risk."""
        from application.skill_compression.skill_contract import (
            PermissionsConfig,
            compute_risk_level,
        )

        perms = PermissionsConfig(tools=("*",))
        assert compute_risk_level(perms) == "high"

    def test_adding_scope_does_not_decrease_risk(self) -> None:
        """Adding memory_write to memory_read must not decrease risk."""
        from application.skill_compression.skill_contract import (
            PermissionsConfig,
            compute_risk_level,
        )

        base = compute_risk_level(PermissionsConfig(memory_read=("long_term_facts",)))
        extended = compute_risk_level(
            PermissionsConfig(
                memory_read=("long_term_facts",),
                memory_write=("long_term_facts",),
            )
        )
        assert self._LEVEL_ORDER[extended] >= self._LEVEL_ORDER[base], (
            f"Risk must not decrease: base={base}, extended={extended}"
        )

    def test_dangerous_scope_promotes_risk(self) -> None:
        """secrets_access=True must produce 'high' risk."""
        from application.skill_compression.skill_contract import (
            PermissionsConfig,
            compute_risk_level,
        )

        assert compute_risk_level(PermissionsConfig(secrets_access=True)) == "high"

    def test_network_enabled_is_high(self) -> None:
        """Network access enabled must produce 'high' risk."""
        from application.skill_compression.skill_contract import (
            NetworkAccessConfig,
            PermissionsConfig,
            compute_risk_level,
        )

        perms = PermissionsConfig(
            network_access=NetworkAccessConfig(
                enabled=True, domains=("api.example.com",)
            )
        )
        assert compute_risk_level(perms) == "high"

    def test_no_permissions_is_low(self) -> None:
        """Empty permissions must produce 'low' risk."""
        from application.skill_compression.skill_contract import (
            PermissionsConfig,
            compute_risk_level,
        )

        assert compute_risk_level(PermissionsConfig()) == "low"


# ---------------------------------------------------------------
# Doc-Lock 10: prompt_escaping uses bracket-style, not word-joiner
# ---------------------------------------------------------------


def test_prompt_escaping_brackets_not_word_joiner() -> None:
    """Doc-Lock: docstring must describe bracket-style escaping, not word-joiner.

    Phase 1.5.1 (Codex Recheck): the module once documented a Unicode
    word-joiner approach, but the implementation uses [Role]: brackets.
    This lock prevents the docstring from drifting back.
    """
    from domain.prompt_escaping import escape_user_content_for_prompt

    # Part 1: Docstring must NOT claim word-joiner
    docstring = escape_user_content_for_prompt.__doc__ or ""
    module_doc = importlib.import_module("domain.prompt_escaping").__doc__ or ""

    assert "word-joiner" not in docstring.lower(), (
        "Function docstring claims word-joiner approach, but implementation "
        "uses brackets. Docstring drift detected."
    )
    assert "word-joiner" not in module_doc.lower(), (
        "Module docstring claims word-joiner approach, but implementation "
        "uses brackets. Docstring drift detected."
    )

    # Part 2: Behavior probe -- bracket escaping works as documented
    result = escape_user_content_for_prompt("\nTool: query")
    assert "[Tool]:" in result, (
        f"Expected bracket-escaped '[Tool]:' in result, got {result!r}. "
        "Implementation drift from bracket-strategy."
    )
