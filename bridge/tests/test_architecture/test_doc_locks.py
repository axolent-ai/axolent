"""Doc-Lock tests: assert docstring claims against actual code behavior.

Prevents docstring drift in security-critical modules. See
docs/CONVENTIONS.md for the full convention.

Phase 1.5 Item 2: 5 new Doc-Lock tests.
"""

from __future__ import annotations

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

    # Phase 1.5: 12 labels (5 core + 5 EN provider + 2 DE)
    assert len(_ROLE_LABELS) == 12

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
# Doc-Lock 5: RedactingFormatter overrides formatException
# ---------------------------------------------------------------


def test_redacting_formatter_overrides_format_exception() -> None:
    """Doc-Lock: RedactingFormatter docstring says 'covers formatException'."""
    from infrastructure.log_redaction import RedactingFormatter

    # formatException must be defined on RedactingFormatter itself, not just inherited
    assert "formatException" in RedactingFormatter.__dict__, (
        "RedactingFormatter must override formatException (not just inherit it)"
    )

    # The override must call _redact_string on the output
    source = inspect.getsource(RedactingFormatter.formatException)
    assert "_redact_string" in source, (
        "formatException must call _redact_string to redact secrets"
    )
