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
