# AXOLENT Code Conventions

## Doc-Lock Tests

### What Is a Doc-Lock?

A Doc-Lock test is a test that **asserts a specific claim made in a
docstring, comment, or constant definition against the actual code
behavior**. If someone changes the claim without changing the code
(or vice versa), the test fails.

Doc-Locks prevent docstring drift: the slow divergence between what
documentation says the code does and what the code actually does.
In security-critical code, docstring drift creates false confidence.

### When a Doc-Lock Is Required

A Doc-Lock test is **mandatory** for:

- Any docstring claim about **security properties** (e.g. "rejects
  bool", "escapes role labels", "covers formatException").
- Any docstring that **enumerates a specific set** (e.g. "iterates
  7 user-text fields", "neutralizes 5 role labels").
- Any docstring claim about **atomicity, thread-safety, or
  concurrency guarantees**.
- Any constant definition with a docstring that states its
  **default value or range** (e.g. "max_depth=64 default").
- Any claim about **homoglyph handling, normalization, or
  character-set coverage** in security modules.

### When a Doc-Lock Is NOT Needed

- Trivial descriptions ("Returns the user ID").
- Implementation details without security implications.
- Docstrings that describe intent without making verifiable claims.

### Naming Convention

```
test_<module>_<claim_description>
```

Examples from the codebase:

- `test_input_normalizer_no_cyrillic_claim`
- `test_injection_detector_no_cyrillic_claim`
- `test_upsert_field_no_atomic_claim`
- `test_no_unqualified_homoglyph_claim`

For positive locks (claim IS true and must stay true):

- `test_iter_user_text_fields_covers_7_fields`
- `test_safe_json_load_default_max_depth_is_64`
- `test_safe_int_rejects_bool`
- `test_escape_role_labels_count_matches_constant`
- `test_redacting_formatter_overrides_format_exception`

### Location

All Doc-Lock tests live in:

```
tests/test_architecture/test_doc_locks.py
```

### Example Pattern

```python
def test_safe_json_load_default_max_depth_is_64() -> None:
    """Doc-Lock: safe_json_load docstring says 'max_depth=64 default'."""
    import inspect
    from domain.json_helpers import safe_json_load

    sig = inspect.signature(safe_json_load)
    assert sig.parameters["max_depth"].default == 64
```

### Review Checklist

When reviewing PRs that touch `application/security/`,
`application/safety/`, or `domain/` modules:

1. Does any new or changed docstring make a verifiable claim?
2. If yes: is there a Doc-Lock test for it?
3. If a Doc-Lock test exists: does the change update both the
   docstring AND the test?
