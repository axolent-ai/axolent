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

Negative locks (claim must NOT be present):

- `test_upsert_field_no_atomic_claim` (bundleE)
- `test_no_unqualified_homoglyph_claim` (bundleF)

Positive locks (claim IS true and must stay true):

- `test_iter_user_text_fields_covers_7_field_groups`
- `test_safe_json_load_default_max_depth_is_64`
- `test_safe_int_rejects_bool`
- `test_escape_role_labels_count_matches_constant`
- `test_redacting_formatter_overrides_format_exception`
- `test_prompt_escaping_brackets_not_word_joiner`

### Location

Architectural doc-locks (cross-module, security-critical claims, default-deny
invariants, monotonicity, normalization coverage) live in:

```
tests/test_architecture/test_doc_locks.py
```

Bundle-specific or incident-specific regression locks may remain co-located
with their original test bundle, but must be recognizable as doc-locks via
naming convention (`test_*_claim`, `test_*_lock`, `test_*_doc_lock`) or a
class/comment that identifies them as doc-locks.

Existing examples of co-located doc-locks:

- `tests/test_bundleE_final_review_fixes.py` (TestMuss8DokuCorrections)
- `tests/test_bundleF_handler_e2e.py` (TestSecretScannerDokuLock)

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

---

## Authorization Decorator Guards

### What Is This?

The wiring guard (`tests/test_architecture/test_wiring_guard.py`,
`TestDecoratorStackConsistency`) ensures that every command/callback
handler registered at runtime is protected by a recognized
authorization decorator (e.g. `@require_whitelist`).

### Update Obligation

If a new permission-granting command decorator is introduced, you
**MUST** update `TestDecoratorStackConsistency.AUTHORIZATION_DECORATORS`
in `tests/test_architecture/test_wiring_guard.py` and add a comment
explaining its authorization semantics. Without this update, the guard
would silently accept new handlers using the new decorator as if they
were unprotected.

### Example

```python
AUTHORIZATION_DECORATORS: frozenset[str] = frozenset(
    [
        "require_whitelist",
        # "require_admin",  # Only bot admins can execute this handler
    ]
)
```
